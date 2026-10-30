"""Active-speaker reframing for the 9:16 crop.

When a shot holds two or more people, the vertical crop should show the person
who is actually talking and cut (not pan) to the other person when they take
over. This module plans the crop for a whole clip before any frame is
rendered:

1. Analysis pass: decode the clip once, sampling ~8 frames/s. Every sample
   runs BlazeFace (full-range) on a downscaled frame; faces are linked into
   identities with an IoU/centroid tracker, reset at every scene cut. When a
   sample has 2+ faces, FaceMesh measures each face's mouth opening (inner lip
   gap divided by a rigid eye/nose scale, so zooming or nodding does not read
   as talking).
2. Audio: RMS energy of the clip's audio says when anyone is talking at all,
   so nothing switches during silence.
3. Decision per scene: faceless or crowd (>4 faces) -> GENERAL (blurred
   letterbox); one person -> follow them; several people who fit in one 9:16
   crop together -> keep them all; otherwise -> the active speaker, chosen
   per sample from mouth-motion variance over a ~0.5s window, with hysteresis
   (new speaker must dominate for 0.4s, min 1.3s hold) and a hard cut back-
   dated to when the new speaker started.

The output is one FramePlan per frame, consumed by main.process_video_to_vertical.
The pure decision logic (mouth_openness, FaceTracker, choose_speakers,
speech_mask_from_rms, faces_fit) has no cv2/mediapipe dependency so it can be
unit-tested anywhere.
"""
import math
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from collections import namedtuple

# --- Tunables -------------------------------------------------------------
SAMPLE_HZ = 8.0             # analysis samples per second (~25ms each with 2 faces)
DETECT_WIDTH = 640          # BlazeFace runs on a frame downscaled to this width
MESH_SIZE = 192             # face crop side fed to FaceMesh
MIN_MESH_FACE_PX = 36       # faces smaller than this (source px) get no mouth reading
MOUTH_WINDOW_S = 0.5        # mouth-motion variance window
MIN_HOLD_S = 1.3            # never cut away from a speaker sooner than this
DOMINATE_S = 0.4            # a new speaker must lead for this long first
DOMINANCE_RATIO = 1.35      # ... by at least this factor over the runner-up
MIN_ACTIVITY = 0.02         # std of normalized mouth opening that counts as talking
LOST_SPEAKER_S = 0.6        # current speaker gone this long -> pick someone visible
TRACK_MAX_GAP_S = 1.0       # identity forgotten after this long unseen
MIN_TRACK_PRESENCE = 0.25   # tracks seen in fewer samples are noise / passers-by
CROWD_FACES = 4             # more faces than this on average -> GENERAL
FIT_PAD = 0.35              # face-width padding each side when checking "both fit"

# FaceMesh landmark ids (468-point topology)
_LIP_PAIRS = ((13, 14), (82, 87), (312, 317))   # inner upper/lower lip pairs
_EYE_L, _EYE_R, _NOSE_TIP = 33, 263, 1

FramePlan = namedtuple("FramePlan", "mode box snap")   # mode: 'GENERAL' | 'TRACK'


# --- Pure logic -----------------------------------------------------------
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def mouth_openness(pts):
    """Inner-lip gap normalized by a rigid face scale.

    ``pts`` maps landmark id -> (x, y) in pixel units (a list of 468 points or
    a dict). The scale is the geometric mean of the eye-corner distance and the
    eye-midpoint-to-nose-tip distance: neither moves with the jaw, the former
    shrinks with yaw and the latter with pitch, so the product is steadier
    than either. Scaling the whole face (zoom, leaning in) leaves the value
    unchanged.
    """
    gap = sum(_dist(pts[a], pts[b]) for a, b in _LIP_PAIRS) / len(_LIP_PAIRS)
    eye_l, eye_r = pts[_EYE_L], pts[_EYE_R]
    inter = _dist(eye_l, eye_r)
    mid = ((eye_l[0] + eye_r[0]) / 2.0, (eye_l[1] + eye_r[1]) / 2.0)
    eye_nose = _dist(mid, pts[_NOSE_TIP])
    scale = math.sqrt(inter * eye_nose)
    if scale <= 1e-6:
        return 0.0
    return gap / scale


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """Greedy IoU/centroid tracker that gives each face a stable id."""

    def __init__(self, max_gap_s=TRACK_MAX_GAP_S, min_iou=0.15, max_center_dist=0.8):
        self.max_gap_s = max_gap_s
        self.min_iou = min_iou
        # centre distance, in face widths, still accepted as the same person
        self.max_center_dist = max_center_dist
        self.tracks = {}      # id -> (box, last_t)
        self.next_id = 0

    def reset(self):
        self.tracks = {}

    def update(self, boxes, t):
        """Return one track id per box (same order)."""
        self.tracks = {i: v for i, v in self.tracks.items() if t - v[1] <= self.max_gap_s}
        pairs = []
        for tid, (tbox, _) in self.tracks.items():
            tcx, tcy = tbox[0] + tbox[2] / 2.0, tbox[1] + tbox[3] / 2.0
            for bi, box in enumerate(boxes):
                ov = iou(tbox, box)
                cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
                d = math.hypot(cx - tcx, cy - tcy) / max(1.0, max(tbox[2], box[2]))
                if ov >= self.min_iou or d <= self.max_center_dist:
                    pairs.append((-ov, d, tid, bi))
        pairs.sort()
        ids = [None] * len(boxes)
        used = set()
        for _, _, tid, bi in pairs:
            if tid in used or ids[bi] is not None:
                continue
            ids[bi] = tid
            used.add(tid)
        for bi, box in enumerate(boxes):
            if ids[bi] is None:
                ids[bi] = self.next_id
                self.next_id += 1
            self.tracks[ids[bi]] = (tuple(box), t)
        return ids


def window_std(times, values, t, half_window):
    """Population std of the non-None values within [t-hw, t+hw]; 0 if <3."""
    vals = [v for ti, v in zip(times, values) if v is not None and abs(ti - t) <= half_window]
    if len(vals) < 3:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def mouth_activity(times, openness, window_s=MOUTH_WINDOW_S):
    """{id: [activity per sample]} from {id: [openness or None per sample]}."""
    return {tid: [window_std(times, vals, t, window_s / 2.0) for t in times]
            for tid, vals in openness.items()}


def speaking_candidate(activity_at, speech, min_activity=MIN_ACTIVITY, ratio=DOMINANCE_RATIO):
    """The one id clearly talking at a sample, else None.

    ``activity_at`` maps id -> activity at this sample (only visible faces).
    """
    if not speech or not activity_at:
        return None
    ranked = sorted(activity_at.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best = ranked[0]
    if best < min_activity:
        return None
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    if best < runner * ratio:
        return None
    return best_id


def choose_speakers(times, activity, present, speech, default_id,
                    min_hold_s=MIN_HOLD_S, dominate_s=DOMINATE_S,
                    lost_s=LOST_SPEAKER_S, min_activity=MIN_ACTIVITY,
                    ratio=DOMINANCE_RATIO):
    """Pick who the crop shows at each sample, with hysteresis.

    times:    sample times (seconds, ascending)
    activity: {id: [mouth activity per sample]}
    present:  {id: [bool per sample]}  (face visible at that sample)
    speech:   [bool per sample]        (audio says someone is talking)
    default_id: who to show before anyone is clearly speaking

    A new speaker takes over once they have been the clear candidate for
    ``dominate_s``; the cut is back-dated to when their run started (this is
    an offline plan, so the cut lands as they start talking), but never
    earlier than ``min_hold_s`` after the previous cut. With nobody clearly
    talking the current person is held. If the current person has left the
    frame for ``lost_s``, the cut goes to the most active visible face.
    """
    n = len(times)
    if n == 0:
        return []
    cand = []
    for i in range(n):
        vis = {tid: activity[tid][i] for tid in activity if present.get(tid, [False] * n)[i]}
        cand.append(speaking_candidate(vis, speech[i], min_activity, ratio))

    out = [None] * n
    current = None
    last_cut_t = times[0]
    run_id, run_start = None, 0
    missing_since = None
    for i in range(n):
        t = times[i]
        c = cand[i]
        if c != run_id:
            run_id, run_start = c, i

        if current is None:
            if c is not None and t - times[run_start] >= dominate_s:
                # First confirmed speaker owns the scene from its start.
                current = c
                for j in range(i):
                    out[j] = c
                last_cut_t = times[0]
        elif c is not None and c != current and t - times[run_start] >= dominate_s:
            cut_t = max(times[run_start], last_cut_t + min_hold_s)
            if cut_t <= t:
                current = c
                last_cut_t = cut_t
                for j in range(i, -1, -1):
                    if times[j] < cut_t:
                        break
                    out[j] = c

        # The person on screen walked out (or the detector lost them for good).
        if current is not None and not present.get(current, [False] * n)[i]:
            missing_since = t if missing_since is None else missing_since
            if t - missing_since >= lost_s:
                vis = [tid for tid in activity if present.get(tid, [False] * n)[i]]
                if vis:
                    current = max(vis, key=lambda tid: activity[tid][i])
                    last_cut_t = t
                    missing_since = None
        else:
            missing_since = None

        out[i] = current

    if current is None:
        return [default_id] * n
    return [default_id if s is None else s for s in out]


def speech_mask_from_rms(rms, abs_floor=0.004, rel=0.12):
    """[bool] per RMS value: True where someone is plausibly talking.

    Threshold = max(absolute floor, ``rel`` x the loud (95th pct) level), so
    the mask adapts to quiet and hot recordings alike. All-silent input -> all
    False; when the audio is uniformly loud (wall-to-wall talk) everything
    passes.
    """
    if not rms:
        return []
    srt = sorted(rms)
    loud = srt[min(len(srt) - 1, int(0.95 * len(srt)))]
    thr = max(abs_floor, rel * loud)
    return [r >= thr for r in rms]


def union_box(boxes):
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[0] + b[2] for b in boxes)
    y2 = max(b[1] + b[3] for b in boxes)
    return (x1, y1, x2 - x1, y2 - y1)


def faces_fit(boxes, crop_width, pad=FIT_PAD):
    """True when every face (with ``pad`` face-widths each side) fits in one crop."""
    if not boxes:
        return False
    left = min(b[0] - pad * b[2] for b in boxes)
    right = max(b[0] + b[2] + pad * b[2] for b in boxes)
    return right - left <= crop_width


def _median(vals):
    s = sorted(vals)
    k = len(s)
    return s[k // 2] if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2.0


def _interp_track(times, boxes):
    """Fill None gaps in a per-sample box list (linear between, hold at ends)."""
    known = [i for i, b in enumerate(boxes) if b is not None]
    if not known:
        return boxes
    out = list(boxes)
    for i in range(len(out)):
        if out[i] is not None:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        if prev is None:
            out[i] = boxes[nxt]
        elif nxt is None:
            out[i] = boxes[prev]
        else:
            a = (times[i] - times[prev]) / max(1e-9, times[nxt] - times[prev])
            out[i] = tuple(p + (q - p) * a for p, q in zip(boxes[prev], boxes[nxt]))
    return out


def plan_scene(times, samples, speech, crop_width, crowd_faces=CROWD_FACES):
    """Decide the scene's mode and per-sample target.

    samples: per sample, a list of dicts {'id', 'box', 'open'} (open may be None)
    Returns (mode, targets, speaker_ids) where mode is 'GENERAL', 'SINGLE',
    'GROUP' or 'SPEAKER', targets is a per-sample box (or None for GENERAL)
    and speaker_ids marks who is framed (changes = hard cuts).
    """
    n = len(times)
    if n == 0:
        return 'GENERAL', [], []
    counts = [len(s) for s in samples]
    avg = sum(counts) / n
    if avg < 0.5 or avg > crowd_faces:
        return 'GENERAL', [None] * n, [None] * n

    boxes, opens = {}, {}
    for i, s in enumerate(samples):
        for f in s:
            boxes.setdefault(f['id'], [None] * n)[i] = tuple(f['box'])
            opens.setdefault(f['id'], [None] * n)[i] = f.get('open')
    present = {tid: [b is not None for b in bl] for tid, bl in boxes.items()}
    main = [tid for tid in boxes if sum(present[tid]) >= MIN_TRACK_PRESENCE * n]
    if not main:   # only flickering detections: follow the most-seen one
        main = [max(boxes, key=lambda tid: sum(present[tid]))]
    filled = {tid: _interp_track(times, boxes[tid]) for tid in main}

    def area(tid):
        return _median([b[2] * b[3] for b in boxes[tid] if b is not None])

    if len(main) == 1:
        tid = main[0]
        return 'SINGLE', filled[tid], [tid] * n

    medians = [tuple(_median([b[k] for b in boxes[tid] if b is not None]) for k in range(4))
               for tid in main]
    if faces_fit(medians, crop_width):
        targets = []
        for i in range(n):
            vis = [boxes[tid][i] for tid in main if boxes[tid][i] is not None]
            targets.append(union_box(vis) if vis else union_box([filled[tid][i] for tid in main]))
        return 'GROUP', targets, [-1] * n

    activity = mouth_activity(times, {tid: opens[tid] for tid in main})
    default_id = max(main, key=area)
    ids = choose_speakers(times, activity, {tid: present[tid] for tid in main}, speech, default_id)
    return 'SPEAKER', [filled[tid][i] for i, tid in enumerate(ids)], ids


# --- Media I/O (cv2 / mediapipe / ffmpeg) ---------------------------------
MESH_WORKERS = 2            # FaceMesh threads (mediapipe releases the GIL)

_face_detector = None
_mesh_pool = None
_mesh_local = threading.local()


def _models():
    """Load BlazeFace + the FaceMesh thread pool once per process."""
    global _face_detector, _mesh_pool
    if _face_detector is None:
        import mediapipe as mp
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5)
        _mesh_pool = ThreadPoolExecutor(MESH_WORKERS, thread_name_prefix="facemesh")
    return _face_detector, _mesh_pool


def _thread_mesh():
    # FaceMesh graphs are not thread-safe: one per worker thread, kept for life.
    mesh = getattr(_mesh_local, "mesh", None)
    if mesh is None:
        import mediapipe as mp
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False,
            min_detection_confidence=0.3)
        _mesh_local.mesh = mesh
    return mesh


def _detect(frame, detector):
    import cv2
    h, w = frame.shape[:2]
    scale = min(1.0, DETECT_WIDTH / float(w))
    small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR) \
        if scale < 1.0 else frame
    res = detector.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    out = []
    for d in (res.detections or []):
        bb = d.location_data.relative_bounding_box
        x, y = max(0.0, bb.xmin * w), max(0.0, bb.ymin * h)
        bw, bh = min(w - x, bb.width * w), min(h - y, bb.height * h)
        if bw > 4 and bh > 4:
            out.append((x, y, bw, bh))
    return out


def _face_crop(frame, box):
    """Square RGB crop around a face for FaceMesh, plus its source size."""
    import cv2
    x, y, w, h = box
    if h < MIN_MESH_FACE_PX:
        return None
    H, W = frame.shape[:2]
    side = max(w, h) * 1.6
    cx, cy = x + w / 2.0, y + h / 2.0
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x2, y2 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    crop = cv2.resize(frame[y1:y2, x1:x2], (MESH_SIZE, MESH_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (x2 - x1), (y2 - y1)


def _mesh_openness(job):
    """Mouth openness for one face crop (runs on a FaceMesh worker thread)."""
    rgb, sx, sy = job
    res = _thread_mesh().process(rgb)
    if not res.multi_face_landmarks:
        return None
    # Back to the crop's true pixel aspect before measuring.
    return mouth_openness([(p.x * sx, p.y * sy) for p in res.multi_face_landmarks[0].landmark])


def audio_rms(video_path, times, half_window=0.15, sr=16000):
    """RMS per sample time from the clip's audio; None if there is no audio."""
    import numpy as np
    try:
        raw = subprocess.run(
            ['ffmpeg', '-v', 'error', '-i', video_path, '-map', '0:a:0', '-ac', '1',
             '-ar', str(sr), '-f', 's16le', '-'],
            capture_output=True, timeout=300).stdout
    except Exception:
        return None
    if not raw:
        return None
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hw = int(half_window * sr)
    out = []
    for t in times:
        c = int(t * sr)
        seg = pcm[max(0, c - hw):c + hw]
        out.append(float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0)
    return out


def plan_clip(video_path, scene_ranges, fps, video_width, video_height, crop_width):
    """Analyse the clip and return a FramePlan per frame (index = frame number).

    scene_ranges: [(start_frame, end_frame)] covering the clip.
    """
    import cv2
    fps = fps or 30.0
    step = max(1, int(round(fps / SAMPLE_HZ)))
    detector, pool = _models()

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # per scene: sample frame indices and per-sample face lists
    scene_samples = [([], []) for _ in scene_ranges]
    tracker = FaceTracker()
    pending = []
    si = 0
    f = 0
    while True:
        if not cap.grab():
            break
        while si < len(scene_ranges) - 1 and f >= scene_ranges[si][1]:
            si += 1
        start_f = scene_ranges[si][0]
        if f == start_f:
            tracker.reset()   # new shot: never carry a face over a cut
        if (f - start_f) % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                t = f / fps
                boxes = _detect(frame, detector)
                ids = tracker.update(boxes, t)
                many = 2 <= len(boxes) <= CROWD_FACES
                faces = []
                for tid, b in zip(ids, boxes):
                    job = _face_crop(frame, b) if many else None
                    # FaceMesh runs on worker threads while this thread decodes
                    # and detects ahead; results are collected after the pass.
                    fut = pool.submit(_mesh_openness, job) if job else None
                    if fut is not None:
                        pending.append(fut)
                        if len(pending) > 4 * MESH_WORKERS:
                            pending.pop(0).result()
                    faces.append({'id': tid, 'box': b, 'open': fut})
                scene_samples[si][0].append(f)
                scene_samples[si][1].append(faces)
        f += 1
    cap.release()
    total = max(total, f)
    for _, samples in scene_samples:
        for faces in samples:
            for face in faces:
                if isinstance(face['open'], Future):
                    try:
                        face['open'] = face['open'].result()
                    except Exception:
                        face['open'] = None

    all_times = [fr / fps for frames, _ in scene_samples for fr in frames]
    rms = audio_rms(video_path, all_times)
    speech_all = speech_mask_from_rms(rms) if rms else [True] * len(all_times)

    plan = [None] * total
    k = 0
    summary = []
    for (s_start, s_end), (frames, samples) in zip(scene_ranges, scene_samples):
        times = [fr / fps for fr in frames]
        speech = speech_all[k:k + len(frames)]
        k += len(frames)
        mode, targets, ids = plan_scene(times, samples, speech, crop_width)
        summary.append(mode)
        s_end = min(s_end, total)
        si_ = 0
        for fr in range(s_start, s_end):
            while si_ + 1 < len(frames) and frames[si_ + 1] <= fr:
                si_ += 1
            if mode == 'GENERAL' or not frames:
                plan[fr] = FramePlan('GENERAL', None, fr == s_start)
                continue
            snap = fr == s_start or (frames[si_] == fr and si_ > 0 and ids[si_] != ids[si_ - 1])
            plan[fr] = FramePlan('TRACK', targets[si_], snap)
    # Frames past the last scene (count mismatch) repeat the last plan.
    last = None
    for i in range(total):
        if plan[i] is None:
            plan[i] = last or FramePlan('GENERAL', None, False)
        last = plan[i]
    cuts = sum(1 for p in plan if p.snap)
    print(f"   🎙️ Speaker plan: scenes={summary} cuts/snaps={cuts}")
    return plan
