"""Active-speaker reframing: the pure decision logic (no cv2/mediapipe needed)."""
import math

import speaker_reframe as sr


# --- mouth_openness --------------------------------------------------------
def _face(open_px, scale=1.0, dx=0.0, dy=0.0):
    """Synthetic landmark dict: eyes 60px apart, nose 45px below, lips open_px."""
    def p(x, y):
        return (dx + x * scale, dy + y * scale)
    pts = {33: p(70, 100), 263: p(130, 100), 1: p(100, 145)}
    for up, lo in sr._LIP_PAIRS:
        pts[up] = p(100, 170)
        pts[lo] = p(100, 170 + open_px)
    return pts


def test_closed_mouth_is_zero():
    assert sr.mouth_openness(_face(0)) == 0


def test_openness_grows_with_lip_gap():
    assert sr.mouth_openness(_face(12)) > sr.mouth_openness(_face(4)) > 0


def test_openness_ignores_zoom_and_position():
    base = sr.mouth_openness(_face(8))
    assert math.isclose(sr.mouth_openness(_face(8, scale=2.5)), base, rel_tol=1e-9)
    assert math.isclose(sr.mouth_openness(_face(8, scale=0.4, dx=900, dy=300)), base, rel_tol=1e-9)


def test_degenerate_face_does_not_divide_by_zero():
    pts = {k: (5.0, 5.0) for k in (33, 263, 1, 13, 14, 82, 87, 312, 317)}
    assert sr.mouth_openness(pts) == 0.0


# --- FaceTracker ----------------------------------------------------------
def test_tracker_keeps_ids_as_faces_drift():
    tr = sr.FaceTracker()
    a, b = (100, 100, 80, 80), (900, 120, 80, 80)
    ids0 = tr.update([a, b], 0.0)
    # small drift, reversed detection order
    ids1 = tr.update([(905, 118, 80, 80), (110, 102, 82, 82)], 0.1)
    assert ids1 == [ids0[1], ids0[0]]


def test_tracker_new_face_gets_new_id():
    tr = sr.FaceTracker()
    (a,) = tr.update([(100, 100, 80, 80)], 0.0)
    ids = tr.update([(100, 100, 80, 80), (600, 100, 80, 80)], 0.1)
    assert ids[0] == a and ids[1] != a


def test_tracker_forgets_after_gap_and_reset():
    tr = sr.FaceTracker(max_gap_s=1.0)
    (a,) = tr.update([(100, 100, 80, 80)], 0.0)
    (b,) = tr.update([(100, 100, 80, 80)], 2.0)      # gone for 2s
    assert b != a
    tr.reset()
    (c,) = tr.update([(100, 100, 80, 80)], 2.1)      # scene cut
    assert c not in (a, b)


def test_tracker_does_not_merge_two_faces_into_one_track():
    tr = sr.FaceTracker()
    tr.update([(100, 100, 80, 80)], 0.0)
    ids = tr.update([(110, 100, 80, 80), (150, 100, 80, 80)], 0.1)
    assert len(set(ids)) == 2


# --- speech mask ----------------------------------------------------------
def test_speech_mask_silence_and_talk():
    rms = [0.0005] * 10 + [0.08] * 10 + [0.001] * 5
    mask = sr.speech_mask_from_rms(rms)
    assert not any(mask[:10]) and all(mask[10:20]) and not any(mask[20:])


def test_speech_mask_all_silent():
    assert not any(sr.speech_mask_from_rms([0.0001] * 20))


# --- speaker choice -------------------------------------------------------
DT = 1 / 12.0


def _timeline(segments):
    """segments: [(seconds, speaker 'A'/'B'/None)] -> times, activity, present."""
    times, act = [], {'A': [], 'B': []}
    t = 0.0
    for dur, who in segments:
        for _ in range(int(round(dur / DT))):
            times.append(t)
            act['A'].append(0.08 if who == 'A' else 0.005)
            act['B'].append(0.08 if who == 'B' else 0.005)
            t += DT
    present = {k: [True] * len(times) for k in act}
    return times, act, present


def _first_time(times, ids, who):
    return next(t for t, s in zip(times, ids) if s == who)


def test_flips_to_the_new_speaker_with_a_backdated_cut():
    times, act, present = _timeline([(3, 'A'), (3, 'B')])
    ids = sr.choose_speakers(times, act, present, [True] * len(times), default_id='A')
    assert ids[0] == 'A' and ids[-1] == 'B'
    # cut lands when B starts talking (t=3), not 0.4s later
    assert abs(_first_time(times, ids, 'B') - 3.0) < 2 * DT
    # exactly one cut
    assert sum(1 for i in range(1, len(ids)) if ids[i] != ids[i - 1]) == 1


def test_first_speaker_owns_the_scene_start():
    times, act, present = _timeline([(1, None), (2, 'B')])
    ids = sr.choose_speakers(times, act, present, [True] * len(times), default_id='A')
    assert set(ids) == {'B'}


def test_short_interjection_does_not_flip():
    times, act, present = _timeline([(3, 'A'), (0.25, 'B'), (3, 'A')])
    ids = sr.choose_speakers(times, act, present, [True] * len(times), default_id='A')
    assert set(ids) == {'A'}


def test_minimum_hold_between_cuts():
    times, act, present = _timeline([(2, 'A'), (0.6, 'B'), (2, 'A')])
    ids = sr.choose_speakers(times, act, present, [True] * len(times), default_id='A',
                             min_hold_s=1.3)
    cuts = [times[i] for i in range(1, len(ids)) if ids[i] != ids[i - 1]]
    for a, b in zip(cuts, cuts[1:]):
        assert b - a >= 1.3 - 1e-9


def test_rapid_back_and_forth_respects_hold():
    segs = [(0.6, 'A'), (0.6, 'B')] * 6
    times, act, present = _timeline(segs)
    ids = sr.choose_speakers(times, act, present, [True] * len(times), default_id='A')
    cuts = [times[i] for i in range(1, len(ids)) if ids[i] != ids[i - 1]]
    for a, b in zip(cuts, cuts[1:]):
        assert b - a >= sr.MIN_HOLD_S - 1e-9


def test_no_switch_during_silence():
    times, act, present = _timeline([(3, 'A'), (3, 'B')])
    speech = [t < 3.0 for t in times]      # B "moves their mouth" silently (chewing)
    ids = sr.choose_speakers(times, act, present, speech, default_id='A')
    assert set(ids) == {'A'}


def test_nobody_clearly_speaking_holds_default():
    times, act, present = _timeline([(4, None)])
    ids = sr.choose_speakers(times, act, present, [True] * len(times), default_id='B')
    assert set(ids) == {'B'}


def test_both_talking_equally_holds_current():
    times, act, present = _timeline([(2, 'A')])
    n = len(times)
    t2 = [times[-1] + DT * (i + 1) for i in range(int(2 / DT))]
    times += t2
    act['A'] += [0.08] * len(t2)
    act['B'] += [0.08] * len(t2)
    present = {k: [True] * len(times) for k in act}
    ids = sr.choose_speakers(times, act, present, [True] * len(times), default_id='A')
    assert set(ids[n:]) == {'A'}


def test_lost_speaker_cuts_to_a_visible_face():
    times, act, present = _timeline([(2, 'A'), (2, None)])
    present['A'] = [t < 2.0 for t in times]
    ids = sr.choose_speakers(times, act, present, [True] * len(times), default_id='A')
    assert ids[-1] == 'B'


# --- scene planning -------------------------------------------------------
def test_faces_fit():
    assert sr.faces_fit([(100, 100, 80, 80), (300, 100, 80, 80)], crop_width=608)
    assert not sr.faces_fit([(100, 100, 80, 80), (1400, 100, 80, 80)], crop_width=608)


def _samples(n, boxes_by_id, opens_by_id=None):
    out = []
    for i in range(n):
        out.append([{'id': tid, 'box': b,
                     'open': (opens_by_id or {}).get(tid, [None] * n)[i]}
                    for tid, b in boxes_by_id.items()])
    return out


def test_plan_faceless_is_general():
    times = [i * DT for i in range(24)]
    mode, targets, _ = sr.plan_scene(times, [[] for _ in times], [True] * 24, 608)
    assert mode == 'GENERAL'


def test_plan_crowd_is_general():
    times = [i * DT for i in range(24)]
    crowd = {k: (k * 300, 100, 60, 60) for k in range(6)}
    mode, _, _ = sr.plan_scene(times, _samples(24, crowd), [True] * 24, 608)
    assert mode == 'GENERAL'


def test_plan_single_face_tracks():
    times = [i * DT for i in range(24)]
    mode, targets, _ = sr.plan_scene(times, _samples(24, {7: (500, 200, 150, 150)}), [True] * 24, 608)
    assert mode == 'SINGLE' and targets[0] == (500, 200, 150, 150)


def test_plan_close_pair_is_group():
    times = [i * DT for i in range(24)]
    pair = {1: (800, 300, 120, 120), 2: (1000, 300, 120, 120)}
    mode, targets, _ = sr.plan_scene(times, _samples(24, pair), [True] * 24, 608)
    assert mode == 'GROUP'
    x, _, w, _ = targets[0]
    assert x == 800 and x + w == 1120


def test_plan_far_pair_follows_the_talker():
    n = 60
    times = [i * DT for i in range(n)]
    pair = {1: (200, 300, 150, 150), 2: (1500, 300, 150, 150)}
    # id 2 talks (oscillating mouth), id 1 is still.
    opens = {1: [0.05] * n, 2: [0.05 + (0.15 if i % 2 else 0.0) for i in range(n)]}
    mode, targets, ids = sr.plan_scene(times, _samples(n, pair, opens), [True] * n, 608)
    assert mode == 'SPEAKER'
    assert set(ids) == {2} and targets[0][0] == 1500
