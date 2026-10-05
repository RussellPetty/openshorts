"""Turn raw pipeline failures into messages that tell the user what to do.

A failed job used to surface as "Process failed with exit code 1", which gives
the user nothing to act on. main.py now classifies any exception it dies with
and prints a ``❌ CODE: message`` line that app.py stores as the job error, so
the caller (the Broker Marketplace clips UI) can show it as-is.

Order matters: the first rule whose needles appear in the error text wins, so
specific causes sit above the generic download/processing fallbacks.
"""

UPLOAD_HINT = "download the video and upload the file directly instead"

# (code, needles (lowercase), user-facing message)
_RULES = [
    ("VIDEO_PRIVATE",
     ("this video is private", "private video", "video is private"),
     "This YouTube video is private. Make it public or unlisted, or "
     f"{UPLOAD_HINT}."),
    ("VIDEO_UNAVAILABLE",
     ("unavailable or has been removed", "video unavailable", "has been removed",
      "does not exist", "no longer available", "video is unavailable"),
     "This YouTube video is unavailable or has been removed. Check that the link "
     "is correct and the video is public, then try again."),
    ("VIDEO_RESTRICTED",
     ("age-restricted", "age restricted", "confirm your age", "members-only",
      "members only", "join this channel"),
     "This video is age-restricted or members-only, so it can't be downloaded. "
     f"Please {UPLOAD_HINT}."),
    ("LIVE_STREAM",
     ("live streams are not supported", "is a live stream", "live event",
      "premieres in", "will begin in"),
     "Live streams and upcoming premieres can't be clipped yet. Try again after "
     "the stream has ended and YouTube has finished processing the replay."),
    ("LINK_FORBIDDEN",
     ("http error 403", "http error 401", "403: forbidden", "401: unauthorized"),
     "The video link refused access — it may have expired or require a login. "
     "Use a public link, or upload the file directly."),
    ("LINK_NOT_FOUND",
     ("http error 404", "404: not found", "http error 410"),
     "The video link no longer exists (404). Check the link and try again."),
    ("NOT_A_VIDEO",
     ("downloaded file too small", "invalid data found when processing input",
      "moov atom not found", "could not open video file", "input file not found",
      "no video stream"),
     "We couldn't read that as a video. Make sure the link points to the video "
     "file itself (MP4 works best), or re-export it as MP4 (H.264) and upload it."),
    ("DOWNLOAD_FAILED",
     ("all download services failed", "savenow", "savefrom", "urlopen error",
      "timed out", "connection reset", "temporary failure in name resolution"),
     "We couldn't download this video right now. Try again in a few minutes, or "
     f"{UPLOAD_HINT}."),
    ("OUT_OF_MEMORY",
     ("memoryerror", "cannot allocate memory", "out of memory"),
     "This video was too large to process. Try a shorter video (under 30 minutes) "
     "or a lower-resolution file."),
    ("NO_SPACE",
     ("no space left on device",),
     "The server ran out of space while processing this video. Try again in a "
     "few minutes, or try a shorter video."),
]

GENERIC = ("PROCESSING_FAILED",
           "Something went wrong while processing this video. Please try again; if it "
           "keeps failing, try a different video or upload the file directly.")


def classify(error_text):
    """(code, message) for a raw error string; GENERIC when nothing matches."""
    text = str(error_text or "").lower()
    for code, needles, message in _RULES:
        if any(n in text for n in needles):
            return code, message
    return GENERIC


# Download-service answers that are final: another service will hit the same
# wall, so the chain should stop instead of spending minutes on a fallback.
# Live streams are deliberately NOT here — savefrom handles finished live VODs
# that savenow refuses.
DEFINITIVE_DOWNLOAD_ERRORS = ("VIDEO_PRIVATE", "VIDEO_UNAVAILABLE", "VIDEO_RESTRICTED")


def is_definitive_download_error(error_text):
    return classify(error_text)[0] in DEFINITIVE_DOWNLOAD_ERRORS


def youtube_link_problem(url):
    """A message when a YouTube URL is not a single video, else None.

    Checked at submit so a channel/playlist link fails in a second with a clear
    reason instead of minutes into the download chain."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    if host.endswith("youtu.be"):
        return None if path.strip("/") else NOT_SINGLE_VIDEO
    if "youtube" not in host:
        return None
    if urllib.parse.parse_qs(parsed.query).get("v"):
        return None
    if any(path.startswith(p) and len(path) > len(p) for p in ("/shorts/", "/live/", "/embed/", "/v/")):
        return None
    return NOT_SINGLE_VIDEO


NOT_SINGLE_VIDEO = ("That link points to a YouTube channel, playlist or page, not a single "
                    "video. Open the video you want and paste its link.")
