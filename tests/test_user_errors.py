"""Failure text -> what the user should do differently."""
import user_errors as ue

# Real error strings seen in prod logs (5-oct-2026) and from the stdlib.
UNAVAILABLE = ("RuntimeError: all download services failed -> savenow: RuntimeError: savenow "
               "cannot handle this video: This video is unavailable or has been removed. | "
               "savefrom: RuntimeError: savefrom /getdata timed out after 120s")
SERVICES_DOWN = ("RuntimeError: all download services failed -> savenow: RuntimeError: savenow "
                 "did not complete within 240s | savefrom: RuntimeError: savefrom /getdata timed out")


def test_removed_video_names_the_cause_not_the_timeout():
    code, msg = ue.classify(UNAVAILABLE)
    assert code == "VIDEO_UNAVAILABLE"
    assert "public" in msg


def test_services_down_says_retry_or_upload():
    code, msg = ue.classify(SERVICES_DOWN)
    assert code == "DOWNLOAD_FAILED"
    assert "upload" in msg and "few minutes" in msg


def test_private_and_restricted():
    assert ue.classify("savenow cannot handle this video: This video is private")[0] == "VIDEO_PRIVATE"
    assert ue.classify("Sign in to confirm your age")[0] == "VIDEO_RESTRICTED"


def test_live_stream_is_not_definitive_savefrom_handles_vods():
    text = "savenow cannot handle this video: Live streams are not supported"
    assert ue.classify(text)[0] == "LIVE_STREAM"
    assert not ue.is_definitive_download_error(text)


def test_removed_is_definitive_so_the_chain_stops():
    assert ue.is_definitive_download_error(
        "savenow cannot handle this video: This video is unavailable or has been removed.")


def test_direct_link_http_errors():
    assert ue.classify("HTTPError: HTTP Error 403: Forbidden")[0] == "LINK_FORBIDDEN"
    assert ue.classify("HTTPError: HTTP Error 404: Not Found")[0] == "LINK_NOT_FOUND"
    assert ue.classify("RuntimeError: downloaded file too small (512 bytes): x.mp4")[0] == "NOT_A_VIDEO"


def test_unknown_error_gets_the_generic_actionable_message():
    code, msg = ue.classify("KeyError: 'segments'")
    assert (code, msg) == ue.GENERIC
    assert "try again" in msg.lower()


def test_youtube_link_problem():
    ok = ["https://www.youtube.com/watch?v=arj7oStGLkU",
          "https://www.youtube.com/watch?v=arj7oStGLkU&list=PL123",
          "https://youtu.be/arj7oStGLkU",
          "https://www.youtube.com/shorts/abc123",
          "https://www.youtube.com/live/abc123",
          "https://example.com/video.mp4"]
    bad = ["https://www.youtube.com/@somechannel",
           "https://www.youtube.com/playlist?list=PL123",
           "https://www.youtube.com/",
           "https://youtu.be/"]
    assert all(ue.youtube_link_problem(u) is None for u in ok)
    assert all(ue.youtube_link_problem(u) for u in bad)
