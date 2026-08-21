import os
from urllib.parse import quote, unquote


def build_video_url(job_id: str, filename: str) -> str:
    """Build a URL whose path segments are safe for browsers and HTTP clients."""
    return f"/videos/{quote(job_id, safe='')}/{quote(filename, safe='')}"


def video_filename_from_url(video_url: str) -> str:
    """Recover the on-disk filename from an encoded video URL."""
    filename = unquote(video_url.rsplit('/', 1)[-1])
    if not filename or os.path.basename(filename) != filename:
        raise ValueError("Invalid video filename")
    return filename
