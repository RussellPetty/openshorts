from urllib.parse import quote


def build_video_url(job_id: str, filename: str) -> str:
    """Build a URL whose path segments are safe for browsers and HTTP clients."""
    return f"/videos/{quote(job_id, safe='')}/{quote(filename, safe='')}"
