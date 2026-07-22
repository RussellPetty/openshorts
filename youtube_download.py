"""
youtube_download.py — shared YouTube download helpers for openshorts.

Provider strategy (used by BOTH app.py's /api/transcribe and main.py's clip pipeline):

    1. savefrom.space, driven by CloakBrowser (stealth Chromium)   <-- pre-yt-dlp fallback
    2. the caller's own yt-dlp path                                <-- last resort

Why CloakBrowser: every keyless downloader middleman we relied on has moved behind
Cloudflare. ytdown.to (our fast path for ~a month) added a full Cloudflare "Just a
moment" interstitial around 2026-06-24; savefrom.space gates POST /getdata behind a
Cloudflare Turnstile widget. A plain HTTP client now gets challenge HTML instead of
JSON, which is why those paths "went dead." CloakBrowser is a source-patched Chromium
that passes Cloudflare's fingerprint/JS challenge, so we let it drive savefrom's real
UI and then read the resulting direct download URL.

Known, accepted caveat: Cloudflare ALSO weighs IP reputation. From a datacenter IP
(Railway) the challenge can escalate to interactive and fail even with a perfect
fingerprint. We intentionally run WITHOUT a proxy first to see whether fingerprint
alone is enough. If it isn't, set CLOAK_PROXY to a residential proxy and this flow
becomes far more reliable (geoip is auto-enabled when a proxy is present).
"""
import os
import re
import time
import urllib.parse
import urllib.request

CHROME_UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
             '(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36')

SAVEFROM_URL = 'https://savefrom.space/'


def sanitize_filename(filename):
    """Remove invalid characters from filename (mirrors main.py.sanitize_filename)."""
    filename = re.sub(r'[<>:"/\\|?*]', '', filename or '')
    filename = filename.replace(' ', '_')
    return filename[:100] or 'youtube_video'


def is_youtube_url(url):
    """Check if a URL is a YouTube URL."""
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or '').lower()
    return any(h in hostname for h in ('youtube.com', 'youtu.be', 'youtube-nocookie.com'))


def _download_file(download_url, output_path, timeout=600):
    """Stream a direct download URL to disk in 1 MB chunks; sanity-check the size."""
    req = urllib.request.Request(download_url, headers={'User-Agent': CHROME_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(output_path, 'wb') as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    if size < 100 * 1024:  # <100 KB means we saved an error page, not a video
        raise RuntimeError(f"downloaded file too small ({size} bytes): {output_path}")
    return output_path


def _wait_for(getter, timeout, interval=1.0, label='result'):
    """Poll getter() until it returns something non-None or we time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        val = getter()
        if val is not None:
            return val
        time.sleep(interval)
    raise RuntimeError(f"timed out after {timeout}s waiting for {label}")


def _wait_past_cloudflare(page, timeout=45):
    """
    If savefrom serves a Cloudflare interstitial ("Just a moment...") on load, give
    CloakBrowser time to clear it and reveal the real form. Best-effort: we don't hard
    fail here — the subsequent wait_for_selector('#url') surfaces a precise error.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            title = (page.title() or '').lower()
        except Exception:
            title = ''
        blocked = ('just a moment' in title) or ('attention required' in title)
        if not blocked:
            try:
                if page.query_selector('#url'):
                    return
            except Exception:
                pass
        time.sleep(1.5)


def download_via_savefrom(url, output_dir='.', want='video', timeout=120):
    """
    Drive savefrom.space in CloakBrowser to obtain a direct download URL, then fetch it.

    Flow (mirrors the real site, which we reverse-engineered from a browser HAR):
      1. goto savefrom.space         — CloakBrowser transparently passes any CF challenge
      2. fill #url, click #submit-btn — renders/executes the Turnstile widget, fires /getdata
      3. capture the /getdata JSON    — gives us the encrypted `id` + title/metadata
      4. POST /getconvert ourselves   — via page.fetch (same-origin cookies + x-csrf-token),
                                        avoiding brittle result-card UI selectors
      5. stream the returned URL to disk

    Returns (output_path, title). Raises on ANY failure so the caller falls back to yt-dlp.
    `want`: 'video' -> mp4 (via googlevideo, reliable; audio included — fine for Whisper),
            'audio' -> mp3 (smaller, routed through savefrom's dlapi relay).
    """
    try:
        from cloakbrowser import launch
    except Exception as e:
        raise RuntimeError(f"cloakbrowser not importable ({type(e).__name__}: {e})")

    fmt = 'mp3' if want == 'audio' else 'mp4'
    proxy = os.environ.get('CLOAK_PROXY') or os.environ.get('YTDLP_PROXY') or None

    launch_kwargs = dict(headless=False, humanize=True, stealth_args=True)
    if proxy:
        launch_kwargs['proxy'] = proxy
        launch_kwargs['geoip'] = True

    print(f"🕵️  savefrom via CloakBrowser (fmt={fmt}, proxy={'yes' if proxy else 'no'}) for {url}")
    step_start = time.time()

    browser = launch(**launch_kwargs)
    try:
        page = browser.new_page()
        captured = {}

        def _on_response(resp):
            try:
                u = resp.url
                if u.endswith('/getdata'):
                    captured['getdata'] = resp.json()
                elif u.endswith('/getconvert'):
                    captured['getconvert'] = resp.json()
            except Exception:
                pass

        page.on('response', _on_response)

        page.goto(SAVEFROM_URL, wait_until='domcontentloaded', timeout=60000)
        _wait_past_cloudflare(page)

        # Submit the URL — this executes the (managed, invisible) Turnstile and fires /getdata.
        page.wait_for_selector('#url', timeout=30000)
        page.fill('#url', url)
        page.click('#submit-btn')

        meta = _wait_for(lambda: captured.get('getdata'), timeout=timeout, label='savefrom /getdata')
        if meta.get('error'):
            raise RuntimeError(f"savefrom getdata error: {meta.get('message')}")
        data = meta.get('data') or {}
        vid_id = data.get('id')
        if not vid_id:
            raise RuntimeError(f"savefrom getdata missing id: {str(meta)[:200]}")
        title = sanitize_filename(data.get('title') or 'youtube_video')
        print(f"   getdata ok: {data.get('title')!r} "
              f"({data.get('duration')}, {data.get('filesize')})")

        # Read Laravel's CSRF token so we can drive /getconvert directly (no fragile UI clicks).
        csrf = page.evaluate(
            "() => { const m = document.querySelector('meta[name=csrf-token]');"
            " return m ? m.content : null; }"
        )

        # Resolve a direct download URL; savefrom may report progress<100 and need re-polling.
        download_url = None
        deadline = time.time() + timeout
        while time.time() < deadline:
            conv = page.evaluate(
                """async ({ id, fmt, csrf }) => {
                    const r = await fetch('/getconvert', {
                        method: 'POST',
                        headers: {
                            'content-type': 'application/json',
                            'accept': '*/*',
                            'x-csrf-token': csrf || '',
                        },
                        body: JSON.stringify({ id, format: fmt }),
                    });
                    try { return await r.json(); }
                    catch (e) { return { error: true, message: 'bad json (http ' + r.status + ')' }; }
                }""",
                {'id': vid_id, 'fmt': fmt, 'csrf': csrf},
            )
            if conv and not conv.get('error'):
                cand = conv.get('download') or conv.get('url')
                if cand and str(cand).startswith(('http://', 'https://')):
                    download_url = cand
                    break
                print(f"   getconvert progress={conv.get('progress')} status={conv.get('status')}")
            else:
                print(f"   getconvert error: {(conv or {}).get('message')}")
            time.sleep(3)

        if not download_url:
            raise RuntimeError("savefrom getconvert never returned a download URL")
    finally:
        try:
            browser.close()
        except Exception:
            pass

    ext = 'mp3' if fmt == 'mp3' else 'mp4'
    output_path = os.path.join(output_dir, f'{title}.{ext}')
    if os.path.exists(output_path):
        os.remove(output_path)
    _download_file(download_url, output_path)
    print(f"✅ savefrom+CloakBrowser done in {time.time() - step_start:.1f}s: {output_path}")
    return output_path, title
