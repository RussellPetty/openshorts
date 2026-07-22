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

# savefrom.space monetizes with popunder + overlay ads. Block these networks and
# auto-close any ad popup/tab so they can't hijack or cover the download flow.
_AD_HOST_HINTS = (
    'googlesyndication', 'doubleclick', 'adservice', 'adexchang', 'adnxs',
    'popads', 'popcash', 'propellerads', 'propeller', 'adsterra', 'hilltopads',
    'clickadu', 'taboola', 'outbrain', 'usrpubtrk', 'acscdn', 'aclib',
    'poperblock', 'pushnotif', 'mgid', 'adcash', 'onclickalgo',
)


def _safe_close(pg):
    try:
        pg.close()
    except Exception:
        pass


def _ad_router(route):
    try:
        if any(h in route.request.url.lower() for h in _AD_HOST_HINTS):
            route.abort()
            return
    except Exception:
        pass
    try:
        route.continue_()
    except Exception:
        pass


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


def _snapshot(page):
    """Best-effort page state for diagnosing why savefrom stalled (CF challenge vs. flow)."""
    def _safe(fn, default=''):
        try:
            return fn()
        except Exception:
            return default
    title = _safe(lambda: page.title())
    url = _safe(lambda: page.url)
    cf_iframes = _safe(lambda: len(page.query_selector_all('iframe[src*="challenges.cloudflare.com"]')), 0)
    captcha_visible = _safe(lambda: page.is_visible('#captchaContainer'), False)
    err = (_safe(lambda: (page.inner_text('#error-text') or '').strip())
           or _safe(lambda: (page.inner_text('#error') or '').strip()))
    body = _safe(lambda: ' '.join((page.inner_text('body') or '').split())[:200])
    return (f"title={title!r} url={url} cf_challenge_iframes={cf_iframes} "
            f"captchaVisible={captcha_visible} errorText={err!r} body[:200]={body!r}")


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
        seen = {'getdata_req': False, 'getconvert_req': False}

        def _on_request(req):
            try:
                if req.url.endswith('/getdata'):
                    seen['getdata_req'] = True
                elif req.url.endswith('/getconvert'):
                    seen['getconvert_req'] = True
            except Exception:
                pass

        def _on_response(resp):
            try:
                u = resp.url
                if u.endswith('/getdata'):
                    captured['getdata'] = resp.json()
                elif u.endswith('/getconvert'):
                    captured['getconvert'] = resp.json()
            except Exception:
                pass

        page.on('request', _on_request)
        page.on('response', _on_response)

        # Neutralize savefrom's ads: block ad networks, auto-close popunders/extra tabs.
        page.route('**/*', _ad_router)
        page.on('popup', _safe_close)
        page.context.on('page', lambda p: None if p is page else _safe_close(p))

        page.goto(SAVEFROM_URL, wait_until='domcontentloaded', timeout=60000)
        _wait_past_cloudflare(page)
        try:
            print(f"   page loaded: title={page.title()!r} url={page.url}")
        except Exception:
            pass

        # Enter the URL like a human. page.fill() alone left savefrom's validator seeing an
        # empty field ("Please paste a valid YouTube URL", getdata never fired), so we type
        # real keystrokes AND set value + dispatch the events its JS listens on, then wait
        # a beat before submitting so validation passes.
        page.wait_for_selector('#url', timeout=30000)
        box = page.locator('#url')
        box.click()
        box.fill('')
        try:
            box.press_sequentially(url, delay=60)
        except Exception:
            page.type('#url', url, delay=60)
        page.evaluate(
            """(v) => {
                const el = document.querySelector('#url');
                if (el) {
                    el.value = v;
                    for (const t of ['input', 'change', 'keyup', 'blur'])
                        el.dispatchEvent(new Event(t, { bubbles: true }));
                }
            }""",
            url,
        )
        time.sleep(1.0)

        # Submit and wait for the Turnstile-gated /getdata to fire. The first click is often
        # swallowed by an ad popunder, so re-click every ~10s until getdata is actually requested.
        meta = None
        deadline = time.time() + timeout
        clicks = 0
        last_click = 0.0
        while time.time() < deadline:
            meta = captured.get('getdata')
            if meta is not None:
                break
            now = time.time()
            if not seen['getdata_req'] and (now - last_click) > 10 and clicks < 6:
                try:
                    page.click('#submit-btn', timeout=5000)
                    clicks += 1
                    print(f"   submit click #{clicks}")
                except Exception as e:
                    print(f"   submit click err: {str(e)[:80]}")
                last_click = now
            time.sleep(1.5)
        if meta is None:
            try:
                page.screenshot(path='/tmp/savefrom_fail.png')
            except Exception:
                pass
            raise RuntimeError(
                f"savefrom /getdata timed out after {timeout}s "
                f"(getdata_requested={seen['getdata_req']}, submit_clicks={clicks}) — {_snapshot(page)}"
            )
        if meta.get('error'):
            raise RuntimeError(f"savefrom getdata error: {meta.get('message')}")
        data = meta.get('data') or {}
        title = sanitize_filename(data.get('title') or 'youtube_video')
        print(f"   getdata ok: {data.get('title')!r} "
              f"({data.get('duration')}, {data.get('filesize')})")

        # Click the "Convert <fmt>" link in the result card. That's what fires /getconvert with
        # the correct per-format id — the DOM link is <a onclick="handleDownload(this,'<id>',
        # '<fmt>')">. (Calling /getconvert via fetch with getdata's id doesn't match every video,
        # e.g. livestream VODs.) We capture the /getconvert response for the direct download URL
        # and re-click through ad popunders.
        fmt_sel = "a[onclick*=\"'%s'\"]" % fmt
        try:
            page.wait_for_selector(fmt_sel, timeout=30000)
        except Exception:
            fmt_sel = "a[onclick*='handleDownload']"
            page.wait_for_selector(fmt_sel, timeout=15000)

        download_url = None
        deadline = time.time() + timeout
        clicks = 0
        last_click = 0.0
        while time.time() < deadline:
            conv = captured.get('getconvert')
            if conv and not conv.get('error'):
                cand = conv.get('download') or conv.get('url')
                if cand and str(cand).startswith(('http://', 'https://')):
                    download_url = cand
                    break
                if conv.get('progress') is not None:
                    print(f"   getconvert progress={conv.get('progress')} status={conv.get('status')}")
            now = time.time()
            if download_url is None and (now - last_click) > 12 and clicks < 6:
                try:
                    page.locator(fmt_sel).first.click(timeout=5000)
                    clicks += 1
                    print(f"   convert click #{clicks}")
                except Exception as e:
                    print(f"   convert click err: {str(e)[:80]}")
                last_click = now
            time.sleep(1.5)

        if not download_url:
            try:
                page.screenshot(path='/tmp/savefrom_fail.png')
            except Exception:
                pass
            raise RuntimeError(
                f"savefrom getconvert never returned a download URL "
                f"(getconvert_requested={seen['getconvert_req']}, convert_clicks={clicks}) — {_snapshot(page)}"
            )
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


def download_via_savenow(url, output_dir='.', want='video'):
    """
    Download via the savenow / video-download-api REST API. Their backend talks to
    YouTube, so Railway's IP never does. Requires SAVENOW_API_KEY. Returns (path, title);
    raises on any failure so the chain can move on to the next service.
    """
    import json
    api_key = os.environ.get('SAVENOW_API_KEY')
    if not api_key:
        raise RuntimeError('SAVENOW_API_KEY not set')
    host = os.environ.get('SAVENOW_API_HOST', 'https://p.savenow.to').rstrip('/')
    fmt = os.environ.get('SAVENOW_FORMAT', '1080')

    def _get_json(u, timeout=30):
        req = urllib.request.Request(u, headers={'User-Agent': CHROME_UA, 'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    start = time.time()
    print(f"📥 savenow (fmt={fmt}) for {url}")
    submit = f"{host}/ajax/download.php?" + urllib.parse.urlencode({
        'url': url, 'format': fmt, 'apikey': api_key,
        'add_info': '1', 'allow_extended_duration': '1', 'no_merge': '0'})
    meta = _get_json(submit)
    if not meta.get('success') or not meta.get('id'):
        raise RuntimeError(f"savenow submit failed: {str(meta)[:200]}")
    title = sanitize_filename(meta.get('title') or (meta.get('info') or {}).get('title') or 'youtube_video')
    prog = meta.get('progress_url') or f"{host}/ajax/progress.php?id={urllib.parse.quote(str(meta['id']))}"

    download_url = None
    deadline = time.time() + 240
    poll = 0
    ERROR_HINTS = ('not supported', 'not available', 'unavailable', 'error', 'failed',
                   'private', 'removed', 'copyright', 'no video', 'invalid')
    while time.time() < deadline:
        poll += 1
        try:
            job = _get_json(prog, timeout=30)
        except Exception as e:
            print(f"   savenow poll {poll} err: {str(e)[:100]}")
            time.sleep(3)
            continue
        cand = job.get('download_url') or job.get('url')
        if cand and str(cand).startswith(('http://', 'https://')):
            download_url = cand
            print(f"   savenow ready after {poll} polls in {time.time() - start:.1f}s")
            break
        # Fail fast on savenow's error states (e.g. progress=1000 "Live streams are not
        # supported") so the chain falls back to savefrom immediately, not after 240s.
        text = str(job.get('text') or '')
        if str(job.get('progress')) == '1000' or any(k in text.lower() for k in ERROR_HINTS):
            raise RuntimeError(f"savenow cannot handle this video: {text[:120]}")
        if poll % 5 == 1:
            print(f"   savenow progress={job.get('progress')} {job.get('text')}")
        time.sleep(3)
    if not download_url:
        raise RuntimeError('savenow did not complete within 240s')

    output_path = os.path.join(output_dir, f'{title}.mp4')
    if os.path.exists(output_path):
        os.remove(output_path)
    _download_file(download_url, output_path)
    print(f"✅ savenow done in {time.time() - start:.1f}s: {output_path}")
    return output_path, title


def download_youtube(url, output_dir='.', want='video'):
    """
    Shared YouTube download chain for /api/transcribe and the clip pipeline.
    Order: savenow (clean REST API, 1080p) -> savefrom via CloakBrowser (stealth browser,
    passes Cloudflare). NO yt-dlp — YouTube blocks Railway's datacenter IP for it.
    Returns (path, title); raises RuntimeError only if EVERY service fails.
    """
    errors = []
    for name, fn in (('savenow', download_via_savenow), ('savefrom', download_via_savefrom)):
        try:
            return fn(url, output_dir, want=want)
        except Exception as e:
            msg = f"{name}: {type(e).__name__}: {e}"
            print(f"⚠️  {msg}")
            errors.append(msg)
    raise RuntimeError("all download services failed -> " + " | ".join(errors))
