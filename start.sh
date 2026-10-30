#!/bin/sh
# Container startup: start the virtual display CloakBrowser needs, then hand
# PID 1 to uvicorn.
#
# Removed 2026-10-30: the bgutil PO-token server launch and the
# `pip install --upgrade yt-dlp` on every boot. Both only served yt-dlp, and no
# download path reaches yt-dlp any more (main.download_youtube_video returns
# from the savenow -> savefrom service chain before the legacy yt-dlp block,
# and /api/transcribe never used it). The upgrade added a network round-trip
# and an unpinned dependency change to every boot.

set -e

cd /app

# Start a virtual display so CloakBrowser can run headed (headless=False), which
# Cloudflare Turnstile requires. The app reads DISPLAY when it launches the browser.
if command -v Xvfb >/dev/null 2>&1; then
  echo "Starting Xvfb virtual display on :99..."
  Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp > /tmp/xvfb.log 2>&1 &
  export DISPLAY=:99
  sleep 1
else
  echo "[startup] Xvfb not found — CloakBrowser headed mode unavailable, savefrom will be skipped"
fi

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
