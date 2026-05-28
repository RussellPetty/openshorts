#!/bin/sh
# Container startup: launch the bgutil POT server in the background, wait
# for it to bind, surface its log to container stdout, upgrade yt-dlp, then
# hand off PID 1 to uvicorn.

set -e

echo "Node.js: $(node --version)"
echo "Node.js (bgutil): $(node-bgutil --version)"
echo "bgutil dir contents:"
ls -la /opt/bgutil

echo "Starting bgutil POT server..."
cd /opt/bgutil
node-bgutil build/main.js --port 4416 > /tmp/bgutil.log 2>&1 &
BGUTIL_PID=$!
echo "bgutil PID=$BGUTIL_PID, waiting for /ping..."

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if curl -sf -o /dev/null --max-time 1 http://127.0.0.1:4416/ping; then
    echo "[startup] bgutil POT server ready on 4416"
    break
  fi
  if ! kill -0 "$BGUTIL_PID" 2>/dev/null; then
    echo "[startup] !!! bgutil CRASHED, log follows:"
    cat /tmp/bgutil.log || true
    echo "[startup] !!! continuing without POT server"
    break
  fi
  sleep 1
done

# Stream bgutil's log to container stdout in the background so subsequent
# crashes/restarts are still visible in Railway runtime logs.
( tail -F /tmp/bgutil.log 2>/dev/null | sed -u 's/^/[bgutil] /' & ) || true

cd /app
pip install --quiet --upgrade 'yt-dlp[default]' || echo "[startup] yt-dlp upgrade failed (continuing)"

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
