# Multi-stage build for smaller final image
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Copy and install Python dependencies
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# bgutil POT HTTP server — pull the official prebuilt image instead of
# building from source. Image ships /app/{build,node_modules} ready to run
# via `node build/main.js`; we copy /app + its node binary into the final
# stage. Pinned to 1.3.1 to match the Python plugin.
FROM brainicism/bgutil-ytdlp-pot-provider:1.3.1 AS bgutil

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install FFmpeg, OpenCV deps, and Node.js for yt-dlp's challenge solver.
# bgutil's canvas runtime libs (cairo/pango/jpeg/gif/pixman/rsvg) come along
# for the ride — same packages are also bundled in the brainicism image, but
# having them here lets us run the bgutil binary with our own Node if needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    ca-certificates \
    xz-utils \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libjpeg62-turbo \
    libgif7 \
    libpixman-1-0 \
    librsvg2-2 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js from official binary (nodesource scripts are unreliable)
RUN ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "amd64" ]; then NODE_ARCH="x64"; else NODE_ARCH="$ARCH"; fi \
    && curl -fsSL "https://nodejs.org/dist/v20.18.3/node-v20.18.3-linux-${NODE_ARCH}.tar.xz" \
       | tar -xJ -C /usr/local --strip-components=1 \
    && node --version

# Pull the prebuilt bgutil POT server out of the official image: /app contains
# build/, node_modules/, package.json. We also copy the image's Node 25 binary
# under a separate name so the bgutil server runs with the runtime it was
# built and tested against (its native deps were compiled for that ABI).
ENV BGUTIL_DIR=/opt/bgutil
COPY --from=bgutil /app ${BGUTIL_DIR}
COPY --from=bgutil /usr/local/bin/node /usr/local/bin/node-bgutil
RUN test -f ${BGUTIL_DIR}/build/main.js \
    && /usr/local/bin/node-bgutil --version

# Copy virtual env from builder and make writable for runtime upgrades
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy application code
COPY . .

# Create a non-root user (Moved up)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Create directories including Ultralytics cache config
RUN mkdir -p /app/uploads /app/output /tmp/Ultralytics

# Symlink bgutil into appuser's home so the Python plugin's script-mode
# fallback lookup (~/bgutil-ytdlp-pot-provider/server/*) resolves to our
# /opt/bgutil install. The HTTP provider is the primary path, but script
# mode is a useful safety net when the server probe fails.
RUN mkdir -p /app/bgutil-ytdlp-pot-provider \
    && ln -s /opt/bgutil /app/bgutil-ytdlp-pot-provider/server

# Fix permissions: /app for code/uploads, /tmp/Ultralytics for AI cache,
# /opt/venv for runtime upgrades, /opt/bgutil so the bgutil HTTP server can
# be started by appuser.
RUN chown -R appuser:appuser /app /tmp/Ultralytics /opt/venv /opt/bgutil \
    && chown -hR appuser:appuser /app/bgutil-ytdlp-pot-provider

# Switch to non-root user
USER appuser

# Pre-download YOLO model on build (now running as appuser)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose FastAPI port
EXPOSE 8000

# Container start: launches the bgutil POT server in the background, waits
# for /ping (or surfaces the crash log if it dies), upgrades yt-dlp, then
# execs uvicorn. yt-dlp's bgutil plugin auto-detects 127.0.0.1:4416.
# `exec` keeps uvicorn as PID 1 so SIGTERM still shuts the container down.
# start.sh came in via `COPY . .` above; we already chowned /app to appuser.
USER root
RUN chmod +x /app/start.sh
USER appuser
CMD ["/app/start.sh"]
