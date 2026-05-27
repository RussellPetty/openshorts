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

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install FFmpeg, OpenCV dependencies, Node.js + git (Node powers yt-dlp's
# challenge solver and the bgutil POT HTTP server; git is needed to clone bgutil).
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
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js from official binary (nodesource scripts are unreliable)
RUN ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "amd64" ]; then NODE_ARCH="x64"; else NODE_ARCH="$ARCH"; fi \
    && curl -fsSL "https://nodejs.org/dist/v20.18.3/node-v20.18.3-linux-${NODE_ARCH}.tar.xz" \
       | tar -xJ -C /usr/local --strip-components=1 \
    && node --version

# bgutil POT HTTP server — yt-dlp's web client needs a PO Token to bypass
# YouTube's "Sign in to confirm you're not a bot" check from datacenter IPs.
# Pairs with the bgutil-ytdlp-pot-provider Python plugin (pinned to the same
# version). Started in the background by the container CMD.
ENV BGUTIL_VERSION=1.3.1
ENV BGUTIL_DIR=/opt/bgutil-ytdlp-pot-provider
RUN git clone --depth 1 --single-branch --branch ${BGUTIL_VERSION} \
       https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git ${BGUTIL_DIR} \
    && cd ${BGUTIL_DIR}/server \
    && npm ci --no-audit --no-fund \
    && npx tsc

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

# Fix permissions: /app for code/uploads, /tmp/Ultralytics for AI cache,
# /opt/venv for runtime upgrades, /opt/bgutil-ytdlp-pot-provider so the
# bgutil HTTP server can be started by appuser.
RUN chown -R appuser:appuser /app /tmp/Ultralytics /opt/venv /opt/bgutil-ytdlp-pot-provider

# Switch to non-root user
USER appuser

# Pre-download YOLO model on build (now running as appuser)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose FastAPI port
EXPOSE 8000

# Start the bgutil POT server (background, 127.0.0.1:4416), upgrade yt-dlp
# at startup so it tracks YouTube API changes, then exec uvicorn.
# yt-dlp's bgutil plugin auto-detects the local provider at 4416.
# `exec` makes uvicorn PID 1 so SIGTERM still shuts the container down cleanly.
CMD ["sh", "-c", "echo 'Node.js:' && node --version && node /opt/bgutil-ytdlp-pot-provider/server/build/main.js --port 4416 > /tmp/bgutil.log 2>&1 & pip install --quiet --upgrade 'yt-dlp[default]' && exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
