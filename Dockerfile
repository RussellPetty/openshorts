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

# Install FFmpeg, OpenCV dependencies, and Node.js (required for yt-dlp YouTube challenge solving)
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
    && rm -rf /var/lib/apt/lists/*

# Install Node.js from official binary (nodesource scripts are unreliable)
RUN ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "amd64" ]; then NODE_ARCH="x64"; else NODE_ARCH="$ARCH"; fi \
    && curl -fsSL "https://nodejs.org/dist/v20.18.3/node-v20.18.3-linux-${NODE_ARCH}.tar.xz" \
       | tar -xJ -C /usr/local --strip-components=1 \
    && node --version

# Copy virtual env from builder and make writable for runtime upgrades
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create a non-root user (Moved up)
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

# Create directories including Ultralytics cache config
RUN mkdir -p /app/uploads /app/output /tmp/Ultralytics
# Fix permissions: /app for code/uploads, /tmp/Ultralytics for AI cache, /opt/venv for runtime upgrades
RUN chown -R appuser:appuser /app /tmp/Ultralytics /opt/venv

# Switch to non-root user
USER appuser

# Pre-download YOLO model on build (now running as appuser)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app (update yt-dlp at startup to handle YouTube API changes)
CMD ["sh", "-c", "echo 'Node.js:' && node --version && pip install --quiet --upgrade 'yt-dlp[default]' && uvicorn app:app --host 0.0.0.0 --port 8000"]
