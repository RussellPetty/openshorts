# OpenShorts API v2 Documentation

Base URL: `https://your-domain.com/api/v2`

## Overview

The v2 API provides Redis-backed video processing with persistent job tracking, progress updates, and 24-hour job retention. Submit YouTube URLs or video links to automatically generate viral short-form clips with AI-powered analysis.

---

## Authentication

All endpoints require a Gemini API key for AI analysis. Provide it via:

1. **Header** (per-request): `X-Gemini-Key: your-api-key`
2. **Environment variable** (server default): `GEMINI_API_KEY`

If both are provided, the header takes precedence.

---

## Endpoints

### Submit Video for Processing

```
POST /api/v2/process
```

Submit a video URL to generate short-form clips.

#### Query Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | string | Yes | - | Video URL or YouTube link |
| `include_captions` | boolean | No | `true` | Add captions to output videos |
| `caption_style` | string | No | `none` | Caption style preset (see below) |
| `caption_color` | string | No | - | Custom text color (hex, e.g., `#FFFFFF`) |
| `caption_outline_color` | string | No | - | Custom outline color (hex, e.g., `#000000`) |

#### Caption Styles

| Style | Description |
|-------|-------------|
| `none` | No captions |
| `classic` | White text with black outline |
| `boxed` | White text on semi-transparent background |
| `yellow` | Yellow cinema-style subtitles |
| `minimal` | Light, lowercase text |
| `bold` | Large uppercase text with thick outline |
| `karaoke` | Word-by-word highlighting |
| `neon` | Magenta/pink glowing effect |
| `gradient` | Red-to-blue gradient text |

#### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-Gemini-Key` | No* | Gemini API key (*required if not set in server env) |

#### Example Request

```bash
curl -X POST "https://your-domain.com/api/v2/process?url=https://www.youtube.com/watch?v=dQw4w9WgXcQ&include_captions=true&caption_style=classic" \
  -H "X-Gemini-Key: your-gemini-api-key"
```

#### Response

```json
{
  "job_id": "93310bfb-7da8-4071-8e51-8fd600cd6575",
  "status": "queued"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Job successfully queued |
| 400 | Missing API key or invalid parameters |
| 503 | Redis not configured |

---

### Get Job Status

```
GET /api/v2/jobs/{job_id}
```

Retrieve the current status, progress, and logs for a processing job.

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | The job ID returned from `/process` |

#### Example Request

```bash
curl "https://your-domain.com/api/v2/jobs/93310bfb-7da8-4071-8e51-8fd600cd6575"
```

#### Response

```json
{
  "job_id": "93310bfb-7da8-4071-8e51-8fd600cd6575",
  "status": "processing",
  "progress_percentage": 50,
  "progress_stage": "AI analysis",
  "logs": [
    "Job 93310bfb-7da8-4071-8e51-8fd600cd6575 queued.",
    "Job started by worker.",
    "[youtube] dQw4w9WgXcQ: Downloading...",
    "Transcribing video with Faster-Whisper...",
    "Analyzing with Gemini..."
  ],
  "created_at": "2025-12-20T22:56:14.446965",
  "started_at": "2025-12-20T22:56:14.453530",
  "error": null
}
```

#### Job Status Values

| Status | Description |
|--------|-------------|
| `queued` | Job is waiting to be processed |
| `processing` | Job is currently being processed |
| `completed` | Job finished successfully |
| `failed` | Job encountered an error |

#### Progress Stages

| Stage | Percentage | Description |
|-------|------------|-------------|
| Downloading video | 10% | Fetching video from URL |
| Transcribing audio | 30% | Speech-to-text with Faster-Whisper |
| AI analysis | 50% | Gemini identifying viral moments |
| Creating clips | 70% | Extracting and processing clips |
| Finalizing | 90% | Final encoding and output |

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | Job not found |
| 503 | Redis not configured |

---

### Get Job Result

```
GET /api/v2/jobs/{job_id}/result
```

Retrieve the completed video clips and metadata.

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | The job ID returned from `/process` |

#### Example Request

```bash
curl "https://your-domain.com/api/v2/jobs/93310bfb-7da8-4071-8e51-8fd600cd6575/result"
```

#### Response

```json
{
  "job_id": "93310bfb-7da8-4071-8e51-8fd600cd6575",
  "status": "completed",
  "result": {
    "clips": [
      {
        "video_url": "/videos/93310bfb-7da8-4071-8e51-8fd600cd6575/Video_Title_clip_1.mp4",
        "title": "AI Generated Title for YouTube Shorts",
        "description_tiktok": "Engaging TikTok description with hashtags... #viral #fyp",
        "description_instagram": "Instagram-optimized caption... #reels #trending",
        "description_youtube": "YouTube Shorts title and description"
      },
      {
        "video_url": "/videos/93310bfb-7da8-4071-8e51-8fd600cd6575/Video_Title_clip_2.mp4",
        "title": "Another Viral Moment",
        "description_tiktok": "...",
        "description_instagram": "...",
        "description_youtube": "..."
      }
    ]
  },
  "completed_at": "2025-12-20T22:59:14.859380"
}
```

#### Clip Object

| Field | Type | Description |
|-------|------|-------------|
| `video_url` | string | Relative URL to download the video clip |
| `title` | string | AI-generated title for the clip |
| `description_tiktok` | string | TikTok-optimized description with hashtags |
| `description_instagram` | string | Instagram Reels description |
| `description_youtube` | string | YouTube Shorts title/description |

#### Downloading Videos

Append the `video_url` to your base domain:

```bash
curl -o clip.mp4 "https://your-domain.com/videos/93310bfb-7da8-4071-8e51-8fd600cd6575/Video_Title_clip_1.mp4"
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | Job not found |
| 503 | Redis not configured |

---

## Polling Strategy

For real-time progress updates, poll the status endpoint:

```javascript
async function waitForCompletion(jobId) {
  const pollInterval = 2000; // 2 seconds

  while (true) {
    const response = await fetch(`/api/v2/jobs/${jobId}`);
    const data = await response.json();

    console.log(`Progress: ${data.progress_percentage}% - ${data.progress_stage}`);

    if (data.status === 'completed') {
      return await fetch(`/api/v2/jobs/${jobId}/result`).then(r => r.json());
    }

    if (data.status === 'failed') {
      throw new Error(data.error);
    }

    await new Promise(resolve => setTimeout(resolve, pollInterval));
  }
}
```

---

## Error Responses

All errors return a JSON object with a `detail` field:

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

| Status | Detail | Solution |
|--------|--------|----------|
| 400 | "Missing Gemini API key..." | Provide `X-Gemini-Key` header or set `GEMINI_API_KEY` env |
| 400 | "Invalid caption_style..." | Use one of the valid caption styles |
| 404 | "Job not found" | Check job ID or job may have expired (24h TTL) |
| 503 | "v2 API requires Redis..." | Configure `REDIS_URL` environment variable |

---

## Job Retention

- Jobs are stored in Redis with a **24-hour TTL**
- After 24 hours, job data and video files are automatically deleted
- Download your videos before expiration

---

## Rate Limits

- Default: **5 concurrent jobs** (configurable via `MAX_CONCURRENT_JOBS` env)
- No per-user rate limiting (jobs are queued FIFO)

---

## Complete Example

```bash
# 1. Submit a video
JOB_ID=$(curl -s -X POST \
  "https://your-domain.com/api/v2/process?url=https://youtube.com/watch?v=VIDEO_ID&caption_style=bold" \
  -H "X-Gemini-Key: your-key" | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# 2. Poll for status
while true; do
  STATUS=$(curl -s "https://your-domain.com/api/v2/jobs/$JOB_ID" | jq -r '.status')
  echo "Status: $STATUS"

  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi

  sleep 5
done

# 3. Get results
curl -s "https://your-domain.com/api/v2/jobs/$JOB_ID/result" | jq '.result.clips'

# 4. Download first clip
VIDEO_URL=$(curl -s "https://your-domain.com/api/v2/jobs/$JOB_ID/result" | jq -r '.result.clips[0].video_url')
curl -o clip.mp4 "https://your-domain.com$VIDEO_URL"
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | Yes | - | Redis connection string |
| `GEMINI_API_KEY` | No | - | Default Gemini API key |
| `MAX_CONCURRENT_JOBS` | No | `5` | Max parallel processing jobs |
| `YOUTUBE_COOKIES` | No | - | YouTube cookies for restricted videos |
