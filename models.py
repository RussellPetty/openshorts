from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from enum import Enum


class CaptionStyleEnum(str, Enum):
    CLASSIC = "classic"
    BOXED = "boxed"
    YELLOW = "yellow"
    MINIMAL = "minimal"
    BOLD = "bold"
    KARAOKE = "karaoke"
    NEON = "neon"
    GRADIENT = "gradient"
    NONE = "none"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CaptionSettings(BaseModel):
    include_captions: bool = True
    style: CaptionStyleEnum = CaptionStyleEnum.NONE
    color: Optional[str] = None  # Hex color e.g., "#FFFFFF"
    outline_color: Optional[str] = None


class ClipResult(BaseModel):
    video_url: str
    title: Optional[str] = None
    description_tiktok: Optional[str] = None
    description_instagram: Optional[str] = None
    description_youtube: Optional[str] = None


class JobResult(BaseModel):
    clips: List[ClipResult] = []
    transcript: Optional[dict] = None  # For editor/subtitle features


class JobData(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    input_url: str
    caption_settings: CaptionSettings
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: int = 0
    progress_stage: Optional[str] = None
    logs: List[str] = []
    result: Optional[JobResult] = None
    error: Optional[str] = None


# Request/Response Models
class ProcessResponseV2(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_percentage: int
    progress_stage: Optional[str]
    logs: List[str]
    created_at: str
    started_at: Optional[str]
    error: Optional[str]


class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[JobResult]
    completed_at: Optional[str]
