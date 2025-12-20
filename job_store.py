from typing import Optional, Any
from datetime import datetime
from redis.asyncio import Redis
from models import JobData, JobStatus, JobResult

JOB_TTL_SECONDS = 24 * 60 * 60  # 24 hours
JOB_KEY_PREFIX = "openshorts:job:"


class RedisJobStore:
    def __init__(self, redis: Redis):
        self.redis = redis

    def _key(self, job_id: str) -> str:
        return f"{JOB_KEY_PREFIX}{job_id}"

    async def create_job(self, job: JobData) -> None:
        """Store a new job in Redis with TTL."""
        key = self._key(job.job_id)
        data = job.model_dump_json()
        await self.redis.set(key, data, ex=JOB_TTL_SECONDS)

    async def get_job(self, job_id: str) -> Optional[JobData]:
        """Retrieve a job from Redis."""
        key = self._key(job_id)
        data = await self.redis.get(key)
        if data:
            return JobData.model_validate_json(data)
        return None

    async def update_job(self, job_id: str, **updates: Any) -> Optional[JobData]:
        """Update specific fields of a job."""
        job = await self.get_job(job_id)
        if not job:
            return None

        for field, value in updates.items():
            if hasattr(job, field):
                setattr(job, field, value)

        # Reset TTL on update
        await self.create_job(job)
        return job

    async def append_log(self, job_id: str, message: str) -> None:
        """Append a log message to job."""
        job = await self.get_job(job_id)
        if job:
            job.logs.append(message)
            await self.create_job(job)

    async def set_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None
    ) -> None:
        """Update job status with appropriate timestamps."""
        job = await self.get_job(job_id)
        if not job:
            return

        job.status = status
        now = datetime.utcnow()

        if status == JobStatus.PROCESSING:
            job.started_at = now
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.completed_at = now
            if error:
                job.error = error

        await self.create_job(job)

    async def update_progress(
        self,
        job_id: str,
        percentage: int,
        stage: Optional[str] = None
    ) -> None:
        """Update job progress."""
        updates: dict[str, Any] = {"progress_percentage": percentage}
        if stage:
            updates["progress_stage"] = stage
        await self.update_job(job_id, **updates)

    async def set_result(self, job_id: str, result: JobResult) -> None:
        """Set the job result."""
        await self.update_job(job_id, result=result)
