"""Concurrent log appends must not clobber progress/results (prod 5-oct-2026)."""
import asyncio
from datetime import datetime

import pytest

fakeredis = pytest.importorskip("fakeredis")

from job_store import RedisJobStore
from models import CaptionSettings, ClipResult, JobData, JobResult, JobStatus


def _job(job_id="j1", status=JobStatus.PROCESSING):
    return JobData(job_id=job_id, status=status, input_url="u",
                   caption_settings=CaptionSettings(), created_at=datetime.utcnow())


def _slow_reads(redis):
    """Yield to the event loop after every read, the way a network round trip
    to real Redis does — that gap is where the lost updates happened."""
    real_get = redis.get

    async def get(key):
        value = await real_get(key)
        await asyncio.sleep(0)
        return value

    redis.get = get
    return redis


def test_interleaved_logs_keep_progress_result_and_every_line():
    async def run():
        store = RedisJobStore(_slow_reads(fakeredis.FakeAsyncRedis()))
        await store.create_job(_job())
        ops = [store.append_log("j1", f"line {i}") for i in range(50)]
        ops.insert(10, store.update_progress("j1", 50, "AI analysis"))
        ops.insert(30, store.set_result("j1", JobResult(clips=[ClipResult(video_url="/v/a.mp4")])))
        ops.insert(40, store.update_progress("j1", 70, "Creating clips"))
        await asyncio.gather(*ops)
        return await store.get_job("j1")

    job = asyncio.run(run())
    assert job.progress_percentage == 70 and job.progress_stage == "Creating clips"
    assert job.result and job.result.clips[0].video_url == "/v/a.mp4"
    assert len(job.logs) == 50


def test_startup_fails_orphaned_jobs_only():
    import app

    async def run():
        store = RedisJobStore(fakeredis.FakeAsyncRedis())
        await store.create_job(_job("running", JobStatus.PROCESSING))
        await store.create_job(_job("waiting", JobStatus.QUEUED))
        await store.create_job(_job("done", JobStatus.COMPLETED))
        await app.fail_orphaned_jobs(store)
        return [await store.get_job(j) for j in ("running", "waiting", "done")]

    running, waiting, done = asyncio.run(run())
    assert running.status == JobStatus.FAILED and "submit the video again" in running.error
    assert waiting.status == JobStatus.FAILED
    assert done.status == JobStatus.COMPLETED
