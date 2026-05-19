"""Simple in-process job scheduler — wraps PMCPipeline for async submission.

V0 uses a ThreadPoolExecutor with one worker by default (training is GPU-bound,
parallelism doesn't help). For production: swap this for a real queue (Celery,
ARQ, Temporal) — the interface stays the same.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from pmc.orchestrator.pipeline import (
    PMCPipeline,
    PipelineConfig,
    PipelineResult,
)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """A submitted pipeline run. Status moves QUEUED → RUNNING → COMPLETED/FAILED."""

    id: str
    user_id: str
    status: JobStatus = JobStatus.QUEUED
    submitted_at: datetime = Field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str = ""
    result: PipelineResult | None = None


class JobScheduler:
    """In-process job queue. Thread-safe; uses one worker by default."""

    def __init__(self, pipeline: PMCPipeline, *, max_workers: int = 1) -> None:
        self.pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._futures: dict[str, Future[PipelineResult]] = {}

    def submit(self, config: PipelineConfig) -> Job:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        job = Job(id=job_id, user_id=config.user_id)
        with self._lock:
            self._jobs[job_id] = job
        future = self._executor.submit(self._run, job_id, config)
        with self._lock:
            self._futures[job_id] = future
        return job

    def _run(self, job_id: str, config: PipelineConfig) -> PipelineResult:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
        try:
            result = self.pipeline.run(config)
            with self._lock:
                job.result = result
                job.status = (
                    JobStatus.FAILED if result.status == "failed" else JobStatus.COMPLETED
                )
                job.completed_at = datetime.now()
                if result.error:
                    job.error = result.error
            return result
        except Exception as e:
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = f"{type(e).__name__}: {e}"
            raise

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(
        self,
        *,
        user_id: str | None = None,
        status: JobStatus | None = None,
    ) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if user_id is not None:
            jobs = [j for j in jobs if j.user_id == user_id]
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.submitted_at, reverse=True)

    def wait(self, job_id: str, timeout: float | None = None) -> Job:
        """Block until the job completes. Returns the final Job state."""
        with self._lock:
            future = self._futures.get(job_id)
            job = self._jobs.get(job_id)
        if future is None or job is None:
            raise KeyError(f"Unknown job_id={job_id!r}")
        try:
            future.result(timeout=timeout)
        except Exception:
            pass  # error is already recorded on the job
        with self._lock:
            return self._jobs[job_id]

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job. Running jobs cannot be cancelled mid-flight."""
        with self._lock:
            future = self._futures.get(job_id)
            job = self._jobs.get(job_id)
        if future is None or job is None:
            return False
        cancelled = future.cancel()
        if cancelled:
            with self._lock:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
        return cancelled

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


__all__ = ["Job", "JobScheduler", "JobStatus"]
