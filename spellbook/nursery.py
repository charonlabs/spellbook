"""A background async job manager.

**Core invariant**: background jobs never silently mutate canonical state when they finish.
Completion only makes an outcome available. A boundary-owned consumer integrates it,
records it, queues footers, or discards it as stale."""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Coroutine, Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from spellbook.config import SpellbookConfig

T = TypeVar("T")

NurseryJobMode = Literal["render_blocking", "best_effort"]
NurseryJobSource = Literal["block_manager", "bash"]
NurseryJobKind = Literal["detect_blocks", "summarize_block", "bash", "block_metrics"]


@dataclass(frozen=True, slots=True)  # Not pydantic because need to hold task
class NurseryJob(Generic[T]):
    """A submitted background job"""

    id: str
    kind: NurseryJobKind
    source: NurseryJobSource
    key: str | None  # for dedupe
    task: asyncio.Task[T]
    mode: NurseryJobMode
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NurseryJobResult(Generic[T]):
    """A harvested job result."""

    job: NurseryJob[T]
    result: T | None = None
    error: BaseException | None = None
    cancelled: bool = False


class AwarenessJobResultSnapshot(BaseModel, Generic[T], frozen=True):
    model_config = ConfigDict(extra="forbid")
    result: T | None = None
    error_message: str | None = None
    cancelled: bool = False


class AwarenessJobSnapshot(BaseModel, Generic[T], frozen=True):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: NurseryJobKind
    source: NurseryJobSource
    key: str | None
    mode: NurseryJobMode
    result: AwarenessJobResultSnapshot[T] | None
    started_at: datetime
    metadata: dict[str, Any]


class AwarenessNurserySnapshot(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    jobs: list[AwarenessJobSnapshot[Any]]


class Nursery:
    """Background async job manager."""

    def __init__(self, config: SpellbookConfig):
        self._config = config
        self._jobs_by_id: dict[str, NurseryJob[Any]] = {}
        self._ids_by_key: dict[str, str] = {}
        self._ready_ids: deque[str] = deque()

    def build_awareness(self) -> AwarenessNurserySnapshot:
        jobs: list[AwarenessJobSnapshot[Any]] = []
        for j in self._jobs_by_id.values():
            result = None
            if j.id in self._ready_ids:
                temp_result = self._result_for(j)
                result = AwarenessJobResultSnapshot(
                    result=temp_result.result,
                    error_message=str(temp_result.error) if temp_result.error else None,
                    cancelled=temp_result.cancelled,
                )
            jobs.append(
                AwarenessJobSnapshot(
                    id=j.id,
                    kind=j.kind,
                    source=j.source,
                    key=j.key,
                    mode=j.mode,
                    result=result,
                    started_at=j.started_at,
                    metadata=j.metadata,
                )
            )
        return AwarenessNurserySnapshot(jobs=jobs)

    def submit(
        self,
        coro: Coroutine[Any, Any, T],
        *,
        kind: NurseryJobKind,
        source: NurseryJobSource,
        key: str | None = None,
        mode: NurseryJobMode = "best_effort",
        metadata: dict[str, Any] | None = None,
    ) -> NurseryJob[T]:
        """Submit a coroutine as a managed background job.

        If `key` matches an already-managed job, the existing job is cancelled before the new one is started."""

        if key is not None:
            existing_id = self._ids_by_key.get(key)
            if existing_id is not None:
                existing = self._jobs_by_id.get(existing_id)
                if existing is not None:
                    existing.task.cancel()
                del self._ids_by_key[key]

        job_id = f"job_{uuid4().hex}"
        task = asyncio.create_task(coro)
        job = NurseryJob(
            id=job_id,
            kind=kind,
            source=source,
            key=key,
            task=task,
            mode=mode,
            metadata=metadata or {},
        )

        self._jobs_by_id[job_id] = job
        if key is not None:
            self._ids_by_key[key] = job.id

        task.add_done_callback(
            lambda _task, ready_id=job.id: self._ready_ids.append(ready_id)
        )
        return job

    def collect_ready(
        self,
        *,
        source: NurseryJobSource | None = None,
        kind: NurseryJobKind | None = None,
        mode: NurseryJobMode | None = None,
    ) -> list[NurseryJobResult[Any]]:

        results: list[NurseryJobResult[Any]] = []
        deferred: deque[str] = deque()

        while self._ready_ids:
            job_id = self._ready_ids.popleft()
            job = self._jobs_by_id.get(job_id)
            if job is None:
                continue
            if not self._matches(job, source=source, kind=kind, mode=mode):
                deferred.append(job_id)
                continue
            self._forget(job)
            results.append(self._result_for(job))
        self._ready_ids = deferred
        return results

    def jobs(
        self,
        *,
        source: NurseryJobSource | None = None,
        kind: NurseryJobKind | None = None,
        mode: NurseryJobMode | None = None,
    ) -> list[NurseryJob[Any]]:
        """Return pending jobs matching the optional filters."""
        return [
            job
            for job in self._jobs_by_id.values()
            if self._matches(job, source=source, kind=kind, mode=mode)
        ]

    def get(self, job_id: str) -> NurseryJob[Any] | None:
        """Get a job by id, if it exists."""
        return self._jobs_by_id.get(job_id)

    def get_by_key(self, key: str) -> NurseryJob[Any] | None:
        """Get a job by key, if it exists."""
        id = self._ids_by_key.get(key)
        if id is None:
            return None
        return self._jobs_by_id.get(id)

    async def wait(self, job_id: str) -> NurseryJobResult[Any] | None:
        """Wait for a specific managed job and harvest the result."""

        job = self._jobs_by_id.get(job_id)
        if job is None:
            return None

        try:
            await job.task
        except BaseException:
            # result error captured below
            pass
        self._forget(job)
        if self._ready_ids:
            self._ready_ids = deque(qid for qid in self._ready_ids if qid != job_id)
        return self._result_for(job)

    async def shutdown(self, *, cancel: bool = True) -> list[NurseryJobResult[Any]]:
        jobs = list(self._jobs_by_id.values())
        if cancel:
            for j in jobs:
                if not j.task.done():
                    j.task.cancel()

        if len(jobs) > 0:
            await asyncio.gather(*(j.task for j in jobs), return_exceptions=True)

        results = [self._result_for(j) for j in jobs]
        self._jobs_by_id.clear()
        self._ids_by_key.clear()
        self._ready_ids.clear()
        return results

    def _forget(self, job: NurseryJob[Any]) -> None:
        self._jobs_by_id.pop(job.id, None)
        if job.key is not None and self._ids_by_key.get(job.key) == job.id:
            del self._ids_by_key[job.key]

    def _matches(
        self,
        job: NurseryJob[Any],
        *,
        source: NurseryJobSource | None,
        kind: NurseryJobKind | None,
        mode: NurseryJobMode | None,
    ) -> bool:
        return (
            (source is None or job.source == source)
            and (kind is None or job.kind == kind)
            and (mode is None or job.mode == mode)
        )

    def _result_for(self, job: NurseryJob[Any]) -> NurseryJobResult[Any]:
        if job.task.cancelled():
            return NurseryJobResult(job=job, cancelled=True)

        try:
            return NurseryJobResult(job=job, result=job.task.result())
        except BaseException as e:
            return NurseryJobResult(job=job, error=e)
