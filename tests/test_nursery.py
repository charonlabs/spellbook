from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from spellbook.config import SpellbookConfig
from spellbook.nursery import Nursery


async def _settle() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def _nursery() -> Nursery:
    return Nursery(config=SpellbookConfig(cwd=Path.cwd()))


async def _returns(value: str) -> str:
    return value


async def _raises() -> str:
    raise RuntimeError("boom")


pytestmark = pytest.mark.asyncio


async def test_collect_ready_harvests_completed_job_and_forgets_it() -> None:
    nursery = _nursery()

    job = nursery.submit(
        _returns("done"),
        kind="detect_blocks",
        source="block_manager",
        key="detector",
    )
    await _settle()

    results = nursery.collect_ready()

    assert len(results) == 1
    assert results[0].job.id == job.id
    assert results[0].result == "done"
    assert results[0].error is None
    assert results[0].cancelled is False
    assert nursery.get(job.id) is None
    assert nursery.get_by_key("detector") is None


async def test_collect_ready_filter_defers_non_matching_jobs() -> None:
    nursery = _nursery()

    block_job = nursery.submit(
        _returns("block"),
        kind="detect_blocks",
        source="block_manager",
    )
    bash_job = nursery.submit(_returns("bash"), kind="bash", source="bash")
    await _settle()

    bash_results = nursery.collect_ready(source="bash")
    later_results = nursery.collect_ready()

    assert [result.job.id for result in bash_results] == [bash_job.id]
    assert [result.job.id for result in later_results] == [block_job.id]


async def test_jobs_filters_pending_jobs_by_mode() -> None:
    nursery = _nursery()
    never = asyncio.Event()

    async def _wait_forever() -> str:
        await never.wait()
        return "done"

    render_job = nursery.submit(
        _wait_forever(),
        kind="detect_blocks",
        source="block_manager",
        mode="render_blocking",
    )
    best_effort_job = nursery.submit(
        _wait_forever(),
        kind="summarize_block",
        source="block_manager",
    )

    try:
        assert [
            job.id
            for job in nursery.jobs(source="block_manager", mode="render_blocking")
        ] == [render_job.id]
        assert [
            job.id for job in nursery.jobs(source="block_manager", mode="best_effort")
        ] == [best_effort_job.id]
    finally:
        await nursery.shutdown(cancel=True)


async def test_duplicate_key_cancels_existing_job_and_keeps_new_mapping() -> None:
    nursery = _nursery()
    never = asyncio.Event()

    async def _wait_forever() -> str:
        await never.wait()
        return "old"

    old_job = nursery.submit(
        _wait_forever(),
        kind="summarize_block",
        source="block_manager",
        key="summary:block_1",
    )
    new_job = nursery.submit(
        _returns("new"),
        kind="summarize_block",
        source="block_manager",
        key="summary:block_1",
    )
    await _settle()

    results = nursery.collect_ready()

    assert nursery.get_by_key("summary:block_1") is None
    results_by_id = {result.job.id: result for result in results}
    assert results_by_id[old_job.id].cancelled is True
    assert results_by_id[new_job.id].result == "new"


async def test_wait_harvests_job_and_removes_pending_ready_id() -> None:
    nursery = _nursery()

    job = nursery.submit(_returns("done"), kind="detect_blocks", source="block_manager")
    result = await nursery.wait(job.id)
    await _settle()

    assert result is not None
    assert result.result == "done"
    assert nursery.collect_ready() == []


async def test_shutdown_cancels_pending_jobs_and_returns_errors() -> None:
    nursery = _nursery()
    never = asyncio.Event()

    async def _wait_forever() -> str:
        await never.wait()
        return "late"

    pending = nursery.submit(_wait_forever(), kind="bash", source="bash")
    erroring = nursery.submit(_raises(), kind="bash", source="bash")
    await _settle()

    results = await nursery.shutdown(cancel=True)

    results_by_id = {result.job.id: result for result in results}
    assert results_by_id[pending.id].cancelled is True
    assert isinstance(results_by_id[erroring.id].error, RuntimeError)
    assert nursery.get(pending.id) is None
    assert nursery.get(erroring.id) is None
