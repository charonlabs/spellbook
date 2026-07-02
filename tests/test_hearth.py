from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from pathlib import Path
from typing import Any, cast

import pytest

from spellbook.config import SpellbookConfig
from spellbook.hearth import (
    HEARTH_HEARTBEAT,
    HearthGaugeSnapshot,
    HearthScheduler,
    HearthSettings,
    compose_hearth_crackle_lines,
)
from spellbook.ir_types import (
    IRInboundMessage,
    IRSemanticBlock,
    IRSemanticBlockFacet,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRUserTextBlock,
    SemanticBlockMode,
)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


class _Clock:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value


@dataclass(frozen=True)
class _Budget:
    current_input_tokens: int | None = 123_000
    max_tokens: int = 1_000_000
    regime: str = "calm"


@dataclass(frozen=True)
class _Awareness:
    budget: _Budget
    semantic_blocks: list[IRSemanticBlock]


class _FakeHomunculus:
    def __init__(
        self,
        *,
        settings: HearthSettings,
        semantic_blocks: list[IRSemanticBlock] | None = None,
    ) -> None:
        self.hearth_settings = settings
        self.semantic_blocks = semantic_blocks or []

    def build_awareness(self) -> _Awareness:
        return _Awareness(
            budget=_Budget(),
            semantic_blocks=self.semantic_blocks,
        )


class _FakeQueue:
    def __init__(self) -> None:
        self.pending = False

    def has_pending(self) -> bool:
        return self.pending


class _FakeSession:
    def __init__(
        self,
        *,
        tmp_path: Path,
        settings: HearthSettings,
        semantic_blocks: list[IRSemanticBlock] | None = None,
    ) -> None:
        self.config = SpellbookConfig(cwd=tmp_path, local_timezone="UTC")
        self.homunculus = _FakeHomunculus(
            settings=settings,
            semantic_blocks=semantic_blocks,
        )
        self.inbound_queue = _FakeQueue()
        self.state = "idle"


class _FakeRuntime:
    def __init__(
        self,
        *,
        session: _FakeSession,
        last_activity_time: datetime,
        clock: _Clock,
    ) -> None:
        self.session = session
        self._last_activity_time = last_activity_time
        self.clock = clock
        self.submitted: list[IRInboundMessage] = []

    @property
    def last_activity_time(self) -> datetime:
        return self._last_activity_time

    async def submit_hearth_crackle(self, message: IRInboundMessage) -> bool:
        if self.session.state != "idle" or self.session.inbound_queue.has_pending():
            return False
        self.submitted.append(message)
        self._last_activity_time = self.clock()
        return True


def _facet(idx: int, title: str = "Doorway moment") -> IRSemanticBlockFacet:
    return IRSemanticBlockFacet(
        id=f"facet_{idx}",
        title=title,
        description="the abstraction collapsed into a concrete next step",
        start_block=idx,
        end_block=idx,
        resources=[],
    )


def _semantic_block(
    idx: int,
    *,
    mode: SemanticBlockMode = "summary",
    facets: list[IRSemanticBlockFacet] | None = None,
) -> IRSemanticBlock:
    block_range = IRSemanticBlockRange(
        id=f"range_{idx}",
        title=f"Block {idx}",
        start_block=idx,
        end_block=idx,
        completed=True,
    )
    return IRSemanticBlock(
        id=f"block_{idx}",
        idx=idx,
        title=f"Block {idx}",
        range=block_range,
        mode=mode,
        toks=None,
        full_toks=None,
        available_modes=["full", "summary"],
        artifacts=[
            IRSemanticBlockSummary(
                headline=f"Summary {idx}",
                text="A compacted chapter.",
                facets=facets or [],
                open_thread=None,
                toks=None,
            )
        ],
    )


def _gauge() -> HearthGaugeSnapshot:
    return HearthGaugeSnapshot(
        current_input_tokens=123_000,
        max_tokens=1_000_000,
        regime="calm",
    )


@pytest.mark.asyncio
async def test_scheduler_disabled_by_default_never_fires(tmp_path: Path) -> None:
    clock = _Clock(_dt("2026-06-12T01:00:00"))
    session = _FakeSession(tmp_path=tmp_path, settings=HearthSettings())
    runtime = _FakeRuntime(
        session=session,
        last_activity_time=_dt("2026-06-11T23:00:00"),
        clock=clock,
    )
    scheduler = HearthScheduler(cast(Any, runtime), clock=clock)

    fired = await scheduler.tick()

    assert fired is False
    assert runtime.submitted == []


@pytest.mark.asyncio
async def test_scheduler_fires_after_interval_and_spaces_evenly(
    tmp_path: Path,
) -> None:
    clock = _Clock(_dt("2026-06-12T00:05:00"))
    session = _FakeSession(
        tmp_path=tmp_path,
        settings=HearthSettings(enabled=True, interval_minutes=5),
    )
    runtime = _FakeRuntime(
        session=session,
        last_activity_time=_dt("2026-06-12T00:00:00"),
        clock=clock,
    )
    scheduler = HearthScheduler(cast(Any, runtime), clock=clock)

    assert await scheduler.tick() is True
    assert runtime.last_activity_time == clock.value
    first = runtime.submitted[-1]
    first_block = first.blocks[0]
    assert isinstance(first_block, IRUserTextBlock)
    assert first_block.origin == "system"
    assert first.source_metadata["source"] == "hearth.scheduler"
    assert first.source_metadata["origin"] == "ambient_scheduler"
    assert first_block.text.splitlines()[0] == HEARTH_HEARTBEAT

    clock.value = _dt("2026-06-12T00:09:59")
    assert await scheduler.tick() is False
    clock.value = _dt("2026-06-12T00:10:00")
    assert await scheduler.tick() is True
    assert len(runtime.submitted) == 2


@pytest.mark.asyncio
async def test_scheduler_respects_quiet_hours_spanning_midnight(
    tmp_path: Path,
) -> None:
    clock = _Clock(_dt("2026-06-12T23:30:00"))
    session = _FakeSession(
        tmp_path=tmp_path,
        settings=HearthSettings(
            enabled=True,
            interval_minutes=5,
            quiet_hours="23:00-07:00",
        ),
    )
    runtime = _FakeRuntime(
        session=session,
        last_activity_time=_dt("2026-06-12T22:00:00"),
        clock=clock,
    )
    scheduler = HearthScheduler(cast(Any, runtime), clock=clock)

    assert await scheduler.tick() is False
    clock.value = _dt("2026-06-13T06:59:00")
    assert await scheduler.tick() is False
    clock.value = _dt("2026-06-13T07:00:00")
    assert await scheduler.tick() is True


@pytest.mark.asyncio
async def test_scheduler_never_fires_while_running_or_queue_nonempty(
    tmp_path: Path,
) -> None:
    clock = _Clock(_dt("2026-06-12T01:00:00"))
    session = _FakeSession(
        tmp_path=tmp_path,
        settings=HearthSettings(enabled=True, interval_minutes=5),
    )
    runtime = _FakeRuntime(
        session=session,
        last_activity_time=_dt("2026-06-12T00:00:00"),
        clock=clock,
    )
    scheduler = HearthScheduler(cast(Any, runtime), clock=clock)

    session.state = "running"
    assert await scheduler.tick() is False
    session.state = "idle"
    session.inbound_queue.pending = True
    assert await scheduler.tick() is False
    assert runtime.submitted == []


@pytest.mark.asyncio
async def test_scheduler_reads_config_changes_live(tmp_path: Path) -> None:
    clock = _Clock(_dt("2026-06-12T00:10:00"))
    session = _FakeSession(tmp_path=tmp_path, settings=HearthSettings())
    runtime = _FakeRuntime(
        session=session,
        last_activity_time=_dt("2026-06-12T00:00:00"),
        clock=clock,
    )
    scheduler = HearthScheduler(cast(Any, runtime), clock=clock)

    assert await scheduler.tick() is False

    session.homunculus.hearth_settings = HearthSettings(
        enabled=True,
        interval_minutes=5,
    )
    assert await scheduler.tick() is True

    session.homunculus.hearth_settings = HearthSettings(
        enabled=True,
        interval_minutes=10,
    )
    clock.value = _dt("2026-06-12T00:19:59")
    assert await scheduler.tick() is False
    clock.value = _dt("2026-06-12T00:20:00")
    assert await scheduler.tick() is True

    session.homunculus.hearth_settings = HearthSettings(
        enabled=False,
        interval_minutes=5,
    )
    clock.value = _dt("2026-06-12T01:00:00")
    assert await scheduler.tick() is False
    assert len(runtime.submitted) == 2


def test_content_function_is_pure_and_heartbeat_is_byte_identical() -> None:
    now = _dt("2026-06-12T03:12:00")
    blocks = [_semantic_block(7, facets=[_facet(7)])]

    first = compose_hearth_crackle_lines(
        now=now,
        gauge=_gauge(),
        semantic_blocks=blocks,
    )
    second = compose_hearth_crackle_lines(
        now=now,
        gauge=_gauge(),
        semantic_blocks=blocks,
    )
    later = compose_hearth_crackle_lines(
        now=_dt("2026-06-12T18:30:00"),
        gauge=_gauge(),
        semantic_blocks=[],
    )

    assert first == second
    assert first[0] == HEARTH_HEARTBEAT
    assert later[0] == HEARTH_HEARTBEAT


def test_content_amber_uses_real_compacted_block_idx_and_format() -> None:
    lines = compose_hearth_crackle_lines(
        now=_dt("2026-06-12T03:12:00"),
        gauge=_gauge(),
        semantic_blocks=[
            _semantic_block(2, mode="full", facets=[_facet(2)]),
            _semantic_block(7, facets=[_facet(7)]),
        ],
    )

    amber = lines[-1]
    match = re.fullmatch(
        r'From the shelf: ".+" — block (?P<idx>\d+), '
        r"if you want to hold it again\.",
        amber,
    )
    assert match is not None
    assert int(match.group("idx")) == 7


def test_content_omits_amber_when_no_compacted_facets_exist() -> None:
    lines = compose_hearth_crackle_lines(
        now=_dt("2026-06-12T03:12:00"),
        gauge=_gauge(),
        semantic_blocks=[
            _semantic_block(2, mode="full", facets=[_facet(2)]),
            _semantic_block(7, facets=[]),
        ],
    )

    assert not any(line.startswith("From the shelf:") for line in lines)
