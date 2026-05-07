from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.footer import FooterController, FooterControllerRoundLifecycle
from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import (
    IRFooterDrainRecord,
    IRFooterQueueRecord,
    IRRecord,
    IRSkillCatalog,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.round_lifecycle import RoundContext
from spellbook.session_lifecycle import SessionContext
from spellbook.timekeeper import (
    Timekeeper,
    TimekeeperRoundLifecycle,
    TimekeeperSessionLifecycle,
)
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def _make_controller(tmp_path: Path) -> tuple[FooterController, Recorder, Path]:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        local_timezone="America/New_York",
        idle_footer_threshold_seconds=300,
    )
    recorder = Recorder(config, transcript, "session_time", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("turn_1", [])
    inbound = InboundMessageQueue()
    return (
        FooterController(inbound_queue=inbound, recorder=recorder),
        recorder,
        transcript,
    )


def _read_records(path: Path) -> list[IRRecord]:
    adapter = TypeAdapter(IRRecord)
    records: list[IRRecord] = []
    with open(path, "r") as f:
        for line in f:
            records.append(adapter.validate_json(line))
    return records


class _Clock:
    def __init__(self, value: datetime):
        self.value = value

    def __call__(self) -> datetime:
        return self.value


@pytest.mark.asyncio
async def test_turn_start_queues_idle_footer(tmp_path: Path) -> None:
    controller, _, transcript = _make_controller(tmp_path)
    clock = _Clock(_dt("2026-03-29T12:00:00"))
    timekeeper = Timekeeper(
        SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
        controller,
        clock=clock,
    )
    lifecycle = TimekeeperSessionLifecycle(timekeeper)
    ctx = SessionContext(session_id="session_time", turn_idx=1)

    await lifecycle.on_enter_idle(ctx)
    clock.value = _dt("2026-03-29T12:08:00")
    await lifecycle.on_turn_started(ctx, "turn_1")

    pending = controller.peek_pending()
    assert len(pending) == 2
    assert pending[0].type == "time"
    assert pending[0].source == "idle"
    assert pending[0].key == "time:turn:1"
    assert pending[0].text == "Idle for 8m before this turn."
    assert pending[1].type == "time"
    assert pending[1].source == "idle"
    assert pending[1].key == "time:rollover:20260329T08"
    assert pending[1].text == "It is now 8 AM EDT, Sunday."

    records = _read_records(transcript)
    queue_records = [r for r in records if isinstance(r, IRFooterQueueRecord)]
    assert [r.footer.key for r in queue_records[-2:]] == [
        "time:turn:1",
        "time:rollover:20260329T08",
    ]


@pytest.mark.asyncio
async def test_turn_start_idle_footer_does_not_duplicate_rollover_time(
    tmp_path: Path,
) -> None:
    controller, recorder, _ = _make_controller(tmp_path)
    clock = _Clock(_dt("2026-03-29T12:00:00"))
    timekeeper = Timekeeper(
        SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
        controller,
        clock=clock,
    )
    session_lifecycle = TimekeeperSessionLifecycle(timekeeper)
    footer_lifecycle = FooterControllerRoundLifecycle(
        controller=controller, recorder=recorder
    )
    ctx = SessionContext(session_id="session_time", turn_idx=1)

    await session_lifecycle.on_enter_idle(ctx)
    clock.value = _dt("2026-03-29T13:08:00")
    await session_lifecycle.on_turn_started(ctx, "turn_1")
    round_ctx = RoundContext(
        blocks=[IRUserTextBlock(text="hello", origin="human")],
        round_number=1,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )
    await TimekeeperRoundLifecycle(timekeeper).before_round(round_ctx)
    await footer_lifecycle.before_round(round_ctx)

    footer_block = round_ctx.blocks[-1]
    assert isinstance(footer_block, IRUserTextBlock)
    assert "Idle for 1h8m before this turn." in footer_block.text
    assert footer_block.text.count("It is now 9 AM EDT, Sunday.") == 1


@pytest.mark.asyncio
async def test_round_rollover_is_drained_in_same_before_round(tmp_path: Path) -> None:
    controller, recorder, transcript = _make_controller(tmp_path)
    clock = _Clock(_dt("2026-03-29T12:59:00"))
    timekeeper = Timekeeper(
        SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
        controller,
        clock=clock,
    )
    time_lifecycle = TimekeeperRoundLifecycle(timekeeper)
    footer_lifecycle = FooterControllerRoundLifecycle(
        controller=controller, recorder=recorder
    )
    ctx = RoundContext(
        blocks=[IRUserTextBlock(text="hello", origin="human")],
        round_number=2,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )

    clock.value = _dt("2026-03-29T13:01:00")
    await time_lifecycle.before_round(ctx)
    await footer_lifecycle.before_round(ctx)

    assert controller.peek_pending() == []
    footer_block = ctx.blocks[-1]
    assert isinstance(footer_block, IRUserTextBlock)
    assert footer_block.origin == "system"
    assert "It is now 9 AM EDT, Sunday." in footer_block.text

    records = _read_records(transcript)
    drain_records = [r for r in records if isinstance(r, IRFooterDrainRecord)]
    assert drain_records[-1].footers[0].key == "time:rollover:20260329T09:2"


def test_resume_queues_footer_for_meaningful_gap(tmp_path: Path) -> None:
    controller, _, _ = _make_controller(tmp_path)
    clock = _Clock(_dt("2026-03-29T05:16:00"))
    timekeeper = Timekeeper(
        SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
        controller,
        clock=clock,
    )

    timekeeper.note_resume(
        previous_activity_time=_dt("2026-03-29T03:50:00"),
        turn_idx=4,
    )

    pending = controller.peek_pending()
    assert len(pending) == 1
    assert pending[0].key == "time:resume:4"
    assert "Resumed at 1:16 AM EDT after 1h26m idle." in pending[0].text


def test_resume_ignores_short_same_hour_gap(tmp_path: Path) -> None:
    controller, _, _ = _make_controller(tmp_path)
    clock = _Clock(_dt("2026-03-29T05:16:00"))
    timekeeper = Timekeeper(
        SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
        controller,
        clock=clock,
    )

    timekeeper.note_resume(
        previous_activity_time=_dt("2026-03-29T05:15:00"),
        turn_idx=4,
    )

    assert controller.peek_pending() == []
