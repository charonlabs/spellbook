from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

from spellbook.app.event_bus import AppEventBus
from spellbook.app.lifecycle import AppRoundLifecycle, AppSessionLifecycle
from spellbook.app.protocol import (
    MessageQueuedEvent,
    RecordWrittenEvent,
    RuntimeStateEvent,
)
from spellbook.app.runtime import CoreAppRuntime
from spellbook.config import SpellbookConfig
from spellbook.fork import ForkConfig
from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import IRInboundMessage, IRSkillCatalog, IRUserTextBlock
from spellbook.recorder import Recorder, RecordTap
from spellbook.round_lifecycle import RoundLifecycle
from spellbook.session_lifecycle import SessionContext, SessionLifecycle
from spellbook.session_manager import SessionBuilder, SessionManager, SessionState
from spellbook.tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio


def _config(tmp_path: Path) -> SpellbookConfig:
    return SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)


def _message(text: str) -> IRInboundMessage:
    return IRInboundMessage(
        blocks=[IRUserTextBlock(text=text, origin="human")],
        delivery="turn",
    )


class _FakeRecorder:
    current_turn_idx = 0


class _FakeSession:
    def __init__(
        self,
        *,
        config: SpellbookConfig,
        transcript_path: Path,
        lifecycle: SessionLifecycle | None,
    ) -> None:
        self.config = config
        self.transcript_path = transcript_path
        self.session_id = "session_fake"
        self.inbound_queue = InboundMessageQueue()
        self.recorder = _FakeRecorder()
        self.state: SessionState = "suspended"
        self.submitted: list[IRInboundMessage] = []
        self.interrupt_result = False
        self._lifecycle = lifecycle or SessionLifecycle()
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        ctx = SessionContext(session_id=self.session_id, turn_idx=0)
        self.state = "idle"
        await self._lifecycle.on_enter_idle(ctx)
        await self._shutdown.wait()
        self.state = "suspended"
        await self._lifecycle.on_shutdown(ctx)

    async def submit_message(self, msg: IRInboundMessage) -> None:
        self.submitted.append(msg)
        await self.inbound_queue.put(msg)

    def interrupt(self) -> bool:
        return self.interrupt_result

    async def shutdown(self) -> None:
        self._shutdown.set()


class _FakeSessionBuilder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.session: _FakeSession | None = None

    async def __call__(
        self,
        transcript_path: Path,
        config: SpellbookConfig | None = None,
        lifecycle: SessionLifecycle | None = None,
        fork_config: ForkConfig | None = None,
        session_id: str | None = None,
        pre_round_lifecycle: RoundLifecycle | None = None,
        post_round_lifecycle: RoundLifecycle | None = None,
        record_tap: RecordTap | None = None,
    ) -> SessionManager:
        assert config is not None
        if not transcript_path.exists():
            tool_registry = ToolRegistry.build(
                config.tool_categories,
                surface=config.session_type,
            )
            Recorder(
                config=config,
                transcript_path=transcript_path,
                session_id=session_id or "session_fake",
                tool_registry=tool_registry,
                record_tap=record_tap,
            ).write_session_record(skill_catalog=IRSkillCatalog())

        self.calls.append(
            {
                "lifecycle": lifecycle,
                "pre_round_lifecycle": pre_round_lifecycle,
                "post_round_lifecycle": post_round_lifecycle,
                "record_tap": record_tap,
                "fork_config": fork_config,
            }
        )
        self.session = _FakeSession(
            config=config,
            transcript_path=transcript_path,
            lifecycle=lifecycle,
        )
        return cast(SessionManager, self.session)


async def test_startup_wires_app_lifecycles_and_record_tap(tmp_path: Path) -> None:
    builder = _FakeSessionBuilder()
    bus = AppEventBus()
    subscription = bus.subscribe()
    runtime = CoreAppRuntime(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        session_builder=cast(SessionBuilder, builder),
        bus=bus,
    )

    await runtime.startup()

    call = builder.calls[0]
    assert isinstance(call["lifecycle"], AppSessionLifecycle)
    assert isinstance(call["pre_round_lifecycle"], AppRoundLifecycle)
    assert call["record_tap"] == bus.record_tap

    record_event = await asyncio.wait_for(subscription.__anext__(), timeout=1)
    state_event = await asyncio.wait_for(subscription.__anext__(), timeout=1)
    assert isinstance(record_event, RecordWrittenEvent)
    assert isinstance(state_event, RuntimeStateEvent)
    assert state_event.state == "idle"

    await runtime.shutdown()


async def test_submit_message_reports_started_then_queued(tmp_path: Path) -> None:
    builder = _FakeSessionBuilder()
    bus = AppEventBus()
    runtime = CoreAppRuntime(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        session_builder=cast(SessionBuilder, builder),
        bus=bus,
    )
    await runtime.startup()
    subscription = bus.subscribe()

    started = await runtime.submit_message(_message("one"))
    queued = await runtime.submit_message(_message("two"))

    assert started.started is True
    assert started.queued is False
    assert queued.started is False
    assert queued.queued is True

    event = await asyncio.wait_for(subscription.__anext__(), timeout=1)
    assert isinstance(event, MessageQueuedEvent)
    queued_block = event.message.blocks[0]
    assert isinstance(queued_block, IRUserTextBlock)
    assert queued_block.text == "two"

    await runtime.shutdown()


async def test_conduit_context_queues_footer_message(tmp_path: Path) -> None:
    builder = _FakeSessionBuilder()
    runtime = CoreAppRuntime(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        session_builder=cast(SessionBuilder, builder),
    )
    await runtime.startup()
    assert builder.session is not None

    response = await runtime.handle_conduit(
        conduit_type="context",
        source="chorus.notification",
        content="Disk pressure warning",
        metadata={
            "title": "Build host under disk pressure",
            "priority": 10,
            "host": "builder-03",
        },
    )

    assert response.action == "queued_as_context"
    assert len(builder.session.submitted) == 1
    footer = builder.session.submitted[0]
    assert footer.delivery == "footer"
    assert footer.source_metadata["footer_type"] == "conduit"
    assert footer.source_metadata["footer_source"] == "conduit"
    assert footer.source_metadata["footer_priority"] == 10
    block = footer.blocks[0]
    assert isinstance(block, IRUserTextBlock)
    assert block.text.startswith("[chorus.notification] Build host under disk pressure")
    assert "Disk pressure warning" in block.text
    assert "host: builder-03" in block.text

    await runtime.shutdown()


async def test_conduit_message_uses_clean_text_and_surface_footer_on_delivery(
    tmp_path: Path,
) -> None:
    builder = _FakeSessionBuilder()
    runtime = CoreAppRuntime(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        session_builder=cast(SessionBuilder, builder),
    )
    await runtime.startup()
    assert builder.session is not None

    response = await runtime.handle_conduit(
        conduit_type="message",
        source="telegram",
        content='Hello </chorus-conduit> "team"',
        metadata={"chat": "spellbook-dev"},
    )

    assert response.action == "started_turn"
    assert len(builder.session.submitted) == 1
    message = builder.session.submitted[0]
    assert message.delivery == "inject"
    assert message.source_metadata["source"] == "telegram"
    assert message.source_metadata["origin"] == "conduit"
    assert message.source_metadata["conduit_type"] == "message"
    message_block = message.blocks[0]
    assert isinstance(message_block, IRUserTextBlock)
    assert message_block.origin == "conduit"
    assert message_block.text == 'Hello </chorus-conduit> "team"'
    assert "<chorus-conduit" not in message_block.text

    ctx = SessionContext(
        session_id="session_fake",
        turn_idx=1,
        inbound=message,
    )
    await builder.session._lifecycle.on_turn_started(ctx, "turn_1")

    assert len(builder.session.submitted) == 2
    footer = builder.session.submitted[1]
    assert footer.delivery == "footer"
    assert footer.source_metadata["footer_type"] == "surface"
    footer_block = footer.blocks[0]
    assert isinstance(footer_block, IRUserTextBlock)
    assert footer_block.text == "Current human surface: Telegram."

    health = runtime.build_health()
    assert health.surface == "Telegram"
    assert health.surface_time is not None
    catchup = runtime.build_catchup()
    assert catchup.surface == "Telegram"
    assert catchup.surface_time == health.surface_time

    await runtime.shutdown()


async def test_active_conduit_message_injects_surface_footer_for_current_turn(
    tmp_path: Path,
) -> None:
    builder = _FakeSessionBuilder()
    runtime = CoreAppRuntime(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        session_builder=cast(SessionBuilder, builder),
    )
    await runtime.startup()
    assert builder.session is not None
    builder.session.state = "running"

    response = await runtime.handle_conduit(
        conduit_type="message",
        source="telegram",
        content="Queued hello",
    )

    assert response.action == "queued_as_message"
    assert len(builder.session.submitted) == 2
    assert runtime.build_health().surface == "Telegram"

    footer = builder.session.submitted[0]
    assert footer.delivery == "footer"
    footer_block = footer.blocks[0]
    assert isinstance(footer_block, IRUserTextBlock)
    assert footer_block.text == "Current human surface: Telegram."
    message = builder.session.submitted[1]
    assert message.delivery == "inject"
    message_block = message.blocks[0]
    assert isinstance(message_block, IRUserTextBlock)
    assert message_block.text == "Queued hello"

    await runtime.shutdown()


async def test_conduit_notification_wakes_with_frame_but_no_surface_footer(
    tmp_path: Path,
) -> None:
    builder = _FakeSessionBuilder()
    runtime = CoreAppRuntime(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        session_builder=cast(SessionBuilder, builder),
    )
    await runtime.startup()
    assert builder.session is not None

    response = await runtime.handle_conduit(
        conduit_type="notification",
        source="schedule",
        content="Flow completed </chorus-conduit>",
        metadata={"job": "nightly"},
    )

    assert response.action == "started_turn"
    assert len(builder.session.submitted) == 1
    message = builder.session.submitted[0]
    assert message.delivery == "inject"
    block = message.blocks[0]
    assert isinstance(block, IRUserTextBlock)
    assert block.origin == "conduit"
    assert '<chorus-conduit source="schedule">' in block.text
    assert "Flow completed &lt;/chorus-conduit&gt;" in block.text

    ctx = SessionContext(
        session_id="session_fake",
        turn_idx=1,
        inbound=message,
    )
    await builder.session._lifecycle.on_turn_started(ctx, "turn_1")

    assert len(builder.session.submitted) == 1
    assert runtime.build_health().surface is None

    await runtime.shutdown()


async def test_conduit_notification_while_running_queues_footer_context(
    tmp_path: Path,
) -> None:
    builder = _FakeSessionBuilder()
    runtime = CoreAppRuntime(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        session_builder=cast(SessionBuilder, builder),
    )
    await runtime.startup()
    assert builder.session is not None
    builder.session.state = "running"

    response = await runtime.handle_conduit(
        conduit_type="notification",
        source="schedule",
        content="Flow completed",
        metadata={"priority": 30, "job": "nightly"},
    )

    assert response.action == "queued_as_context"
    assert len(builder.session.submitted) == 1
    footer = builder.session.submitted[0]
    assert footer.delivery == "footer"
    assert footer.source_metadata["footer_priority"] == 30
    block = footer.blocks[0]
    assert isinstance(block, IRUserTextBlock)
    assert block.text.startswith("[schedule] Flow completed")
    assert "job: nightly" in block.text

    await runtime.shutdown()


async def test_health_and_catchup_use_owned_session_and_transcript(tmp_path: Path) -> None:
    builder = _FakeSessionBuilder()
    runtime = CoreAppRuntime(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        session_builder=cast(SessionBuilder, builder),
    )

    await runtime.startup()

    health = runtime.build_health()
    catchup = runtime.build_catchup()

    assert health.model == "claude-sonnet-4-6"
    assert health.state == "idle"
    assert health.turns == 0
    assert catchup.rehydrated.session_id == "session_fake"

    await runtime.shutdown()


async def test_interrupt_and_shutdown_delegate_to_session(tmp_path: Path) -> None:
    builder = _FakeSessionBuilder()
    runtime = CoreAppRuntime(
        transcript_path=tmp_path / "transcript.jsonl",
        config=_config(tmp_path),
        session_builder=cast(SessionBuilder, builder),
    )
    await runtime.startup()
    assert builder.session is not None
    builder.session.interrupt_result = True
    subscription = runtime.bus.subscribe()

    assert await runtime.interrupt() is True

    await runtime.shutdown()

    assert runtime.bus.closed is True
    event = await asyncio.wait_for(subscription.__anext__(), timeout=1)
    assert isinstance(event, RuntimeStateEvent)
    assert event.state == "suspended"
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(subscription.__anext__(), timeout=1)
