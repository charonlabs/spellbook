from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

from spellbook.app.event_bus import AppEventBus
from spellbook.app.protocol import (
    AwarenessResponse,
    ContextBlockAddedEvent,
    HealthResponse,
    StreamEvent,
    SubmitMessageResponse,
    TurnEndedEvent,
    TurnStartedEvent,
)
from spellbook.app.runtime import CoreAppRuntime
from spellbook.config import SpellbookConfig
from spellbook.custom import CustomSurface
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRGeneration,
    IRInboundMessage,
    IRLoopResult,
    IRStreamTextDeltaEvent,
    IRUserTextBlock,
)
from spellbook.sdk import SDK_REQUEST_ID_KEY, Spell


def _config(tmp_path: Path) -> SpellbookConfig:
    return SpellbookConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        system_prompt="Test entity.",
    )


class _FakeRuntime:
    def __init__(
        self,
        *,
        transcript_path: Path,
        config: SpellbookConfig | None,
        custom_surface: CustomSurface | None,
        bus: AppEventBus,
        responses: list[list[IRBlock]] | None = None,
    ) -> None:
        self.transcript_path = transcript_path
        self.config = config
        self.custom_surface = custom_surface
        self.bus = bus
        default_response: list[IRBlock] = [
            IRAssistantTextBlock(text="default response", origin="model")
        ]
        self.responses = list(responses or [default_response])
        self.started = False
        self.shutdown_called = False
        self.submitted: list[IRInboundMessage] = []
        self._session_task: asyncio.Task[None] | None = None

    async def startup(self) -> None:
        self.started = True
        self._session_task = asyncio.create_task(self._idle_forever())

    async def _idle_forever(self) -> None:
        await asyncio.Event().wait()

    async def shutdown(self) -> None:
        self.shutdown_called = True
        task = self._session_task
        self._session_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.bus.close()

    async def submit_message(self, message: IRInboundMessage) -> SubmitMessageResponse:
        self.submitted.append(message)
        turn = len(self.submitted)
        turn_id = f"turn_{turn}"
        self.bus.publish(TurnStartedEvent(turn=turn, turn_id=turn_id, message=message))

        blocks = self.responses.pop(0)
        self.bus.publish(
            StreamEvent(event=IRStreamTextDeltaEvent(text=_assistant_text(blocks)))
        )
        for block in blocks:
            self.bus.publish(ContextBlockAddedEvent(block=block))

        generation = IRGeneration(
            model=(self.config.model if self.config is not None else "resumed-model"),
            blocks=blocks,
            stop_reason="end_turn",
            usage=None,
        )
        result = IRLoopResult(
            blocks=[*message.blocks, *blocks],
            generations=[generation],
            executions=[],
            stop_reason="end_turn",
            rounds=1,
        )
        self.bus.publish(TurnEndedEvent(turn=turn, turn_id=turn_id, result=result))
        return SubmitMessageResponse(started=True, queued=False)

    async def interrupt(self) -> bool:
        return False

    def build_health(self) -> HealthResponse:
        raise NotImplementedError

    def build_awareness(self) -> AwarenessResponse:
        raise NotImplementedError


def _assistant_text(blocks: list[IRBlock]) -> str:
    return "\n\n".join(
        block.text for block in blocks if isinstance(block, IRAssistantTextBlock)
    )


class _RuntimeFactory:
    def __init__(self, responses: list[list[IRBlock]] | None = None) -> None:
        self.responses = responses
        self.runtimes: list[_FakeRuntime] = []

    def __call__(
        self,
        transcript_path: Path,
        config: SpellbookConfig | None,
        custom_surface: CustomSurface | None,
        bus: AppEventBus,
    ) -> CoreAppRuntime:
        runtime = _FakeRuntime(
            transcript_path=transcript_path,
            config=config,
            custom_surface=custom_surface,
            bus=bus,
            responses=self.responses,
        )
        self.runtimes.append(runtime)
        return cast(CoreAppRuntime, runtime)


@pytest.mark.asyncio
async def test_spell_cast_uses_session_dir_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    factory = _RuntimeFactory()
    spell = Spell(config=_config(tmp_path), _runtime_factory=factory)

    async with spell.cast() as entity:
        assert entity.transcript_path.parent == home / ".spellbook" / "sessions"
        assert entity.transcript_path.name.startswith("sdk_")
        assert factory.runtimes[0].started is True

    assert factory.runtimes[0].shutdown_called is True


@pytest.mark.asyncio
async def test_spell_cast_resumes_existing_transcript_without_config(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "existing.jsonl"
    transcript.write_text("", encoding="utf-8")
    factory = _RuntimeFactory()
    spell = Spell(
        config=_config(tmp_path),
        transcript_path=transcript,
        _runtime_factory=factory,
    )

    async with spell.cast():
        pass

    assert factory.runtimes[0].transcript_path == transcript.resolve()
    assert factory.runtimes[0].config is None


@pytest.mark.asyncio
async def test_entity_send_returns_matching_turn_result(tmp_path: Path) -> None:
    response: list[IRBlock] = [
        IRAssistantTextBlock(text="The lantern wakes.", origin="model")
    ]
    factory = _RuntimeFactory(responses=[response])
    spell = Spell(config=_config(tmp_path), _runtime_factory=factory)

    async with spell.cast() as entity:
        result = await entity.send("Begin.", metadata={"chapter": 1})

    runtime = factory.runtimes[0]
    assert len(runtime.submitted) == 1
    submitted = runtime.submitted[0]
    assert isinstance(submitted.blocks[0], IRUserTextBlock)
    assert submitted.blocks[0].text == "Begin."
    assert submitted.source_metadata["source"] == "sdk"
    assert submitted.source_metadata["chapter"] == 1
    assert isinstance(submitted.source_metadata[SDK_REQUEST_ID_KEY], str)

    assert result.text == "The lantern wakes."
    assert result.blocks == response
    assert result.turn_id == "turn_1"
    assert result.stop_reason == "end_turn"
    assert result.loop_result.rounds == 1
    assert len(result.stream_events) == 1
    assert isinstance(result.stream_events[0], IRStreamTextDeltaEvent)


@pytest.mark.asyncio
async def test_entity_stream_yields_events_and_returns_result(tmp_path: Path) -> None:
    response: list[IRBlock] = [
        IRAssistantTextBlock(text="A streamed answer.", origin="model")
    ]
    factory = _RuntimeFactory(responses=[response])
    spell = Spell(config=_config(tmp_path), _runtime_factory=factory)

    async with spell.cast() as entity:
        stream = entity.stream("Stream this.")
        events = [event async for event in stream]
        result = await stream.result()

    assert len(events) == 1
    assert isinstance(events[0], IRStreamTextDeltaEvent)
    assert events[0].text == "A streamed answer."
    assert result.text == "A streamed answer."
    assert result.stream_events == events


@pytest.mark.asyncio
async def test_spell_once_casts_sends_and_shuts_down(tmp_path: Path) -> None:
    factory = _RuntimeFactory(
        responses=[[IRAssistantTextBlock(text="One turn only.", origin="model")]]
    )
    spell = Spell(config=_config(tmp_path), _runtime_factory=factory)

    result = await spell.once("Say one thing.")

    assert result.text == "One turn only."
    assert len(factory.runtimes) == 1
    assert factory.runtimes[0].shutdown_called is True
