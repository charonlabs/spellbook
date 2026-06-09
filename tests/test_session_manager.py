from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Sequence, cast, Never

import pytest
from pydantic import BaseModel

from spellbook.backends.model_backend import RequestSurface, TokenCounter
from spellbook.cancel_token import CancelToken
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.custom import CustomSurface
from spellbook.executor import Executor
from spellbook.footer import FooterController
from spellbook.fork import BlockDetectorConfig, ForkRunner
from spellbook.generator import Generator
from spellbook.homunculus import Homunculus
from spellbook.inbound import InboundInjectionRoundLifecycle, InboundMessageQueue
from spellbook.ir_types import (
    InboundDelivery,
    IRAssistantTextBlock,
    IRBlock,
    IRBlockRecord,
    IRExecution,
    IRFooter,
    IRGeneration,
    IRInboundMessage,
    IRLoopResult,
    IRSemanticBlockRange,
    IRSkillCatalog,
    IRSkillCatalogUpdateRecord,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUsage,
    IRUserTextBlock,
    StopReason,
)
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.round_lifecycle import (
    CompositeRoundLifecycle,
    RoundContext,
    RoundLifecycle,
)
from spellbook.session_lifecycle import SessionContext, SessionLifecycle
from spellbook.session_manager import SessionManager
from spellbook.skills.manager import SkillManager
from spellbook.tools.common import (
    BlockDetectorToolMetadata,
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY
from spellbook.tools.skills import SkillInput, exec_skill


def _config(tmp_path: Path) -> SpellbookConfig:
    return SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)


def _user_msg(text: str, delivery: InboundDelivery = "turn") -> IRInboundMessage:
    return IRInboundMessage(
        blocks=[IRUserTextBlock(text=text, origin="human")],
        delivery=delivery,
    )


def _gen(
    blocks: list[IRBlock] | None = None,
    stop_reason: StopReason = "end_turn",
) -> IRGeneration:
    return IRGeneration(
        model="test-model",
        blocks=blocks or [],
        stop_reason=stop_reason,
        usage=IRUsage(),
    )


def _tool_call(call_id: str, tool: str = "Bash") -> IRToolCallBlock:
    return IRToolCallBlock(
        origin="model",
        call_id=call_id,
        tool=tool,
        input={"command": "echo hi"},
    )


def _tool_result(call_id: str, text: str = "ok") -> IRToolResultBlock:
    return IRToolResultBlock(
        call_id=call_id,
        tool="Bash",
        content=[IRToolTextBlock(text=text)],
    )


class _FakeGenerator:
    def __init__(self, responses: list[IRGeneration]):
        self._queue = list(responses)
        self.calls_seen: list[list[IRBlock]] = []

    async def run(
        self,
        blocks: list[IRBlock],
        cancel_token: CancelToken,
        lifecycle: RoundLifecycle,
    ) -> IRGeneration:
        self.calls_seen.append(list(blocks))
        if not self._queue:
            raise RuntimeError("FakeGenerator ran out of responses")
        return self._queue.pop(0)


class _FakeExecutor:
    def __init__(self, responses: list[IRExecution]):
        self._queue = list(responses)
        self.calls_received: list[list[IRToolCallBlock]] = []

    async def run(self, calls, cancel_token) -> IRExecution:
        self.calls_received.append(list(calls))
        if not self._queue:
            raise RuntimeError("FakeExecutor ran out of responses")
        return self._queue.pop(0)


class _RecordingRoundLifecycle(RoundLifecycle):
    def __init__(self, label: str):
        self.label = label
        self.events: list[tuple[str, int]] = []

    async def before_round(self, ctx: RoundContext) -> None:
        self.events.append((f"{self.label}:before_round", ctx.round_number))

    async def after_generate(self, ctx: RoundContext, generation: IRGeneration) -> None:
        self.events.append((f"{self.label}:after_generate", ctx.round_number))

    async def after_execute(self, ctx: RoundContext, execution: IRExecution) -> None:
        self.events.append((f"{self.label}:after_execute", ctx.round_number))

    async def between_rounds(self, ctx: RoundContext) -> None:
        self.events.append((f"{self.label}:between_rounds", ctx.round_number))

    async def on_loop_exit(self, ctx: RoundContext, stop_reason: StopReason) -> None:
        self.events.append((f"{self.label}:on_loop_exit", ctx.round_number))


class _RecordingSessionLifecycle(SessionLifecycle):
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None]] = []

    async def on_enter_idle(self, ctx: SessionContext) -> None:
        self.events.append(("on_enter_idle", None))

    async def on_exit_idle(self, ctx: SessionContext, reason: str) -> None:
        self.events.append(("on_exit_idle", reason))

    async def on_turn_started(self, ctx: SessionContext, turn_id: str) -> None:
        self.events.append(("on_turn_started", turn_id))

    async def on_turn_ended(
        self, ctx: SessionContext, result: IRLoopResult, turn_id: str
    ) -> None:
        self.events.append(("on_turn_ended", result.stop_reason))

    async def on_shutdown(self, ctx: SessionContext) -> None:
        self.events.append(("on_shutdown", None))


class _FakeTokenCounter(TokenCounter):
    async def count_block_content(self, block: IRBlock) -> int | None:
        return None

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return None

    async def count_frame(self) -> int | None:
        return None

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return None


class _FakeForkRunner:
    async def run_fork(self, fork_config) -> Never:
        raise AssertionError("Fork runner should not be used in this test")


class _CustomToolInput(BaseModel):
    value: str = "ok"


async def _exec_custom_tool(
    meta: ToolMetadata, input: _CustomToolInput
) -> ToolExecutionResult:
    return ToolExecutionResult(content=[IRToolTextBlock(text=input.value)])


CUSTOM_TEST_TOOL: Tool[_CustomToolInput] = Tool(
    name="CustomTool",
    input_model=_CustomToolInput,
    exec=_exec_custom_tool,
    category="filesystem",
)


def _nursery(config: SpellbookConfig | HomunculusConfig) -> Nursery:
    spellbook_config = (
        SpellbookConfig(cwd=Path.cwd(), hom_config=config)
        if isinstance(config, HomunculusConfig)
        else config
    )
    return Nursery(config=spellbook_config)


def _make_footer_controller(
    *, inbound_queue: InboundMessageQueue, recorder: Recorder
) -> FooterController:
    return FooterController(inbound_queue=inbound_queue, recorder=recorder)


def _make_homunculus(
    *,
    config: SpellbookConfig | HomunculusConfig,
    recorder: Recorder,
    inbound_queue: InboundMessageQueue | None = None,
) -> Homunculus:
    hom_config = config if isinstance(config, HomunculusConfig) else config.hom_config
    queue = inbound_queue or InboundMessageQueue()
    footer_c = _make_footer_controller(inbound_queue=queue, recorder=recorder)
    return Homunculus(
        config=hom_config,
        footer_c=footer_c,
        recorder=recorder,
        token_counter=_FakeTokenCounter(),
        nursery=_nursery(config),
        fork_runner=cast(ForkRunner, _FakeForkRunner()),
    )


class _TrackingHomunculus(Homunculus):
    def __init__(
        self,
        *,
        config: SpellbookConfig | HomunculusConfig,
        recorder: Recorder,
        inbound_queue: InboundMessageQueue | None = None,
    ):
        hom_config = (
            config if isinstance(config, HomunculusConfig) else config.hom_config
        )
        queue = inbound_queue or InboundMessageQueue()
        footer_c = _make_footer_controller(inbound_queue=queue, recorder=recorder)
        super().__init__(
            config=hom_config,
            footer_c=footer_c,
            recorder=recorder,
            token_counter=_FakeTokenCounter(),
            nursery=_nursery(config),
            fork_runner=cast(ForkRunner, _FakeForkRunner()),
        )
        self.render_calls: list[list[IRBlock]] = []
        self.rehydrate_calls: list[RehydrationResult] = []

    async def rehydrate(self, rehydrated: RehydrationResult) -> None:
        self.rehydrate_calls.append(rehydrated)
        await super().rehydrate(rehydrated)

    async def render_context(self, new_blocks: Sequence[IRBlock]):
        self.render_calls.append(list(new_blocks))
        return await super().render_context(new_blocks)


class _DummyBackend:
    def build_tool_schemas(self, registry):
        return []

    def build_token_counter(self, config: SpellbookConfig, surface_builder):
        return _FakeTokenCounter()


class _ScriptedGenerationStream:
    def __init__(self, response: IRGeneration):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def get_final_response(self) -> IRGeneration:
        return self._response

    def get_current_response(self, *, stop_reason: StopReason) -> IRGeneration:
        return self._response.model_copy(update={"stop_reason": stop_reason})


class _ScriptedBackend:
    def __init__(self, responses: list[IRGeneration]):
        self._responses = list(responses)
        self.surfaces: list[RequestSurface] = []
        self.blocks_seen: list[list[IRBlock]] = []

    def build_tool_schemas(self, registry):
        return [{"name": tool.name} for tool in registry.tools]

    def build_request_surface(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        blocks: Sequence[IRBlock],
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        effort: str,
    ) -> RequestSurface:
        self.blocks_seen.append(list(blocks))
        surface = RequestSurface(
            model=model,
            system=system,
            tools=tools,
            messages=[],
            max_output_tokens=max_output_tokens,
            output_config={"effort": effort},
        )
        self.surfaces.append(surface)
        return surface

    def stream(self, surface: RequestSurface, cancel_token: CancelToken):
        if not self._responses:
            raise RuntimeError("Scripted backend ran out of responses")
        return _ScriptedGenerationStream(self._responses.pop(0))

    def build_token_counter(self, config: SpellbookConfig, surface_builder):
        return _FakeTokenCounter()


def _make_manager(
    tmp_path: Path,
    *,
    generator: Generator | None = None,
    executor: Executor | None = None,
    round_lifecycle: RoundLifecycle | None = None,
    session_lifecycle: SessionLifecycle | None = None,
    homunculus: Homunculus | None = None,
) -> SessionManager:
    transcript = tmp_path / "transcript.jsonl"
    config = _config(tmp_path)
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    inbound_queue = InboundMessageQueue()
    return SessionManager(
        session_id="session_test",
        inbound_queue=inbound_queue,
        homunculus=homunculus
        or _make_homunculus(
            config=config,
            recorder=recorder,
            inbound_queue=inbound_queue,
        ),
        generator=generator or cast(Generator, _FakeGenerator([_gen()])),
        executor=executor or cast(Executor, _FakeExecutor([])),
        round_lifecycle=round_lifecycle or RoundLifecycle(),
        session_lifecycle=session_lifecycle or SessionLifecycle(),
        recorder=recorder,
        config=config,
        tool_registry=DEFAULT_TOOL_REGISTRY,
        transcript_path=transcript,
        nursery=Nursery(config=config),
        skill_manager=SkillManager(config=config),
    )


def _write_skill(
    root: Path,
    *,
    skill_dir: str = "compose",
    name: str = "compose",
    description: str = "Draft careful prose.",
    body: str = "Use crisp paragraphs.",
) -> Path:
    path = root / ".test-skills" / "skills" / skill_dir / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n")
    return path


class TestInboundQueueSemantics:
    @pytest.mark.asyncio
    async def test_idle_ignores_footer_until_turn_and_preserves_footer(
        self, tmp_path: Path
    ) -> None:
        manager = _make_manager(tmp_path)

        footer = _user_msg("footer note", delivery="footer")
        turn = _user_msg("hello", delivery="turn")

        await manager.submit_message(footer)
        await manager.submit_message(turn)

        await manager._idle_phase()

        assert manager.state == "idle"
        assert len(manager.inbound_queue._messages) == 2
        queued = list(manager.inbound_queue._messages)
        assert queued[0].delivery == "turn"
        assert isinstance(queued[0].blocks[0], IRUserTextBlock)
        assert queued[0].blocks[0].text == "hello"
        assert queued[1].delivery == "footer"
        assert isinstance(queued[1].blocks[0], IRUserTextBlock)
        assert queued[1].blocks[0].text == "footer note"

    @pytest.mark.asyncio
    async def test_injected_message_starts_idle_phase(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)

        injection = _user_msg("background notification", delivery="inject")
        await manager.submit_message(injection)

        await manager._idle_phase()

        queued = list(manager.inbound_queue._messages)
        assert len(queued) == 1
        assert queued[0].delivery == "inject"
        assert isinstance(queued[0].blocks[0], IRUserTextBlock)
        assert queued[0].blocks[0].text == "background notification"

    @pytest.mark.asyncio
    async def test_shutdown_wakes_idle_queue_and_run_exits(
        self, tmp_path: Path
    ) -> None:
        lifecycle = _RecordingSessionLifecycle()
        manager = _make_manager(tmp_path, session_lifecycle=lifecycle)

        task = asyncio.create_task(manager.run())
        await asyncio.sleep(0)

        await manager.shutdown()
        await asyncio.wait_for(task, timeout=0.2)

        assert manager.state == "suspended"
        assert lifecycle.events[0] == ("on_enter_idle", None)
        assert ("on_shutdown", None) in lifecycle.events
        assert ("on_exit_idle", "shutdown") in lifecycle.events


class TestInboundInjectionRoundLifecycle:
    @pytest.mark.asyncio
    async def test_injected_messages_join_current_round_and_are_recorded(
        self, tmp_path: Path
    ) -> None:
        transcript = tmp_path / "injected_round.jsonl"
        config = _config(tmp_path)
        recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn(
            "turn_1",
            [IRUserTextBlock(text="initial", origin="human")],
        )
        inbound_queue = InboundMessageQueue()
        injected = _user_msg("active-loop hello", delivery="inject")
        next_turn = _user_msg("later", delivery="turn")
        await inbound_queue.put(next_turn)
        await inbound_queue.put(injected)
        lifecycle = InboundInjectionRoundLifecycle(
            inbound_queue=inbound_queue,
            recorder=recorder,
        )
        ctx = RoundContext(
            blocks=[],
            round_number=2,
            cancel_token=CancelToken(),
            blocks_this_round=[],
        )

        await lifecycle.before_round(ctx)

        assert len(ctx.blocks) == 1
        assert ctx.blocks == ctx.blocks_this_round
        assert isinstance(ctx.blocks[0], IRUserTextBlock)
        assert ctx.blocks[0].text == "active-loop hello"
        assert [msg.delivery for msg in inbound_queue._messages] == ["turn"]

        rehydrated = Rehydrator(transcript).run()
        block_records = [
            record for record in rehydrated.records if isinstance(record, IRBlockRecord)
        ]
        assert len(block_records) == 2
        assert block_records[-1].turn == 1
        assert block_records[-1].event.turn_id == "turn_1"
        assert isinstance(block_records[-1].event, IRUserTextBlock)
        assert block_records[-1].event.text == "active-loop hello"


class TestRunningPhase:
    @pytest.mark.asyncio
    async def test_running_phase_processes_pending_turn_messages(
        self, tmp_path: Path
    ) -> None:
        gen = _FakeGenerator(
            [
                _gen(blocks=[IRAssistantTextBlock(text="done", origin="model")]),
            ]
        )
        ex = _FakeExecutor([])
        tracking_config = _config(tmp_path)
        tracking_inbound = InboundMessageQueue()
        hom = _TrackingHomunculus(
            config=tracking_config,
            recorder=Recorder(
                tracking_config,
                tmp_path / "tracking.jsonl",
                "tracking_session",
                DEFAULT_TOOL_REGISTRY,
            ),
            inbound_queue=tracking_inbound,
        )
        manager = _make_manager(
            tmp_path,
            generator=cast(Generator, gen),
            executor=cast(Executor, ex),
            homunculus=hom,
        )

        await manager.submit_message(_user_msg("hello"))

        await manager._running_phase()

        assert manager.state == "running"
        assert len(hom.render_calls) == 1
        assert isinstance(hom.render_calls[0][0], IRUserTextBlock)
        assert hom.render_calls[0][0].text == "hello"
        assert manager.cancel_token is None
        assert not manager.inbound_queue.has_pending_turn()

    @pytest.mark.asyncio
    async def test_running_phase_turn_lifecycle_sequence(self, tmp_path: Path) -> None:
        gen = _FakeGenerator([_gen(stop_reason="end_turn")])
        ex = _FakeExecutor([])
        lifecycle = _RecordingSessionLifecycle()
        manager = _make_manager(
            tmp_path,
            generator=cast(Generator, gen),
            executor=cast(Executor, ex),
            session_lifecycle=lifecycle,
        )

        await manager.submit_message(_user_msg("hello"))
        await manager._running_phase()

        names = [event[0] for event in lifecycle.events]
        assert names == ["on_turn_started", "on_turn_ended"]
        assert lifecycle.events[1] == ("on_turn_ended", "end_turn")

    @pytest.mark.asyncio
    async def test_running_phase_drains_multiple_turn_messages(
        self, tmp_path: Path
    ) -> None:
        gen = _FakeGenerator(
            [
                _gen(blocks=[IRAssistantTextBlock(text="first", origin="model")]),
                _gen(blocks=[IRAssistantTextBlock(text="second", origin="model")]),
            ]
        )
        ex = _FakeExecutor([])
        lifecycle = _RecordingSessionLifecycle()
        manager = _make_manager(
            tmp_path,
            generator=cast(Generator, gen),
            executor=cast(Executor, ex),
            session_lifecycle=lifecycle,
        )

        await manager.submit_message(_user_msg("one"))
        await manager.submit_message(_user_msg("two"))

        await manager._running_phase()

        started = [event for event in lifecycle.events if event[0] == "on_turn_started"]
        ended = [event for event in lifecycle.events if event[0] == "on_turn_ended"]
        assert len(started) == 2
        assert len(ended) == 2
        assert not manager.inbound_queue.has_pending_turn()

    @pytest.mark.asyncio
    async def test_running_phase_handles_tool_use_roundtrip(
        self, tmp_path: Path
    ) -> None:
        gen = _FakeGenerator(
            [
                _gen(blocks=[_tool_call("toolu_1")], stop_reason="tool_use"),
                _gen(blocks=[IRAssistantTextBlock(text="done", origin="model")]),
            ]
        )
        ex = _FakeExecutor([IRExecution(blocks=[_tool_result("toolu_1")])])
        manager = _make_manager(
            tmp_path,
            generator=cast(Generator, gen),
            executor=cast(Executor, ex),
        )

        await manager.submit_message(_user_msg("use a tool"))
        await manager._running_phase()

        assert len(ex.calls_received) == 1
        assert [call.call_id for call in ex.calls_received[0]] == ["toolu_1"]


class TestCompositeRoundLifecycle:
    @pytest.mark.asyncio
    async def test_composite_round_lifecycle_fans_out_in_list_order(self) -> None:
        left = _RecordingRoundLifecycle("left")
        right = _RecordingRoundLifecycle("right")
        lifecycle = CompositeRoundLifecycle([left, right])

        ctx = RoundContext(
            blocks=[],
            round_number=1,
            cancel_token=CancelToken(),
            blocks_this_round=[],
        )
        generation = _gen(stop_reason="tool_use")
        execution = IRExecution(blocks=[_tool_result("toolu_1")])

        await lifecycle.before_round(ctx)
        await lifecycle.after_generate(ctx, generation)
        await lifecycle.after_execute(ctx, execution)
        await lifecycle.between_rounds(ctx)
        await lifecycle.on_loop_exit(ctx, "end_turn")

        combined = left.events + right.events
        assert combined == [
            ("left:before_round", 1),
            ("left:after_generate", 1),
            ("left:after_execute", 1),
            ("left:between_rounds", 1),
            ("left:on_loop_exit", 1),
            ("right:before_round", 1),
            ("right:after_generate", 1),
            ("right:after_execute", 1),
            ("right:between_rounds", 1),
            ("right:on_loop_exit", 1),
        ]


class TestBuildResumeBehavior:
    @pytest.mark.asyncio
    async def test_build_new_session_creates_transcript_and_initializes_manager(
        self, tmp_path: Path
    ) -> None:
        transcript = tmp_path / "brand_new.jsonl"

        manager = await SessionManager.build(
            transcript_path=transcript,
            config=_config(tmp_path),
        )

        assert transcript.exists()
        assert manager.transcript_path == transcript
        assert manager.session_id.startswith("session_")
        assert manager.state == "suspended"

    @pytest.mark.asyncio
    async def test_build_new_main_session_discovers_records_and_wires_skill_manager(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "skills.jsonl"
        _write_skill(tmp_path)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(
            "spellbook.session_manager.build_backend",
            lambda config: _DummyBackend(),
        )
        config = _config(tmp_path).model_copy(
            update={"skill_discovery_dirs": [".test-skills"]}
        )

        manager = await SessionManager.build(
            transcript_path=transcript,
            config=config,
        )

        assert manager.skill_manager.catalog is not None
        skill = manager.skill_manager.catalog.skills["compose"]
        assert skill.scope == "project"
        assert skill.description == "Draft careful prose."
        assert manager.executor.meta.skill_manager is manager.skill_manager
        assert "<available-skills>" in manager.config.system_prompt
        assert "<name>compose</name>" in manager.config.system_prompt

        result = await exec_skill(
            manager.executor.meta,
            SkillInput(name="compose", args="forward-compatible"),
        )

        assert len(result.content) == 1
        assert isinstance(result.content[0], IRToolTextBlock)
        assert '<skill-content name="compose">' in result.content[0].text
        assert "Use crisp paragraphs." in result.content[0].text

        rehydrated = Rehydrator(transcript).run()
        assert "compose" in rehydrated.skill_catalog.skills

    @pytest.mark.asyncio
    async def test_skill_tool_missing_manager_surfaces_tool_error(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ToolError, match="Skills were not initialized"):
            await exec_skill(
                ToolMetadata(cwd=tmp_path, transcript_path=Path()),
                SkillInput(name="compose"),
            )

    @pytest.mark.asyncio
    async def test_runtime_skill_discovery_records_and_injects_footer(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "runtime_skills.jsonl"
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(
            "spellbook.session_manager.build_backend",
            lambda config: _DummyBackend(),
        )
        config = _config(tmp_path).model_copy(
            update={"skill_discovery_dirs": [".test-skills"]}
        )
        manager = await SessionManager.build(
            transcript_path=transcript,
            config=config,
        )
        assert manager.skill_manager.catalog == IRSkillCatalog()

        _write_skill(tmp_path)
        gen = _FakeGenerator([_gen(stop_reason="end_turn")])
        manager.generator = cast(Generator, gen)
        manager.executor = cast(Executor, _FakeExecutor([]))

        await manager.submit_message(_user_msg("notice the new skill"))
        await manager._running_phase()

        assert len(gen.calls_seen) == 1
        seen_blocks = gen.calls_seen[0]
        assert len(seen_blocks) == 2
        assert isinstance(seen_blocks[1], IRUserTextBlock)
        assert seen_blocks[1].origin == "system"
        assert "Skill catalog updated:" in seen_blocks[1].text
        assert "<name>compose</name>" in seen_blocks[1].text

        rehydrated = Rehydrator(transcript).run()
        assert "compose" in rehydrated.skill_catalog.skills
        updates = [
            record
            for record in rehydrated.records
            if isinstance(record, IRSkillCatalogUpdateRecord)
        ]
        assert len(updates) == 1
        assert set(updates[0].delta.added) == {"compose"}
        assert updates[0].delta.updated == {}
        assert updates[0].delta.removed == []

    @pytest.mark.asyncio
    async def test_build_without_existing_transcript_and_without_config_raises(
        self, tmp_path: Path
    ) -> None:
        transcript = tmp_path / "missing.jsonl"

        with pytest.raises(ValueError, match="Need one or the other"):
            await SessionManager.build(transcript_path=transcript)

    @pytest.mark.asyncio
    async def test_build_resume_restores_recorder_state(self, tmp_path: Path) -> None:
        transcript = tmp_path / "resume_recorder.jsonl"
        config = _config(tmp_path)
        session_id = "session_resume"
        recorder = Recorder(config, transcript, session_id, DEFAULT_TOOL_REGISTRY)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn(
            "turn_live",
            [IRUserTextBlock(text="hello", origin="human")],
        )
        recorder.write_block(IRAssistantTextBlock(text="partial", origin="model"))

        manager = await SessionManager.build(transcript_path=transcript)

        manager.recorder.write_block(
            IRAssistantTextBlock(text="after resume", origin="model")
        )

        rehydrated = manager.recorder
        assert rehydrated is manager.recorder

        from spellbook.ir_types import IRBlockRecord
        from spellbook.rehydrator import Rehydrator

        result = Rehydrator(transcript).run()
        block_records = [r for r in result.records if isinstance(r, IRBlockRecord)]
        assert [r.seq for r in block_records] == [0, 1, 2]
        assert [r.turn for r in block_records] == [1, 1, 1]
        assert block_records[-1].event.turn_id == "turn_live"

    @pytest.mark.asyncio
    async def test_build_resume_rehydrates_homunculus_state(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "resume_homunculus.jsonl"
        config = _config(tmp_path)
        recorder = Recorder(config, transcript, "session_resume", DEFAULT_TOOL_REGISTRY)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn(
            "turn_1",
            [IRUserTextBlock(text="persisted user", origin="human")],
        )
        recorder.write_block(
            IRAssistantTextBlock(text="persisted assistant", origin="model")
        )
        recorder.end_turn()

        tracking: dict[str, _TrackingHomunculus] = {}

        def _factory(
            *,
            config,
            footer_c,
            recorder: Recorder,
            token_counter,
            fork_runner,
            **kwargs,
        ):
            hom = _TrackingHomunculus(config=config, recorder=recorder)
            tracking["homunculus"] = hom
            return hom

        import spellbook.session_manager as session_manager_module

        monkeypatch.setattr(session_manager_module, "Homunculus", _factory)
        monkeypatch.setattr(
            session_manager_module,
            "build_backend",
            lambda config: _DummyBackend(),
        )

        manager = await SessionManager.build(transcript_path=transcript)

        hom = tracking["homunculus"]
        assert manager.homunculus is hom
        assert len(hom.rehydrate_calls) == 1

        rendered = await hom.render_context(
            [IRUserTextBlock(text="new message", origin="human")]
        )
        texts = [block.text for block in rendered if hasattr(block, "text")]
        assert texts == ["persisted user", "persisted assistant", "new message"]

    @pytest.mark.asyncio
    async def test_build_resume_rehydrates_pending_footers_and_injects_them_on_first_round(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "resume_pending_footers.jsonl"
        config = _config(tmp_path).model_copy(update={"skill_discovery_dirs": []})
        recorder = Recorder(config, transcript, "session_resume", DEFAULT_TOOL_REGISTRY)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("turn_1", [])
        recorder.queue_footer(
            IRFooter(
                text="restored footer",
                type="notif",
                source="conduit",
                key="restored",
                priority=15,
            )
        )
        recorder.end_turn()

        class _CapturingGenerator(_FakeGenerator):
            def __init__(self) -> None:
                super().__init__([_gen(stop_reason="end_turn")])

        gen = _CapturingGenerator()
        ex = _FakeExecutor([])

        import spellbook.session_manager as session_manager_module

        monkeypatch.setattr(
            session_manager_module,
            "build_backend",
            lambda config: _DummyBackend(),
        )

        manager = await SessionManager.build(transcript_path=transcript)
        manager.generator = cast(Generator, gen)
        manager.executor = cast(Executor, ex)

        await manager.submit_message(_user_msg("hello after resume"))
        await manager._running_phase()

        assert len(gen.calls_seen) == 1
        seen_blocks = gen.calls_seen[0]
        assert len(seen_blocks) == 2
        assert isinstance(seen_blocks[0], IRUserTextBlock)
        assert seen_blocks[0].text == "hello after resume"
        assert isinstance(seen_blocks[1], IRUserTextBlock)
        assert seen_blocks[1].origin == "system"
        assert seen_blocks[1].text == "<spellbook>\nrestored footer\n</spellbook>"

        from spellbook.rehydrator import Rehydrator

        result = Rehydrator(transcript).run()
        assert "restored" not in result.pending_footers
        assert "gas_gauge" in result.pending_footers
        assert result.pending_footers["gas_gauge"].type == "gas_gauge"

    @pytest.mark.asyncio
    async def test_build_new_then_resume_round_trips_session_id(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "round_trip.jsonl"

        monkeypatch.setattr(
            "spellbook.session_manager.build_backend",
            lambda config: _DummyBackend(),
        )

        created = await SessionManager.build(
            transcript_path=transcript,
            config=_config(tmp_path),
        )
        resumed = await SessionManager.build(transcript_path=transcript)

        assert resumed.session_id == created.session_id
        assert resumed.transcript_path == created.transcript_path

    @pytest.mark.asyncio
    async def test_build_block_detector_session_uses_detector_surface_and_metadata(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "detector.jsonl"
        _write_skill(tmp_path)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        config = _config(tmp_path).model_copy(
            update={
                "session_type": "block_detector",
                "tool_categories": {"block_detection"},
                "skill_discovery_dirs": [".test-skills"],
            }
        )
        inbound = IRUserTextBlock(
            text="<block_detector_context />",
            origin="system",
        )
        fork_config = BlockDetectorConfig(
            prev_semantic_blocks=[],
            full_context_blocks=[inbound],
            context_block_buffer=[inbound],
            context_block_start_id=0,
            semantic_block_buffer=[],
            inbound_block=inbound,
        )

        monkeypatch.setattr(
            "spellbook.session_manager.build_backend",
            lambda config: _DummyBackend(),
        )

        manager = await SessionManager.build(
            transcript_path=transcript,
            config=config,
            fork_config=fork_config,
        )

        assert manager.session_id.startswith("bd_session_")
        assert manager.tool_registry.tool_names == {
            "ProposeBlock",
            "AmendBlock",
            "CompleteBlock",
        }
        assert manager.fork_config == fork_config
        assert isinstance(manager.executor.meta, BlockDetectorToolMetadata)
        assert manager.executor.meta.full_context_blocks == [inbound]
        assert manager.executor.meta.skill_manager is None
        assert "<available-skills>" not in manager.config.system_prompt
        assert manager.skill_manager.catalog == IRSkillCatalog()

        rehydrated = Rehydrator(transcript).run()
        assert rehydrated.skill_catalog == IRSkillCatalog()

    @pytest.mark.asyncio
    async def test_build_custom_session_uses_custom_surface_and_metadata(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "custom.jsonl"
        config = _config(tmp_path).model_copy(update={"session_type": "custom"})
        custom_surface = CustomSurface(
            tools=[CUSTOM_TEST_TOOL],
            include_tool_categories={"memory"},
        )

        monkeypatch.setattr(
            "spellbook.session_manager.build_backend",
            lambda config: _DummyBackend(),
        )

        manager = await SessionManager.build(
            transcript_path=transcript,
            config=config,
            custom_surface=custom_surface,
        )

        assert manager.session_id.startswith("custom_session_")
        assert manager.tool_registry.tool_names == {
            "CustomTool",
            "Reflect",
            "ReflectToolResults",
            "Forget",
            "ForgetToolResult",
            "Configure",
            "Pin",
            "Recall",
        }
        assert isinstance(manager.executor.meta, ToolMetadata)
        assert manager.executor.meta.cwd == tmp_path
        assert manager.executor.meta.transcript_path == transcript
        assert manager.executor.meta.homunculus is manager.homunculus
        assert manager.executor.meta.skill_manager is manager.skill_manager
        assert manager.skill_manager.catalog == IRSkillCatalog()

        rehydrated = Rehydrator(
            transcript,
            custom_tools=[CUSTOM_TEST_TOOL],
        ).run()
        assert {tool.name for tool in rehydrated.tools} == {
            "CustomTool",
            "Reflect",
            "ReflectToolResults",
            "Forget",
            "ForgetToolResult",
            "Configure",
            "Pin",
            "Recall",
        }

    @pytest.mark.asyncio
    async def test_custom_session_drains_footer_messages_into_round(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "custom_footer.jsonl"
        config = _config(tmp_path).model_copy(update={"session_type": "custom"})
        backend = _ScriptedBackend([_gen(stop_reason="end_turn")])

        monkeypatch.setattr(
            "spellbook.session_manager.build_backend",
            lambda config: backend,
        )

        manager = await SessionManager.build(
            transcript_path=transcript,
            config=config,
            custom_surface=CustomSurface(tools=[]),
        )

        await manager.submit_message(
            IRInboundMessage(
                blocks=[
                    IRUserTextBlock(
                        text="Ambient custom-surface note.",
                        origin="system",
                    )
                ],
                source_metadata={
                    "footer_type": "notif",
                    "footer_source": "runtime",
                    "footer_key": "custom_note",
                },
                delivery="footer",
            )
        )
        await manager.submit_message(_user_msg("Start the scene."))
        await manager._running_phase()

        assert len(backend.blocks_seen) == 1
        assert len(backend.blocks_seen[0]) == 2
        assert isinstance(backend.blocks_seen[0][0], IRUserTextBlock)
        assert backend.blocks_seen[0][0].text == "Start the scene."
        assert isinstance(backend.blocks_seen[0][1], IRUserTextBlock)
        assert backend.blocks_seen[0][1].origin == "system"
        assert "<spellbook>" in backend.blocks_seen[0][1].text
        assert "Ambient custom-surface note." in backend.blocks_seen[0][1].text

    @pytest.mark.asyncio
    async def test_custom_session_refreshes_skills_when_skill_tool_is_enabled(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "custom_runtime_skills.jsonl"
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        backend = _ScriptedBackend([_gen(stop_reason="end_turn")])
        config = _config(tmp_path).model_copy(
            update={
                "session_type": "custom",
                "skill_discovery_dirs": [".test-skills"],
            }
        )
        custom_surface = CustomSurface(
            tools=[],
            include_tool_categories={"skills"},
        )

        monkeypatch.setattr(
            "spellbook.session_manager.build_backend",
            lambda config: backend,
        )

        manager = await SessionManager.build(
            transcript_path=transcript,
            config=config,
            custom_surface=custom_surface,
        )

        assert manager.tool_registry.tool_names == {"Skill"}
        assert manager.skill_manager.catalog == IRSkillCatalog()

        _write_skill(tmp_path)
        await manager.submit_message(_user_msg("Check for new skills."))
        await manager._running_phase()

        assert len(backend.blocks_seen) == 1
        assert len(backend.blocks_seen[0]) == 2
        assert isinstance(backend.blocks_seen[0][1], IRUserTextBlock)
        assert backend.blocks_seen[0][1].origin == "system"
        assert "Skill catalog updated:" in backend.blocks_seen[0][1].text
        assert "<name>compose</name>" in backend.blocks_seen[0][1].text

        rehydrated = Rehydrator(
            transcript,
            custom_tools=[],
        ).run()
        assert "compose" in rehydrated.skill_catalog.skills

    @pytest.mark.asyncio
    async def test_custom_session_keeps_assistant_generations_in_live_context(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "custom_memory.jsonl"
        config = _config(tmp_path).model_copy(update={"session_type": "custom"})
        backend = _ScriptedBackend(
            [
                _gen(
                    blocks=[
                        IRAssistantTextBlock(
                            text="The brass notebook rests beside the teacup.",
                            origin="model",
                        )
                    ],
                    stop_reason="end_turn",
                ),
                _gen(
                    blocks=[
                        IRAssistantTextBlock(
                            text="The notebook is still beside the teacup.",
                            origin="model",
                        )
                    ],
                    stop_reason="end_turn",
                ),
            ]
        )

        monkeypatch.setattr(
            "spellbook.session_manager.build_backend",
            lambda config: backend,
        )

        manager = await SessionManager.build(
            transcript_path=transcript,
            config=config,
            custom_surface=CustomSurface(tools=[]),
        )

        await manager.submit_message(_user_msg("Describe the room."))
        await manager._running_phase()
        await manager.submit_message(_user_msg("What objects are already here?"))
        await manager._running_phase()

        assert len(backend.blocks_seen) == 2
        second_request_texts = [
            (type(block).__name__, block.text)
            for block in backend.blocks_seen[1]
            if isinstance(block, (IRUserTextBlock, IRAssistantTextBlock))
        ]
        assert second_request_texts[:3] == [
            ("IRUserTextBlock", "Describe the room."),
            (
                "IRAssistantTextBlock",
                "The brass notebook rests beside the teacup.",
            ),
            ("IRUserTextBlock", "What objects are already here?"),
        ]

    @pytest.mark.asyncio
    async def test_block_detector_session_records_tool_rounds(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        transcript = tmp_path / "detector_recorded.jsonl"
        config = _config(tmp_path).model_copy(
            update={
                "session_type": "block_detector",
                "tool_categories": {"block_detection"},
            }
        )
        inbound = IRUserTextBlock(
            text="<block_detector_context />",
            origin="system",
        )
        buffered_block = IRSemanticBlockRange(
            title="Old Topic",
            start_block=0,
            end_block=0,
        )
        fork_config = BlockDetectorConfig(
            prev_semantic_blocks=[],
            full_context_blocks=[inbound],
            context_block_buffer=[],
            context_block_start_id=1,
            semantic_block_buffer=[buffered_block],
            inbound_block=inbound,
        )
        tool_call = IRToolCallBlock(
            call_id="toolu_complete",
            tool="CompleteBlock",
            input={"existing_title": "Old Topic"},
        )
        backend = _ScriptedBackend(
            [
                _gen(blocks=[tool_call], stop_reason="tool_use"),
                _gen(
                    blocks=[
                        IRAssistantTextBlock(
                            text="Detector pass complete.",
                            origin="model",
                        )
                    ],
                    stop_reason="end_turn",
                ),
            ]
        )

        monkeypatch.setattr(
            "spellbook.session_manager.build_backend",
            lambda config: backend,
        )

        manager = await SessionManager.build(
            transcript_path=transcript,
            config=config,
            fork_config=fork_config,
        )
        await manager.submit_message(
            IRInboundMessage(blocks=[inbound], delivery="turn")
        )
        await manager._running_phase()

        assert isinstance(manager.executor.meta, BlockDetectorToolMetadata)
        completed = manager.executor.meta.semantic_block_buffer[0]
        assert completed.title == "Old Topic"
        assert completed.completed is True
        assert completed.completed_at is not None

        result = Rehydrator(transcript).run()
        events = [
            record.event
            for record in result.records
            if isinstance(record, IRBlockRecord)
        ]

        assert any(
            isinstance(event, IRToolCallBlock) and event.tool == "CompleteBlock"
            for event in events
        )
        assert any(
            isinstance(event, IRToolResultBlock) and event.tool == "CompleteBlock"
            for event in events
        )
        assert any(
            isinstance(event, IRAssistantTextBlock)
            and event.text == "Detector pass complete."
            for event in events
        )
