"""Focused core unit tests for fork runner dispatch and block detector fork execution."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from spellbook.config import SpellbookConfig
from spellbook.fork import (
    BlockDetectorConfig,
    BlockDetectorResult,
    ForkRunner,
    PreparedFork,
    QuantumForkConfig,
    QuantumForkResult,
)
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRGeneration,
    IRInboundMessage,
    IRLoopResult,
    IRSemanticBlockRange,
    IRSkillCatalog,
    IRUsage,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.session_lifecycle import SessionContext
from spellbook.tools.common import BlockDetectorToolMetadata, QuantumForkToolMetadata
from spellbook.tools.registry import ToolRegistry


def _parent_config(tmp_path: Path) -> SpellbookConfig:
    return SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)


def _detector_config() -> BlockDetectorConfig:
    return BlockDetectorConfig(
        prev_semantic_blocks=[
            IRSemanticBlockRange(
                title="Completed before fork", start_block=0, end_block=2
            )
        ],
        full_context_blocks=[],
        context_block_buffer=[],
        context_block_start_id=3,
        semantic_block_buffer=[
            IRSemanticBlockRange(
                title="Buffered before fork", start_block=3, end_block=4
            )
        ],
        inbound_block=IRUserTextBlock(
            text="<block_detector_context />",
            origin="system",
        ),
    )


def _quantum_config(label: str | None = "Chapter One") -> QuantumForkConfig:
    return QuantumForkConfig(
        instruction=IRUserTextBlock(
            text="Write the fork result.",
            origin="system",
        ),
        fork_label=label,
    )


def _write_parent_transcript(tmp_path: Path) -> Path:
    session_dir = tmp_path / "session"
    transcript = session_dir / "transcript.jsonl"
    config = _parent_config(tmp_path).model_copy(
        update={"system_prompt": "parent system"}
    )
    recorder = Recorder(
        config=config,
        transcript_path=transcript,
        session_id="parent_session",
        tool_registry=ToolRegistry.build(
            config.tool_categories, body_url=config.body_url
        ),
    )
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn(
        "turn_parent",
        [IRUserTextBlock(text="parent input", origin="human")],
    )
    recorder.write_block(IRAssistantTextBlock(text="parent answer", origin="model"))
    recorder.end_turn("end_turn")
    (session_dir / "blobs").mkdir()
    (session_dir / "tool_outputs").mkdir()
    return transcript


def _unfinished_parent_transcript(tmp_path: Path) -> Path:
    transcript = tmp_path / "session" / "transcript.jsonl"
    config = _parent_config(tmp_path)
    recorder = Recorder(
        config=config,
        transcript_path=transcript,
        session_id="parent_session",
        tool_registry=ToolRegistry.build(
            config.tool_categories, body_url=config.body_url
        ),
    )
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn(
        "turn_open",
        [IRUserTextBlock(text="still running", origin="human")],
    )
    return transcript


def _loop_result(final_text: str, *, rounds: int = 1) -> IRLoopResult:
    final_block = IRAssistantTextBlock(text=final_text, origin="model")
    generation = IRGeneration(
        model="test-model",
        blocks=[final_block],
        stop_reason="end_turn",
        usage=IRUsage(),
    )
    return IRLoopResult(
        blocks=[final_block],
        generations=[generation],
        executions=[],
        stop_reason="end_turn",
        rounds=rounds,
    )


def _prepared(result: Any, fork_id: str = "detector_test"):
    async def _run():
        return result

    return PreparedFork(coro=_run(), fork_id=fork_id)


class _FakeForkSession:
    def __init__(self, final_meta: object):
        self._final_meta = final_meta
        self.submitted_messages: list[IRInboundMessage] = []
        self.run_calls = 0
        self.shutdown_calls = 0

    async def run(self) -> None:
        self.run_calls += 1

    async def submit_message(self, msg: IRInboundMessage) -> None:
        self.submitted_messages.append(msg)

    async def get_tool_meta(self) -> object:
        return self._final_meta

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeRecorder:
    def __init__(self) -> None:
        self.summons: list[tuple[str, str, str]] = []
        self.shutdowns: list[str] = []
        self.shutdown_notes: list[tuple[str, str | None]] = []

    def summon_fork(
        self, fork_id: str, fork_type: str, child_transcript_path: str
    ) -> None:
        self.summons.append((fork_id, fork_type, child_transcript_path))

    def shutdown_fork(self, fork_id: str, error_note: str | None = None) -> None:
        self.shutdowns.append(fork_id)
        self.shutdown_notes.append((fork_id, error_note))


class _FakeDebugEmitter:
    def __init__(self) -> None:
        self.alerts: list[dict[str, object]] = []

    def alert(self, **kwargs: object) -> None:
        self.alerts.append(kwargs)


class TestForkDispatch:
    @pytest.mark.asyncio
    async def test_run_fork_dispatches_block_detector_config(
        self, tmp_path: Path
    ) -> None:
        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, lambda **kwargs: None),
        )
        fork_config = _detector_config()
        expected = BlockDetectorResult(completed=[], still_buffered=[])
        prepared = _prepared(expected)

        async def _fake_run_block_detector(
            config: BlockDetectorConfig,
        ) -> PreparedFork:
            assert config is fork_config
            return prepared

        cast(Any, runner)._run_block_detector = _fake_run_block_detector

        result = await runner.run_fork(fork_config)

        assert result is prepared
        prepared.coro.close()

    @pytest.mark.asyncio
    async def test_run_fork_dispatches_quantum_config(self, tmp_path: Path) -> None:
        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, lambda **kwargs: None),
        )
        fork_config = _quantum_config()
        expected = QuantumForkResult(
            final_text="done",
            submitted=None,
            fork_transcript_path=str(tmp_path / "fork.jsonl"),
            rounds=1,
            stop_reason="end_turn",
        )
        prepared = _prepared(expected, fork_id="quantum_test")

        async def _fake_run_quantum(config: QuantumForkConfig) -> PreparedFork:
            assert config is fork_config
            return prepared

        cast(Any, runner)._run_quantum = _fake_run_quantum

        result = await runner.run_fork(fork_config)

        assert result is prepared
        prepared.coro.close()

    @pytest.mark.asyncio
    async def test_block_detector_build_failure_alerts_and_shutdowns_fork(
        self, tmp_path: Path
    ) -> None:
        recorder = _FakeRecorder()
        debug = _FakeDebugEmitter()

        async def _build_session(**kwargs):
            raise RuntimeError("build boom")

        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, recorder),
            session_builder=cast(Any, _build_session),
            debug_emitter=cast(Any, debug),
        )

        with pytest.raises(RuntimeError, match="build boom"):
            await runner._run_block_detector(_detector_config())  # noqa: SLF001

        fork_id = recorder.summons[0][0]
        assert recorder.shutdowns == [fork_id]
        assert len(debug.alerts) == 1
        assert debug.alerts[0]["subsystem"] == "fork"
        assert debug.alerts[0]["event"] == "build_failed"
        metadata = cast(dict[str, object], debug.alerts[0]["metadata"])
        assert metadata["fork_id"] == fork_id
        assert metadata["fork_type"] == "block_detector"

    def test_orientation_loader_reads_markdown_fork_orientation(
        self, tmp_path: Path
    ) -> None:
        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, lambda **kwargs: None),
        )

        orientation = runner._get_orientation(_detector_config())  # noqa: SLF001

        assert "block detector" in orientation.lower()


class TestQuantumForkRun:
    @pytest.mark.asyncio
    async def test_run_quantum_snapshots_parent_transcript_directory(
        self, tmp_path: Path
    ) -> None:
        parent_path = _write_parent_transcript(tmp_path)
        fork_config = _quantum_config(label="Dream Chapter")
        built: dict[str, Any] = {}

        async def _build_session(**kwargs):
            built.update(kwargs)
            return _FakeForkSession(
                QuantumForkToolMetadata(
                    cwd=tmp_path,
                    transcript_path=kwargs["transcript_path"],
                )
            )

        recorder = _FakeRecorder()
        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=parent_path,
            recorder=cast(Recorder, recorder),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_quantum(fork_config)  # noqa: SLF001

        child_path = built["transcript_path"]
        assert child_path == Path(recorder.summons[0][2])
        assert child_path.parent.parent == parent_path.parent / "forks"
        assert child_path.name == "transcript.jsonl"
        assert child_path.parent.name.startswith("quantum_Dream_Chapter_")
        assert (child_path.parent / "blobs").is_symlink()
        assert (child_path.parent / "tool_outputs").is_symlink()

        rehydrated = Rehydrator(child_path).run()
        assert rehydrated.session_id == prepared.fork_id
        assert rehydrated.config.session_type == "quantum"
        assert rehydrated.config.profile.name == "quantum"
        assert rehydrated.config.system_prompt == "parent system"
        assert [type(block).__name__ for block in rehydrated.blocks] == [
            "IRUserTextBlock",
            "IRAssistantTextBlock",
        ]
        assert {tool.name for tool in rehydrated.tools} == {
            "Read",
            "Reflect",
            "ReflectToolResults",
            "Recall",
            "SubmitResult",
        }
        assert built["config"].session_type == "quantum"
        assert built["fork_config"] == fork_config
        assert built["session_id"] == prepared.fork_id
        assert recorder.shutdowns == []
        prepared.coro.close()

    @pytest.mark.asyncio
    async def test_run_quantum_returns_submitted_payload_and_final_text(
        self, tmp_path: Path
    ) -> None:
        parent_path = _write_parent_transcript(tmp_path)
        fork_config = _quantum_config()
        final_meta = QuantumForkToolMetadata(
            cwd=tmp_path,
            transcript_path=Path(),
            submitted={"chapter": 1, "status": "ok"},
            submit_called=True,
        )
        fake_session = _FakeForkSession(final_meta)

        async def _build_session(**kwargs):
            lifecycle = kwargs["lifecycle"]

            async def _submit_and_release(msg: IRInboundMessage) -> None:
                fake_session.submitted_messages.append(msg)
                await lifecycle.on_turn_ended(
                    SessionContext(session_id=kwargs["session_id"], turn_idx=1),
                    _loop_result("final fork text", rounds=2),
                    "turn_quantum",
                )

            cast(Any, fake_session).submit_message = _submit_and_release
            return fake_session

        recorder = _FakeRecorder()
        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=parent_path,
            recorder=cast(Recorder, recorder),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_quantum(fork_config)  # noqa: SLF001
        result = await prepared.coro

        assert isinstance(result, QuantumForkResult)
        assert result.final_text == "final fork text"
        assert result.submitted == {"chapter": 1, "status": "ok"}
        assert result.rounds == 2
        assert result.stop_reason == "end_turn"
        assert Path(result.fork_transcript_path).exists()
        assert fake_session.submitted_messages == [
            IRInboundMessage(
                blocks=[fork_config.instruction],
                delivery="turn",
                source_metadata={
                    "source": "quantum_fork",
                    "fork_id": prepared.fork_id,
                    "fork_label": fork_config.fork_label,
                },
            )
        ]
        assert recorder.shutdowns == []

        runner.integrate_result(prepared.fork_id)

        assert recorder.shutdown_notes == [(prepared.fork_id, None)]

    @pytest.mark.asyncio
    async def test_run_quantum_without_submit_uses_final_text(
        self, tmp_path: Path
    ) -> None:
        parent_path = _write_parent_transcript(tmp_path)
        fork_config = _quantum_config(label=None)
        fake_session = _FakeForkSession(
            QuantumForkToolMetadata(cwd=tmp_path, transcript_path=Path())
        )

        async def _build_session(**kwargs):
            lifecycle = kwargs["lifecycle"]

            async def _submit_and_release(msg: IRInboundMessage) -> None:
                fake_session.submitted_messages.append(msg)
                await lifecycle.on_turn_ended(
                    SessionContext(session_id=kwargs["session_id"], turn_idx=1),
                    _loop_result("plain result"),
                    "turn_quantum",
                )

            cast(Any, fake_session).submit_message = _submit_and_release
            return fake_session

        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=parent_path,
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_quantum(fork_config)  # noqa: SLF001
        result = await prepared.coro

        assert isinstance(result, QuantumForkResult)
        assert result.final_text == "plain result"
        assert result.submitted is None

    @pytest.mark.asyncio
    async def test_run_quantum_failure_keeps_transcript_and_writes_error_shutdown(
        self, tmp_path: Path
    ) -> None:
        parent_path = _write_parent_transcript(tmp_path)
        fork_config = _quantum_config()

        class _ErroringSession:
            async def run(self) -> None:
                raise RuntimeError("child boom")

            async def submit_message(self, msg: IRInboundMessage) -> None:
                return None

            async def get_tool_meta(self) -> QuantumForkToolMetadata:
                return QuantumForkToolMetadata(cwd=tmp_path, transcript_path=Path())

            async def shutdown(self) -> None:
                return None

        async def _build_session(**kwargs):
            return _ErroringSession()

        recorder = _FakeRecorder()
        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=parent_path,
            recorder=cast(Recorder, recorder),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_quantum(fork_config)  # noqa: SLF001

        with pytest.raises(RuntimeError, match="child boom"):
            await asyncio.wait_for(prepared.coro, timeout=1)

        fork_path = Path(recorder.summons[0][2])
        assert fork_path.exists()
        assert recorder.shutdowns == [prepared.fork_id]
        assert recorder.shutdown_notes[0][0] == prepared.fork_id
        assert "RuntimeError: child boom" in (recorder.shutdown_notes[0][1] or "")

    @pytest.mark.asyncio
    async def test_run_quantum_quiescence_failure_touches_nothing(
        self, tmp_path: Path
    ) -> None:
        parent_path = _unfinished_parent_transcript(tmp_path)
        recorder = _FakeRecorder()
        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=parent_path,
            recorder=cast(Recorder, recorder),
            session_builder=cast(Any, lambda **kwargs: None),
        )

        with pytest.raises(RuntimeError, match="in-progress turn"):
            await runner._run_quantum(_quantum_config())  # noqa: SLF001

        assert recorder.summons == []
        assert recorder.shutdowns == []
        assert not (parent_path.parent / "forks").exists()

    @pytest.mark.asyncio
    async def test_run_quantum_can_omit_submit_result_tool(
        self, tmp_path: Path
    ) -> None:
        parent_path = _write_parent_transcript(tmp_path)
        fork_config = _quantum_config().model_copy(update={"submit_tool": False})
        built: dict[str, Any] = {}

        async def _build_session(**kwargs):
            built.update(kwargs)
            return _FakeForkSession(
                QuantumForkToolMetadata(
                    cwd=tmp_path,
                    transcript_path=kwargs["transcript_path"],
                )
            )

        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=parent_path,
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_quantum(fork_config)  # noqa: SLF001
        rehydrated = Rehydrator(built["transcript_path"]).run()

        assert {tool.name for tool in rehydrated.tools} == {
            "Read",
            "Reflect",
            "ReflectToolResults",
            "Recall",
        }
        prepared.coro.close()


class TestBlockDetectorForkRun:
    @pytest.mark.asyncio
    async def test_run_block_detector_derives_child_config_and_submits_inbound_block(
        self, tmp_path: Path
    ) -> None:
        parent_config = SpellbookConfig(
            provider="openai",
            model="gpt-5.5",
            cwd=tmp_path,
        )
        parent_path = tmp_path / "parent_transcript.jsonl"
        fork_config = _detector_config()

        final_meta = BlockDetectorToolMetadata(
            cwd=tmp_path,
            transcript_path=Path(),
            prev_semantic_blocks=fork_config.prev_semantic_blocks,
            full_context_blocks=fork_config.full_context_blocks,
            context_block_buffer=fork_config.context_block_buffer,
            context_block_start_id=fork_config.context_block_start_id,
            semantic_block_buffer=[
                IRSemanticBlockRange(
                    title="Completed in child",
                    start_block=3,
                    end_block=5,
                    completed=True,
                ),
                IRSemanticBlockRange(
                    title="Still buffered in child",
                    start_block=6,
                    end_block=7,
                ),
            ],
            new_semantic_blocks=[],
            touched_block_titles=set(),
        )
        fake_session = _FakeForkSession(final_meta)
        built: dict[str, Any] = {}

        async def _build_session(**kwargs):
            built.update(kwargs)
            lifecycle = kwargs["lifecycle"]

            async def _submit_and_release(msg: IRInboundMessage) -> None:
                fake_session.submitted_messages.append(msg)
                lifecycle.turn_end_event.set()

            cast(Any, fake_session).submit_message = _submit_and_release
            return fake_session

        recorder = _FakeRecorder()
        runner = ForkRunner(
            parent_config=parent_config,
            parent_transcript_path=parent_path,
            recorder=cast(Recorder, recorder),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_block_detector(fork_config)
        result = await prepared.coro
        assert isinstance(result, BlockDetectorResult)

        assert built["config"].session_type == "block_detector"
        assert built["config"].tool_categories == {"block_detection"}
        assert built["config"].provider == parent_config.provider
        assert built["config"].model == parent_config.model
        assert built["fork_config"] == fork_config
        assert built["transcript_path"].parent == parent_path.parent / "forks"
        assert built["transcript_path"].name.startswith("detector_")
        assert built["transcript_path"].suffix == ".jsonl"
        assert built["session_id"] == built["transcript_path"].stem
        assert recorder.summons == [
            (
                built["session_id"],
                "block_detector",
                str(built["transcript_path"]),
            )
        ]
        assert recorder.shutdowns == []

        assert fake_session.run_calls == 1
        assert fake_session.submitted_messages == [
            IRInboundMessage(blocks=[fork_config.inbound_block], delivery="turn")
        ]
        assert fake_session.shutdown_calls == 1

        assert [b.title for b in result.completed] == ["Completed in child"]
        assert [b.title for b in result.still_buffered] == ["Still buffered in child"]
        runner.integrate_result(prepared.fork_id)
        assert recorder.shutdowns == [built["session_id"]]

    @pytest.mark.asyncio
    async def test_run_block_detector_honors_explicit_detector_model(
        self, tmp_path: Path
    ) -> None:
        parent_config = _parent_config(tmp_path)
        fork_config = _detector_config().model_copy(
            update={"detector_model": "claude-opus-4-7"}
        )
        final_meta = BlockDetectorToolMetadata(
            cwd=tmp_path,
            transcript_path=Path(),
            prev_semantic_blocks=[],
            full_context_blocks=[],
            context_block_buffer=[],
            context_block_start_id=0,
            semantic_block_buffer=[],
            new_semantic_blocks=[],
            touched_block_titles=set(),
        )
        fake_session = _FakeForkSession(final_meta)
        built: dict[str, Any] = {}

        async def _build_session(**kwargs):
            built.update(kwargs)
            kwargs["lifecycle"].turn_end_event.set()
            return fake_session

        runner = ForkRunner(
            parent_config=parent_config,
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_block_detector(fork_config)
        await prepared.coro

        assert built["config"].provider == "anthropic"
        assert built["config"].model == "claude-opus-4-7"

    @pytest.mark.asyncio
    async def test_run_block_detector_waits_for_turn_end_before_reading_tool_meta(
        self, tmp_path: Path
    ) -> None:
        fork_config = _detector_config()
        events: list[str] = []

        class _LifecycleAwareSession:
            async def run(self) -> None:
                events.append("run")

            async def submit_message(self, msg: IRInboundMessage) -> None:
                events.append("submit")

            async def get_tool_meta(self) -> BlockDetectorToolMetadata:
                events.append("get_tool_meta")
                return BlockDetectorToolMetadata(
                    cwd=tmp_path,
                    transcript_path=Path(),
                    prev_semantic_blocks=[],
                    full_context_blocks=[],
                    context_block_buffer=[],
                    context_block_start_id=0,
                    semantic_block_buffer=[],
                    new_semantic_blocks=[],
                    touched_block_titles=set(),
                )

            async def shutdown(self) -> None:
                events.append("shutdown")

        async def _build_session(**kwargs):
            lifecycle = kwargs["lifecycle"]

            async def _release_lifecycle() -> None:
                events.append("set_turn_end")
                lifecycle.turn_end_event.set()

            async def _submit_then_release(msg: IRInboundMessage) -> None:
                events.append("submit")
                await _release_lifecycle()

            session = _LifecycleAwareSession()
            cast(Any, session).submit_message = _submit_then_release
            return session

        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_block_detector(fork_config)
        await prepared.coro

        assert events == ["submit", "set_turn_end", "get_tool_meta", "shutdown", "run"]

    @pytest.mark.asyncio
    async def test_run_block_detector_requires_block_detector_tool_metadata(
        self, tmp_path: Path
    ) -> None:
        fork_config = _detector_config()

        class _WrongMetaSession:
            async def run(self) -> None:
                return None

            async def submit_message(self, msg: IRInboundMessage) -> None:
                return None

            async def get_tool_meta(self) -> object:
                return object()

            async def shutdown(self) -> None:
                return None

        async def _build_session(**kwargs):
            kwargs["lifecycle"].turn_end_event.set()
            return _WrongMetaSession()

        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, _build_session),
        )

        with pytest.raises(AssertionError):
            prepared = await runner._run_block_detector(fork_config)
            await prepared.coro

    @pytest.mark.asyncio
    async def test_run_block_detector_shuts_down_child_session_after_collecting_result(
        self, tmp_path: Path
    ) -> None:
        fork_config = _detector_config()
        events: list[str] = []

        class _ShutdownTrackingSession:
            async def run(self) -> None:
                events.append("run")

            async def submit_message(self, msg: IRInboundMessage) -> None:
                events.append("submit")

            async def get_tool_meta(self) -> BlockDetectorToolMetadata:
                events.append("get_tool_meta")
                return BlockDetectorToolMetadata(
                    cwd=tmp_path,
                    transcript_path=Path(),
                    prev_semantic_blocks=[],
                    full_context_blocks=[],
                    context_block_buffer=[],
                    context_block_start_id=0,
                    semantic_block_buffer=[
                        IRSemanticBlockRange(
                            title="Completed block",
                            start_block=0,
                            end_block=1,
                            completed=True,
                        )
                    ],
                    new_semantic_blocks=[],
                    touched_block_titles=set(),
                )

            async def shutdown(self) -> None:
                events.append("shutdown")

        async def _build_session(**kwargs):
            kwargs["lifecycle"].turn_end_event.set()
            return _ShutdownTrackingSession()

        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_block_detector(fork_config)
        result = await prepared.coro
        assert isinstance(result, BlockDetectorResult)

        assert [b.title for b in result.completed] == ["Completed block"]
        assert events == ["submit", "get_tool_meta", "shutdown", "run"]

    @pytest.mark.asyncio
    async def test_prepared_fork_fails_if_child_errors_before_turn_end(
        self, tmp_path: Path
    ) -> None:
        fork_config = _detector_config()
        events: list[str] = []

        class _ErroringSession:
            async def run(self) -> None:
                events.append("run")
                raise RuntimeError("child boom")

            async def submit_message(self, msg: IRInboundMessage) -> None:
                events.append("submit")

            async def get_tool_meta(self) -> BlockDetectorToolMetadata:
                events.append("get_tool_meta")
                return BlockDetectorToolMetadata(
                    cwd=tmp_path,
                    transcript_path=Path(),
                    prev_semantic_blocks=[],
                    full_context_blocks=[],
                    context_block_buffer=[],
                    context_block_start_id=0,
                    semantic_block_buffer=[],
                    new_semantic_blocks=[],
                    touched_block_titles=set(),
                )

            async def shutdown(self) -> None:
                events.append("shutdown")

        async def _build_session(**kwargs):
            return _ErroringSession()

        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_block_detector(fork_config)

        with pytest.raises(RuntimeError, match="child boom"):
            await asyncio.wait_for(prepared.coro, timeout=1)

        assert events == ["submit", "run", "shutdown"]

    @pytest.mark.asyncio
    async def test_prepared_fork_shutdowns_child_when_cancelled(
        self, tmp_path: Path
    ) -> None:
        fork_config = _detector_config()
        events: list[str] = []

        never = asyncio.Event()

        class _CancellableSession:
            async def run(self) -> None:
                events.append("run")
                await never.wait()

            async def submit_message(self, msg: IRInboundMessage) -> None:
                events.append("submit")

            async def get_tool_meta(self) -> BlockDetectorToolMetadata:
                events.append("get_tool_meta")
                return BlockDetectorToolMetadata(
                    cwd=tmp_path,
                    transcript_path=Path(),
                    prev_semantic_blocks=[],
                    full_context_blocks=[],
                    context_block_buffer=[],
                    context_block_start_id=0,
                    semantic_block_buffer=[],
                    new_semantic_blocks=[],
                    touched_block_titles=set(),
                )

            async def shutdown(self) -> None:
                events.append("shutdown")

        async def _build_session(**kwargs):
            return _CancellableSession()

        runner = ForkRunner(
            parent_config=_parent_config(tmp_path),
            parent_transcript_path=tmp_path / "parent.jsonl",
            recorder=cast(Recorder, _FakeRecorder()),
            session_builder=cast(Any, _build_session),
        )

        prepared = await runner._run_block_detector(fork_config)
        task = asyncio.create_task(prepared.coro)
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert "shutdown" in events
        assert "get_tool_meta" not in events
