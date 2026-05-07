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
)
from spellbook.ir_types import (
    IRInboundMessage,
    IRSemanticBlockRange,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.tools.common import BlockDetectorToolMetadata


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


def _prepared(result: BlockDetectorResult, fork_id: str = "detector_test"):
    async def _run() -> BlockDetectorResult:
        return result

    return PreparedFork(coro=_run(), fork_id=fork_id)


class _FakeForkSession:
    def __init__(self, final_meta: BlockDetectorToolMetadata):
        self._final_meta = final_meta
        self.submitted_messages: list[IRInboundMessage] = []
        self.run_calls = 0
        self.shutdown_calls = 0

    async def run(self) -> None:
        self.run_calls += 1

    async def submit_message(self, msg: IRInboundMessage) -> None:
        self.submitted_messages.append(msg)

    async def get_tool_meta(self) -> BlockDetectorToolMetadata:
        return self._final_meta

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeRecorder:
    def __init__(self) -> None:
        self.summons: list[tuple[str, str, str]] = []
        self.shutdowns: list[str] = []

    def summon_fork(
        self, fork_id: str, fork_type: str, child_transcript_path: str
    ) -> None:
        self.summons.append((fork_id, fork_type, child_transcript_path))

    def shutdown_fork(self, fork_id: str) -> None:
        self.shutdowns.append(fork_id)


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
