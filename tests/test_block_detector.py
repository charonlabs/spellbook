"""Focused core unit tests for block detector behavior and rendering."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.fork import (
    BlockDetectorConfig,
    BlockDetectorResult,
    ForkRunner,
    PreparedFork,
)
from spellbook.homunculus.block_detector import BlockDetector
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRImageBase64Source,
    IRImageBlock,
    IRSemanticBlockRange,
    IRSkillCatalog,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _prepared(result: BlockDetectorResult, fork_id: str = "detector_test"):
    async def _run() -> BlockDetectorResult:
        return result

    return PreparedFork(coro=_run(), fork_id=fork_id)


def _make_detector(tmp_path: Path, *, detect_interval: int = 100) -> BlockDetector:
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    recorder = Recorder(
        config,
        tmp_path / "transcript.jsonl",
        "session_test",
        DEFAULT_TOOL_REGISTRY,
    )
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("t1", [])
    fake_runner = SimpleNamespace(run_fork=None, integrate_result=lambda fork_id: None)
    return BlockDetector(
        config=HomunculusConfig(detect_interval=detect_interval),
        fork_runner=cast(ForkRunner, fake_runner),
        recorder=recorder,
    )


def _user(text: str) -> IRUserTextBlock:
    return IRUserTextBlock(text=text, origin="human")


def _assistant(text: str) -> IRAssistantTextBlock:
    return IRAssistantTextBlock(text=text, origin="model")


class TestBuildContextBuffer:
    def test_uses_accumulated_start_id_when_no_completed_or_buffered_blocks(
        self, tmp_path: Path
    ) -> None:
        detector = _make_detector(tmp_path)
        detector._accumulated = [_user("a"), _assistant("b"), _user("c")]
        detector._accumulated_start_block_id = 25

        detector.build_context_buffer()

        assert detector._start_block_id == 25
        assert detector._context_buffer == detector._accumulated

    def test_starts_after_last_completed_block_when_no_semantic_buffer(
        self, tmp_path: Path
    ) -> None:
        detector = _make_detector(tmp_path)
        detector._accumulated = [
            _user("a"),
            _assistant("b"),
            _user("c"),
            _assistant("d"),
        ]
        detector._accumulated_start_block_id = 10
        detector.completed_blocks = [
            IRSemanticBlockRange(title="done", start_block=10, end_block=11)
        ]

        detector.build_context_buffer()

        assert detector._start_block_id == 12
        assert detector._context_buffer == detector._accumulated[2:]

    def test_starts_after_last_buffered_semantic_block(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        detector._accumulated = [
            _user("a"),
            _assistant("b"),
            _user("c"),
            _assistant("d"),
            _user("e"),
        ]
        detector._accumulated_start_block_id = 50
        detector.completed_blocks = [
            IRSemanticBlockRange(title="done", start_block=50, end_block=51)
        ]
        detector._semantic_buffer = [
            IRSemanticBlockRange(title="buffered", start_block=52, end_block=53)
        ]

        detector.build_context_buffer()

        assert detector._start_block_id == 54
        assert detector._context_buffer == [detector._accumulated[4]]

    def test_empty_when_start_is_past_accumulated_tail(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        detector._accumulated = [_user("a"), _assistant("b")]
        detector._accumulated_start_block_id = 100
        detector._semantic_buffer = [
            IRSemanticBlockRange(title="buffered", start_block=100, end_block=101)
        ]

        detector.build_context_buffer()

        assert detector._start_block_id == 102
        assert detector._context_buffer == []

    def test_handles_no_accumulated_start_id(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        detector._accumulated = [_user("a")]

        detector.build_context_buffer()

        assert detector._start_block_id == 0
        assert detector._context_buffer == []


class TestMaybeDetect:
    @pytest.mark.asyncio
    async def test_under_threshold_only_accumulates_and_updates_counter(
        self, tmp_path: Path
    ) -> None:
        detector = _make_detector(tmp_path)
        blocks = [_user(f"msg {i}") for i in range(3)]

        prepared = await detector.maybe_detect(blocks, first_block_id=40)

        assert prepared is None
        assert detector._accumulated == blocks
        assert detector._accumulated_start_block_id == 40
        assert detector._counter == 3

    @pytest.mark.asyncio
    async def test_empty_batch_is_noop(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        detector._counter = 7

        prepared = await detector.maybe_detect([], first_block_id=10)

        assert prepared is None
        assert detector._counter == 7
        assert detector._accumulated == []
        assert detector._accumulated_start_block_id is None

    @pytest.mark.asyncio
    async def test_threshold_triggers_single_fork_and_updates_state(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        detector = _make_detector(tmp_path, detect_interval=3)

        seen: dict[str, BlockDetectorConfig] = {}

        async def _run_fork(*, fork_config: BlockDetectorConfig) -> PreparedFork:
            seen["fork_config"] = fork_config
            return _prepared(
                BlockDetectorResult(
                    completed=[
                        IRSemanticBlockRange(
                            title="completed", start_block=0, end_block=1
                        )
                    ],
                    still_buffered=[
                        IRSemanticBlockRange(
                            title="buffered", start_block=2, end_block=2
                        )
                    ],
                )
            )

        cast(Any, detector._fork_runner).run_fork = _run_fork

        blocks = [_user("a"), _assistant("b"), _user("c")]
        prepared = await detector.maybe_detect(blocks, first_block_id=0)

        assert prepared is not None
        result = await prepared.coro
        assert isinstance(result, BlockDetectorResult)
        completed = detector.integrate_result(result, prepared.fork_id)
        assert [b.title for b in completed] == ["completed"]
        assert [b.title for b in detector.completed_blocks] == ["completed"]
        assert [b.title for b in detector._semantic_buffer] == ["buffered"]

        fork_config = seen["fork_config"]
        assert fork_config.context_block_start_id == 0
        assert fork_config.full_context_blocks == blocks
        assert fork_config.context_block_buffer == blocks

        assert detector._start_block_id == 3
        assert detector._context_buffer == []
        assert detector._counter == 0

    @pytest.mark.asyncio
    async def test_global_indexing_survives_multi_batch_threshold_crossing(
        self, tmp_path: Path
    ) -> None:
        detector = _make_detector(tmp_path, detect_interval=4)
        seen_starts: list[int] = []

        async def _run_fork(*, fork_config: BlockDetectorConfig) -> PreparedFork:
            seen_starts.append(fork_config.context_block_start_id)
            return _prepared(
                BlockDetectorResult(
                    completed=[],
                    still_buffered=[
                        IRSemanticBlockRange(
                            title="buffered", start_block=12, end_block=13
                        )
                    ],
                )
            )

        cast(Any, detector._fork_runner).run_fork = _run_fork

        first = [_user("a"), _assistant("b")]
        second = [_user("c"), _assistant("d"), _user("e")]

        first_prepared = await detector.maybe_detect(first, first_block_id=10)
        second_prepared = await detector.maybe_detect(second, first_block_id=12)

        assert first_prepared is None
        assert second_prepared is not None
        result = await second_prepared.coro
        assert isinstance(result, BlockDetectorResult)
        detector.integrate_result(result, second_prepared.fork_id)
        assert detector._accumulated_start_block_id == 10
        assert seen_starts == [10]
        assert detector._start_block_id == 14
        assert detector._context_buffer == [second[-1]]
        assert detector._counter == 1

    @pytest.mark.asyncio
    async def test_multiple_threshold_crossings_schedule_one_fork_and_keep_remainder(
        self, tmp_path: Path
    ) -> None:
        detector = _make_detector(tmp_path, detect_interval=2)
        call_count = {"value": 0}

        async def _run_fork(*, fork_config: BlockDetectorConfig) -> PreparedFork:
            call_count["value"] += 1
            return _prepared(
                BlockDetectorResult(
                    completed=[
                        IRSemanticBlockRange(title="first", start_block=0, end_block=1)
                    ],
                    still_buffered=[],
                )
            )

        cast(Any, detector._fork_runner).run_fork = _run_fork

        blocks = [_user("a"), _assistant("b"), _user("c"), _assistant("d")]
        prepared = await detector.maybe_detect(blocks, first_block_id=0)

        assert prepared is not None
        result = await prepared.coro
        assert isinstance(result, BlockDetectorResult)
        completed = detector.integrate_result(result, prepared.fork_id)
        assert [b.title for b in completed] == ["first"]
        assert [b.title for b in detector.completed_blocks] == ["first"]
        assert call_count["value"] == 1
        assert detector._counter == 2


class TestRehydrate:
    def test_restores_detector_state_from_rehydrated_transcript(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path, detect_interval=4)
        blocks: list[
            IRUserTextBlock
            | IRAssistantTextBlock
            | IRImageBlock
            | IRThinkingBlock
            | IRToolCallBlock
            | IRToolResultBlock
        ] = [_user("done"), _assistant("buffered"), _user("raw")]
        completed = [IRSemanticBlockRange(title="Done", start_block=0, end_block=0)]
        buffered = [IRSemanticBlockRange(title="Buffered", start_block=1, end_block=1)]

        rehydrated = RehydrationResult(
            session_id="session_test",
            records=[],
            blocks=blocks,
            config=SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
            tools=[],
            last_completed_turn=1,
            pending_footers={},
            completed_semantic_block_ranges=completed,
            buffered_semantic_block_ranges=buffered,
            semantic_blocks=[],
            plan_proposal=None,
            skill_catalog=IRSkillCatalog(),
        )

        detector.rehydrate(rehydrated)

        assert detector.completed_blocks == completed
        assert detector.buffered_blocks == buffered
        assert detector._accumulated == blocks
        assert detector._accumulated_start_block_id == 0
        assert detector._counter == 3
        assert detector._start_block_id == 2
        assert detector._context_buffer == [blocks[2]]

    def test_rehydrate_empty_transcript_resets_detector_buffers(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        detector.completed_blocks = [
            IRSemanticBlockRange(title="stale", start_block=0, end_block=0)
        ]
        detector._semantic_buffer = [
            IRSemanticBlockRange(title="stale buffer", start_block=1, end_block=1)
        ]
        detector._accumulated = [_user("stale")]
        detector._accumulated_start_block_id = 0
        detector._context_buffer = [_user("stale")]
        detector._counter = 1

        rehydrated = RehydrationResult(
            session_id="session_test",
            records=[],
            blocks=[],
            config=SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
            tools=[],
            last_completed_turn=0,
            pending_footers={},
            completed_semantic_block_ranges=[],
            buffered_semantic_block_ranges=[],
            semantic_blocks=[],
            plan_proposal=None,
            skill_catalog=IRSkillCatalog(),
        )

        detector.rehydrate(rehydrated)

        assert detector.completed_blocks == []
        assert detector.buffered_blocks == []
        assert detector._accumulated == []
        assert detector._accumulated_start_block_id is None
        assert detector._counter == 0
        assert detector._context_buffer == []


class TestInboundRendering:
    def test_empty_sections_render_self_closing_tags(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)

        inbound = detector.build_inbound_block()

        assert inbound.origin == "system"
        assert "<block_detector_context>" in inbound.text
        assert "<completed_semantic_blocks />" in inbound.text
        assert "<buffered_semantic_blocks />" in inbound.text
        assert "<context_block_buffer />" in inbound.text
        assert "<instructions>" in inbound.text

    def test_renders_completed_buffered_and_context_sections(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        detector.completed_blocks = [
            IRSemanticBlockRange(title="Already done", start_block=0, end_block=1)
        ]
        detector._accumulated_start_block_id = 2
        detector._accumulated = [
            _user("buffered a"),
            _assistant("buffered b"),
            _user("raw c"),
        ]
        detector._semantic_buffer = [
            IRSemanticBlockRange(title="Buffered block", start_block=2, end_block=3)
        ]
        detector._start_block_id = 4
        detector._context_buffer = [detector._accumulated[2]]

        inbound = detector.build_inbound_block()
        text = inbound.text

        assert '<completed_semantic_block title="Already done" range="0-1" />' in text
        assert '<buffered_semantic_block title="Buffered block" range="2-3">' in text
        assert '<context_block id="2">' in text
        assert "**User:** buffered a" in text
        assert '<context_block id="3">' in text
        assert "**Assistant:** buffered b" in text
        assert "<context_block_buffer>" in text
        assert '<context_block id="4">' in text
        assert "**User:** raw c" in text

    def test_escapes_xml_sensitive_characters(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        detector.completed_blocks = [
            IRSemanticBlockRange(
                title='A <done> & "quoted"', start_block=0, end_block=0
            )
        ]
        detector._accumulated_start_block_id = 1
        detector._accumulated = [_user("5 < 6 & 7 > 3")]
        detector._start_block_id = 1
        detector._context_buffer = detector._accumulated[:]

        inbound = detector.build_inbound_block()
        text = inbound.text

        assert 'title="A &lt;done&gt; &amp; &quot;quoted&quot;"' in text
        assert "5 &lt; 6 &amp; 7 &gt; 3" in text

    def test_renders_mixed_block_kinds_as_markdown(self, tmp_path: Path) -> None:
        detector = _make_detector(tmp_path)
        detector._accumulated_start_block_id = 10
        detector._start_block_id = 10
        detector._context_buffer = [
            IRThinkingBlock(text="hidden reasoning", signature="sig"),
            IRToolCallBlock(
                call_id="toolu_1",
                tool="Bash",
                input={"command": "ls"},
            ),
            IRToolResultBlock(
                call_id="toolu_1",
                tool="Bash",
                content=[
                    IRToolTextBlock(text="file_a.py"),
                    IRImageBlock(
                        origin="tool",
                        source=IRImageBase64Source(
                            media_type="image/png", data="abc123"
                        ),
                    ),
                ],
            ),
        ]

        inbound = detector.build_inbound_block()
        text = inbound.text

        assert "**Thinking:** hidden reasoning" in text
        assert "**Tool call (`Bash`):** call_id=toolu_1, input={" in text
        assert "**Tool result (`Bash`):** file_a.py" in text
        assert "&lt;base64:image/png&gt;" in text
