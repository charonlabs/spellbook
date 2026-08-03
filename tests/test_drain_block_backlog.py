from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from scripts.dreaming.drain_block_backlog import (
    DrainRuntime,
    drain_block_backlog,
)
from spellbook.backends.model_backend import RequestSurface
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.footer import FooterController
from spellbook.fork import (
    BlockDetectorConfig,
    BlockDetectorResult,
    BlockSummarizerConfig,
    BlockSummarizerResult,
    ForkConfig,
    ForkRunner,
    ForkResult,
    PreparedFork,
)
from spellbook.homunculus.block_manager import BlockManager
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRImageBase64Source,
    IRImageBlock,
    IRSemanticBlock,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTokenRangeCount,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY

pytestmark = pytest.mark.asyncio


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _user(text: str) -> IRUserTextBlock:
    return IRUserTextBlock(text=text, origin="human")


def _assistant(text: str) -> IRAssistantTextBlock:
    return IRAssistantTextBlock(text=text)


def _image_result(tmp_path: Path, call_id: str) -> IRToolResultBlock:
    blob_path = Path("blobs") / "capture.png"
    full_blob_path = tmp_path / blob_path
    full_blob_path.parent.mkdir(parents=True, exist_ok=True)
    full_blob_path.write_bytes(b"image-data")
    return IRToolResultBlock(
        call_id=call_id,
        tool="Read",
        content=[
            IRImageBlock(
                origin="tool",
                source=IRImageBase64Source(media_type="image/png", data="aW1hZ2U="),
                blob_path=str(blob_path),
            )
        ],
        display={"kind": "text", "title": "Read Image"},
    )


def _write_source_transcript(
    tmp_path: Path,
    *,
    blocks: list[IRBlock],
    completed_prefix: int = 1,
    ttl_threshold: int = 20,
    buffered_ranges: list[IRSemanticBlockRange] | None = None,
) -> Path:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        cwd=tmp_path,
        hom_config=HomunculusConfig(
            detect_interval=2,
            tool_result_ttl_char_threshold=ttl_threshold,
        ),
    )
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("turn_1", blocks)
    recorder.end_turn()

    if completed_prefix > 0:
        semantic_range = IRSemanticBlockRange(
            title="Existing block",
            start_block=0,
            end_block=completed_prefix - 1,
            completed=True,
        )
        recorder.detect_blocks(
            BlockDetectorResult(completed=[semantic_range], still_buffered=[])
        )
        semantic_block = IRSemanticBlock(
            idx=0,
            title=semantic_range.title,
            range=semantic_range,
            toks=_count(5),
            full_toks=_count(5),
        )
        recorder.write_semantic_block(semantic_block)
        recorder.write_block_artifact(
            IRSemanticBlockSummary(
                headline="Existing summary",
                text="Already summarized.",
                facets=[],
                open_thread=None,
                toks=_count(3),
            ),
            semantic_block.id,
        )
    if buffered_ranges:
        recorder.detect_blocks(
            BlockDetectorResult(completed=[], still_buffered=buffered_ranges)
        )
    return transcript


class _FakeTokenCounter:
    async def count_block_content(self, block: IRBlock) -> int | None:
        return 1

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return len(blocks)

    async def count_frame(self) -> int | None:
        return 0

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return 0


class _FakeForkRunner:
    def __init__(self, recorder: Recorder, transcript_path: Path) -> None:
        self._recorder = recorder
        self._transcript_path = transcript_path
        self._counter = 0

    async def run_fork(self, fork_config: ForkConfig) -> PreparedFork:
        self._counter += 1
        fork_id = f"fake_fork_{self._counter}"
        self._recorder.summon_fork(
            fork_id=fork_id,
            fork_type=fork_config.type,
            child_transcript_path=str(
                self._transcript_path.parent / "forks" / f"{fork_id}.jsonl"
            ),
        )

        async def _run() -> ForkResult:
            match fork_config:
                case BlockDetectorConfig():
                    if not fork_config.context_block_buffer:
                        return BlockDetectorResult(completed=[], still_buffered=[])
                    start = fork_config.context_block_start_id
                    end = start + len(fork_config.context_block_buffer) - 1
                    return BlockDetectorResult(
                        completed=[
                            IRSemanticBlockRange(
                                title=f"Detected {start}-{end}",
                                start_block=start,
                                end_block=end,
                                completed=True,
                            )
                        ],
                        still_buffered=[],
                    )
                case BlockSummarizerConfig():
                    return BlockSummarizerResult(
                        summary=IRSemanticBlockSummary(
                            headline="Generated summary",
                            text="Generated by fake summarizer.",
                            facets=[],
                            open_thread=None,
                            toks=None,
                        )
                    )
                case _:
                    raise AssertionError(f"Unexpected fork config: {fork_config}")

        return PreparedFork(coro=_run(), fork_id=fork_id)

    def integrate_result(self, fork_id: str) -> None:
        self._recorder.shutdown_fork(fork_id)


def _runtime_builder(
    config: SpellbookConfig,
    transcript_path: Path,
    recorder: Recorder,
    context_projector: Any,
) -> DrainRuntime:
    nursery = Nursery(config=config)
    fork_runner = _FakeForkRunner(recorder, transcript_path)
    manager = BlockManager(
        config=config.hom_config,
        fork_runner=cast(ForkRunner, fork_runner),
        footer_c=FooterController(
            inbound_queue=InboundMessageQueue(),
            recorder=recorder,
        ),
        nursery=nursery,
        recorder=recorder,
        token_meter=TokenMeter(
            config=config.hom_config,
            tok_counter=_FakeTokenCounter(),  # type: ignore[arg-type]
        ),
        context_projector=context_projector,
        enable_block_metrics=True,
    )
    return DrainRuntime(block_manager=manager, nursery=nursery)


async def test_drain_block_backlog_dry_run_does_not_mutate(tmp_path: Path) -> None:
    transcript = _write_source_transcript(
        tmp_path,
        blocks=[_user("done"), _assistant("tail one"), _user("tail two")],
    )
    before = transcript.read_text(encoding="utf-8")

    report = await drain_block_backlog(
        transcript_path=transcript,
        apply=False,
        show_progress=False,
        write_report=False,
    )

    assert transcript.read_text(encoding="utf-8") == before
    assert report.dry_run is True
    assert report.start_block == 1
    assert report.target_blocks == 2
    assert report.processed_blocks == 0


async def test_drain_block_backlog_discard_buffered_policy_replays_from_completed_tail(
    tmp_path: Path,
) -> None:
    transcript = _write_source_transcript(
        tmp_path,
        blocks=[_user("done"), _assistant("tail one"), _user("tail two")],
        buffered_ranges=[
            IRSemanticBlockRange(title="Stale proposal", start_block=1, end_block=2)
        ],
    )

    report = await drain_block_backlog(
        transcript_path=transcript,
        apply=False,
        buffered_policy="discard",
        show_progress=False,
        write_report=False,
    )

    assert report.start_block == 1
    assert report.target_blocks == 2
    assert report.starting_buffered_blocks == 1
    assert report.ignored_buffered_blocks == 1


async def test_drain_block_backlog_preserve_buffered_policy_keeps_old_start(
    tmp_path: Path,
) -> None:
    transcript = _write_source_transcript(
        tmp_path,
        blocks=[_user("done"), _assistant("tail one"), _user("tail two")],
        buffered_ranges=[
            IRSemanticBlockRange(title="Stale proposal", start_block=1, end_block=2)
        ],
    )

    report = await drain_block_backlog(
        transcript_path=transcript,
        apply=False,
        buffered_policy="preserve",
        show_progress=False,
        write_report=False,
    )

    assert report.start_block == 3
    assert report.target_blocks == 0
    assert report.starting_buffered_blocks == 1
    assert report.ignored_buffered_blocks == 0


async def test_drain_block_backlog_appends_detection_and_summaries(
    tmp_path: Path,
) -> None:
    transcript = _write_source_transcript(
        tmp_path,
        blocks=[
            _user("done"),
            _assistant("tail one"),
            _user("tail two"),
            _assistant("tail three"),
            _user("tail four"),
        ],
    )

    report = await drain_block_backlog(
        transcript_path=transcript,
        chunk_size=2,
        interval=2,
        max_finalize_passes=0,
        apply=True,
        backup=False,
        show_progress=False,
        stream_events=False,
        write_report=False,
        runtime_builder=_runtime_builder,
    )

    rehydrated = Rehydrator(transcript).run()
    assert report.processed_blocks == 4
    assert report.detector_passes == 2
    assert report.new_semantic_blocks == 2
    assert report.new_summaries == 2
    assert len(rehydrated.semantic_blocks) == 3
    assert [block.range.start_block for block in rehydrated.semantic_blocks] == [
        0,
        1,
        3,
    ]
    assert all(
        "summary" in block.available_modes for block in rehydrated.semantic_blocks
    )
    new_blocks = rehydrated.semantic_blocks[1:]
    assert [
        block.full_toks.tokens if block.full_toks else None for block in new_blocks
    ] == [
        2,
        2,
    ]
    assert [block.toks.tokens if block.toks else None for block in new_blocks] == [2, 2]
    assert [
        next(
            artifact.toks.tokens
            for artifact in block.artifacts
            if isinstance(artifact, IRSemanticBlockSummary)
            and artifact.toks is not None
        )
        for block in new_blocks
    ] == [1, 1]


async def test_drain_block_backlog_discard_policy_supersedes_stale_buffered_ranges(
    tmp_path: Path,
) -> None:
    transcript = _write_source_transcript(
        tmp_path,
        blocks=[_user("done"), _assistant("tail one"), _user("tail two")],
        buffered_ranges=[
            IRSemanticBlockRange(title="Stale proposal", start_block=1, end_block=2)
        ],
    )

    report = await drain_block_backlog(
        transcript_path=transcript,
        chunk_size=2,
        interval=2,
        max_finalize_passes=0,
        apply=True,
        backup=False,
        buffered_policy="discard",
        show_progress=False,
        stream_events=False,
        write_report=False,
        runtime_builder=_runtime_builder,
    )

    rehydrated = Rehydrator(transcript).run()
    assert report.start_block == 1
    assert report.processed_blocks == 2
    assert report.new_semantic_blocks == 1
    assert report.final_buffered_blocks == 0
    assert rehydrated.buffered_semantic_block_ranges == []
    assert [
        (block.range.start_block, block.range.end_block)
        for block in rehydrated.semantic_blocks
    ] == [
        (0, 0),
        (1, 2),
    ]


async def test_drain_block_backlog_streams_blocks_and_summaries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript = _write_source_transcript(
        tmp_path,
        blocks=[
            _user("done"),
            _assistant("tail one"),
            _user("tail two"),
        ],
    )

    await drain_block_backlog(
        transcript_path=transcript,
        chunk_size=2,
        interval=2,
        max_finalize_passes=0,
        apply=True,
        backup=False,
        show_progress=False,
        stream_events=True,
        write_report=False,
        runtime_builder=_runtime_builder,
    )

    output = capsys.readouterr().out
    assert "Completed Blocks" in output
    assert "Detected 1-2" in output
    assert "Summary:" in output
    assert "Generated summary" in output
    assert "Generated by fake summarizer." in output


async def test_drain_block_backlog_streams_cleanly_with_progress(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transcript = _write_source_transcript(
        tmp_path,
        blocks=[
            _user("done"),
            _assistant("tail one"),
            _user("tail two"),
        ],
    )

    await drain_block_backlog(
        transcript_path=transcript,
        chunk_size=2,
        interval=2,
        max_finalize_passes=0,
        apply=True,
        backup=False,
        show_progress=True,
        stream_events=True,
        write_report=False,
        runtime_builder=_runtime_builder,
    )

    output = capsys.readouterr().out
    assert "Completed Blocks" in output
    assert "Generated summary" in output
    assert "Backlog" in output


async def test_drain_block_backlog_rejects_large_untracked_tool_result(
    tmp_path: Path,
) -> None:
    transcript = _write_source_transcript(
        tmp_path,
        blocks=[
            _user("done"),
            IRToolResultBlock(
                call_id="toolu_big",
                tool="Read",
                content=[IRToolTextBlock(text="large output\n" * 4)],
            ),
        ],
        ttl_threshold=20,
    )

    with pytest.raises(ValueError, match="large tool result"):
        await drain_block_backlog(
            transcript_path=transcript,
            apply=True,
            backup=False,
            show_progress=False,
            write_report=False,
            runtime_builder=_runtime_builder,
        )


async def test_drain_block_backlog_rejects_untracked_image_tool_result(
    tmp_path: Path,
) -> None:
    transcript = _write_source_transcript(
        tmp_path,
        blocks=[
            _user("done"),
            _image_result(tmp_path, "toolu_image"),
        ],
        ttl_threshold=20,
    )

    with pytest.raises(ValueError, match="large tool result"):
        await drain_block_backlog(
            transcript_path=transcript,
            apply=True,
            backup=False,
            show_progress=False,
            write_report=False,
            runtime_builder=_runtime_builder,
        )
