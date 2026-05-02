from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from scripts import analyze_initial_context_plan
from scripts.analyze_initial_context_plan import (
    InitialContextPlanReport,
)
from scripts.analyze_initial_context_plan import (
    analyze_initial_context_plan as analyze,
)
from spellbook.backends.model_backend import RequestSurface, TokenCounter
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRBlock,
    IRSemanticBlock,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTokenRangeCount,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY

pytestmark = pytest.mark.asyncio


class _FakeTokenCounter:
    def __init__(self, counts: list[int | None]):
        self.counts = list(counts)

    async def count_block_content(self, block: IRBlock) -> int | None:
        return None

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return None

    async def count_frame(self) -> int | None:
        return None

    async def count_surface(self, surface: RequestSurface) -> int | None:
        if not self.counts:
            raise AssertionError("No fake token counts left.")
        return self.counts.pop(0)


class _FakeSurfaceBuilder:
    def __init__(self) -> None:
        self.built_blocks: list[list[IRBlock]] = []

    def build(self, blocks: list[IRBlock]) -> RequestSurface:
        self.built_blocks.append(blocks)
        return RequestSurface(model="test-model", messages=[])


def _user(text: str) -> IRUserTextBlock:
    return IRUserTextBlock(text=text, origin="human")


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _write_transcript(
    tmp_path: Path,
    *,
    summary_ready: bool = True,
) -> Path:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        hom_config=HomunculusConfig(soft_threshold=300),
    )
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("turn_1", [_user("one"), _user("two"), _user("three")])

    ranges = [
        IRSemanticBlockRange(
            title=f"Block {idx}",
            start_block=idx,
            end_block=idx,
            completed=True,
        )
        for idx in range(3)
    ]
    recorder.detect_blocks(BlockDetectorResult(completed=ranges, still_buffered=[]))
    for idx, semantic_range in enumerate(ranges):
        block = IRSemanticBlock(
            idx=idx,
            title=semantic_range.title,
            range=semantic_range,
            toks=_count(100),
            full_toks=_count(100),
        )
        recorder.write_semantic_block(block)
        if summary_ready or idx != 1:
            recorder.write_block_artifact(
                IRSemanticBlockSummary(
                    headline=f"Summary {idx}",
                    text=f"Summary text {idx}.",
                    facets=[],
                    open_thread=None,
                    toks=_count(10),
                ),
                block.id,
            )
    recorder.end_turn()
    return transcript


async def test_analyze_initial_context_plan_walks_newest_full_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = _write_transcript(tmp_path)
    builder = _FakeSurfaceBuilder()
    monkeypatch.setattr(
        analyze_initial_context_plan,
        "_build_surface_builder",
        lambda config: builder,
    )

    report = await analyze(
        transcript_path=transcript,
        threshold=300,
        after_over=1,
        token_counter=cast(TokenCounter, _FakeTokenCounter([100, 200, 350, 500])),
    )

    assert isinstance(report, InitialContextPlanReport)
    assert [candidate.full_suffix_blocks for candidate in report.candidates] == [
        0,
        1,
        2,
        3,
    ]
    assert [candidate.tokens for candidate in report.candidates] == [
        100,
        200,
        350,
        500,
    ]
    assert report.last_under_threshold is not None
    assert report.last_under_threshold.full_suffix_blocks == 1
    assert report.first_over_threshold is not None
    assert report.first_over_threshold.full_suffix_blocks == 2

    all_summary = builder.built_blocks[0]
    newest_one_full = builder.built_blocks[1]
    assert [
        block.origin for block in all_summary if isinstance(block, IRUserTextBlock)
    ] == ["memory", "memory", "memory"]
    assert [
        block.origin for block in newest_one_full if isinstance(block, IRUserTextBlock)
    ] == ["memory", "memory", "human"]


async def test_analyze_initial_context_plan_requires_summary_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = _write_transcript(tmp_path, summary_ready=False)
    monkeypatch.setattr(
        analyze_initial_context_plan,
        "_build_surface_builder",
        lambda config: _FakeSurfaceBuilder(),
    )

    with pytest.raises(ValueError, match="Missing summaries"):
        await analyze(
            transcript_path=transcript,
            token_counter=cast(TokenCounter, _FakeTokenCounter([100])),
        )


async def test_analyze_initial_context_plan_applies_tool_result_ttls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        hom_config=HomunculusConfig(
            soft_threshold=300,
            tool_result_ttl_turns=1,
        ),
    )
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("turn_1", [_user("please read this")])
    recorder.write_block(IRToolCallBlock(call_id="toolu_big", tool="Read", input={}))
    recorder.write_block(
        IRToolResultBlock(
            call_id="toolu_big",
            tool="Read",
            content=[IRToolTextBlock(text="huge result\n" * 10)],
        )
    )
    recorder.write_tool_result_ttl(
        call_id="toolu_big",
        replace_content="[Read: collapsed]",
        ttl=1,
        trigger="end_turn",
    )
    recorder.end_turn()

    builder = _FakeSurfaceBuilder()
    monkeypatch.setattr(
        analyze_initial_context_plan,
        "_build_surface_builder",
        lambda config: builder,
    )

    report = await analyze(
        transcript_path=transcript,
        threshold=300,
        token_counter=cast(TokenCounter, _FakeTokenCounter([100])),
    )

    assert report.semantic_blocks == 0
    built = builder.built_blocks[0]
    result_block = next(
        block for block in built if isinstance(block, IRToolResultBlock)
    )
    assert result_block.content == [IRToolTextBlock(text="[Read: collapsed]")]
