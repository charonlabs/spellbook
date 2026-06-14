from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from core_scripts.merge_stats import merge_stats
from spellbook.backends.model_backend import RequestSurface, TokenCounter
from spellbook.config import SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRSemanticBlock,
    IRSemanticBlockFacet,
    IRSemanticBlockPin,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTokenRangeCount,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY

pytestmark = pytest.mark.asyncio


class _LenTokenCounter:
    async def count_block_content(self, block: IRBlock) -> int | None:
        return None

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return len(blocks)

    async def count_frame(self) -> int | None:
        return 0

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return 999


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _summary(
    headline: str,
    *,
    facets: list[IRSemanticBlockFacet] | None = None,
) -> IRSemanticBlockSummary:
    return IRSemanticBlockSummary(
        headline=headline,
        text=f"{headline} text.",
        facets=facets or [],
        open_thread=None,
        toks=_count(1),
    )


def _write_transcript(tmp_path: Path) -> Path:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(cwd=tmp_path, model="claude-sonnet-4-6")
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn(
        "turn_1",
        [
            IRUserTextBlock(text="block zero", origin="human"),
            IRAssistantTextBlock(text="pinned sentence", origin="model"),
            IRUserTextBlock(text="unpinned sentence", origin="human"),
        ],
    )

    ranges = [
        IRSemanticBlockRange(
            title="Block zero",
            start_block=0,
            end_block=0,
            completed=True,
        ),
        IRSemanticBlockRange(
            title="Block one",
            start_block=1,
            end_block=2,
            completed=True,
        ),
    ]
    for semantic_range in ranges:
        recorder.detect_blocks(
            BlockDetectorResult(completed=[semantic_range], still_buffered=[])
        )

    blocks = [
        IRSemanticBlock(
            idx=0,
            title="Block zero",
            range=ranges[0],
            toks=_count(10),
            full_toks=_count(10),
        ),
        IRSemanticBlock(
            idx=1,
            title="Block one",
            range=ranges[1],
            toks=_count(20),
            full_toks=_count(20),
        ),
    ]
    facet = IRSemanticBlockFacet(
        id="facet_pin",
        title="Pinned moment",
        description="Keep this exact exchange.",
        start_block=1,
        end_block=2,
        resources=[],
    )
    for block in blocks:
        recorder.write_semantic_block(block)
    recorder.write_block_artifact(_summary("Summary zero"), blocks[0].id)
    recorder.write_block_artifact(
        _summary("Summary one", facets=[facet]),
        blocks[1].id,
    )
    recorder.apply_block_pin(
        IRSemanticBlockPin(
            kind="facet",
            facet_id="facet_pin",
            reason="Important exact sentence.",
        ),
        blocks[1].id,
    )
    recorder.end_turn()
    return transcript


async def test_merge_stats_counts_full_summary_context_and_pins(
    tmp_path: Path,
) -> None:
    director_opening = tmp_path / "director-opening.md"
    director_opening.write_text("# Director\n\nTell the story.\n", encoding="utf-8")
    render_path = tmp_path / "blocks-0-1.md"

    report = await merge_stats(
        transcript_path=_write_transcript(tmp_path),
        first_idx=0,
        second_idx=1,
        token_counter=cast(TokenCounter, _LenTokenCounter()),
        render_path=render_path,
        director_opening_path=director_opening,
    )

    assert report.full_count.tokens == 3
    assert report.summary_count.tokens == 5
    assert [line.title for line in report.block_lines] == [
        "Summary zero",
        "Summary one",
    ]
    assert len(report.pins) == 1
    assert report.pins[0].kind == "facet"
    assert report.pins[0].title == "Pinned moment"
    assert report.pins[0].tokens == 2
    assert report.render_path == render_path.resolve()

    rendered = render_path.read_text(encoding="utf-8")
    assert rendered.startswith(
        "# Director\n\nTell the story.\n\n---\n\n# Source Blocks"
    )
    assert "## Block 0: Summary zero" in rendered
    assert "### User Message (context block 0)" in rendered
    assert "block zero" in rendered
    assert "### Assistant Message (context block 1)" in rendered
    assert "pinned sentence" in rendered
    assert "### User Message (context block 2)" in rendered
    assert "unpinned sentence" in rendered
    assert "<!-- context_block=1 -->" in rendered
    assert "turn_id=" not in rendered
    assert "event_id=" not in rendered
    assert (
        '<pin kind="facet" block_idx="1" title="Pinned moment" '
        'reason="Important exact sentence." facet_id="facet_pin">'
    ) in rendered
    assert rendered.count('<pin kind="facet"') == 1
    assert rendered.count("</pin>") == 1
    assert "is_error:" not in rendered
    pin_start = rendered.index('<pin kind="facet"')
    pin_end = rendered.index("</pin>", pin_start)
    assert "### Assistant Message (context block 1)" in rendered[pin_start:pin_end]
    assert "### User Message (context block 2)" in rendered[pin_start:pin_end]
    assert "### User Message (context block 0)" not in rendered[pin_start:pin_end]
