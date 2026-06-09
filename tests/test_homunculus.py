from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter

from spellbook.backends.model_backend import RequestSurface, TokenCounter
from spellbook.cancel_token import CancelToken
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.footer import FooterController
from spellbook.fork import ForkRunner
from spellbook.homunculus import Homunculus, HomunculusRoundLifecycle
from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRCompactBlockIntent,
    IRContextPlanProposalRecord,
    IRExecution,
    IRFooterQueueRecord,
    IRGeneration,
    IRRecord,
    IRRuntimeConfigRecord,
    IRSemanticBlock,
    IRSemanticBlockApplyModeRecord,
    IRSemanticBlockFacet,
    IRSemanticBlockPin,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTokenRangeCount,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolResultTTLRecord,
    IRToolTextBlock,
    IRUsage,
    IRUserTextBlock,
)
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.round_lifecycle import RoundContext
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY

pytestmark = pytest.mark.asyncio


class _FakeTokenCounter:
    async def count_block_content(self, block: IRBlock) -> int | None:
        return 10

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return len(blocks) * 10

    async def count_frame(self) -> int | None:
        return 100

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return None


def _user(text: str) -> IRUserTextBlock:
    return IRUserTextBlock(text=text, origin="human")


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="prefix_delta", exact=True)


def _tool_result(call_id: str, text: str, *, tool: str = "Read") -> IRToolResultBlock:
    return IRToolResultBlock(
        call_id=call_id,
        tool=tool,
        content=[IRToolTextBlock(text=text)],
    )


def _summary_block(
    *,
    full_tokens: int | None = None,
    summary_tokens: int | None = None,
) -> IRSemanticBlock:
    semantic_range = IRSemanticBlockRange(
        title="Full block",
        start_block=0,
        end_block=0,
        completed=True,
    )
    summary = IRSemanticBlockSummary(
        headline="Compact headline",
        text="Compact summary text.",
        facets=[],
        open_thread=None,
        toks=_count(summary_tokens) if summary_tokens is not None else None,
    )
    full_toks = _count(full_tokens) if full_tokens is not None else None
    return IRSemanticBlock(
        idx=0,
        title="Full block",
        range=semantic_range,
        toks=full_toks,
        full_toks=full_toks,
        available_modes=["full", "summary"],
        artifacts=[summary],
    )


def _rehydrated(
    tmp_path: Path,
    *,
    blocks: list[IRBlock],
    semantic_blocks: list[IRSemanticBlock],
) -> RehydrationResult:
    return RehydrationResult(
        session_id="session_test",
        records=[],
        config=SpellbookConfig(cwd=tmp_path),
        blocks=blocks,
        tools=[],
        last_completed_turn=0,
        pending_footers={},
        completed_semantic_block_ranges=[block.range for block in semantic_blocks],
        buffered_semantic_block_ranges=[],
        semantic_blocks=semantic_blocks,
        plan_proposal=None,
        skill_catalog=IRSkillCatalog(),
    )


def _read_records(path: Path) -> list[IRRecord]:
    adapter = TypeAdapter(IRRecord)
    records: list[IRRecord] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(adapter.validate_json(line))
    return records


def _homunculus(
    tmp_path: Path, homunculus_config: HomunculusConfig | None = None
) -> Homunculus:
    return _homunculus_with_transcript(
        tmp_path,
        tmp_path / "transcript.jsonl",
        homunculus_config,
        initialize=True,
    )


def _homunculus_with_transcript(
    tmp_path: Path,
    transcript: Path,
    homunculus_config: HomunculusConfig | None = None,
    *,
    initialize: bool,
) -> Homunculus:
    config = SpellbookConfig(cwd=tmp_path)
    recorder = Recorder(
        config,
        transcript,
        "session_test",
        DEFAULT_TOOL_REGISTRY,
    )
    if initialize:
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("turn_1", [])
    footer_c = FooterController(
        inbound_queue=InboundMessageQueue(),
        recorder=recorder,
    )
    return Homunculus(
        config=homunculus_config or HomunculusConfig(),
        footer_c=footer_c,
        recorder=recorder,
        token_counter=cast(TokenCounter, _FakeTokenCounter()),
        nursery=Nursery(config=config),
        fork_runner=cast(ForkRunner, object()),
    )


def _make_core_recorder(tmp_path: Path) -> tuple[Recorder, Path]:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(cwd=tmp_path)
    recorder = Recorder(config, transcript, "session_test", DEFAULT_TOOL_REGISTRY)
    return recorder, transcript


async def test_forget_invalidates_reflect_counts_and_rerenders_between_rounds(
    tmp_path: Path,
) -> None:
    full_block = _user("Full transcript text.")
    tail_block = IRAssistantTextBlock(text="Tail remains full.", origin="model")
    semantic_block = _summary_block()
    homunculus = _homunculus(tmp_path)
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[full_block, tail_block],
            semantic_blocks=[semantic_block],
        )
    )

    await homunculus.forget(0)
    reflect, _ = await homunculus.render_reflect()
    ctx = RoundContext(
        blocks=[full_block, tail_block],
        round_number=1,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )

    await HomunculusRoundLifecycle(homunculus).between_rounds(ctx)

    assert "Currently calculating rendered count" in reflect
    assert len(ctx.blocks) == 2
    assert isinstance(ctx.blocks[0], IRUserTextBlock)
    assert ctx.blocks[0].origin == "memory"
    assert 'block_idx="0"' in ctx.blocks[0].text
    assert "Compact headline" in ctx.blocks[0].text
    assert isinstance(ctx.blocks[1], IRAssistantTextBlock)
    assert ctx.blocks[1].text == "Tail remains full."


async def test_between_rounds_is_noop_without_scheduled_rerender(
    tmp_path: Path,
) -> None:
    full_block = _user("Full transcript text.")
    homunculus = _homunculus(tmp_path)
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[full_block],
            semantic_blocks=[],
        )
    )
    ctx = RoundContext(
        blocks=[full_block],
        round_number=1,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )

    await HomunculusRoundLifecycle(homunculus).between_rounds(ctx)

    assert ctx.blocks == [full_block]


async def test_pin_summary_block_invalidates_and_rerenders_between_rounds(
    tmp_path: Path,
) -> None:
    full_block = _user("Full transcript text.")
    tail_block = IRAssistantTextBlock(text="Tail remains full.", origin="model")
    semantic_block = _summary_block().model_copy(
        update={
            "mode": "summary",
            "toks": _count(7),
            "full_toks": _count(40),
        }
    )
    homunculus = _homunculus(tmp_path)
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[full_block, tail_block],
            semantic_blocks=[semantic_block],
        )
    )
    ctx = RoundContext(
        blocks=[
            IRUserTextBlock(text="previous compact summary", origin="memory"),
            tail_block,
        ],
        round_number=1,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )

    await homunculus.pin(0, "Needs the exact wording.")
    await HomunculusRoundLifecycle(homunculus).between_rounds(ctx)

    assert ctx.blocks == [full_block, tail_block]


async def test_reflect_shows_pin_reason(tmp_path: Path) -> None:
    full_block = _user("Full transcript text.")
    semantic_block = _summary_block().model_copy(
        update={
            "pin": IRSemanticBlockPin(
                kind="block",
                reason="This setup was hard won.",
            )
        }
    )
    homunculus = _homunculus(tmp_path)
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[full_block],
            semantic_blocks=[semantic_block],
        )
    )

    reflect, _ = await homunculus.render_reflect()

    assert "pinned: This setup was hard won." in reflect


async def test_reflect_shows_summary_facets_and_facet_pin_reason(
    tmp_path: Path,
) -> None:
    full_block = _user("Full transcript text.")
    facet = IRSemanticBlockFacet(
        id="facet_design",
        title="Design moment",
        description="They found the shape.",
        start_block=0,
        end_block=0,
        resources=[],
    )
    semantic_block = _summary_block().model_copy(
        update={
            "artifacts": [
                IRSemanticBlockSummary(
                    headline="Compact headline",
                    text="Compact summary text.",
                    facets=[facet],
                    open_thread=None,
                    toks=None,
                )
            ],
            "available_modes": ["full", "summary"],
            "facet_pins": [
                IRSemanticBlockPin(
                    kind="facet",
                    reason="Keep Ryan's exact question.",
                    facet_id="facet_design",
                )
            ],
        }
    )
    homunculus = _homunculus(tmp_path)
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[full_block],
            semantic_blocks=[semantic_block],
        )
    )

    reflect, _ = await homunculus.render_reflect()

    assert "facet facet_design: Design moment (0-0)" in reflect
    assert "[pinned: Keep Ryan's exact question.]" in reflect


async def test_reflect_block_idx_previews_summary_rendering(tmp_path: Path) -> None:
    full_block = _user("Ryan asks the key question.")
    answer_block = IRAssistantTextBlock(text="The answer lands.", origin="model")
    facet = IRSemanticBlockFacet(
        id="facet_decision",
        title="Decision moment",
        description="The exact exchange should stay vivid.",
        start_block=0,
        end_block=1,
        resources=[],
    )
    summary = IRSemanticBlockSummary(
        headline="Decision captured",
        text="The summarized decision.",
        facets=[facet],
        open_thread="Keep checking the preview.",
        toks=_count(12),
    )
    semantic_block = _summary_block().model_copy(
        update={
            "title": "Design decision",
            "range": IRSemanticBlockRange(
                title="Design decision",
                start_block=0,
                end_block=1,
                completed=True,
            ),
            "available_modes": ["full", "summary"],
            "artifacts": [summary],
            "facet_pins": [
                IRSemanticBlockPin(
                    kind="facet",
                    reason="Keep the original exchange.",
                    facet_id="facet_decision",
                )
            ],
        }
    )
    homunculus = _homunculus(tmp_path)
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[full_block, answer_block],
            semantic_blocks=[semantic_block],
        )
    )

    reflect, _ = await homunculus.render_reflect(block_idx=0)

    assert '# Block 0: "Design decision"' in reflect
    assert "Preview mode: summary." in reflect
    assert "This preview renders as 4 content blocks" in reflect
    assert '<spellbook-memory block_idx="0" mode="summary" turns="0-1">' in reflect
    assert "Decision captured" in reflect
    assert "The summarized decision." in reflect
    assert 'Pinned facets follow as original conversation: "Decision moment"' in reflect
    assert "**User:** Ryan asks the key question." in reflect
    assert "**Assistant:** The answer lands." in reflect
    assert "End of pinned facets." in reflect
    assert "Open thread: Keep checking the preview." in reflect


async def test_reflect_block_idx_requires_summary_artifact(tmp_path: Path) -> None:
    full_block = _user("Full transcript text.")
    semantic_block = IRSemanticBlock(
        idx=0,
        title="Unsummary-ready block",
        range=IRSemanticBlockRange(
            title="Unsummary-ready block",
            start_block=0,
            end_block=0,
            completed=True,
        ),
        toks=None,
        full_toks=None,
    )
    homunculus = _homunculus(tmp_path)
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[full_block],
            semantic_blocks=[semantic_block],
        )
    )

    with pytest.raises(ValueError, match="no summary artifact"):
        await homunculus.render_reflect(block_idx=0)


async def test_planner_compacts_between_rounds_records_source_and_rerenders(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    full_block = _user("Full transcript text.")
    tail_block = IRAssistantTextBlock(text="Tail remains full.", origin="model")
    semantic_block = _summary_block(full_tokens=120, summary_tokens=15)
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(soft_threshold=50, medium_threshold=100, hard_threshold=200),
    )
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[full_block, tail_block],
            semantic_blocks=[semantic_block],
        )
    )
    await homunculus.integrate_generation(
        IRGeneration(
            model="test-model",
            blocks=[],
            stop_reason="end_turn",
            usage=IRUsage(input_tokens=100),
        )
    )
    ctx = RoundContext(
        blocks=[full_block, tail_block],
        round_number=1,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )

    await HomunculusRoundLifecycle(homunculus).between_rounds(ctx)

    assert len(ctx.blocks) == 2
    assert isinstance(ctx.blocks[0], IRUserTextBlock)
    assert ctx.blocks[0].origin == "memory"
    assert 'block_idx="0"' in ctx.blocks[0].text
    assert "Compact headline" in ctx.blocks[0].text
    assert ctx.blocks[1] == tail_block

    records = _read_records(transcript)
    mode_records = [
        record
        for record in records
        if isinstance(record, IRSemanticBlockApplyModeRecord)
    ]
    assert len(mode_records) == 1
    assert mode_records[0].mode == "summary"
    assert mode_records[0].source == "planner"
    assert mode_records[0].block_id == semantic_block.id

    footer_records = [
        record for record in records if isinstance(record, IRFooterQueueRecord)
    ]
    compaction_records = [
        record for record in footer_records if record.footer.type == "compaction"
    ]
    assert len(compaction_records) == 1
    assert compaction_records[0].footer.source == "planner"
    assert "Planner:" in compaction_records[0].footer.text
    assert "compacted [Block 0]" in compaction_records[0].footer.text
    assert "savings: ~105 tokens" in compaction_records[0].footer.text


async def test_planner_proposal_records_plan_without_rerendering(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    full_block = _user("Full transcript text.")
    semantic_block = _summary_block(full_tokens=120, summary_tokens=15)
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(soft_threshold=50, medium_threshold=100, hard_threshold=200),
    )
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[full_block],
            semantic_blocks=[semantic_block],
        )
    )
    await homunculus.integrate_generation(
        IRGeneration(
            model="test-model",
            blocks=[],
            stop_reason="end_turn",
            usage=IRUsage(input_tokens=50),
        )
    )
    ctx = RoundContext(
        blocks=[full_block],
        round_number=1,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )

    await HomunculusRoundLifecycle(homunculus).between_rounds(ctx)
    reflect, _ = await homunculus.render_reflect()

    assert ctx.blocks == [full_block]
    assert "## Planner" in reflect
    assert "Pending proposal:" in reflect
    assert 'compact [Block 0]: "Full block" (savings: ~105 tokens)' in reflect

    records = _read_records(transcript)
    proposal_records = [
        record for record in records if isinstance(record, IRContextPlanProposalRecord)
    ]
    assert len(proposal_records) == 1
    intent = proposal_records[0].plan.intents[0]
    assert isinstance(intent, IRCompactBlockIntent)
    assert intent.block_idx == semantic_block.idx

    mode_records = [
        record
        for record in records
        if isinstance(record, IRSemanticBlockApplyModeRecord)
    ]
    assert mode_records == []

    footer_records = [
        record for record in records if isinstance(record, IRFooterQueueRecord)
    ]
    compaction_records = [
        record for record in footer_records if record.footer.type == "compaction"
    ]
    assert len(compaction_records) == 1
    assert "new proposal" in compaction_records[0].footer.text
    assert "savings: ~105 tokens" in compaction_records[0].footer.text


async def test_large_tool_result_auto_ttl_persists_and_collapses_after_turn(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(
            tool_result_ttl_turns=1,
            tool_result_ttl_char_threshold=20,
        ),
    )
    call = IRToolCallBlock(call_id="toolu_big", tool="Read", input={})
    output = "large output\n" * 4
    result = _tool_result("toolu_big", output)
    result = result.model_copy(
        update={
            "display": {
                "kind": "read",
                "path": "/tmp/big.txt",
                "start_line": 1,
                "end_line": 4,
                "total_lines": 4,
            }
        }
    )
    await homunculus.rehydrate(_rehydrated(tmp_path, blocks=[], semantic_blocks=[]))

    await homunculus.integrate_generation(
        IRGeneration(
            model="test-model",
            blocks=[call],
            stop_reason="tool_use",
            usage=None,
        )
    )
    await homunculus.integrate_execution(IRExecution(blocks=[result]))
    before_tick = await homunculus.render_context([])
    ctx = RoundContext(
        blocks=before_tick,
        round_number=1,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )

    await HomunculusRoundLifecycle(homunculus).on_loop_exit(ctx, "end_turn")
    after_tick = await homunculus.render_context([])

    assert before_tick == [call, result]
    assert after_tick[0] == call
    assert isinstance(after_tick[1], IRToolResultBlock)
    assert after_tick[1].content == [
        IRToolTextBlock(
            text="[Read: /tmp/big.txt - lines 1-4 of 4. Full output saved to tool-outputs/toolu_big.txt]"
        )
    ]
    assert (tmp_path / "tool-outputs" / "toolu_big.txt").read_text() == output

    records = _read_records(transcript)
    ttl_records = [
        record for record in records if isinstance(record, IRToolResultTTLRecord)
    ]
    assert len(ttl_records) == 1
    assert ttl_records[0].call_id == "toolu_big"
    assert ttl_records[0].ttl == 1
    assert ttl_records[0].trigger == "end_turn"
    assert ttl_records[0].delivered_turn == 1


async def test_small_tool_result_does_not_auto_register_ttl(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(tool_result_ttl_char_threshold=50),
    )
    await homunculus.rehydrate(_rehydrated(tmp_path, blocks=[], semantic_blocks=[]))

    await homunculus.integrate_execution(
        IRExecution(blocks=[_tool_result("toolu_small", "short")])
    )

    records = _read_records(transcript)
    assert not any(isinstance(record, IRToolResultTTLRecord) for record in records)


async def test_reflect_tool_results_default_shows_pending_ttls_only(
    tmp_path: Path,
) -> None:
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(
            tool_result_ttl_turns=2,
            tool_result_ttl_char_threshold=20,
        ),
    )
    await homunculus.rehydrate(_rehydrated(tmp_path, blocks=[], semantic_blocks=[]))
    big = _tool_result("toolu_big", "large output\n" * 4).model_copy(
        update={
            "display": {
                "kind": "read",
                "path": "/tmp/big.txt",
                "start_line": 1,
                "end_line": 4,
                "total_lines": 4,
            }
        }
    )
    small = _tool_result("toolu_small", "short", tool="Bash")
    skill = _tool_result("toolu_skill", "large skill output\n" * 4, tool="Skill")

    await homunculus.integrate_execution(IRExecution(blocks=[big, small, skill]))
    text, display = homunculus.render_tool_results()

    assert display["kind"] == "reflect_tool_results"
    assert display["shown"] == 1
    assert display["total"] == 3
    assert display["pending"] == 1
    assert display["untracked"] == 2
    assert "Showing 1 of 3 tool result(s) (default view)." in text
    assert "Tracked: 1 pending, 0 collapsed. Untracked: 2 (0 above threshold)." in text
    assert "toolu_big Read (/tmp/big.txt)" in text
    assert "status: pending TTL, 2 turns remaining" in text
    assert "saved: tool-outputs/toolu_big.txt" in text
    assert "toolu_small" not in text
    assert "toolu_skill" not in text


async def test_reflect_tool_results_verbose_includes_untracked_and_ignored(
    tmp_path: Path,
) -> None:
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(tool_result_ttl_char_threshold=20),
    )
    small = _tool_result("toolu_small", "short", tool="Bash")
    skill = _tool_result("toolu_skill", "large skill output\n" * 4, tool="Skill")
    historical = _tool_result("toolu_old", "large historical output\n" * 4)
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[small, skill, historical],
            semantic_blocks=[],
        )
    )

    default_text, default_display = homunculus.render_tool_results()
    verbose_text, verbose_display = homunculus.render_tool_results(verbose=True)

    assert default_display["shown"] == 1
    assert "toolu_old Read" in default_text
    assert "status: untracked, above TTL threshold" in default_text
    assert "toolu_small" not in default_text
    assert "toolu_skill" not in default_text

    assert verbose_display["shown"] == 3
    assert "Showing 3 of 3 tool result(s) (verbose view)." in verbose_text
    assert "toolu_small Bash" in verbose_text
    assert "status: untracked, below TTL threshold" in verbose_text
    assert "toolu_skill Skill" in verbose_text
    assert "status: ignored, tool is excluded from auto-TTL" in verbose_text


async def test_forget_tool_result_registers_manual_ttl_and_rerenders(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(tool_result_ttl_char_threshold=10_000),
    )
    call = IRToolCallBlock(call_id="toolu_big_manual", tool="Read", input={})
    output = "large output\n" * 4
    result = _tool_result("toolu_big_manual", output).model_copy(
        update={
            "display": {
                "kind": "read",
                "path": "/tmp/big.txt",
                "start_line": 1,
                "end_line": 4,
                "total_lines": 4,
            }
        }
    )
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[call, result],
            semantic_blocks=[],
        )
    )

    message = await homunculus.forget_tool_result("toolu_big")
    ctx = RoundContext(
        blocks=[call, result],
        round_number=1,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )
    await HomunculusRoundLifecycle(homunculus).between_rounds(ctx)

    assert "Tool result toolu_big_manual successfully forgotten." in message
    assert "Full output saved to tool-outputs/toolu_big_manual.txt" in message
    assert (tmp_path / "tool-outputs" / "toolu_big_manual.txt").read_text() == output
    assert ctx.blocks[0] == call
    collapsed = ctx.blocks[1]
    assert isinstance(collapsed, IRToolResultBlock)
    assert collapsed.content == [
        IRToolTextBlock(
            text="[Read: /tmp/big.txt - lines 1-4 of 4. Full output saved to tool-outputs/toolu_big_manual.txt]"
        )
    ]

    ttl_records = [
        record
        for record in _read_records(transcript)
        if isinstance(record, IRToolResultTTLRecord)
    ]
    assert len(ttl_records) == 1
    assert ttl_records[0].call_id == "toolu_big_manual"
    assert ttl_records[0].ttl == 0
    assert ttl_records[0].source == "manual"


async def test_forget_tool_result_overrides_pending_auto_ttl(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(
            tool_result_ttl_turns=5,
            tool_result_ttl_char_threshold=20,
        ),
    )
    output = "large output\n" * 4
    result = _tool_result("toolu_auto", output, tool="Bash").model_copy(
        update={
            "display": {
                "kind": "command",
                "command": "cat big.txt",
                "exit_code": 0,
            }
        }
    )
    await homunculus.rehydrate(_rehydrated(tmp_path, blocks=[], semantic_blocks=[]))
    await homunculus.integrate_execution(IRExecution(blocks=[result]))

    await homunculus.forget_tool_result("toolu_auto")
    rendered = await homunculus.render_context([])

    collapsed = rendered[0]
    assert isinstance(collapsed, IRToolResultBlock)
    assert collapsed.content == [
        IRToolTextBlock(
            text="[Bash: `cat big.txt` - exit 0, 4 lines. Full output saved to tool-outputs/toolu_auto.txt]"
        )
    ]

    ttl_records = [
        record
        for record in _read_records(transcript)
        if isinstance(record, IRToolResultTTLRecord)
    ]
    assert [record.source for record in ttl_records] == ["auto", "manual"]
    assert [record.ttl for record in ttl_records] == [5, 0]
    assert ttl_records[0].output_ref == ttl_records[1].output_ref


async def test_forget_tool_result_rejects_ambiguous_prefix(tmp_path: Path) -> None:
    homunculus = _homunculus(tmp_path)
    await homunculus.rehydrate(
        _rehydrated(
            tmp_path,
            blocks=[
                _tool_result("toolu_same_a", "a"),
                _tool_result("toolu_same_b", "b"),
            ],
            semantic_blocks=[],
        )
    )

    with pytest.raises(ValueError, match="ambiguous"):
        await homunculus.forget_tool_result("toolu_same")


async def test_configure_reads_runtime_ttl_settings_without_recording(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(
            tool_result_ttl_enabled=True,
            tool_result_ttl_turns=7,
            tool_result_ttl_char_threshold=1234,
        ),
    )
    await homunculus.rehydrate(_rehydrated(tmp_path, blocks=[], semantic_blocks=[]))

    text, display = homunculus.configure()

    assert display == {
        "kind": "configure",
        "action": "read",
        "namespace": "tool_result_ttl",
        "updates": {},
        "effective": {
            "enabled": True,
            "ttl_turns": 7,
            "char_threshold": 1234,
        },
    }
    assert "- ttl_enabled: True" in text
    assert "- ttl_turns: 7" in text
    assert "- ttl_char_threshold: 1234" in text
    assert not any(
        isinstance(record, IRRuntimeConfigRecord)
        for record in _read_records(transcript)
    )


async def test_configure_persists_and_applies_ttl_runtime_settings(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "transcript.jsonl"
    homunculus = _homunculus(
        tmp_path,
        HomunculusConfig(
            tool_result_ttl_enabled=True,
            tool_result_ttl_turns=3,
            tool_result_ttl_char_threshold=20,
        ),
    )
    await homunculus.rehydrate(_rehydrated(tmp_path, blocks=[], semantic_blocks=[]))

    text, display = homunculus.configure(key="ttl_enabled", value=False)

    assert display["action"] == "update"
    assert display["updates"] == {
        "enabled": False,
    }
    assert display["effective"] == {
        "enabled": False,
        "ttl_turns": 3,
        "char_threshold": 20,
    }
    assert "- ttl_enabled: True -> False" in text

    records = _read_records(transcript)
    config_records = [
        record for record in records if isinstance(record, IRRuntimeConfigRecord)
    ]
    assert len(config_records) == 1
    assert config_records[0].namespace == "tool_result_ttl"
    assert config_records[0].updates == display["updates"]
    assert config_records[0].effective == display["effective"]

    await homunculus.integrate_execution(
        IRExecution(blocks=[_tool_result("toolu_disabled", "large output\n" * 20)])
    )
    ttl_records = [
        record
        for record in _read_records(transcript)
        if isinstance(record, IRToolResultTTLRecord)
    ]
    assert ttl_records == []


async def test_configure_rehydrates_runtime_ttl_settings(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    homunculus = _homunculus(tmp_path)
    await homunculus.rehydrate(_rehydrated(tmp_path, blocks=[], semantic_blocks=[]))
    homunculus.configure(key="ttl_turns", value=4)
    homunculus.configure(key="ttl_char_threshold", value="99")

    rehydrated = Rehydrator(transcript).run()
    resumed = _homunculus_with_transcript(
        tmp_path,
        transcript,
        initialize=False,
    )
    await resumed.rehydrate(rehydrated)
    text, display = resumed.configure()

    assert len(rehydrated.runtime_config_updates) == 2
    assert display["effective"] == {
        "enabled": True,
        "ttl_turns": 4,
        "char_threshold": 99,
    }
    assert "- ttl_turns: 4" in text
    assert "- ttl_char_threshold: 99" in text


async def test_ttl_rehydrates_remaining_from_completed_turns(tmp_path: Path) -> None:
    call = IRToolCallBlock(call_id="toolu_rehydrate", tool="Bash", input={})
    result = _tool_result("toolu_rehydrate", "full output", tool="Bash")
    recorder, transcript = _make_core_recorder(tmp_path)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("t1", [IRUserTextBlock(text="run command", origin="human")])
    recorder.write_block(call)
    recorder.write_block(result)
    recorder.write_tool_result_ttl(
        call_id="toolu_rehydrate",
        replace_content="[Bash: collapsed]",
        ttl=2,
        trigger="end_turn",
    )
    recorder.end_turn()
    recorder.start_turn("t2", [IRUserTextBlock(text="next turn", origin="human")])
    recorder.end_turn()

    rehydrated = Rehydrator(transcript).run()
    homunculus = _homunculus_with_transcript(
        tmp_path,
        transcript,
        initialize=False,
    )
    await homunculus.rehydrate(rehydrated)
    rendered = await homunculus.render_context([])

    assert rehydrated.last_completed_turn == 2
    assert len(rehydrated.tool_result_ttls) == 1
    assert isinstance(rendered[1], IRToolCallBlock)
    assert rendered[1].call_id == call.call_id
    assert rendered[1].tool == call.tool
    assert isinstance(rendered[2], IRToolResultBlock)
    assert rendered[2].content == [IRToolTextBlock(text="[Bash: collapsed]")]


async def test_loop_exit_harvests_ready_nursery_jobs_without_blocking() -> None:
    class _FakeHomunculus:
        def __init__(self) -> None:
            self.checks = 0

        async def check_nursery(self) -> None:
            self.checks += 1

        async def maybe_rerender(self) -> list[IRBlock] | None:
            return None

        def tick_end_turn_ttls(self) -> None:
            pass

    fake = _FakeHomunculus()
    ctx = RoundContext(
        blocks=[],
        round_number=1,
        cancel_token=CancelToken(),
        blocks_this_round=[],
    )

    await HomunculusRoundLifecycle(cast(Homunculus, fake)).on_loop_exit(ctx, "end_turn")

    assert fake.checks == 1
