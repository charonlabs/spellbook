from __future__ import annotations

from dataclasses import dataclass

import pytest

from spellbook.dreaming.frontier import (
    DreamDurationCalibration,
    DreamTreeShape,
    DurationEstimate,
    FrontierPolicy,
    build_morning_manifest,
    derive_frontier,
    forecast_deep_sleep,
    forecast_sleep,
    plan_frontier_advance,
)
from spellbook.ir_types import (
    IRSemanticBlock,
    IRSemanticBlockPairNarrative,
    IRSemanticBlockPairNarrativeChild,
    IRSemanticBlockPin,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRTokenRangeCount,
    IRUserTextBlock,
    SemanticBlockMode,
)


def _count(tokens: int, *, exact: bool = True) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=exact)


def _summary(tokens: int = 10) -> IRSemanticBlockSummary:
    return IRSemanticBlockSummary(
        headline="A summary",
        text="What mattered.",
        facets=[],
        open_thread=None,
        toks=_count(tokens),
    )


def _block(
    idx: int,
    *,
    title: str | None = None,
    mode: SemanticBlockMode = "full",
    full_tokens: int = 100,
    summary_tokens: int | None = None,
    pinned: bool = False,
) -> IRSemanticBlock:
    summary = _summary(summary_tokens) if summary_tokens is not None else None
    current_tokens = (
        summary.toks if mode == "summary" and summary else _count(full_tokens)
    )
    return IRSemanticBlock(
        id=f"block_{idx}",
        idx=idx,
        title=title or f"Block {idx}",
        range=IRSemanticBlockRange(
            id=f"range_{idx}",
            title=title or f"Block {idx}",
            start_block=idx,
            end_block=idx,
            completed=True,
        ),
        mode=mode,
        toks=current_tokens,
        full_toks=_count(full_tokens),
        available_modes=["full", "summary"] if summary else ["full"],
        artifacts=[summary] if summary else [],
        pin=(
            IRSemanticBlockPin(kind="block", reason="Keep the exact exchange.")
            if pinned
            else None
        ),
    )


def _with_pair_narrative(
    blocks: list[IRSemanticBlock],
    first: int,
    *,
    active: bool,
    narrative_tokens: int = 30,
    include_child: bool = True,
) -> list[IRSemanticBlock]:
    second = first + 1
    chapter = first // 2 + 1
    narrative_id = f"narrative_{chapter}"
    parent = IRSemanticBlockPairNarrative(
        narrative_id=narrative_id,
        pair=(first, second),
        chapter_number=chapter,
        title=f"Chapter {chapter}",
        blocks=[IRUserTextBlock(text="Woven memory.", origin="memory")],
        toks=_count(narrative_tokens),
        source_chapter_path=f"dreams/chapter-{chapter:02d}.md",
        compiled_json_path=f"dreams/chapter-{chapter:02d}.compiled.json",
    )
    parent_block = blocks[first]
    blocks[first] = parent_block.model_copy(
        update={
            "mode": "pair_narrative" if active else parent_block.mode,
            "toks": parent.toks if active else parent_block.toks,
            "available_modes": [*parent_block.available_modes, "pair_narrative"],
            "artifacts": [*parent_block.artifacts, parent],
        }
    )
    if include_child:
        child = IRSemanticBlockPairNarrativeChild(
            narrative_id=narrative_id,
            parent_block_idx=first,
            parent_block_id=blocks[first].id,
            pair=(first, second),
            chapter_number=chapter,
        )
        child_block = blocks[second]
        blocks[second] = child_block.model_copy(
            update={
                "mode": "pair_narrative" if active else child_block.mode,
                "toks": child.toks if active else child_block.toks,
                "available_modes": [*child_block.available_modes, "pair_narrative"],
                "artifacts": [*child_block.artifacts, child],
            }
        )
    return blocks


@dataclass
class _World:
    semantic_blocks: list[IRSemanticBlock]


def test_derive_frontier_from_mixed_rehydrated_world() -> None:
    blocks = [_block(idx) for idx in range(5)]
    blocks = _with_pair_narrative(blocks, 0, active=True)
    blocks[2] = _block(2, mode="summary", summary_tokens=9)
    blocks[3] = _block(3, summary_tokens=11, pinned=True)

    frontier = derive_frontier(_World(blocks))

    assert frontier.full_block_indices == (3, 4)
    assert frontier.summary_block_indices == (2,)
    assert frontier.active_narrative_chapters == (1,)
    assert frontier.block(0).narrative_id == "narrative_1"
    assert frontier.block(0).narrative_tokens == _count(30)
    child_narrative_tokens = frontier.block(1).narrative_tokens
    assert child_narrative_tokens is not None
    assert child_narrative_tokens.tokens == 0
    assert child_narrative_tokens.exact is True
    assert frontier.block(2).has_summary is True
    assert frontier.block(3).pinned is True
    assert frontier.block(4).has_summary is False


def test_derive_frontier_rejects_half_active_narrative() -> None:
    blocks = _with_pair_narrative([_block(0), _block(1)], 0, active=False)
    blocks[0] = blocks[0].model_copy(update={"mode": "pair_narrative"})

    with pytest.raises(ValueError, match="only active on one block"):
        derive_frontier(blocks)


def test_advance_prefers_narratives_then_summaries_and_is_minimal() -> None:
    blocks = [
        _block(0, full_tokens=100),
        _block(1, full_tokens=120),
        _block(2, full_tokens=80, summary_tokens=10),
        _block(3, summary_tokens=12, pinned=True),
        _block(4, summary_tokens=8),
        _block(5, summary_tokens=7),
    ]
    blocks = _with_pair_narrative(blocks, 0, active=False, narrative_tokens=30)
    original = list(blocks)

    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=2))

    assert plan.refused is False
    assert [(delta.block_idx, delta.to_mode) for delta in plan.transitions] == [
        (0, "pair_narrative"),
        (1, "pair_narrative"),
        (2, "summary"),
    ]
    assert plan.tokens_freed == 260
    assert plan.token_delta_exact is True
    assert [narrative.chapter_number for narrative in plan.narratives_applied] == [1]
    assert any(reason.code == "pinned" for reason in plan.reasons)
    assert blocks == original
    assert all(block.mode == "full" for block in blocks)


def test_advance_refuses_entire_plan_when_an_eligible_summary_is_missing() -> None:
    blocks = [
        _block(0, summary_tokens=10),
        _block(1),
        _block(2, summary_tokens=12),
    ]

    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=1))

    assert plan.refused is True
    assert plan.transitions == ()
    assert plan.narratives_applied == ()
    assert [reason.code for reason in plan.reasons] == ["missing_summary"]
    assert plan.reasons[0].block_indices == (1,)
    assert "whole frontier advance was refused" in plan.reasons[0].message


def test_advance_never_moves_a_pin_and_defers_its_pair_narrative() -> None:
    blocks = [
        _block(0, summary_tokens=10),
        _block(1, pinned=True),
    ]
    blocks = _with_pair_narrative(blocks, 0, active=False)

    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=0))

    assert [(delta.block_idx, delta.to_mode) for delta in plan.transitions] == [
        (0, "summary")
    ]
    assert all(delta.block_idx != 1 for delta in plan.transitions)
    assert {reason.code for reason in plan.reasons} == {
        "pinned",
        "narrative_deferred",
    }


def test_advance_heals_recent_and_pinned_blocks_back_to_full() -> None:
    blocks = [
        _block(0, mode="summary", summary_tokens=10, pinned=True),
        _block(1, mode="summary", summary_tokens=12),
    ]

    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=1))

    assert [(delta.block_idx, delta.to_mode) for delta in plan.transitions] == [
        (0, "full"),
        (1, "full"),
    ]
    assert plan.tokens_freed == -178


def test_incomplete_narrative_falls_back_to_summary_and_names_the_debt() -> None:
    blocks = [_block(0, summary_tokens=10), _block(1, summary_tokens=10)]
    blocks = _with_pair_narrative(blocks, 0, active=False, include_child=False)

    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=0))

    assert [(delta.block_idx, delta.to_mode) for delta in plan.transitions] == [
        (0, "summary"),
        (1, "summary"),
    ]
    assert any(reason.code == "narrative_incomplete" for reason in plan.reasons)


def test_morning_manifest_renders_deltas_debts_and_dream_pointer() -> None:
    blocks = [
        _block(0, full_tokens=100),
        _block(1, full_tokens=120),
        _block(2, summary_tokens=10, pinned=True),
    ]
    blocks = _with_pair_narrative(blocks, 0, active=False, narrative_tokens=30)
    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=0))

    manifest = build_morning_manifest(
        plan,
        sleep_kind="forced_sleep",
        deferred_deep_sleep_chapters=2,
        dream_transcript_paths=("forks/quantum_dream_1/transcript.jsonl",),
    )
    rendered = manifest.render()

    assert len(manifest.deltas) == 2
    assert {debt.code for debt in manifest.debts} == {
        "pinned",
        "deep_sleep_deferred",
        "forced_sleep_minimalism",
    }
    assert "Deltas\n" in rendered
    assert "Debts\n" in rendered
    assert "2 blocks moved; 190 tokens freed; 1 narrative applied" in rendered
    assert "Deep Sleep still owes 2 chapters" in rendered
    assert "no new memory may be authored" in rendered
    assert "The dream itself is kept, if you ever want to hold it" in rendered
    assert "forks/quantum_dream_1/transcript.jsonl" in rendered


def test_manifest_discloses_unknown_token_measurement_and_uses_default_pointer() -> (
    None
):
    blocks = [_block(0), _block(1)]
    blocks[0] = blocks[0].model_copy(update={"toks": None, "full_toks": None})
    blocks = _with_pair_narrative(blocks, 0, active=False)
    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=0))

    manifest = build_morning_manifest(plan)

    assert manifest.tokens_freed is None
    assert any(debt.code == "token_measurement_missing" for debt in manifest.debts)
    assert manifest.dream_transcript_paths == ("forks/",)
    assert "known portion" in manifest.render()


def test_duration_forecasts_are_structured_bounded_and_consent_renderable() -> None:
    sleep = forecast_sleep()
    deep = forecast_deep_sleep(
        DreamTreeShape(pending_chapter_writes=1, pending_merges=2)
    )

    assert sleep.estimate.upper_seconds == 2
    assert sleep.estimate.contains(0.5)
    assert "Sleep: ~seconds" in sleep.render()
    assert deep.estimate == DurationEstimate(28 * 60, 40 * 60, 64 * 60)
    assert deep.estimate.contains(40 * 60)
    assert "Deep Sleep tonight: ~40min" in deep.render()
    assert "range 28min-1h 4min" in deep.render()
    assert "1 chapter + 2 merges" in deep.render()
    assert deep.confidence == "low"


def test_duration_forecaster_accepts_measured_calibration() -> None:
    calibration = DreamDurationCalibration(
        chapter_write=DurationEstimate(100, 120, 150),
        merge=DurationEstimate(40, 50, 70),
        deep_sleep_overhead=DurationEstimate(5, 10, 20),
        basis="Observed batch timings.",
        confidence="high",
    )

    forecast = forecast_deep_sleep(
        DreamTreeShape(pending_chapter_writes=2, pending_merges=3),
        calibration,
    )

    assert forecast.estimate == DurationEstimate(325, 400, 530)
    assert forecast.estimate.contains(470)
    assert forecast.basis == "Observed batch timings."
    assert forecast.confidence == "high"
