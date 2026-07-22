from __future__ import annotations

from dataclasses import dataclass

import pytest

from spellbook.config import DEFAULT_SOFT_THRESHOLD
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
    narrative_tokens: int | None = 30,
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
        toks=_count(narrative_tokens) if narrative_tokens is not None else None,
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


def test_default_policy_keeps_largest_tiny_block_window_that_reaches_calm() -> None:
    blocks = [_block(idx, full_tokens=10, summary_tokens=1) for idx in range(20)]

    plan = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=150),
        current_render_tokens=200,
    )

    assert plan.policy.mode == "calm_targeted"
    assert [delta.block_idx for delta in plan.transitions] == list(range(6))
    assert plan.projection.kept_full_blocks == 14
    assert plan.projection.projected_render_tokens == 146
    assert plan.projection.estimate_quality == "exact"
    assert plan.projection.outcome == "calm_reached"
    assert (
        "kept 14 recent blocks full; calm reached without touching them"
        in plan.projection.render()
    )
    manifest = build_morning_manifest(plan)
    assert "Policy\n- Target: calm below 150 tokens" in manifest.render()
    assert "projected result 146 tokens" in manifest.render()


def test_default_policy_stops_at_four_block_floor_for_large_blocks() -> None:
    blocks = [_block(idx, full_tokens=200, summary_tokens=10) for idx in range(8)]

    plan = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=500),
        current_render_tokens=1_600,
    )

    assert [delta.block_idx for delta in plan.transitions] == [0, 1, 2, 3]
    assert plan.projection.kept_full_blocks == 4
    assert plan.projection.projected_render_tokens == 840
    assert plan.projection.calm_reached is False
    assert plan.projection.outcome == "floor_reached"


def test_default_policy_does_nothing_when_render_is_already_calm() -> None:
    blocks = [_block(idx, full_tokens=10, summary_tokens=1) for idx in range(12)]

    plan = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=150),
        current_render_tokens=149,
    )

    assert plan.empty is True
    assert plan.refused is False
    assert plan.reasons == ()
    assert plan.projection.kept_full_blocks == 12
    assert plan.projection.outcome == "already_calm"


def test_unknown_summary_size_counts_as_zero_relief_and_marks_projection() -> None:
    blocks = [_block(idx, full_tokens=100, summary_tokens=10) for idx in range(10)]
    unknown_summary = blocks[0].artifacts[0].model_copy(update={"toks": None})
    blocks[0] = blocks[0].model_copy(update={"artifacts": [unknown_summary]})

    plan = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=850),
        current_render_tokens=1_000,
    )

    assert [delta.block_idx for delta in plan.transitions] == [0, 1, 2]
    assert plan.projection.kept_full_blocks == 7
    assert plan.projection.estimated_tokens_freed == 180
    assert plan.projection.projected_render_tokens == 820
    assert plan.projection.estimate_quality == "conservative"
    assert plan.tokens_freed is None


def test_rendered_current_sizes_replace_raw_metrics_and_unknowns_claim_no_relief() -> (
    None
):
    blocks = [_block(idx, full_tokens=1_000, summary_tokens=100) for idx in range(6)]
    rendered_current_tokens = {
        block.id: (_count(150) if block.idx == 1 else block.toks) for block in blocks
    }
    rendered_current_tokens[blocks[0].id] = None

    plan = plan_frontier_advance(
        blocks,
        FrontierPolicy(recent_full_blocks=4),
        current_render_tokens=4_300,
        rendered_current_tokens=rendered_current_tokens,
    )

    assert [delta.block_idx for delta in plan.transitions] == [0, 1]
    assert plan.transitions[0].before_tokens is None
    assert plan.transitions[1].before_tokens == _count(150)
    assert plan.projection.estimated_tokens_freed == 50
    assert plan.projection.estimate_quality == "conservative"


def test_summary_pair_deepening_uses_rendered_pair_delta_and_moves_atomically() -> None:
    blocks = [
        _block(0, mode="summary", summary_tokens=10),
        _block(1, mode="summary", summary_tokens=10),
        *[_block(idx, full_tokens=100, summary_tokens=10) for idx in range(2, 6)],
    ]
    blocks = _with_pair_narrative(
        blocks,
        0,
        active=False,
        narrative_tokens=30,
    )
    rendered_current_tokens = {
        block.id: (_count(50) if block.idx < 2 else block.toks) for block in blocks
    }
    rendered_narrative_tokens = {
        blocks[0].id: _count(60),
        blocks[1].id: _count(0),
    }

    plan = plan_frontier_advance(
        blocks,
        FrontierPolicy(recent_full_blocks=4),
        current_render_tokens=500,
        rendered_current_tokens=rendered_current_tokens,
        rendered_narrative_tokens=rendered_narrative_tokens,
    )

    assert [
        (delta.block_idx, delta.from_mode, delta.to_mode) for delta in plan.transitions
    ] == [
        (0, "summary", "pair_narrative"),
        (1, "summary", "pair_narrative"),
    ]
    assert len(plan.deepening_candidates) == 1
    candidate = plan.deepening_candidates[0]
    assert candidate.classification == "relief"
    assert candidate.token_delta == 40
    assert candidate.estimated_tokens_freed == 40
    assert plan.selected_deepenings == (candidate,)
    assert "1 chapter rendered into place: Ch1 — 40 tokens freed" in (
        build_morning_manifest(plan).render()
    )


def test_nonrelieving_summary_pair_stays_as_enrichment_candidate() -> None:
    blocks = [
        _block(0, mode="summary", summary_tokens=10),
        _block(1, mode="summary", summary_tokens=10),
        *[_block(idx, summary_tokens=10) for idx in range(2, 6)],
    ]
    blocks = _with_pair_narrative(
        blocks,
        0,
        active=False,
        narrative_tokens=30,
    )

    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=4))

    assert plan.transitions == ()
    assert plan.selected_deepenings == ()
    assert len(plan.deepening_candidates) == 1
    candidate = plan.deepening_candidates[0]
    assert candidate.classification == "enrichment"
    assert candidate.token_delta == -10
    assert candidate.estimated_tokens_freed == 0


@pytest.mark.parametrize("approximate_side", ["summary", "narrative"])
def test_uncountable_summary_pair_claims_zero_deepening_relief(
    approximate_side: str,
) -> None:
    blocks = [
        _block(0, mode="summary", summary_tokens=40),
        _block(1, mode="summary", summary_tokens=40),
        *[_block(idx, summary_tokens=10) for idx in range(2, 6)],
    ]
    blocks = _with_pair_narrative(
        blocks,
        0,
        active=False,
        narrative_tokens=50,
    )
    if approximate_side == "summary":
        blocks[0] = blocks[0].model_copy(update={"toks": _count(40, exact=False)})
    else:
        blocks[0] = blocks[0].model_copy(
            update={
                "artifacts": [
                    artifact.model_copy(update={"toks": _count(50, exact=False)})
                    if isinstance(artifact, IRSemanticBlockPairNarrative)
                    else artifact
                    for artifact in blocks[0].artifacts
                ]
            }
        )

    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=4))

    assert plan.transitions == ()
    candidate = plan.deepening_candidates[0]
    assert candidate.classification == "enrichment"
    assert candidate.token_delta is None
    assert candidate.token_delta_exact is False
    assert candidate.estimated_tokens_freed == 0


@pytest.mark.parametrize("excluded_by", ["pin", "kept_window"])
def test_pin_or_kept_window_excludes_summary_pair_deepening(
    excluded_by: str,
) -> None:
    blocks = [
        _block(
            0,
            mode="summary",
            summary_tokens=40,
            pinned=excluded_by == "pin",
        ),
        _block(1, mode="summary", summary_tokens=40),
        *[_block(idx, summary_tokens=10) for idx in range(2, 6)],
    ]
    blocks = _with_pair_narrative(
        blocks,
        0,
        active=False,
        narrative_tokens=50,
    )
    kept = 6 if excluded_by == "kept_window" else 4

    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=kept))

    assert plan.deepening_candidates == ()
    assert all(delta.to_mode != "pair_narrative" for delta in plan.transitions)


def test_standing_narrative_inside_kept_window_remains_untouched() -> None:
    blocks = [_block(idx, summary_tokens=10) for idx in range(6)]
    blocks = _with_pair_narrative(
        blocks,
        4,
        active=True,
        narrative_tokens=30,
    )

    plan = plan_frontier_advance(blocks, FrontierPolicy(recent_full_blocks=4))

    assert [delta.block_idx for delta in plan.transitions] == [0, 1]
    assert all(delta.block_idx not in {4, 5} for delta in plan.transitions)
    assert plan.narratives_applied == ()
    assert all(reason.chapter_number != 3 for reason in plan.reasons)


def test_calm_search_orders_summary_deepening_before_newer_full_advance() -> None:
    blocks = [
        _block(0, mode="summary", summary_tokens=40),
        _block(1, mode="summary", summary_tokens=40),
        *[_block(idx, full_tokens=100, summary_tokens=10) for idx in range(2, 8)],
    ]
    blocks = _with_pair_narrative(
        blocks,
        0,
        active=False,
        narrative_tokens=50,
    )

    plan = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=600),
        current_render_tokens=680,
    )

    assert [
        (delta.block_idx, delta.from_mode, delta.to_mode) for delta in plan.transitions
    ] == [
        (0, "summary", "pair_narrative"),
        (1, "summary", "pair_narrative"),
        (2, "full", "summary"),
    ]
    assert plan.projection.kept_full_blocks == 5
    assert plan.projection.projected_render_tokens == 560
    assert [item.narrative.chapter_number for item in plan.selected_deepenings] == [1]


def test_unknown_atomic_narrative_counts_neither_halfs_apparent_relief() -> None:
    blocks = [_block(idx, full_tokens=100, summary_tokens=10) for idx in range(6)]
    blocks[0] = _block(0, full_tokens=100)
    blocks[1] = _block(1, full_tokens=100)
    blocks = _with_pair_narrative(
        blocks,
        0,
        active=False,
        narrative_tokens=None,
    )

    plan = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=550),
        current_render_tokens=600,
    )

    assert [(delta.block_idx, delta.to_mode) for delta in plan.transitions] == [
        (0, "pair_narrative"),
        (1, "pair_narrative"),
    ]
    assert plan.projection.estimated_tokens_freed == 0
    assert plan.projection.projected_render_tokens == 600
    assert plan.projection.estimate_quality == "conservative"


def test_policy_uses_configured_soft_threshold_constant_and_explicit_fixed_mode() -> (
    None
):
    default_policy = FrontierPolicy()
    fixed_policy = FrontierPolicy(recent_full_blocks=2)

    assert default_policy.calm_target_tokens == DEFAULT_SOFT_THRESHOLD
    assert default_policy.mode == "calm_targeted"
    assert fixed_policy.mode == "fixed"


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


def test_morning_manifest_names_material_gauge_gap_beside_estimate() -> None:
    blocks = [
        _block(0, full_tokens=100_000),
        _block(1, full_tokens=120_000),
    ]
    blocks = _with_pair_narrative(
        blocks,
        0,
        active=False,
        narrative_tokens=30_000,
    )
    plan = plan_frontier_advance(
        blocks,
        FrontierPolicy(recent_full_blocks=0),
        current_render_tokens=220_000,
    )

    manifest = build_morning_manifest(
        plan,
        gauge_tokens_before=220_000,
        gauge_tokens_after=121_000,
    )

    assert manifest.tokens_freed == 190_000
    assert manifest.gauge_tokens_freed == 99_000
    assert manifest.gauge_projection_gap == 91_000
    assert (
        "estimated 190,000 tokens freed; gauge shows 99,000 tokens freed; "
        "gap: 91,000 fewer than estimated"
    ) in manifest.render()


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


def test_min_kept_floor_raises_kept_blocks_above_calm_requirement() -> None:
    """Night-002 guard: min_kept widens the kept-full window even when calm
    would be reached with fewer blocks; values below 4 raise to the hard
    floor; fixed-window policy ignores min_kept."""
    blocks = [_block(idx) for idx in range(12)]

    calm_only = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=10_000_000),
        current_render_tokens=100,
    )
    floored = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=10_000_000, min_kept=8),
        current_render_tokens=100,
    )
    # A generous calm target is trivially reached; both plans should be
    # empty/already calm — min_kept must not FORCE advancing.
    assert calm_only.projection.outcome == "already_calm"
    assert floored.projection.outcome == "already_calm"

    tight = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=1, min_kept=8),
        current_render_tokens=10_000,
    )
    # With an unreachable calm target the plan advances to its floor —
    # which min_kept has raised from 4 to 8.
    assert tight.projection.kept_full_blocks == 8
    assert all(t.block_idx < 4 for t in tight.transitions)

    below_hard_floor = plan_frontier_advance(
        blocks,
        FrontierPolicy(calm_target_tokens=1, min_kept=2),
        current_render_tokens=10_000,
    )
    assert below_hard_floor.projection.kept_full_blocks == 4
