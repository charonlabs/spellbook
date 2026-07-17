from spellbook.config import HomunculusConfig
from spellbook.dreaming.frontier import forecast_sleep, plan_frontier_advance
from spellbook.homunculus.planner import Planner
from spellbook.ir_types import (
    IRCompactBlockIntent,
    IRPlannerResult,
    IRSemanticBlock,
    IRSemanticBlockPin,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRTokenRangeCount,
    SemanticBlockMode,
)
from spellbook.tools.sleep import SleepDryRun


def _config() -> HomunculusConfig:
    return HomunculusConfig(soft_threshold=50, medium_threshold=100)


def _summary(tokens: int | None = None) -> IRSemanticBlockSummary:
    return IRSemanticBlockSummary(
        headline="Summary headline",
        text="Summary text.",
        facets=[],
        open_thread=None,
        toks=_count(tokens) if tokens is not None else None,
    )


def _count(tokens: int) -> IRTokenRangeCount:
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _semantic_block(
    idx: int,
    *,
    mode: SemanticBlockMode = "full",
    summary_ready: bool = True,
    pinned: bool = False,
    full_tokens: int | None = None,
    summary_tokens: int | None = None,
) -> IRSemanticBlock:
    summary = _summary(summary_tokens)
    full_toks = _count(full_tokens) if full_tokens is not None else None
    return IRSemanticBlock(
        idx=idx,
        title=f"Block {idx}",
        range=IRSemanticBlockRange(
            title=f"Block {idx}",
            start_block=idx,
            end_block=idx,
            completed=True,
        ),
        toks=full_toks,
        full_toks=full_toks,
        mode=mode,
        available_modes=["full", "summary"] if summary_ready else ["full"],
        artifacts=[summary] if summary_ready else [],
        pin=(
            IRSemanticBlockPin(kind="block", reason="Keep it exact.")
            if pinned
            else None
        ),
    )


def test_planner_compacts_oldest_viable_block_using_real_block_idx() -> None:
    planner = Planner(config=_config())
    blocks = [
        _semantic_block(0, pinned=True),
        _semantic_block(1, mode="summary"),
        _semantic_block(2, summary_ready=False),
        _semantic_block(3),
    ]

    result = planner.plan(blocks, input_tokens=100)

    assert result is not None
    assert result.kind == "action"
    assert len(result.plan.intents) == 1
    intent = result.plan.intents[0]
    assert isinstance(intent, IRCompactBlockIntent)
    assert intent.block_idx == 3


def test_planner_proposes_once_before_medium_then_compacts() -> None:
    planner = Planner(config=_config())
    blocks = [_semantic_block(0)]

    proposal = planner.plan(blocks, input_tokens=50)
    repeated = planner.plan(blocks, input_tokens=99)
    action = planner.plan(blocks, input_tokens=100)

    assert isinstance(proposal, IRPlannerResult)
    assert proposal.kind == "proposal"
    assert repeated is None
    assert isinstance(action, IRPlannerResult)
    assert action.kind == "action"
    assert action.plan == proposal.plan


def test_planner_does_nothing_below_soft_threshold() -> None:
    planner = Planner(config=_config())

    plan = planner.plan([_semantic_block(0)], input_tokens=49)

    assert plan is None


def _sleep_config() -> HomunculusConfig:
    return HomunculusConfig(
        soft_threshold=500,
        medium_threshold=850,
        hard_threshold=933,
        max_tokens=1_000,
    )


def _sleep_preview(*, blocks: list[IRSemanticBlock] | None = None) -> SleepDryRun:
    resolved = blocks or [
        _semantic_block(idx, full_tokens=100, summary_tokens=10) for idx in range(6)
    ]
    return SleepDryRun(
        plan=plan_frontier_advance(resolved),
        forecast=forecast_sleep(),
    )


def test_sleep_nudge_uses_real_dry_run_price_once_per_warning_entry() -> None:
    planner = Planner(config=_sleep_config())
    calls = 0

    def dry_run() -> SleepDryRun:
        nonlocal calls
        calls += 1
        return _sleep_preview()

    entered = planner.observe_sleep_pressure(500, dry_run)
    repeated = planner.observe_sleep_pressure(700, dry_run)
    planner.observe_sleep_pressure(499, dry_run)
    reentered = planner.observe_sleep_pressure(500, dry_run)

    assert calls == 2
    assert len(entered) == 1
    assert "Sleep now: 180 tokens freed" in entered[0]
    assert "~0.25s (range 0.05-2s)" in entered[0]
    assert "no known debts would remain" in entered[0]
    assert repeated == ()
    assert len(reentered) == 1


def test_sleep_nudge_names_refusal_without_guessing_and_skips_empty_frontier() -> None:
    refused_blocks = [
        _semantic_block(idx, full_tokens=100, summary_tokens=10) for idx in range(6)
    ]
    refused_blocks[0] = _semantic_block(
        0,
        summary_ready=False,
        full_tokens=100,
    )
    refused = Planner(config=_sleep_config()).observe_sleep_pressure(
        500,
        lambda: _sleep_preview(blocks=refused_blocks),
    )
    empty = Planner(config=_sleep_config()).observe_sleep_pressure(
        500,
        lambda: _sleep_preview(
            blocks=[
                _semantic_block(idx, full_tokens=100, summary_tokens=10)
                for idx in range(4)
            ]
        ),
    )

    assert len(refused) == 1
    assert "Sleep now" not in refused[0]
    assert "no price was guessed" in refused[0]
    assert "whole frontier advance was refused" in refused[0]
    assert empty == ()


def test_sleep_prewarning_is_exact_once_per_ninety_percent_entry() -> None:
    planner = Planner(config=_sleep_config())

    planner.observe_sleep_pressure(800, _sleep_preview)
    entered = planner.observe_sleep_pressure(900, _sleep_preview)
    repeated = planner.observe_sleep_pressure(940, _sleep_preview)
    planner.observe_sleep_pressure(899, _sleep_preview)
    reentered = planner.observe_sleep_pressure(900, _sleep_preview)

    expected = (
        "Sleep pre-warning: at 95% the system will run a frontier-only sleep "
        "automatically; sleeping now by your own hand would be gentler and keep "
        "the choice yours."
    )
    assert entered == (expected,)
    assert repeated == ()
    assert reentered == (expected,)


def test_sleep_floor_requires_prior_prewarning_and_is_consumed_once() -> None:
    direct_jump = Planner(config=_sleep_config())

    warning = direct_jump.observe_sleep_pressure(950, _sleep_preview)

    assert any("Sleep pre-warning" in message for message in warning)
    assert direct_jump.take_forced_sleep_history(950) is None

    direct_jump.observe_sleep_pressure(950, _sleep_preview)
    history = direct_jump.take_forced_sleep_history(950)

    assert history is not None
    assert history.prewarning_tokens == 950
    assert history.floor_tokens == 950
    assert direct_jump.take_forced_sleep_history(950) is None


def test_planner_does_nothing_when_no_summary_ready_unpinned_full_block() -> None:
    planner = Planner(config=_config())
    blocks = [
        _semantic_block(0, pinned=True),
        _semantic_block(1, mode="summary"),
        _semantic_block(2, summary_ready=False),
    ]

    plan = planner.plan(blocks, input_tokens=100)

    assert plan is None
