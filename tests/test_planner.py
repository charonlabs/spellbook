import pytest

from spellbook.config import HomunculusConfig
from spellbook.dreaming.frontier import (
    FrontierPolicy,
    forecast_sleep,
    plan_frontier_advance,
)
from spellbook.homunculus.planner import Planner
from spellbook.ir_types import (
    IRCompactBlockIntent,
    IRFooter,
    IRFooterQueueRecord,
    IRPlannerResult,
    IRSemanticBlock,
    IRSemanticBlockPin,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRTokenRangeCount,
    SemanticBlockMode,
)
from spellbook.rehydrator import RehydrationResult
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


async def _sleep_preview(*, blocks: list[IRSemanticBlock] | None = None) -> SleepDryRun:
    resolved = blocks or [
        _semantic_block(idx, full_tokens=100, summary_tokens=10) for idx in range(6)
    ]
    current_render_tokens = sum(
        block.toks.tokens for block in resolved if block.toks is not None
    )
    return SleepDryRun(
        plan=plan_frontier_advance(
            resolved,
            FrontierPolicy(calm_target_tokens=_sleep_config().soft_threshold),
            current_render_tokens=current_render_tokens,
        ),
        forecast=forecast_sleep(),
    )


def _planner_footer_record(text: str, *, turn: int = 1) -> IRFooterQueueRecord:
    return IRFooterQueueRecord(
        session_id="session_test",
        footer=IRFooter(
            text=f"Planner:\n{text}",
            type="compaction",
            source="planner",
            key=f"planner_{turn}",
        ),
        turn=turn,
        turn_id=f"turn_{turn}",
    )


def _gauge_footer_record(
    text: str,
    *,
    turn: int = 1,
) -> IRFooterQueueRecord:
    return IRFooterQueueRecord(
        session_id="session_test",
        footer=IRFooter(
            text=text,
            type="gas_gauge",
            source="telemetry",
            key="gas_gauge",
        ),
        turn=turn,
        turn_id=f"turn_{turn}",
    )


def _rehydrated_planner(
    config: HomunculusConfig,
    records: list[IRFooterQueueRecord],
) -> Planner:
    planner = Planner(config=config)
    planner.rehydrate(
        RehydrationResult.model_construct(
            plan_proposal=None,
            records=records,
        )
    )
    return planner


@pytest.mark.asyncio
async def test_sleep_nudge_uses_real_price_once_across_sustained_warning() -> None:
    planner = Planner(config=_sleep_config())
    calls = 0

    async def dry_run() -> SleepDryRun:
        nonlocal calls
        calls += 1
        return await _sleep_preview()

    observations = [500, 550, 600, 650, 700, 800]
    updates = [
        await planner.observe_sleep_pressure(input_tokens, dry_run)
        for input_tokens in observations
    ]

    assert calls == len(observations)
    assert len(updates[0]) == 1
    assert "Sleep now: 180 tokens freed" in updates[0][0]
    assert "~0.25s (range 0.05-2s)" in updates[0][0]
    assert "no known debts would remain" in updates[0][0]
    assert updates[1:] == [()] * (len(observations) - 1)


@pytest.mark.asyncio
async def test_sleep_nudge_reoffers_once_on_hard_escalation() -> None:
    planner = Planner(config=_sleep_config())

    warning = await planner.observe_sleep_pressure(500, _sleep_preview)
    await planner.observe_sleep_pressure(920, _sleep_preview)
    hard = await planner.observe_sleep_pressure(933, _sleep_preview)
    repeated = await planner.observe_sleep_pressure(940, _sleep_preview)

    assert sum("Sleep now:" in message for message in warning) == 1
    assert sum("Sleep now:" in message for message in hard) == 1
    assert repeated == ()


@pytest.mark.asyncio
async def test_sleep_nudge_requires_hysteretic_exit_before_reentry() -> None:
    planner = Planner(config=_sleep_config())

    entered = await planner.observe_sleep_pressure(500, _sleep_preview)
    fluttered_low = await planner.observe_sleep_pressure(499, _sleep_preview)
    fluttered_back = await planner.observe_sleep_pressure(500, _sleep_preview)
    exited = await planner.observe_sleep_pressure(479, _sleep_preview)
    reentered = await planner.observe_sleep_pressure(500, _sleep_preview)

    assert sum("Sleep now:" in message for message in entered) == 1
    assert fluttered_low == ()
    assert fluttered_back == ()
    assert exited == ()
    assert sum("Sleep now:" in message for message in reentered) == 1


@pytest.mark.asyncio
async def test_sleep_nudge_names_refusal_without_guessing_and_skips_empty_frontier() -> (
    None
):
    refused_blocks = [
        _semantic_block(idx, full_tokens=100, summary_tokens=10) for idx in range(6)
    ]
    refused_blocks[0] = _semantic_block(
        0,
        summary_ready=False,
        full_tokens=100,
    )

    async def refused_preview() -> SleepDryRun:
        return await _sleep_preview(blocks=refused_blocks)

    async def empty_preview() -> SleepDryRun:
        return await _sleep_preview(
            blocks=[
                _semantic_block(idx, full_tokens=100, summary_tokens=10)
                for idx in range(4)
            ]
        )

    refused = await Planner(config=_sleep_config()).observe_sleep_pressure(
        500,
        refused_preview,
    )
    empty = await Planner(config=_sleep_config()).observe_sleep_pressure(
        500,
        empty_preview,
    )

    assert len(refused) == 1
    assert "Sleep now" not in refused[0]
    assert "no price was guessed" in refused[0]
    assert "whole frontier advance was refused" in refused[0]
    assert empty == ()


@pytest.mark.asyncio
async def test_sleep_nudge_reoffers_only_above_twenty_five_percent_relief_delta() -> (
    None
):
    planner = Planner(config=_sleep_config())

    def blocks_with_relief(
        full_tokens: int, summary_tokens: int
    ) -> list[IRSemanticBlock]:
        return [
            _semantic_block(
                0,
                full_tokens=full_tokens,
                summary_tokens=summary_tokens,
            ),
            *[
                _semantic_block(idx, full_tokens=100, summary_tokens=10)
                for idx in range(1, 5)
            ],
        ]

    previews = [
        await _sleep_preview(blocks=blocks_with_relief(200, 20)),
        await _sleep_preview(blocks=blocks_with_relief(250, 25)),
        await _sleep_preview(blocks=blocks_with_relief(251, 25)),
    ]

    async def changing_preview() -> SleepDryRun:
        return previews.pop(0)

    entered = await planner.observe_sleep_pressure(500, changing_preview)
    exactly_twenty_five_percent = await planner.observe_sleep_pressure(
        500,
        changing_preview,
    )
    materially_changed = await planner.observe_sleep_pressure(
        500,
        changing_preview,
    )

    assert "Sleep now: 180 tokens freed" in entered[0]
    assert exactly_twenty_five_percent == ()
    assert "Sleep now: 226 tokens freed" in materially_changed[0]


@pytest.mark.asyncio
async def test_sleep_nudge_does_not_repeat_after_rehydration_mid_regime() -> None:
    first = Planner(config=_sleep_config())
    nudge = await first.observe_sleep_pressure(500, _sleep_preview)
    resumed = _rehydrated_planner(
        _sleep_config(),
        [
            _gauge_footer_record("[context: 0K / 1M - warning]"),
            _planner_footer_record(nudge[0]),
        ],
    )

    unchanged = await resumed.observe_sleep_pressure(700, _sleep_preview)
    escalated = await resumed.observe_sleep_pressure(933, _sleep_preview)

    assert unchanged == ()
    assert sum("Sleep now:" in message for message in escalated) == 1


@pytest.mark.asyncio
async def test_sleep_prewarning_is_exact_once_per_ninety_percent_entry() -> None:
    planner = Planner(config=_sleep_config())

    await planner.observe_sleep_pressure(800, _sleep_preview)
    entered = await planner.observe_sleep_pressure(900, _sleep_preview)
    repeated = await planner.observe_sleep_pressure(920, _sleep_preview)
    await planner.observe_sleep_pressure(899, _sleep_preview)
    fluttered_back = await planner.observe_sleep_pressure(900, _sleep_preview)
    await planner.observe_sleep_pressure(879, _sleep_preview)
    reentered = await planner.observe_sleep_pressure(900, _sleep_preview)

    expected = (
        "Sleep pre-warning: at 95% the system will run a frontier-only sleep "
        "automatically; sleeping now by your own hand would be gentler and keep "
        "the choice yours."
    )
    assert entered == (expected,)
    assert repeated == ()
    assert fluttered_back == ()
    assert reentered == (expected,)


@pytest.mark.asyncio
async def test_sleep_prewarning_does_not_repeat_after_rehydration_mid_regime() -> None:
    first = Planner(config=_sleep_config())
    nudge = await first.observe_sleep_pressure(800, _sleep_preview)
    prewarning = await first.observe_sleep_pressure(900, _sleep_preview)
    resumed = _rehydrated_planner(
        _sleep_config(),
        [
            _gauge_footer_record("[context: 0K / 1M - warning]"),
            _planner_footer_record(nudge[0]),
            _gauge_footer_record(
                "[context: 0K / 1M - forced]",
                turn=2,
            ),
            _planner_footer_record(prewarning[0], turn=2),
        ],
    )

    unchanged = await resumed.observe_sleep_pressure(920, _sleep_preview)
    assert resumed.take_forced_sleep_history(920) is None
    await resumed.observe_sleep_pressure(879, _sleep_preview)
    reentered = await resumed.observe_sleep_pressure(900, _sleep_preview)

    assert unchanged == ()
    assert sum("Sleep pre-warning" in message for message in reentered) == 1


@pytest.mark.asyncio
async def test_sleep_floor_requires_prior_prewarning_and_is_consumed_once() -> None:
    direct_jump = Planner(config=_sleep_config())

    warning = await direct_jump.observe_sleep_pressure(950, _sleep_preview)

    assert any("Sleep pre-warning" in message for message in warning)
    assert direct_jump.take_forced_sleep_history(950) is None

    await direct_jump.observe_sleep_pressure(950, _sleep_preview)
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
