from spellbook.config import HomunculusConfig
from spellbook.homunculus.planner import Planner
from spellbook.ir_types import (
    IRCompactBlockIntent,
    IRPlannerResult,
    IRSemanticBlock,
    IRSemanticBlockPin,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    SemanticBlockMode,
)


def _config() -> HomunculusConfig:
    return HomunculusConfig(soft_threshold=50, medium_threshold=100)


def _summary() -> IRSemanticBlockSummary:
    return IRSemanticBlockSummary(
        headline="Summary headline",
        text="Summary text.",
        facets=[],
        open_thread=None,
        toks=None,
    )


def _semantic_block(
    idx: int,
    *,
    mode: SemanticBlockMode = "full",
    summary_ready: bool = True,
    pinned: bool = False,
) -> IRSemanticBlock:
    summary = _summary()
    return IRSemanticBlock(
        idx=idx,
        title=f"Block {idx}",
        range=IRSemanticBlockRange(
            title=f"Block {idx}",
            start_block=idx,
            end_block=idx,
            completed=True,
        ),
        toks=None,
        full_toks=None,
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


def test_planner_does_nothing_when_no_summary_ready_unpinned_full_block() -> None:
    planner = Planner(config=_config())
    blocks = [
        _semantic_block(0, pinned=True),
        _semantic_block(1, mode="summary"),
        _semantic_block(2, summary_ready=False),
    ]

    plan = planner.plan(blocks, input_tokens=100)

    assert plan is None
