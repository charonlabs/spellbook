"""Generate context plans for the homunculus to consume."""

from spellbook.config import HomunculusConfig
from spellbook.ir_types import (
    IRCompactBlockIntent,
    IRContextPlan,
    IRPlannerResult,
    IRSemanticBlock,
)
from spellbook.rehydrator import RehydrationResult


class Planner:
    def __init__(self, *, config: HomunculusConfig):
        self._config = config
        self._proposal: IRContextPlan | None = None

    def rehydrate(self, rehydrated: RehydrationResult) -> None:
        self._proposal = rehydrated.plan_proposal

    @property
    def proposal(self) -> IRContextPlan | None:
        return self._proposal

    def _compact_oldest_ready_block(
        self, blocks: list[IRSemanticBlock]
    ) -> IRCompactBlockIntent | None:
        for block in blocks:
            if (
                block.mode == "full"
                and "summary" in block.available_modes
                and block.pin is None
            ):
                return IRCompactBlockIntent(block_idx=block.idx)

    def _propose_plan(
        self, semantic_blocks: list[IRSemanticBlock]
    ) -> IRContextPlan | None:
        compact_intent = self._compact_oldest_ready_block(semantic_blocks)
        if compact_intent is not None:
            return IRContextPlan(intents=[compact_intent])

    def invalidate(self) -> None:
        self._proposal = None

    def plan(
        self, semantic_blocks: list[IRSemanticBlock], input_tokens: int
    ) -> IRPlannerResult | None:
        """Currently, on medium threshold, just compacts the oldest non-pinned
        summary-ready semantic block. Returns None on no changes."""
        if input_tokens < self._config.soft_threshold:
            return
        new_proposal = False
        if self._proposal is None:
            self._proposal = self._propose_plan(semantic_blocks)
            new_proposal = True
        if self._proposal is None:
            return
        if input_tokens < self._config.medium_threshold:
            if new_proposal:
                return IRPlannerResult(kind="proposal", plan=self._proposal)
            return None
        return IRPlannerResult(kind="action", plan=self._proposal)
