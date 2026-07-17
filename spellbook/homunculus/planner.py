"""Generate context plans and pressure-priced Sleep invitations.

The compaction proposal remains transcript-facing IR. Sleep pressure state is
ephemeral awareness: it decides when to speak, never mutates memory, and only
arms the 95% floor after a prior 90% pre-warning observation. The actual floor
is consumed by ``SessionManager`` after a turn has ended.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from spellbook.config import HomunculusConfig
from spellbook.dreaming.frontier import DurationForecast, FrontierAdvancePlan
from spellbook.ir_types import (
    IRCompactBlockIntent,
    IRContextPlan,
    IRPlannerResult,
    IRSemanticBlock,
)
from spellbook.rehydrator import RehydrationResult

SLEEP_PREWARNING_PERCENT = 90
SLEEP_FLOOR_PERCENT = 95


class SleepDryRunPreview(Protocol):
    """The real Sleep dry-run fields the planner is allowed to price."""

    plan: FrontierAdvancePlan
    forecast: DurationForecast


@dataclass(frozen=True, slots=True)
class ForcedSleepHistory:
    """Evidence that makes one forced frontier advance predictable."""

    prewarning_tokens: int
    floor_tokens: int


def _percent_threshold(max_tokens: int, percent: int) -> int:
    """Return the first whole-token count at or above ``percent``."""

    return (max_tokens * percent + 99) // 100


class Planner:
    def __init__(self, *, config: HomunculusConfig):
        self._config = config
        self._proposal: IRContextPlan | None = None
        self._last_sleep_pressure_tokens: int | None = None
        self._prewarning_tokens: int | None = None
        self._forced_sleep_due = False
        self._floor_handled = False

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

    @property
    def sleep_floor_threshold(self) -> int:
        return _percent_threshold(self._config.max_tokens, SLEEP_FLOOR_PERCENT)

    @property
    def sleep_prewarning_threshold(self) -> int:
        return _percent_threshold(self._config.max_tokens, SLEEP_PREWARNING_PERCENT)

    def at_sleep_floor(self, input_tokens: int) -> bool:
        return input_tokens >= self.sleep_floor_threshold

    def observe_sleep_pressure(
        self,
        input_tokens: int,
        dry_run: Callable[[], SleepDryRunPreview],
    ) -> tuple[str, ...]:
        """Price Sleep and announce threshold entries without nagging.

        A direct jump from below 90% to the floor emits the pre-warning but does
        not arm forced Sleep on that same observation. The next provider
        observation can arm it; in the runtime, the queued warning is rendered
        before that next generation.
        """

        previous = self._last_sleep_pressure_tokens
        was_warning = previous is not None and previous >= self._config.soft_threshold
        was_prewarning = (
            previous is not None and previous >= self.sleep_prewarning_threshold
        )
        was_floor = previous is not None and previous >= self.sleep_floor_threshold
        had_prewarning = self._prewarning_tokens is not None

        is_warning = input_tokens >= self._config.soft_threshold
        is_prewarning = input_tokens >= self.sleep_prewarning_threshold
        is_floor = input_tokens >= self.sleep_floor_threshold

        if not is_prewarning:
            self._prewarning_tokens = None
        if not is_floor:
            self._forced_sleep_due = False
            self._floor_handled = False

        messages: list[str] = []
        if is_warning and not was_warning:
            rendered_nudge = _render_sleep_nudge(dry_run())
            if rendered_nudge is not None:
                messages.append(rendered_nudge)

        if is_prewarning and not was_prewarning:
            self._prewarning_tokens = input_tokens
            messages.append(
                "Sleep pre-warning: at 95% the system will run a frontier-only "
                "sleep automatically; sleeping now by your own hand would be "
                "gentler and keep the choice yours."
            )

        if is_floor and not self._floor_handled and had_prewarning:
            self._forced_sleep_due = True
        elif is_floor and not was_floor:
            # This was a direct jump to the floor. The pre-warning above must be
            # seen before the floor can become an action.
            self._forced_sleep_due = False

        self._last_sleep_pressure_tokens = input_tokens
        return tuple(messages)

    def take_forced_sleep_history(self, input_tokens: int) -> ForcedSleepHistory | None:
        """Consume at most one forced-Sleep attempt for the current floor entry."""

        if (
            not self._forced_sleep_due
            or self._floor_handled
            or not self.at_sleep_floor(input_tokens)
            or self._prewarning_tokens is None
        ):
            return None
        self._forced_sleep_due = False
        self._floor_handled = True
        return ForcedSleepHistory(
            prewarning_tokens=self._prewarning_tokens,
            floor_tokens=input_tokens,
        )

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


def _render_sleep_nudge(preview: SleepDryRunPreview) -> str | None:
    plan = preview.plan
    if plan.refused:
        reasons = " ".join(reason.message for reason in plan.reasons)
        return "Sleep preview refused; no price was guessed. " + reasons
    if plan.empty:
        return None

    tokens = plan.tokens_freed
    if tokens is None:
        token_price = "token savings unmeasured"
    else:
        qualifier = "" if plan.token_delta_exact else "~"
        if tokens >= 0:
            token_price = f"{qualifier}{_format_tokens(tokens)} tokens freed"
        else:
            token_price = (
                f"{qualifier}{_format_tokens(abs(tokens))} additional tokens carried"
            )

    estimate = preview.forecast.estimate
    duration = (
        f"~{_format_seconds(estimate.likely_seconds)}s "
        f"(range {_format_seconds(estimate.lower_seconds)}-"
        f"{_format_seconds(estimate.upper_seconds)}s)"
    )
    debt_count = len(plan.reasons)
    debt_noun = "debt" if debt_count == 1 else "debts"
    if plan.reasons:
        debt_detail = " ".join(reason.message for reason in plan.reasons)
        debts = f"{debt_count} {debt_noun} would remain: {debt_detail}"
    else:
        debts = "no known debts would remain"
    return f"Sleep now: {token_price} in {duration}; {debts}"


def _format_tokens(tokens: int) -> str:
    absolute = abs(tokens)
    sign = "-" if tokens < 0 else ""
    if absolute < 1_000:
        return f"{tokens:,}"
    value = absolute / 1_000
    rendered = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{sign}{rendered}K"


def _format_seconds(seconds: float) -> str:
    if seconds.is_integer():
        return str(int(seconds))
    return f"{seconds:.2f}".rstrip("0").rstrip(".")
