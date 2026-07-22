"""Generate context plans and pressure-priced Sleep invitations.

The compaction proposal remains transcript-facing IR. Sleep pressure state is
awareness: it decides when to speak, never mutates memory, and only arms the
95% floor after a prior 90% pre-warning observation. Invitation and warning
cadence rehydrates from their explicit footer records; exact forced-Sleep
consent history does not. The actual floor is consumed by ``SessionManager``
after a turn has ended.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import re
from typing import Literal, Protocol

from spellbook.config import HomunculusConfig
from spellbook.dreaming.frontier import DurationForecast, FrontierAdvancePlan
from spellbook.ir_types import (
    IRCompactBlockIntent,
    IRContextPlan,
    IRFooterQueueRecord,
    IRPlannerResult,
    IRSemanticBlock,
)
from spellbook.rehydrator import RehydrationResult

SLEEP_PREWARNING_PERCENT = 90
SLEEP_FLOOR_PERCENT = 95
SLEEP_PRESSURE_HYSTERESIS_PERCENT = 2
SLEEP_RELIEF_REOFFER_PERCENT = 25

SleepNudgeRegime = Literal["calm", "warning", "hard"]

_SLEEP_PREWARNING_MESSAGE = (
    "Sleep pre-warning: at 95% the system will run a frontier-only sleep "
    "automatically; sleeping now by your own hand would be gentler and keep "
    "the choice yours."
)
_SLEEP_NUDGE_MARKERS = ("Sleep now:", "Sleep preview refused;")
_GAS_GAUGE_RE = re.compile(
    r"\[context: (?P<thousands>\d+)K / 1M - "
    r"(?P<regime>calm|warning|forced|critical|unknown)\]"
)


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
        self._sleep_nudge_regime: SleepNudgeRegime = "calm"
        self._has_sleep_relief_baseline = False
        self._last_sleep_relief_tokens: int | None = None
        self._prewarning_active = False
        self._prewarning_tokens: int | None = None
        self._floor_active = False
        self._forced_sleep_due = False
        self._floor_handled = False

    def rehydrate(self, rehydrated: RehydrationResult) -> None:
        self._proposal = rehydrated.plan_proposal
        self._rehydrate_sleep_cadence(rehydrated)

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

    @property
    def sleep_pressure_hysteresis(self) -> int:
        return _percent_threshold(
            self._config.max_tokens,
            SLEEP_PRESSURE_HYSTERESIS_PERCENT,
        )

    def at_sleep_floor(self, input_tokens: int) -> bool:
        return input_tokens >= self.sleep_floor_threshold

    async def observe_sleep_pressure(
        self,
        input_tokens: int,
        dry_run: Callable[[], Awaitable[SleepDryRunPreview]],
    ) -> tuple[str, ...]:
        """Price Sleep and announce threshold entries without nagging.

        A direct jump from below 90% to the floor emits the pre-warning but does
        not arm forced Sleep on that same observation. The next provider
        observation can arm it; in the runtime, the queued warning is rendered
        before that next generation.
        """

        previous_regime = self._sleep_nudge_regime
        current_regime = self._next_sleep_nudge_regime(input_tokens)
        self._sleep_nudge_regime = current_regime

        was_prewarning = self._prewarning_active
        is_prewarning = self._threshold_active(
            was_prewarning,
            input_tokens,
            self.sleep_prewarning_threshold,
        )
        self._prewarning_active = is_prewarning

        was_floor = self._floor_active
        is_floor = self._threshold_active(
            was_floor,
            input_tokens,
            self.sleep_floor_threshold,
        )
        self._floor_active = is_floor
        had_prewarning = self._prewarning_tokens is not None

        if not is_prewarning:
            self._prewarning_tokens = None
        if not is_floor:
            self._forced_sleep_due = False
            self._floor_handled = False

        messages: list[str] = []
        if current_regime != "calm":
            preview = await dry_run()
            relief_tokens = preview.plan.tokens_freed
            regime_entry = _regime_rank(current_regime) > _regime_rank(previous_regime)
            materially_changed = self._sleep_relief_materially_changed(relief_tokens)
            if regime_entry or materially_changed:
                rendered_nudge = _render_sleep_nudge(preview)
                if rendered_nudge is not None:
                    messages.append(rendered_nudge)
                self._remember_sleep_relief(relief_tokens)
            elif not self._has_sleep_relief_baseline:
                # A resumed planner knows that it already spoke in this regime,
                # but footer text is not an exact pricing record. Establish the
                # current exact baseline silently before considering deltas.
                self._remember_sleep_relief(relief_tokens)
        elif previous_regime != "calm":
            self._has_sleep_relief_baseline = False
            self._last_sleep_relief_tokens = None

        if is_prewarning and not was_prewarning:
            self._prewarning_tokens = input_tokens
            messages.append(_SLEEP_PREWARNING_MESSAGE)

        if (
            input_tokens >= self.sleep_floor_threshold
            and not self._floor_handled
            and had_prewarning
        ):
            self._forced_sleep_due = True
        elif is_floor and not was_floor:
            # This was a direct jump to the floor. The pre-warning above must be
            # seen before the floor can become an action.
            self._forced_sleep_due = False

        return tuple(messages)

    def _next_sleep_nudge_regime(self, input_tokens: int) -> SleepNudgeRegime:
        current = self._sleep_nudge_regime
        margin = self.sleep_pressure_hysteresis
        warning_exit = max(0, self._config.soft_threshold - margin)
        hard_exit = max(0, self._config.hard_threshold - margin)

        if current == "hard":
            if input_tokens >= hard_exit:
                return "hard"
            if input_tokens >= warning_exit:
                return "warning"
            return "calm"
        if current == "warning":
            if input_tokens >= self._config.hard_threshold:
                return "hard"
            if input_tokens >= warning_exit:
                return "warning"
            return "calm"
        if input_tokens >= self._config.hard_threshold:
            return "hard"
        if input_tokens >= self._config.soft_threshold:
            return "warning"
        return "calm"

    def _threshold_active(
        self,
        was_active: bool,
        input_tokens: int,
        threshold: int,
    ) -> bool:
        entry_threshold = (
            max(0, threshold - self.sleep_pressure_hysteresis)
            if was_active
            else threshold
        )
        return input_tokens >= entry_threshold

    def _sleep_relief_materially_changed(self, relief_tokens: int | None) -> bool:
        if not self._has_sleep_relief_baseline:
            return False
        previous = self._last_sleep_relief_tokens
        if previous is None or relief_tokens is None:
            return False
        if previous == 0:
            return relief_tokens != 0
        difference = abs(relief_tokens - previous)
        return difference * 100 > abs(previous) * SLEEP_RELIEF_REOFFER_PERCENT

    def _remember_sleep_relief(self, relief_tokens: int | None) -> None:
        self._has_sleep_relief_baseline = True
        self._last_sleep_relief_tokens = relief_tokens

    def _rehydrate_sleep_cadence(self, rehydrated: RehydrationResult) -> None:
        """Recover quieting state from canonical footer events.

        Planner footer text proves that an invitation or pre-warning was made.
        Gas-gauge records after those events provide coarse pressure observations
        for genuine exits. They do not prove the exact token count at which a
        pre-warning was delivered, so rehydration never synthesizes
        ``_prewarning_tokens`` or arms forced Sleep from them.
        """

        self._sleep_nudge_regime = "calm"
        self._has_sleep_relief_baseline = False
        self._last_sleep_relief_tokens = None
        self._prewarning_active = False
        self._prewarning_tokens = None
        self._floor_active = False
        self._forced_sleep_due = False
        self._floor_handled = False

        latest_gauge: tuple[int, int, str] | None = None
        nudge_seen = False
        prewarning_seen = False
        for record in rehydrated.records:
            if not isinstance(record, IRFooterQueueRecord):
                continue
            footer = record.footer
            if footer.source == "telemetry" and footer.key == "gas_gauge":
                latest_gauge = _parse_gas_gauge(footer.text)
                if nudge_seen:
                    self._rehydrate_nudge_gauge(latest_gauge)
                if prewarning_seen:
                    self._rehydrate_prewarning_gauge(latest_gauge)
                continue
            if footer.source != "planner":
                continue
            if any(marker in footer.text for marker in _SLEEP_NUDGE_MARKERS):
                nudge_seen = True
                self._sleep_nudge_regime = _nudge_regime_from_gauge(latest_gauge)
            if _SLEEP_PREWARNING_MESSAGE in footer.text:
                prewarning_seen = True
                self._prewarning_active = True

    def _rehydrate_nudge_gauge(
        self,
        gauge: tuple[int, int, str] | None,
    ) -> None:
        if gauge is None:
            return
        _lower_tokens, upper_tokens, _regime = gauge
        warning_exit = max(
            0,
            self._config.soft_threshold - self.sleep_pressure_hysteresis,
        )
        hard_exit = max(
            0,
            self._config.hard_threshold - self.sleep_pressure_hysteresis,
        )
        if self._sleep_nudge_regime == "hard" and upper_tokens < hard_exit:
            self._sleep_nudge_regime = (
                "calm" if upper_tokens < warning_exit else "warning"
            )
        elif self._sleep_nudge_regime == "warning" and upper_tokens < warning_exit:
            self._sleep_nudge_regime = "calm"

    def _rehydrate_prewarning_gauge(
        self,
        gauge: tuple[int, int, str] | None,
    ) -> None:
        if gauge is None or not self._prewarning_active:
            return
        _lower_tokens, upper_tokens, _regime = gauge
        prewarning_exit = max(
            0,
            self.sleep_prewarning_threshold - self.sleep_pressure_hysteresis,
        )
        if upper_tokens < prewarning_exit:
            self._prewarning_active = False

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


def _regime_rank(regime: SleepNudgeRegime) -> int:
    return {"calm": 0, "warning": 1, "hard": 2}[regime]


def _parse_gas_gauge(text: str) -> tuple[int, int, str] | None:
    match = _GAS_GAUGE_RE.search(text)
    if match is None:
        return None
    lower_tokens = int(match.group("thousands")) * 1_000
    return lower_tokens, lower_tokens + 999, match.group("regime")


def _nudge_regime_from_gauge(
    gauge: tuple[int, int, str] | None,
) -> SleepNudgeRegime:
    if gauge is not None and gauge[2] == "critical":
        return "hard"
    return "warning"


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
