"""Pure state and planning primitives for the Dreamer's resolution frontier.

The transcript remains canonical. This module derives a view from rehydrated
semantic blocks, plans mode changes, describes a completed sleep, and forecasts
sleep duration. It never records or applies a mode transition.

The public structures are deliberately frozen. Slice 2 may execute a
``FrontierAdvancePlan`` through the existing semantic-block apply-mode record
machinery, then build a ``MorningManifest`` from the transitions that actually
landed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from spellbook.config import DEFAULT_SOFT_THRESHOLD
from spellbook.ir_types import (
    IRSemanticBlock,
    IRSemanticBlockPairNarrative,
    IRSemanticBlockPairNarrativeChild,
    IRSemanticBlockSummary,
    IRTokenRangeCount,
    SemanticBlockMode,
)

DEFAULT_RECENT_FULL_BLOCKS = 4
DEFAULT_DREAM_TRANSCRIPT_POINTER = "forks/"

PlanReasonCode = Literal[
    "missing_summary",
    "pinned",
    "narrative_deferred",
    "narrative_incomplete",
]
SleepKind = Literal["sleep", "deep_sleep", "forced_sleep"]
ManifestDebtCode = Literal[
    "missing_summary",
    "pinned",
    "narrative_deferred",
    "narrative_incomplete",
    "deep_sleep_deferred",
    "forced_sleep_minimalism",
    "token_measurement_missing",
    "transition_not_applied",
]
ForecastKind = Literal["sleep", "deep_sleep"]
ForecastConfidence = Literal["low", "medium", "high"]
FrontierPolicyMode = Literal["calm_targeted", "fixed"]
ProjectionEstimateQuality = Literal[
    "exact",
    "approximate",
    "conservative",
    "unavailable",
]
ProjectionOutcome = Literal[
    "already_calm",
    "calm_reached",
    "floor_reached",
    "fixed",
    "refused",
]


class SemanticBlockWorld(Protocol):
    """The part of a rehydrated world needed to derive the frontier."""

    semantic_blocks: list[IRSemanticBlock]


@dataclass(frozen=True, slots=True)
class FrontierNarrative:
    """A complete, usable pair narrative found in semantic-block artifacts."""

    narrative_id: str
    chapter_number: int
    title: str
    block_ids: tuple[str, str]
    block_indices: tuple[int, int]
    active: bool
    source_chapter_path: str | None
    compiled_json_path: str | None


@dataclass(frozen=True, slots=True)
class FrontierBlock:
    """One semantic block as seen by the resolution frontier."""

    block_id: str
    block_idx: int
    title: str
    mode: SemanticBlockMode
    pinned: bool
    facet_pin_count: int
    current_tokens: IRTokenRangeCount | None
    full_tokens: IRTokenRangeCount | None
    summary_artifact_id: str | None
    summary_tokens: IRTokenRangeCount | None
    narrative_artifact_present: bool
    narrative_id: str | None
    narrative_chapter: int | None
    narrative_tokens: IRTokenRangeCount | None

    @property
    def has_summary(self) -> bool:
        return self.summary_artifact_id is not None

    @property
    def has_narrative(self) -> bool:
        return self.narrative_id is not None


@dataclass(frozen=True, slots=True)
class FrontierState:
    """Derived resolution state over the ordered semantic-block prefix."""

    blocks: tuple[FrontierBlock, ...]
    narratives: tuple[FrontierNarrative, ...]

    def block(self, block_idx: int) -> FrontierBlock:
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise ValueError(
                f"Block {block_idx} is outside the frontier of {len(self.blocks)} blocks."
            )
        block = self.blocks[block_idx]
        if block.block_idx != block_idx:
            raise ValueError(
                f"Frontier block coordinate mismatch: expected {block_idx}, "
                f"found {block.block_idx}."
            )
        return block

    @property
    def full_block_indices(self) -> tuple[int, ...]:
        return tuple(block.block_idx for block in self.blocks if block.mode == "full")

    @property
    def summary_block_indices(self) -> tuple[int, ...]:
        return tuple(
            block.block_idx for block in self.blocks if block.mode == "summary"
        )

    @property
    def active_narrative_chapters(self) -> tuple[int, ...]:
        return tuple(
            narrative.chapter_number
            for narrative in self.narratives
            if narrative.active
        )


@dataclass(frozen=True, slots=True)
class FrontierPolicy:
    """Policy for a frontier-only Sleep.

    By default, Sleep advances the oldest memory only until its conservative
    projection falls below the Homunculus warning threshold. Four recent blocks
    is the hard floor because it retains two adjacent level-1 source pairs at
    full resolution. Passing ``recent_full_blocks`` explicitly selects the
    backward-compatible fixed-window policy.
    """

    recent_full_blocks: int | None = None
    calm_target_tokens: int = DEFAULT_SOFT_THRESHOLD
    prefer_narratives: bool = True
    min_kept: int | None = None
    """Optional floor on kept-full blocks for calm-targeted plans.

    The calm search still runs, but the plan never advances past the point
    that would leave fewer than ``min_kept`` recent blocks at full fidelity.
    Values below the hard four-block floor are raised to it. Ignored when
    ``recent_full_blocks`` selects the fixed-window policy.
    """

    def __post_init__(self) -> None:
        if self.recent_full_blocks is not None and self.recent_full_blocks < 0:
            raise ValueError("recent_full_blocks must be non-negative.")
        if self.calm_target_tokens < 0:
            raise ValueError("calm_target_tokens must be non-negative.")

    @property
    def mode(self) -> FrontierPolicyMode:
        return "calm_targeted" if self.recent_full_blocks is None else "fixed"


@dataclass(frozen=True, slots=True)
class FrontierPlanProjection:
    """The pressure calculation that selected a frontier boundary."""

    policy_mode: FrontierPolicyMode
    target_tokens: int
    current_render_tokens: int | None
    projected_render_tokens: int | None
    estimated_tokens_freed: int
    estimate_quality: ProjectionEstimateQuality
    kept_full_blocks: int
    outcome: ProjectionOutcome

    @property
    def calm_reached(self) -> bool:
        return (
            self.projected_render_tokens is not None
            and self.projected_render_tokens < self.target_tokens
        )

    def render(self) -> str:
        target = f"calm below {self.target_tokens:,} tokens"
        kept = _counted_noun(
            self.kept_full_blocks,
            "recent block full",
            "recent blocks full",
        )
        if self.projected_render_tokens is None:
            projected = "projected result unavailable"
        else:
            projected = (
                f"projected result {self.projected_render_tokens:,} tokens "
                f"({self.estimate_quality} estimate)"
            )

        if self.outcome == "already_calm":
            return (
                f"Target: {target}; current render is already calm at "
                f"{self.current_render_tokens:,} tokens; kept {kept} and advanced "
                "nothing."
            )
        if self.outcome == "calm_reached":
            return (
                f"Target: {target}; {projected}; kept {kept}; calm reached without "
                "touching them."
            )
        if self.outcome == "floor_reached":
            return (
                f"Target: {target}; {projected}; kept {kept}; the recent-full floor "
                "stopped further advance."
            )
        if self.outcome == "refused":
            return (
                f"Target: {target}; {projected}; kept {kept}; the selected frontier "
                "was refused rather than projecting unlandable relief."
            )
        return f"Fixed-N policy kept {kept}; {projected}."


@dataclass(frozen=True, slots=True)
class FrontierPlanReason:
    code: PlanReasonCode
    message: str
    block_indices: tuple[int, ...] = ()
    chapter_number: int | None = None


@dataclass(frozen=True, slots=True)
class FrontierTransition:
    """A requested mode record; this type does not apply it."""

    block_id: str
    block_idx: int
    title: str
    from_mode: SemanticBlockMode
    to_mode: SemanticBlockMode
    before_tokens: IRTokenRangeCount | None
    after_tokens: IRTokenRangeCount | None
    reason: str
    narrative_id: str | None = None
    narrative_chapter: int | None = None

    @property
    def tokens_freed(self) -> int | None:
        if self.before_tokens is None or self.after_tokens is None:
            return None
        return self.before_tokens.tokens - self.after_tokens.tokens

    @property
    def token_delta_exact(self) -> bool:
        return (
            self.before_tokens is not None
            and self.after_tokens is not None
            and self.before_tokens.exact
            and self.after_tokens.exact
        )


@dataclass(frozen=True, slots=True)
class FrontierAdvancePlan:
    """The complete, mutation-free decision for one frontier advance."""

    frontier: FrontierState
    policy: FrontierPolicy
    transitions: tuple[FrontierTransition, ...]
    reasons: tuple[FrontierPlanReason, ...]
    narratives_applied: tuple[FrontierNarrative, ...]
    projection: FrontierPlanProjection
    refused: bool = False

    @property
    def empty(self) -> bool:
        return not self.transitions

    @property
    def known_tokens_freed(self) -> int:
        return sum(
            tokens
            for transition in self.transitions
            if (tokens := transition.tokens_freed) is not None
        )

    @property
    def tokens_freed(self) -> int | None:
        if any(transition.tokens_freed is None for transition in self.transitions):
            return None
        return self.known_tokens_freed

    @property
    def token_delta_exact(self) -> bool:
        return all(
            transition.tokens_freed is not None and transition.token_delta_exact
            for transition in self.transitions
        )


def derive_frontier(
    world: SemanticBlockWorld | Sequence[IRSemanticBlock],
) -> FrontierState:
    """Derive current frontier state without changing the rehydrated world.

    Active pair narratives are validated as an atomic two-block rendering. An
    orphaned inactive artifact is represented as present-but-unusable so the
    planner can disclose the debt and safely fall back to an existing summary.
    """

    semantic_blocks = _semantic_blocks(world)
    for expected_idx, block in enumerate(semantic_blocks):
        if block.idx != expected_idx:
            raise ValueError(
                "Frontier requires ordered, gapless semantic block indices: "
                f"expected {expected_idx}, found {block.idx}."
            )

    narratives = _derive_narratives(semantic_blocks)
    narrative_by_block = {
        block_idx: narrative
        for narrative in narratives
        for block_idx in narrative.block_indices
    }
    _validate_active_narratives(semantic_blocks, narrative_by_block)

    blocks: list[FrontierBlock] = []
    for block in semantic_blocks:
        summary = _latest_summary(block)
        narrative_artifact = _latest_narrative_artifact(block)
        narrative = narrative_by_block.get(block.idx)
        narrative_tokens = None
        if narrative is not None and narrative_artifact is not None:
            narrative_tokens = narrative_artifact.toks
        blocks.append(
            FrontierBlock(
                block_id=block.id,
                block_idx=block.idx,
                title=block.title,
                mode=block.mode,
                pinned=block.pin is not None,
                facet_pin_count=len(block.facet_pins),
                current_tokens=_current_token_count(
                    block,
                    summary=summary,
                    narrative_tokens=narrative_tokens,
                ),
                full_tokens=block.full_toks,
                summary_artifact_id=summary.id if summary is not None else None,
                summary_tokens=summary.toks if summary is not None else None,
                narrative_artifact_present=narrative_artifact is not None,
                narrative_id=narrative.narrative_id if narrative is not None else None,
                narrative_chapter=(
                    narrative.chapter_number if narrative is not None else None
                ),
                narrative_tokens=narrative_tokens,
            )
        )
    return FrontierState(blocks=tuple(blocks), narratives=narratives)


def plan_frontier_advance(
    world: SemanticBlockWorld | Sequence[IRSemanticBlock] | FrontierState,
    policy: FrontierPolicy | None = None,
    *,
    current_render_tokens: int | None = None,
) -> FrontierAdvancePlan:
    """Compute the gentlest safe transition set for a frontier-only Sleep.

    Missing summaries are a global refusal, not an invitation to make a partial
    guess: when any eligible block has neither a usable narrative nor a summary,
    the returned plan has no transitions and explains every known debt. The
    default policy searches from the full recent window toward the four-block
    floor and selects the first boundary whose conservative projection is calm.
    """

    resolved_policy = policy or FrontierPolicy()
    frontier = world if isinstance(world, FrontierState) else derive_frontier(world)
    current_tokens, baseline_quality = _projection_baseline(
        frontier,
        current_render_tokens,
    )

    if resolved_policy.mode == "fixed":
        fixed_count = resolved_policy.recent_full_blocks
        assert fixed_count is not None
        return _plan_fixed_frontier(
            frontier,
            resolved_policy,
            recent_full_blocks=fixed_count,
            current_render_tokens=current_tokens,
            baseline_quality=baseline_quality,
        )

    kept_all = len(frontier.blocks)
    if (
        current_tokens is not None
        and current_tokens < resolved_policy.calm_target_tokens
    ):
        return FrontierAdvancePlan(
            frontier=frontier,
            policy=resolved_policy,
            transitions=(),
            reasons=(),
            narratives_applied=(),
            projection=FrontierPlanProjection(
                policy_mode="calm_targeted",
                target_tokens=resolved_policy.calm_target_tokens,
                current_render_tokens=current_tokens,
                projected_render_tokens=current_tokens,
                estimated_tokens_freed=0,
                estimate_quality=baseline_quality,
                kept_full_blocks=kept_all,
                outcome="already_calm",
            ),
        )

    requested_floor = max(
        DEFAULT_RECENT_FULL_BLOCKS,
        resolved_policy.min_kept if resolved_policy.min_kept is not None else 0,
    )
    recent_floor = min(requested_floor, len(frontier.blocks))
    floor_plan: FrontierAdvancePlan | None = None
    for kept_full_blocks in range(kept_all, recent_floor - 1, -1):
        candidate = _plan_fixed_frontier(
            frontier,
            resolved_policy,
            recent_full_blocks=kept_full_blocks,
            current_render_tokens=current_tokens,
            baseline_quality=baseline_quality,
        )
        if kept_full_blocks == recent_floor:
            floor_plan = candidate
        if not candidate.refused and candidate.projection.calm_reached:
            return candidate

    assert floor_plan is not None
    return floor_plan


def _plan_fixed_frontier(
    frontier: FrontierState,
    policy: FrontierPolicy,
    *,
    recent_full_blocks: int,
    current_render_tokens: int | None,
    baseline_quality: ProjectionEstimateQuality,
) -> FrontierAdvancePlan:
    """Apply the established fixed boundary rules without mutating the world."""

    recent_start = max(0, len(frontier.blocks) - recent_full_blocks)
    protected_recent = set(range(recent_start, len(frontier.blocks)))
    protected_pins = {block.block_idx for block in frontier.blocks if block.pinned}
    protected = protected_recent | protected_pins
    eligible = {
        block.block_idx for block in frontier.blocks if block.block_idx not in protected
    }

    reasons: list[FrontierPlanReason] = []
    targets: dict[int, SemanticBlockMode] = {
        block_idx: "full" for block_idx in protected
    }
    target_narratives: dict[int, FrontierNarrative] = {}

    for block_idx in sorted(protected_pins - protected_recent):
        block = frontier.block(block_idx)
        reasons.append(
            FrontierPlanReason(
                code="pinned",
                block_indices=(block_idx,),
                message=(
                    f'Block {block_idx} "{block.title}" targets full resolution '
                    "because its block pin is absolute."
                ),
            )
        )

    if policy.prefer_narratives:
        for narrative in frontier.narratives:
            pair = set(narrative.block_indices)
            if pair <= eligible:
                for block_idx in narrative.block_indices:
                    targets[block_idx] = "pair_narrative"
                    target_narratives[block_idx] = narrative
                continue
            eligible_half = tuple(sorted(pair & eligible))
            if eligible_half or narrative.active:
                reasons.append(
                    FrontierPlanReason(
                        code="narrative_deferred",
                        block_indices=narrative.block_indices,
                        chapter_number=narrative.chapter_number,
                        message=(
                            f"Chapter {narrative.chapter_number} was not applied because "
                            "its adjacent pair crosses the pinned or recent-full boundary."
                        ),
                    )
                )

    missing: list[FrontierPlanReason] = []
    for block_idx in sorted(eligible):
        block = frontier.block(block_idx)
        if block_idx in targets:
            continue
        if block.has_summary:
            targets[block_idx] = "summary"
            if block.narrative_artifact_present and not block.has_narrative:
                reasons.append(
                    FrontierPlanReason(
                        code="narrative_incomplete",
                        block_indices=(block_idx,),
                        message=(
                            f'Block {block_idx} "{block.title}" has an incomplete pair '
                            "narrative; Sleep can use its summary but owes a repaired "
                            "chapter pair."
                        ),
                    )
                )
            continue
        detail = ""
        if block.narrative_artifact_present:
            detail = " Its pair narrative artifact is incomplete or mismatched."
        missing.append(
            FrontierPlanReason(
                code="missing_summary",
                block_indices=(block_idx,),
                message=(
                    f'Block {block_idx} "{block.title}" has neither a usable pair '
                    f"narrative nor a summary.{detail} The whole frontier advance was "
                    "refused rather than guessing."
                ),
            )
        )

    if missing:
        return FrontierAdvancePlan(
            frontier=frontier,
            policy=policy,
            transitions=(),
            reasons=tuple([*reasons, *missing]),
            narratives_applied=(),
            projection=_project_frontier_plan(
                frontier,
                policy,
                transitions=(),
                current_render_tokens=current_render_tokens,
                baseline_quality=baseline_quality,
                kept_full_blocks=min(recent_full_blocks, len(frontier.blocks)),
                refused=True,
            ),
            refused=True,
        )

    transitions: list[FrontierTransition] = []
    for block_idx, target in sorted(targets.items()):
        block = frontier.block(block_idx)
        if block.mode == target:
            continue
        narrative = target_narratives.get(block_idx)
        if target == "pair_narrative":
            after_tokens = block.narrative_tokens
            transition_reason = (
                "an existing adjacent-pair narrative is the preferred older-memory "
                "resolution"
            )
        elif target == "summary":
            after_tokens = block.summary_tokens
            transition_reason = (
                "no usable pair narrative exists and a summary is available"
            )
        else:
            after_tokens = block.full_tokens
            transition_reason = "the block is pinned or inside the recent-full window"
        transitions.append(
            FrontierTransition(
                block_id=block.block_id,
                block_idx=block_idx,
                title=block.title,
                from_mode=block.mode,
                to_mode=target,
                before_tokens=block.current_tokens,
                after_tokens=after_tokens,
                reason=transition_reason,
                narrative_id=(
                    narrative.narrative_id if narrative is not None else None
                ),
                narrative_chapter=(
                    narrative.chapter_number if narrative is not None else None
                ),
            )
        )

    changed_narrative_ids = {
        transition.narrative_id
        for transition in transitions
        if transition.narrative_id is not None
    }
    narratives_applied = tuple(
        narrative
        for narrative in frontier.narratives
        if narrative.narrative_id in changed_narrative_ids
    )
    return FrontierAdvancePlan(
        frontier=frontier,
        policy=policy,
        transitions=tuple(transitions),
        reasons=tuple(reasons),
        narratives_applied=narratives_applied,
        projection=_project_frontier_plan(
            frontier,
            policy,
            transitions=tuple(transitions),
            current_render_tokens=current_render_tokens,
            baseline_quality=baseline_quality,
            kept_full_blocks=min(recent_full_blocks, len(frontier.blocks)),
            refused=False,
        ),
    )


def _projection_baseline(
    frontier: FrontierState,
    observed_render_tokens: int | None,
) -> tuple[int | None, ProjectionEstimateQuality]:
    """Prefer the gauge observation, while refusing an obviously stale low value."""

    if observed_render_tokens is not None and observed_render_tokens < 0:
        raise ValueError("current_render_tokens must be non-negative.")

    counts = [block.current_tokens for block in frontier.blocks]
    frontier_total = None
    if all(count is not None for count in counts):
        frontier_total = sum(count.tokens for count in counts if count is not None)

    if observed_render_tokens is None:
        if frontier_total is None:
            return None, "unavailable"
        # This is a useful pure-planner fallback, but it omits frame and tail
        # overhead and therefore is never presented as an exact render count.
        return frontier_total, "approximate"

    if frontier_total is not None and frontier_total > observed_render_tokens:
        # The gauge is an input-side observation and can lag newly integrated
        # output. Taking the larger known component avoids claiming false calm.
        return frontier_total, "conservative"
    return observed_render_tokens, "exact"


def _project_frontier_plan(
    frontier: FrontierState,
    policy: FrontierPolicy,
    *,
    transitions: tuple[FrontierTransition, ...],
    current_render_tokens: int | None,
    baseline_quality: ProjectionEstimateQuality,
    kept_full_blocks: int,
    refused: bool,
) -> FrontierPlanProjection:
    savings, savings_quality = _estimated_transition_savings(frontier, transitions)
    if current_render_tokens is None:
        projected_tokens = None
        quality: ProjectionEstimateQuality = "unavailable"
    else:
        projected_tokens = max(0, current_render_tokens - savings)
        quality = _least_certain_quality(baseline_quality, savings_quality)

    if refused:
        outcome: ProjectionOutcome = "refused"
    elif policy.mode == "fixed":
        outcome = "fixed"
    elif projected_tokens is not None and projected_tokens < policy.calm_target_tokens:
        outcome = "calm_reached"
    else:
        outcome = "floor_reached"

    return FrontierPlanProjection(
        policy_mode=policy.mode,
        target_tokens=policy.calm_target_tokens,
        current_render_tokens=current_render_tokens,
        projected_render_tokens=projected_tokens,
        estimated_tokens_freed=savings,
        estimate_quality=quality,
        kept_full_blocks=kept_full_blocks,
        outcome=outcome,
    )


def _estimated_transition_savings(
    frontier: FrontierState,
    transitions: tuple[FrontierTransition, ...],
) -> tuple[int, ProjectionEstimateQuality]:
    """Under-promise relief for unknown destinations and atomic narratives."""

    grouped: dict[str, list[FrontierTransition]] = {}
    for transition in transitions:
        key = (
            f"narrative:{transition.narrative_id}"
            if transition.narrative_id is not None
            else f"block:{transition.block_idx}"
        )
        grouped.setdefault(key, []).append(transition)

    estimated_savings = 0
    quality: ProjectionEstimateQuality = "exact"
    for group in grouped.values():
        summary_with_facet_pins = any(
            transition.to_mode == "summary"
            and frontier.block(transition.block_idx).facet_pin_count > 0
            for transition in group
        )
        if summary_with_facet_pins or any(
            transition.tokens_freed is None for transition in group
        ):
            # A narrative is one atomic render. If either half is unknown, even
            # the known half's apparent savings cannot safely price the pair.
            quality = _least_certain_quality(quality, "conservative")
            continue

        group_savings = sum(transition.tokens_freed or 0 for transition in group)
        estimated_savings += group_savings
        if not all(transition.token_delta_exact for transition in group):
            quality = _least_certain_quality(quality, "approximate")
    return estimated_savings, quality


def _least_certain_quality(
    first: ProjectionEstimateQuality,
    second: ProjectionEstimateQuality,
) -> ProjectionEstimateQuality:
    order: tuple[ProjectionEstimateQuality, ...] = (
        "exact",
        "approximate",
        "conservative",
        "unavailable",
    )
    return order[max(order.index(first), order.index(second))]


def advance_frontier(
    world: SemanticBlockWorld | Sequence[IRSemanticBlock] | FrontierState,
    policy: FrontierPolicy | None = None,
    *,
    current_render_tokens: int | None = None,
) -> FrontierAdvancePlan:
    """Readable alias for ``plan_frontier_advance``; still pure and non-mutating."""

    return plan_frontier_advance(
        world,
        policy,
        current_render_tokens=current_render_tokens,
    )


@dataclass(frozen=True, slots=True)
class ManifestDebt:
    code: ManifestDebtCode
    message: str
    block_indices: tuple[int, ...] = ()
    chapter_number: int | None = None


@dataclass(frozen=True, slots=True)
class MorningManifest:
    """Covenant-grade account of one completed sleep."""

    sleep_kind: SleepKind
    projection: FrontierPlanProjection
    deltas: tuple[FrontierTransition, ...]
    debts: tuple[ManifestDebt, ...]
    narratives_applied: tuple[FrontierNarrative, ...]
    chapters_authored: int
    dream_transcript_paths: tuple[str, ...]
    prewarning_tokens: int | None = None
    floor_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.chapters_authored < 0:
            raise ValueError("chapters_authored must be non-negative.")
        if self.sleep_kind != "deep_sleep" and self.chapters_authored:
            raise ValueError("Only Deep Sleep may author new chapters.")
        if (self.narratives_applied or self.chapters_authored) and not (
            self.dream_transcript_paths
        ):
            raise ValueError(
                "Chapter-bearing manifests need a dream transcript pointer."
            )
        if self.prewarning_tokens is not None and self.prewarning_tokens < 0:
            raise ValueError("Pre-warning tokens must be non-negative.")
        if self.floor_tokens is not None and self.floor_tokens < 0:
            raise ValueError("Floor tokens must be non-negative.")
        if not self.forced and (
            self.prewarning_tokens is not None or self.floor_tokens is not None
        ):
            raise ValueError("Only forced Sleep carries pre-warning history.")

    @property
    def forced(self) -> bool:
        return self.sleep_kind == "forced_sleep"

    @property
    def known_tokens_freed(self) -> int:
        return sum(
            tokens
            for delta in self.deltas
            if (tokens := delta.tokens_freed) is not None
        )

    @property
    def tokens_freed(self) -> int | None:
        if any(delta.tokens_freed is None for delta in self.deltas):
            return None
        return self.known_tokens_freed

    @property
    def token_delta_exact(self) -> bool:
        return all(
            delta.tokens_freed is not None and delta.token_delta_exact
            for delta in self.deltas
        )

    def render(self) -> str:
        """Render the structured manifest without hiding empty sections."""

        lines = [self._opening()]
        if self.forced:
            lines.append("forced=true")
            if self.prewarning_tokens is None or self.floor_tokens is None:
                lines.append("Pre-warning history: unavailable.")
            else:
                lines.append(
                    "Pre-warning history: issued at "
                    f"{self.prewarning_tokens:,} tokens before this "
                    f"{self.floor_tokens:,}-token floor."
                )
        lines.extend(["", "Policy", f"- {self.projection.render()}"])
        lines.extend(["", "Deltas"])
        if self.deltas:
            for delta in self.deltas:
                destination = delta.to_mode.replace("pair_narrative", "narrative")
                if delta.narrative_chapter is not None:
                    destination += f" (chapter {delta.narrative_chapter})"
                lines.append(
                    f'- Block {delta.block_idx} "{delta.title}": '
                    f"{delta.from_mode} -> {destination}; "
                    f"{_render_token_delta(delta)}."
                )
        else:
            lines.append("- No block modes moved.")
        lines.append(f"- {_render_delta_total(self)}")

        lines.extend(["", "Debts"])
        if self.debts:
            lines.extend(f"- {debt.message}" for debt in self.debts)
        else:
            lines.append("- None. Nothing was skipped or left unmeasured.")

        if self.narratives_applied or self.chapters_authored:
            pointers = ", ".join(self.dream_transcript_paths)
            lines.extend(
                [
                    "",
                    "The dream itself is kept, if you ever want to hold it: "
                    f"{pointers}.",
                ]
            )
        return "\n".join(lines)

    def _opening(self) -> str:
        if self.sleep_kind == "deep_sleep":
            if self.chapters_authored == 0:
                return (
                    "Good morning. Deep Sleep completed without authoring a new "
                    "chapter; the debts below say what remains owed."
                )
            return (
                "Good morning. You slept and dreamed; the account below says what "
                "changed and what remains owed."
            )
        if self.sleep_kind == "forced_sleep":
            return (
                "Good morning. Forced Sleep occurred at the hard limit. It advanced "
                "existing memory only; no new chapter was authored."
            )
        return (
            "Good morning. You slept. This Sleep advanced existing memory only; "
            "no new chapter was authored."
        )

    def __str__(self) -> str:
        return self.render()


def build_morning_manifest(
    plan: FrontierAdvancePlan,
    *,
    applied_deltas: Sequence[FrontierTransition] | None = None,
    sleep_kind: SleepKind = "sleep",
    deferred_deep_sleep_chapters: int = 0,
    chapters_authored: int = 0,
    dream_transcript_paths: Sequence[str] = (),
    additional_debts: Sequence[ManifestDebt] = (),
    prewarning_tokens: int | None = None,
    floor_tokens: int | None = None,
) -> MorningManifest:
    """Build a manifest from an advance decision and the deltas that landed.

    ``applied_deltas`` lets the executor report actual outcomes. Omitting it is
    appropriate only when the complete pure plan was applied successfully.
    Forced-sleep minimalism, deferred Deep Sleep work, and unknown token deltas
    become debts automatically so callers cannot accidentally conceal them.
    """

    if deferred_deep_sleep_chapters < 0:
        raise ValueError("deferred_deep_sleep_chapters must be non-negative.")
    if chapters_authored < 0:
        raise ValueError("chapters_authored must be non-negative.")
    deltas = tuple(plan.transitions if applied_deltas is None else applied_deltas)
    planned_by_key = {_transition_key(delta): delta for delta in plan.transitions}
    applied_keys = {_transition_key(delta) for delta in deltas}
    unexpected = applied_keys - planned_by_key.keys()
    if unexpected:
        raise ValueError("applied_deltas must be a subset of the frontier plan.")
    applied_narrative_ids = {
        delta.narrative_id for delta in deltas if delta.narrative_id is not None
    }
    narratives_applied = tuple(
        narrative
        for narrative in plan.narratives_applied
        if narrative.narrative_id in applied_narrative_ids
    )
    debts = [
        ManifestDebt(
            code=reason.code,
            message=reason.message,
            block_indices=reason.block_indices,
            chapter_number=reason.chapter_number,
        )
        for reason in plan.reasons
    ]
    for key, delta in planned_by_key.items():
        if key in applied_keys:
            continue
        debts.append(
            ManifestDebt(
                code="transition_not_applied",
                block_indices=(delta.block_idx,),
                message=(
                    f'Block {delta.block_idx} "{delta.title}" did not make its planned '
                    f"{delta.from_mode} -> {delta.to_mode} transition; execution still "
                    "owes the specific reason."
                ),
            )
        )
    if deferred_deep_sleep_chapters:
        noun = "chapter" if deferred_deep_sleep_chapters == 1 else "chapters"
        debts.append(
            ManifestDebt(
                code="deep_sleep_deferred",
                message=(
                    f"Deep Sleep still owes {deferred_deep_sleep_chapters} {noun}; "
                    "this sleep did not author them."
                ),
            )
        )
    if sleep_kind == "forced_sleep":
        debts.append(
            ManifestDebt(
                code="forced_sleep_minimalism",
                message=(
                    "Forced Sleep was deliberately frontier-only because no new "
                    "memory may be authored at the mind's hard limit."
                ),
            )
        )
    unknown_block_indices = tuple(
        delta.block_idx for delta in deltas if delta.tokens_freed is None
    )
    if unknown_block_indices:
        rendered = ", ".join(str(idx) for idx in unknown_block_indices)
        debts.append(
            ManifestDebt(
                code="token_measurement_missing",
                block_indices=unknown_block_indices,
                message=(
                    f"Token change was not measurable for blocks {rendered}; known "
                    "savings are reported only as the measured portion."
                ),
            )
        )
    debts.extend(additional_debts)

    pointers = tuple(dict.fromkeys(str(path) for path in dream_transcript_paths))
    if (narratives_applied or chapters_authored) and not pointers:
        pointers = (DEFAULT_DREAM_TRANSCRIPT_POINTER,)
    return MorningManifest(
        sleep_kind=sleep_kind,
        projection=plan.projection,
        deltas=deltas,
        debts=tuple(debts),
        narratives_applied=narratives_applied,
        chapters_authored=chapters_authored,
        dream_transcript_paths=pointers,
        prewarning_tokens=prewarning_tokens,
        floor_tokens=floor_tokens,
    )


@dataclass(frozen=True, slots=True)
class DurationEstimate:
    """An honest three-point duration estimate in seconds."""

    lower_seconds: float
    likely_seconds: float
    upper_seconds: float

    def __post_init__(self) -> None:
        if self.lower_seconds < 0:
            raise ValueError("Duration bounds cannot be negative.")
        if not self.lower_seconds <= self.likely_seconds <= self.upper_seconds:
            raise ValueError("Duration bounds must contain the likely estimate.")

    def scaled(self, count: int) -> DurationEstimate:
        if count < 0:
            raise ValueError("Duration work counts must be non-negative.")
        return DurationEstimate(
            lower_seconds=self.lower_seconds * count,
            likely_seconds=self.likely_seconds * count,
            upper_seconds=self.upper_seconds * count,
        )

    def plus(self, other: DurationEstimate) -> DurationEstimate:
        return DurationEstimate(
            lower_seconds=self.lower_seconds + other.lower_seconds,
            likely_seconds=self.likely_seconds + other.likely_seconds,
            upper_seconds=self.upper_seconds + other.upper_seconds,
        )

    def contains(self, actual_seconds: float) -> bool:
        return self.lower_seconds <= actual_seconds <= self.upper_seconds


@dataclass(frozen=True, slots=True)
class DreamTreeShape:
    """Pending work visible to the duration forecaster."""

    pending_chapter_writes: int = 0
    pending_merges: int = 0

    def __post_init__(self) -> None:
        if self.pending_chapter_writes < 0 or self.pending_merges < 0:
            raise ValueError("Dream-tree work counts must be non-negative.")


@dataclass(frozen=True, slots=True)
class DreamDurationCalibration:
    """Replaceable unit costs; seconds keep the math inspectable."""

    sleep: DurationEstimate = field(
        default_factory=lambda: DurationEstimate(0.05, 0.25, 2.0)
    )
    chapter_write: DurationEstimate = field(
        default_factory=lambda: DurationEstimate(8 * 60, 12 * 60, 20 * 60)
    )
    merge: DurationEstimate = field(
        default_factory=lambda: DurationEstimate(10 * 60, 14 * 60, 22 * 60)
    )
    deep_sleep_overhead: DurationEstimate = field(
        default_factory=lambda: DurationEstimate(0, 0, 0)
    )
    basis: str = (
        "Conservative initial calibration for the narrator/Director loop; replace "
        "with measured production samples when per-job timing telemetry exists."
    )
    confidence: ForecastConfidence = "low"


@dataclass(frozen=True, slots=True)
class DurationForecast:
    kind: ForecastKind
    estimate: DurationEstimate
    tree_shape: DreamTreeShape
    basis: str
    confidence: ForecastConfidence

    def render(self) -> str:
        if self.kind == "sleep":
            return (
                "Sleep: ~seconds (usually <1s; honest range "
                f"{_format_duration(self.estimate.lower_seconds)}-"
                f"{_format_duration(self.estimate.upper_seconds)}), frontier-only."
            )
        chapters = _counted_noun(
            self.tree_shape.pending_chapter_writes, "chapter", "chapters"
        )
        merges = _counted_noun(self.tree_shape.pending_merges, "merge", "merges")
        return (
            "Deep Sleep tonight: "
            f"~{_format_duration(self.estimate.likely_seconds)} "
            f"(range {_format_duration(self.estimate.lower_seconds)}-"
            f"{_format_duration(self.estimate.upper_seconds)}), "
            f"{chapters} + {merges}."
        )

    def __str__(self) -> str:
        return self.render()


def forecast_sleep(
    calibration: DreamDurationCalibration | None = None,
) -> DurationForecast:
    """Forecast the pure-Python frontier advance honestly, even though it is tiny."""

    resolved = calibration or DreamDurationCalibration()
    return DurationForecast(
        kind="sleep",
        estimate=resolved.sleep,
        tree_shape=DreamTreeShape(),
        basis="Pure frontier derivation and planning; no model calls. "
        + resolved.basis,
        confidence=resolved.confidence,
    )


def forecast_deep_sleep(
    tree_shape: DreamTreeShape,
    calibration: DreamDurationCalibration | None = None,
) -> DurationForecast:
    """Forecast authored chapter work plus recursive merge work."""

    resolved = calibration or DreamDurationCalibration()
    estimate = resolved.deep_sleep_overhead.plus(
        resolved.chapter_write.scaled(tree_shape.pending_chapter_writes)
    ).plus(resolved.merge.scaled(tree_shape.pending_merges))
    return DurationForecast(
        kind="deep_sleep",
        estimate=estimate,
        tree_shape=tree_shape,
        basis=resolved.basis,
        confidence=resolved.confidence,
    )


def _semantic_blocks(
    world: SemanticBlockWorld | Sequence[IRSemanticBlock],
) -> tuple[IRSemanticBlock, ...]:
    if isinstance(world, Sequence):
        return tuple(cast(Sequence[IRSemanticBlock], world))
    return tuple(world.semantic_blocks)


def _derive_narratives(
    blocks: tuple[IRSemanticBlock, ...],
) -> tuple[FrontierNarrative, ...]:
    narratives: list[FrontierNarrative] = []
    for parent_block in blocks:
        parent = _latest_narrative_artifact(parent_block)
        if not isinstance(parent, IRSemanticBlockPairNarrative):
            continue
        first_idx, second_idx = parent.pair
        if first_idx < 0 or first_idx != parent_block.idx or second_idx >= len(blocks):
            continue
        child_block = blocks[second_idx]
        child = _latest_narrative_artifact(child_block)
        if not isinstance(child, IRSemanticBlockPairNarrativeChild):
            continue
        if (
            child.narrative_id != parent.narrative_id
            or child.pair != parent.pair
            or child.chapter_number != parent.chapter_number
            or child.parent_block_idx != parent_block.idx
            or child.parent_block_id != parent_block.id
        ):
            continue
        active_modes = (parent_block.mode, child_block.mode)
        narratives.append(
            FrontierNarrative(
                narrative_id=parent.narrative_id,
                chapter_number=parent.chapter_number,
                title=parent.title,
                block_ids=(parent_block.id, child_block.id),
                block_indices=parent.pair,
                active=active_modes == ("pair_narrative", "pair_narrative"),
                source_chapter_path=parent.source_chapter_path,
                compiled_json_path=parent.compiled_json_path,
            )
        )
    return tuple(narratives)


def _validate_active_narratives(
    blocks: tuple[IRSemanticBlock, ...],
    narrative_by_block: dict[int, FrontierNarrative],
) -> None:
    for block in blocks:
        narrative = narrative_by_block.get(block.idx)
        if block.mode == "pair_narrative" and narrative is None:
            raise ValueError(
                f"Block {block.idx} is in pair_narrative mode without a complete "
                "matching parent/child artifact pair."
            )
        if narrative is None:
            continue
        pair_modes = tuple(blocks[idx].mode for idx in narrative.block_indices)
        active_count = sum(mode == "pair_narrative" for mode in pair_modes)
        if active_count == 1:
            raise ValueError(
                f"Chapter {narrative.chapter_number} is only active on one block; "
                "pair narratives must move atomically."
            )


def _latest_summary(block: IRSemanticBlock) -> IRSemanticBlockSummary | None:
    return next(
        (
            artifact
            for artifact in reversed(block.artifacts)
            if isinstance(artifact, IRSemanticBlockSummary)
        ),
        None,
    )


def _latest_narrative_artifact(
    block: IRSemanticBlock,
) -> IRSemanticBlockPairNarrative | IRSemanticBlockPairNarrativeChild | None:
    return next(
        (
            artifact
            for artifact in reversed(block.artifacts)
            if isinstance(
                artifact,
                IRSemanticBlockPairNarrative | IRSemanticBlockPairNarrativeChild,
            )
        ),
        None,
    )


def _current_token_count(
    block: IRSemanticBlock,
    *,
    summary: IRSemanticBlockSummary | None,
    narrative_tokens: IRTokenRangeCount | None,
) -> IRTokenRangeCount | None:
    if block.toks is not None:
        return block.toks
    if block.mode == "full":
        return block.full_toks
    if block.mode == "summary" and summary is not None:
        return summary.toks
    if block.mode == "pair_narrative":
        return narrative_tokens
    return None


def _render_token_delta(delta: FrontierTransition) -> str:
    tokens = delta.tokens_freed
    if tokens is None:
        return "token change unmeasured"
    qualifier = "" if delta.token_delta_exact else "about "
    if tokens >= 0:
        return f"{qualifier}{tokens:,} tokens freed"
    return f"{qualifier}{abs(tokens):,} additional tokens carried"


def _render_delta_total(manifest: MorningManifest) -> str:
    moved = _counted_noun(len(manifest.deltas), "block moved", "blocks moved")
    narratives = _counted_noun(
        len(manifest.narratives_applied), "narrative applied", "narratives applied"
    )
    if manifest.tokens_freed is None:
        token_text = _render_token_total(
            manifest.known_tokens_freed,
            exact=manifest.token_delta_exact,
            suffix=" in the known portion",
        )
    else:
        token_text = _render_token_total(
            manifest.tokens_freed,
            exact=manifest.token_delta_exact,
        )
    return f"Total: {moved}; {token_text}; {narratives}."


def _render_token_total(tokens: int, *, exact: bool, suffix: str = "") -> str:
    qualifier = "" if exact else "about "
    if tokens >= 0:
        return f"{qualifier}{tokens:,} tokens freed{suffix}"
    return f"{qualifier}{abs(tokens):,} additional tokens carried{suffix}"


def _transition_key(
    transition: FrontierTransition,
) -> tuple[str, SemanticBlockMode, SemanticBlockMode]:
    return transition.block_id, transition.from_mode, transition.to_mode


def _counted_noun(count: int, singular: str, plural: str) -> str:
    return f"{count} {singular if count == 1 else plural}"


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{round(seconds):d}s"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes}min"
    hours, remaining_minutes = divmod(minutes, 60)
    if remaining_minutes == 0:
        return f"{hours}h"
    return f"{hours}h {remaining_minutes}min"
