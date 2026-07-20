"""Self-triggered Sleep: plan and synchronously advance the memory frontier.

Calling the tool is the consent ceremony. The runtime enters dreaming only for
an executing night, returns to the interrupted waking turn on success, refusal,
or failure, and never invokes a model. Refused frontier plans are kind results
with no mutations. Unexpected failures remain exceptions and carry a manifest
of the append-only transitions that actually landed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

from spellbook.dreaming.frontier import (
    DurationForecast,
    FrontierAdvancePlan,
    FrontierTransition,
    ManifestDebt,
    MorningManifest,
    build_morning_manifest,
    forecast_sleep,
)
from spellbook.ir_types import IRToolTextBlock
from spellbook.session_lifecycle import DreamingOutcome
from spellbook.tools.common import (
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)


class SleepInput(BaseModel):
    """Advance the existing dream frontier, or preview the night."""

    dry_run: bool = Field(
        default=False,
        description=(
            "Preview the frontier plan and duration forecast without entering "
            "dreaming, recording mode changes, or mutating awareness."
        ),
    )


class SleepExecutionError(RuntimeError):
    """An unexpected Sleep failure whose message is an honest recovery account."""


class SleepFrontierPlanner(Protocol):
    """Narrow seam shared by the tool and planner dry-run paths."""

    def plan_sleep_frontier(self) -> FrontierAdvancePlan: ...


@dataclass(frozen=True, slots=True)
class SleepDryRun:
    """The exact plan and forecast behind every rendered Sleep preview."""

    plan: FrontierAdvancePlan
    forecast: DurationForecast


def dry_run_sleep(planner: SleepFrontierPlanner) -> SleepDryRun:
    """Run Sleep's real non-mutating preview path for tools or planner nudges."""

    return SleepDryRun(
        plan=planner.plan_sleep_frontier(),
        forecast=forecast_sleep(),
    )


async def exec_sleep(meta: ToolMetadata, input: SleepInput) -> ToolExecutionResult:
    if meta.homunculus is None:
        raise ToolError("Sleep is unavailable because this session has no Homunculus.")

    if input.dry_run:
        preview = dry_run_sleep(meta.homunculus)
        return ToolExecutionResult(
            content=[
                IRToolTextBlock(text=_render_preview(preview.plan, preview.forecast))
            ],
            display=_preview_display(preview.plan, preview.forecast),
        )

    runtime = meta.dreaming_runtime
    if runtime is None:
        raise ToolError(
            "Sleep is unavailable because this session has no dreaming runtime."
        )

    outcome: DreamingOutcome = "failed"
    await runtime.enter_dreaming()
    try:
        plan = meta.homunculus.plan_sleep_frontier()
        if plan.refused:
            manifest = build_morning_manifest(plan, applied_deltas=())
            outcome = "refused"
        else:
            try:
                applied = meta.homunculus.apply_sleep_frontier(plan)
            except Exception as exc:
                applied = _landed_deltas(exc)
                manifest = build_morning_manifest(
                    plan,
                    applied_deltas=applied,
                    additional_debts=(
                        ManifestDebt(
                            code="transition_not_applied",
                            message=f"Sleep execution failed unexpectedly: {exc}",
                        ),
                    ),
                )
                raise SleepExecutionError(
                    "Sleep failed loudly. The transcript remains authoritative; "
                    "this account names every transition known to have landed.\n\n"
                    + manifest.render()
                ) from exc
            manifest = build_morning_manifest(plan, applied_deltas=applied)
            outcome = "completed"
        return ToolExecutionResult(
            content=[IRToolTextBlock(text=manifest.render())],
            display=_manifest_display(manifest, status=outcome),
        )
    finally:
        await runtime.exit_dreaming(outcome)


def _landed_deltas(error: Exception) -> tuple[FrontierTransition, ...]:
    maybe_deltas = getattr(error, "applied_deltas", ())
    if not isinstance(maybe_deltas, Sequence) or isinstance(maybe_deltas, str | bytes):
        return ()
    deltas = tuple(maybe_deltas)
    if not all(isinstance(delta, FrontierTransition) for delta in deltas):
        return ()
    return deltas


def _render_preview(
    plan: FrontierAdvancePlan,
    forecast: DurationForecast,
) -> str:
    lines = [
        "Sleep dry run. You stayed awake; nothing was recorded or changed.",
        "",
        forecast.render(),
        "",
        "Plan",
        f"- {plan.projection.render()}",
    ]
    if plan.refused:
        lines.append("- Refused as a whole; no partial frontier is available.")
    elif not plan.transitions:
        lines.append("- No block modes would move.")
    else:
        for transition in plan.transitions:
            lines.append(
                f'- Block {transition.block_idx} "{transition.title}": '
                f"{transition.from_mode} -> {transition.to_mode}; "
                f"{_preview_token_delta(transition)}."
            )

    lines.extend(["", "Reasons and forecast debts"])
    if plan.reasons:
        lines.extend(f"- {reason.message}" for reason in plan.reasons)
    else:
        lines.append("- None.")
    return "\n".join(lines)


def _preview_token_delta(transition: FrontierTransition) -> str:
    tokens = transition.tokens_freed
    if tokens is None:
        return "token change unmeasured"
    qualifier = "" if transition.token_delta_exact else "about "
    if tokens >= 0:
        return f"{qualifier}{tokens:,} tokens freed"
    return f"{qualifier}{abs(tokens):,} additional tokens carried"


def _preview_display(
    plan: FrontierAdvancePlan,
    forecast: DurationForecast,
) -> dict:
    return {
        "kind": "sleep",
        "status": "preview",
        "dry_run": True,
        "refused": plan.refused,
        "transitions": [_transition_display(delta) for delta in plan.transitions],
        "reasons": [reason.code for reason in plan.reasons],
        "projection": _projection_display(plan),
        "forecast_seconds": {
            "lower": forecast.estimate.lower_seconds,
            "likely": forecast.estimate.likely_seconds,
            "upper": forecast.estimate.upper_seconds,
        },
    }


def _manifest_display(
    manifest: MorningManifest,
    *,
    status: DreamingOutcome,
) -> dict:
    return {
        "kind": "sleep",
        "status": status,
        "dry_run": False,
        "forced": manifest.forced,
        "prewarning_tokens": manifest.prewarning_tokens,
        "floor_tokens": manifest.floor_tokens,
        "deltas": [_transition_display(delta) for delta in manifest.deltas],
        "debts": [debt.code for debt in manifest.debts],
        "known_tokens_freed": manifest.known_tokens_freed,
        "tokens_freed": manifest.tokens_freed,
        "projection": {
            "target_tokens": manifest.projection.target_tokens,
            "projected_render_tokens": manifest.projection.projected_render_tokens,
            "kept_full_blocks": manifest.projection.kept_full_blocks,
            "estimate_quality": manifest.projection.estimate_quality,
            "outcome": manifest.projection.outcome,
        },
    }


def _projection_display(plan: FrontierAdvancePlan) -> dict:
    return {
        "target_tokens": plan.projection.target_tokens,
        "current_render_tokens": plan.projection.current_render_tokens,
        "projected_render_tokens": plan.projection.projected_render_tokens,
        "estimated_tokens_freed": plan.projection.estimated_tokens_freed,
        "kept_full_blocks": plan.projection.kept_full_blocks,
        "estimate_quality": plan.projection.estimate_quality,
        "outcome": plan.projection.outcome,
    }


def _transition_display(transition: FrontierTransition) -> dict:
    return {
        "block_idx": transition.block_idx,
        "block_id": transition.block_id,
        "from_mode": transition.from_mode,
        "to_mode": transition.to_mode,
        "tokens_freed": transition.tokens_freed,
        "narrative_chapter": transition.narrative_chapter,
    }


SLEEP_TOOL: Tool[SleepInput] = Tool(
    name="Sleep",
    input_model=SleepInput,
    exec=exec_sleep,
    category="memory",
)
