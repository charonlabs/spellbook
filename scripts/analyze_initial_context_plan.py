"""Analyze initial full-vs-summary context plan candidates.

This is a read-only transplant utility. It loads a prepared core transcript,
renders candidate context views using the real BlockManager projection, and
counts each view against the provider surface.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import rich
from dotenv import load_dotenv
from rich.table import Table

from spellbook.backends.anthropic import AnthropicBackend
from spellbook.backends.model_backend import RequestSurface, TokenCounter
from spellbook.config import SpellbookConfig
from spellbook.footer import FooterController
from spellbook.fork import ForkRunner
from spellbook.homunculus.block_manager import BlockManager
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.homunculus.tool_result_ttl import ToolResultTTLRegistry
from spellbook.ir_types import IRBlock, IRSemanticBlock, SemanticBlockMode
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.registry import ToolRegistry

DEFAULT_ENV_PATH = Path.home() / ".chorus" / ".env"


class _NoopTokenCounter:
    async def count_block_content(self, block: IRBlock) -> int | None:
        return None

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return None

    async def count_frame(self) -> int | None:
        return None

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return None


@dataclass(frozen=True)
class InitialContextPlanCandidate:
    full_suffix_blocks: int
    summary_blocks: int
    rendered_blocks: int
    full_start_idx: int | None
    full_end_idx: int | None
    full_start_title: str | None
    full_end_title: str | None
    tokens: int | None
    delta_tokens: int | None
    threshold_slack: int | None
    status: str

    @property
    def over_threshold(self) -> bool | None:
        if self.tokens is None:
            return None
        return self.threshold_slack is not None and self.threshold_slack < 0

    def to_dict(self) -> dict[str, object]:
        return {
            "full_suffix_blocks": self.full_suffix_blocks,
            "summary_blocks": self.summary_blocks,
            "rendered_blocks": self.rendered_blocks,
            "full_start_idx": self.full_start_idx,
            "full_end_idx": self.full_end_idx,
            "full_start_title": self.full_start_title,
            "full_end_title": self.full_end_title,
            "tokens": self.tokens,
            "delta_tokens": self.delta_tokens,
            "threshold_slack": self.threshold_slack,
            "over_threshold": self.over_threshold,
            "status": self.status,
        }


@dataclass(frozen=True)
class InitialContextPlanReport:
    transcript_path: Path
    session_id: str
    model: str
    threshold: int
    semantic_blocks: int
    summary_ready_blocks: int
    tail_start: int
    tail_blocks: int
    candidates: list[InitialContextPlanCandidate] = field(default_factory=list)

    @property
    def last_under_threshold(self) -> InitialContextPlanCandidate | None:
        candidates = [
            candidate
            for candidate in self.candidates
            if candidate.tokens is not None and candidate.tokens <= self.threshold
        ]
        return candidates[-1] if candidates else None

    @property
    def first_over_threshold(self) -> InitialContextPlanCandidate | None:
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.tokens is not None and candidate.tokens > self.threshold
            ),
            None,
        )

    @property
    def first_failed(self) -> InitialContextPlanCandidate | None:
        return next(
            (candidate for candidate in self.candidates if candidate.tokens is None),
            None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "session_id": self.session_id,
            "model": self.model,
            "threshold": self.threshold,
            "semantic_blocks": self.semantic_blocks,
            "summary_ready_blocks": self.summary_ready_blocks,
            "tail_start": self.tail_start,
            "tail_blocks": self.tail_blocks,
            "last_under_threshold": self.last_under_threshold.to_dict()
            if self.last_under_threshold is not None
            else None,
            "first_over_threshold": self.first_over_threshold.to_dict()
            if self.first_over_threshold is not None
            else None,
            "first_failed": self.first_failed.to_dict()
            if self.first_failed is not None
            else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


async def analyze_initial_context_plan(
    *,
    transcript_path: Path,
    threshold: int | None = None,
    after_over: int = 2,
    max_full: int | None = None,
    count_all: bool = False,
    token_counter: TokenCounter | None = None,
    model: str | None = None,
) -> InitialContextPlanReport:
    if after_over < 0:
        raise ValueError("after_over must be greater than or equal to zero.")
    if max_full is not None and max_full < 0:
        raise ValueError("max_full must be greater than or equal to zero.")

    transcript_path = transcript_path.expanduser().resolve()
    rehydrated = Rehydrator(transcript_path).run()
    _validate_summary_ready(rehydrated)

    config = rehydrated.config
    if model is not None:
        config = config.model_copy(update={"model": model})
    soft_threshold = (
        threshold if threshold is not None else config.hom_config.soft_threshold
    )
    counter = token_counter or _build_token_counter(config)
    builder = _build_surface_builder(config)

    semantic_count = len(rehydrated.semantic_blocks)
    max_suffix = semantic_count if max_full is None else min(max_full, semantic_count)
    candidates: list[InitialContextPlanCandidate] = []
    previous_tokens: int | None = None
    over_count = 0
    for full_suffix_blocks in range(max_suffix + 1):
        rendered = _render_candidate_blocks(
            rehydrated=rehydrated,
            full_suffix_blocks=full_suffix_blocks,
        )
        surface = builder.build(rendered)
        tokens = await counter.count_surface(surface)
        delta = (
            tokens - previous_tokens
            if tokens is not None and previous_tokens is not None
            else None
        )
        candidate = _candidate_report(
            rehydrated=rehydrated,
            full_suffix_blocks=full_suffix_blocks,
            rendered_blocks=len(rendered),
            tokens=tokens,
            delta_tokens=delta,
            threshold=soft_threshold,
        )
        candidates.append(candidate)
        if tokens is not None:
            previous_tokens = tokens
        if not count_all and tokens is not None and tokens > soft_threshold:
            over_count += 1
            if over_count > after_over:
                break
        if not count_all and tokens is None:
            break

    tail_start = _tail_start(rehydrated.semantic_blocks)
    return InitialContextPlanReport(
        transcript_path=transcript_path,
        session_id=rehydrated.session_id,
        model=config.model,
        threshold=soft_threshold,
        semantic_blocks=semantic_count,
        summary_ready_blocks=sum(
            1 for block in rehydrated.semantic_blocks if _has_summary(block)
        ),
        tail_start=tail_start,
        tail_blocks=len(rehydrated.blocks[tail_start:]),
        candidates=candidates,
    )


def _validate_summary_ready(rehydrated: RehydrationResult) -> None:
    missing = [
        f'{block.idx} "{block.title}"'
        for block in rehydrated.semantic_blocks
        if not _has_summary(block)
    ]
    if missing:
        preview = "\n".join(f"- {line}" for line in missing[:20])
        suffix = "\n..." if len(missing) > 20 else ""
        raise ValueError(
            "Every semantic block must have a summary artifact before initial "
            f"context plan analysis. Missing summaries:\n{preview}{suffix}"
        )


def _has_summary(block: IRSemanticBlock) -> bool:
    return any(artifact.type == "summary" for artifact in block.artifacts)


def _render_candidate_blocks(
    *,
    rehydrated: RehydrationResult,
    full_suffix_blocks: int,
) -> list[IRBlock]:
    manager = _build_block_manager(rehydrated)
    semantic_blocks = _candidate_semantic_blocks(
        rehydrated.semantic_blocks,
        full_suffix_blocks=full_suffix_blocks,
    )
    manager.semantic_blocks = semantic_blocks
    rendered: list[IRBlock] = []
    for block in semantic_blocks:
        rendered.extend(manager.render_block(semantic_block=block))
    rendered.extend(manager.render_tail())
    return _apply_tool_result_ttls(rehydrated, rendered)


def _apply_tool_result_ttls(
    rehydrated: RehydrationResult,
    blocks: list[IRBlock],
) -> list[IRBlock]:
    ttl_registry = ToolResultTTLRegistry(
        config=rehydrated.config.hom_config,
        recorder=cast(Recorder, object()),
    )
    ttl_registry.rehydrate(
        rehydrated.tool_result_ttls,
        last_completed_turn=rehydrated.last_completed_turn,
    )
    return ttl_registry.collapse_blocks(blocks)


def _candidate_semantic_blocks(
    semantic_blocks: list[IRSemanticBlock],
    *,
    full_suffix_blocks: int,
) -> list[IRSemanticBlock]:
    full_start = len(semantic_blocks) - full_suffix_blocks
    candidates: list[IRSemanticBlock] = []
    for idx, block in enumerate(semantic_blocks):
        mode: SemanticBlockMode = "full" if idx >= full_start else "summary"
        candidates.append(_with_mode(block, mode))
    return candidates


def _with_mode(block: IRSemanticBlock, mode: SemanticBlockMode) -> IRSemanticBlock:
    if mode == "summary":
        summary = next(
            artifact for artifact in block.artifacts if artifact.type == "summary"
        )
        available_modes = block.available_modes
        if "summary" not in available_modes:
            available_modes = [*available_modes, "summary"]
        return block.model_copy(
            update={
                "mode": "summary",
                "toks": summary.toks,
                "available_modes": available_modes,
            }
        )
    return block.model_copy(
        update={
            "mode": "full",
            "toks": block.full_toks or block.toks,
        }
    )


def _build_block_manager(rehydrated: RehydrationResult) -> BlockManager:
    manager = BlockManager(
        config=rehydrated.config.hom_config,
        fork_runner=cast(ForkRunner, object()),
        footer_c=cast(FooterController, object()),
        nursery=Nursery(config=rehydrated.config),
        recorder=cast(Recorder, object()),
        token_meter=TokenMeter(
            config=rehydrated.config.hom_config,
            tok_counter=cast(TokenCounter, _NoopTokenCounter()),
        ),
    )
    manager.context_blocks = list(rehydrated.blocks)
    manager.semantic_blocks = list(rehydrated.semantic_blocks)
    manager.next_block_id = len(manager.context_blocks)
    return manager


def _candidate_report(
    *,
    rehydrated: RehydrationResult,
    full_suffix_blocks: int,
    rendered_blocks: int,
    tokens: int | None,
    delta_tokens: int | None,
    threshold: int,
) -> InitialContextPlanCandidate:
    semantic_count = len(rehydrated.semantic_blocks)
    full_start_idx = semantic_count - full_suffix_blocks if full_suffix_blocks else None
    full_end_idx = semantic_count - 1 if full_suffix_blocks else None
    full_start = (
        rehydrated.semantic_blocks[full_start_idx]
        if full_start_idx is not None
        else None
    )
    full_end = (
        rehydrated.semantic_blocks[full_end_idx] if full_end_idx is not None else None
    )
    return InitialContextPlanCandidate(
        full_suffix_blocks=full_suffix_blocks,
        summary_blocks=semantic_count - full_suffix_blocks,
        rendered_blocks=rendered_blocks,
        full_start_idx=full_start_idx,
        full_end_idx=full_end_idx,
        full_start_title=full_start.title if full_start is not None else None,
        full_end_title=full_end.title if full_end is not None else None,
        tokens=tokens,
        delta_tokens=delta_tokens,
        threshold_slack=(threshold - tokens) if tokens is not None else None,
        status="counted" if tokens is not None else "failed",
    )


def _tail_start(semantic_blocks: list[IRSemanticBlock]) -> int:
    if not semantic_blocks:
        return 0
    return semantic_blocks[-1].range.end_block + 1


def _build_surface_builder(config: SpellbookConfig) -> RequestSurfaceBuilder:
    match config.provider:
        case "anthropic":
            backend = AnthropicBackend()
        case _:
            raise NotImplementedError(
                f"Initial context plan analysis does not support provider `{config.provider}` yet."
            )
    registry = ToolRegistry.build(config.tool_categories, surface=config.session_type)
    return RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=registry,
    )


def _build_token_counter(config: SpellbookConfig) -> TokenCounter:
    match config.provider:
        case "anthropic":
            backend = AnthropicBackend()
        case _:
            raise NotImplementedError(
                f"Initial context plan analysis does not support provider `{config.provider}` yet."
            )
    registry = ToolRegistry.build(config.tool_categories, surface=config.session_type)
    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=registry,
    )
    return backend.build_token_counter(config=config, surface_builder=surface_builder)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.analyze_initial_context_plan",
        description=(
            "Render and count initial context plan candidates for a prepared "
            "core replay transcript."
        ),
    )
    parser.add_argument("transcript", type=Path, help="Prepared core transcript JSONL.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Token threshold to inspect around. Defaults to the transcript soft threshold.",
    )
    parser.add_argument(
        "--after-over",
        type=int,
        default=2,
        help="After the first over-threshold candidate, count this many more. Defaults to 2.",
    )
    parser.add_argument(
        "--max-full",
        type=int,
        default=None,
        help="Maximum number of newest semantic blocks to render full.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Count every candidate up to --max-full or all semantic blocks.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the transcript model for count_tokens.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before token counting. Defaults to {DEFAULT_ENV_PATH}.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for a JSON report.",
    )
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)

    report = await analyze_initial_context_plan(
        transcript_path=args.transcript,
        threshold=args.threshold,
        after_over=args.after_over,
        max_full=args.max_full,
        count_all=args.all,
        model=args.model,
    )
    _print_report(report)
    if args.report_json is not None:
        report_path = args.report_json.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rich.print(f"\nReport: {report_path}")


def _print_report(report: InitialContextPlanReport) -> None:
    rich.print("[bold]Initial Context Plan Analysis[/bold]")
    rich.print(f"Transcript: {report.transcript_path}")
    rich.print(f"Session:    {report.session_id}")
    rich.print(f"Model:      {report.model}")
    rich.print(f"Threshold:  {_format_tokens(report.threshold)}")
    rich.print(
        f"Blocks:     {report.semantic_blocks} semantic, "
        f"{report.summary_ready_blocks} summary-ready, {report.tail_blocks} tail"
    )
    rich.print("")

    table = Table(title="Candidates")
    table.add_column("Newest Full", justify="right")
    table.add_column("Full Range")
    table.add_column("Tokens", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Slack", justify="right")
    table.add_column("Rendered", justify="right")
    table.add_column("Status")
    for candidate in report.candidates:
        table.add_row(
            str(candidate.full_suffix_blocks),
            _format_full_range(candidate),
            _format_optional_tokens(candidate.tokens),
            _format_delta(candidate.delta_tokens),
            _format_slack(candidate.threshold_slack),
            str(candidate.rendered_blocks),
            candidate.status,
        )
    rich.print(table)

    under = report.last_under_threshold
    over = report.first_over_threshold
    if under is not None:
        rich.print(
            "\n[green]Last under threshold:[/green] "
            f"newest {under.full_suffix_blocks} block(s) full, "
            f"{_format_optional_tokens(under.tokens)}."
        )
    else:
        rich.print("\n[yellow]No counted candidate was under the threshold.[/yellow]")
    if over is not None:
        rich.print(
            "[yellow]First over threshold:[/yellow] "
            f"newest {over.full_suffix_blocks} block(s) full, "
            f"{_format_optional_tokens(over.tokens)}."
        )
    if report.first_failed is not None:
        rich.print(
            "[red]First failed count:[/red] "
            f"newest {report.first_failed.full_suffix_blocks} block(s) full."
        )


def _format_full_range(candidate: InitialContextPlanCandidate) -> str:
    if candidate.full_start_idx is None or candidate.full_end_idx is None:
        return "(none)"
    if candidate.full_start_idx == candidate.full_end_idx:
        return f"[{candidate.full_start_idx}] {candidate.full_start_title}"
    return (
        f"[{candidate.full_start_idx}-{candidate.full_end_idx}] "
        f"{candidate.full_start_title} -> {candidate.full_end_title}"
    )


def _format_tokens(tokens: int) -> str:
    return f"{tokens:,}"


def _format_optional_tokens(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    return _format_tokens(tokens)


def _format_delta(delta: int | None) -> str:
    if delta is None:
        return "-"
    sign = "+" if delta >= 0 else "-"
    return f"{sign}{abs(delta):,}"


def _format_slack(slack: int | None) -> str:
    if slack is None:
        return "unknown"
    sign = "+" if slack >= 0 else "-"
    return f"{sign}{abs(slack):,}"


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()
