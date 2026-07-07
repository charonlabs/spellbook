"""Count merge-pass context stats for adjacent semantic blocks.

This script is intentionally read-only. It rehydrates a Spellbook transcript,
renders the same block projections the live context renderer uses, and measures
them with the configured provider token-counting API.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from html import escape as html_escape
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from spellbook.backends import build_backend, infer_provider_for_model
from spellbook.backends.model_backend import RequestSurface, TokenCounter
from spellbook.config import SpellbookConfig
from spellbook.footer import FooterController
from spellbook.fork import ForkRunner
from spellbook.homunculus.block_manager import BlockManager
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.homunculus.tool_result_ttl import ToolResultTTLRegistry
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRImageBase64Source,
    IRImageBlobSource,
    IRImageBlock,
    IRImageURLSource,
    IRSemanticBlock,
    IRSemanticBlockFacet,
    IRSemanticBlockPin,
    IRSemanticBlockSummary,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRTokenRangeCount,
    IRUserTextBlock,
)
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.registry import ToolRegistry

DEFAULT_TRANSCRIPT = (
    Path.home() / ".chorus/spellbook/sessions/meta-claude/transcript.jsonl"
)
DEFAULT_ENV_PATH = Path.home() / ".chorus/.env"
DIRECTOR_OPENING_PATH = Path(
    "/home/rheaton64/code/spellbook-legacy/spellbook/design/playground/director-opening-v1.md"
)


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
class PinCount:
    block_idx: int
    block_title: str
    kind: str
    title: str
    tokens: int
    exact: bool
    method: str
    reason: str
    facet_id: str | None = None


@dataclass(frozen=True)
class BlockLine:
    idx: int
    title: str
    start_block: int
    end_block: int


@dataclass(frozen=True)
class MarkdownPin:
    kind: str
    block_idx: int
    title: str
    reason: str
    facet_id: str | None = None


@dataclass(frozen=True)
class MarkdownPinRange:
    pin: MarkdownPin
    start: int
    end: int


@dataclass(frozen=True)
class MergeStatsReport:
    transcript_path: Path
    model: str
    block_lines: list[BlockLine]
    full_count: IRTokenRangeCount
    summary_count: IRTokenRangeCount
    pins: list[PinCount] = field(default_factory=list)
    render_path: Path | None = None


async def merge_stats(
    *,
    transcript_path: Path = DEFAULT_TRANSCRIPT,
    first_idx: int,
    second_idx: int,
    token_counter: TokenCounter | None = None,
    model: str | None = None,
    render_path: Path | None = None,
    director_opening_path: Path = DIRECTOR_OPENING_PATH,
) -> MergeStatsReport:
    if second_idx != first_idx + 1:
        raise ValueError(
            "merge_stats expects adjacent semantic block indices: N and N+1."
        )

    transcript_path = transcript_path.expanduser().resolve()
    rehydrated = Rehydrator(transcript_path).run()
    config = _count_config(rehydrated.config, model)
    counter = token_counter or _build_token_counter(config)
    manager = _build_block_manager(rehydrated)

    first = _block_by_idx(rehydrated.semantic_blocks, first_idx)
    second = _block_by_idx(rehydrated.semantic_blocks, second_idx)
    blocks = [first, second]

    full_render = _render_full_blocks(rehydrated, blocks)
    full_count = await _count_blocks(
        counter=counter,
        config=config,
        blocks=full_render,
        label="full fidelity blocks",
    )

    summary_render = _apply_tool_result_ttls(
        rehydrated,
        _render_merge_summary_blocks(manager, blocks),
    )
    summary_count = await _count_blocks(
        counter=counter,
        config=config,
        blocks=summary_render,
        label="summary renderings",
    )

    pins = await _count_pins(
        counter=counter,
        config=config,
        rehydrated=rehydrated,
        manager=manager,
        blocks=blocks,
    )

    resolved_render_path: Path | None = None
    if render_path is not None:
        resolved_render_path = _write_markdown_render(
            rehydrated=rehydrated,
            blocks=blocks,
            render_path=render_path,
            director_opening_path=director_opening_path,
        )

    return MergeStatsReport(
        transcript_path=transcript_path,
        model=config.model,
        block_lines=[_block_line(block) for block in blocks],
        full_count=full_count,
        summary_count=summary_count,
        pins=pins,
        render_path=resolved_render_path,
    )


def _count_config(config: SpellbookConfig, model: str | None) -> SpellbookConfig:
    if model is None:
        return config
    return config.model_copy(
        update={"model": model, "provider": infer_provider_for_model(model)}
    )


def _block_by_idx(blocks: list[IRSemanticBlock], idx: int) -> IRSemanticBlock:
    block = next((block for block in blocks if block.idx == idx), None)
    if block is None:
        available = ", ".join(str(block.idx) for block in blocks)
        raise ValueError(f"Semantic block {idx} not found. Available: {available}")
    return block


def _block_line(block: IRSemanticBlock) -> BlockLine:
    return BlockLine(
        idx=block.idx,
        title=_display_title(block),
        start_block=block.range.start_block,
        end_block=block.range.end_block,
    )


def _display_title(block: IRSemanticBlock) -> str:
    summary = _summary_artifact(block)
    if summary is not None:
        return summary.headline
    return block.title


def _summary_artifact(block: IRSemanticBlock) -> IRSemanticBlockSummary | None:
    return next(
        (
            artifact
            for artifact in block.artifacts
            if isinstance(artifact, IRSemanticBlockSummary)
        ),
        None,
    )


def _render_full_blocks(
    rehydrated: RehydrationResult,
    blocks: list[IRSemanticBlock],
) -> list[IRBlock]:
    rendered: list[IRBlock] = []
    for block in blocks:
        rendered.extend(
            rehydrated.blocks[block.range.start_block : block.range.end_block + 1]
        )
    return rendered


def _render_merge_summary_blocks(
    manager: BlockManager,
    blocks: list[IRSemanticBlock],
) -> list[IRBlock]:
    rendered: list[IRBlock] = []
    for block in blocks:
        rendered.extend(
            manager.render_block(semantic_block=_merge_summary_block(block))
        )
    return rendered


def _merge_summary_block(block: IRSemanticBlock) -> IRSemanticBlock:
    if block.pin is not None:
        full_toks = block.full_toks or block.toks
        return block.model_copy(update={"mode": "full", "toks": full_toks})

    summary = _summary_artifact(block)
    if summary is None:
        raise ValueError(f'Block {block.idx} "{block.title}" has no summary artifact.')
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
        config_records=rehydrated.runtime_config_updates,
    )
    return ttl_registry.collapse_blocks(blocks)


async def _count_blocks(
    *,
    counter: TokenCounter,
    config: SpellbookConfig,
    blocks: list[IRBlock],
    label: str,
) -> IRTokenRangeCount:
    meter = TokenMeter(config=config.hom_config, tok_counter=counter)
    count = await meter.count_slice(blocks, 0, len(blocks))
    if count is None:
        raise RuntimeError(f"Token counting failed for {label}.")
    return count


async def _count_pins(
    *,
    counter: TokenCounter,
    config: SpellbookConfig,
    rehydrated: RehydrationResult,
    manager: BlockManager,
    blocks: list[IRSemanticBlock],
) -> list[PinCount]:
    pin_counts: list[PinCount] = []
    for block in blocks:
        if block.pin is not None:
            count = await _count_blocks(
                counter=counter,
                config=config,
                blocks=manager.render_block(semantic_block=_merge_summary_block(block)),
                label=f"block pin {block.idx}",
            )
            pin_counts.append(
                _block_pin_count(
                    block=block,
                    pin=block.pin,
                    count=count,
                )
            )
        pin_counts.extend(
            await _count_facet_pins(
                counter=counter,
                config=config,
                rehydrated=rehydrated,
                manager=manager,
                block=block,
            )
        )
    return pin_counts


async def _count_facet_pins(
    *,
    counter: TokenCounter,
    config: SpellbookConfig,
    rehydrated: RehydrationResult,
    manager: BlockManager,
    block: IRSemanticBlock,
) -> list[PinCount]:
    summary = _summary_artifact(block)
    if summary is None or not block.facet_pins:
        return []

    by_id: dict[str, IRSemanticBlockFacet] = {
        facet.id: facet for facet in summary.facets
    }
    counts: list[PinCount] = []
    for pin in block.facet_pins:
        if pin.facet_id is None:
            continue
        facet = by_id.get(pin.facet_id)
        if facet is None:
            continue
        intervals = manager._expanded_pinned_facet_intervals(block, [facet])  # noqa: SLF001
        pinned_blocks: list[IRBlock] = []
        for start, end in intervals:
            pinned_blocks.extend(manager.context_blocks[start : end + 1])
        pinned_blocks = _apply_tool_result_ttls(rehydrated, pinned_blocks)
        count = await _count_blocks(
            counter=counter,
            config=config,
            blocks=pinned_blocks,
            label=f"facet pin {pin.facet_id}",
        )
        counts.append(
            _facet_pin_count(
                block=block,
                pin=pin,
                facet=facet,
                count=count,
            )
        )
    return counts


def _block_pin_count(
    *,
    block: IRSemanticBlock,
    pin: IRSemanticBlockPin,
    count: IRTokenRangeCount,
) -> PinCount:
    return PinCount(
        block_idx=block.idx,
        block_title=_display_title(block),
        kind="block",
        title=_display_title(block),
        tokens=count.tokens,
        exact=count.exact,
        method=count.method,
        reason=pin.reason,
    )


def _write_markdown_render(
    *,
    rehydrated: RehydrationResult,
    blocks: list[IRSemanticBlock],
    render_path: Path,
    director_opening_path: Path,
) -> Path:
    director_path = director_opening_path.expanduser().resolve()
    if not director_path.exists():
        raise FileNotFoundError(f"Director opening not found: {director_path}")

    output_path = render_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_markdown_document(
            director_text=director_path.read_text(encoding="utf-8"),
            rehydrated=rehydrated,
            blocks=blocks,
        ),
        encoding="utf-8",
    )
    return output_path


def _render_markdown_document(
    *,
    director_text: str,
    rehydrated: RehydrationResult,
    blocks: list[IRSemanticBlock],
) -> str:
    manager = _build_block_manager(rehydrated)
    parts: list[str] = [
        director_text.rstrip(),
        "",
        "---",
        "",
        "# Source Blocks",
        "",
    ]
    for block in blocks:
        parts.extend(_render_semantic_block_markdown(rehydrated, manager, block))
    return "\n".join(parts).rstrip() + "\n"


def _render_semantic_block_markdown(
    rehydrated: RehydrationResult,
    manager: BlockManager,
    block: IRSemanticBlock,
) -> list[str]:
    pin_ranges = _markdown_pin_ranges(manager, block)
    parts: list[str] = [
        f"## Block {block.idx}: {_display_title(block)}",
        "",
        f"Context block range: {block.range.start_block}-{block.range.end_block}",
        "",
    ]
    for context_idx in range(block.range.start_block, block.range.end_block + 1):
        for pin_range in _pin_ranges_starting_at(pin_ranges, context_idx):
            parts.append(_pin_open_tag(pin_range.pin))
        section = _render_context_block_markdown(
            rehydrated.blocks[context_idx],
            context_idx=context_idx,
        )
        parts.extend(section)
        for _ in _pin_ranges_ending_at(pin_ranges, context_idx):
            parts.extend(["</pin>", ""])
    return parts


def _markdown_pin_ranges(
    manager: BlockManager,
    block: IRSemanticBlock,
) -> list[MarkdownPinRange]:
    ranges: list[MarkdownPinRange] = []
    if block.pin is not None:
        pin = MarkdownPin(
            kind="block",
            block_idx=block.idx,
            title=_display_title(block),
            reason=block.pin.reason,
        )
        ranges.append(
            MarkdownPinRange(
                pin=pin,
                start=block.range.start_block,
                end=block.range.end_block,
            )
        )

    summary = _summary_artifact(block)
    if summary is None or not block.facet_pins:
        return ranges

    facets = {facet.id: facet for facet in summary.facets}
    for pin in block.facet_pins:
        if pin.facet_id is None:
            continue
        facet = facets.get(pin.facet_id)
        if facet is None:
            continue
        markdown_pin = MarkdownPin(
            kind="facet",
            block_idx=block.idx,
            title=facet.title,
            reason=pin.reason,
            facet_id=facet.id,
        )
        intervals = manager._expanded_pinned_facet_intervals(block, [facet])  # noqa: SLF001
        for start, end in intervals:
            ranges.append(MarkdownPinRange(pin=markdown_pin, start=start, end=end))
    return ranges


def _pin_ranges_starting_at(
    ranges: list[MarkdownPinRange],
    context_idx: int,
) -> list[MarkdownPinRange]:
    starting = [pin_range for pin_range in ranges if pin_range.start == context_idx]
    return sorted(starting, key=lambda pin_range: pin_range.end, reverse=True)


def _pin_ranges_ending_at(
    ranges: list[MarkdownPinRange],
    context_idx: int,
) -> list[MarkdownPinRange]:
    ending = [pin_range for pin_range in ranges if pin_range.end == context_idx]
    return sorted(ending, key=lambda pin_range: pin_range.start, reverse=True)


def _pin_open_tag(pin: MarkdownPin) -> str:
    attrs = {
        "kind": pin.kind,
        "block_idx": str(pin.block_idx),
        "title": pin.title,
        "reason": pin.reason,
    }
    if pin.facet_id is not None:
        attrs["facet_id"] = pin.facet_id
    rendered = " ".join(
        f'{key}="{html_escape(value, quote=True)}"' for key, value in attrs.items()
    )
    return f"<pin {rendered}>"


def _render_context_block_markdown(block: IRBlock, *, context_idx: int) -> list[str]:
    metadata = f"<!-- context_block={context_idx} -->"
    match block:
        case IRUserTextBlock():
            title = {
                "human": "User Message",
                "conduit": "Conduit Message",
                "system": "System Message",
                "memory": "Memory Message",
            }[block.origin]
            return [
                f"### {title} (context block {context_idx})",
                metadata,
                "",
                block.text.rstrip(),
                "",
            ]
        case IRAssistantTextBlock():
            return [
                f"### Assistant Message (context block {context_idx})",
                metadata,
                "",
                block.text.rstrip(),
                "",
            ]
        case IRThinkingBlock():
            return [
                f"### Thinking (context block {context_idx})",
                metadata,
                "",
                _fenced(block.text, "text"),
                "",
            ]
        case IRToolCallBlock():
            return [
                f"### Tool Call: {block.tool} (context block {context_idx})",
                metadata,
                "",
                f"call_id: `{block.call_id}`",
                "",
                _fenced(json.dumps(block.input, indent=2, sort_keys=True), "json"),
                "",
            ]
        case IRToolResultBlock():
            return _render_tool_result_markdown(block, context_idx, metadata)
        case IRImageBlock():
            return [
                f"### Image ({block.origin}) (context block {context_idx})",
                metadata,
                "",
                _render_image_markdown(block),
                "",
            ]


def _render_tool_result_markdown(
    block: IRToolResultBlock,
    context_idx: int,
    metadata: str,
) -> list[str]:
    parts = [
        f"### Tool Result: {block.tool} (context block {context_idx})",
        metadata,
        "",
        f"call_id: `{block.call_id}`",
        "",
    ]
    if not block.content:
        parts.extend(["(no content)", ""])
        return parts

    for item_idx, item in enumerate(block.content, start=1):
        match item:
            case IRToolTextBlock():
                title = "Text" if len(block.content) == 1 else f"Text {item_idx}"
                parts.extend(
                    [
                        f"#### {title}",
                        "",
                        _fenced(item.text, "text"),
                        "",
                    ]
                )
            case IRImageBlock():
                title = "Image" if len(block.content) == 1 else f"Image {item_idx}"
                parts.extend(
                    [
                        f"#### {title}",
                        "",
                        _render_image_markdown(item),
                        "",
                    ]
                )
    return parts


def _render_image_markdown(block: IRImageBlock) -> str:
    source = block.source
    if isinstance(source, IRImageURLSource):
        return f"- origin: `{block.origin}`\n- source: `{source.url}`"
    if isinstance(source, IRImageBlobSource):
        return f"- origin: `{block.origin}`\n- blob_path: `{block.blob_path}`"
    if isinstance(source, IRImageBase64Source):
        return (
            f"- origin: `{block.origin}`\n"
            f"- source: base64 `{source.media_type}` "
            f"({len(source.data):,} chars)"
        )
    raise TypeError(f"Unsupported image source: {type(source)}")


def _fenced(text: str, language: str) -> str:
    fence_len = 3
    while "`" * fence_len in text:
        fence_len += 1
    fence = "`" * fence_len
    return f"{fence}{language}\n{text.rstrip()}\n{fence}"


def _facet_pin_count(
    *,
    block: IRSemanticBlock,
    pin: IRSemanticBlockPin,
    facet: IRSemanticBlockFacet,
    count: IRTokenRangeCount,
) -> PinCount:
    return PinCount(
        block_idx=block.idx,
        block_title=_display_title(block),
        kind="facet",
        title=facet.title,
        tokens=count.tokens,
        exact=count.exact,
        method=count.method,
        reason=pin.reason,
        facet_id=facet.id,
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


def _build_surface_builder(config: SpellbookConfig) -> RequestSurfaceBuilder:
    backend = build_backend(config)
    registry = ToolRegistry.build(
        config.tool_categories,
        surface=config.profile.tool_surface,
    )
    return RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=registry,
    )


def _build_token_counter(config: SpellbookConfig) -> TokenCounter:
    backend = build_backend(config)
    registry = ToolRegistry.build(
        config.tool_categories,
        surface=config.profile.tool_surface,
    )
    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=registry,
    )
    return backend.build_token_counter(config=config, surface_builder=surface_builder)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="merge_stats",
        description="Count full and summary tokens for a merge pair.",
    )
    parser.add_argument("first_idx", type=int, help="First semantic block index.")
    parser.add_argument("second_idx", type=int, help="Second semantic block index.")
    parser.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_TRANSCRIPT,
        help=f"Transcript JSONL path. Defaults to {DEFAULT_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the transcript model for token counting.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before token counting. Defaults to {DEFAULT_ENV_PATH}.",
    )
    parser.add_argument(
        "--render",
        nargs="?",
        default=None,
        const=True,
        help=(
            "Also write a markdown render of the full-fidelity source blocks. "
            "Pass a path, or omit the value to write ./blocks-N-N+1.md."
        ),
    )
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)

    report = await merge_stats(
        transcript_path=args.transcript,
        first_idx=args.first_idx,
        second_idx=args.second_idx,
        model=args.model,
        render_path=_resolve_render_path(args.render, args.first_idx, args.second_idx),
    )
    _print_report(report)


def _print_report(report: MergeStatsReport) -> None:
    first, second = report.block_lines
    print(f"Merge pair: Block {first.idx} + Block {second.idx}")
    for block in report.block_lines:
        print(
            f'Block {block.idx}: "{_clip(block.title)}" '
            f"(turns {block.start_block}-{block.end_block})"
        )
    print("")
    print(f"Full fidelity (both): {_format_count(report.full_count)}")
    print(
        f"Summaries (both):      {_format_count(report.summary_count)} (target ceiling)"
    )
    if report.render_path is not None:
        print(f"Rendered markdown:     {report.render_path}")
    print("")
    _print_pins(report.pins)


def _resolve_render_path(
    value: object,
    first_idx: int,
    second_idx: int,
) -> Path | None:
    if value is None:
        return None
    if value is True:
        return Path(f"blocks-{first_idx}-{second_idx}.md")
    return Path(str(value))


def _print_pins(pins: list[PinCount]) -> None:
    if not pins:
        print("Pins: none")
        return
    print(f"Pins: {len(pins)}")
    for pin in pins:
        qualifier = "full block" if pin.kind == "block" else f"facet {pin.facet_id}"
        exact = "" if pin.exact else f" ({pin.method})"
        print(
            f'- Block {pin.block_idx} {qualifier}: "{_clip(pin.title)}" - '
            f"{_format_tokens(pin.tokens)} tokens{exact}"
        )


def _format_count(count: IRTokenRangeCount) -> str:
    suffix = "" if count.exact else f" ({count.method})"
    return f"{_format_tokens(count.tokens)} tokens{suffix}"


def _format_tokens(tokens: int) -> str:
    return f"{tokens:,}"


def _clip(value: str, limit: int = 72) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()
