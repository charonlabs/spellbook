"""Preview core summary rendering before and after pinning a facet.

Copies the input transcript before appending a facet pin, so the source
transcript is never mutated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from scripts.repair_session_skill_catalog import repair_session_skill_catalog
from spellbook.backends.anthropic import AnthropicBackend
from spellbook.backends.model_backend import TokenCounter
from spellbook.config import SpellbookConfig
from spellbook.footer import FooterController
from spellbook.fork import ForkRunner
from spellbook.homunculus.block_manager import BlockManager
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRImageBlock,
    IRSemanticBlock,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)
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

    async def count_surface(self, surface: object) -> int | None:
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.facet_pin_render",
        description=(
            "Copy a core transcript, render a summary block before and after "
            "pinning one facet, and validate the rendered IR with Anthropic count_tokens."
        ),
    )
    parser.add_argument("transcript", type=Path, help="Core transcript JSONL path.")
    parser.add_argument("facet_id", help="Facet id to pin.")
    parser.add_argument(
        "--block-idx",
        type=int,
        default=None,
        help="Optional block idx if the facet id appears in multiple blocks.",
    )
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Where to write the copied transcript. Defaults to a temp path.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model slug for Anthropic count_tokens. Defaults to transcript config model.",
    )
    parser.add_argument(
        "--reason",
        default="Manual facet pin render validation.",
        help="Reason stored on the copied transcript's facet pin record.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before token counting. Defaults to {DEFAULT_ENV_PATH}.",
    )
    parser.add_argument(
        "--skip-count",
        action="store_true",
        help="Print renders without calling Anthropic count_tokens.",
    )
    parser.add_argument(
        "--max-block-chars",
        type=int,
        default=12_000,
        help="Max characters printed per rendered IR block. Use 0 for no truncation.",
    )
    return parser.parse_args(argv)


async def main_async(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)

    source_path = args.transcript.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Transcript not found: {source_path}")

    copy_path = _copy_transcript(source_path, args.copy_to)
    rehydrated = Rehydrator(copy_path).run()
    match = _find_facet(
        rehydrated,
        facet_id=args.facet_id,
        block_idx=args.block_idx,
    )
    block, facet_title = match

    unpinned_block = _as_summary_without_target_facet_pin(block, args.facet_id)
    unpinned_render = _render_block(
        rehydrated=rehydrated,
        transcript_path=copy_path,
        block=unpinned_block,
    )

    _print_header(source_path, copy_path, block, args.facet_id, facet_title)
    _print_render(
        "Unpinned Summary Render",
        unpinned_render,
        max_block_chars=args.max_block_chars,
    )

    counter: TokenCounter | None = None
    count_config = rehydrated.config
    if args.model is not None:
        count_config = count_config.model_copy(update={"model": args.model})
    if not args.skip_count:
        counter = _build_token_counter(count_config)
        await _print_count("Unpinned count_blocks", counter, unpinned_render)

    pinned_rehydrated = _append_pin_and_rehydrate(
        copy_path=copy_path,
        rehydrated=rehydrated,
        block=block,
        facet_id=args.facet_id,
        reason=args.reason,
    )
    pinned_block = _semantic_block_by_id(pinned_rehydrated, block.id)
    pinned_block = _as_summary_block(pinned_block)
    pinned_render = _render_block(
        rehydrated=pinned_rehydrated,
        transcript_path=copy_path,
        block=pinned_block,
    )
    _print_render(
        "Pinned Summary Render",
        pinned_render,
        max_block_chars=args.max_block_chars,
    )
    if counter is not None:
        await _print_count("Pinned count_blocks", counter, pinned_render)


def _copy_transcript(source_path: Path, copy_to: Path | None) -> Path:
    if copy_to is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="spellbook_facet_pin_"))
        copy_path = temp_dir / source_path.name
    else:
        copy_path = copy_to.expanduser().resolve()
        if copy_path == source_path:
            raise ValueError("--copy-to must not be the source transcript path.")
        copy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, copy_path)
    repair_session_skill_catalog(copy_path, backup=False, write_report=False)
    return copy_path


def _find_facet(
    rehydrated: RehydrationResult,
    *,
    facet_id: str,
    block_idx: int | None,
) -> tuple[IRSemanticBlock, str]:
    matches: list[tuple[IRSemanticBlock, str]] = []
    for block in rehydrated.semantic_blocks:
        if block_idx is not None and block.idx != block_idx:
            continue
        summary = next((a for a in block.artifacts if a.type == "summary"), None)
        if summary is None:
            continue
        for facet in summary.facets:
            if facet.id == facet_id:
                matches.append((block, facet.title))

    if not matches:
        candidates = _facet_candidates(rehydrated)
        raise ValueError(
            f'Facet id "{facet_id}" was not found in summary artifacts.\n\n'
            f"Available facets:\n{candidates}"
        )
    if len(matches) > 1:
        locations = ", ".join(f"block {block.idx}" for block, _ in matches)
        raise ValueError(
            f'Facet id "{facet_id}" appears in multiple blocks: {locations}. '
            "Pass --block-idx to disambiguate."
        )
    return matches[0]


def _facet_candidates(rehydrated: RehydrationResult) -> str:
    lines: list[str] = []
    for block in rehydrated.semantic_blocks:
        summary = next((a for a in block.artifacts if a.type == "summary"), None)
        if summary is None:
            continue
        for facet in summary.facets:
            lines.append(
                f'- block {block.idx} "{block.title}": {facet.id} '
                f"({facet.start_block}-{facet.end_block}) - {facet.title}"
            )
    return "\n".join(lines) if lines else "(no summary facets found)"


def _as_summary_without_target_facet_pin(
    block: IRSemanticBlock,
    facet_id: str,
) -> IRSemanticBlock:
    summary_block = _as_summary_block(block)
    return summary_block.model_copy(
        update={
            "facet_pins": [
                pin for pin in summary_block.facet_pins if pin.facet_id != facet_id
            ]
        }
    )


def _as_summary_block(block: IRSemanticBlock) -> IRSemanticBlock:
    summary = next((a for a in block.artifacts if a.type == "summary"), None)
    if summary is None:
        raise ValueError(f"Block {block.idx} has no summary artifact.")
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


def _append_pin_and_rehydrate(
    *,
    copy_path: Path,
    rehydrated: RehydrationResult,
    block: IRSemanticBlock,
    facet_id: str,
    reason: str,
) -> RehydrationResult:
    if not any(pin.facet_id == facet_id for pin in block.facet_pins):
        manager, _ = _build_block_manager(
            rehydrated=rehydrated,
            transcript_path=copy_path,
        )
        manager.pin_facet(block.idx, facet_id, reason)
    return Rehydrator(copy_path).run()


def _build_block_manager(
    *,
    rehydrated: RehydrationResult,
    transcript_path: Path,
) -> tuple[BlockManager, Recorder]:
    tool_registry = ToolRegistry.build(
        rehydrated.config.tool_categories,
        surface=rehydrated.config.session_type,
    )
    recorder = Recorder(
        rehydrated.config,
        transcript_path,
        rehydrated.session_id,
        tool_registry,
    )
    recorder.set_state(
        rehydrated.current_turn_id or "manual_facet_pin",
        rehydrated.in_progress_turn or rehydrated.last_completed_turn,
        (rehydrated.last_seq or 0) + 1,
    )
    manager = BlockManager(
        config=rehydrated.config.hom_config,
        fork_runner=cast(ForkRunner, object()),
        footer_c=cast(FooterController, object()),
        nursery=Nursery(config=rehydrated.config),
        recorder=recorder,
        token_meter=TokenMeter(
            config=rehydrated.config.hom_config,
            tok_counter=cast(TokenCounter, _NoopTokenCounter()),
        ),
    )
    manager.context_blocks = list(rehydrated.blocks)
    manager.semantic_blocks = list(rehydrated.semantic_blocks)
    manager.next_block_id = len(manager.context_blocks)
    return manager, recorder


def _semantic_block_by_id(
    rehydrated: RehydrationResult,
    block_id: str,
) -> IRSemanticBlock:
    block = next((b for b in rehydrated.semantic_blocks if b.id == block_id), None)
    if block is None:
        raise ValueError(f'Rehydrated copy no longer contains block "{block_id}".')
    return block


def _render_block(
    *,
    rehydrated: RehydrationResult,
    transcript_path: Path,
    block: IRSemanticBlock,
) -> list[IRBlock]:
    manager, _ = _build_block_manager(
        rehydrated=rehydrated,
        transcript_path=transcript_path,
    )
    return manager.render_block(semantic_block=block)


def _build_token_counter(config: SpellbookConfig) -> TokenCounter:
    if config.provider != "anthropic":
        raise NotImplementedError(
            f"Facet render counting only supports Anthropic, got {config.provider}."
        )
    backend = AnthropicBackend()
    tool_registry = ToolRegistry.build(
        config.tool_categories,
        surface=config.session_type,
    )
    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=tool_registry,
    )
    return backend.build_token_counter(config=config, surface_builder=surface_builder)


async def _print_count(
    label: str,
    counter: TokenCounter,
    blocks: list[IRBlock],
) -> None:
    count = await counter.count_blocks(blocks)
    print(f"\n## {label}")
    if count is None:
        print(
            "FAILED: count_blocks returned None. See warning above for provider error."
        )
    else:
        print(f"OK: {count} input tokens")


def _print_header(
    source_path: Path,
    copy_path: Path,
    block: IRSemanticBlock,
    facet_id: str,
    facet_title: str,
) -> None:
    print("# Facet Pin Render Preview")
    print("")
    print(f"- Source transcript: `{source_path}`")
    print(f"- Copied transcript: `{copy_path}`")
    print(
        f'- Block: [{block.idx}] "{block.title}" ({block.range.start_block}-{block.range.end_block})'
    )
    print(f'- Facet: `{facet_id}` - "{facet_title}"')


def _print_render(
    title: str,
    blocks: list[IRBlock],
    *,
    max_block_chars: int,
) -> None:
    print("")
    print(f"## {title}")
    print("")
    print(f"Rendered IR blocks: {len(blocks)}")
    for idx, block in enumerate(blocks):
        print("")
        print(f"### Rendered block {idx}: `{block.type}`")
        print("")
        print("```")
        print(_truncate(_format_block(block), max_block_chars))
        print("```")


def _format_block(block: IRBlock) -> str:
    match block:
        case IRUserTextBlock():
            return f"role=user origin={block.origin}\n\n{block.text}"
        case IRAssistantTextBlock():
            return f"role=assistant\n\n{block.text}"
        case IRThinkingBlock():
            return (
                f"role=assistant thinking signature={block.signature}\n\n{block.text}"
            )
        case IRToolCallBlock():
            return (
                f"role=assistant tool_call tool={block.tool} call_id={block.call_id}\n\n"
                f"{json.dumps(block.input, indent=2, sort_keys=True)}"
            )
        case IRToolResultBlock():
            parts = [
                f"role=user tool_result tool={block.tool} call_id={block.call_id} is_error={block.is_error}"
            ]
            for content in block.content:
                match content:
                    case IRToolTextBlock():
                        parts.append(content.text)
                    case IRImageBlock():
                        parts.append(_format_image(content))
            return "\n\n".join(parts)
        case IRImageBlock():
            return f"role=user image\n\n{_format_image(block)}"


def _format_image(block: IRImageBlock) -> str:
    return block.model_dump_json(indent=2)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    remaining = len(text) - max_chars
    return f"{text[:max_chars]}\n\n... truncated {remaining} chars ..."


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
