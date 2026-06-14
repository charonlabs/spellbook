"""Compile narrative dream chapters into Spellbook IR memory blocks."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core_scripts.dream_director import DEFAULT_DREAM_REVIEW_DIR, DEFAULT_ENV_PATH, Pair
from core_scripts.dream_merge import DEFAULT_DREAM_CHAPTER_DIR, _progress
from core_scripts.merge_stats import (
    DEFAULT_TRANSCRIPT,
    _apply_tool_result_ttls,
    _block_by_idx,
    _build_block_manager,
    _build_token_counter,
    _count_blocks,
    _count_config,
    _markdown_pin_ranges,
    _render_context_block_markdown,
)
from spellbook.backends import infer_provider_for_model
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.custom import CustomSurface
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
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.sdk import Spell
from spellbook.tools.common import Tool, ToolExecutionResult, ToolMetadata

DEFAULT_COMPILE_MODEL = "claude-opus-4-6"
DEFAULT_SANITY_MODEL = "claude-sonnet-4-6"
DEFAULT_WORK_ROOT = Path("/tmp/spellbook-dream-compiler")

OPENING_MEMORY_TEXT = """\
<spellbook-memory source="dream_chapter" fidelity="narrative">
The following is narrative memory reconstructed from prior conversation. It is
not an exact transcript. Quoted exchanges and tool events are curated to preserve
continuity, meaning, and load-bearing context; bracketed passages are
retrospective marginalia.
</spellbook-memory>
"""

SANITY_SYSTEM_PROMPT = """\
You sanity check compiled narrative memory chapters before they are applied to a
Spellbook transcript.

The user message contains a markdown rendering of compiled IR blocks. Check:
- Does it read as coherent conversation?
- Are message boundaries coherent, with no obviously broken role transitions?
- Does every tool call have a matching tool result?
- Is anything malformed, truncated, garbled, or impossible to render as memory?

Use the SubmitSanity tool exactly once. Do not answer only in prose.
"""

BeatKind = Literal["assistant_text", "user_text", "tool_call", "tool_result", "pin"]
CompileStatus = Literal["compiled", "failed"]
SanityStatus = Literal["pass", "fail", "skipped", "error"]
TokenCountFn = Callable[[list[IRBlock]], Awaitable[int]]
SanityRunner = Callable[["CompiledChapter"], Awaitable["SanityRunResult"]]

_DIALOGUE_RE = re.compile(
    r"^(Ryan|User|Claude|Assistant|Meta-Claude|meta-Claude):\s*(.*)$"
)
_TOOL_CALL_RE = re.compile(r"^\s*->\s*(.+?)\s*$|^\s*\u2192\s*(.+?)\s*$")
_TOOL_RESULT_RE = re.compile(r"^\s*<-\s*(.+?)\s*$|^\s*\u2190\s*(.+?)\s*$")
_PIN_RE = re.compile(r"^<!--\s*pin:\s*(.*?)\s*-->\s*$", re.IGNORECASE)
_CALL_STYLE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\((.*)\)$")
_COLON_STYLE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$")
_DASH_SPLIT_RE = re.compile(r"\s+(?:-|--|\u2014)\s+")
_CHAPTER_RE = re.compile(r"chapter-(\d+)-.*\.md$")
_TOOLISH_NAMES = {
    "Bash",
    "BashOutput",
    "Browse",
    "Configure",
    "Dive",
    "Edit",
    "Forget",
    "ForgetToolResult",
    "Glob",
    "Grep",
    "Kill",
    "LS",
    "Load",
    "MultiEdit",
    "Read",
    "Recall",
    "Reflect",
    "ReflectToolResults",
    "Skill",
    "Task",
    "TodoWrite",
    "ToolSearch",
    "WebRead",
    "WebSearch",
    "Write",
}


class SanityCheckInput(BaseModel):
    """Submit a compiled-chapter sanity check."""

    verdict: Literal["pass", "fail"] = Field(
        description="Use pass only if the compiled rendering is coherent."
    )
    notes: str = Field(description="Concise notes for a human reviewer.")
    boundary_notes: str = Field(description="Notes on role/message boundaries.")
    tool_pairing_notes: str = Field(description="Notes on tool call/result pairing.")
    malformed_items: list[str] = Field(
        description="Specific malformed, truncated, or garbled items; empty if clean."
    )


@dataclass(frozen=True)
class SanityRunResult:
    review: SanityCheckInput
    transcript_path: Path | None = None


@dataclass(frozen=True)
class MarkdownToken:
    text: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class ChapterBeat:
    kind: BeatKind
    line_start: int
    line_end: int
    source: str
    text: str | None = None
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    call_id: str | None = None
    marker: str | None = None


@dataclass(frozen=True)
class ExpandedPin:
    marker: str
    title: str
    block_idx: int
    kind: str
    start_context_block: int
    end_context_block: int
    inserted_at_beat: int
    inserted_blocks: int
    facet_id: str | None = None


@dataclass(frozen=True)
class SourcePin:
    title: str
    block_idx: int
    kind: str
    start_context_block: int
    end_context_block: int
    blocks: tuple[IRBlock, ...]
    facet_id: str | None = None


@dataclass(frozen=True)
class SourceContext:
    transcript_path: Path
    rehydrated: RehydrationResult

    @classmethod
    def from_transcript(cls, transcript_path: Path) -> "SourceContext":
        resolved = transcript_path.expanduser().resolve()
        return cls(transcript_path=resolved, rehydrated=Rehydrator(resolved).run())

    def pins_for_pair(self, pair: Pair) -> tuple[SourcePin, ...]:
        manager = _build_block_manager(self.rehydrated)
        pins: list[SourcePin] = []
        for block_idx in (pair.first, pair.second):
            block = _block_by_idx(self.rehydrated.semantic_blocks, block_idx)
            for pin_range in _markdown_pin_ranges(manager, block):
                pins.append(
                    SourcePin(
                        title=pin_range.pin.title,
                        block_idx=pin_range.pin.block_idx,
                        kind=pin_range.pin.kind,
                        start_context_block=pin_range.start,
                        end_context_block=pin_range.end,
                        blocks=tuple(
                            self.rehydrated.blocks[pin_range.start : pin_range.end + 1]
                        ),
                        facet_id=pin_range.pin.facet_id,
                    )
                )
        return tuple(pins)

    def semantic_blocks_for_pair(self, pair: Pair) -> tuple[IRSemanticBlock, ...]:
        return (
            _block_by_idx(self.rehydrated.semantic_blocks, pair.first),
            _block_by_idx(self.rehydrated.semantic_blocks, pair.second),
        )


@dataclass(frozen=True)
class CompiledChapter:
    chapter_path: Path
    chapter_number: int
    pair: Pair
    title: str
    beats: tuple[ChapterBeat, ...]
    ir_blocks: tuple[IRBlock, ...]
    rendered_ir_blocks: tuple[IRBlock, ...]
    expanded_pins: tuple[ExpandedPin, ...]
    token_count: int | None
    provider_message_count: int
    output_path: Path
    preview_path: Path
    sanity: SanityCheckInput | None = None
    sanity_status: SanityStatus = "skipped"
    sanity_transcript_path: Path | None = None
    sanity_error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "dream_compiled_chapter.v1",
            "chapter": {
                "number": self.chapter_number,
                "title": self.title,
                "source_path": str(self.chapter_path),
            },
            "target": {
                "pair": [self.pair.first, self.pair.second],
                "semantic_block_indices": [self.pair.first, self.pair.second],
            },
            "parse": {
                "beat_count": len(self.beats),
                "beats": [_beat_json(beat) for beat in self.beats],
            },
            "compiled": {
                "ir_block_count": len(self.ir_blocks),
                "ir_blocks": [_block_json(block) for block in self.ir_blocks],
                "rendered_ir_block_count": len(self.rendered_ir_blocks),
                "provider_message_count": self.provider_message_count,
                "token_count": self.token_count,
                "token_count_scope": "ttl_collapsed_render",
                "expanded_pins": [
                    {
                        "marker": pin.marker,
                        "title": pin.title,
                        "block_idx": pin.block_idx,
                        "kind": pin.kind,
                        "facet_id": pin.facet_id,
                        "start_context_block": pin.start_context_block,
                        "end_context_block": pin.end_context_block,
                        "inserted_at_beat": pin.inserted_at_beat,
                        "inserted_blocks": pin.inserted_blocks,
                    }
                    for pin in self.expanded_pins
                ],
            },
            "sanity": {
                "status": self.sanity_status,
                "review": self.sanity.model_dump(mode="json")
                if self.sanity is not None
                else None,
                "transcript_path": str(self.sanity_transcript_path)
                if self.sanity_transcript_path is not None
                else None,
                "error": self.sanity_error,
            },
            "artifacts": {
                "compiled_json": str(self.output_path),
                "compiled_markdown": str(self.preview_path),
            },
        }


@dataclass(frozen=True)
class CompileErrorDetail:
    message: str
    chapter_path: Path
    line_start: int | None = None
    line_end: int | None = None
    beat_index: int | None = None
    source: str | None = None
    nearby_lines: tuple[str, ...] = ()


class DreamCompileError(Exception):
    def __init__(self, detail: CompileErrorDetail):
        super().__init__(detail.message)
        self.detail = detail


@dataclass(frozen=True)
class ChapterRunResult:
    chapter_path: Path
    status: CompileStatus
    output_path: Path | None = None
    preview_path: Path | None = None
    beat_count: int = 0
    ir_block_count: int = 0
    provider_message_count: int = 0
    token_count: int | None = None
    ceiling_tokens: int | None = None
    savings_tokens: int | None = None
    sanity_status: SanityStatus = "skipped"
    sanity_notes: str = ""
    error: CompileErrorDetail | None = None


@dataclass(frozen=True)
class CompileOptions:
    chapter_dir: Path = DEFAULT_DREAM_CHAPTER_DIR
    transcript_path: Path = DEFAULT_TRANSCRIPT
    review_dir: Path = DEFAULT_DREAM_REVIEW_DIR
    out_dir: Path | None = None
    work_dir: Path = field(default_factory=lambda: _default_work_dir())
    model: str = DEFAULT_COMPILE_MODEL
    sanity_model: str = DEFAULT_SANITY_MODEL
    concurrency: int = 3
    skip_sanity: bool = False
    show_progress: bool = True
    chapters: tuple[int, ...] | None = None


@dataclass(frozen=True)
class CompileRunReport:
    options: CompileOptions
    results: tuple[ChapterRunResult, ...]


@dataclass
class _SanityHolder:
    review: SanityCheckInput | None = None


def parse_chapter_markdown(markdown: str) -> tuple[ChapterBeat, ...]:
    beats: list[ChapterBeat] = []
    call_index = 1
    for token in _merge_open_bracket_tokens(_paragraph_tokens(markdown)):
        pin = _PIN_RE.match(token.text.strip())
        if pin:
            beats.append(
                ChapterBeat(
                    kind="pin",
                    line_start=token.line_start,
                    line_end=token.line_end,
                    source=token.text,
                    marker=pin.group(1).strip(),
                )
            )
            continue

        dialogue = _DIALOGUE_RE.match(token.text.strip())
        if dialogue:
            speaker = dialogue.group(1)
            text = dialogue.group(2).strip()
            kind: BeatKind = (
                "user_text"
                if speaker.casefold() in {"ryan", "user"}
                else "assistant_text"
            )
            beats.append(
                ChapterBeat(
                    kind=kind,
                    line_start=token.line_start,
                    line_end=token.line_end,
                    source=token.text,
                    text=text,
                )
            )
            continue

        tool_call = _TOOL_CALL_RE.match(token.text)
        if tool_call:
            body = (tool_call.group(1) or tool_call.group(2) or "").strip()
            if not _is_tool_call_body(body):
                beats.append(
                    ChapterBeat(
                        kind="assistant_text",
                        line_start=token.line_start,
                        line_end=token.line_end,
                        source=token.text,
                        text=token.text,
                    )
                )
                continue
            name, tool_input = _parse_tool_call_body(body)
            call_id = f"dream_toolu_{call_index:04d}"
            call_index += 1
            beats.append(
                ChapterBeat(
                    kind="tool_call",
                    line_start=token.line_start,
                    line_end=token.line_end,
                    source=token.text,
                    tool_name=name,
                    tool_input=tool_input,
                    call_id=call_id,
                )
            )
            continue

        tool_result = _TOOL_RESULT_RE.match(token.text)
        if tool_result:
            text = (tool_result.group(1) or tool_result.group(2) or "").strip()
            beats.append(
                ChapterBeat(
                    kind="tool_result",
                    line_start=token.line_start,
                    line_end=token.line_end,
                    source=token.text,
                    text=text,
                )
            )
            continue

        beats.append(
            ChapterBeat(
                kind="assistant_text",
                line_start=token.line_start,
                line_end=token.line_end,
                source=token.text,
                text=token.text,
            )
        )
    return tuple(beats)


def compile_beats_to_ir_blocks(
    *,
    beats: tuple[ChapterBeat, ...],
    chapter_number: int,
    pair: Pair,
    source_pins: tuple[SourcePin, ...] = (),
) -> tuple[tuple[IRBlock, ...], tuple[ExpandedPin, ...]]:
    blocks: list[IRBlock] = []
    if chapter_number == 1:
        blocks.append(
            IRUserTextBlock(
                origin="memory", text=_opening_memory_text(chapter_number, pair)
            )
        )
    expanded_pins: list[ExpandedPin] = []
    pending_calls: list[ChapterBeat] = []

    def flush_pending_with(text: str | None = None) -> None:
        nonlocal pending_calls
        if not pending_calls:
            return
        for pending in pending_calls:
            blocks.append(_tool_result_for_call(pending, text))
        pending_calls = []

    for beat_index, beat in enumerate(beats):
        try:
            match beat.kind:
                case "assistant_text":
                    flush_pending_with()
                    blocks.append(IRAssistantTextBlock(text=beat.text or ""))
                case "user_text":
                    flush_pending_with()
                    blocks.append(
                        IRUserTextBlock(origin="memory", text=beat.text or "")
                    )
                case "tool_call":
                    blocks.append(
                        IRToolCallBlock(
                            call_id=beat.call_id or f"dream_toolu_{beat_index:04d}",
                            tool=beat.tool_name or "Tool",
                            input=beat.tool_input or {},
                        )
                    )
                    pending_calls.append(beat)
                case "tool_result":
                    if pending_calls:
                        flush_pending_with(beat.text or "")
                    else:
                        blocks.append(
                            IRAssistantTextBlock(text=f"Tool result: {beat.text or ''}")
                        )
                case "pin":
                    flush_pending_with()
                    source_pin = _match_source_pin(beat.marker or "", source_pins)
                    blocks.extend(source_pin.blocks)
                    expanded_pins.append(
                        ExpandedPin(
                            marker=beat.marker or "",
                            title=source_pin.title,
                            block_idx=source_pin.block_idx,
                            kind=source_pin.kind,
                            facet_id=source_pin.facet_id,
                            start_context_block=source_pin.start_context_block,
                            end_context_block=source_pin.end_context_block,
                            inserted_at_beat=beat_index,
                            inserted_blocks=len(source_pin.blocks),
                        )
                    )
                case _:
                    raise ValueError(f"Unsupported beat kind: {beat.kind}")
        except DreamCompileError as exc:
            if exc.detail.line_start is not None:
                raise
            raise DreamCompileError(
                CompileErrorDetail(
                    message=exc.detail.message,
                    chapter_path=exc.detail.chapter_path,
                    line_start=beat.line_start,
                    line_end=beat.line_end,
                    beat_index=beat_index,
                    source=beat.source,
                )
            ) from exc
        except Exception as exc:
            raise DreamCompileError(
                CompileErrorDetail(
                    message=str(exc),
                    chapter_path=Path("<markdown>"),
                    line_start=beat.line_start,
                    line_end=beat.line_end,
                    beat_index=beat_index,
                    source=beat.source,
                )
            ) from exc

    flush_pending_with()
    _validate_compiled_blocks(blocks)
    return tuple(blocks), tuple(expanded_pins)


async def compile_chapter(
    *,
    chapter_path: Path,
    source_context: SourceContext,
    out_dir: Path | None,
    token_count_fn: TokenCountFn,
    sanity_runner: SanityRunner | None = None,
) -> CompiledChapter:
    chapter_path = chapter_path.expanduser().resolve()
    markdown = chapter_path.read_text(encoding="utf-8")
    chapter_number = _chapter_number_from_path(chapter_path)
    pair = _pair_for_chapter(chapter_number)
    source_pins = source_context.pins_for_pair(pair)
    beats = parse_chapter_markdown(markdown)
    try:
        blocks, expanded_pins = compile_beats_to_ir_blocks(
            beats=beats,
            chapter_number=chapter_number,
            pair=pair,
            source_pins=source_pins,
        )
    except DreamCompileError as exc:
        raise _with_nearby_lines(exc, chapter_path, markdown) from exc

    rendered_blocks = tuple(_rendered_context_blocks(source_context, blocks))
    token_count = await token_count_fn(list(rendered_blocks))
    provider_message_count = _provider_message_count(rendered_blocks)
    title = _chapter_title(markdown, fallback=chapter_path.stem)
    output_path = _compiled_output_path(chapter_path, out_dir=out_dir, suffix=".json")
    preview_path = _compiled_output_path(chapter_path, out_dir=out_dir, suffix=".md")
    compiled = CompiledChapter(
        chapter_path=chapter_path,
        chapter_number=chapter_number,
        pair=pair,
        title=title,
        beats=beats,
        ir_blocks=blocks,
        rendered_ir_blocks=rendered_blocks,
        expanded_pins=expanded_pins,
        token_count=token_count,
        provider_message_count=provider_message_count,
        output_path=output_path,
        preview_path=preview_path,
    )

    sanity_status: SanityStatus = "skipped"
    sanity: SanityCheckInput | None = None
    sanity_error: str | None = None
    sanity_transcript_path: Path | None = None
    if sanity_runner is not None:
        try:
            sanity_result = await sanity_runner(compiled)
            sanity = sanity_result.review
            sanity_transcript_path = sanity_result.transcript_path
            sanity_status = sanity.verdict
        except Exception as exc:  # noqa: BLE001 - record per chapter.
            sanity_status = "error"
            sanity_error = str(exc)

    compiled = CompiledChapter(
        chapter_path=compiled.chapter_path,
        chapter_number=compiled.chapter_number,
        pair=compiled.pair,
        title=compiled.title,
        beats=compiled.beats,
        ir_blocks=compiled.ir_blocks,
        rendered_ir_blocks=compiled.rendered_ir_blocks,
        expanded_pins=compiled.expanded_pins,
        token_count=compiled.token_count,
        provider_message_count=compiled.provider_message_count,
        output_path=compiled.output_path,
        preview_path=compiled.preview_path,
        sanity=sanity,
        sanity_status=sanity_status,
        sanity_transcript_path=sanity_transcript_path,
        sanity_error=sanity_error,
    )
    _write_compiled_artifacts(compiled)
    return compiled


async def run_compile(
    options: CompileOptions,
    *,
    token_count_fn: TokenCountFn | None = None,
    sanity_runner: SanityRunner | None = None,
) -> CompileRunReport:
    _validate_options(options)
    source_context = SourceContext.from_transcript(options.transcript_path)
    chapters = _chapter_paths(options.chapter_dir, options.chapters)
    count_fn = token_count_fn or _default_token_count_fn(
        source_context.rehydrated.config,
        model=options.model,
    )

    if options.skip_sanity:
        runner = None
    elif sanity_runner is not None:
        runner = sanity_runner
    else:
        runner = _DefaultSanityRunner(
            model=options.sanity_model,
            work_dir=options.work_dir.expanduser().resolve(),
        )

    results = await _run_compile_phase(
        chapters,
        options,
        source_context=source_context,
        token_count_fn=count_fn,
        sanity_runner=runner,
    )
    return CompileRunReport(options=options, results=tuple(results))


async def _run_compile_phase(
    chapters: list[Path],
    options: CompileOptions,
    *,
    source_context: SourceContext,
    token_count_fn: TokenCountFn,
    sanity_runner: SanityRunner | None,
) -> list[ChapterRunResult]:
    semaphore = asyncio.Semaphore(options.concurrency)

    async def run_one(idx: int, chapter_path: Path) -> tuple[int, ChapterRunResult]:
        async with semaphore:
            result = await _run_one_compile(
                chapter_path,
                options,
                source_context=source_context,
                token_count_fn=token_count_fn,
                sanity_runner=sanity_runner,
            )
            return idx, result

    return await _run_results_with_progress(
        chapters,
        coros=(run_one(idx, chapter) for idx, chapter in enumerate(chapters)),
        description="Compile chapters",
        show_progress=options.show_progress,
    )


async def _run_one_compile(
    chapter_path: Path,
    options: CompileOptions,
    *,
    source_context: SourceContext,
    token_count_fn: TokenCountFn,
    sanity_runner: SanityRunner | None,
) -> ChapterRunResult:
    try:
        compiled = await compile_chapter(
            chapter_path=chapter_path,
            source_context=source_context,
            out_dir=options.out_dir,
            token_count_fn=token_count_fn,
            sanity_runner=sanity_runner,
        )
        sanity_notes = ""
        if compiled.sanity is not None:
            sanity_notes = compiled.sanity.notes
        elif compiled.sanity_error:
            sanity_notes = compiled.sanity_error
        ceiling_tokens = _review_ceiling_tokens(
            options.review_dir,
            compiled.chapter_number,
        )
        return ChapterRunResult(
            chapter_path=chapter_path,
            status="compiled",
            output_path=compiled.output_path,
            preview_path=compiled.preview_path,
            beat_count=len(compiled.beats),
            ir_block_count=len(compiled.ir_blocks),
            provider_message_count=compiled.provider_message_count,
            token_count=compiled.token_count,
            ceiling_tokens=ceiling_tokens,
            savings_tokens=_savings_tokens(ceiling_tokens, compiled.token_count),
            sanity_status=compiled.sanity_status,
            sanity_notes=sanity_notes,
        )
    except DreamCompileError as exc:
        return ChapterRunResult(
            chapter_path=chapter_path, status="failed", error=exc.detail
        )
    except Exception as exc:  # noqa: BLE001 - keep runner moving.
        return ChapterRunResult(
            chapter_path=chapter_path,
            status="failed",
            error=CompileErrorDetail(message=str(exc), chapter_path=chapter_path),
        )


async def _run_results_with_progress(
    chapters: list[Path],
    *,
    coros: Iterable[Awaitable[tuple[int, ChapterRunResult]]],
    description: str,
    show_progress: bool,
) -> list[ChapterRunResult]:
    tasks = [asyncio.create_task(coro) for coro in coros]
    results: list[ChapterRunResult | None] = [None] * len(chapters)
    if not tasks:
        return []

    if not show_progress:
        for idx, result in await asyncio.gather(*tasks):
            results[idx] = result
        return _completed_results(results)

    with _progress() as progress:
        task_id = progress.add_task(description, total=len(tasks))
        for task in asyncio.as_completed(tasks):
            idx, result = await task
            results[idx] = result
            progress.advance(task_id)
    return _completed_results(results)


def _completed_results(
    results: list[ChapterRunResult | None],
) -> list[ChapterRunResult]:
    completed: list[ChapterRunResult] = []
    for result in results:
        if result is None:
            raise RuntimeError("Compile task completed without a recorded result.")
        completed.append(result)
    return completed


class _DefaultSanityRunner:
    def __init__(self, *, model: str, work_dir: Path):
        self.model = model
        self.work_dir = work_dir

    async def __call__(self, compiled: CompiledChapter) -> SanityRunResult:
        holder = _SanityHolder()
        transcript_path = (
            self.work_dir
            / "sanity-transcripts"
            / f"chapter-{compiled.chapter_number:02d}-{uuid4().hex}.jsonl"
        )
        spell = Spell(
            config=_sanity_config(self.model),
            transcript_path=transcript_path,
            custom_surface=CustomSurface(tools=[_submit_sanity_tool(holder)]),
        )
        result = await spell.once(_sanity_user_message(compiled))
        if holder.review is None:
            raise RuntimeError(
                "Sanity model did not call SubmitSanity; no structured output captured."
            )
        _ = result.text
        return SanityRunResult(review=holder.review, transcript_path=transcript_path)


def _submit_sanity_tool(holder: _SanityHolder) -> Tool[SanityCheckInput]:
    async def submit_sanity(
        meta: ToolMetadata,
        input: SanityCheckInput,
    ) -> ToolExecutionResult:
        holder.review = input
        return ToolExecutionResult(
            content=[IRToolTextBlock(text="Compiled-chapter sanity check recorded.")]
        )

    return Tool(
        name="SubmitSanity",
        input_model=SanityCheckInput,
        exec=submit_sanity,
        category="thinking",
    )


def _sanity_config(model: str) -> SpellbookConfig:
    return SpellbookConfig(
        provider=infer_provider_for_model(model),
        model=model,
        session_type="custom",
        cwd=Path.cwd(),
        max_output_tokens=8_000,
        system_prompt=SANITY_SYSTEM_PROMPT,
        hom_config=HomunculusConfig(detect_interval=10_000),
    )


def _sanity_user_message(compiled: CompiledChapter) -> str:
    rendered = render_compiled_markdown(compiled)
    return f"""\
# Compiled Dream Chapter Sanity Check

Chapter: {compiled.title}
Source path: {compiled.chapter_path}
Block pair: {compiled.pair.first}-{compiled.pair.second}
Compiled IR blocks: {len(compiled.ir_blocks)}
Provider message count: {compiled.provider_message_count}
Token count: {compiled.token_count}

## Rendered Compiled Blocks

{rendered}
"""


def _default_token_count_fn(
    config: SpellbookConfig,
    *,
    model: str,
) -> TokenCountFn:
    count_config = _count_config(config, model)
    counter = _build_token_counter(count_config)

    async def count(blocks: list[IRBlock]) -> int:
        result = await _count_blocks(
            counter=counter,
            config=count_config,
            blocks=blocks,
            label="compiled dream chapter",
        )
        return result.tokens

    return count


def _paragraph_tokens(markdown: str) -> tuple[MarkdownToken, ...]:
    tokens: list[MarkdownToken] = []
    freeform: list[str] = []
    freeform_start: int | None = None

    def flush_freeform(end_line: int) -> None:
        nonlocal freeform, freeform_start
        if not freeform:
            return
        start = freeform_start if freeform_start is not None else end_line
        text = "\n".join(freeform).strip()
        if text:
            tokens.append(MarkdownToken(text=text, line_start=start, line_end=end_line))
        freeform = []
        freeform_start = None

    for line_no, raw_line in enumerate(markdown.splitlines(), start=1):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_freeform(line_no - 1)
            continue
        if _is_special_line(stripped):
            flush_freeform(line_no - 1)
            tokens.append(
                MarkdownToken(text=stripped, line_start=line_no, line_end=line_no)
            )
            continue
        if freeform_start is None:
            freeform_start = line_no
        freeform.append(stripped)

    flush_freeform(len(markdown.splitlines()))
    return tuple(tokens)


def _merge_open_bracket_tokens(
    tokens: tuple[MarkdownToken, ...],
) -> tuple[MarkdownToken, ...]:
    merged: list[MarkdownToken] = []
    pending: list[MarkdownToken] = []
    balance = 0

    for token in tokens:
        if pending:
            pending.append(token)
            balance += _square_bracket_balance(token.text)
            if balance <= 0:
                merged.append(_merge_markdown_tokens(pending))
                pending = []
                balance = 0
            continue

        stripped = token.text.lstrip()
        if stripped.startswith("["):
            balance = _square_bracket_balance(token.text)
            if balance > 0:
                pending = [token]
                continue

        merged.append(token)

    if pending:
        merged.append(_merge_markdown_tokens(pending))
    return tuple(merged)


def _merge_markdown_tokens(tokens: list[MarkdownToken]) -> MarkdownToken:
    if not tokens:
        raise ValueError("Cannot merge an empty token list.")
    return MarkdownToken(
        text="\n\n".join(token.text for token in tokens),
        line_start=tokens[0].line_start,
        line_end=tokens[-1].line_end,
    )


def _square_bracket_balance(text: str) -> int:
    return text.count("[") - text.count("]")


def _is_special_line(stripped: str) -> bool:
    return (
        stripped.startswith("#")
        or _is_tool_call_line(stripped)
        or stripped.startswith("<-")
        or stripped.startswith("\u2190")
        or _PIN_RE.match(stripped) is not None
        or _DIALOGUE_RE.match(stripped) is not None
    )


def _is_tool_call_line(stripped: str) -> bool:
    match = _TOOL_CALL_RE.match(stripped)
    if match is None:
        return False
    body = (match.group(1) or match.group(2) or "").strip()
    return _is_tool_call_body(body)


def _is_tool_call_body(body: str) -> bool:
    call_match = _CALL_STYLE_RE.match(body)
    if call_match:
        return call_match.group(1) in _TOOLISH_NAMES

    colon_match = _COLON_STYLE_RE.match(body)
    if colon_match:
        return colon_match.group(1) in _TOOLISH_NAMES

    return False


def _parse_tool_call_body(body: str) -> tuple[str, dict[str, Any]]:
    match = _CALL_STYLE_RE.match(body)
    if match:
        name = match.group(1)
        return name, _parse_call_style_args(name, match.group(2))

    match = _COLON_STYLE_RE.match(body)
    if match:
        name = match.group(1)
        return name, _parse_colon_style_args(name, match.group(2))

    parts = body.split(maxsplit=1)
    name = parts[0] if parts else "Tool"
    tool_input = {"value": parts[1]} if len(parts) == 2 else {}
    return name, tool_input


def _parse_call_style_args(name: str, raw_args: str) -> dict[str, Any]:
    try:
        expr = ast.parse(f"_f({raw_args})", mode="eval").body
    except SyntaxError:
        expr = None

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    if isinstance(expr, ast.Call):
        args = [_literal_or_text(arg) for arg in expr.args]
        kwargs = {
            str(keyword.arg): _literal_or_text(keyword.value)
            for keyword in expr.keywords
            if keyword.arg
        }

    if kwargs:
        return kwargs
    if len(args) == 1:
        key = "prompt" if name in {"Dive", "Task"} else "value"
        return {key: args[0]}
    if args:
        return {"args": args}
    if raw_args.strip():
        return {"raw": raw_args.strip()}
    return {}


def _literal_or_text(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except Exception:
        if hasattr(ast, "unparse"):
            return ast.unparse(node)
        return ""


def _parse_colon_style_args(name: str, body: str) -> dict[str, Any]:
    body = body.strip()
    parts = _DASH_SPLIT_RE.split(body, maxsplit=1)
    head = parts[0].strip()
    tail = parts[1].strip() if len(parts) == 2 else ""
    if name == "Bash":
        payload = {"command": head}
    elif name in {"Read", "Write", "Edit"}:
        payload = {"file_path": head}
    elif name in {"Browse", "WebRead"}:
        payload = {"target": head}
    else:
        payload = {"target": head} if head else {}
    if tail:
        payload["description"] = tail
    return payload


def _tool_result_for_call(
    beat: ChapterBeat, text: str | None = None
) -> IRToolResultBlock:
    return IRToolResultBlock(
        call_id=beat.call_id or "dream_toolu_missing",
        tool=beat.tool_name or "Tool",
        content=[IRToolTextBlock(text=text or _synthesized_tool_result_text(beat))],
    )


def _synthesized_tool_result_text(beat: ChapterBeat) -> str:
    name = beat.tool_name or "Tool"
    payload = beat.tool_input or {}
    if name == "Read":
        target = payload.get("file_path") or payload.get("target")
        return f"Read completed: {target}" if target else "Read completed."
    if name == "Bash":
        command = payload.get("command")
        return f"Command completed: {command}" if command else "Command completed."
    if name in {"Write", "Edit"}:
        target = payload.get("file_path") or payload.get("target")
        return f"Applied changes to {target}." if target else "Applied changes."
    if name == "Browse":
        target = payload.get("target") or payload.get("value")
        return (
            f"Browser observation captured: {target}"
            if target
            else "Browser observation captured."
        )
    return f"{name} completed."


def _match_source_pin(marker: str, source_pins: tuple[SourcePin, ...]) -> SourcePin:
    if not source_pins:
        raise DreamCompileError(
            CompileErrorDetail(
                message=f'Pin marker "{marker}" found, but source block pair has no pins.',
                chapter_path=Path("<markdown>"),
                source=f"<!-- pin: {marker} -->",
            )
        )
    marker_norm = _normalize_title(marker)
    exact = [pin for pin in source_pins if _normalize_title(pin.title) == marker_norm]
    if len(exact) == 1:
        return exact[0]
    contains = [
        pin
        for pin in source_pins
        if marker_norm in _normalize_title(pin.title)
        or _normalize_title(pin.title) in marker_norm
    ]
    if len(contains) == 1:
        return contains[0]
    available = ", ".join(f'"{pin.title}"' for pin in source_pins)
    if not contains and not exact:
        message = f'Pin marker "{marker}" did not match source pins: {available}'
    else:
        message = f'Pin marker "{marker}" matched multiple source pins: {available}'
    raise DreamCompileError(
        CompileErrorDetail(
            message=message,
            chapter_path=Path("<markdown>"),
            source=f"<!-- pin: {marker} -->",
        )
    )


def _normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _validate_compiled_blocks(blocks: list[IRBlock]) -> None:
    if not blocks:
        raise ValueError("Compiled block list is empty.")

    open_calls: dict[str, str] = {}
    seen_calls: set[str] = set()
    for idx, block in enumerate(blocks):
        if isinstance(block, IRToolCallBlock):
            if block.call_id in seen_calls:
                raise ValueError(
                    f"Duplicate tool call id {block.call_id} at block {idx}."
                )
            seen_calls.add(block.call_id)
            open_calls[block.call_id] = block.tool
        elif isinstance(block, IRToolResultBlock):
            if block.call_id not in seen_calls:
                raise ValueError(
                    f"Tool result {block.call_id} at block {idx} has no prior call."
                )
            open_calls.pop(block.call_id, None)
    if open_calls:
        missing = ", ".join(sorted(open_calls))
        raise ValueError(f"Tool calls without matching results: {missing}")


def _provider_message_count(blocks: tuple[IRBlock, ...] | list[IRBlock]) -> int:
    count = 0
    last_role: str | None = None
    for block in blocks:
        role = _provider_role(block)
        if role != last_role:
            count += 1
            last_role = role
    return count


def _provider_role(block: IRBlock) -> Literal["user", "assistant"]:
    if isinstance(block, IRUserTextBlock | IRImageBlock | IRToolResultBlock):
        return "user"
    if isinstance(block, IRAssistantTextBlock | IRThinkingBlock | IRToolCallBlock):
        return "assistant"
    raise TypeError(f"Unsupported IR block type: {type(block)}")


def _rendered_context_blocks(
    source_context: SourceContext,
    blocks: tuple[IRBlock, ...],
) -> list[IRBlock]:
    return _apply_tool_result_ttls(source_context.rehydrated, list(blocks))


def render_compiled_markdown(compiled: CompiledChapter) -> str:
    parts = [
        f"# Compiled {compiled.title}",
        "",
        f"- Source chapter: `{compiled.chapter_path}`",
        f"- Block pair: `{compiled.pair.first}-{compiled.pair.second}`",
        f"- Raw IR blocks: `{len(compiled.ir_blocks)}`",
        f"- Rendered IR blocks: `{len(compiled.rendered_ir_blocks)}`",
        f"- Provider messages: `{compiled.provider_message_count}`",
        f"- Tokens (TTL-aware render): `{compiled.token_count}`",
        "",
        "## Rendered Blocks",
        "",
    ]
    for idx, block in enumerate(compiled.rendered_ir_blocks):
        parts.extend(_render_context_block_markdown(block, context_idx=idx))
    return "\n".join(parts).rstrip() + "\n"


def _write_compiled_artifacts(compiled: CompiledChapter) -> None:
    compiled.output_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.output_path.write_text(
        json.dumps(compiled.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compiled.preview_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.preview_path.write_text(
        render_compiled_markdown(compiled), encoding="utf-8"
    )


def _block_json(block: IRBlock) -> dict[str, Any]:
    return cast(dict[str, Any], block.model_dump(mode="json"))


def _beat_json(beat: ChapterBeat) -> dict[str, Any]:
    return {
        "kind": beat.kind,
        "line_start": beat.line_start,
        "line_end": beat.line_end,
        "source": beat.source,
        "text": beat.text,
        "tool_name": beat.tool_name,
        "tool_input": beat.tool_input,
        "call_id": beat.call_id,
        "marker": beat.marker,
    }


def _opening_memory_text(chapter_number: int, pair: Pair) -> str:
    return (
        OPENING_MEMORY_TEXT.rstrip()
        + f'\n<!-- dream_chapter="{chapter_number:02d}" block_pair="{pair.slug}" -->'
    )


def _chapter_number_from_path(path: Path) -> int:
    match = _CHAPTER_RE.match(path.name)
    if match is None:
        raise DreamCompileError(
            CompileErrorDetail(
                message=f"Could not infer chapter number from {path.name}",
                chapter_path=path,
            )
        )
    return int(match.group(1))


def _pair_for_chapter(chapter_number: int) -> Pair:
    if chapter_number < 1:
        raise ValueError(f"Chapter number must be positive: {chapter_number}")
    first = (chapter_number - 1) * 2
    return Pair(first=first, second=first + 1)


def _chapter_title(markdown: str, *, fallback: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return fallback


def _compiled_output_path(
    chapter_path: Path,
    *,
    out_dir: Path | None,
    suffix: str,
) -> Path:
    directory = (
        chapter_path.parent if out_dir is None else out_dir.expanduser().resolve()
    )
    return directory / f"{chapter_path.stem}.compiled{suffix}"


def _with_nearby_lines(
    exc: DreamCompileError,
    chapter_path: Path,
    markdown: str,
) -> DreamCompileError:
    detail = exc.detail
    if detail.line_start is None:
        updated = CompileErrorDetail(
            message=detail.message,
            chapter_path=chapter_path,
            source=detail.source,
            beat_index=detail.beat_index,
        )
        return DreamCompileError(updated)
    lines = markdown.splitlines()
    start = max(1, detail.line_start - 3)
    end = min(len(lines), (detail.line_end or detail.line_start) + 3)
    nearby = tuple(f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1))
    updated = CompileErrorDetail(
        message=detail.message,
        chapter_path=chapter_path,
        line_start=detail.line_start,
        line_end=detail.line_end,
        beat_index=detail.beat_index,
        source=detail.source,
        nearby_lines=nearby,
    )
    return DreamCompileError(updated)


def _chapter_paths(chapter_dir: Path, chapters: tuple[int, ...] | None) -> list[Path]:
    paths = [
        path
        for path in sorted(chapter_dir.expanduser().resolve().glob("chapter-*.md"))
        if not path.name.endswith(".compiled.md")
    ]
    if chapters is None:
        return paths
    wanted = set(chapters)
    return [path for path in paths if _chapter_number_from_path(path) in wanted]


def _review_ceiling_tokens(review_dir: Path, chapter_number: int) -> int | None:
    review_path = (
        review_dir.expanduser().resolve() / f"review-{chapter_number:02d}.json"
    )
    if not review_path.exists():
        return None
    try:
        data = json.loads(review_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("ceiling_tokens")
    if value is None and isinstance(data.get("request"), dict):
        value = data["request"].get("ceiling_tokens")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _savings_tokens(ceiling_tokens: int | None, token_count: int | None) -> int | None:
    if ceiling_tokens is None or token_count is None:
        return None
    return ceiling_tokens - token_count


def _parse_chapters(value: str) -> tuple[int, ...]:
    chapters: list[int] = []
    seen: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", maxsplit=1)
            start = int(start_raw)
            end = int(end_raw)
            if end < start:
                raise argparse.ArgumentTypeError(
                    f"Chapter range {part!r} must be ascending."
                )
            values = range(start, end + 1)
        else:
            values = (int(part),)
        for chapter in values:
            if chapter < 1:
                raise argparse.ArgumentTypeError("Chapter numbers must be positive.")
            if chapter not in seen:
                seen.add(chapter)
                chapters.append(chapter)
    if not chapters:
        raise argparse.ArgumentTypeError("At least one chapter is required.")
    return tuple(chapters)


def _default_work_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_WORK_ROOT / f"run-{stamp}-{uuid4().hex[:8]}"


def _validate_options(options: CompileOptions) -> None:
    if options.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    if not options.chapter_dir.expanduser().exists():
        raise FileNotFoundError(options.chapter_dir.expanduser())
    if not options.transcript_path.expanduser().exists():
        raise FileNotFoundError(options.transcript_path.expanduser())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dream_compiler",
        description="Compile dream chapter markdown into Spellbook IR memory blocks.",
    )
    parser.add_argument(
        "--chapter-dir",
        type=Path,
        default=DEFAULT_DREAM_CHAPTER_DIR,
        help=f"Dream chapter directory. Defaults to {DEFAULT_DREAM_CHAPTER_DIR}.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_TRANSCRIPT,
        help=f"Meta-Claude transcript path. Defaults to {DEFAULT_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=DEFAULT_DREAM_REVIEW_DIR,
        help=(
            "Director review directory for original summary ceilings. "
            f"Defaults to {DEFAULT_DREAM_REVIEW_DIR}."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for compiled sidecars. Defaults beside each chapter.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Run workspace. Defaults to a unique directory under /tmp/spellbook-dream-compiler.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_COMPILE_MODEL,
        help=f"Token-count model. Defaults to {DEFAULT_COMPILE_MODEL}.",
    )
    parser.add_argument(
        "--sanity-model",
        default=DEFAULT_SANITY_MODEL,
        help=f"Sanity-check model. Defaults to {DEFAULT_SANITY_MODEL}.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Maximum concurrent chapter compile/sanity tasks. Defaults to 3.",
    )
    parser.add_argument(
        "--chapters",
        type=_parse_chapters,
        default=None,
        help='Optional chapter filter, e.g. "3,5-8". Defaults to all chapters.',
    )
    parser.add_argument(
        "--skip-sanity",
        action="store_true",
        help="Skip live Sonnet sanity checks.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the Rich progress bar.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before API calls. Defaults to {DEFAULT_ENV_PATH}.",
    )
    return parser.parse_args(argv)


def _options_from_args(args: argparse.Namespace) -> CompileOptions:
    return CompileOptions(
        chapter_dir=args.chapter_dir,
        transcript_path=args.transcript,
        review_dir=args.review_dir,
        out_dir=args.out_dir,
        work_dir=args.work_dir or _default_work_dir(),
        model=args.model,
        sanity_model=args.sanity_model,
        concurrency=args.concurrency,
        skip_sanity=args.skip_sanity,
        show_progress=not args.no_progress,
        chapters=args.chapters,
    )


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)
    report = await run_compile(_options_from_args(args))
    _print_run_summary(report)


def _print_run_summary(report: CompileRunReport) -> None:
    print(f"Chapters: {report.options.chapter_dir.expanduser().resolve()}")
    print(f"Transcript: {report.options.transcript_path.expanduser().resolve()}")
    print(f"Reviews: {report.options.review_dir.expanduser().resolve()}")
    print(f"Work dir: {report.options.work_dir.expanduser().resolve()}")
    if report.options.out_dir is not None:
        print(f"Out dir: {report.options.out_dir.expanduser().resolve()}")
    print("")
    print(
        "Chapter     Beats  IR blocks  Messages  Tokens  Ceiling  Savings  Sanity   Status    File"
    )
    print(
        "----------  -----  ---------  --------  ------  -------  -------  -------  --------  -------------------------------------------"
    )
    for result in report.results:
        token_text = "-" if result.token_count is None else f"{result.token_count:,}"
        ceiling_text = (
            "-" if result.ceiling_tokens is None else f"{result.ceiling_tokens:,}"
        )
        savings_text = _format_savings(result.savings_tokens)
        chapter_id = f"chapter_{_safe_chapter_number(result.chapter_path):02d}"
        print(
            f"{chapter_id:<10}  "
            f"{result.beat_count:>5}  "
            f"{result.ir_block_count:>9}  "
            f"{result.provider_message_count:>8}  "
            f"{token_text:>6}  "
            f"{ceiling_text:>7}  "
            f"{savings_text:>7}  "
            f"{result.sanity_status:<7}  "
            f"{result.status:<8}  "
            f"{result.chapter_path.name[:43]:<43}"
        )

    _print_total_savings(report.results)

    failures = [result for result in report.results if result.error is not None]
    if failures:
        print("")
        print("Failures:")
        for result in failures:
            _print_failure(result.error)

    sanity_notes = [
        result
        for result in report.results
        if result.sanity_notes and result.error is None
    ]
    if sanity_notes:
        print("")
        print("Sanity notes:")
        for result in sanity_notes:
            chapter_id = f"chapter_{_safe_chapter_number(result.chapter_path):02d}"
            print(f"- {chapter_id} [{result.sanity_status}]: {result.sanity_notes}")


def _format_savings(value: int | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):,}"


def _print_total_savings(results: tuple[ChapterRunResult, ...]) -> None:
    counted = [
        result
        for result in results
        if result.token_count is not None and result.ceiling_tokens is not None
    ]
    if not counted:
        return
    compiled_total = sum(result.token_count or 0 for result in counted)
    ceiling_total = sum(result.ceiling_tokens or 0 for result in counted)
    savings_total = ceiling_total - compiled_total
    print("")
    print(
        "Totals: "
        f"{compiled_total:,} compiled / {ceiling_total:,} original ceiling "
        f"({_format_savings(savings_total)} tokens)"
    )


def _safe_chapter_number(path: Path) -> int:
    try:
        return _chapter_number_from_path(path)
    except DreamCompileError:
        return 0


def _print_failure(error: CompileErrorDetail | None) -> None:
    if error is None:
        return
    print(f"- {error.chapter_path}: {error.message}")
    if error.line_start is not None:
        span = str(error.line_start)
        if error.line_end is not None and error.line_end != error.line_start:
            span = f"{error.line_start}-{error.line_end}"
        print(f"  line: {span}")
    if error.beat_index is not None:
        print(f"  beat: {error.beat_index}")
    if error.source:
        print(f"  source: {error.source}")
    if error.nearby_lines:
        print("  nearby:")
        for line in error.nearby_lines:
            print(f"    {line}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()
