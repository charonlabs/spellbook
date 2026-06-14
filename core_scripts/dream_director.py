"""Review one dream chapter with a Director Spell."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core_scripts.mdtoks import count_markdown_tokens
from core_scripts.merge_stats import (
    DEFAULT_TRANSCRIPT,
    MergeStatsReport,
    merge_stats,
)
from spellbook.backends import infer_provider_for_model
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.custom import CustomSurface
from spellbook.ir_types import IRToolTextBlock
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.sdk import Spell
from spellbook.tools.common import Tool, ToolExecutionResult, ToolMetadata

DEFAULT_ENV_PATH = Path.home() / ".chorus/.env"
DEFAULT_DIRECTOR_MODEL = "claude-opus-4-8"
DEFAULT_DREAM_REVIEW_DIR = Path.home() / ".chorus/dream-reviews"
DEFAULT_DREAM_RENDER_DIR = Path("/tmp")
PIN_MARKER_RE = re.compile(r"<!--\s*pin:\s*(.*?)\s*-->", re.IGNORECASE | re.DOTALL)
PIN_TAG_RE = re.compile(r"<pin\b[^>]*>|</pin>", re.IGNORECASE)
PIN_ATTR_RE = re.compile(r'([A-Za-z_][\w:-]*)="([^"]*)"')

DIRECTOR_SYSTEM_PROMPT = """\
You are the Director for a narrative dream-merge pass.

Your job is to evaluate one drafted dream chapter against its source material.
The user message will contain:
- the two original semantic block summaries being replaced
- exact budget data
- the rendered full source blocks
- the drafted dream chapter, with any pin markers expanded for review

Use the SubmitReview tool exactly once. Do not merely answer in prose.

Rubric:

Coverage is critical. Compare the original summaries against the chapter line
by line. Every load-bearing decision, commit hash, entity name, file path, and
outcome mentioned in the summaries should appear in the chapter, unless omitted
for a clear reason. Report present items and missing items.

Rendered source blocks may contain <pin>...</pin> sections. The chapter may
place a matching <!-- pin: Name --> marker instead of restating that material.
The user message expands matched markers by inserting the full source pin
section at that point in the chapter. Treat expanded pin content as present for
coverage: it will be placed fully where the marker appears in the merged
chapter. Do not flag facts as missing merely because they live inside expanded
pin content. Do flag unmatched markers, source pins with no marker, or pin
placements that make the assembled chapter incoherent.

Accuracy is critical. Claims about what happened in these source blocks must be
verifiable against the rendered source blocks. Assume future-tense marginalia is
true when it reaches beyond the source blocks; the author has context you do not.
Flag only claims about these blocks that do not match the source material.

Budget is a hard constraint. Use the provided exact mdtoks result and ceiling.

Quality is advisory. Note whether the chapter works as narrative: coherence,
voice consistency, useful marginalia, and well-chosen exchanges.
"""


class DirectorReviewInput(BaseModel):
    """Submit the structured Director review for one dream chapter."""

    verdict: Literal["pass", "needs_human_review", "fail"] = Field(
        description=(
            "Overall verdict. Use pass only if coverage, accuracy, and budget are "
            "all acceptable. Use needs_human_review for missing/uncertain items."
        )
    )
    coverage_present: list[str] = Field(
        description="Load-bearing summary items that the chapter does cover."
    )
    coverage_missing: list[str] = Field(
        description="Load-bearing summary items missing from the chapter."
    )
    accuracy_flags: list[str] = Field(
        description=(
            "Factual claims about these blocks that are not supported by the source. "
            "Use an empty list if clean."
        )
    )
    budget_token_count: int = Field(
        description="Exact mdtoks token count for the chapter."
    )
    budget_ceiling: int = Field(
        description="Two-summary token ceiling for this block pair."
    )
    budget_passed: bool = Field(
        description="Whether budget_token_count is less than or equal to budget_ceiling."
    )
    budget_notes: str = Field(description="Brief budget notes.")
    quality_notes: str = Field(description="Advisory narrative-quality notes.")
    general_feedback: str = Field(
        description="Concise human-facing feedback paragraph."
    )


@dataclass
class _ReviewHolder:
    review: DirectorReviewInput | None = None


@dataclass(frozen=True)
class PinSection:
    idx: int
    start: int
    end: int
    kind: str
    block_idx: str
    title: str
    reason: str
    facet_id: str | None
    markdown: str

    @property
    def label(self) -> str:
        if self.title:
            return self.title
        if self.facet_id:
            return self.facet_id
        if self.block_idx:
            return f"Block {self.block_idx} {self.kind} pin"
        return f"Pin {self.idx + 1}"


@dataclass(frozen=True)
class PinExpansionResult:
    chapter_markdown: str
    source_pins: tuple[PinSection, ...]
    matched_markers: tuple[str, ...]
    unmatched_markers: tuple[str, ...]
    unused_pins: tuple[str, ...]
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "source_pin_count": len(self.source_pins),
            "matched_markers": list(self.matched_markers),
            "unmatched_markers": list(self.unmatched_markers),
            "unused_pins": list(self.unused_pins),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Pair:
    first: int
    second: int

    @property
    def chapter_number(self) -> int:
        return (self.first // 2) + 1

    @property
    def slug(self) -> str:
        return f"{self.first}-{self.second}"


@dataclass(frozen=True)
class DirectorReviewRequest:
    pair: Pair
    transcript_path: Path
    render_path: Path
    chapter_path: Path
    chapter_tokens: int
    chapter_chars: int
    ceiling_tokens: int
    stats: MergeStatsReport
    summaries_markdown: str
    source_blocks_markdown: str
    chapter_markdown: str


@dataclass(frozen=True)
class DirectorReviewReport:
    pair: Pair
    review: DirectorReviewInput
    output_path: Path
    director_model: str
    director_transcript_path: Path
    request: DirectorReviewRequest
    turn_text: str

    def to_json_dict(self) -> dict[str, object]:
        pin_expansion = _expand_pin_markers(
            chapter_markdown=self.request.chapter_markdown,
            source_blocks_markdown=self.request.source_blocks_markdown,
        )
        return {
            "pair": [self.pair.first, self.pair.second],
            "chapter_number": self.pair.chapter_number,
            "chapter_path": str(self.request.chapter_path),
            "render_path": str(self.request.render_path),
            "transcript_path": str(self.request.transcript_path),
            "director_model": self.director_model,
            "director_transcript_path": str(self.director_transcript_path),
            "chapter_tokens": self.request.chapter_tokens,
            "chapter_chars": self.request.chapter_chars,
            "ceiling_tokens": self.request.ceiling_tokens,
            "budget_passed": self.request.chapter_tokens <= self.request.ceiling_tokens,
            "pin_expansion": pin_expansion.to_json_dict(),
            "review": self.review.model_dump(mode="json"),
            "director_text": self.turn_text,
        }


async def run_director_review(
    *,
    pair: Pair,
    chapter_path: Path,
    transcript_path: Path = DEFAULT_TRANSCRIPT,
    render_path: Path | None = None,
    output_path: Path | None = None,
    director_model: str = DEFAULT_DIRECTOR_MODEL,
    director_transcript_path: Path | None = None,
) -> DirectorReviewReport:
    request = await prepare_director_request(
        pair=pair,
        chapter_path=chapter_path,
        transcript_path=transcript_path,
        render_path=render_path,
    )
    holder = _ReviewHolder()
    custom_surface = CustomSurface(tools=[_submit_review_tool(holder)])
    resolved_director_transcript_path = director_transcript_path or (
        DEFAULT_DREAM_REVIEW_DIR
        / "director-transcripts"
        / f"review-{pair.chapter_number:02d}-{uuid4().hex}.jsonl"
    )
    config = _director_config(director_model)
    spell = Spell(
        config=config,
        transcript_path=resolved_director_transcript_path,
        custom_surface=custom_surface,
    )
    result = await spell.once(_director_user_message(request))
    if holder.review is None:
        raise RuntimeError(
            "Director did not call SubmitReview; no structured review was captured."
        )

    resolved_output_path = (
        (
            output_path
            or DEFAULT_DREAM_REVIEW_DIR / f"review-{pair.chapter_number:02d}.json"
        )
        .expanduser()
        .resolve()
    )
    report = DirectorReviewReport(
        pair=pair,
        review=holder.review,
        output_path=resolved_output_path,
        director_model=director_model,
        director_transcript_path=resolved_director_transcript_path.expanduser().resolve(),
        request=request,
        turn_text=result.text,
    )
    _write_report(report)
    return report


async def prepare_director_request(
    *,
    pair: Pair,
    chapter_path: Path,
    transcript_path: Path,
    render_path: Path | None,
) -> DirectorReviewRequest:
    chapter_path = chapter_path.expanduser().resolve()
    transcript_path = transcript_path.expanduser().resolve()
    resolved_render_path = _resolve_render_path(pair, render_path)
    stats = await merge_stats(
        transcript_path=transcript_path,
        first_idx=pair.first,
        second_idx=pair.second,
        render_path=resolved_render_path,
    )
    rehydrated = Rehydrator(transcript_path).run()
    chapter_count = await count_markdown_tokens(chapter_path)
    return DirectorReviewRequest(
        pair=pair,
        transcript_path=transcript_path,
        render_path=stats.render_path or resolved_render_path.expanduser().resolve(),
        chapter_path=chapter_path,
        chapter_tokens=chapter_count.tokens,
        chapter_chars=chapter_count.characters,
        ceiling_tokens=stats.summary_count.tokens,
        stats=stats,
        summaries_markdown=_summaries_markdown(rehydrated, pair),
        source_blocks_markdown=_source_blocks_from_render(
            stats.render_path or resolved_render_path
        ),
        chapter_markdown=chapter_path.read_text(encoding="utf-8"),
    )


def _director_config(model: str) -> SpellbookConfig:
    return SpellbookConfig(
        provider=infer_provider_for_model(model),
        model=model,
        session_type="custom",
        cwd=Path.cwd(),
        max_output_tokens=16_000,
        system_prompt=DIRECTOR_SYSTEM_PROMPT,
        hom_config=HomunculusConfig(detect_interval=10_000),
    )


def _submit_review_tool(holder: _ReviewHolder) -> Tool[DirectorReviewInput]:
    async def submit_review(
        meta: ToolMetadata,
        input: DirectorReviewInput,
    ) -> ToolExecutionResult:
        holder.review = input
        return ToolExecutionResult(
            content=[IRToolTextBlock(text="Director review recorded.")]
        )

    return Tool(
        name="SubmitReview",
        input_model=DirectorReviewInput,
        exec=submit_review,
        category="thinking",
    )


def _director_user_message(request: DirectorReviewRequest) -> str:
    budget_passed = request.chapter_tokens <= request.ceiling_tokens
    budget_status = "PASS" if budget_passed else "FAIL"
    pin_expansion = _expand_pin_markers(
        chapter_markdown=request.chapter_markdown,
        source_blocks_markdown=request.source_blocks_markdown,
    )
    return f"""\
Review this dream chapter.

## Metadata

- Block pair: {request.pair.first}-{request.pair.second}
- Chapter path: {request.chapter_path}
- Render path: {request.render_path}
- Exact chapter tokens from mdtoks: {request.chapter_tokens:,}
- Token ceiling from two summary renderings: {request.ceiling_tokens:,}
- Budget status: {budget_status}

## Original Summaries

{request.summaries_markdown}

## Rendered Source Blocks

The following is ground truth for what happened in these blocks.

{request.source_blocks_markdown}

## Pin Expansion Notes

{_pin_expansion_notes_markdown(pin_expansion)}

## Dream Chapter (Pins Expanded For Review)

The following is the drafted chapter with matched `<!-- pin: ... -->` markers
followed by the full source `<pin>` section that will be inserted there. Use
this assembled view for coverage and accuracy. The exact budget above still
applies to the raw chapter file, not to inserted pin text.

{pin_expansion.chapter_markdown}

## Required Output

Call SubmitReview exactly once with:
- coverage_present: load-bearing summary facts present in the chapter
- coverage_missing: load-bearing summary facts absent from the chapter
- accuracy_flags: factual errors about THESE blocks only, or [] if clean
- budget_token_count: {request.chapter_tokens}
- budget_ceiling: {request.ceiling_tokens}
- budget_passed: {str(budget_passed).lower()}
- budget_notes
- quality_notes
- general_feedback
"""


def _summaries_markdown(rehydrated: RehydrationResult, pair: Pair) -> str:
    parts: list[str] = []
    for block_idx in (pair.first, pair.second):
        block = next(
            (block for block in rehydrated.semantic_blocks if block.idx == block_idx),
            None,
        )
        if block is None:
            raise ValueError(f"Semantic block {block_idx} not found.")
        summary = next(
            (artifact for artifact in block.artifacts if artifact.type == "summary"),
            None,
        )
        if summary is None:
            raise ValueError(f"Semantic block {block_idx} has no summary artifact.")
        parts.extend(
            [
                f"### Block {block.idx}: {summary.headline}",
                "",
                summary.text,
                "",
            ]
        )
        if summary.facets:
            parts.extend(["#### Facets", ""])
            for facet in summary.facets:
                resources = (
                    f" Resources: {'; '.join(facet.resources)}"
                    if facet.resources
                    else ""
                )
                parts.extend(
                    [
                        (
                            f"- {facet.title} "
                            f"(context blocks {facet.start_block}-{facet.end_block})."
                            f"{resources}"
                        ),
                        f"  {facet.description}",
                    ]
                )
            parts.append("")
        if summary.open_thread:
            parts.extend(["#### Open Thread", "", summary.open_thread, ""])
    return "\n".join(parts).rstrip()


def _source_blocks_from_render(render_path: Path) -> str:
    text = render_path.expanduser().resolve().read_text(encoding="utf-8")
    marker = "# Source Blocks"
    marker_idx = text.find(marker)
    if marker_idx == -1:
        return text
    return text[marker_idx:]


def _expand_pin_markers(
    *,
    chapter_markdown: str,
    source_blocks_markdown: str,
) -> PinExpansionResult:
    source_pins = _extract_pin_sections(source_blocks_markdown)
    used_pin_indices: set[int] = set()
    matched_markers: list[str] = []
    unmatched_markers: list[str] = []
    notes: list[str] = []

    def replace_marker(match: re.Match[str]) -> str:
        marker = match.group(0)
        marker_label = " ".join(match.group(1).split())
        pin_idx = _find_matching_pin(marker_label, source_pins, used_pin_indices)
        if pin_idx is None:
            unmatched_markers.append(marker_label)
            return (
                f"{marker}\n\n"
                "<!-- pin expansion missing: no matching source pin found -->"
            )

        used_pin_indices.add(pin_idx)
        pin = source_pins[pin_idx]
        matched_markers.append(marker_label)
        notes.append(
            f'Matched chapter pin "{marker_label}" to source pin "{pin.label}".'
        )
        return f"{marker}\n\n{pin.markdown}"

    expanded_chapter = PIN_MARKER_RE.sub(replace_marker, chapter_markdown)
    unused_pins = _unused_pin_labels(source_pins, used_pin_indices)

    if source_pins and not matched_markers:
        notes.append("Source pins were present, but no chapter pin markers matched.")
    if not source_pins and not matched_markers and not unmatched_markers:
        notes.append("No source pins or chapter pin markers were present.")
    for marker in unmatched_markers:
        notes.append(f'Chapter pin marker "{marker}" did not match any source pin.')
    for pin_label in unused_pins:
        notes.append(f'Source pin "{pin_label}" was not placed by a chapter marker.')

    return PinExpansionResult(
        chapter_markdown=expanded_chapter,
        source_pins=source_pins,
        matched_markers=tuple(matched_markers),
        unmatched_markers=tuple(unmatched_markers),
        unused_pins=tuple(unused_pins),
        notes=tuple(notes),
    )


def _extract_pin_sections(source_blocks_markdown: str) -> tuple[PinSection, ...]:
    stack: list[tuple[int, dict[str, str]]] = []
    sections: list[PinSection] = []
    for match in PIN_TAG_RE.finditer(source_blocks_markdown):
        tag = match.group(0)
        if tag.lower().startswith("<pin"):
            stack.append((match.start(), _pin_attrs(tag)))
            continue
        if not stack:
            continue
        start, attrs = stack.pop()
        sections.append(
            PinSection(
                idx=len(sections),
                start=start,
                end=match.end(),
                kind=attrs.get("kind", ""),
                block_idx=attrs.get("block_idx", ""),
                title=attrs.get("title", ""),
                reason=attrs.get("reason", ""),
                facet_id=attrs.get("facet_id"),
                markdown=source_blocks_markdown[start : match.end()],
            )
        )
    return tuple(sorted(sections, key=lambda pin: pin.start))


def _pin_attrs(tag: str) -> dict[str, str]:
    return {
        match.group(1): unescape(match.group(2)) for match in PIN_ATTR_RE.finditer(tag)
    }


def _find_matching_pin(
    marker_label: str,
    source_pins: tuple[PinSection, ...],
    used_pin_indices: set[int],
) -> int | None:
    scored: list[tuple[int, int]] = []
    for idx, pin in enumerate(source_pins):
        if idx in used_pin_indices:
            continue
        score = _pin_match_score(marker_label, pin)
        if score > 0:
            scored.append((score, idx))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][1]


def _pin_match_score(marker_label: str, pin: PinSection) -> int:
    marker = _normalize_pin_label(marker_label)
    if not marker:
        return 0
    candidates = [pin.title, pin.facet_id or "", pin.label]
    if pin.block_idx:
        candidates.append(f"block {pin.block_idx}")
    for candidate in candidates:
        normalized = _normalize_pin_label(candidate)
        if not normalized:
            continue
        if marker == normalized:
            return 3
        if len(marker) >= 8 and marker in normalized:
            return 2
        if len(normalized) >= 8 and normalized in marker:
            return 1
    return 0


def _normalize_pin_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _unused_pin_labels(
    source_pins: tuple[PinSection, ...],
    used_pin_indices: set[int],
) -> list[str]:
    unused: list[str] = []
    used_sections = [source_pins[idx] for idx in sorted(used_pin_indices)]
    for idx, pin in enumerate(source_pins):
        if idx in used_pin_indices:
            continue
        if any(
            used.start <= pin.start and pin.end <= used.end for used in used_sections
        ):
            continue
        unused.append(pin.label)
    return unused


def _pin_expansion_notes_markdown(pin_expansion: PinExpansionResult) -> str:
    lines = [
        f"- Source pins in render: {len(pin_expansion.source_pins)}",
        f"- Matched chapter markers: {len(pin_expansion.matched_markers)}",
        f"- Unmatched chapter markers: {len(pin_expansion.unmatched_markers)}",
        f"- Unplaced source pins: {len(pin_expansion.unused_pins)}",
    ]
    if pin_expansion.notes:
        lines.append("")
        lines.extend(f"- {note}" for note in pin_expansion.notes)
    return "\n".join(lines)


def _resolve_render_path(pair: Pair, render_path: Path | None) -> Path:
    if render_path is not None:
        return render_path.expanduser().resolve()
    return (DEFAULT_DREAM_RENDER_DIR / f"merge-blocks-{pair.slug}.md").resolve()


def _write_report(report: DirectorReviewReport) -> None:
    report.output_path.parent.mkdir(parents=True, exist_ok=True)
    report.output_path.write_text(
        json.dumps(report.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_pair(value: str) -> Pair:
    try:
        first_raw, second_raw = value.split("-", maxsplit=1)
        pair = Pair(first=int(first_raw), second=int(second_raw))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'Pair must look like "N-N+1", e.g. "0-1".'
        ) from exc
    if pair.second != pair.first + 1:
        raise argparse.ArgumentTypeError("Pair must contain adjacent block indices.")
    return pair


def _default_output_path(pair: Pair) -> Path:
    return DEFAULT_DREAM_REVIEW_DIR / f"review-{pair.chapter_number:02d}.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dream_director",
        description="Run a Director Spell against one dream chapter.",
    )
    parser.add_argument("--pair", type=_parse_pair, required=True)
    parser.add_argument("--chapter", type=Path, required=True)
    parser.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_TRANSCRIPT,
        help=f"Transcript JSONL path. Defaults to {DEFAULT_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--render",
        type=Path,
        default=None,
        help="Existing/source render path. Defaults to /tmp/merge-blocks-N-N+1.md.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Review JSON path. Defaults to ~/.chorus/dream-reviews/review-NN.json.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_DIRECTOR_MODEL,
        help=f"Director model. Defaults to {DEFAULT_DIRECTOR_MODEL}.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before model calls. Defaults to {DEFAULT_ENV_PATH}.",
    )
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)
    output_path = args.out or _default_output_path(args.pair)
    report = await run_director_review(
        pair=args.pair,
        chapter_path=args.chapter,
        transcript_path=args.transcript,
        render_path=args.render,
        output_path=output_path,
        director_model=args.model,
    )
    _print_report_summary(report)


def _print_report_summary(report: DirectorReviewReport) -> None:
    review = report.review
    print(f"Review:  {report.output_path}")
    print(f"Pair:    {report.pair.first}-{report.pair.second}")
    print(f"Chapter: {report.request.chapter_path}")
    print(f"Verdict: {review.verdict}")
    print(
        "Budget:  "
        f"{report.request.chapter_tokens:,} / {report.request.ceiling_tokens:,} "
        f"({'pass' if report.request.chapter_tokens <= report.request.ceiling_tokens else 'fail'})"
    )
    if review.coverage_missing:
        print("\nMissing coverage:")
        for item in review.coverage_missing:
            print(f"- {item}")
    if review.accuracy_flags:
        print("\nAccuracy flags:")
        for item in review.accuracy_flags:
            print(f"- {item}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()
