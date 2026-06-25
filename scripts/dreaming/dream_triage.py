"""Triage Director coverage-missing items for dream chapters."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scripts.dreaming.dream_director import DEFAULT_DREAM_REVIEW_DIR, DEFAULT_ENV_PATH
from spellbook.backends import infer_provider_for_model
from spellbook.config import HomunculusConfig, SpellbookConfig
from spellbook.custom import CustomSurface
from spellbook.ir_types import IRToolTextBlock
from spellbook.sdk import Spell
from spellbook.tools.common import Tool, ToolExecutionResult, ToolMetadata

DEFAULT_TRIAGE_MODEL = "claude-sonnet-4-6"
DEFAULT_TRIAGE_OUTPUT = DEFAULT_DREAM_REVIEW_DIR / "triage.json"

Category = Literal[
    "relational",
    "philosophical",
    "architectural",
    "implementation",
    "ephemeral",
]
Disposition = Literal["keep", "compress", "drop"]

KEEP_CATEGORIES = {"relational", "philosophical", "architectural"}
COMPRESS_CATEGORIES = {"implementation"}
DROP_CATEGORIES = {"ephemeral"}

TRIAGE_SYSTEM_PROMPT = """\
You are triaging Director coverage-missing items for narrative dream chapters.

Your job is not to re-review the chapters. Classify each missing item by what
kind of memory it represents, so a human can decide what should be restored.

Use the SubmitTriage tool exactly once. Do not answer in prose.

Categories:

- relational: knowledge about Ryan, family, relationships, personal context, or
  moments between people/minds. These are KEEP.
- philosophical: framework insights, design principles, key reframes, naming
  moments, or conceptual positions. These are KEEP.
- architectural: major system design decisions or structural choices that shaped
  everything after. These are KEEP.
- implementation: specific commits, line counts, test numbers, file paths, CLI
  flags, bug fixes, or local mechanics. These are COMPRESS.
- ephemeral: specific PIDs, daemon restarts, transient debugging steps, temporary
  state, or routine operational noise. These are DROP.

Prefer implementation over architectural for ordinary implementation details,
even if they were useful at the time. Reserve architectural for durable design
choices that changed later work. Prefer philosophical for named ideas, explicit
reframes, or principles that should survive as meaning rather than mechanics.
Prefer relational for human/mind context even when there are technical details
nearby.
"""


class TriageClassification(BaseModel):
    """Classification for one missing item."""

    item_id: str = Field(description="Stable item ID from the user message.")
    category: Category = Field(description="Triage category for this item.")
    reason: str = Field(description="One concise sentence explaining the category.")


class SubmitTriageInput(BaseModel):
    """Submit classifications for all missing items."""

    classifications: list[TriageClassification] = Field(
        description="One classification for every item ID provided."
    )


@dataclass
class _TriageHolder:
    triage: SubmitTriageInput | None = None


@dataclass(frozen=True)
class MissingItem:
    chapter_id: str
    chapter_number: int
    item_id: str
    item: str
    review_path: Path


@dataclass(frozen=True)
class TriageRunResult:
    output_path: Path
    model: str
    transcript_paths: tuple[Path, ...]
    missing_items: tuple[MissingItem, ...]
    classifications: tuple[TriageClassification, ...]
    triage_json: dict[str, dict[str, list[dict[str, str]]]]
    turn_text: str


async def run_triage(
    *,
    review_dir: Path = DEFAULT_DREAM_REVIEW_DIR,
    output_path: Path = DEFAULT_TRIAGE_OUTPUT,
    model: str = DEFAULT_TRIAGE_MODEL,
    transcript_path: Path | None = None,
    chapters_per_call: int = 4,
) -> TriageRunResult:
    if chapters_per_call < 1:
        raise ValueError("chapters_per_call must be at least 1.")
    review_dir = review_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    missing_items = tuple(_load_missing_items(review_dir))
    if not missing_items:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}\n", encoding="utf-8")
        return TriageRunResult(
            output_path=output_path,
            model=model,
            transcript_paths=(),
            missing_items=(),
            classifications=(),
            triage_json={},
            turn_text="",
        )

    classifications: list[TriageClassification] = []
    transcript_paths: list[Path] = []
    turn_texts: list[str] = []
    batches = _item_batches(missing_items, chapters_per_call)
    for idx, batch in enumerate(batches, start=1):
        batch_transcript_path = (
            transcript_path
            if transcript_path is not None and len(batches) == 1
            else review_dir
            / "triage-transcripts"
            / f"triage-batch-{idx:02d}-{uuid4().hex}.jsonl"
        )
        batch_classifications, batch_turn_text = await _run_triage_batch(
            batch,
            model=model,
            transcript_path=batch_transcript_path,
        )
        classifications.extend(batch_classifications)
        transcript_paths.append(batch_transcript_path.expanduser().resolve())
        turn_texts.append(batch_turn_text)

    resolved_classifications = tuple(classifications)
    _validate_classifications(missing_items, resolved_classifications)
    triage_json = _triage_json(missing_items, resolved_classifications)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(triage_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TriageRunResult(
        output_path=output_path,
        model=model,
        transcript_paths=tuple(transcript_paths),
        missing_items=missing_items,
        classifications=resolved_classifications,
        triage_json=triage_json,
        turn_text="\n\n".join(turn_texts),
    )


async def _run_triage_batch(
    items: tuple[MissingItem, ...],
    *,
    model: str,
    transcript_path: Path,
) -> tuple[tuple[TriageClassification, ...], str]:
    holder = _TriageHolder()
    spell = Spell(
        config=_triage_config(model),
        transcript_path=transcript_path,
        custom_surface=CustomSurface(tools=[_submit_triage_tool(holder)]),
    )
    result = await spell.once(_triage_user_message(items))
    if holder.triage is None:
        raise RuntimeError(
            "Triage model did not call SubmitTriage; no structured output captured."
        )
    classifications = tuple(holder.triage.classifications)
    _validate_classifications(items, classifications)
    return classifications, result.text


def _triage_config(model: str) -> SpellbookConfig:
    return SpellbookConfig(
        provider=infer_provider_for_model(model),
        model=model,
        session_type="custom",
        cwd=Path.cwd(),
        max_output_tokens=32_000,
        system_prompt=TRIAGE_SYSTEM_PROMPT,
        hom_config=HomunculusConfig(detect_interval=10_000),
    )


def _submit_triage_tool(holder: _TriageHolder) -> Tool[SubmitTriageInput]:
    async def submit_triage(
        meta: ToolMetadata,
        input: SubmitTriageInput,
    ) -> ToolExecutionResult:
        holder.triage = input
        return ToolExecutionResult(
            content=[IRToolTextBlock(text="Triage classifications recorded.")]
        )

    return Tool(
        name="SubmitTriage",
        input_model=SubmitTriageInput,
        exec=submit_triage,
        category="thinking",
    )


def _load_missing_items(review_dir: Path) -> list[MissingItem]:
    items: list[MissingItem] = []
    for review_path in sorted(review_dir.glob("review-*.json")):
        data = json.loads(review_path.read_text(encoding="utf-8"))
        missing = data.get("review", {}).get("coverage_missing") or []
        if not missing:
            continue
        chapter_number = int(data.get("chapter_number") or _chapter_number(review_path))
        chapter_id = f"chapter_{chapter_number:02d}"
        for idx, item in enumerate(missing, start=1):
            items.append(
                MissingItem(
                    chapter_id=chapter_id,
                    chapter_number=chapter_number,
                    item_id=f"{chapter_id}_item_{idx:02d}",
                    item=str(item),
                    review_path=review_path,
                )
            )
    return items


def _chapter_number(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.removeprefix("review-"))
    except ValueError as exc:
        raise ValueError(f"Could not infer chapter number from {path}") from exc


def _triage_user_message(items: tuple[MissingItem, ...]) -> str:
    chapters: dict[str, list[MissingItem]] = {}
    for item in items:
        chapters.setdefault(item.chapter_id, []).append(item)

    parts = [
        "Classify each Director coverage-missing item below.",
        "",
        "Return exactly one classification for every item_id.",
        "Do not merge, split, rewrite, or omit items.",
        "",
    ]
    for chapter_id, chapter_items in chapters.items():
        parts.extend([f"## {chapter_id}", ""])
        for item in chapter_items:
            parts.extend([f"- {item.item_id}: {item.item}", ""])
    return "\n".join(parts).rstrip()


def _item_batches(
    items: tuple[MissingItem, ...],
    chapters_per_call: int,
) -> tuple[tuple[MissingItem, ...], ...]:
    chapters: dict[str, list[MissingItem]] = {}
    for item in items:
        chapters.setdefault(item.chapter_id, []).append(item)

    batches: list[tuple[MissingItem, ...]] = []
    current: list[MissingItem] = []
    current_chapters = 0
    for chapter_id in sorted(chapters):
        if current and current_chapters >= chapters_per_call:
            batches.append(tuple(current))
            current = []
            current_chapters = 0
        current.extend(chapters[chapter_id])
        current_chapters += 1
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _validate_classifications(
    items: tuple[MissingItem, ...],
    classifications: tuple[TriageClassification, ...],
) -> None:
    expected = {item.item_id for item in items}
    returned = [classification.item_id for classification in classifications]
    returned_set = set(returned)
    missing = sorted(expected - returned_set)
    extra = sorted(returned_set - expected)
    duplicates = sorted(
        item_id for item_id, count in Counter(returned).items() if count > 1
    )
    if missing or extra or duplicates:
        raise RuntimeError(
            "Triage classification IDs did not match input IDs. "
            f"Missing: {missing}; extra: {extra}; duplicates: {duplicates}"
        )


def _triage_json(
    items: tuple[MissingItem, ...],
    classifications: tuple[TriageClassification, ...],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    by_id = {item.item_id: item for item in items}
    output: dict[str, dict[str, list[dict[str, str]]]] = {}
    for classification in classifications:
        item = by_id[classification.item_id]
        disposition = _disposition(classification.category)
        chapter = output.setdefault(
            item.chapter_id, {"keep": [], "compress": [], "drop": []}
        )
        chapter[disposition].append(
            {
                "item": item.item,
                "category": classification.category,
                "reason": classification.reason,
            }
        )

    return {
        chapter_id: groups
        for chapter_id, groups in sorted(output.items())
        if groups["keep"] or groups["compress"] or groups["drop"]
    }


def _disposition(category: Category) -> Disposition:
    if category in KEEP_CATEGORIES:
        return "keep"
    if category in COMPRESS_CATEGORIES:
        return "compress"
    if category in DROP_CATEGORIES:
        return "drop"
    raise ValueError(f"Unknown category: {category}")


def _summary_counts(
    triage_json: dict[str, dict[str, list[dict[str, str]]]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for groups in triage_json.values():
        for disposition, items in groups.items():
            counts[disposition] += len(items)
            for item in items:
                counts[item["category"]] += 1
    return counts


def _keep_items(
    triage_json: dict[str, dict[str, list[dict[str, str]]]],
) -> list[tuple[str, dict[str, str]]]:
    keep: list[tuple[str, dict[str, str]]] = []
    for chapter_id, groups in sorted(triage_json.items()):
        for item in groups["keep"]:
            keep.append((chapter_id, item))
    return keep


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dream_triage",
        description="Triage Director coverage-missing items into keep/compress/drop.",
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=DEFAULT_DREAM_REVIEW_DIR,
        help=f"Review JSON directory. Defaults to {DEFAULT_DREAM_REVIEW_DIR}.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_TRIAGE_OUTPUT,
        help=f"Triage JSON output path. Defaults to {DEFAULT_TRIAGE_OUTPUT}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_TRIAGE_MODEL,
        help=f"Triage model. Defaults to {DEFAULT_TRIAGE_MODEL}.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before model calls. Defaults to {DEFAULT_ENV_PATH}.",
    )
    parser.add_argument(
        "--chapters-per-call",
        type=int,
        default=4,
        help="Number of chapters to classify per Sonnet call. Defaults to 4.",
    )
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)
    result = await run_triage(
        review_dir=args.review_dir,
        output_path=args.out,
        model=args.model,
        chapters_per_call=args.chapters_per_call,
    )
    _print_summary(result)


def _print_summary(result: TriageRunResult) -> None:
    counts = _summary_counts(result.triage_json)
    keep = _keep_items(result.triage_json)
    print(f"Triage: {result.output_path}")
    print(f"Model:  {result.model}")
    print(f"Items:  {len(result.missing_items)}")
    print("")
    print("Buckets:")
    print(f"- keep:     {counts['keep']}")
    print(f"- compress: {counts['compress']}")
    print(f"- drop:     {counts['drop']}")
    print("")
    print("Categories:")
    for category in (
        "relational",
        "philosophical",
        "architectural",
        "implementation",
        "ephemeral",
    ):
        print(f"- {category}: {counts[category]}")
    if keep:
        print("")
        print("Keep items:")
        for chapter_id, item in keep:
            print(f"- {chapter_id} [{item['category']}]: {item['item']}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()
