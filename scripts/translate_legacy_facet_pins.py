"""Suggest core facet pins corresponding to legacy pinned facets.

Legacy facet ids do not survive replay: the old transcript has block/facet ids
from the legacy detector, while the core replay has fresh semantic blocks and
fresh summary facets. This script compares the old pinned facet turn ranges
against the core replay's facet block ranges and reports overlap candidates.

It intentionally does not mutate either transcript. The output is a review aid
for deciding which core facets should receive durable facet-pin records.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spellbook.rehydrator import Rehydrator

DEFAULT_LEGACY_TRANSCRIPT = (
    Path.home()
    / ".chorus"
    / "spellbook"
    / "sessions"
    / "meta-claude"
    / "transcript.jsonl"
)
DEFAULT_CORE_TRANSCRIPT = (
    Path(__file__).resolve().parents[1]
    / "archive"
    / "core_replays"
    / "meta-claude-4"
    / "transcript.jsonl"
)


@dataclass(frozen=True)
class LegacyBlock:
    id: str
    title: str
    turn_start: int
    turn_end: int
    facets: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class LegacyFacetPin:
    block_id: str
    block_title: str | None
    facet_id: str
    facet_title: str | None
    facet_summary: str | None
    turn_start: int | None
    turn_end: int | None
    source: str
    created_facet: bool
    line_number: int

    @property
    def resolved(self) -> bool:
        return self.turn_start is not None and self.turn_end is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_title": self.block_title,
            "facet_id": self.facet_id,
            "facet_title": self.facet_title,
            "facet_summary": self.facet_summary,
            "turn_start": self.turn_start,
            "turn_end": self.turn_end,
            "source": self.source,
            "created_facet": self.created_facet,
            "line_number": self.line_number,
        }


@dataclass(frozen=True)
class CoreFacet:
    block_id: str
    block_idx: int
    block_title: str
    facet_id: str
    facet_title: str
    facet_description: str
    start_block: int
    end_block: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "block_idx": self.block_idx,
            "block_title": self.block_title,
            "facet_id": self.facet_id,
            "facet_title": self.facet_title,
            "facet_description": self.facet_description,
            "start_block": self.start_block,
            "end_block": self.end_block,
        }


@dataclass(frozen=True)
class FacetCandidate:
    facet: CoreFacet
    overlap_blocks: int
    union_blocks: int
    legacy_coverage: float
    core_coverage: float
    jaccard: float
    gap_blocks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.facet.to_dict(),
            "overlap_blocks": self.overlap_blocks,
            "union_blocks": self.union_blocks,
            "legacy_coverage": self.legacy_coverage,
            "core_coverage": self.core_coverage,
            "jaccard": self.jaccard,
            "gap_blocks": self.gap_blocks,
        }


@dataclass(frozen=True)
class FacetPinMatch:
    legacy_pin: LegacyFacetPin
    mapped_start_block: int | None
    mapped_end_block: int | None
    candidates: list[FacetCandidate]
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_pin": self.legacy_pin.to_dict(),
            "mapped_start_block": self.mapped_start_block,
            "mapped_end_block": self.mapped_end_block,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "warning": self.warning,
        }


@dataclass(frozen=True)
class FacetPinTranslationReport:
    legacy_path: Path
    core_path: Path
    matches: list[FacetPinMatch]
    core_facets_seen: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy_path": str(self.legacy_path),
            "core_path": str(self.core_path),
            "core_facets_seen": self.core_facets_seen,
            "matches": [match.to_dict() for match in self.matches],
        }


def analyze_legacy_facet_pins(
    *,
    legacy_path: Path,
    core_path: Path,
    top: int = 5,
) -> FacetPinTranslationReport:
    legacy_path = legacy_path.expanduser().resolve()
    core_path = core_path.expanduser().resolve()
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy transcript not found: {legacy_path}")
    if not core_path.exists():
        raise FileNotFoundError(f"Core transcript not found: {core_path}")

    legacy_pins = load_legacy_facet_pins(legacy_path)
    core_index = load_core_turn_index(core_path)
    core_facets = load_core_facets(core_path)

    matches = [_match_pin(pin, core_index, core_facets, top=top) for pin in legacy_pins]
    return FacetPinTranslationReport(
        legacy_path=legacy_path,
        core_path=core_path,
        matches=matches,
        core_facets_seen=len(core_facets),
    )


def load_legacy_facet_pins(path: Path) -> list[LegacyFacetPin]:
    blocks: dict[str, LegacyBlock] = {}
    raw_pins: list[tuple[int, dict[str, Any]]] = []

    with path.open("r", encoding="utf-8") as src:
        for line_number, raw_line in enumerate(src, 1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            data = _system_event_data(record)
            kind = data.get("kind")

            if kind == "block_detected":
                block = _parse_legacy_block(data)
                if block is not None:
                    existing = blocks.get(block.id)
                    facets = existing.facets if existing is not None else {}
                    blocks[block.id] = LegacyBlock(
                        id=block.id,
                        title=block.title,
                        turn_start=block.turn_start,
                        turn_end=block.turn_end,
                        facets=facets,
                    )
                continue

            if kind == "block_artifact_created":
                parsed = _parse_legacy_facet_artifact(data)
                if parsed is not None:
                    block_id, facets = parsed
                    block = blocks.get(block_id)
                    if block is None:
                        continue
                    merged_facets = {**block.facets, **facets}
                    blocks[block_id] = LegacyBlock(
                        id=block.id,
                        title=block.title,
                        turn_start=block.turn_start,
                        turn_end=block.turn_end,
                        facets=merged_facets,
                    )
                continue

            if kind == "facet_pinned":
                raw_pins.append((line_number, data))

    return [
        _resolve_legacy_pin(line_number, data, blocks) for line_number, data in raw_pins
    ]


def load_core_turn_index(path: Path) -> dict[int, list[int]]:
    by_turn: dict[int, list[int]] = defaultdict(list)
    context_block_idx = 0

    with path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("ir") != "event":
                continue
            turn = record.get("turn")
            if isinstance(turn, int):
                by_turn[turn].append(context_block_idx)
            context_block_idx += 1

    return dict(by_turn)


def load_core_facets(path: Path) -> list[CoreFacet]:
    rehydrated = Rehydrator(path).run()
    facets: list[CoreFacet] = []
    for block in rehydrated.semantic_blocks:
        summary = next(
            (artifact for artifact in block.artifacts if artifact.type == "summary"),
            None,
        )
        if summary is None:
            continue
        for facet in summary.facets:
            facets.append(
                CoreFacet(
                    block_id=block.id,
                    block_idx=block.idx,
                    block_title=block.title,
                    facet_id=facet.id,
                    facet_title=facet.title,
                    facet_description=facet.description,
                    start_block=facet.start_block,
                    end_block=facet.end_block,
                )
            )
    return facets


def _match_pin(
    pin: LegacyFacetPin,
    core_index: dict[int, list[int]],
    core_facets: list[CoreFacet],
    *,
    top: int,
) -> FacetPinMatch:
    if not pin.resolved:
        return FacetPinMatch(
            legacy_pin=pin,
            mapped_start_block=None,
            mapped_end_block=None,
            candidates=[],
            warning="Legacy pin could not be resolved to a facet turn range.",
        )

    assert pin.turn_start is not None
    assert pin.turn_end is not None
    mapped_range = map_turn_range_to_core_blocks(
        core_index,
        turn_start=pin.turn_start,
        turn_end=pin.turn_end,
    )
    if mapped_range is None:
        return FacetPinMatch(
            legacy_pin=pin,
            mapped_start_block=None,
            mapped_end_block=None,
            candidates=[],
            warning="No core transcript event blocks were found for this turn range.",
        )

    mapped_start, mapped_end = mapped_range
    candidates = rank_core_facets(
        mapped_start=mapped_start,
        mapped_end=mapped_end,
        core_facets=core_facets,
    )
    return FacetPinMatch(
        legacy_pin=pin,
        mapped_start_block=mapped_start,
        mapped_end_block=mapped_end,
        candidates=candidates[:top],
    )


def map_turn_range_to_core_blocks(
    core_index: dict[int, list[int]],
    *,
    turn_start: int,
    turn_end: int,
) -> tuple[int, int] | None:
    if turn_start > turn_end:
        return None
    indices: list[int] = []
    for turn in range(turn_start, turn_end + 1):
        indices.extend(core_index.get(turn, []))
    if not indices:
        return None
    return min(indices), max(indices)


def rank_core_facets(
    *,
    mapped_start: int,
    mapped_end: int,
    core_facets: list[CoreFacet],
) -> list[FacetCandidate]:
    candidates = [
        _score_candidate(mapped_start, mapped_end, facet) for facet in core_facets
    ]
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.overlap_blocks > 0,
            candidate.jaccard,
            candidate.legacy_coverage,
            candidate.core_coverage,
            -candidate.gap_blocks,
        ),
        reverse=True,
    )


def _score_candidate(
    mapped_start: int,
    mapped_end: int,
    facet: CoreFacet,
) -> FacetCandidate:
    overlap = _inclusive_overlap(
        mapped_start,
        mapped_end,
        facet.start_block,
        facet.end_block,
    )
    legacy_len = _span_len(mapped_start, mapped_end)
    core_len = _span_len(facet.start_block, facet.end_block)
    union = legacy_len + core_len - overlap
    gap = _inclusive_gap(mapped_start, mapped_end, facet.start_block, facet.end_block)
    return FacetCandidate(
        facet=facet,
        overlap_blocks=overlap,
        union_blocks=union,
        legacy_coverage=overlap / legacy_len if legacy_len else 0.0,
        core_coverage=overlap / core_len if core_len else 0.0,
        jaccard=overlap / union if union else 0.0,
        gap_blocks=gap,
    )


def _inclusive_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0, end - start + 1)


def _inclusive_gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    if _inclusive_overlap(a_start, a_end, b_start, b_end):
        return 0
    if a_end < b_start:
        return b_start - a_end
    return a_start - b_end


def _span_len(start: int, end: int) -> int:
    return max(0, end - start + 1)


def _parse_legacy_block(data: dict[str, Any]) -> LegacyBlock | None:
    block_id = data.get("block_id")
    title = data.get("title")
    turn_range = _coerce_range(data.get("turn_range"))
    if (
        not isinstance(block_id, str)
        or not isinstance(title, str)
        or turn_range is None
    ):
        return None
    return LegacyBlock(
        id=block_id,
        title=title,
        turn_start=turn_range[0],
        turn_end=turn_range[1],
    )


def _parse_legacy_facet_artifact(
    data: dict[str, Any],
) -> tuple[str, dict[str, dict[str, Any]]] | None:
    block_id = data.get("block_id")
    artifact = data.get("artifact")
    if not isinstance(block_id, str) or not isinstance(artifact, dict):
        return None
    if artifact.get("kind") != "facet_index":
        return None
    payload = artifact.get("payload")
    if not isinstance(payload, dict):
        return None
    raw_facets = payload.get("facets")
    if not isinstance(raw_facets, list):
        return None

    facets: dict[str, dict[str, Any]] = {}
    for raw_facet in raw_facets:
        if not isinstance(raw_facet, dict):
            continue
        facet_id = raw_facet.get("id")
        if isinstance(facet_id, str):
            facets[facet_id] = raw_facet
    return block_id, facets


def _resolve_legacy_pin(
    line_number: int,
    data: dict[str, Any],
    blocks: dict[str, LegacyBlock],
) -> LegacyFacetPin:
    block_id = data.get("block_id")
    facet_id = data.get("facet_id")
    source = data.get("source", "unknown")
    if not isinstance(block_id, str):
        block_id = "<missing>"
    if not isinstance(facet_id, str):
        facet_id = "<missing>"
    if not isinstance(source, str):
        source = "unknown"

    block = blocks.get(block_id)
    created_facet = data.get("created_facet")
    uses_created_facet = isinstance(created_facet, dict)
    raw_facet = created_facet if uses_created_facet else None
    if raw_facet is None and block is not None:
        raw_facet = block.facets.get(facet_id)

    turn_range = _coerce_range(raw_facet.get("turn_range")) if raw_facet else None
    return LegacyFacetPin(
        block_id=block_id,
        block_title=block.title if block is not None else None,
        facet_id=facet_id,
        facet_title=_legacy_facet_title(raw_facet),
        facet_summary=_legacy_facet_summary(raw_facet),
        turn_start=turn_range[0] if turn_range else None,
        turn_end=turn_range[1] if turn_range else None,
        source=source,
        created_facet=uses_created_facet,
        line_number=line_number,
    )


def _legacy_facet_title(raw_facet: dict[str, Any] | None) -> str | None:
    if raw_facet is None:
        return None
    for key in ("headline", "title", "description", "summary"):
        value = raw_facet.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _legacy_facet_summary(raw_facet: dict[str, Any] | None) -> str | None:
    if raw_facet is None:
        return None
    value = raw_facet.get("summary") or raw_facet.get("description")
    return value if isinstance(value, str) else None


def _coerce_range(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        start = int(value[0])
        end = int(value[1])
    except (TypeError, ValueError):
        return None
    if start > end:
        return None
    return start, end


def _system_event_data(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("ir") != "event":
        return {}
    event = record.get("event")
    if not isinstance(event, dict) or event.get("type") != "system":
        return {}
    data = event.get("data")
    return data if isinstance(data, dict) else {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.translate_legacy_facet_pins",
        description=(
            "Compare legacy pinned facets to a core replay transcript and "
            "print likely core facet pin targets."
        ),
    )
    parser.add_argument(
        "--legacy",
        type=Path,
        default=DEFAULT_LEGACY_TRANSCRIPT,
        help=f"Legacy Spellbook transcript. Defaults to {DEFAULT_LEGACY_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--core",
        type=Path,
        default=DEFAULT_CORE_TRANSCRIPT,
        help=f"Core replay transcript. Defaults to {DEFAULT_CORE_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of candidates to show per legacy pin.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for the full JSON report.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = analyze_legacy_facet_pins(
        legacy_path=args.legacy,
        core_path=args.core,
        top=args.top,
    )
    if args.json_output is not None:
        output_path = args.json_output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print_markdown_report(report)


def print_markdown_report(report: FacetPinTranslationReport) -> None:
    print("# Legacy Facet Pin Translation Candidates")
    print()
    print(f"- Legacy: `{report.legacy_path}`")
    print(f"- Core replay: `{report.core_path}`")
    print(f"- Legacy facet pins: {len(report.matches)}")
    print(f"- Core facets scanned: {report.core_facets_seen}")
    print()

    for index, match in enumerate(report.matches, 1):
        pin = match.legacy_pin
        title = pin.facet_title or pin.facet_id
        print(f"## {index}. {title}")
        print()
        print(f"- Legacy block: `{pin.block_id}` {pin.block_title or ''}".rstrip())
        print(f"- Legacy facet: `{pin.facet_id}`")
        print(f"- Source: `{pin.source}`; legacy line: {pin.line_number}")
        if pin.turn_start is not None and pin.turn_end is not None:
            print(f"- Legacy turns: {pin.turn_start}-{pin.turn_end}")
        if match.mapped_start_block is not None and match.mapped_end_block is not None:
            print(
                "- Core mapped blocks: "
                f"{match.mapped_start_block}-{match.mapped_end_block}"
            )
        if match.warning is not None:
            print(f"- Warning: {match.warning}")
        print()

        if not match.candidates:
            print("_No candidates._")
            print()
            continue

        print(
            "| Rank | Score | Overlap | Legacy Cov. | Core Cov. | Core Block | Core Facet | Range |"
        )
        print("| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
        for rank, candidate in enumerate(match.candidates, 1):
            facet = candidate.facet
            print(
                "| "
                f"{rank} | "
                f"{candidate.jaccard:.3f} | "
                f"{candidate.overlap_blocks} | "
                f"{candidate.legacy_coverage:.1%} | "
                f"{candidate.core_coverage:.1%} | "
                f"{_md_cell(f'[{facet.block_idx}] {facet.block_title}')} | "
                f"{_md_cell(f'`{facet.facet_id}` {facet.facet_title}')} | "
                f"{facet.start_block}-{facet.end_block} |"
            )
        print()


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    main()
