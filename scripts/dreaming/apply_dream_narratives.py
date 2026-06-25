"""Apply compiled dream chapters as pair narrative semantic block artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter
from scripts.dreaming.dream_merge import DEFAULT_DREAM_CHAPTER_DIR
from scripts.dreaming.merge_stats import (
    DEFAULT_TRANSCRIPT,
    _apply_tool_result_ttls,
    _build_block_manager,
)
from spellbook.ir_types import (
    IRBlock,
    IRRecord,
    IRSemanticBlock,
    IRSemanticBlockApplyModeRecord,
    IRSemanticBlockArtifactRecord,
    IRSemanticBlockPairNarrative,
    IRSemanticBlockPairNarrativeChild,
    IRTokenRangeCount,
    IRToolCallBlock,
    IRToolResultBlock,
    IRTurnEndRecord,
    SemanticBlockMode,
)
from spellbook.rehydrator import Rehydrator

ApplyStatus = Literal["appended", "would_append", "skipped_existing"]


@dataclass(frozen=True)
class ChapterApplyEntry:
    chapter: int
    pair: tuple[int, int]
    narrative_id: str
    compiled_path: Path
    source_chapter_path: str | None
    parent_block_id: str
    parent_block_idx: int
    parent_title: str
    parent_previous_mode: SemanticBlockMode
    child_block_id: str
    child_block_idx: int
    child_title: str
    child_previous_mode: SemanticBlockMode
    status: ApplyStatus
    artifact_records: int
    mode_records: int
    token_count: int | None

    @property
    def records_to_append(self) -> int:
        return self.artifact_records + self.mode_records

    def to_dict(self) -> dict[str, object]:
        return {
            "chapter": self.chapter,
            "pair": list(self.pair),
            "narrative_id": self.narrative_id,
            "compiled_path": str(self.compiled_path),
            "source_chapter_path": self.source_chapter_path,
            "parent": {
                "block_id": self.parent_block_id,
                "block_idx": self.parent_block_idx,
                "title": self.parent_title,
                "previous_mode": self.parent_previous_mode,
            },
            "child": {
                "block_id": self.child_block_id,
                "block_idx": self.child_block_idx,
                "title": self.child_title,
                "previous_mode": self.child_previous_mode,
            },
            "status": self.status,
            "artifact_records": self.artifact_records,
            "mode_records": self.mode_records,
            "records_to_append": self.records_to_append,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class DreamNarrativeApplyReport:
    transcript_path: Path
    chapter_dir: Path
    backup_path: Path | None
    dry_run: bool
    force: bool
    chapters_requested: tuple[int, ...]
    entries: list[ChapterApplyEntry] = field(default_factory=list)

    @property
    def records_to_append(self) -> int:
        return sum(entry.records_to_append for entry in self.entries)

    @property
    def records_appended(self) -> int:
        return 0 if self.dry_run else self.records_to_append

    @property
    def skipped_existing(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "skipped_existing")

    def to_dict(self) -> dict[str, object]:
        return {
            "transcript_path": str(self.transcript_path),
            "chapter_dir": str(self.chapter_dir),
            "backup_path": str(self.backup_path) if self.backup_path else None,
            "dry_run": self.dry_run,
            "force": self.force,
            "chapters_requested": list(self.chapters_requested),
            "chapter_count": len(self.entries),
            "records_to_append": self.records_to_append,
            "records_appended": self.records_appended,
            "skipped_existing": self.skipped_existing,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class _ChapterPlan:
    entry: ChapterApplyEntry
    artifact_records: list[IRSemanticBlockArtifactRecord]
    mode_records: list[IRSemanticBlockApplyModeRecord]

    @property
    def records(self) -> list[IRRecord]:
        return [*self.artifact_records, *self.mode_records]


def apply_dream_narratives(
    *,
    transcript_path: Path = DEFAULT_TRANSCRIPT,
    chapter_dir: Path = DEFAULT_DREAM_CHAPTER_DIR,
    chapters: tuple[int, ...],
    apply: bool = False,
    backup: bool = True,
    validate: bool = True,
    allow_unfinished: bool = False,
    force: bool = False,
    report_json: Path | None = None,
    write_report: bool = True,
) -> DreamNarrativeApplyReport:
    """Append pair narrative artifacts and mode records for compiled chapters."""

    if not chapters:
        raise ValueError("At least one chapter must be selected.")

    transcript_path = transcript_path.expanduser().resolve()
    chapter_dir = chapter_dir.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    if not chapter_dir.exists():
        raise FileNotFoundError(f"Chapter directory not found: {chapter_dir}")

    rehydrated = Rehydrator(transcript_path).run()
    if rehydrated.is_unfinished_turn and not allow_unfinished:
        raise ValueError(
            "Transcript has an unfinished turn. Pass --allow-unfinished if you "
            "really want to append dream narrative records after it."
        )

    turn_id = _last_completed_turn_id(
        rehydrated.records, rehydrated.last_completed_turn
    )
    plans = [
        _chapter_plan(
            chapter=chapter,
            chapter_dir=chapter_dir,
            semantic_blocks=rehydrated.semantic_blocks,
            session_id=rehydrated.session_id,
            turn=rehydrated.last_completed_turn,
            turn_id=turn_id,
            apply=apply,
            force=force,
        )
        for chapter in chapters
    ]

    records = [record for plan in plans for record in plan.records]
    backup_path: Path | None = None
    dry_run = not apply
    if records and apply:
        backup_path = _backup_path(transcript_path) if backup else None
        _append_records(
            transcript_path=transcript_path,
            records=records,
            backup_path=backup_path,
            validate=validate,
        )

    report = DreamNarrativeApplyReport(
        transcript_path=transcript_path,
        chapter_dir=chapter_dir,
        backup_path=backup_path,
        dry_run=dry_run,
        force=force,
        chapters_requested=chapters,
        entries=[plan.entry for plan in plans],
    )
    if write_report:
        _write_report(report, report_json or _default_report_path(transcript_path))
    return report


def _chapter_plan(
    *,
    chapter: int,
    chapter_dir: Path,
    semantic_blocks: list[IRSemanticBlock],
    session_id: str,
    turn: int,
    turn_id: str,
    apply: bool,
    force: bool,
) -> _ChapterPlan:
    first_idx, second_idx = _pair_for_chapter(chapter)
    if second_idx >= len(semantic_blocks):
        raise ValueError(
            f"Chapter {chapter} maps to blocks {first_idx}-{second_idx}, "
            f"but there are only {len(semantic_blocks)} semantic blocks."
        )
    parent_block = semantic_blocks[first_idx]
    child_block = semantic_blocks[second_idx]
    if parent_block.idx != first_idx or child_block.idx != second_idx:
        raise ValueError(
            f"Semantic block indices do not match expected pair {first_idx}-{second_idx}."
        )

    compiled_path = _compiled_path(chapter_dir, chapter)
    compiled = _load_compiled(compiled_path)
    _validate_compiled_target(compiled, chapter=chapter, pair=(first_idx, second_idx))
    ir_blocks = _compiled_ir_blocks(compiled)
    _validate_tool_pairs(ir_blocks)

    token_count = _compiled_token_count(compiled)
    narrative_id = _narrative_id(chapter, first_idx, second_idx)
    source_chapter_path = _source_chapter_path(compiled)

    parent_has_artifact = _has_pair_narrative(parent_block, narrative_id)
    child_has_artifact = _has_pair_narrative(child_block, narrative_id)
    active = (
        parent_block.mode == "pair_narrative"
        and child_block.mode == "pair_narrative"
        and parent_has_artifact
        and child_has_artifact
    )
    should_append_artifacts = force or not (parent_has_artifact and child_has_artifact)
    should_append_parent_mode = force or parent_block.mode != "pair_narrative"
    should_append_child_mode = force or child_block.mode != "pair_narrative"

    artifact_records: list[IRSemanticBlockArtifactRecord] = []
    if should_append_artifacts:
        parent_artifact = IRSemanticBlockPairNarrative(
            narrative_id=narrative_id,
            pair=(first_idx, second_idx),
            chapter_number=chapter,
            title=_compiled_title(compiled, chapter),
            blocks=ir_blocks,
            toks=_token_count(token_count),
            source_chapter_path=source_chapter_path,
            compiled_json_path=str(compiled_path),
        )
        child_artifact = IRSemanticBlockPairNarrativeChild(
            narrative_id=narrative_id,
            parent_block_idx=first_idx,
            parent_block_id=parent_block.id,
            pair=(first_idx, second_idx),
            chapter_number=chapter,
        )
        artifact_records.extend(
            [
                IRSemanticBlockArtifactRecord(
                    session_id=session_id,
                    block_id=parent_block.id,
                    artifact=parent_artifact,
                    turn=turn,
                    turn_id=turn_id,
                ),
                IRSemanticBlockArtifactRecord(
                    session_id=session_id,
                    block_id=child_block.id,
                    artifact=child_artifact,
                    turn=turn,
                    turn_id=turn_id,
                ),
            ]
        )

    mode_records: list[IRSemanticBlockApplyModeRecord] = []
    if should_append_parent_mode:
        mode_records.append(
            _mode_record(
                session_id=session_id,
                block_id=parent_block.id,
                turn=turn,
                turn_id=turn_id,
            )
        )
    if should_append_child_mode:
        mode_records.append(
            _mode_record(
                session_id=session_id,
                block_id=child_block.id,
                turn=turn,
                turn_id=turn_id,
            )
        )

    status: ApplyStatus
    if active and not force:
        status = "skipped_existing"
    else:
        status = "appended" if apply else "would_append"

    entry = ChapterApplyEntry(
        chapter=chapter,
        pair=(first_idx, second_idx),
        narrative_id=narrative_id,
        compiled_path=compiled_path,
        source_chapter_path=source_chapter_path,
        parent_block_id=parent_block.id,
        parent_block_idx=parent_block.idx,
        parent_title=parent_block.title,
        parent_previous_mode=parent_block.mode,
        child_block_id=child_block.id,
        child_block_idx=child_block.idx,
        child_title=child_block.title,
        child_previous_mode=child_block.mode,
        status=status,
        artifact_records=len(artifact_records),
        mode_records=len(mode_records),
        token_count=token_count,
    )
    return _ChapterPlan(
        entry=entry,
        artifact_records=artifact_records,
        mode_records=mode_records,
    )


def _pair_for_chapter(chapter: int) -> tuple[int, int]:
    if chapter < 1:
        raise ValueError(f"Chapter number must be positive: {chapter}.")
    first = (chapter - 1) * 2
    return first, first + 1


def _compiled_path(chapter_dir: Path, chapter: int) -> Path:
    matches = sorted(chapter_dir.glob(f"chapter-{chapter:02d}-*.compiled.json"))
    if not matches:
        raise FileNotFoundError(
            f"No compiled chapter found for chapter {chapter:02d} in {chapter_dir}."
        )
    if len(matches) > 1:
        rendered = "\n".join(f"- {path}" for path in matches)
        raise ValueError(
            f"Multiple compiled chapters found for chapter {chapter:02d}:\n{rendered}"
        )
    return matches[0].resolve()


def _load_compiled(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Compiled chapter JSON is invalid: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Compiled chapter JSON must be an object: {path}")
    return data


def _validate_compiled_target(
    compiled: dict[str, object],
    *,
    chapter: int,
    pair: tuple[int, int],
) -> None:
    chapter_obj = compiled.get("chapter")
    if (
        not isinstance(chapter_obj, dict)
        or int(chapter_obj.get("number", -1)) != chapter
    ):
        raise ValueError(f"Compiled chapter number does not match chapter {chapter}.")

    target = compiled.get("target")
    if not isinstance(target, dict):
        raise ValueError(f"Compiled chapter {chapter} has no target metadata.")
    raw_pair = target.get("pair")
    if not isinstance(raw_pair, list | tuple) or tuple(raw_pair) != pair:
        raise ValueError(
            f"Compiled chapter {chapter} targets pair {raw_pair}, expected {pair}."
        )


def _compiled_ir_blocks(compiled: dict[str, object]) -> list[IRBlock]:
    compiled_obj = compiled.get("compiled")
    if not isinstance(compiled_obj, dict):
        raise ValueError("Compiled chapter has no compiled payload.")
    raw_blocks = compiled_obj.get("ir_blocks")
    if not isinstance(raw_blocks, list):
        raise ValueError("Compiled chapter payload has no ir_blocks list.")
    adapter = TypeAdapter(list[IRBlock])
    return adapter.validate_python(raw_blocks)


def _validate_tool_pairs(blocks: list[IRBlock]) -> None:
    seen_calls: dict[str, str] = {}
    open_calls: set[str] = set()
    seen_results: set[str] = set()
    for idx, block in enumerate(blocks):
        if isinstance(block, IRToolCallBlock):
            if block.call_id in seen_calls:
                raise ValueError(
                    f"Duplicate tool call id {block.call_id} at block {idx}."
                )
            seen_calls[block.call_id] = block.tool
            open_calls.add(block.call_id)
        elif isinstance(block, IRToolResultBlock):
            if block.call_id not in seen_calls:
                raise ValueError(
                    f"Tool result {block.call_id} at block {idx} has no prior call."
                )
            if block.call_id in seen_results:
                raise ValueError(
                    f"Duplicate tool result for call {block.call_id} at block {idx}."
                )
            if block.tool != seen_calls[block.call_id]:
                raise ValueError(
                    f"Tool result {block.call_id} at block {idx} is for tool "
                    f"{block.tool}, expected {seen_calls[block.call_id]}."
                )
            seen_results.add(block.call_id)
            open_calls.discard(block.call_id)
    if open_calls:
        missing = ", ".join(sorted(open_calls))
        raise ValueError(f"Tool calls without matching results: {missing}")


def _compiled_token_count(compiled: dict[str, object]) -> int | None:
    compiled_obj = compiled.get("compiled")
    if not isinstance(compiled_obj, dict):
        return None
    value = compiled_obj.get("token_count")
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _source_chapter_path(compiled: dict[str, object]) -> str | None:
    chapter_obj = compiled.get("chapter")
    if not isinstance(chapter_obj, dict):
        return None
    value = chapter_obj.get("source_path")
    return str(value) if value is not None else None


def _compiled_title(compiled: dict[str, object], chapter: int) -> str:
    chapter_obj = compiled.get("chapter")
    if isinstance(chapter_obj, dict):
        title = chapter_obj.get("title")
        if title:
            return str(title)
    return f"Dream Chapter {chapter:02d}"


def _token_count(tokens: int | None) -> IRTokenRangeCount | None:
    if tokens is None:
        return None
    return IRTokenRangeCount(tokens=tokens, method="api", exact=True)


def _narrative_id(chapter: int, first: int, second: int) -> str:
    return f"dream_chapter_{chapter:02d}_blocks_{first}_{second}"


def _has_pair_narrative(block: IRSemanticBlock, narrative_id: str) -> bool:
    return any(
        getattr(artifact, "narrative_id", None) == narrative_id
        and artifact.mode == "pair_narrative"
        for artifact in block.artifacts
    )


def _mode_record(
    *,
    session_id: str,
    block_id: str,
    turn: int,
    turn_id: str,
) -> IRSemanticBlockApplyModeRecord:
    return IRSemanticBlockApplyModeRecord(
        session_id=session_id,
        block_id=block_id,
        mode="pair_narrative",
        source="model",
        turn=turn,
        turn_id=turn_id,
    )


def _last_completed_turn_id(records: list[IRRecord], last_completed_turn: int) -> str:
    for record in reversed(records):
        if isinstance(record, IRTurnEndRecord) and record.turn == last_completed_turn:
            return record.turn_id
    raise ValueError(
        "Could not find the last completed turn_id needed for appended records."
    )


def _append_records(
    *,
    transcript_path: Path,
    records: list[IRRecord],
    backup_path: Path | None,
    validate: bool,
) -> None:
    original_text = transcript_path.read_text(encoding="utf-8")
    append_text = "".join(record.model_dump_json() + "\n" for record in records)
    tmp_path = transcript_path.with_name(transcript_path.name + ".tmp")
    try:
        tmp_path.write_text(
            _ensure_trailing_newline(original_text) + append_text,
            encoding="utf-8",
        )
        if validate:
            _validate_transcript(tmp_path)
        if backup_path is not None:
            shutil.copy2(transcript_path, backup_path)
        tmp_path.replace(transcript_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _validate_transcript(transcript_path: Path) -> None:
    rehydrated = Rehydrator(transcript_path).run()
    manager = _build_block_manager(rehydrated)
    context: list[IRBlock] = []
    for block in manager.semantic_blocks:
        context.extend(manager.render_block(semantic_block=block))
    context.extend(manager.render_tail())
    _apply_tool_result_ttls(rehydrated, context)


def _ensure_trailing_newline(text: str) -> str:
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def _backup_path(transcript_path: Path) -> Path:
    candidate = transcript_path.with_suffix(
        transcript_path.suffix + ".dream-narratives.bak"
    )
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return transcript_path.with_suffix(
        transcript_path.suffix + f".dream-narratives.bak.{timestamp}"
    )


def _default_report_path(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".dream-narratives-report.json")


def _write_report(report: DreamNarrativeApplyReport, report_path: Path) -> None:
    report_path = report_path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
                    f"Chapter range ends before it starts: {part}"
                )
            values = range(start, end + 1)
        else:
            values = [int(part)]
        for chapter in values:
            if chapter < 1:
                raise argparse.ArgumentTypeError(
                    f"Chapter numbers must be positive: {chapter}"
                )
            if chapter not in seen:
                chapters.append(chapter)
                seen.add(chapter)
    if not chapters:
        raise argparse.ArgumentTypeError("No chapters selected.")
    return tuple(chapters)


def _all_compiled_chapters(chapter_dir: Path) -> tuple[int, ...]:
    chapters: list[int] = []
    for path in sorted(
        chapter_dir.expanduser().resolve().glob("chapter-*.compiled.json")
    ):
        prefix = path.name.removeprefix("chapter-").split("-", maxsplit=1)[0]
        try:
            chapters.append(int(prefix))
        except ValueError:
            continue
    if not chapters:
        raise ValueError(f"No compiled chapters found in {chapter_dir}.")
    return tuple(sorted(set(chapters)))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="apply_dream_narratives",
        description="Append compiled dream chapters as pair narrative memory artifacts.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=DEFAULT_TRANSCRIPT,
        help=f"Meta-Claude transcript path. Defaults to {DEFAULT_TRANSCRIPT}.",
    )
    parser.add_argument(
        "--chapter-dir",
        type=Path,
        default=DEFAULT_DREAM_CHAPTER_DIR,
        help=f"Compiled chapter directory. Defaults to {DEFAULT_DREAM_CHAPTER_DIR}.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--chapters",
        type=_parse_chapters,
        help='Chapters to apply, e.g. "1-26" or "1,3,5-7".',
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Apply every compiled chapter found in --chapter-dir.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Append records to the transcript. Without this, only report the plan.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Append fresh artifacts/mode records even when chapters already appear applied.",
    )
    parser.add_argument(
        "--allow-unfinished",
        action="store_true",
        help="Allow appending after a transcript with an unfinished turn.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Append without creating transcript.jsonl.dream-narratives.bak.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validating the rewritten transcript before replacement.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for the report JSON. Defaults beside the transcript.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    chapters = (
        _all_compiled_chapters(args.chapter_dir) if args.all else tuple(args.chapters)
    )
    report_path = args.report_json or _default_report_path(
        args.transcript.expanduser().resolve()
    )
    report = apply_dream_narratives(
        transcript_path=args.transcript,
        chapter_dir=args.chapter_dir,
        chapters=chapters,
        apply=args.apply,
        backup=not args.no_backup,
        validate=not args.no_validate,
        allow_unfinished=args.allow_unfinished,
        force=args.force,
        report_json=args.report_json,
    )
    _print_report(report, report_path)


def _print_report(report: DreamNarrativeApplyReport, report_path: Path) -> None:
    mode = "Dry run" if report.dry_run else "Applied dream narratives"
    print(f"{mode}: {report.transcript_path}")
    print(f"Chapters: {len(report.entries)}")
    print(f"Records to append: {report.records_to_append}")
    if not report.dry_run:
        print(f"Records appended: {report.records_appended}")
    print(f"Skipped existing: {report.skipped_existing}")
    print("")
    print("Chapter  Pair   Artifacts  Modes  Status            Tokens  Parent")
    print(
        "-------  -----  ---------  -----  ----------------  ------  --------------------"
    )
    for entry in report.entries:
        tokens = "-" if entry.token_count is None else f"{entry.token_count:,}"
        print(
            f"{entry.chapter:>7}  "
            f"{entry.pair[0]}-{entry.pair[1]:<3}  "
            f"{entry.artifact_records:>9}  "
            f"{entry.mode_records:>5}  "
            f"{entry.status:<16}  "
            f"{tokens:>6}  "
            f"{entry.parent_title[:20]:<20}"
        )
    if report.backup_path is not None:
        print(f"\nBackup: {report.backup_path}")
    print(f"Report: {report_path.expanduser().resolve()}")


if __name__ == "__main__":
    main()
