"""Repair mail-ui-bard semantic block boundaries that split tool pairs.

This is a guarded transcript surgery script for the June 2026 mail-ui-bard
boundary incident. It edits only three known integers inside block_detection
records' completed ranges:

  block 3  range 7d0c6919...: end_block   249 -> 250
  block 4  range 4f87d931...: start_block 250 -> 251
  block 5  range 59784664...: end_block   367 -> 368

The repair moves each tool_result to the same side of the semantic boundary as
its matching tool_call. The script patches a copy by default. Use --execute only
after the session is stopped and the owner has consented.

Usage:
  uv run python scripts/repair_mail_ui_bard_boundaries.py --dry-run
  uv run python scripts/repair_mail_ui_bard_boundaries.py --execute
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from spellbook.rehydrator import Rehydrator

TRANSCRIPT = Path(
    "/home/rheaton64/.chorus/spellbook/sessions/mail-ui-bard/transcript.jsonl"
)
WORKDIR = Path("/home/rheaton64/.forge/spellbook/workspace/surgery")

RangeField = Literal["start_block", "end_block"]


@dataclass(frozen=True)
class BoundaryEdit:
    range_id: str
    field: RangeField
    old_value: int
    new_value: int

    @property
    def key(self) -> str:
        return f"{self.range_id}:{self.field}"


EDITS = [
    BoundaryEdit(
        "block_range_7d0c691910ca4717bdf2a4b681abbabc",
        "end_block",
        249,
        250,
    ),
    BoundaryEdit(
        "block_range_4f87d9318897466fabe11e77be820237",
        "start_block",
        250,
        251,
    ),
    BoundaryEdit(
        "block_range_5978466406de4eb090cab27b7226fb8b",
        "end_block",
        367,
        368,
    ),
]


def patch(path: Path) -> int:
    applied = {edit.key: 0 for edit in EDITS}
    range_ids = {edit.range_id for edit in EDITS}
    out_lines: list[str] = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if '"ir":"block_detection"' in stripped and any(
                range_id in stripped for range_id in range_ids
            ):
                record = json.loads(stripped)
                changed = False
                for entry in record.get("completed", []):
                    for edit in EDITS:
                        if (
                            entry.get("id") == edit.range_id
                            and entry.get(edit.field) == edit.old_value
                        ):
                            entry[edit.field] = edit.new_value
                            applied[edit.key] += 1
                            changed = True
                if changed:
                    stripped = json.dumps(record, separators=(",", ":"))
            out_lines.append(stripped)

    bad_counts = {key: count for key, count in applied.items() if count != 1}
    if bad_counts:
        details = ", ".join(
            f"{key} applied {count}x" for key, count in bad_counts.items()
        )
        raise SystemExit(f"ABORT: expected each edit to apply exactly once; {details}")

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return sum(applied.values())


def verify(path: Path) -> bool:
    result = Rehydrator(transcript_path=path).run()
    context_blocks = result.blocks
    semantic_blocks = result.semantic_blocks
    print(
        f"context blocks: {len(context_blocks)}, semantic blocks: {len(semantic_blocks)}"
    )

    ok = True
    expected_start = 0
    for expected_idx, block in enumerate(semantic_blocks):
        if block.idx != expected_idx:
            print(f"TILING BROKEN: expected idx {expected_idx}, got {block.idx}")
            ok = False
        if block.range.start_block != expected_start:
            print(
                "TILING BROKEN: "
                f"idx={block.idx} starts {block.range.start_block}, "
                f"expected {expected_start}"
            )
            ok = False
        expected_start = block.range.end_block + 1

    for edit in EDITS:
        matches = [
            block for block in semantic_blocks if block.range.id == edit.range_id
        ]
        if len(matches) != 1:
            print(
                f"VERIFY BROKEN: range {edit.range_id} has {len(matches)} semantic matches"
            )
            ok = False
            continue
        actual = getattr(matches[0].range, edit.field)
        if actual != edit.new_value:
            print(
                "VERIFY BROKEN: "
                f"range {edit.range_id} {edit.field} is {actual}, "
                f"expected {edit.new_value}"
            )
            ok = False

    splits = scan_pair_splits(path)
    for split in splits:
        print(
            "SPLIT REMAINS: "
            f"idx={split.block_idx} ctx[{split.boundary_block}] "
            f"call_id={split.call_id} -> {split.next_location}"
        )
        ok = False

    for idx in (3, 4, 5):
        if idx < len(semantic_blocks):
            block = semantic_blocks[idx]
            print(
                f"  idx={idx} [{block.range.start_block},{block.range.end_block}] "
                f"mode={block.mode} title={block.title!r}"
            )

    print(f"splits={len(splits)}")
    print("VERIFY:", "PASS" if ok else "FAIL")
    return ok


@dataclass(frozen=True)
class PairSplit:
    block_idx: int
    boundary_block: int
    call_id: str
    next_location: str


def scan_pair_splits(path: Path) -> list[PairSplit]:
    result = Rehydrator(transcript_path=path).run()
    context_blocks = result.blocks
    semantic_blocks = result.semantic_blocks
    splits: list[PairSplit] = []

    for block in semantic_blocks:
        end = block.range.end_block
        if end + 1 >= len(context_blocks):
            continue
        last = context_blocks[end]
        next_block = context_blocks[end + 1]
        if (
            getattr(last, "type", None) == "tool_call"
            and getattr(next_block, "type", None) == "tool_result"
            and getattr(last, "call_id", "a") == getattr(next_block, "call_id", "b")
        ):
            next_location = "unblocked tail"
            next_idx = block.idx + 1
            if (
                next_idx < len(semantic_blocks)
                and semantic_blocks[next_idx].range.start_block == end + 1
            ):
                next_location = f"block idx={next_idx}"
            splits.append(
                PairSplit(
                    block_idx=block.idx,
                    boundary_block=end,
                    call_id=last.call_id,
                    next_location=next_location,
                )
            )

    return splits


def _copy_for_dry_run(source: Path) -> Path:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    target = WORKDIR / "mail_ui_bard_boundary_dryrun.jsonl"
    shutil.copy2(source, target)
    return target


def _backup(source: Path) -> Path:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    backup = (
        WORKDIR / f"mail_ui_bard_boundary_backup_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )
    shutil.copy2(source, backup)
    return backup


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair mail-ui-bard tool_call/tool_result semantic boundary splits."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Patch a copy under the surgery workdir and verify it. This is the default.",
    )
    group.add_argument(
        "--execute",
        action="store_true",
        help="Backup and patch the real transcript. Stop the session before using this.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=TRANSCRIPT,
        help=f"Transcript to repair. Defaults to {TRANSCRIPT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    transcript = args.transcript.expanduser().resolve()
    if not transcript.exists():
        raise SystemExit(f"Transcript not found: {transcript}")

    if args.execute:
        backup = _backup(transcript)
        print(f"backup written: {backup}")
        target = transcript
    else:
        target = _copy_for_dry_run(transcript)
        print(f"dry-run copy: {target}")

    applied = patch(target)
    print(f"applied {applied} field edits")
    if not verify(target):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
