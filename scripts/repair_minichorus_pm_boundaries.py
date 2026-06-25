"""Repair minichorus-pm semantic block boundaries that split tool pairs.

This is a guarded transcript surgery script for the June 2026 minichorus-pm
boundary incident. It applies only ten known field transitions inside
block_detection records' completed/buffered ranges:

  block 13 range fa49c01d...: end_block    2999 -> 3000
  block 14 range 477605c5...: start_block  3000 -> 3001
  block 17 range 3638c944...: end_block    3973 -> 3974
  block 18 range 0e913ef0...: start_block  3974 -> 3975
  block 19 range df74cd7a...: end_block    4454 -> 4455
  block 20 range 4d8a2c22...: start_block  4455 -> 4456
  block 20 range 4d8a2c22...: end_block    4499 -> 4500
  block 21 range 18efb86c...: start_block  4500 -> 4501
  block 21 range 18efb86c...: end_block    4568 -> 4569
  buffer   range b72d2b3a...: start_block  4569 -> 4570

The repair moves each tool_result to the same side of the semantic boundary as
its matching tool_call. It also shifts the current first buffered proposal so it
does not overlap the repaired final completed block. The script patches a copy
by default. Use --execute only after the session is stopped and the owner has
consented.

Usage:
  uv run python scripts/repair_minichorus_pm_boundaries.py --dry-run
  uv run python scripts/repair_minichorus_pm_boundaries.py --execute
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

TRANSCRIPT = Path("/home/rheaton64/.spellbook/sessions/server_20260513_203434.jsonl")
WORKDIR = Path("/tmp/spellbook-boundary-surgery")

RangeField = Literal["start_block", "end_block"]


@dataclass(frozen=True)
class BoundaryEdit:
    range_id: str
    field: RangeField
    old_value: int
    new_value: int
    expected_count: int

    @property
    def key(self) -> str:
        return f"{self.range_id}:{self.field}"


EDITS = [
    BoundaryEdit(
        "block_range_fa49c01d703c497f936c4ef73e12ccab",
        "end_block",
        2999,
        3000,
        2,
    ),
    BoundaryEdit(
        "block_range_477605c568b941c78207796ea193350a",
        "start_block",
        3000,
        3001,
        2,
    ),
    BoundaryEdit(
        "block_range_3638c944edd3456e9b0a0a61ecc5e7d5",
        "end_block",
        3973,
        3974,
        2,
    ),
    BoundaryEdit(
        "block_range_0e913ef00186498190c21151fbda0763",
        "start_block",
        3974,
        3975,
        2,
    ),
    BoundaryEdit(
        "block_range_df74cd7ab6f04ef883ad65818d5ca37e",
        "end_block",
        4454,
        4455,
        2,
    ),
    BoundaryEdit(
        "block_range_4d8a2c22166f4c0db5541e6114b26880",
        "start_block",
        4455,
        4456,
        2,
    ),
    BoundaryEdit(
        "block_range_4d8a2c22166f4c0db5541e6114b26880",
        "end_block",
        4499,
        4500,
        2,
    ),
    BoundaryEdit(
        "block_range_18efb86cfb9c4d368166a16fb75db5c6",
        "start_block",
        4500,
        4501,
        2,
    ),
    BoundaryEdit(
        "block_range_18efb86cfb9c4d368166a16fb75db5c6",
        "end_block",
        4568,
        4569,
        2,
    ),
    BoundaryEdit(
        "block_range_b72d2b3aeae845378b7bbf0c539181c3",
        "start_block",
        4569,
        4570,
        2,
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
                for collection_name in ("completed", "still_buffered"):
                    for entry in record.get(collection_name, []):
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

    bad_counts = {
        edit.key: applied[edit.key]
        for edit in EDITS
        if applied[edit.key] != edit.expected_count
    }
    if bad_counts:
        details = ", ".join(
            f"{key} applied {count}x" for key, count in bad_counts.items()
        )
        raise SystemExit(f"ABORT: unexpected edit application counts; {details}")

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return sum(applied.values())


def verify(path: Path) -> bool:
    result = Rehydrator(transcript_path=path).run()
    context_blocks = result.blocks
    semantic_blocks = result.semantic_blocks
    buffered = result.buffered_semantic_block_ranges
    print(
        f"context blocks: {len(context_blocks)}, semantic blocks: {len(semantic_blocks)}"
    )
    print(f"unfinished turn: {result.is_unfinished_turn}")

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

    if buffered:
        expected_buffer_start = expected_start
        for block in buffered:
            if block.start_block != expected_buffer_start:
                print(
                    "BUFFER TILING BROKEN: "
                    f"range {block.id} starts {block.start_block}, "
                    f"expected {expected_buffer_start}"
                )
                ok = False
            expected_buffer_start = block.end_block + 1

    for edit in EDITS:
        matches = [
            block
            for block in [*(semantic.range for semantic in semantic_blocks), *buffered]
            if block.id == edit.range_id
        ]
        if not matches:
            print(f"VERIFY BROKEN: range {edit.range_id} is missing")
            ok = False
            continue
        if not any(getattr(match, edit.field) == edit.new_value for match in matches):
            print(
                "VERIFY BROKEN: "
                f"range {edit.range_id} has no visible {edit.field}="
                f"{edit.new_value}"
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

    for idx in (13, 14, 17, 18, 19, 20, 21):
        if idx < len(semantic_blocks):
            block = semantic_blocks[idx]
            print(
                f"  idx={idx} [{block.range.start_block},{block.range.end_block}] "
                f"mode={block.mode} title={block.title!r}"
            )
    if buffered:
        first = buffered[0]
        print(
            f"  buffered[0] [{first.start_block},{first.end_block}] "
            f"title={first.title!r}"
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
    target = WORKDIR / "minichorus_pm_boundary_dryrun.jsonl"
    shutil.copy2(source, target)
    return target


def _backup(source: Path) -> Path:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    backup = (
        WORKDIR / f"minichorus_pm_boundary_backup_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )
    shutil.copy2(source, backup)
    return backup


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair minichorus-pm tool_call/tool_result semantic boundary splits."
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
        "--allow-unfinished",
        action="store_true",
        help="Allow --execute even if the transcript has an unfinished turn.",
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

    rehydrated = Rehydrator(transcript_path=transcript).run()
    if args.execute and rehydrated.is_unfinished_turn and not args.allow_unfinished:
        raise SystemExit(
            "ABORT: transcript has an unfinished turn. Stop the session or pass "
            "--allow-unfinished if you intentionally want to patch it anyway."
        )

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
