"""Repair legacy-imported tool result ordering.

Legacy transcripts can contain user text between an assistant tool_use run and
the corresponding tool_result blocks. Anthropic rejects that provider projection:
the next user message after tool_use must start with those tool_result blocks.

This module performs a conservative transcript-level repair by reordering
existing IRBlockRecord records within their local event segment. It does not
invent missing tool results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRBlockRecord,
    IRImageBlock,
    IRRecord,
    IRSemanticBlockRange,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRUserTextBlock,
)


@dataclass(frozen=True)
class ToolResultOrderMove:
    turn: int
    turn_id: str | None
    call_id: str
    tool: str
    original_block_index: int
    repaired_block_index: int
    semantic_range_id: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "turn": self.turn,
            "turn_id": self.turn_id,
            "call_id": self.call_id,
            "tool": self.tool,
            "original_block_index": self.original_block_index,
            "repaired_block_index": self.repaired_block_index,
            "semantic_range_id": self.semantic_range_id,
        }


@dataclass(frozen=True)
class ToolResultOrderRepairReport:
    moves: list[ToolResultOrderMove] = field(default_factory=list)
    skipped_cross_boundary: int = 0

    @property
    def records_moved(self) -> int:
        return len(self.moves)

    @property
    def turns_repaired(self) -> int:
        return len({move.turn for move in self.moves})

    def to_dict(self) -> dict[str, object]:
        return {
            "records_moved": self.records_moved,
            "turns_repaired": self.turns_repaired,
            "skipped_cross_boundary": self.skipped_cross_boundary,
            "moves": [move.to_dict() for move in self.moves],
        }


@dataclass(frozen=True)
class _BlockRecordRef:
    record: IRBlockRecord
    block_index: int
    semantic_range_id: str | None


def repair_tool_result_order_records(
    records: Sequence[IRRecord],
    *,
    semantic_ranges: Sequence[IRSemanticBlockRange] | None = None,
) -> tuple[list[IRRecord], ToolResultOrderRepairReport]:
    """Return records with legacy tool_result ordering repaired.

    Repairs are limited to contiguous IRBlockRecord segments in a single turn.
    When semantic ranges are supplied, any candidate move that would cross a
    semantic range boundary is skipped.
    """

    range_ids_by_block_index = _range_ids_by_block_index(semantic_ranges or [])
    indexed: list[IRRecord | _BlockRecordRef] = []
    block_index = 0
    for record in records:
        if isinstance(record, IRBlockRecord):
            indexed.append(
                _BlockRecordRef(
                    record=record,
                    block_index=block_index,
                    semantic_range_id=range_ids_by_block_index.get(block_index),
                )
            )
            block_index += 1
        else:
            indexed.append(record)

    repaired_indexed, report = _repair_indexed_records(indexed)
    repaired_records = [
        item.record if isinstance(item, _BlockRecordRef) else item
        for item in repaired_indexed
    ]
    return _renumber_event_sequences(repaired_records), report


def _repair_indexed_records(
    indexed: list[IRRecord | _BlockRecordRef],
) -> tuple[list[IRRecord | _BlockRecordRef], ToolResultOrderRepairReport]:
    repaired: list[IRRecord | _BlockRecordRef] = list(indexed)
    moves: list[ToolResultOrderMove] = []
    skipped_cross_boundary = 0

    segment_start: int | None = None
    for idx, item in enumerate([*indexed, None]):
        if isinstance(item, _BlockRecordRef):
            if segment_start is None:
                segment_start = idx
            continue

        if segment_start is not None:
            segment = indexed[segment_start:idx]
            assert all(isinstance(ref, _BlockRecordRef) for ref in segment)
            repaired_segment, segment_moves, skipped = _repair_segment(
                [ref for ref in segment if isinstance(ref, _BlockRecordRef)]
            )
            repaired[segment_start:idx] = repaired_segment
            moves.extend(segment_moves)
            skipped_cross_boundary += skipped
            segment_start = None

    return repaired, ToolResultOrderRepairReport(
        moves=moves,
        skipped_cross_boundary=skipped_cross_boundary,
    )


def _repair_segment(
    segment: list[_BlockRecordRef],
) -> tuple[list[_BlockRecordRef], list[ToolResultOrderMove], int]:
    if len(segment) < 2:
        return segment, [], 0

    repaired: list[_BlockRecordRef] = []
    moves: list[ToolResultOrderMove] = []
    skipped_cross_boundary = 0
    runs = _role_runs(segment)

    idx = 0
    while idx < len(runs):
        role, refs = runs[idx]
        if role != "assistant":
            candidate = _repair_user_run_tool_results_first(refs)
            if candidate == refs:
                repaired.extend(refs)
            elif _crosses_semantic_boundary(refs, candidate):
                skipped_cross_boundary += 1
                repaired.extend(refs)
            else:
                repaired.extend(candidate)
                moves.extend(_moves_for_reordered_run(refs, candidate))
            idx += 1
            continue

        repaired.extend(refs)
        tool_calls = [
            ref for ref in refs if isinstance(ref.record.event, IRToolCallBlock)
        ]
        if not tool_calls:
            idx += 1
            continue

        if idx + 1 >= len(runs) or runs[idx + 1][0] != "user":
            idx += 1
            continue

        next_user_refs = runs[idx + 1][1]
        candidate = _repair_user_run_after_tool_calls(tool_calls, next_user_refs)
        if candidate == next_user_refs:
            repaired.extend(next_user_refs)
        elif _crosses_semantic_boundary(next_user_refs, candidate):
            skipped_cross_boundary += 1
            repaired.extend(next_user_refs)
        else:
            repaired.extend(candidate)
            moves.extend(_moves_for_reordered_run(next_user_refs, candidate))
        idx += 2

    return repaired, moves, skipped_cross_boundary


def _repair_user_run_after_tool_calls(
    tool_calls: list[_BlockRecordRef],
    user_refs: list[_BlockRecordRef],
) -> list[_BlockRecordRef]:
    results_by_call_id: dict[str, _BlockRecordRef] = {
        ref.record.event.call_id: ref
        for ref in user_refs
        if isinstance(ref.record.event, IRToolResultBlock)
    }

    ordered_results: list[_BlockRecordRef] = []
    moved_result_ids: set[str] = set()
    for call_ref in tool_calls:
        call = call_ref.record.event
        assert isinstance(call, IRToolCallBlock)
        result_ref = results_by_call_id.get(call.call_id)
        if result_ref is not None:
            ordered_results.append(result_ref)
            moved_result_ids.add(call.call_id)

    if not ordered_results:
        return user_refs

    remaining = [
        ref
        for ref in user_refs
        if not (
            isinstance(ref.record.event, IRToolResultBlock)
            and ref.record.event.call_id in moved_result_ids
        )
    ]
    extra_results = [
        ref for ref in remaining if isinstance(ref.record.event, IRToolResultBlock)
    ]
    non_results = [
        ref for ref in remaining if not isinstance(ref.record.event, IRToolResultBlock)
    ]
    candidate = [*ordered_results, *extra_results, *non_results]
    return candidate if candidate != user_refs else user_refs


def _repair_user_run_tool_results_first(
    user_refs: list[_BlockRecordRef],
) -> list[_BlockRecordRef]:
    results = [
        ref for ref in user_refs if isinstance(ref.record.event, IRToolResultBlock)
    ]
    if not results:
        return user_refs
    non_results = [
        ref for ref in user_refs if not isinstance(ref.record.event, IRToolResultBlock)
    ]
    candidate = [*results, *non_results]
    return candidate if candidate != user_refs else user_refs


def _role_runs(
    segment: list[_BlockRecordRef],
) -> list[tuple[str, list[_BlockRecordRef]]]:
    runs: list[tuple[str, list[_BlockRecordRef]]] = []
    for ref in segment:
        role = _provider_role(ref.record.event)
        if not runs or runs[-1][0] != role:
            runs.append((role, [ref]))
        else:
            runs[-1][1].append(ref)
    return runs


def _provider_role(block: IRBlock) -> str:
    if isinstance(block, IRUserTextBlock | IRImageBlock | IRToolResultBlock):
        return "user"
    if isinstance(block, IRAssistantTextBlock | IRThinkingBlock | IRToolCallBlock):
        return "assistant"
    raise TypeError(f"Unsupported IR block type: {type(block)}")


def _crosses_semantic_boundary(
    original: list[_BlockRecordRef],
    candidate: list[_BlockRecordRef],
) -> bool:
    for original_ref, candidate_ref in zip(original, candidate, strict=True):
        if original_ref.semantic_range_id != candidate_ref.semantic_range_id:
            return True
    return False


def _moves_for_reordered_run(
    original: list[_BlockRecordRef],
    candidate: list[_BlockRecordRef],
) -> list[ToolResultOrderMove]:
    moves: list[ToolResultOrderMove] = []
    for offset, (original_ref, candidate_ref) in enumerate(
        zip(original, candidate, strict=True)
    ):
        if original_ref is candidate_ref:
            continue
        block = candidate_ref.record.event
        if not isinstance(block, IRToolResultBlock):
            continue
        destination = original[offset]
        moves.append(
            ToolResultOrderMove(
                turn=candidate_ref.record.turn,
                turn_id=block.turn_id,
                call_id=block.call_id,
                tool=block.tool,
                original_block_index=candidate_ref.block_index,
                repaired_block_index=destination.block_index,
                semantic_range_id=candidate_ref.semantic_range_id,
            )
        )
    return moves


def _renumber_event_sequences(records: list[IRRecord]) -> list[IRRecord]:
    next_seq_by_turn: dict[int, int] = {}
    renumbered: list[IRRecord] = []
    for record in records:
        if not isinstance(record, IRBlockRecord):
            renumbered.append(record)
            continue
        seq = next_seq_by_turn.get(record.turn, 0)
        next_seq_by_turn[record.turn] = seq + 1
        renumbered.append(record.model_copy(update={"seq": seq}))
    return renumbered


def _range_ids_by_block_index(
    ranges: Sequence[IRSemanticBlockRange],
) -> dict[int, str]:
    result: dict[int, str] = {}
    for semantic_range in ranges:
        for block_index in range(
            semantic_range.start_block,
            semantic_range.end_block + 1,
        ):
            result[block_index] = semantic_range.id
    return result
