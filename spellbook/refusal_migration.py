"""Audited, append-only refusal rendering amendments for legacy transcripts.

The migration does not rewrite historical blocks. It appends an operator-owned
runtime-config record selecting a refusal projection, while the compatibility
renderer normalizes legacy flattened refusal strings in derived request
surfaces. Block indexes and all inherited transcript bytes remain unchanged.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlockDetectionRecord,
    IRBlockRecord,
    IRRecord,
    IRRefusalBlock,
    IRRuntimeConfigRecord,
    IRSemanticBlockApplyModeRecord,
    IRSemanticBlockArtifact,
    IRSemanticBlockArtifactRecord,
    IRSemanticBlockRange,
    IRSemanticBlockRecord,
    IRSessionRecord,
    IRTurnEndRecord,
    IRTurnStartRecord,
    IRUserTextBlock,
)
from spellbook.refusal import (
    REFUSAL_OPEN,
    REFUSAL_RUNTIME_CONFIG_NAMESPACE,
    SYSTEM_INTERRUPTION_TEXT,
    RefusalRenderer,
    RefusalRenderPolicy,
    canonical_refusal,
    canonicalize_legacy_refusal,
    refusal_policy_from_runtime_config_records,
    refusal_policy_runtime_values,
)

_adapter: TypeAdapter[IRRecord] = TypeAdapter(IRRecord)
_DERIVED_REFUSAL_TERMS = ("refusal", "classifier", "stop_reason", "arousal")


class RefusalTranscriptAnalysis(BaseModel, frozen=True):
    """Preflight facts for one transcript and proposed rendering policy."""

    model_config = ConfigDict(extra="forbid")
    path: str
    source_sha256: str
    record_count: int
    block_count: int
    last_completed_turn: int
    unfinished_turn: bool
    proposed_policy: RefusalRenderPolicy
    explicit_policy: RefusalRenderPolicy | None = None
    refusal_turns: list[int] = Field(default_factory=list)
    legacy_refusal_turns: list[int] = Field(default_factory=list)
    canonical_refusal_turns: list[int] = Field(default_factory=list)
    lifecycle_only_refusal_turns: list[int] = Field(default_factory=list)
    ambiguous_refusal_turns: list[int] = Field(default_factory=list)
    zero_partial_refusals: int = 0
    partial_refusals: int = 0
    thinking_summary_refusals: int = 0
    projected_system_notes: int = 0
    projected_assistant_partials: int = 0
    projected_refusal_markers: int = 0
    refusals_in_materialized_semantic_blocks: list[int] = Field(default_factory=list)
    refusals_in_non_full_semantic_blocks: list[int] = Field(default_factory=list)
    active_derived_blocks_with_refusal_language: list[int] = Field(default_factory=list)
    safe_to_amend: bool


def analyze_refusal_transcript(
    path: Path,
    policy: RefusalRenderPolicy,
) -> RefusalTranscriptAnalysis:
    """Read and validate a transcript without changing it."""
    source = path.expanduser().resolve()
    payload = source.read_bytes()
    records = _parse_records(source, payload)
    refusal_turns = sorted(
        {
            record.turn
            for record in records
            if isinstance(record, IRTurnEndRecord) and record.stop_reason == "refusal"
        }
    )
    refusal_turn_set = set(refusal_turns)
    represented: dict[int, list[tuple[int, IRRefusalBlock, str]]] = {
        turn: [] for turn in refusal_turns
    }
    block_count = 0
    for record in records:
        if not isinstance(record, IRBlockRecord):
            continue
        block_idx = block_count
        block_count += 1
        if record.turn not in refusal_turn_set:
            continue
        refusal = canonical_refusal(record.event) or canonicalize_legacy_refusal(
            record.event
        )
        if refusal is None:
            continue
        shape = "canonical" if isinstance(record.event, IRRefusalBlock) else "legacy"
        represented[record.turn].append((block_idx, refusal, shape))

    ranges, semantic_records, modes, artifacts = _semantic_state(records)
    materialized: list[int] = []
    non_full: list[int] = []
    for turn, matches in represented.items():
        for block_idx, _, _ in matches:
            owner = _semantic_owner(
                block_idx,
                ranges=ranges,
                semantic_records=semantic_records,
            )
            if owner is None:
                continue
            materialized.append(turn)
            if modes.get(owner.id, "full") != "full":
                non_full.append(turn)

    derived_language: list[int] = []
    for block_id, semantic_record in semantic_records.items():
        if modes.get(block_id, "full") == "full":
            continue
        active_artifacts = [
            artifact
            for artifact in artifacts.get(block_id, [])
            if artifact.mode == modes[block_id]
        ]
        if any(
            any(
                term in artifact.model_dump_json().lower()
                for term in _DERIVED_REFUSAL_TERMS
            )
            for artifact in active_artifacts
        ):
            derived_language.append(semantic_record.idx)

    renderer = RefusalRenderer(policy)
    projected_notes = 0
    projected_partials = 0
    projected_markers = 0
    zero_partial = 0
    partial = 0
    with_summary = 0
    legacy_turns: list[int] = []
    canonical_turns: list[int] = []
    for turn in refusal_turns:
        for _, refusal, shape in represented[turn]:
            if shape == "legacy":
                legacy_turns.append(turn)
            else:
                canonical_turns.append(turn)
            if refusal.partial_text:
                partial += 1
            else:
                zero_partial += 1
            if any(segment.kind == "thinking_summary" for segment in refusal.segments):
                with_summary += 1
            for block in renderer.render_refusal(refusal):
                if isinstance(block, IRAssistantTextBlock):
                    projected_partials += 1
                elif (
                    isinstance(block, IRUserTextBlock)
                    and block.text == SYSTEM_INTERRUPTION_TEXT
                ):
                    projected_notes += 1
                if isinstance(block, IRAssistantTextBlock | IRUserTextBlock):
                    projected_markers += block.text.count(REFUSAL_OPEN)

    ambiguous = sorted(
        turn for turn, matches in represented.items() if len(matches) > 1
    )
    lifecycle_only = sorted(
        turn for turn, matches in represented.items() if not matches
    )
    last_completed, unfinished = _turn_state(records)
    runtime_records = [
        record for record in records if isinstance(record, IRRuntimeConfigRecord)
    ]
    safe = not unfinished and not ambiguous and projected_markers == 0
    return RefusalTranscriptAnalysis(
        path=str(source),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        record_count=len(records),
        block_count=block_count,
        last_completed_turn=last_completed,
        unfinished_turn=unfinished,
        proposed_policy=policy,
        explicit_policy=refusal_policy_from_runtime_config_records(runtime_records),
        refusal_turns=refusal_turns,
        legacy_refusal_turns=sorted(legacy_turns),
        canonical_refusal_turns=sorted(canonical_turns),
        lifecycle_only_refusal_turns=lifecycle_only,
        ambiguous_refusal_turns=ambiguous,
        zero_partial_refusals=zero_partial,
        partial_refusals=partial,
        thinking_summary_refusals=with_summary,
        projected_system_notes=projected_notes,
        projected_assistant_partials=projected_partials,
        projected_refusal_markers=projected_markers,
        refusals_in_materialized_semantic_blocks=sorted(set(materialized)),
        refusals_in_non_full_semantic_blocks=sorted(set(non_full)),
        active_derived_blocks_with_refusal_language=sorted(set(derived_language)),
        safe_to_amend=safe,
    )


def write_amended_copy(
    source: Path,
    destination: Path,
    policy: RefusalRenderPolicy,
) -> RefusalTranscriptAnalysis:
    """Create a byte-prefix-identical copy with one appended policy record."""
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Refusing to overwrite {destination}.")
    analysis = analyze_refusal_transcript(source, policy)
    _require_safe(analysis)
    record = _policy_record(source, policy)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "xb") as dest:
        payload = source.read_bytes()
        dest.write(payload)
        if payload and not payload.endswith(b"\n"):
            dest.write(b"\n")
        dest.write(record.model_dump_json().encode() + b"\n")
        dest.flush()
        os.fsync(dest.fileno())
    _verify_appended_copy(source, destination)
    return analyze_refusal_transcript(destination, policy)


def append_refusal_policy(
    source: Path,
    policy: RefusalRenderPolicy,
    *,
    expected_sha256: str,
    backup: Path,
) -> RefusalTranscriptAnalysis:
    """Append a policy in place after hash-locking and backing up the source."""
    source = source.expanduser().resolve()
    backup = backup.expanduser().resolve()
    payload = source.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Transcript hash changed: expected {expected_sha256}, got {actual_sha256}."
        )
    analysis = analyze_refusal_transcript(source, policy)
    _require_safe(analysis)
    if analysis.explicit_policy == policy:
        raise ValueError("Transcript already carries the requested refusal policy.")
    if backup.exists():
        raise ValueError(f"Refusing to overwrite backup {backup}.")
    backup.parent.mkdir(parents=True, exist_ok=True)
    with open(backup, "xb") as dest:
        dest.write(payload)
        dest.flush()
        os.fsync(dest.fileno())
    shutil.copystat(source, backup)

    record = _policy_record(source, policy)
    with open(source, "ab") as transcript:
        if payload and not payload.endswith(b"\n"):
            transcript.write(b"\n")
        transcript.write(record.model_dump_json().encode() + b"\n")
        transcript.flush()
        os.fsync(transcript.fileno())
    if source.read_bytes()[: len(payload)] != payload:
        raise RuntimeError("Append verification failed: inherited bytes changed.")
    result = analyze_refusal_transcript(source, policy)
    if result.explicit_policy != policy:
        raise RuntimeError("Append verification failed: policy did not rehydrate.")
    return result


def _parse_records(path: Path, payload: bytes) -> list[IRRecord]:
    records: list[IRRecord] = []
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            records.append(_adapter.validate_json(line))
        except Exception as exc:
            raise ValueError(
                f"Malformed record at {path}:{line_number}: {exc}"
            ) from exc
    if not records:
        raise ValueError(f"Transcript {path} is empty.")
    return records


def _policy_record(path: Path, policy: RefusalRenderPolicy) -> IRRuntimeConfigRecord:
    records = _parse_records(path, path.read_bytes())
    session = next(
        (record for record in records if isinstance(record, IRSessionRecord)), None
    )
    if session is None:
        raise ValueError(f"Transcript {path} has no session record.")
    last_completed, unfinished = _turn_state(records)
    if unfinished:
        raise ValueError(
            "Cannot append a rendering amendment during an unfinished turn."
        )
    values = refusal_policy_runtime_values(policy)
    return IRRuntimeConfigRecord(
        session_id=session.session_id,
        namespace=REFUSAL_RUNTIME_CONFIG_NAMESPACE,
        updates=values,
        effective=values,
        source="operator",
        turn=last_completed,
        turn_id="",
    )


def _turn_state(records: list[IRRecord]) -> tuple[int, bool]:
    last_completed = 0
    open_turn: int | None = None
    for record in records:
        if isinstance(record, IRTurnStartRecord):
            open_turn = record.turn
        elif isinstance(record, IRTurnEndRecord):
            last_completed = max(last_completed, record.turn)
            if open_turn == record.turn:
                open_turn = None
    return last_completed, open_turn is not None


def _semantic_state(
    records: list[IRRecord],
) -> tuple[
    dict[str, IRSemanticBlockRange],
    dict[str, IRSemanticBlockRecord],
    dict[str, str],
    dict[str, list[IRSemanticBlockArtifact]],
]:
    ranges: dict[str, IRSemanticBlockRange] = {}
    semantic_records: dict[str, IRSemanticBlockRecord] = {}
    modes: dict[str, str] = {}
    artifacts: dict[str, list[IRSemanticBlockArtifact]] = {}
    for record in records:
        if isinstance(record, IRBlockDetectionRecord):
            for semantic_range in record.completed:
                ranges[semantic_range.id] = semantic_range
        elif isinstance(record, IRSemanticBlockRecord):
            semantic_records[record.id] = record
            modes.setdefault(record.id, "full")
        elif isinstance(record, IRSemanticBlockApplyModeRecord):
            modes[record.block_id] = record.mode
        elif isinstance(record, IRSemanticBlockArtifactRecord):
            artifacts.setdefault(record.block_id, []).append(record.artifact)
    return ranges, semantic_records, modes, artifacts


def _semantic_owner(
    block_idx: int,
    *,
    ranges: dict[str, IRSemanticBlockRange],
    semantic_records: dict[str, IRSemanticBlockRecord],
) -> IRSemanticBlockRecord | None:
    for record in semantic_records.values():
        semantic_range = ranges.get(record.range_id)
        if (
            semantic_range is not None
            and semantic_range.start_block <= block_idx <= semantic_range.end_block
        ):
            return record
    return None


def _require_safe(analysis: RefusalTranscriptAnalysis) -> None:
    if not analysis.safe_to_amend:
        raise ValueError(
            "Refusal amendment preflight failed: "
            f"unfinished={analysis.unfinished_turn}, "
            f"ambiguous_turns={analysis.ambiguous_refusal_turns}, "
            f"projected_markers={analysis.projected_refusal_markers}."
        )


def _verify_appended_copy(source: Path, destination: Path) -> None:
    source_payload = source.read_bytes()
    destination_payload = destination.read_bytes()
    if destination_payload[: len(source_payload)] != source_payload:
        raise RuntimeError("Copy verification failed: inherited bytes changed.")
    _parse_records(destination, destination_payload)
