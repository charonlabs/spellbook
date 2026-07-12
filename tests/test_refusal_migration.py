"""Append-only refusal policy migration and preflight."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRSkillCatalog,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.refusal import DEFAULT_REFUSAL_RENDER_POLICY
from spellbook.refusal_migration import (
    analyze_refusal_transcript,
    append_refusal_policy,
    write_amended_copy,
)
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _legacy_transcript(tmp_path: Path) -> Path:
    transcript = tmp_path / "legacy" / "transcript.jsonl"
    transcript.parent.mkdir(parents=True)
    recorder = Recorder(
        SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
        transcript,
        "legacy_session",
        DEFAULT_TOOL_REGISTRY,
    )
    recorder.write_session_record(IRSkillCatalog())
    recorder.start_turn("turn_1", [IRUserTextBlock(origin="human", text="hello")])
    recorder.write_block(
        IRAssistantTextBlock(
            text=(
                "<thinking_summary>\nsteady\n</thinking_summary>\n\n"
                "Partial answer\n\n<refusal>\nstop_reason: refusal\n"
                "details: unavailable\n</refusal>"
            )
        )
    )
    recorder.end_turn("refusal")
    recorder.start_turn("turn_2", [IRUserTextBlock(origin="human", text="again")])
    recorder.write_block(
        IRAssistantTextBlock(
            text=("<refusal>\nstop_reason: refusal\ndetails: unavailable\n</refusal>")
        )
    )
    recorder.end_turn("refusal")
    recorder.start_turn("turn_3", [IRUserTextBlock(origin="human", text="tool")])
    recorder.write_block(
        IRToolCallBlock(call_id="call_1", tool="Bash", input={"command": "true"})
    )
    recorder.write_block(
        IRToolResultBlock(
            call_id="call_1",
            tool="Bash",
            content=[IRToolTextBlock(text="ok")],
        )
    )
    recorder.end_turn("refusal")
    return transcript


def test_preflight_counts_represented_and_lifecycle_only_refusals(
    tmp_path: Path,
) -> None:
    transcript = _legacy_transcript(tmp_path)

    report = analyze_refusal_transcript(transcript, DEFAULT_REFUSAL_RENDER_POLICY)

    assert report.refusal_turns == [1, 2, 3]
    assert report.legacy_refusal_turns == [1, 2]
    assert report.canonical_refusal_turns == []
    assert report.lifecycle_only_refusal_turns == [3]
    assert report.partial_refusals == 1
    assert report.zero_partial_refusals == 1
    assert report.thinking_summary_refusals == 1
    assert report.projected_assistant_partials == 1
    assert report.projected_system_notes == 2
    assert report.projected_refusal_markers == 0
    assert report.safe_to_amend
    assert report.explicit_policy is None


def test_amended_copy_preserves_source_as_exact_prefix(tmp_path: Path) -> None:
    transcript = _legacy_transcript(tmp_path)
    original = transcript.read_bytes()
    destination = tmp_path / "copies" / "partial-note.jsonl"

    report = write_amended_copy(
        transcript,
        destination,
        DEFAULT_REFUSAL_RENDER_POLICY,
    )

    assert transcript.read_bytes() == original
    assert destination.read_bytes().startswith(original)
    assert report.explicit_policy == DEFAULT_REFUSAL_RENDER_POLICY
    assert report.record_count == 15


def test_in_place_append_requires_hash_and_creates_exact_backup(
    tmp_path: Path,
) -> None:
    transcript = _legacy_transcript(tmp_path)
    original = transcript.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    backup = tmp_path / "backups" / "before.jsonl"

    report = append_refusal_policy(
        transcript,
        DEFAULT_REFUSAL_RENDER_POLICY,
        expected_sha256=digest,
        backup=backup,
    )

    assert backup.read_bytes() == original
    assert transcript.read_bytes().startswith(original)
    assert report.explicit_policy == DEFAULT_REFUSAL_RENDER_POLICY


def test_hash_mismatch_refuses_without_writing(tmp_path: Path) -> None:
    transcript = _legacy_transcript(tmp_path)
    original = transcript.read_bytes()

    with pytest.raises(ValueError, match="hash changed"):
        append_refusal_policy(
            transcript,
            DEFAULT_REFUSAL_RENDER_POLICY,
            expected_sha256="0" * 64,
            backup=tmp_path / "backup.jsonl",
        )

    assert transcript.read_bytes() == original
    assert not (tmp_path / "backup.jsonl").exists()
