"""Tests for Rehydrator — the read side of the transcript foundation.

Locks in the invariants:
- Clean-ended transcripts produce None for in-progress fields
- Unfinished transcripts detect the dangling turn correctly
- Session record is required; missing config or session_id raises
- Malformed JSON fails loudly (not silently)
- Round-trip: write → read produces structurally equivalent records
- Tool validation: mismatch between recorded and live registry errors
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import pytest

from spellbook.config import SpellbookConfig
from spellbook.fork import BlockDetectorResult
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRBlockRecord,
    IRCompactBlockIntent,
    IRContextPlan,
    IRContextPlanProposalRecord,
    IRFooter,
    IRImageBase64Source,
    IRImageBlock,
    IRSemanticBlock,
    IRSemanticBlockApplyModeRecord,
    IRSemanticBlockMetricsRecord,
    IRSemanticBlockPin,
    IRSemanticBlockPinRecord,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSessionRecord,
    IRSkill,
    IRSkillCatalog,
    IRSkillCatalogDelta,
    IRSkillCatalogUpdateRecord,
    IRTokenRangeCount,
    IRToolCallBlock,
    IRToolResultBlock,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _make_recorder(tmp_path: Path, session_id: str = "s1") -> tuple[Recorder, Path]:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    recorder = Recorder(config, transcript, session_id, DEFAULT_TOOL_REGISTRY)
    return recorder, transcript


def _block_texts(blocks: list[IRBlock]) -> list[str]:
    texts: list[str] = []
    for block in blocks:
        assert isinstance(block, IRUserTextBlock | IRAssistantTextBlock)
        texts.append(block.text)
    return texts


def _skill(tmp_path: Path, name: str, description: str) -> IRSkill:
    directory = tmp_path / "skills" / name
    return IRSkill(
        name=name,
        description=description,
        location=directory / "SKILL.md",
        directory=directory,
        scope="project",
    )


# --- Clean-ended transcripts ---


class TestCleanEnded:
    def test_session_only(self, tmp_path) -> None:
        """Transcript with just a session record: no turns, no blocks."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        result = Rehydrator(transcript).run()
        assert result.session_id == "s1"
        assert result.blocks == []
        assert result.last_completed_turn == 0
        assert result.is_unfinished_turn is False
        assert result.current_turn_id is None
        assert result.in_progress_turn is None
        assert result.last_seq is None

    def test_single_completed_turn(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="hi", origin="human")])
        recorder.end_turn()

        result = Rehydrator(transcript).run()
        assert result.is_unfinished_turn is False
        assert result.current_turn_id is None
        assert result.in_progress_turn is None
        assert result.last_seq is None
        assert result.last_completed_turn == 1
        assert len(result.blocks) == 1
        assert _block_texts(result.blocks) == ["hi"]

    def test_multiple_completed_turns(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        recorder.start_turn("t1", [IRUserTextBlock(text="a", origin="human")])
        recorder.write_block(IRAssistantTextBlock(text="b", origin="model"))
        recorder.end_turn()

        recorder.start_turn("t2", [IRUserTextBlock(text="c", origin="human")])
        recorder.write_block(IRAssistantTextBlock(text="d", origin="model"))
        recorder.end_turn()

        result = Rehydrator(transcript).run()
        assert result.is_unfinished_turn is False
        assert result.last_completed_turn == 2
        assert len(result.blocks) == 4
        assert _block_texts(result.blocks) == ["a", "b", "c", "d"]


class TestImageBlobRehydration:
    def test_relative_blob_refs_rehydrate_after_session_directory_moves(
        self, tmp_path: Path
    ) -> None:
        session_dir = tmp_path / "session"
        transcript = session_dir / "transcript.jsonl"
        blob_path = session_dir / "blobs" / "tiny.png"
        image_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
            "/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
        )
        blob_path.parent.mkdir(parents=True)
        blob_path.write_bytes(image_bytes)

        config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
        recorder = Recorder(config, transcript, "s1", DEFAULT_TOOL_REGISTRY)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])
        recorder.write_block(
            IRToolResultBlock(
                call_id="toolu_image",
                tool="Read",
                content=[
                    IRImageBlock(
                        origin="tool",
                        source=IRImageBase64Source(
                            media_type="image/png",
                            data="stale-provider-payload",
                        ),
                        blob_path="blobs/tiny.png",
                    )
                ],
            )
        )
        recorder.end_turn()

        copied_dir = tmp_path / "copied-session"
        shutil.copytree(session_dir, copied_dir)
        result = Rehydrator(copied_dir / "transcript.jsonl").run()

        assert len(result.blocks) == 1
        block = result.blocks[0]
        assert isinstance(block, IRToolResultBlock)
        image = block.content[0]
        assert isinstance(image, IRImageBlock)
        assert image.blob_path == "blobs/tiny.png"
        assert isinstance(image.source, IRImageBase64Source)
        assert image.source.media_type == "image/png"
        assert image.source.data == base64.standard_b64encode(image_bytes).decode(
            "ascii"
        )


class TestBlockDetectionRecords:
    def test_block_detection_records_rehydrate_semantic_state(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])
        completed = [IRSemanticBlockRange(title="Done", start_block=0, end_block=1)]
        buffered = [IRSemanticBlockRange(title="Buffered", start_block=2, end_block=3)]
        recorder.detect_blocks(
            BlockDetectorResult(completed=completed, still_buffered=buffered)
        )
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert result.completed_semantic_block_ranges == completed
        assert result.buffered_semantic_block_ranges == buffered

    def test_later_block_detection_record_replaces_buffered_semantic_state(
        self, tmp_path
    ) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])
        old_buffer = [IRSemanticBlockRange(title="Old", start_block=0, end_block=0)]
        new_completion = [
            IRSemanticBlockRange(
                title="Old",
                start_block=0,
                end_block=0,
                completed=True,
            )
        ]
        new_buffer = [IRSemanticBlockRange(title="New", start_block=1, end_block=2)]
        recorder.detect_blocks(
            BlockDetectorResult(completed=[], still_buffered=old_buffer)
        )
        recorder.detect_blocks(
            BlockDetectorResult(completed=new_completion, still_buffered=new_buffer)
        )
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert result.completed_semantic_block_ranges == new_completion
        assert result.buffered_semantic_block_ranges == new_buffer


class TestSkillCatalogUpdateRecords:
    def test_skill_catalog_update_rehydrates_without_mutating_session_record(
        self, tmp_path
    ) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        compose_initial = _skill(tmp_path, "compose", "Draft careful prose.")
        old_skill = _skill(tmp_path, "old", "Old workflow.")
        initial_catalog = IRSkillCatalog(
            skills={"compose": compose_initial, "old": old_skill}
        )
        recorder.write_session_record(skill_catalog=initial_catalog)
        recorder.start_turn("t1", [])

        compose_updated = _skill(tmp_path, "compose", "Draft extra careful prose.")
        review = _skill(tmp_path, "review", "Review code changes.")
        delta = IRSkillCatalogDelta(
            added={"review": review},
            updated={"compose": compose_updated},
            removed=["old"],
        )
        recorder.update_skill_catalog(delta)
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert set(result.skill_catalog.skills) == {"compose", "review"}
        assert (
            result.skill_catalog.skills["compose"].description
            == "Draft extra careful prose."
        )
        assert result.skill_catalog.skills["review"] == review

        session_record = next(
            record for record in result.records if isinstance(record, IRSessionRecord)
        )
        assert session_record.skill_catalog == initial_catalog

        update_record = next(
            record
            for record in result.records
            if isinstance(record, IRSkillCatalogUpdateRecord)
        )
        assert update_record.delta == delta


class TestSemanticBlockModeRecords:
    def test_semantic_block_apply_mode_rehydrates_current_mode(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="full text", origin="human")])
        full_toks = IRTokenRangeCount(
            tokens=40,
            method="prefix_delta",
            exact=True,
        )
        summary_toks = IRTokenRangeCount(
            tokens=7,
            method="prefix_delta",
            exact=True,
        )
        completed_range = IRSemanticBlockRange(
            title="Completed block",
            start_block=0,
            end_block=0,
            completed=True,
        )
        semantic_block = IRSemanticBlock(
            idx=0,
            title="Completed block",
            range=completed_range,
            toks=full_toks,
            full_toks=full_toks,
        )
        summary = IRSemanticBlockSummary(
            headline="Summary headline",
            text="Summary text.",
            facets=[],
            open_thread=None,
            toks=summary_toks,
        )
        recorder.detect_blocks(
            BlockDetectorResult(completed=[completed_range], still_buffered=[])
        )
        recorder.write_semantic_block(semantic_block)
        recorder.write_block_artifact(summary, semantic_block.id)
        recorder.apply_semantic_block_mode("summary", semantic_block.id, "model")
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert len(result.semantic_blocks) == 1
        rehydrated_block = result.semantic_blocks[0]
        assert rehydrated_block.mode == "summary"
        assert rehydrated_block.toks == summary_toks
        assert rehydrated_block.full_toks == full_toks
        assert rehydrated_block.available_modes == ["full", "summary"]
        records = [
            record
            for record in result.records
            if isinstance(record, IRSemanticBlockApplyModeRecord)
        ]
        assert len(records) == 1
        assert records[0].mode == "summary"
        assert records[0].block_id == semantic_block.id
        assert records[0].source == "model"

    def test_old_semantic_block_apply_mode_record_defaults_source_to_model(
        self, tmp_path
    ) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="full text", origin="human")])
        full_toks = IRTokenRangeCount(
            tokens=40,
            method="prefix_delta",
            exact=True,
        )
        summary_toks = IRTokenRangeCount(
            tokens=7,
            method="prefix_delta",
            exact=True,
        )
        completed_range = IRSemanticBlockRange(
            title="Completed block",
            start_block=0,
            end_block=0,
            completed=True,
        )
        semantic_block = IRSemanticBlock(
            idx=0,
            title="Completed block",
            range=completed_range,
            toks=full_toks,
            full_toks=full_toks,
        )
        summary = IRSemanticBlockSummary(
            headline="Summary headline",
            text="Summary text.",
            facets=[],
            open_thread=None,
            toks=summary_toks,
        )
        recorder.detect_blocks(
            BlockDetectorResult(completed=[completed_range], still_buffered=[])
        )
        recorder.write_semantic_block(semantic_block)
        recorder.write_block_artifact(summary, semantic_block.id)
        with open(transcript, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "session_id": "s1",
                        "ir": "semantic_block_apply_mode",
                        "block_id": semantic_block.id,
                        "mode": "summary",
                        "turn": 1,
                        "turn_id": "t1",
                    }
                )
                + "\n"
            )
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert len(result.semantic_blocks) == 1
        assert result.semantic_blocks[0].mode == "summary"
        records = [
            record
            for record in result.records
            if isinstance(record, IRSemanticBlockApplyModeRecord)
        ]
        assert len(records) == 1
        assert records[0].source == "model"


class TestContextPlanProposalRecords:
    def test_context_plan_proposal_rehydrates_pending_plan(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])
        proposal = IRContextPlan(intents=[IRCompactBlockIntent(block_idx=0)])

        recorder.propose_plan(proposal)
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert result.plan_proposal == proposal
        records = [
            record
            for record in result.records
            if isinstance(record, IRContextPlanProposalRecord)
        ]
        assert len(records) == 1
        assert records[0].plan == proposal

    def test_later_mode_change_clears_pending_plan_proposal(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="full text", origin="human")])
        summary_toks = IRTokenRangeCount(
            tokens=7,
            method="prefix_delta",
            exact=True,
        )
        completed_range = IRSemanticBlockRange(
            title="Completed block",
            start_block=0,
            end_block=0,
            completed=True,
        )
        semantic_block = IRSemanticBlock(
            idx=0,
            title="Completed block",
            range=completed_range,
            toks=None,
            full_toks=None,
        )
        summary = IRSemanticBlockSummary(
            headline="Summary headline",
            text="Summary text.",
            facets=[],
            open_thread=None,
            toks=summary_toks,
        )
        proposal = IRContextPlan(intents=[IRCompactBlockIntent(block_idx=0)])
        recorder.detect_blocks(
            BlockDetectorResult(completed=[completed_range], still_buffered=[])
        )
        recorder.write_semantic_block(semantic_block)
        recorder.write_block_artifact(summary, semantic_block.id)
        recorder.propose_plan(proposal)
        recorder.apply_semantic_block_mode("summary", semantic_block.id, "planner")
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert result.plan_proposal is None


class TestSemanticBlockMetricsRecords:
    def test_metrics_rehydrate_full_count_without_overwriting_summary_mode_toks(
        self, tmp_path
    ) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="full text", origin="human")])
        full_toks = IRTokenRangeCount(
            tokens=40,
            method="prefix_delta",
            exact=True,
        )
        summary_toks = IRTokenRangeCount(
            tokens=7,
            method="prefix_delta",
            exact=True,
        )
        completed_range = IRSemanticBlockRange(
            title="Completed block",
            start_block=0,
            end_block=0,
            completed=True,
        )
        semantic_block = IRSemanticBlock(
            idx=0,
            title="Completed block",
            range=completed_range,
            toks=None,
            full_toks=None,
        )
        summary = IRSemanticBlockSummary(
            headline="Summary headline",
            text="Summary text.",
            facets=[],
            open_thread=None,
            toks=summary_toks,
        )
        recorder.detect_blocks(
            BlockDetectorResult(completed=[completed_range], still_buffered=[])
        )
        recorder.write_semantic_block(semantic_block)
        recorder.write_block_artifact(summary, semantic_block.id)
        recorder.apply_semantic_block_mode("summary", semantic_block.id, "model")
        recorder.write_block_metrics(full_toks, semantic_block.id)
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert len(result.semantic_blocks) == 1
        rehydrated_block = result.semantic_blocks[0]
        assert rehydrated_block.mode == "summary"
        assert rehydrated_block.toks == summary_toks
        assert rehydrated_block.full_toks == full_toks
        records = [
            record
            for record in result.records
            if isinstance(record, IRSemanticBlockMetricsRecord)
        ]
        assert len(records) == 1
        assert records[0].block_id == semantic_block.id
        assert records[0].toks == full_toks


class TestSemanticBlockPinRecords:
    def test_pin_record_rehydrates_block_pin(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="full text", origin="human")])
        completed_range = IRSemanticBlockRange(
            title="Completed block",
            start_block=0,
            end_block=0,
            completed=True,
        )
        semantic_block = IRSemanticBlock(
            idx=0,
            title="Completed block",
            range=completed_range,
            toks=None,
            full_toks=None,
        )
        pin = IRSemanticBlockPin(kind="block", reason="Keep the exact wording.")
        recorder.detect_blocks(
            BlockDetectorResult(completed=[completed_range], still_buffered=[])
        )
        recorder.write_semantic_block(semantic_block)
        recorder.apply_block_pin(pin, semantic_block.id)
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert len(result.semantic_blocks) == 1
        assert result.semantic_blocks[0].pin == pin
        records = [
            record
            for record in result.records
            if isinstance(record, IRSemanticBlockPinRecord)
        ]
        assert len(records) == 1
        assert records[0].block_id == semantic_block.id
        assert records[0].pin == pin

    def test_pin_record_rehydrates_facet_pin(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="full text", origin="human")])
        completed_range = IRSemanticBlockRange(
            title="Completed block",
            start_block=0,
            end_block=0,
            completed=True,
        )
        semantic_block = IRSemanticBlock(
            idx=0,
            title="Completed block",
            range=completed_range,
            toks=None,
            full_toks=None,
        )
        pin = IRSemanticBlockPin(
            kind="facet",
            reason="Keep this moment.",
            facet_id="facet_design",
        )
        recorder.detect_blocks(
            BlockDetectorResult(completed=[completed_range], still_buffered=[])
        )
        recorder.write_semantic_block(semantic_block)
        recorder.apply_block_pin(pin, semantic_block.id)
        recorder.end_turn()

        result = Rehydrator(transcript).run()

        assert len(result.semantic_blocks) == 1
        assert result.semantic_blocks[0].pin is None
        assert result.semantic_blocks[0].facet_pins == [pin]
        records = [
            record
            for record in result.records
            if isinstance(record, IRSemanticBlockPinRecord)
        ]
        assert len(records) == 1
        assert records[0].block_id == semantic_block.id
        assert records[0].pin == pin


# --- Unfinished turns ---


class TestUnfinishedTurn:
    def test_turn_started_without_end(self, tmp_path) -> None:
        """Turn started, blocks written, no end record: unfinished."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="hi", origin="human")])
        recorder.write_block(IRAssistantTextBlock(text="partial", origin="model"))
        # No end_turn — simulates crash mid-turn

        result = Rehydrator(transcript).run()
        assert result.is_unfinished_turn is True
        assert result.current_turn_id == "t1"
        assert result.in_progress_turn == 1
        assert result.last_seq == 1  # second block (0-indexed)
        assert result.last_completed_turn == 0  # no turn ever completed

    def test_last_completed_turn_tracked_correctly_when_unfinished(self, tmp_path) -> None:
        """When the unfinished turn is not the first, last_completed_turn is current_turn - 1."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        # Turn 1 — clean end
        recorder.start_turn("t1", [IRUserTextBlock(text="a", origin="human")])
        recorder.end_turn()
        # Turn 2 — clean end
        recorder.start_turn("t2", [IRUserTextBlock(text="b", origin="human")])
        recorder.end_turn()
        # Turn 3 — unfinished
        recorder.start_turn("t3", [IRUserTextBlock(text="c", origin="human")])

        result = Rehydrator(transcript).run()
        assert result.is_unfinished_turn is True
        assert result.in_progress_turn == 3
        assert result.last_completed_turn == 2

    def test_unfinished_first_turn(self, tmp_path) -> None:
        """First turn is unfinished: last_completed_turn stays 0."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="a", origin="human")])

        result = Rehydrator(transcript).run()
        assert result.is_unfinished_turn is True
        assert result.in_progress_turn == 1
        assert result.last_completed_turn == 0

    def test_unfinished_with_no_blocks(self, tmp_path) -> None:
        """Turn started but no blocks written before crash."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        result = Rehydrator(transcript).run()
        assert result.is_unfinished_turn is True
        assert result.current_turn_id == "t1"
        assert result.last_seq is None  # no block ever written


# --- Missing session record ---


class TestMissingSession:
    def test_empty_transcript_raises(self, tmp_path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")

        with pytest.raises(ValueError, match="Session record broken"):
            Rehydrator(empty).run()


# --- Malformed input ---


class TestMalformed:
    def test_malformed_json_raises(self, tmp_path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("this is not json\n")

        with pytest.raises(Exception):
            Rehydrator(bad).run()

    def test_valid_json_wrong_shape_raises(self, tmp_path) -> None:
        """A JSON line that parses but isn't a valid IRRecord."""
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"ir": "unknown_kind", "session_id": "x"}\n')

        with pytest.raises(Exception):
            Rehydrator(bad).run()

    def test_legacy_session_without_skill_catalog_raises_friendly_error(self, tmp_path) -> None:
        legacy = tmp_path / "legacy.jsonl"
        config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
        legacy.write_text(
            json.dumps(
                {
                    "session_id": "s1",
                    "ir": "session",
                    "time": "2026-04-29T00:00:00Z",
                    "config": config.model_dump(mode="json"),
                    "tools": [],
                }
            )
            + "\n"
        )

        with pytest.raises(ValueError, match="before Skill support"):
            Rehydrator(legacy).run()

    def test_blank_lines_ignored(self, tmp_path) -> None:
        """Blank lines in the transcript are skipped, not parsed."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="hi", origin="human")])
        recorder.end_turn()

        # Manually append blank lines
        with open(transcript, "a") as f:
            f.write("\n\n\n")

        # Rehydration succeeds despite blank trailing lines
        result = Rehydrator(transcript).run()
        assert result.is_unfinished_turn is False


# --- Pending footer replay ---


class TestPendingFooters:
    def test_queued_footer_without_drain_rehydrates_as_pending(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        footer = IRFooter(
            text="queued footer",
            type="notif",
            source="conduit",
            key="notif_1",
            priority=40,
        )
        recorder.queue_footer(footer)

        result = Rehydrator(transcript).run()

        assert result.pending_footers == {"notif_1": footer}

    def test_drained_footer_is_removed_from_pending(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        footer = IRFooter(
            text="queued footer",
            type="notif",
            source="conduit",
            key="notif_1",
            priority=40,
        )
        recorder.queue_footer(footer)
        recorder.drain_footers([footer])

        result = Rehydrator(transcript).run()

        assert result.pending_footers == {}

    def test_partial_drain_leaves_only_undrained_footers_pending(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        first = IRFooter(
            text="first",
            type="notif",
            source="conduit",
            key="first",
            priority=50,
        )
        second = IRFooter(
            text="second",
            type="reminder",
            source="planner",
            key="second",
            priority=20,
        )
        third = IRFooter(
            text="third",
            type="gas_gauge",
            source="telemetry",
            key="gas_gauge",
            priority=10,
        )

        recorder.queue_footer(first)
        recorder.queue_footer(second)
        recorder.queue_footer(third)
        recorder.drain_footers([second])

        result = Rehydrator(transcript).run()

        assert result.pending_footers == {
            "first": first,
            "gas_gauge": third,
        }

    def test_latest_queued_footer_for_same_key_wins_on_rehydrate(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        first = IRFooter(
            text="old footer",
            type="notif",
            source="conduit",
            key="shared",
            priority=50,
        )
        second = IRFooter(
            text="new footer",
            type="reminder",
            source="planner",
            key="shared",
            priority=10,
        )

        recorder.queue_footer(first)
        recorder.queue_footer(second)

        result = Rehydrator(transcript).run()

        assert result.pending_footers == {"shared": second}


# --- Tool validation ---


class TestToolValidation:
    def test_mismatched_tool_name_raises(self, tmp_path) -> None:
        """Session recorded a tool that's not in the live registry: error."""
        from spellbook.ir_types import IRToolRecord

        transcript = tmp_path / "transcript.jsonl"

        # Construct a synthetic session record with a bogus tool
        config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
        session = IRSessionRecord(
            config=config,
            session_id="s1",
            tools=[
                IRToolRecord(
                    name="NonExistentTool",
                    input_schema={"type": "object", "properties": {}},
                    category="filesystem",
                ),
            ],
            skill_catalog=IRSkillCatalog(),
        )
        transcript.write_text(session.model_dump_json() + "\n")

        with pytest.raises(ValueError, match="not registry"):
            Rehydrator(transcript).run()

    def test_category_mismatch_tolerated(self, tmp_path) -> None:
        """Categories are 'config concerns' — mismatches don't raise."""
        from spellbook.tools.common import tool_to_record

        transcript = tmp_path / "transcript.jsonl"

        bash_tool = DEFAULT_TOOL_REGISTRY.get("Bash")
        assert bash_tool is not None
        real_record = tool_to_record(bash_tool)

        # Rewrite category to something different
        altered = real_record.model_copy(update={"category": "web"})
        config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
        session = IRSessionRecord(
            config=config,
            session_id="s1",
            tools=[altered],
            skill_catalog=IRSkillCatalog(),
        )
        transcript.write_text(session.model_dump_json() + "\n")

        # Should not raise — category mismatches are tolerated
        result = Rehydrator(transcript).run()
        assert len(result.tools) == 1
        # The recorded category (not the live one) is preserved in the result
        assert result.tools[0].category == "web"

    def test_tools_surfaced_on_result(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        result = Rehydrator(transcript).run()

        tool_names = {t.name for t in result.tools}
        assert "Bash" in tool_names

    def test_fork_tool_record_validates_against_known_registry(self, tmp_path) -> None:
        """Fork transcripts record internal tools that are known but not default."""
        from spellbook.tools.common import tool_to_record
        from spellbook.tools.homunculus.block_detector import PROPOSE_BLOCK_TOOL

        transcript = tmp_path / "transcript.jsonl"
        config = SpellbookConfig(
            model="claude-sonnet-4-6",
            cwd=tmp_path,
            session_type="block_detector",
            tool_categories={"block_detection"},
        )
        session = IRSessionRecord(
            config=config,
            session_id="bd_session_1",
            tools=[tool_to_record(PROPOSE_BLOCK_TOOL)],
            skill_catalog=IRSkillCatalog(),
        )
        transcript.write_text(session.model_dump_json() + "\n")

        result = Rehydrator(transcript).run()

        assert result.session_id == "bd_session_1"
        assert {tool.name for tool in result.tools} == {"ProposeBlock"}


# --- Config surfaced ---


class TestConfigSurfaced:
    def test_config_on_result(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        result = Rehydrator(transcript).run()
        assert isinstance(result.config, SpellbookConfig)
        assert result.config.model == "claude-sonnet-4-6"


# --- Round-trip structural equality ---


class TestRoundTrip:
    def test_records_survive_write_read(self, tmp_path) -> None:
        """Writing a sequence of records and reading them back yields equal objects."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn(
            "t1",
            [
                IRUserTextBlock(text="first", origin="human"),
                IRUserTextBlock(text="second", origin="human"),
            ],
        )
        recorder.write_block(IRAssistantTextBlock(text="reply", origin="model"))
        recorder.write_block(
            IRToolCallBlock(
                origin="model",
                call_id="toolu_1",
                tool="Bash",
                input={"command": "ls"},
            )
        )
        recorder.end_turn()

        result = Rehydrator(transcript).run()
        assert isinstance(result, RehydrationResult)

        # Record-level structural equality via sequence check
        # (records include session, turn_start, 4 blocks, turn_end = 7 records)
        assert len(result.records) == 7

        # Block-level check
        assert len(result.blocks) == 4
        assert _block_texts(result.blocks[:3]) == ["first", "second", "reply"]
        assert isinstance(result.blocks[3], IRToolCallBlock)
        assert result.blocks[3].call_id == "toolu_1"

    def test_turn_seq_reconstruction(self, tmp_path) -> None:
        """Block records carry their turn+seq correctly through the round trip."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])
        recorder.write_block(IRUserTextBlock(text="a", origin="human"))
        recorder.write_block(IRUserTextBlock(text="b", origin="human"))
        recorder.end_turn()
        recorder.start_turn("t2", [])
        recorder.write_block(IRUserTextBlock(text="c", origin="human"))
        recorder.end_turn()

        result = Rehydrator(transcript).run()
        block_records = [r for r in result.records if isinstance(r, IRBlockRecord)]
        # (turn, seq) pairs
        pairs = [(r.turn, r.seq) for r in block_records]
        assert pairs == [(1, 0), (1, 1), (2, 0)]

    def test_event_ids_stamped_through_round_trip(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])
        recorder.write_block(IRUserTextBlock(text="hi", origin="human"))

        result = Rehydrator(transcript).run()
        assert result.blocks[0].event_id is not None
        assert len(result.blocks[0].event_id) > 0

    def test_turn_ids_stamped_through_round_trip(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])
        recorder.write_block(IRUserTextBlock(text="hi", origin="human"))

        result = Rehydrator(transcript).run()
        assert result.blocks[0].turn_id == "t1"
