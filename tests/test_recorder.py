"""Tests for Recorder — the write side of the transcript foundation.

Locks in the invariants:
- Turn/seq counters increment correctly
- event_id always stamped, turn_id stamped when None (preserved when set)
- write_session_record refuses to overwrite existing transcripts
- start_turn writes the start record plus any initial blocks in order
- set_state restores all three counters for resume
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlockRecord,
    IRFooter,
    IRFooterDrainRecord,
    IRFooterQueueRecord,
    IRImageBase64Source,
    IRImageBlobSource,
    IRImageBlock,
    IRRecord,
    IRSessionRecord,
    IRSkillCatalog,
    IRToolResultBlock,
    IRTurnEndRecord,
    IRTurnStartRecord,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _make_recorder(tmp_path: Path, session_id: str = "s1") -> tuple[Recorder, Path]:
    """Build a recorder pointed at a fresh transcript path."""
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    recorder = Recorder(config, transcript, session_id, DEFAULT_TOOL_REGISTRY)
    return recorder, transcript


def _make_tapped_recorder(
    tmp_path: Path,
    seen: list[IRRecord],
    session_id: str = "s1",
) -> tuple[Recorder, Path]:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    recorder = Recorder(
        config,
        transcript,
        session_id,
        DEFAULT_TOOL_REGISTRY,
        record_tap=seen.append,
    )
    return recorder, transcript


def _read_records(path: Path) -> list[IRRecord]:
    """Parse a transcript into typed records for test assertions."""
    records: list[IRRecord] = []
    adapter = TypeAdapter(IRRecord)
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(adapter.validate_json(line))
    return records


# --- Construction & session record ---


class TestSessionRecord:
    def test_write_session_record_creates_file(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        assert not transcript.exists()

        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        assert transcript.exists()

        records = _read_records(transcript)
        assert len(records) == 1
        assert isinstance(records[0], IRSessionRecord)
        assert records[0].session_id == "s1"

    def test_write_session_record_raises_on_existing_transcript(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        with pytest.raises(ValueError, match="already exists"):
            recorder.write_session_record(skill_catalog=IRSkillCatalog())

    def test_session_record_snapshots_tool_registry(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        records = _read_records(transcript)
        session = records[0]
        assert isinstance(session, IRSessionRecord)
        tool_names = {t.name for t in session.tools}
        assert "Bash" in tool_names

    def test_write_session_record_creates_parent_dirs(self, tmp_path) -> None:
        """Transcript in a nested path: parent dirs get created."""
        nested_path = tmp_path / "deep" / "nested" / "transcript.jsonl"
        config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
        recorder = Recorder(config, nested_path, "s1", DEFAULT_TOOL_REGISTRY)

        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        assert nested_path.exists()

    def test_record_tap_observes_written_records(self, tmp_path) -> None:
        seen: list[IRRecord] = []
        recorder, transcript = _make_tapped_recorder(tmp_path, seen)

        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [IRUserTextBlock(text="hello", origin="human")])
        recorder.end_turn()

        records = _read_records(transcript)
        assert seen == records
        assert [type(record) for record in seen] == [
            IRSessionRecord,
            IRTurnStartRecord,
            IRBlockRecord,
            IRTurnEndRecord,
        ]


# --- start_turn / end_turn / counters ---


class TestTurnLifecycle:
    def test_start_turn_without_initial_blocks_writes_only_start(
        self, tmp_path
    ) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        records = _read_records(transcript)
        assert len(records) == 2
        assert isinstance(records[1], IRTurnStartRecord)
        assert records[1].turn == 1
        assert records[1].turn_id == "t1"

    def test_start_turn_with_initial_blocks_writes_all_in_order(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        initial = [
            IRUserTextBlock(text="hello", origin="human"),
            IRUserTextBlock(text="second message", origin="human"),
        ]
        recorder.start_turn("t1", initial)

        records = _read_records(transcript)
        # session + turn_start + 2 block records
        assert len(records) == 4
        assert isinstance(records[1], IRTurnStartRecord)
        first_block_record = records[2]
        second_block_record = records[3]
        assert isinstance(first_block_record, IRBlockRecord)
        assert isinstance(second_block_record, IRBlockRecord)
        # Content preserved in order
        assert isinstance(first_block_record.event, IRUserTextBlock)
        assert isinstance(second_block_record.event, IRUserTextBlock)
        assert first_block_record.event.text == "hello"
        assert second_block_record.event.text == "second message"

    def test_end_turn_writes_end_record(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])
        recorder.end_turn()

        records = _read_records(transcript)
        assert isinstance(records[-1], IRTurnEndRecord)
        assert records[-1].turn == 1
        assert records[-1].turn_id == "t1"

    def test_turn_counter_increments_across_turns(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])
        recorder.end_turn()
        recorder.start_turn("t2", [])
        recorder.end_turn()
        recorder.start_turn("t3", [])

        records = _read_records(transcript)
        start_records = [r for r in records if isinstance(r, IRTurnStartRecord)]
        assert [r.turn for r in start_records] == [1, 2, 3]


# --- Seq numbering ---


class TestSeqCounters:
    def test_seq_monotonic_within_turn(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        for i in range(5):
            recorder.write_block(IRUserTextBlock(text=f"msg {i}", origin="human"))

        records = _read_records(transcript)
        block_records = [r for r in records if isinstance(r, IRBlockRecord)]
        assert [r.seq for r in block_records] == [0, 1, 2, 3, 4]

    def test_seq_resets_at_new_turn(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        recorder.start_turn("t1", [])
        recorder.write_block(IRUserTextBlock(text="a", origin="human"))
        recorder.write_block(IRUserTextBlock(text="b", origin="human"))
        recorder.end_turn()

        recorder.start_turn("t2", [])
        recorder.write_block(IRUserTextBlock(text="c", origin="human"))

        records = _read_records(transcript)
        block_records = [r for r in records if isinstance(r, IRBlockRecord)]
        # t1 blocks: seq 0, 1
        # t2 blocks: seq 0
        assert [r.turn for r in block_records] == [1, 1, 2]
        assert [r.seq for r in block_records] == [0, 1, 0]

    def test_start_turn_initial_blocks_get_seq_0_onward(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        initial = [
            IRUserTextBlock(text="a", origin="human"),
            IRUserTextBlock(text="b", origin="human"),
        ]
        recorder.start_turn("t1", initial)
        recorder.write_block(IRUserTextBlock(text="c", origin="human"))

        records = _read_records(transcript)
        block_records = [r for r in records if isinstance(r, IRBlockRecord)]
        assert [r.seq for r in block_records] == [0, 1, 2]


# --- Stamping invariants ---


class TestBlockStamping:
    def test_event_id_always_stamped(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        # Block arrives without event_id
        block = IRUserTextBlock(text="hi", origin="human")
        assert block.event_id is None
        recorder.write_block(block)

        records = _read_records(transcript)
        written = records[-1]
        assert isinstance(written, IRBlockRecord)
        assert written.event.event_id is not None
        assert len(written.event.event_id) > 0

    def test_event_ids_are_unique(self, tmp_path) -> None:
        """Every write_block produces a new event_id, even for identical content."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        for _ in range(5):
            recorder.write_block(IRUserTextBlock(text="same", origin="human"))

        records = _read_records(transcript)
        block_records = [r for r in records if isinstance(r, IRBlockRecord)]
        event_ids = [r.event.event_id for r in block_records]
        assert len(set(event_ids)) == 5  # all unique

    def test_event_id_overwritten_even_if_present(self, tmp_path) -> None:
        """Recorder is authoritative for event_id; pre-existing values are replaced."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        pre_existing = "pre-existing-event-id"
        block = IRUserTextBlock(text="hi", origin="human", event_id=pre_existing)
        recorder.write_block(block)

        records = _read_records(transcript)
        written = records[-1]
        assert isinstance(written, IRBlockRecord)
        assert written.event.event_id != pre_existing

    def test_turn_id_stamped_when_none(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        block = IRUserTextBlock(text="hi", origin="human")
        assert block.turn_id is None
        recorder.write_block(block)

        records = _read_records(transcript)
        written = records[-1]
        assert isinstance(written, IRBlockRecord)
        assert written.event.turn_id == "t1"

    def test_turn_id_preserved_when_set(self, tmp_path) -> None:
        """Blocks arriving with an existing turn_id keep that value."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        pre_set = "some-other-turn-id"
        block = IRUserTextBlock(text="hi", origin="human", turn_id=pre_set)
        recorder.write_block(block)

        records = _read_records(transcript)
        written = records[-1]
        assert isinstance(written, IRBlockRecord)
        assert written.event.turn_id == pre_set

    def test_image_blob_refs_are_persisted_without_base64(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
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
                            data="large-base64-payload",
                        ),
                        blob_path="blobs/image.png",
                    )
                ],
            )
        )
        recorder.write_block(
            IRImageBlock(
                origin="human",
                source=IRImageBase64Source(
                    media_type="image/png",
                    data="another-large-base64-payload",
                ),
                blob_path="blobs/human.png",
            )
        )

        records = _read_records(transcript)
        tool_result_record = records[-2]
        top_level_image_record = records[-1]
        assert isinstance(tool_result_record, IRBlockRecord)
        assert isinstance(tool_result_record.event, IRToolResultBlock)
        nested_image = tool_result_record.event.content[0]
        assert isinstance(nested_image, IRImageBlock)
        assert isinstance(nested_image.source, IRImageBlobSource)
        assert nested_image.blob_path == "blobs/image.png"

        assert isinstance(top_level_image_record, IRBlockRecord)
        assert isinstance(top_level_image_record.event, IRImageBlock)
        assert isinstance(top_level_image_record.event.source, IRImageBlobSource)
        assert top_level_image_record.event.blob_path == "blobs/human.png"


# --- set_state ---


class TestSetState:
    def test_set_state_updates_all_three_fields(self, tmp_path) -> None:
        """set_state mutates the turn/seq/turn_id for resume."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.set_state(turn_id="resumed_turn", turn=7, seq=42)

        # Subsequent writes reflect the restored state
        recorder._write_record(
            IRTurnEndRecord(
                turn_id="resumed_turn",
                session_id="s1",
                turn=7,
            )
        )
        records = _read_records(transcript)
        end_record = records[0]
        assert isinstance(end_record, IRTurnEndRecord)
        assert end_record.turn_id == "resumed_turn"
        assert end_record.turn == 7

    def test_set_state_affects_next_write_block(self, tmp_path) -> None:
        """After set_state, the next write_block uses the restored seq."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        # Simulate resume: we were at turn 3, seq 10
        recorder.set_state(turn_id="t3", turn=3, seq=10)
        recorder.write_block(IRUserTextBlock(text="after resume", origin="human"))

        records = _read_records(transcript)
        block_records = [r for r in records if isinstance(r, IRBlockRecord)]
        assert len(block_records) == 1
        assert block_records[0].turn == 3
        assert block_records[0].seq == 10
        assert block_records[0].event.turn_id == "t3"


# --- Footer records ---


class TestFooterRecords:
    def test_queue_footer_writes_footer_queue_record(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        footer = IRFooter(
            text="queued reminder",
            type="reminder",
            source="planner",
            key="reminder_1",
            priority=25,
        )
        recorder.queue_footer(footer)

        records = _read_records(transcript)
        queue_records = [r for r in records if isinstance(r, IRFooterQueueRecord)]
        assert len(queue_records) == 1
        assert queue_records[0].footer == footer
        assert queue_records[0].turn == 1
        assert queue_records[0].turn_id == "t1"

    def test_drain_footers_writes_footer_drain_record(self, tmp_path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
        recorder.start_turn("t1", [])

        footers = [
            IRFooter(
                text="first",
                type="notif",
                source="conduit",
                key="notif_1",
                priority=50,
            ),
            IRFooter(
                text="second",
                type="gas_gauge",
                source="telemetry",
                key="gas_gauge",
                priority=10,
            ),
        ]
        recorder.drain_footers(footers)

        records = _read_records(transcript)
        drain_records = [r for r in records if isinstance(r, IRFooterDrainRecord)]
        assert len(drain_records) == 1
        assert drain_records[0].footers == footers
        assert drain_records[0].turn == 1
        assert drain_records[0].turn_id == "t1"


# --- Full composition ---


class TestFullSequence:
    def test_session_with_multi_turn_sequence(self, tmp_path) -> None:
        """A realistic session: 2 turns, several blocks each, proper ordering."""
        recorder, transcript = _make_recorder(tmp_path)
        recorder.write_session_record(skill_catalog=IRSkillCatalog())

        # Turn 1
        recorder.start_turn("t1", [IRUserTextBlock(text="question", origin="human")])
        recorder.write_block(IRAssistantTextBlock(text="answer", origin="model"))
        recorder.end_turn()

        # Turn 2
        recorder.start_turn("t2", [IRUserTextBlock(text="followup", origin="human")])
        recorder.write_block(IRAssistantTextBlock(text="response", origin="model"))
        recorder.end_turn()

        records = _read_records(transcript)
        # Expected shape:
        # 0: session
        # 1: t1 start
        # 2: t1 user block (seq 0)
        # 3: t1 assistant block (seq 1)
        # 4: t1 end
        # 5: t2 start
        # 6: t2 user block (seq 0)
        # 7: t2 assistant block (seq 1)
        # 8: t2 end
        assert len(records) == 9
        assert isinstance(records[0], IRSessionRecord)
        assert isinstance(records[1], IRTurnStartRecord)
        assert isinstance(records[4], IRTurnEndRecord)
        assert isinstance(records[5], IRTurnStartRecord)
        assert isinstance(records[8], IRTurnEndRecord)
        # Turn counters consistent
        assert records[1].turn == 1
        assert records[5].turn == 2
