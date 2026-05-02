"""Tests for footer controller behavior and footer lifecycle injection."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.footer import (
    FooterController,
    FooterControllerRoundLifecycle,
)
from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import (
    IRBlockRecord,
    IRFooter,
    IRFooterDrainRecord,
    IRFooterQueueRecord,
    IRImageBlock,
    IRImageURLSource,
    IRInboundMessage,
    IRRecord,
    IRSkillCatalog,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder
from spellbook.round_lifecycle import RoundContext
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


def _make_recorder(tmp_path: Path, session_id: str = "s1") -> tuple[Recorder, Path]:
    transcript = tmp_path / "transcript.jsonl"
    config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
    recorder = Recorder(config, transcript, session_id, DEFAULT_TOOL_REGISTRY)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("t1", [])
    return recorder, transcript


def _read_records(path: Path) -> list[IRRecord]:
    adapter = TypeAdapter(IRRecord)
    records: list[IRRecord] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(adapter.validate_json(line))
    return records


def _footer_msg(
    text: str,
    *,
    footer_type: str = "notif",
    footer_source: str = "conduit",
    footer_key: str = "key1",
    footer_priority: int = 50,
) -> IRInboundMessage:
    return IRInboundMessage(
        blocks=[IRUserTextBlock(text=text, origin="human")],
        delivery="footer",
        source_metadata={
            "footer_type": footer_type,
            "footer_source": footer_source,
            "footer_key": footer_key,
            "footer_priority": footer_priority,
        },
    )


class TestFooterControllerQueueing:
    def test_queue_footer_stores_pending_by_key(self, tmp_path: Path) -> None:
        recorder, _ = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        controller.queue_footer(
            text="remember this",
            footer_type="reminder",
            source="planner",
            key="rem-1",
            priority=30,
        )

        pending = controller.peek_pending()
        assert len(pending) == 1
        assert pending[0].text == "remember this"
        assert pending[0].type == "reminder"
        assert pending[0].source == "planner"
        assert pending[0].key == "rem-1"
        assert pending[0].priority == 30

    def test_same_key_replaces_prior_footer(self, tmp_path: Path) -> None:
        recorder, _ = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        controller.queue_footer(
            text="old",
            footer_type="notif",
            source="conduit",
            key="shared",
            priority=90,
        )
        controller.queue_footer(
            text="new",
            footer_type="reminder",
            source="planner",
            key="shared",
            priority=10,
        )

        pending = controller.peek_pending()
        assert len(pending) == 1
        assert pending[0].text == "new"
        assert pending[0].type == "reminder"
        assert pending[0].source == "planner"
        assert pending[0].priority == 10

    def test_peek_pending_returns_priority_sorted(self, tmp_path: Path) -> None:
        recorder, _ = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        controller.queue_footer(
            text="lowest number first",
            footer_type="notif",
            source="conduit",
            key="high-prio",
            priority=5,
        )
        controller.queue_footer(
            text="later",
            footer_type="notif",
            source="conduit",
            key="low-prio",
            priority=50,
        )
        controller.queue_footer(
            text="middle",
            footer_type="notif",
            source="conduit",
            key="mid-prio",
            priority=20,
        )

        pending = controller.peek_pending()
        assert [f.key for f in pending] == ["high-prio", "mid-prio", "low-prio"]

    def test_collect_and_drain_returns_sorted_and_clears_pending(self, tmp_path: Path) -> None:
        recorder, _ = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        controller.queue_footer(
            text="b",
            footer_type="notif",
            source="conduit",
            key="b",
            priority=20,
        )
        controller.queue_footer(
            text="a",
            footer_type="notif",
            source="conduit",
            key="a",
            priority=10,
        )

        drained = controller.collect_and_drain()

        assert [f.key for f in drained] == ["a", "b"]
        assert controller.peek_pending() == []

    def test_render_footers_wraps_spellbook_tags_and_joins_with_separator(
        self, tmp_path: Path
    ) -> None:
        recorder, _ = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        footers = [
            IRFooter(
                text="first footer",
                type="notif",
                source="conduit",
                key="a",
                priority=10,
            ),
            IRFooter(
                text="second footer",
                type="reminder",
                source="planner",
                key="b",
                priority=20,
            ),
        ]

        rendered = controller.render_footers(footers)

        assert rendered.startswith("<spellbook>\n")
        assert rendered.endswith("\n</spellbook>")
        assert "first footer\n---\nsecond footer" in rendered


class TestFooterControllerInboundDrain:
    @pytest.mark.asyncio
    async def test_collect_and_drain_converts_footer_messages_from_inbound_queue(
        self, tmp_path: Path
    ) -> None:
        recorder, _ = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        await inbound.put(
            _footer_msg(
                "from inbound",
                footer_type="notif",
                footer_source="conduit",
                footer_key="notif-1",
                footer_priority=17,
            )
        )

        drained = controller.collect_and_drain()

        assert len(drained) == 1
        assert drained[0].text == "from inbound"
        assert drained[0].type == "notif"
        assert drained[0].source == "conduit"
        assert drained[0].key == "notif-1"
        assert drained[0].priority == 17

    @pytest.mark.asyncio
    async def test_collect_and_drain_preserves_non_footer_messages_in_queue(
        self, tmp_path: Path
    ) -> None:
        recorder, _ = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        turn = IRInboundMessage(
            blocks=[IRUserTextBlock(text="turn me", origin="human")],
            delivery="turn",
        )
        injection = IRInboundMessage(
            blocks=[IRUserTextBlock(text="inject me", origin="human")],
            delivery="inject",
        )
        footer = _footer_msg("footer only", footer_key="footer-1")

        await inbound.put(turn)
        await inbound.put(footer)
        await inbound.put(injection)

        drained = controller.collect_and_drain()

        assert [f.key for f in drained] == ["footer-1"]
        retained = list(inbound._messages)
        assert [msg.delivery for msg in retained] == ["turn", "inject"]
        first_block = retained[0].blocks[0]
        second_block = retained[1].blocks[0]
        assert isinstance(first_block, IRUserTextBlock)
        assert isinstance(second_block, IRUserTextBlock)
        assert first_block.text == "turn me"
        assert second_block.text == "inject me"

    @pytest.mark.asyncio
    async def test_collect_and_drain_rejects_footer_message_with_multiple_blocks(
        self, tmp_path: Path
    ) -> None:
        recorder, _ = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        bad = IRInboundMessage(
            blocks=[
                IRUserTextBlock(text="one", origin="human"),
                IRUserTextBlock(text="two", origin="human"),
            ],
            delivery="footer",
            source_metadata={
                "footer_type": "notif",
                "footer_source": "conduit",
                "footer_key": "bad",
            },
        )
        await inbound.put(bad)

        with pytest.raises(ValueError, match="malformed inbound footer message"):
            controller.collect_and_drain()

    @pytest.mark.asyncio
    async def test_collect_and_drain_rejects_footer_message_with_non_text_block(
        self, tmp_path: Path
    ) -> None:
        recorder, _ = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        bad = IRInboundMessage(
            blocks=[
                IRImageBlock(
                    origin="human",
                    source=IRImageURLSource(url="https://example.com/x.png"),
                )
            ],
            delivery="footer",
            source_metadata={
                "footer_type": "notif",
                "footer_source": "conduit",
                "footer_key": "bad",
            },
        )

        await inbound.put(bad)

        with pytest.raises(ValueError, match="malformed inbound footer message"):
            controller.collect_and_drain()


class TestFooterControllerTranscriptEffects:
    def test_queue_footer_writes_footer_queue_record(self, tmp_path: Path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        controller.queue_footer(
            text="queued footer",
            footer_type="notif",
            source="conduit",
            key="q1",
            priority=50,
        )

        records = _read_records(transcript)
        queue_records = [r for r in records if isinstance(r, IRFooterQueueRecord)]
        assert len(queue_records) == 1
        assert queue_records[0].footer.text == "queued footer"
        assert queue_records[0].footer.key == "q1"
        assert queue_records[0].turn == 1
        assert queue_records[0].turn_id == "t1"

    def test_collect_and_drain_writes_footer_drain_record(self, tmp_path: Path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)

        controller.queue_footer(
            text="first",
            footer_type="notif",
            source="conduit",
            key="a",
            priority=20,
        )
        controller.queue_footer(
            text="second",
            footer_type="reminder",
            source="planner",
            key="b",
            priority=10,
        )

        controller.collect_and_drain()

        records = _read_records(transcript)
        drain_records = [r for r in records if isinstance(r, IRFooterDrainRecord)]
        assert len(drain_records) == 1
        assert [f.key for f in drain_records[0].footers] == ["b", "a"]
        assert drain_records[0].turn == 1
        assert drain_records[0].turn_id == "t1"


class TestFooterControllerRoundLifecycle:
    @pytest.mark.asyncio
    async def test_before_round_is_noop_when_no_pending_footers(self, tmp_path: Path) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)
        lifecycle = FooterControllerRoundLifecycle(
            controller=controller, recorder=recorder
        )
        original = IRUserTextBlock(text="hello", origin="human")
        ctx = RoundContext(
            blocks=[original],
            round_number=1,
            cancel_token=CancelToken(),
            blocks_this_round=[],
        )

        await lifecycle.before_round(ctx)

        assert ctx.blocks == [original]
        records = _read_records(transcript)
        assert not any(isinstance(r, IRFooterDrainRecord) for r in records)
        block_records = [r for r in records if isinstance(r, IRBlockRecord)]
        assert len(block_records) == 0

    @pytest.mark.asyncio
    async def test_before_round_appends_rendered_footer_block_and_records_it(
        self, tmp_path: Path
    ) -> None:
        recorder, transcript = _make_recorder(tmp_path)
        inbound = InboundMessageQueue()
        controller = FooterController(inbound_queue=inbound, recorder=recorder)
        lifecycle = FooterControllerRoundLifecycle(
            controller=controller, recorder=recorder
        )

        controller.queue_footer(
            text="alpha",
            footer_type="notif",
            source="conduit",
            key="a",
            priority=20,
        )
        controller.queue_footer(
            text="beta",
            footer_type="reminder",
            source="planner",
            key="b",
            priority=10,
        )

        ctx = RoundContext(
            blocks=[IRUserTextBlock(text="hello", origin="human")],
            round_number=1,
            cancel_token=CancelToken(),
            blocks_this_round=[],
        )

        await lifecycle.before_round(ctx)

        assert len(ctx.blocks) == 2
        footer_block = ctx.blocks[-1]
        assert isinstance(footer_block, IRUserTextBlock)
        assert footer_block.origin == "system"
        assert footer_block.text == "<spellbook>\nbeta\n---\nalpha\n</spellbook>"

        records = _read_records(transcript)
        drain_records = [r for r in records if isinstance(r, IRFooterDrainRecord)]
        assert len(drain_records) == 1
        assert [f.key for f in drain_records[0].footers] == ["b", "a"]

        block_records = [r for r in records if isinstance(r, IRBlockRecord)]
        assert len(block_records) == 1
        written_block = block_records[0].event
        assert isinstance(written_block, IRUserTextBlock)
        assert written_block.origin == "system"
        assert written_block.turn_id == "t1"
        assert written_block.event_id is not None
        assert written_block.text == "<spellbook>\nbeta\n---\nalpha\n</spellbook>"
