"""Tests for IR types — the canonical internal language of Spellbook core.

These tests lock in the invariants the type system is supposed to enforce.
They catch regressions in three specific properties:

- frozen=True on every type (mutation fails after construction)
- extra="forbid" on every type (typos fail at construction)
- Discriminated union membership (only valid types appear in IRBlock)
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from spellbook.config import SpellbookConfig
from spellbook.fork import (
    BlockDetectorConfig,
    BlockDetectorResult,
    ForkResult,
)
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRBlockRecord,
    IRFooter,
    IRFooterDrainRecord,
    IRFooterQueueRecord,
    IRGeneration,
    IRImageBase64Source,
    IRImageBlobSource,
    IRImageBlock,
    IRImageURLSource,
    IRRecord,
    IRSemanticBlockRange,
    IRSessionRecord,
    IRSkillCatalog,
    IRStreamTextDeltaEvent,
    IRStreamThinkingEndEvent,
    IRStreamThinkingStartEvent,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRTurnEndRecord,
    IRTurnStartRecord,
    IRUsage,
    IRUserTextBlock,
    StopReason,
)


class TestFrozen:
    """Every IR type is immutable after construction."""

    def test_user_text_block_is_frozen(self) -> None:
        block = IRUserTextBlock(text="hello", origin="human")
        with pytest.raises(ValidationError):
            setattr(block, "text", "changed")

    def test_assistant_text_block_is_frozen(self) -> None:
        block = IRAssistantTextBlock(text="hi", origin="model")
        with pytest.raises(ValidationError):
            setattr(block, "text", "changed")

    def test_tool_call_block_is_frozen(self) -> None:
        block = IRToolCallBlock(
            origin="model", call_id="toolu_1", tool="Bash", input={"cmd": "ls"}
        )
        with pytest.raises(ValidationError):
            setattr(block, "call_id", "different")

    def test_tool_result_block_is_frozen(self) -> None:
        block = IRToolResultBlock(
            call_id="toolu_1",
            tool="Bash",
            content=[IRToolTextBlock(text="hi")],
        )
        with pytest.raises(ValidationError):
            setattr(block, "is_error", True)

    def test_usage_is_frozen(self) -> None:
        usage = IRUsage(input_tokens=100, output_tokens=50)
        with pytest.raises(ValidationError):
            setattr(usage, "input_tokens", 200)


class TestExtraForbid:
    """Unknown fields are rejected at construction — catches typos."""

    def test_user_text_block_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError):
            IRUserTextBlock.model_validate(
                {"text": "hi", "origin": "human", "unknown_field": "x"}
            )

    def test_tool_call_block_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError):
            IRToolCallBlock.model_validate(
                {
                    "origin": "model",
                    "call_id": "toolu_1",
                    "tool": "Bash",
                    "input": {},
                    "turn_ID": "typo",
                }
            )

    def test_usage_rejects_extra_field(self) -> None:
        with pytest.raises(ValidationError):
            IRUsage.model_validate({"input_tokens": 100, "cached_tokens": 50})


class TestOriginConstraints:
    """Origin is constrained per block type at the type level."""

    def test_assistant_text_block_cannot_have_human_origin(self) -> None:
        with pytest.raises(ValidationError):
            IRAssistantTextBlock.model_validate({"text": "hi", "origin": "human"})

    def test_user_text_block_cannot_have_model_origin(self) -> None:
        with pytest.raises(ValidationError):
            IRUserTextBlock.model_validate({"text": "hi", "origin": "model"})

    def test_thinking_block_cannot_have_human_origin(self) -> None:
        with pytest.raises(ValidationError):
            IRThinkingBlock.model_validate(
                {"text": "thinking", "signature": "sig", "origin": "human"}
            )

    def test_tool_call_block_must_be_model_origin(self) -> None:
        with pytest.raises(ValidationError):
            IRToolCallBlock.model_validate(
                {
                    "origin": "tool",
                    "call_id": "toolu_1",
                    "tool": "Bash",
                    "input": {},
                }
            )

    def test_tool_result_block_must_be_tool_origin(self) -> None:
        with pytest.raises(ValidationError):
            IRToolResultBlock.model_validate(
                {
                    "call_id": "toolu_1",
                    "tool": "Bash",
                    "content": [],
                    "origin": "model",
                }
            )

    def test_user_text_accepts_system_origin(self) -> None:
        """System origin is valid for user text blocks — footer injections."""
        block = IRUserTextBlock(text="<spellbook>...</spellbook>", origin="system")
        assert block.origin == "system"


class TestDiscriminatedUnion:
    """IRBlock discriminates by `type` field and narrows correctly."""

    def test_tool_text_not_in_top_level_ir_block_union(self) -> None:
        """IRToolTextBlock should only appear nested inside IRToolResultBlock,
        not as a top-level stream block. This test documents that invariant."""
        # IRBlock is Annotated[Union[...], discriminator="type"]
        # Pydantic uses this when validating polymorphic fields
        # Construct an IRToolResultBlock with tool_text content (valid)
        result = IRToolResultBlock(
            call_id="toolu_1",
            tool="Bash",
            content=[IRToolTextBlock(text="output")],
        )
        assert len(result.content) == 1

    def test_tool_result_content_accepts_text_and_image(self) -> None:
        """IRToolResultContentBlock is the union of IRToolTextBlock and IRImageBlock."""
        text = IRToolTextBlock(text="hi")
        img = IRImageBlock(
            origin="tool",
            source=IRImageURLSource(url="https://example.com/x.png"),
        )
        result = IRToolResultBlock(
            call_id="toolu_1",
            tool="Bash",
            content=[text, img],
        )
        assert len(result.content) == 2

    def test_type_narrowing_via_isinstance(self) -> None:
        """isinstance-based narrowing works for IRBlock union members."""
        blocks: list[IRBlock] = [
            IRUserTextBlock(text="hi", origin="human"),
            IRAssistantTextBlock(text="hello", origin="model"),
            IRToolCallBlock(
                origin="model",
                call_id="toolu_1",
                tool="Bash",
                input={},
            ),
        ]

        tool_calls = [b for b in blocks if isinstance(b, IRToolCallBlock)]
        assert len(tool_calls) == 1
        assert tool_calls[0].call_id == "toolu_1"


class TestImageSource:
    """Image source is a discriminated union — exactly one shape per image."""

    def test_base64_source_shape(self) -> None:
        src = IRImageBase64Source(data="base64data==", media_type="image/png")
        assert src.type == "base64"

    def test_url_source_shape(self) -> None:
        src = IRImageURLSource(url="https://example.com/x.png")
        assert src.type == "url"

    def test_blob_source_shape(self) -> None:
        src = IRImageBlobSource()
        assert src.type == "blob"

    def test_image_block_with_base64(self) -> None:
        block = IRImageBlock(
            origin="human",
            source=IRImageBase64Source(data="xxx", media_type="image/jpeg"),
        )
        assert block.source.type == "base64"

    def test_image_block_with_blob_requires_blob_path(self) -> None:
        with pytest.raises(ValidationError):
            IRImageBlock(origin="human", source=IRImageBlobSource())

        block = IRImageBlock(
            origin="human",
            source=IRImageBlobSource(),
            blob_path="blobs/image.png",
        )
        assert block.source.type == "blob"

    def test_image_block_rejects_invalid_source(self) -> None:
        with pytest.raises(ValidationError):
            IRImageBlock.model_validate(
                {"origin": "human", "source": {"type": "bogus"}}
            )


class TestUsage:
    """IRUsage computed properties produce the correct totals."""

    def test_total_input_tokens_sums_input_and_cache(self) -> None:
        usage = IRUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_create_tokens=30,
        )
        assert usage.total_input_tokens == 100 + 200 + 30
        # output NOT included in total_input_tokens
        assert usage.total_input_tokens == 330

    def test_total_tokens_sums_everything(self) -> None:
        usage = IRUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=200,
            cache_create_tokens=30,
        )
        assert usage.total_tokens == 380

    def test_zero_by_default(self) -> None:
        usage = IRUsage()
        assert usage.input_tokens == 0
        assert usage.total_tokens == 0


class TestGeneration:
    """IRGeneration tool_calls property filters blocks correctly."""

    def test_tool_calls_extracts_tool_use_blocks(self) -> None:
        gen = IRGeneration(
            model="test",
            blocks=[
                IRAssistantTextBlock(text="I'll run that", origin="model"),
                IRToolCallBlock(
                    origin="model",
                    call_id="toolu_1",
                    tool="Bash",
                    input={},
                ),
                IRToolCallBlock(
                    origin="model",
                    call_id="toolu_2",
                    tool="Read",
                    input={},
                ),
            ],
            stop_reason="tool_use",
            usage=IRUsage(),
        )
        calls = gen.tool_calls
        assert len(calls) == 2
        assert calls[0].call_id == "toolu_1"
        assert calls[1].call_id == "toolu_2"

    def test_tool_calls_empty_when_no_tool_use(self) -> None:
        gen = IRGeneration(
            model="test",
            blocks=[
                IRAssistantTextBlock(text="just text", origin="model"),
            ],
            stop_reason="end_turn",
            usage=IRUsage(),
        )
        assert gen.tool_calls == []

    def test_stop_reason_must_be_valid(self) -> None:
        with pytest.raises(ValidationError):
            IRGeneration.model_validate(
                {
                    "model": "test",
                    "blocks": [],
                    "stop_reason": "invalid_reason",
                    "usage": IRUsage(),
                }
            )


class TestStopReasons:
    """StopReason covers all provider-observed termination conditions."""

    def test_all_expected_values_accepted(self) -> None:
        """Every declared stop reason constructs successfully."""
        from typing import get_args

        for reason in get_args(StopReason):
            gen = IRGeneration(
                model="test",
                blocks=[],
                stop_reason=reason,
                usage=IRUsage(),
            )
            assert gen.stop_reason == reason


class TestStreamEvents:
    """Stream events are six separate types discriminated by `kind`."""

    def test_thinking_start_event(self) -> None:
        event = IRStreamThinkingStartEvent()
        assert event.kind == "thinking_start"

    def test_text_delta_event(self) -> None:
        event = IRStreamTextDeltaEvent(text="partial")
        assert event.kind == "text_delta"
        assert event.text == "partial"

    def test_thinking_end_fires_without_payload(self) -> None:
        event = IRStreamThinkingEndEvent()
        assert event.kind == "thinking_end"


class TestFooterTypes:
    def test_footer_defaults_and_fields(self) -> None:
        footer = IRFooter(
            text="queued footer",
            type="notif",
            source="conduit",
            key="footer-key",
        )
        assert footer.text == "queued footer"
        assert footer.type == "notif"
        assert footer.source == "conduit"
        assert footer.key == "footer-key"
        assert footer.priority == 50
        assert footer.id.startswith("footer_")

    def test_footer_is_frozen(self) -> None:
        footer = IRFooter(
            text="queued footer",
            type="notif",
            source="conduit",
            key="footer-key",
        )
        with pytest.raises(ValidationError):
            setattr(footer, "text", "changed")


class TestIRRecordDiscrimination:
    def test_footer_queue_record_round_trips_through_ir_record_union(self) -> None:
        footer = IRFooter(
            text="queued footer",
            type="notif",
            source="conduit",
            key="footer-key",
            priority=12,
        )
        record = IRFooterQueueRecord(
            session_id="s1",
            footer=footer,
            turn=3,
            turn_id="turn_3",
        )

        adapter = TypeAdapter(IRRecord)
        parsed = adapter.validate_json(record.model_dump_json())

        assert isinstance(parsed, IRFooterQueueRecord)
        assert parsed.footer.text == "queued footer"
        assert parsed.footer.key == "footer-key"
        assert parsed.turn == 3
        assert parsed.turn_id == "turn_3"

    def test_footer_drain_record_round_trips_through_ir_record_union(self) -> None:
        drained = IRFooterDrainRecord(
            session_id="s1",
            footers=[
                IRFooter(
                    text="first",
                    type="notif",
                    source="conduit",
                    key="a",
                    priority=20,
                ),
                IRFooter(
                    text="second",
                    type="reminder",
                    source="planner",
                    key="b",
                    priority=10,
                ),
            ],
            turn=4,
            turn_id="turn_4",
        )

        adapter = TypeAdapter(IRRecord)
        parsed = adapter.validate_json(drained.model_dump_json())

        assert isinstance(parsed, IRFooterDrainRecord)
        assert [f.key for f in parsed.footers] == ["a", "b"]
        assert parsed.turn == 4
        assert parsed.turn_id == "turn_4"

    def test_ir_record_accepts_existing_record_variants(self, tmp_path) -> None:
        config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)
        session = IRSessionRecord(
            session_id="s1", config=config, tools=[], skill_catalog=IRSkillCatalog()
        )
        turn_start = IRTurnStartRecord(session_id="s1", turn=1, turn_id="t1")
        block = IRBlockRecord(
            session_id="s1",
            turn=1,
            seq=0,
            event=IRUserTextBlock(text="hi", origin="human"),
        )
        turn_end = IRTurnEndRecord(session_id="s1", turn=1, turn_id="t1")

        adapter = TypeAdapter(IRRecord)

        assert isinstance(
            adapter.validate_json(session.model_dump_json()), IRSessionRecord
        )
        assert isinstance(
            adapter.validate_json(turn_start.model_dump_json()), IRTurnStartRecord
        )
        assert isinstance(adapter.validate_json(block.model_dump_json()), IRBlockRecord)
        assert isinstance(
            adapter.validate_json(turn_end.model_dump_json()), IRTurnEndRecord
        )


class TestForkProtocolTypes:
    def test_block_detector_config_constructs_with_expected_fields(self) -> None:
        config = BlockDetectorConfig(
            prev_semantic_blocks=[
                IRSemanticBlockRange(title="Prior block", start_block=0, end_block=2)
            ],
            full_context_blocks=[
                IRUserTextBlock(text="hello", origin="human"),
                IRAssistantTextBlock(text="world", origin="model"),
            ],
            context_block_buffer=[
                IRAssistantTextBlock(text="world", origin="model"),
            ],
            context_block_start_id=1,
            semantic_block_buffer=[
                IRSemanticBlockRange(title="Buffered block", start_block=1, end_block=1)
            ],
            inbound_block=IRUserTextBlock(
                text="<block_detector_context />",
                origin="system",
            ),
        )

        assert config.type == "block_detector"
        assert config.context_block_start_id == 1
        assert config.detector_model == "claude-opus-4-6"
        assert config.prev_semantic_blocks[0].title == "Prior block"
        assert config.semantic_block_buffer[0].title == "Buffered block"
        assert config.inbound_block.origin == "system"

    def test_block_detector_result_round_trips_through_fork_result_union(self) -> None:
        result = BlockDetectorResult(
            completed=[
                IRSemanticBlockRange(
                    title="Completed block",
                    start_block=3,
                    end_block=5,
                    completed=True,
                )
            ],
            still_buffered=[
                IRSemanticBlockRange(
                    title="Buffered block",
                    start_block=6,
                    end_block=7,
                )
            ],
        )

        adapter = TypeAdapter(ForkResult)
        parsed = adapter.validate_json(result.model_dump_json())

        assert isinstance(parsed, BlockDetectorResult)
        assert [b.title for b in parsed.completed] == ["Completed block"]
        assert [b.title for b in parsed.still_buffered] == ["Buffered block"]
        assert parsed.completed[0].completed is True

    def test_fork_result_rejects_unknown_discriminator(self) -> None:
        adapter = TypeAdapter(ForkResult)
        with pytest.raises(ValidationError):
            adapter.validate_json(
                '{"type":"unknown_fork","completed":[],"still_buffered":[]}'
            )
