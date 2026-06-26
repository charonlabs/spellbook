from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from spellbook.config import SpellbookConfig
from spellbook.homunculus.common import (
    AwarenessBudgetSnapshot,
    AwarenessHomunculusSnapshot,
    AwarenessTailSnapshot,
)
from spellbook.ir_types import (
    InboundDelivery,
    IRInboundMessage,
    IRSemanticBlock,
    IRSemanticBlockRange,
    IRTokenRangeCount,
    IRUserTextBlock,
)
from spellbook.slash_commands import (
    SlashCommandHandler,
    SlashCommandSession,
    parse_slash_command_message,
)


def _message(text: str, *, delivery: InboundDelivery = "turn") -> IRInboundMessage:
    return IRInboundMessage(
        blocks=[IRUserTextBlock(text=text, origin="human")],
        delivery=delivery,
    )


def _awareness() -> AwarenessHomunculusSnapshot:
    return AwarenessHomunculusSnapshot(
        budget=AwarenessBudgetSnapshot(
            max_tokens=1_000_000,
            reserve_output_tokens=50_000,
            current_input_tokens=12_345,
            current_slack_tokens=987_655,
            regime="calm",
            warning_threshold=700_000,
            forced_threshold=850_000,
            critical_threshold=950_000,
        ),
        semantic_blocks=[
            IRSemanticBlock(
                idx=0,
                title="Opening work",
                range=IRSemanticBlockRange(
                    title="Opening work",
                    start_block=0,
                    end_block=2,
                    completed=True,
                ),
                mode="summary",
                toks=IRTokenRangeCount(tokens=321, method="api", exact=True),
                full_toks=IRTokenRangeCount(tokens=900, method="api", exact=True),
            )
        ],
        proposed_blocks=[],
        tail=AwarenessTailSnapshot(tail_start=3, tail_end=4, toks=None),
        plan_proposal=None,
    )


def _handler(tmp_path: Path) -> SlashCommandHandler:
    session = SimpleNamespace(
        config=SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path),
        homunculus=SimpleNamespace(build_awareness=_awareness),
        recorder=SimpleNamespace(current_turn_idx=7),
        state="idle",
    )
    return SlashCommandHandler(cast(SlashCommandSession, session))


def test_parse_slash_command_only_accepts_human_turn_messages() -> None:
    assert parse_slash_command_message(_message("  /status")) == ("/status", "")
    assert parse_slash_command_message(_message("/reflect now")) == (
        "/reflect",
        "now",
    )
    assert parse_slash_command_message(_message("/status", delivery="inject")) is None


@pytest.mark.asyncio
async def test_status_command_outputs_plaintext_content_and_metadata(
    tmp_path: Path,
) -> None:
    response = await _handler(tmp_path).handle("/status", "")

    assert response.command == "/status"
    assert "claude-sonnet-4-6" in response.plaintext
    assert "| Context tokens | `12,345` |" in response.content
    assert response.metadata is not None
    assert response.metadata["semantic_blocks"] == 1


@pytest.mark.asyncio
async def test_reflect_command_outputs_block_table(tmp_path: Path) -> None:
    response = await _handler(tmp_path).handle("/reflect", "")

    assert response.command == "/reflect"
    assert "Reflect: 1 blocks" in response.plaintext
    assert "| Block | Title | Fidelity | Range | Tokens | Pins |" in response.content
    assert "| 0 | Opening work | summary | 0-2 | 321 | - |" in response.content


@pytest.mark.asyncio
async def test_help_command_lists_registered_commands(tmp_path: Path) -> None:
    response = await _handler(tmp_path).handle("/help", "")

    assert response.command == "/help"
    assert "/status" in response.plaintext
    assert "| `/reflect` |" in response.content


@pytest.mark.asyncio
async def test_unknown_command_returns_system_response(tmp_path: Path) -> None:
    response = await _handler(tmp_path).handle("/missing", "")

    assert response.command == "/missing"
    assert "Unknown command" in response.plaintext
    assert response.metadata is not None
    assert "/help" in response.metadata["known_commands"]
