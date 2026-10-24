"""Conversation history must survive between turns in every session profile.

Regression tests for two bugs:

- Profiles with ``homunculus_lifecycle=False`` never handed generations or
  executions to the Homunculus, so the next turn was rendered from inbound
  messages alone. The entity lost all of its own earlier output: thinking,
  text, tool calls and tool results.
- A turn interrupted mid-thinking stores a thinking block with an empty
  signature. Anthropic rejects unsigned thinking, so every later request
  failed, including after rehydration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest
from pydantic import BaseModel

from spellbook.backends.anthropic import _ir_blocks_to_message_params
from spellbook.backends.model_backend import RequestSurface, TokenCounter
from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.custom import CustomSurface
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRGeneration,
    IRInboundMessage,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUsage,
    IRUserTextBlock,
    StopReason,
)
from spellbook.profiles import CUSTOM
from spellbook.session_manager import SessionManager
from spellbook.tools.common import Tool, ToolExecutionResult, ToolMetadata


class _NullTokenCounter(TokenCounter):
    async def count_block_content(self, block: IRBlock) -> int | None:
        return None

    async def count_blocks(self, blocks: list[IRBlock]) -> int | None:
        return None

    async def count_frame(self) -> int | None:
        return None

    async def count_surface(self, surface: RequestSurface) -> int | None:
        return None


class _ScriptedStream:
    def __init__(self, response: IRGeneration):
        self._response = response

    async def __aenter__(self) -> "_ScriptedStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> "_ScriptedStream":
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def get_final_response(self) -> IRGeneration:
        return self._response

    def get_current_response(self, *, stop_reason: StopReason) -> IRGeneration:
        return self._response.model_copy(update={"stop_reason": stop_reason})


class _ScriptedBackend:
    """Returns canned generations and records the blocks of every request."""

    def __init__(self, responses: list[IRGeneration]):
        self._responses = list(responses)
        self.blocks_seen: list[list[IRBlock]] = []

    def build_tool_schemas(self, registry: Any) -> list[dict[str, Any]]:
        return [{"name": tool.name} for tool in registry.tools]

    def build_request_surface(
        self,
        *,
        model: str,
        system: str | list[dict[str, Any]],
        blocks: Sequence[IRBlock],
        tools: list[dict[str, Any]],
        max_output_tokens: int,
        effort: str,
    ) -> RequestSurface:
        self.blocks_seen.append(list(blocks))
        return RequestSurface(
            model=model,
            system=system,
            tools=tools,
            messages=[],
            max_output_tokens=max_output_tokens,
        )

    def stream(
        self, surface: RequestSurface, cancel_token: CancelToken
    ) -> _ScriptedStream:
        if not self._responses:
            raise RuntimeError("scripted backend ran out of responses")
        return _ScriptedStream(self._responses.pop(0))

    def build_token_counter(
        self, config: SpellbookConfig, surface_builder: Any
    ) -> TokenCounter:
        return _NullTokenCounter()


class _EchoInput(BaseModel):
    """Echo a word back."""

    word: str


async def _echo(meta: ToolMetadata, inp: _EchoInput) -> ToolExecutionResult:
    return ToolExecutionResult(content=[IRToolTextBlock(text=f"echo: {inp.word}")])


ECHO_TOOL = Tool(name="Echo", input_model=_EchoInput, exec=_echo, category="thinking")


def _gen(blocks: list[IRBlock], stop_reason: StopReason = "end_turn") -> IRGeneration:
    return IRGeneration(
        model="test-model", blocks=blocks, stop_reason=stop_reason, usage=IRUsage()
    )


def _user(text: str) -> IRInboundMessage:
    return IRInboundMessage(
        blocks=[IRUserTextBlock(text=text, origin="human")], delivery="turn"
    )


async def _history_only_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: _ScriptedBackend
) -> SessionManager:
    """A custom session with every awareness feature off, like an eval harness."""
    profile = CUSTOM.model_copy(
        update={
            "homunculus_lifecycle": False,
            "block_detection": False,
            "skill_discovery": False,
        }
    )
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        cwd=tmp_path,
        session_type="custom",
        profile=profile,
        system_prompt="test",
    )
    monkeypatch.setattr(
        "spellbook.session_manager.build_backend", lambda config: backend
    )
    return await SessionManager.build(
        transcript_path=tmp_path / "history.jsonl",
        config=config,
        custom_surface=CustomSurface(tools=[ECHO_TOOL]),
    )


async def _turn(manager: SessionManager, text: str) -> None:
    await manager.submit_message(_user(text))
    await manager._running_phase()


def _round_lifecycle_names(manager: SessionManager) -> list[str]:
    return [type(lc).__name__ for lc in manager.round_lifecycle._lifecycles]


class TestHistoryWithoutHomunculusLifecycle:
    @pytest.mark.asyncio
    async def test_history_only_profile_wires_history_lifecycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = await _history_only_session(
            tmp_path, monkeypatch, _ScriptedBackend([])
        )

        assert _round_lifecycle_names(manager) == [
            "RecordingRoundLifecycle",
            "ContextHistoryRoundLifecycle",
        ]

    @pytest.mark.asyncio
    async def test_thinking_only_max_tokens_turn_reaches_next_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thinking = IRThinkingBlock(text="long deliberation", signature="sig-123")
        backend = _ScriptedBackend(
            [
                _gen([thinking], stop_reason="max_tokens"),
                _gen([IRAssistantTextBlock(text="resuming")]),
            ]
        )
        manager = await _history_only_session(tmp_path, monkeypatch, backend)

        await _turn(manager, "solve the puzzle")
        await _turn(manager, "keep going")

        second = backend.blocks_seen[1]
        assert [type(b).__name__ for b in second] == [
            "IRUserTextBlock",
            "IRThinkingBlock",
            "IRUserTextBlock",
        ]
        carried = second[1]
        assert isinstance(carried, IRThinkingBlock)
        assert carried.text == "long deliberation"
        assert carried.signature == "sig-123"

        messages = _ir_blocks_to_message_params(second)
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert list(messages[1]["content"]) == [
            {
                "type": "thinking",
                "thinking": "long deliberation",
                "signature": "sig-123",
            }
        ]

    @pytest.mark.asyncio
    async def test_tool_calls_results_and_text_reach_next_turn(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _ScriptedBackend(
            [
                _gen(
                    [
                        IRToolCallBlock(
                            call_id="call_1", tool="Echo", input={"word": "hi"}
                        )
                    ],
                    stop_reason="tool_use",
                ),
                _gen([IRAssistantTextBlock(text="done with turn one")]),
                _gen([IRAssistantTextBlock(text="turn two")]),
            ]
        )
        manager = await _history_only_session(tmp_path, monkeypatch, backend)

        await _turn(manager, "first")
        await _turn(manager, "second")

        third_request = backend.blocks_seen[2]
        kinds = [type(b).__name__ for b in third_request]
        assert kinds == [
            "IRUserTextBlock",
            "IRToolCallBlock",
            "IRToolResultBlock",
            "IRAssistantTextBlock",
            "IRUserTextBlock",
        ]
        result = third_request[2]
        assert isinstance(result, IRToolResultBlock)
        assert result.content == [IRToolTextBlock(text="echo: hi")]

    @pytest.mark.asyncio
    async def test_history_only_profile_injects_no_footers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _ScriptedBackend(
            [
                _gen([IRAssistantTextBlock(text="one")]),
                _gen([IRAssistantTextBlock(text="two")]),
            ]
        )
        manager = await _history_only_session(tmp_path, monkeypatch, backend)

        await _turn(manager, "first")
        await _turn(manager, "second")

        texts = [
            b.text for b in backend.blocks_seen[1] if isinstance(b, IRUserTextBlock)
        ]
        assert texts == ["first", "second"]


class TestInterruptedThinking:
    @pytest.mark.asyncio
    async def test_interrupted_unsigned_thinking_is_kept_but_never_sent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = _ScriptedBackend(
            [
                _gen(
                    [IRThinkingBlock(text="half a thought", signature="")],
                    stop_reason="cancelled",
                ),
                _gen([IRAssistantTextBlock(text="back again")]),
            ]
        )
        manager = await _history_only_session(tmp_path, monkeypatch, backend)

        await _turn(manager, "think hard")
        await _turn(manager, "what were you doing?")

        second = backend.blocks_seen[1]
        # Transcript truth is carried forward untouched...
        assert any(isinstance(b, IRThinkingBlock) and b.signature == "" for b in second)
        # ...but the provider request never contains the unsigned block.
        messages = _ir_blocks_to_message_params(second)
        assert [m["role"] for m in messages] == ["user"]
        assert [part["text"] for part in messages[0]["content"]] == [
            "think hard",
            "what were you doing?",
        ]
