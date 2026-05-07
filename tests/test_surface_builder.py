"""Tests for the core RequestSurfaceBuilder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from spellbook.backends.model_backend import (
    GenerationStream,
    RequestSurface,
    TokenCounter,
)
from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.ir_types import IRBlock, IRUserTextBlock
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.common import Tool, ToolExecutionResult, ToolMetadata
from spellbook.tools.registry import ToolRegistry


@dataclass
class _FakeBackend:
    """Minimal fake backend that records request-surface inputs."""

    tool_schema_calls: list[ToolRegistry] = field(default_factory=list)
    request_surface_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def provider(self) -> str:
        return "fake"

    def build_tool_schemas(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        self.tool_schema_calls.append(registry)
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
        self.request_surface_calls.append(
            {
                "model": model,
                "system": system,
                "blocks": list(blocks),
                "tools": list(tools),
                "max_output_tokens": max_output_tokens,
                "effort": effort,
            }
        )
        return RequestSurface(
            model=model,
            system=system,
            tools=list(tools),
            messages=[{"role": "fake", "content": "stub"}],
            thinking={"budget": effort},
            output_config={"kind": "fake"},
            cache_control={"scope": "test"},
            max_output_tokens=max_output_tokens,
        )

    def stream(
        self,
        surface: RequestSurface,
        cancel_token: CancelToken,
    ) -> GenerationStream:
        raise NotImplementedError

    def build_token_counter(
        self, config: SpellbookConfig, surface_builder: RequestSurfaceBuilder
    ) -> TokenCounter:
        raise NotImplementedError


class _FakeToolInput(BaseModel):
    pass


async def _fake_exec(meta: ToolMetadata, input: _FakeToolInput) -> ToolExecutionResult:
    return ToolExecutionResult(content=[])


FAKE_TOOL: Tool[_FakeToolInput] = Tool(
    name="FakeRead",
    input_model=_FakeToolInput,
    exec=_fake_exec,
    category="filesystem",
)


class TestRequestSurfaceBuilderDirect:
    def test_request_surface_preserves_provider_payloads(self) -> None:
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

        surface = RequestSurface(model="test-model", messages=messages)
        other_surface = RequestSurface(model="test-model")

        assert surface.messages is messages
        assert isinstance(surface.messages[0]["content"], list)
        assert other_surface.messages == []
        assert other_surface.tools == []
        assert other_surface.messages is not RequestSurface(model="test-model").messages

    def test_build_delegates_to_backend_with_blocks_and_config(self) -> None:
        backend = _FakeBackend()
        builder = RequestSurfaceBuilder(
            model="test-model",
            system_provider=lambda: "system prompt",
            tool_schemas=[{"name": "FakeRead"}],
            backend=backend,
            max_output_tokens=64_000,
            effort="medium",
        )
        blocks = [IRUserTextBlock(text="hello", origin="human")]

        surface = builder.build(blocks)

        assert len(backend.request_surface_calls) == 1
        call = backend.request_surface_calls[0]
        assert call["model"] == "test-model"
        assert call["system"] == "system prompt"
        assert call["blocks"] == blocks
        assert call["tools"] == [{"name": "FakeRead"}]
        assert call["max_output_tokens"] == 64_000
        assert call["effort"] == "medium"

        assert surface.model == "test-model"
        assert surface.system == "system prompt"
        assert surface.tools == [{"name": "FakeRead"}]
        assert surface.thinking == {"budget": "medium"}
        assert surface.output_config == {"kind": "fake"}
        assert surface.cache_control == {"scope": "test"}
        assert surface.max_output_tokens == 64_000

    def test_system_provider_is_called_on_each_build(self) -> None:
        backend = _FakeBackend()
        counter = {"value": 0}

        def _system_provider() -> str:
            counter["value"] += 1
            return f"prompt v{counter['value']}"

        builder = RequestSurfaceBuilder(
            model="test-model",
            system_provider=_system_provider,
            tool_schemas=[],
            backend=backend,
        )

        builder.build([IRUserTextBlock(text="one", origin="human")])
        builder.build([IRUserTextBlock(text="two", origin="human")])
        builder.build([IRUserTextBlock(text="three", origin="human")])

        assert [call["system"] for call in backend.request_surface_calls] == [
            "prompt v1",
            "prompt v2",
            "prompt v3",
        ]

    def test_model_property_exposes_builder_model(self) -> None:
        backend = _FakeBackend()
        builder = RequestSurfaceBuilder(
            model="claude-opus-test",
            system_provider=lambda: "",
            tool_schemas=[],
            backend=backend,
        )

        assert builder.model == "claude-opus-test"

    def test_has_backend_is_true_for_live_builder(self) -> None:
        backend = _FakeBackend()
        builder = RequestSurfaceBuilder(
            model="test-model",
            system_provider=lambda: "",
            tool_schemas=[],
            backend=backend,
        )

        assert builder.has_backend is True


class TestRequestSurfaceBuilderFromConfig:
    def test_from_config_uses_backend_tool_schemas_and_spellbook_config(
        self, tmp_path
    ) -> None:
        backend = _FakeBackend()
        registry = ToolRegistry(tools=[FAKE_TOOL])
        config = SpellbookConfig(
            model="claude-sonnet-4-6",
            cwd=tmp_path,
            system_prompt="You are Spellbook.",
            max_output_tokens=12_345,
            effort="low",
        )

        builder = RequestSurfaceBuilder.from_config(
            backend=backend,
            config=config,
            tool_registry=registry,
        )

        assert builder.model == "claude-sonnet-4-6"
        assert builder.has_backend is True
        assert backend.tool_schema_calls == [registry]

        blocks = [IRUserTextBlock(text="hi", origin="human")]
        surface = builder.build(blocks)

        assert len(backend.request_surface_calls) == 1
        call = backend.request_surface_calls[0]
        assert call["model"] == "claude-sonnet-4-6"
        assert call["system"] == "You are Spellbook."
        assert call["blocks"] == blocks
        assert call["tools"] == [{"name": "FakeRead"}]
        assert call["max_output_tokens"] == 12_345
        assert call["effort"] == "low"

        assert surface.tools == [{"name": "FakeRead"}]

    def test_from_config_rebuilds_system_from_current_config_value(
        self, tmp_path
    ) -> None:
        backend = _FakeBackend()
        registry = ToolRegistry(tools=[])

        config = SpellbookConfig(
            model="claude-sonnet-4-6",
            cwd=tmp_path,
            system_prompt="Initial prompt",
        )

        builder = RequestSurfaceBuilder.from_config(
            backend=backend,
            config=config,
            tool_registry=registry,
        )

        builder.build([IRUserTextBlock(text="first", origin="human")])
        assert backend.request_surface_calls[0]["system"] == "Initial prompt"

        updated_config = config.model_copy(update={"system_prompt": "Updated prompt"})
        builder = RequestSurfaceBuilder.from_config(
            backend=backend,
            config=updated_config,
            tool_registry=registry,
        )

        builder.build([IRUserTextBlock(text="second", origin="human")])
        assert backend.request_surface_calls[1]["system"] == "Updated prompt"

    def test_from_config_passes_empty_tool_registry_through_cleanly(
        self, tmp_path
    ) -> None:
        backend = _FakeBackend()
        registry = ToolRegistry(tools=[])
        config = SpellbookConfig(model="claude-sonnet-4-6", cwd=tmp_path)

        builder = RequestSurfaceBuilder.from_config(
            backend=backend,
            config=config,
            tool_registry=registry,
        )
        builder.build([])

        assert backend.tool_schema_calls == [registry]
        assert backend.request_surface_calls[0]["tools"] == []
