"""Tests for the tool registry — lookup and schema generation."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from spellbook.backends.anthropic import AnthropicBackend
from spellbook.custom import CustomSurface
from spellbook.ir_types import IRToolTextBlock
from spellbook.tools.common import (
    Tool,
    ToolExecutionResult,
    ToolMetadata,
)
from spellbook.tools.registry import (
    DEFAULT_TOOL_REGISTRY,
    KNOWN_TOOL_REGISTRY,
    ToolRegistry,
)

# --- Fake tools for testing ---


class _FakeFSInput(BaseModel):
    path: str = Field(description="A path")


class _FakeMemInput(BaseModel):
    block_id: str


class _FakeWebInput(BaseModel):
    query: str


async def _fake_exec(meta: ToolMetadata, input: BaseModel) -> ToolExecutionResult:
    return ToolExecutionResult(content=[IRToolTextBlock(text="fake")])


FAKE_FS_TOOL: Tool[_FakeFSInput] = Tool(
    name="FakeRead",
    input_model=_FakeFSInput,
    exec=_fake_exec,
    category="filesystem",
)

FAKE_MEM_TOOL: Tool[_FakeMemInput] = Tool(
    name="FakeReflect",
    input_model=_FakeMemInput,
    exec=_fake_exec,
    category="memory",
)

FAKE_WEB_TOOL: Tool[_FakeWebInput] = Tool(
    name="FakeSearch",
    input_model=_FakeWebInput,
    exec=_fake_exec,
    category="web",
)


# --- Tests ---


class TestToolRegistryLookup:
    def test_empty_registry(self) -> None:
        registry = ToolRegistry(tools=[])
        assert registry.tool_names == set()
        assert registry.get("Bash") is None

    def test_single_tool_lookup(self) -> None:
        registry = ToolRegistry(tools=[FAKE_FS_TOOL])
        assert registry.tool_names == {"FakeRead"}
        tool = registry.get("FakeRead")
        assert tool is not None
        assert tool.name == "FakeRead"

    def test_unknown_tool_returns_none(self) -> None:
        registry = ToolRegistry(tools=[FAKE_FS_TOOL])
        assert registry.get("NonExistent") is None

    def test_multiple_tools(self) -> None:
        registry = ToolRegistry(tools=[FAKE_FS_TOOL, FAKE_MEM_TOOL, FAKE_WEB_TOOL])
        assert registry.tool_names == {"FakeRead", "FakeReflect", "FakeSearch"}
        assert registry.get("FakeMemTool") is None
        reflect = registry.get("FakeReflect")
        assert reflect is not None
        assert reflect.name == "FakeReflect"
        assert reflect.category == "memory"

    def test_registry_is_frozen(self) -> None:
        """Registry cannot be mutated after construction."""
        from pydantic import ValidationError

        registry = ToolRegistry(tools=[FAKE_FS_TOOL])
        with pytest.raises(ValidationError):
            registry.tools = [FAKE_MEM_TOOL]  # ty:ignore[invalid-assignment]


class TestSchemaGeneration:
    """Backend owns schema generation; registry provides the tools."""

    def test_anthropic_backend_generates_schema_for_all_tools(self) -> None:
        registry = ToolRegistry(tools=[FAKE_FS_TOOL, FAKE_MEM_TOOL, FAKE_WEB_TOOL])
        backend = AnthropicBackend()
        schemas = backend.build_tool_schemas(registry)
        names = {s["name"] for s in schemas}
        assert names == {"FakeRead", "FakeReflect", "FakeSearch"}

    def test_filtered_registry_narrows_schemas(self) -> None:
        registry = ToolRegistry(tools=[FAKE_FS_TOOL, FAKE_WEB_TOOL])
        backend = AnthropicBackend()
        schemas = backend.build_tool_schemas(registry)
        names = {s["name"] for s in schemas}
        assert names == {"FakeRead", "FakeSearch"}
        assert "FakeReflect" not in names

    def test_registry_build_empty_category_set_returns_nothing(self) -> None:
        registry = ToolRegistry.build(categories=set())
        assert registry.tools == []

        backend = AnthropicBackend()
        schemas = backend.build_tool_schemas(registry)
        assert schemas == []

    def test_registry_build_none_categories_means_all(self) -> None:
        registry = ToolRegistry.build(categories=None)
        assert registry.tool_names == DEFAULT_TOOL_REGISTRY.tool_names

        backend = AnthropicBackend()
        schemas = backend.build_tool_schemas(registry)
        names = {s["name"] for s in schemas}
        assert names == DEFAULT_TOOL_REGISTRY.tool_names

    def test_schema_has_name_description_and_input_schema(self) -> None:
        """Each generated schema has the three required Anthropic fields."""
        registry = ToolRegistry(tools=[FAKE_FS_TOOL])
        backend = AnthropicBackend()
        schemas = backend.build_tool_schemas(registry)
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["name"] == "FakeRead"
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"

    def test_schema_description_falls_back_to_docstring(self) -> None:
        """Without a markdown file, the input model's docstring provides description."""
        # _FakeFSInput doesn't have a markdown file in tools/descs/
        registry = ToolRegistry(tools=[FAKE_FS_TOOL])
        backend = AnthropicBackend()
        schemas = backend.build_tool_schemas(registry)
        # _FakeFSInput has no docstring, so description is an empty string
        # (the model docstring fallback yields "" if no docstring)
        assert schemas[0]["description"] == ""


class TestDefaultRegistry:
    """The DEFAULT_TOOL_REGISTRY contains the shipped tools."""

    def test_contains_bash(self) -> None:
        assert "Bash" in DEFAULT_TOOL_REGISTRY.tool_names

    def test_bash_is_filesystem_category(self) -> None:
        bash = DEFAULT_TOOL_REGISTRY.get("Bash")
        assert bash is not None
        assert bash.category == "filesystem"

    def test_default_registry_excludes_fork_tools(self) -> None:
        assert DEFAULT_TOOL_REGISTRY.tool_names == {
            "Read",
            "Write",
            "Edit",
            "Bash",
            "WebSearch",
            "WebRead",
            "WebAnswer",
            "Skill",
            "Reflect",
            "ReflectToolResults",
            "Forget",
            "ForgetToolResult",
            "Configure",
            "Pin",
            "Recall",
        }
        assert "ProposeBlock" not in DEFAULT_TOOL_REGISTRY.tool_names
        assert "AmendBlock" not in DEFAULT_TOOL_REGISTRY.tool_names
        assert "CompleteBlock" not in DEFAULT_TOOL_REGISTRY.tool_names


class TestToolSurfaces:
    """Tool surfaces separate normal entity tools from fork protocol tools."""

    def test_known_registry_contains_main_and_fork_tools(self) -> None:
        assert KNOWN_TOOL_REGISTRY.tool_names == {
            "Read",
            "Write",
            "Edit",
            "Bash",
            "WebSearch",
            "WebRead",
            "WebAnswer",
            "Skill",
            "Reflect",
            "ReflectToolResults",
            "Forget",
            "ForgetToolResult",
            "Configure",
            "Pin",
            "Recall",
            "ProposeBlock",
            "AmendBlock",
            "CompleteBlock",
            "Summarize",
        }

    def test_main_surface_default_is_normal_entity_registry(self) -> None:
        registry = ToolRegistry.build(categories=None, surface="main")

        assert registry.tool_names == DEFAULT_TOOL_REGISTRY.tool_names
        assert registry.tool_names == {
            "Read",
            "Write",
            "Edit",
            "Bash",
            "WebSearch",
            "WebRead",
            "WebAnswer",
            "Skill",
            "Reflect",
            "ReflectToolResults",
            "Forget",
            "ForgetToolResult",
            "Configure",
            "Pin",
            "Recall",
        }

    def test_main_surface_does_not_expose_block_detection_category(self) -> None:
        registry = ToolRegistry.build(categories={"block_detection"}, surface="main")

        assert registry.tool_names == set()

    def test_main_surface_exposes_memory_tools(self) -> None:
        registry = ToolRegistry.build(categories={"memory"}, surface="main")

        assert registry.tool_names == {
            "Reflect",
            "ReflectToolResults",
            "Forget",
            "ForgetToolResult",
            "Configure",
            "Pin",
            "Recall",
        }

    def test_block_detector_surface_default_exposes_detector_tools(self) -> None:
        registry = ToolRegistry.build(categories=None, surface="block_detector")

        assert registry.tool_names == {
            "ProposeBlock",
            "AmendBlock",
            "CompleteBlock",
        }

    def test_block_detector_surface_filters_within_surface(self) -> None:
        registry = ToolRegistry.build(
            categories={"block_detection"},
            surface="block_detector",
        )

        assert registry.tool_names == {
            "ProposeBlock",
            "AmendBlock",
            "CompleteBlock",
        }

    def test_block_detector_surface_does_not_expose_main_tools(self) -> None:
        registry = ToolRegistry.build(
            categories={"filesystem"},
            surface="block_detector",
        )

        assert registry.tool_names == set()

    def test_custom_surface_includes_selected_main_categories_and_custom_tools(
        self,
    ) -> None:
        registry = ToolRegistry.build(
            surface="custom",
            custom=CustomSurface(
                tools=[FAKE_FS_TOOL],
                include_tool_categories={"memory"},
            ),
        )

        assert registry.tool_names == {
            "FakeRead",
            "Reflect",
            "ReflectToolResults",
            "Forget",
            "ForgetToolResult",
            "Configure",
            "Pin",
            "Recall",
        }

    def test_custom_surface_requires_custom_surface_definition(self) -> None:
        with pytest.raises(ValueError, match="Custom tool surfaces require"):
            ToolRegistry.build(surface="custom")
