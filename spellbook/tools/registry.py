"""The tool registry.

``ToolRegistry`` is a frozen list of ``Tool`` instances. It supports
lookup by name and category-based filtering. The registry itself is
provider-agnostic — the backend consumes it and generates
provider-specific schemas via ``backend.build_tool_schemas(registry)``.

``DEFAULT_TOOL_REGISTRY`` is the normal Spellbook entity registry.
Internal fork tools live on their own tool surfaces: they are known to
the binary for execution and transcript validation, but they are not
available to the main entity unless its surface explicitly exposes them.
The registry is immutable — there's no ``registry.add``; to change the
tool surface, construct a new registry.
"""

from typing import Any, Literal

from pydantic import BaseModel

from spellbook.tools.skills import SKILL_TOOL
from spellbook.tools.web import WEB_ANSWER_TOOL, WEB_READ_TOOL, WEB_SEARCH_TOOL

from ..ir_types import IRToolRecord
from .common import Tool, tool_to_record
from .filesystem import BASH_TOOL, EDIT_TOOL, READ_TOOL, WRITE_TOOL
from .homunculus.block_detector import (
    AMEND_BLOCK_TOOL,
    COMPLETE_BLOCK_TOOL,
    PROPOSE_BLOCK_TOOL,
)
from .homunculus.block_summarizer import SUMMARIZE_TOOL
from .self_work import FORGET_TOOL, PIN_TOOL, RECALL_TOOL, REFLECT_TOOL

ToolSurface = Literal["main", "block_detector", "block_summarizer"]


class ToolRegistry(BaseModel, frozen=True):
    """Global, immutable tool registry."""

    tools: list[Tool[Any]]

    @property
    def tool_names(self) -> set[str]:
        return set([tool.name for tool in self.tools])

    @property
    def records(self) -> list[IRToolRecord]:
        return [tool_to_record(tool) for tool in self.tools]

    def get(self, name: str) -> Tool[Any] | None:
        """Get a `Tool` object given the tool's name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    @classmethod
    def build(
        cls,
        categories: set[str] | None = None,
        *,
        surface: ToolSurface = "main",
    ) -> "ToolRegistry":
        surface_tools = TOOLS_BY_SURFACE[surface]
        if categories is None:
            return cls(tools=surface_tools)
        filtered_tools = [tool for tool in surface_tools if tool.category in categories]
        return cls(tools=filtered_tools)


# Main entity tools. This is the surface a normal Spellbook session sees.
MAIN_TOOLS: list[Tool[Any]] = [
    READ_TOOL,
    WRITE_TOOL,
    EDIT_TOOL,
    BASH_TOOL,
    WEB_SEARCH_TOOL,
    WEB_READ_TOOL,
    WEB_ANSWER_TOOL,
    SKILL_TOOL,
    REFLECT_TOOL,
    FORGET_TOOL,
    PIN_TOOL,
    RECALL_TOOL,
]

# Fork-scoped tools. These are protocol tools for child sessions, not part of
# the normal model-facing Spellbook surface.
BLOCK_DETECTOR_TOOLS: list[Tool[Any]] = [
    PROPOSE_BLOCK_TOOL,
    AMEND_BLOCK_TOOL,
    COMPLETE_BLOCK_TOOL,
]

BLOCK_SUMMARIZER_TOOLS: list[Tool[Any]] = [SUMMARIZE_TOOL]

TOOLS_BY_SURFACE: dict[ToolSurface, list[Tool[Any]]] = {
    "main": MAIN_TOOLS,
    "block_detector": BLOCK_DETECTOR_TOOLS,
    "block_summarizer": BLOCK_SUMMARIZER_TOOLS,
}

# Every tool this binary knows how to validate and execute.
ALL_TOOLS: list[Tool[Any]] = MAIN_TOOLS + BLOCK_DETECTOR_TOOLS + BLOCK_SUMMARIZER_TOOLS

DEFAULT_TOOL_REGISTRY = ToolRegistry(tools=MAIN_TOOLS)
KNOWN_TOOL_REGISTRY = ToolRegistry(tools=ALL_TOOLS)
