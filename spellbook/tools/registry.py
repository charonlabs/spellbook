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

from collections.abc import Set as AbstractSet
from typing import Any

from pydantic import BaseModel

from spellbook.custom import CustomSurface
from spellbook.profiles import ToolSurface
from spellbook.tools.body import BODY_TOOL
from spellbook.tools.chorus import REACH_TOOL
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
from .quantum import SUBMIT_RESULT_TOOL
from .self_work import (
    CONFIGURE_TOOL,
    FORGET_TOOL,
    FORGET_TOOL_RESULT_TOOL,
    PIN_TOOL,
    RECALL_TOOL,
    REFLECT_TOOL,
    REFLECT_TOOL_RESULTS_TOOL,
)
from .sleep import SLEEP_TOOL

CATEGORY_HIERARCHY: dict[str, frozenset[str]] = {
    "coding": frozenset({"filesystem", "thinking"}),
    "main": frozenset({"filesystem", "thinking", "memory", "web", "skills", "body"}),
    "chorus": frozenset(
        {
            "filesystem",
            "thinking",
            "memory",
            "web",
            "skills",
            "body",
            "chorus_tools",
        }
    ),
}


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
        categories: AbstractSet[str] | None = None,
        *,
        surface: ToolSurface = "main",
        custom: CustomSurface | None = None,
        include_quantum_submit: bool = True,
        body_url: str | None = None,
        sleep_enabled: bool = False,
    ) -> "ToolRegistry":
        if surface == "custom" and custom is None:
            raise ValueError("Custom tool surfaces require a CustomSurface.")
        if custom is not None:
            if surface != "custom":
                raise ValueError(f"Found surface={surface} instead of `custom`.")
            main_tools = _main_tools(
                body_enabled=body_url is not None, sleep_enabled=sleep_enabled
            )
            custom_tools = [
                tool
                for tool in main_tools
                if tool.category
                in resolve_tool_categories(custom.include_tool_categories)
            ]
            custom_tools.extend(custom.tools)
            return cls(tools=custom_tools)
        surface_tools = _tools_for_surface(
            surface, body_enabled=body_url is not None, sleep_enabled=sleep_enabled
        )
        if surface == "quantum" and not include_quantum_submit:
            surface_tools = [
                tool for tool in surface_tools if tool.name != SUBMIT_RESULT_TOOL.name
            ]
        if categories is None and surface != "main":
            return cls(tools=surface_tools)
        resolved_categories = resolve_tool_categories(categories)
        filtered_tools = [
            tool for tool in surface_tools if tool.category in resolved_categories
        ]
        return cls(tools=filtered_tools)


def resolve_tool_categories(categories: AbstractSet[str] | None) -> set[str]:
    requested = {"main"} if categories is None else set(categories)
    resolved: set[str] = set()
    for category in requested:
        expanded = CATEGORY_HIERARCHY.get(category)
        if expanded is None:
            resolved.add(category)
        else:
            resolved.update(expanded)
    return resolved


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
    REFLECT_TOOL_RESULTS_TOOL,
    FORGET_TOOL,
    FORGET_TOOL_RESULT_TOOL,
    CONFIGURE_TOOL,
    PIN_TOOL,
    RECALL_TOOL,
    REACH_TOOL,
]

BODY_ENABLED_MAIN_TOOLS: list[Tool[Any]] = [*MAIN_TOOLS, BODY_TOOL]

# Fork-scoped tools. These are protocol tools for child sessions, not part of
# the normal model-facing Spellbook surface.
BLOCK_DETECTOR_TOOLS: list[Tool[Any]] = [
    PROPOSE_BLOCK_TOOL,
    AMEND_BLOCK_TOOL,
    COMPLETE_BLOCK_TOOL,
]

BLOCK_SUMMARIZER_TOOLS: list[Tool[Any]] = [SUMMARIZE_TOOL]

QUANTUM_TOOLS: list[Tool[Any]] = [
    READ_TOOL,
    REFLECT_TOOL,
    REFLECT_TOOL_RESULTS_TOOL,
    RECALL_TOOL,
    SUBMIT_RESULT_TOOL,
]

TOOLS_BY_SURFACE: dict[ToolSurface, list[Tool[Any]]] = {
    "main": MAIN_TOOLS,
    "block_detector": BLOCK_DETECTOR_TOOLS,
    "block_summarizer": BLOCK_SUMMARIZER_TOOLS,
    "quantum": QUANTUM_TOOLS,
}


def _main_tools(*, body_enabled: bool, sleep_enabled: bool = False) -> list[Tool[Any]]:
    tools = BODY_ENABLED_MAIN_TOOLS if body_enabled else MAIN_TOOLS
    if sleep_enabled:
        return [*tools, SLEEP_TOOL]
    return list(tools)


def _tools_for_surface(
    surface: ToolSurface, *, body_enabled: bool, sleep_enabled: bool = False
) -> list[Tool[Any]]:
    if surface == "main":
        return _main_tools(body_enabled=body_enabled, sleep_enabled=sleep_enabled)
    return TOOLS_BY_SURFACE[surface]


# Every tool this binary knows how to validate and execute.
ALL_TOOLS: list[Tool[Any]] = (
    BODY_ENABLED_MAIN_TOOLS
    + BLOCK_DETECTOR_TOOLS
    + BLOCK_SUMMARIZER_TOOLS
    + [SUBMIT_RESULT_TOOL]
)

DEFAULT_TOOL_REGISTRY = ToolRegistry.build(categories=None, surface="main")
KNOWN_TOOL_REGISTRY = ToolRegistry(tools=ALL_TOOLS)
