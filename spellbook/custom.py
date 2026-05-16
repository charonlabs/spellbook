from dataclasses import dataclass, field

from spellbook.tools.common import Tool, ToolCategory


@dataclass(frozen=True, slots=True)
class CustomSurface:
    tools: list[Tool]
    include_tool_categories: set[ToolCategory] = field(default_factory=set)
