from dataclasses import dataclass, field
from collections.abc import Set as AbstractSet

from spellbook.tools.common import Tool


@dataclass(frozen=True, slots=True)
class CustomSurface:
    tools: list[Tool]
    include_tool_categories: AbstractSet[str] = field(default_factory=set)
