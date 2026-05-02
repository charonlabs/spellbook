from pydantic import BaseModel, Field

from spellbook.ir_types import IRToolTextBlock
from spellbook.tools.common import (
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)


class SkillInput(BaseModel):
    """Execute a skill — load specialized instructions into the conversation."""

    name: str = Field(
        description='The skill name. E.g., "summon", "compose", or "browse".',
    )
    args: str | None = Field(
        default=None,
        description="Optional arguments for the skill.",
    )


async def exec_skill(meta: ToolMetadata, input: SkillInput) -> ToolExecutionResult:
    if meta.skill_manager is None or meta.skill_manager.catalog is None:
        raise ToolError(
            (
                "Skills were not initialized for this session. "
                "Skill support is not currently functional."
            )
        )
    try:
        text = meta.skill_manager.invoke(input.name, args=input.args)
    except ValueError as e:
        raise ToolError(str(e)) from e
    return ToolExecutionResult(content=[IRToolTextBlock(text=text)])


SKILL_TOOL: Tool[SkillInput] = Tool(
    name="Skill", input_model=SkillInput, exec=exec_skill, category="skills"
)
