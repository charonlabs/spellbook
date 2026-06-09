from pydantic import BaseModel, Field

from spellbook.ir_types import IRToolTextBlock
from spellbook.tools.common import (
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)


class ReflectInput(BaseModel):
    """Inspect current awareness state."""

    block_idx: int | None = Field(
        default=None,
        description=(
            "Optional semantic block index from Reflect output. When provided, "
            "Reflect drills into that block and previews the summary rendering "
            "that would be shown after compaction."
        ),
    )


async def exec_reflect(meta: ToolMetadata, input: ReflectInput) -> ToolExecutionResult:
    if meta.homunculus is None:
        raise ToolError(
            "Reflect is unavailable because this session has no Homunculus."
        )
    try:
        result, display = await meta.homunculus.render_reflect(input.block_idx)
    except ValueError as e:
        raise ToolError(str(e)) from e
    return ToolExecutionResult(content=[IRToolTextBlock(text=result)], display=display)


class ReflectToolResultsInput(BaseModel):
    """Inspect current tool results, sizes, and TTL status."""

    verbose: bool = Field(
        default=False,
        description=(
            "When false, show only token-relevant tool results: pending TTLs "
            "and large untracked outputs. When true, show every tool result."
        ),
    )


async def exec_reflect_tool_results(
    meta: ToolMetadata, input: ReflectToolResultsInput
) -> ToolExecutionResult:
    if meta.homunculus is None:
        raise ToolError(
            "ReflectToolResults is unavailable because this session has no Homunculus."
        )
    result, display = meta.homunculus.render_tool_results(verbose=input.verbose)
    return ToolExecutionResult(content=[IRToolTextBlock(text=result)], display=display)


class ForgetInput(BaseModel):
    """Compact a semantic block to its summary."""

    block_idx: int = Field(
        description="The index of the block to compact, as show in `Reflect` output."
    )

    confirm: bool = Field(
        default=False,
        description=(
            "Defaults to false. Required only when forgetting a pinned block. "
            "The first call without confirm returns a warning; "
            "call again with confirm=true to proceed."
        ),
    )


async def exec_forget(meta: ToolMetadata, input: ForgetInput) -> ToolExecutionResult:
    if meta.homunculus is None:
        raise ToolError("Forget is unavailable because this session has no Homunculus.")
    try:
        await meta.homunculus.forget(input.block_idx, input.confirm)
    except ValueError as e:
        raise ToolError(str(e)) from e
    return ToolExecutionResult(
        content=[
            IRToolTextBlock(text=f"Block {input.block_idx} successfully compacted.")
        ]
    )


class ForgetToolResultInput(BaseModel):
    """Forget a specific tool result by collapsing it immediately."""

    call_id: str = Field(
        description=(
            "The tool result call_id to forget. Unique prefixes from "
            "`ReflectToolResults` are accepted."
        ),
    )


async def exec_forget_tool_result(
    meta: ToolMetadata, input: ForgetToolResultInput
) -> ToolExecutionResult:
    if meta.homunculus is None:
        raise ToolError(
            "ForgetToolResult is unavailable because this session has no Homunculus."
        )
    try:
        text = await meta.homunculus.forget_tool_result(input.call_id)
    except ValueError as e:
        raise ToolError(str(e)) from e
    return ToolExecutionResult(
        content=[IRToolTextBlock(text=text)],
        display={"kind": "forget_tool_result", "call_id": input.call_id},
    )


class ConfigureInput(BaseModel):
    """Inspect or update runtime configuration."""

    key: str | None = Field(
        default=None,
        description=(
            "Optional runtime config key to update. Omit both key and value to read "
            "the current configuration."
        ),
    )

    value: str | int | bool | None = Field(
        default=None,
        description=(
            "Optional value for the runtime config key. Omit both key and value to "
            "read the current configuration."
        ),
    )


async def exec_configure(
    meta: ToolMetadata, input: ConfigureInput
) -> ToolExecutionResult:
    if meta.homunculus is None:
        raise ToolError(
            "Configure is unavailable because this session has no Homunculus."
        )
    try:
        text, display = meta.homunculus.configure(
            key=input.key,
            value=input.value,
        )
    except ValueError as e:
        raise ToolError(str(e)) from e
    return ToolExecutionResult(content=[IRToolTextBlock(text=text)], display=display)


class PinInput(BaseModel):
    """Pin a semantic block or summary facet to protect it from compaction."""

    block_idx: int = Field(
        description="The index of the block to pin, as shown in `Reflect` output."
    )

    facet_id: str | None = Field(
        default=None,
        description=(
            "Optional summary facet id to pin within the block. When omitted, "
            "the whole block is pinned."
        ),
    )

    reason: str = Field(
        description="The reason why you're pinning this block - only visible to future-you."
    )


async def exec_pin(meta: ToolMetadata, input: PinInput) -> ToolExecutionResult:
    if meta.homunculus is None:
        raise ToolError("Pin is unavailable because this session has no Homunculus.")
    try:
        await meta.homunculus.pin(input.block_idx, input.reason, input.facet_id)
    except ValueError as e:
        raise ToolError(str(e)) from e
    if input.facet_id is not None:
        text = (
            f'Facet "{input.facet_id}" in block {input.block_idx} successfully pinned. '
            "It will be preserved as original conversation when the block is compacted."
        )
    else:
        text = f"Block {input.block_idx} successfully pinned. It will no longer be compacted."
    return ToolExecutionResult(content=[IRToolTextBlock(text=text)])


class RecallInput(BaseModel):
    """Recall content from a compacted semantic block back into your awareness."""

    block_idx: int = Field(
        description=(
            "The index of the block to recall, as shown in the opening tag of the summary, "
            "or in `Reflect` output."
        )
    )


async def exec_recall(meta: ToolMetadata, input: RecallInput) -> ToolExecutionResult:
    if meta.homunculus is None:
        raise ToolError("Recall is unavailable because this session has no Homunculus.")
    try:
        text = await meta.homunculus.recall(input.block_idx)
    except ValueError as e:
        raise ToolError(str(e)) from e
    return ToolExecutionResult(content=[IRToolTextBlock(text=text)])


REFLECT_TOOL: Tool[ReflectInput] = Tool(
    name="Reflect",
    input_model=ReflectInput,
    exec=exec_reflect,
    category="memory",
)

REFLECT_TOOL_RESULTS_TOOL: Tool[ReflectToolResultsInput] = Tool(
    name="ReflectToolResults",
    input_model=ReflectToolResultsInput,
    exec=exec_reflect_tool_results,
    category="memory",
)

FORGET_TOOL: Tool[ForgetInput] = Tool(
    name="Forget",
    input_model=ForgetInput,
    exec=exec_forget,
    category="memory",
)

FORGET_TOOL_RESULT_TOOL: Tool[ForgetToolResultInput] = Tool(
    name="ForgetToolResult",
    input_model=ForgetToolResultInput,
    exec=exec_forget_tool_result,
    category="memory",
)

CONFIGURE_TOOL: Tool[ConfigureInput] = Tool(
    name="Configure",
    input_model=ConfigureInput,
    exec=exec_configure,
    category="memory",
)

PIN_TOOL: Tool[PinInput] = Tool(
    name="Pin",
    input_model=PinInput,
    exec=exec_pin,
    category="memory",
)

RECALL_TOOL: Tool[RecallInput] = Tool(
    name="Recall",
    input_model=RecallInput,
    exec=exec_recall,
    category="memory",
)
