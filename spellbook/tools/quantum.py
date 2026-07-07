from pydantic import BaseModel, Field, JsonValue

from spellbook.ir_types import IRToolTextBlock
from spellbook.tools.common import (
    QuantumForkToolMetadata,
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)


class SubmitResultInput(BaseModel):
    """Submit a structured JSON result and end this quantum fork turn."""

    payload: JsonValue = Field(description="The JSON payload to return to the caller.")


async def exec_submit_result(
    meta: ToolMetadata, input: SubmitResultInput
) -> ToolExecutionResult:
    if not isinstance(meta, QuantumForkToolMetadata):
        raise ToolError("SubmitResult is only available inside quantum forks.")
    meta.submitted = input.payload
    meta.submit_called = True
    return ToolExecutionResult(
        content=[IRToolTextBlock(text="Result submitted.")],
        display={"kind": "submit_result"},
        terminal_stop_reason="end_turn",
    )


SUBMIT_RESULT_TOOL = Tool(
    name="SubmitResult",
    input_model=SubmitResultInput,
    exec=exec_submit_result,
    category="thinking",
)
