"""Tool execution service.

The ``Executor`` takes a list of ``IRToolCallBlock``s produced by the
model, dispatches each through the ``ToolRegistry``, and returns an
``IRExecution`` containing the result blocks.

Invariants:
- Every call produces exactly one ``IRToolResultBlock`` in the result,
  in declared order. ``zip(calls, execution.blocks)`` is safe.
- Unknown tool names and validation errors become error result blocks
  (``is_error=True``), not exceptions — the loop continues so the model
  can see what went wrong and recover.
- Tool-raised ``ToolError`` becomes an error result block with the
  message text as content.

Cancellation is cooperative — checked between calls and (eventually)
threaded into tool execution itself.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from spellbook.fork import ForkConfig
from spellbook.skills.manager import SkillManager

from .cancel_token import CancelToken
from .config import SpellbookConfig
from .ir_types import IRExecution, IRToolCallBlock, IRToolResultBlock, IRToolTextBlock
from .tools.common import (
    ToolError,
    build_tool_metadata,
)
from .tools.registry import ToolRegistry

if TYPE_CHECKING:
    from .homunculus.homunculus import Homunculus


class Executor:
    def __init__(
        self,
        config: SpellbookConfig,
        transcript_path: Path,
        registry: ToolRegistry,
        skill_manager: SkillManager | None = None,
        homunculus: Homunculus | None = None,
        fork_config: ForkConfig | None = None,
    ):
        self._registry = registry
        self.meta = build_tool_metadata(
            config, transcript_path, homunculus, skill_manager, fork_config
        )

    # TODO: actually implement the cancel stuff
    async def run(
        self, calls: list[IRToolCallBlock], cancel_token: CancelToken
    ) -> IRExecution:
        """Dispatch tool calls in declared order.

        Checks cancellation between calls and during long-running individual calls.
        Cancellation is authoritative — in-flight tools are killed, not awaited.

        ALL calls will always have a corresponding result block. When `cancelled_early`,
        results for calls that didn't run will include an error reporting execution was interrupted
        before running, kind of like the existing interrupt error result path."""
        result_blocks: list[IRToolResultBlock] = []
        cancelled_early = False
        for idx, call in enumerate(calls):
            if cancel_token.cancelled:
                cancelled_early = True
                result_blocks.extend(
                    self._cancelled_result_block(pending) for pending in calls[idx:]
                )
                break
            try:
                tool = self._registry.get(call.tool)
                if tool is None:
                    raise ToolError(f"'{call.tool}' is not a valid tool name.")
                try:
                    validated_input = tool.input_model.model_validate(call.input)
                except ValidationError as e:
                    raise ToolError(
                        f"Error validation tool input arguments:\n{str(e)}"
                    ) from e
                exec_task = asyncio.ensure_future(tool.exec(self.meta, validated_input))
                cancel_task = asyncio.create_task(cancel_token.wait_cancelled())
                done, _ = await asyncio.wait(
                    {exec_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
                )
                if exec_task in done:
                    cancel_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await cancel_task
                    exec_result = exec_task.result()
                else:
                    cancelled_early = True
                    exec_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await exec_task
                    result_blocks.append(self._cancelled_result_block(call))
                    result_blocks.extend(
                        self._cancelled_result_block(pending)
                        for pending in calls[idx + 1 :]
                    )
                    break
                result_blocks.append(
                    IRToolResultBlock(
                        call_id=call.call_id,
                        tool=call.tool,
                        content=exec_result.content,
                        display=exec_result.display,
                    )
                )
            except ToolError as e:
                result_blocks.append(
                    IRToolResultBlock(
                        call_id=call.call_id,
                        tool=call.tool,
                        content=[IRToolTextBlock(text=e.message)],
                        is_error=True,
                    )
                )

        return IRExecution(blocks=result_blocks, cancelled_early=cancelled_early)

    def _cancelled_result_block(self, call: IRToolCallBlock) -> IRToolResultBlock:
        return IRToolResultBlock(
            call_id=call.call_id,
            tool=call.tool,
            content=[IRToolTextBlock(text="Tool execution interrupted by shutdown.")],
            is_error=True,
        )
