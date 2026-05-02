from datetime import datetime, timezone

from pydantic import BaseModel, Field

from spellbook.ir_types import IRBlock, IRSemanticBlockRange, IRToolTextBlock
from spellbook.tools.common import (
    BlockDetectorToolMetadata,
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolMetadata,
)


class ProposeBlockInput(BaseModel):
    """Propose a new semantic block. The block will include context blocks from the beginning of the buffer through
    end_block **inclusive**.So, if you want a semantic block to include context blocks 1, 2, and 3, you would set
    `end_block`=3"""

    title: str = Field(description="Short descriptive title for the block (2-6 words)")

    end_block: int = Field(
        description="The block index of the last context block in the new semantic block, **inclusive**"
    )


def _remaining_buffer_start(meta: BlockDetectorToolMetadata) -> int:
    if not meta.semantic_block_buffer:
        return meta.context_block_start_id
    return meta.semantic_block_buffer[-1].end_block + 1


def _full_context_last_block(meta: BlockDetectorToolMetadata) -> int:
    return meta.context_block_start_id + len(meta.full_context_blocks) - 1


def _context_slice_from_block_id(
    meta: BlockDetectorToolMetadata, start_block: int
) -> list[IRBlock]:
    offset = start_block - meta.context_block_start_id
    return meta.full_context_blocks[offset:]


def _recompute_context_block_buffer(meta: BlockDetectorToolMetadata) -> None:
    next_start = _remaining_buffer_start(meta)
    last = _full_context_last_block(meta)
    if next_start > last:
        meta.context_block_buffer = []
        return
    meta.context_block_buffer = _context_slice_from_block_id(meta, next_start)


def _check_in_bounds(end_block: int, meta: BlockDetectorToolMetadata) -> None:
    first = _remaining_buffer_start(meta)
    last = _full_context_last_block(meta)
    if not first <= end_block <= last:
        raise ToolError(
            f"Invalid block bounds. Make sure {first} <= `end_block` <= {last}"
        )


def _find_existing_block(
    meta: BlockDetectorToolMetadata, existing_title: str
) -> tuple[int, IRSemanticBlockRange]:
    for i, b in enumerate(meta.semantic_block_buffer):
        if b.title == existing_title:
            return i, b
    existing_titles = [b.title for b in meta.semantic_block_buffer]
    raise ToolError(
        f"No block with the title `{existing_title}` was found. Try one of: [{existing_titles}]"
    )


def _check_last_block_only(meta: BlockDetectorToolMetadata, found_idx: int) -> None:
    if found_idx != len(meta.semantic_block_buffer) - 1:
        raise ToolError(
            "Only the most recently proposed semantic block can be amended."
        )


async def exec_propose_block(
    meta: ToolMetadata, input: ProposeBlockInput
) -> ToolExecutionResult:
    assert isinstance(meta, BlockDetectorToolMetadata)
    _check_in_bounds(input.end_block, meta)
    meta.semantic_block_buffer.append(
        IRSemanticBlockRange(
            title=input.title,
            start_block=_remaining_buffer_start(meta),
            end_block=input.end_block,
        )
    )
    meta.touched_block_titles.add(input.title)
    _recompute_context_block_buffer(meta)

    return ToolExecutionResult(
        content=[
            IRToolTextBlock(
                text=f"Block successfully proposed. The context block buffer now starts at block {_remaining_buffer_start(meta)}"
            )
        ]
    )


class AmendBlockInput(BaseModel):
    """Amend a buffered semantic block, keyed by block title. Use this to change the title and/or
    the `end_block` based on new information."""

    existing_title: str = Field(
        description="The title of the buffered semantic block to amend."
    )

    new_end_block: int | None = Field(
        default=None,
        description="The new `end_block` **inclusive** of the block. The existing `end_block` is left unchanged when omitted.",
    )

    new_title: str | None = Field(
        default=None,
        description="The new title of the block. The existing title is left unchanged when omitted.",
    )


async def exec_amend_block(
    meta: ToolMetadata, input: AmendBlockInput
) -> ToolExecutionResult:
    assert isinstance(meta, BlockDetectorToolMetadata)
    if input.new_end_block is None and input.new_title is None:
        raise ToolError(
            "Invalid input. Either one of `new_end_block` or `new_title` must be set to call `AmendBlock`."
        )

    found_idx, existing_block = _find_existing_block(meta, input.existing_title)
    _check_last_block_only(meta, found_idx)

    new_block = existing_block
    if input.new_end_block is not None:
        if input.new_end_block < existing_block.start_block:
            raise ToolError(
                f"Invalid block bounds. Make sure {existing_block.start_block} <= `new_end_block` <= {_full_context_last_block(meta)}"
            )

        meta.semantic_block_buffer.pop()
        try:
            _check_in_bounds(input.new_end_block, meta)
        except ToolError:
            meta.semantic_block_buffer.append(existing_block)
            raise
        meta.semantic_block_buffer.append(existing_block)

        new_block = new_block.model_copy(update={"end_block": input.new_end_block})

    if input.new_title is not None:
        new_block = new_block.model_copy(update={"title": input.new_title})

    meta.semantic_block_buffer[found_idx] = new_block
    meta.touched_block_titles.discard(input.existing_title)
    meta.touched_block_titles.add(new_block.title)
    _recompute_context_block_buffer(meta)

    return ToolExecutionResult(
        content=[
            IRToolTextBlock(
                text=f"Block successfully amended. The context block buffer now starts at block {_remaining_buffer_start(meta)}"
            )
        ]
    )


class CompleteBlockInput(BaseModel):
    """Mark an existing buffered semantic block as completed. Blocks proposed or amended in the
    current detector session cannot be completed until a later invocation."""

    existing_title: str = Field(
        description="The title of the buffered semantic block to mark completed."
    )


async def exec_complete_block(
    meta: ToolMetadata, input: CompleteBlockInput
) -> ToolExecutionResult:
    assert isinstance(meta, BlockDetectorToolMetadata)
    found_idx, existing_block = _find_existing_block(meta, input.existing_title)

    if existing_block.title in meta.touched_block_titles:
        raise ToolError(
            f"Block `{existing_block.title}` was proposed or amended in this detector session and cannot be completed until a later invocation."
        )

    completed_block = existing_block.model_copy(
        update={
            "completed": True,
            "completed_at": datetime.now(timezone.utc),
        }
    )
    meta.semantic_block_buffer[found_idx] = completed_block

    return ToolExecutionResult(
        content=[
            IRToolTextBlock(
                text=f"Block `{completed_block.title}` successfully marked complete."
            )
        ]
    )


PROPOSE_BLOCK_TOOL = Tool(
    name="ProposeBlock",
    input_model=ProposeBlockInput,
    exec=exec_propose_block,
    category="block_detection",
)

AMEND_BLOCK_TOOL = Tool(
    name="AmendBlock",
    input_model=AmendBlockInput,
    exec=exec_amend_block,
    category="block_detection",
)

COMPLETE_BLOCK_TOOL = Tool(
    name="CompleteBlock",
    input_model=CompleteBlockInput,
    exec=exec_complete_block,
    category="block_detection",
)
