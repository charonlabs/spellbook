from pydantic import BaseModel, Field

from spellbook.ir_types import (
    IRSemanticBlockFacet,
    IRSemanticBlockSummary,
    IRToolTextBlock,
)
from spellbook.tools.common import (
    BlockSummarizerToolMetadata,
    Tool,
    ToolExecutionResult,
    ToolMetadata,
)


class SummaryFacet(BaseModel):
    """A distinct thread or sub-topic within the block."""

    title: str = Field(
        description="Short descriptive title for this facet. Be specific: 'ForkRunner Callback Design' not 'Design Discussion'."
    )
    description: str = Field(
        description="1-2 sentences describing what happened in this facet. Prefer directly evidenced facts over inferred details."
    )
    block_range: tuple[int, int] = Field(
        description="The (start, end) context block numbers this facet spans, inclusive."
    )
    resources: list[str] = Field(
        default_factory=list,
        description="File paths, doc paths, commit hashes, or URLs — the breadcrumbs a future mind follows to re-ground. Only include resources explicitly mentioned in the conversation.",
    )


class SummarizeInput(BaseModel):
    """Write a structured summary for the given semantic block. Capture both meaning (why it mattered, how understanding changed) and evidence (what was decided, what was built, what files were touched)."""

    headline: str = Field(
        description="One-line headline for the block. This appears in the block list and Reflect output. Be specific and descriptive: 'Wire BlockManager with Mode Rendering and Gapless-Prefix Validation' not 'Homunculus Work'."
    )
    summary: str = Field(
        description="2-4 sentence paragraph capturing the arc of the block: what started, what happened, what resulted. Write plainly and concretely. Do not narrate the conversation ('the user asked...') — state what happened directly."
    )
    facets: list[SummaryFacet] = Field(
        description="The key threads within the block. Each facet is a distinct piece of work, decision, or topic. Order them chronologically. Aim for 2-6 facets per block."
    )
    open_thread: str | None = Field(
        default=None,
        description="If the block ends with unfinished work, describe what's pending. A future mind needs to know what's dangling. None if the block reaches a clean stopping point.",
    )


async def exec_summarize(
    meta: ToolMetadata, input: SummarizeInput
) -> ToolExecutionResult:
    assert isinstance(meta, BlockSummarizerToolMetadata)
    facets: list[IRSemanticBlockFacet] = []
    for f in input.facets:
        facets.append(
            IRSemanticBlockFacet(
                title=f.title,
                description=f.description,
                start_block=f.block_range[0],
                end_block=f.block_range[1],
                resources=f.resources,
            )
        )
    meta.new_summary = IRSemanticBlockSummary(
        headline=input.headline,
        text=input.summary,
        facets=facets,
        toks=None,
        open_thread=input.open_thread,
    )
    return ToolExecutionResult(content=[IRToolTextBlock(text="Summary recorded.")])


SUMMARIZE_TOOL = Tool(
    name="Summarize",
    input_model=SummarizeInput,
    exec=exec_summarize,
    category="block_summarization",
)
