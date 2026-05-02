from collections.abc import Sequence
from html import escape
from typing import Literal

from pydantic import BaseModel, ConfigDict

from spellbook.config import HomunculusConfig
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRCompactBlockIntent,
    IRContextPlan,
    IRContextPlanIntent,
    IRImageBlobSource,
    IRImageBlock,
    IRImageURLSource,
    IRSemanticBlock,
    IRSemanticBlockRange,
    IRThinkingBlock,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
)

RegimeType = Literal["calm", "warning", "forced", "critical", "unknown"]


class AwarenessBudgetSnapshot(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    max_tokens: int
    reserve_output_tokens: int  # max_tokens - hard_threshold
    current_input_tokens: int | None
    current_slack_tokens: int | None
    regime: RegimeType
    warning_threshold: int  # soft threshold
    forced_threshold: int  # medium threshold
    critical_threshold: int  # hard threshold


class AwarenessTailSnapshot(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    tail_start: int  # start block idx
    tail_end: int  # end block idx
    toks: int | None


class AwarenessHomunculusSnapshot(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    budget: AwarenessBudgetSnapshot
    semantic_blocks: list[IRSemanticBlock]
    proposed_blocks: list[IRSemanticBlockRange]
    tail: AwarenessTailSnapshot
    plan_proposal: IRContextPlan | None


def calc_regime(config: HomunculusConfig, tokens: int | None) -> RegimeType:
    if tokens is None:
        return "unknown"
    if tokens >= config.hard_threshold:
        return "critical"
    if tokens >= config.medium_threshold:
        return "forced"
    if tokens >= config.soft_threshold:
        return "warning"
    return "calm"


def render_context_block(block: IRBlock, block_id: int | None = None) -> str:
    content = escape(_render_block_markdown(block))
    id_slug = f' id="{block_id}"' if block_id is not None else ""
    return f"<context_block{id_slug}>\n{content}\n</context_block>"


def render_summary(block: IRSemanticBlock) -> IRUserTextBlock:
    artifact = next(a for a in block.artifacts if a.type == "summary")

    parts: list[str] = []
    parts.append(
        f'<spellbook-memory block_idx="{block.idx}" mode="summary" turns="{block.range.start_block}-{block.range.end_block}">'
    )
    parts.append(f"# {artifact.headline}")
    parts.append("")
    parts.append(artifact.text)

    if artifact.facets:
        parts.append("")
        parts.append("## Facets")
        for facet in artifact.facets:
            parts.append(
                f"- {facet.title} (blocks {facet.start_block}-{facet.end_block})"
            )
            parts.append(f"  {facet.description}")
            if facet.resources:
                parts.append(f"  Resources: {'; '.join(facet.resources)}")

    if artifact.open_thread:
        parts.append("")
        parts.append(f"Open thread: {artifact.open_thread}")

    parts.append("</spellbook-memory>")

    return IRUserTextBlock(text="\n".join(parts), origin="memory")


def render_plan(
    plan: IRContextPlan,
    semantic_blocks: Sequence[IRSemanticBlock],
) -> str:
    return "\n".join(
        f"- {render_intent(intent, semantic_blocks)}" for intent in plan.intents
    )


def render_intent(
    intent: IRContextPlanIntent,
    semantic_blocks: Sequence[IRSemanticBlock],
    *,
    verb: Literal["compact", "compacted"] = "compact",
) -> str:
    match intent:
        case IRCompactBlockIntent():
            block = _block_by_idx(semantic_blocks, intent.block_idx)
            if block is None:
                return (
                    f"{verb} [Block {intent.block_idx}]: "
                    "(block unavailable, savings: unknown)"
                )
            savings = estimate_intent_savings(intent, semantic_blocks)
            return (
                f'{verb} [Block {intent.block_idx}]: "{block.title}" '
                f"({_format_token_savings(savings)})"
            )
        case _:
            raise NotImplementedError(
                f"Cannot render context plan intent of type {type(intent)}."
            )


def estimate_intent_savings(
    intent: IRContextPlanIntent,
    semantic_blocks: Sequence[IRSemanticBlock],
) -> int | None:
    match intent:
        case IRCompactBlockIntent():
            block = _block_by_idx(semantic_blocks, intent.block_idx)
            if block is None:
                return None
            full_toks = block.full_toks
            if full_toks is None and block.mode == "full":
                full_toks = block.toks
            summary = next(
                (
                    artifact
                    for artifact in block.artifacts
                    if artifact.mode == "summary"
                ),
                None,
            )
            if full_toks is None or summary is None or summary.toks is None:
                return None
            return full_toks.tokens - summary.toks.tokens
        case _:
            raise NotImplementedError(
                f"Cannot estimate savings for context plan intent of type {type(intent)}."
            )


def _render_block_markdown(block: IRBlock) -> str:
    match block:
        case IRUserTextBlock():
            label = {
                "human": "User",
                "conduit": "Conduit",
                "system": "System",
            }[block.origin]
            return f"**{label}:** {block.text}"
        case IRAssistantTextBlock():
            return f"**Assistant:** {block.text}"
        case IRThinkingBlock():
            return f"**Thinking:** {block.text}"
        case IRToolCallBlock():
            return (
                f"**Tool call (`{block.tool}`):** "
                f"call_id={block.call_id}, input={block.input}"
            )
        case IRToolResultBlock():
            parts: list[str] = []
            for item in block.content:
                if isinstance(item, IRToolTextBlock):
                    parts.append(item.text)
                elif isinstance(item, IRImageBlock):
                    parts.append(_render_image_markdown(item))
            body = "\n".join(parts).strip()
            if not body:
                body = "(no textual content)"
            prefix = f"**Tool result (`{block.tool}`):** "
            if block.is_error:
                prefix = f"**Tool result error (`{block.tool}`):** "
            return prefix + body
        case IRImageBlock():
            return _render_image_markdown(block)


def _render_image_markdown(block: IRImageBlock) -> str:
    source = block.source
    if isinstance(source, IRImageURLSource):
        return f"**Image ({block.origin}):** {source.url}"
    if isinstance(source, IRImageBlobSource):
        return f"**Image ({block.origin}):** <blob:{block.blob_path}>"
    return f"**Image ({block.origin}):** <base64:{source.media_type}>"


def _block_by_idx(
    semantic_blocks: Sequence[IRSemanticBlock],
    block_idx: int,
) -> IRSemanticBlock | None:
    return next((block for block in semantic_blocks if block.idx == block_idx), None)


def _format_token_savings(savings: int | None) -> str:
    if savings is None:
        return "savings: unknown"
    if savings < 0:
        return f"cost: ~{abs(savings)} tokens"
    return f"savings: ~{savings} tokens"
