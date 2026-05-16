"""Shared primitives for the tool system.

A ``Tool`` is a frozen Pydantic model bundling the tool's name, input
schema (a Pydantic BaseModel subclass), exec function, and category.
Tools are generic over their input type — ``Tool[BashInput]`` binds
the exec signature to ``Callable[[ToolMetadata, BashInput], ...]``,
so the type system catches shape mismatches at construction.

``ToolExecutionResult`` is what a tool returns on success: IR content
blocks (text or image) and an arbitrary display dict for tool-card
rendering. Errors flow through ``ToolError`` instead; the Executor
catches and builds an error result block.

``ToolMetadata`` is the mutable dispatch context passed to every tool
(cwd, cancel_token). Kept as a dataclass because mutation and rich
types (CancelToken) make Pydantic awkward.

``TOOL_DESCS_DIR`` points at the markdown description directory.
Providers read per-tool descriptions from here when generating API
schemas; tools without a markdown file fall back to their input
model's docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from spellbook.config import SpellbookConfig
from spellbook.fork import BlockDetectorConfig, BlockSummarizerConfig, ForkConfig
from spellbook.skills.manager import SkillManager

from ..cancel_token import CancelToken
from ..ir_types import (
    IRBlock,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRToolRecord,
    IRToolResultContentBlock,
)

if TYPE_CHECKING:
    from ..homunculus.homunculus import Homunculus

ToolCategory = Literal[
    "filesystem",
    "memory",
    "skills",
    "web",
    "thinking",
    "block_detection",
    "block_summarization",
]

TOOL_DESCS_DIR = Path(__file__).parent / "descs"


class ToolExecutionResult(BaseModel, frozen=True):
    """A result of a tool execution"""

    content: list[IRToolResultContentBlock]
    display: dict = Field(
        default_factory=dict
    )  # TODO: replace with real display types once those exist


@dataclass  # not pydantic because mutable, internal state
class ToolMetadata:
    cwd: Path
    transcript_path: Path
    skill_manager: SkillManager | None = None
    homunculus: Homunculus | None = None
    cancel_token: CancelToken | None = None


# I'm using inheritance here b/c idk how to do the union thingy with dataclasses
# This also feels weird to keep here - in general, we probably should think some
# more about how to do this in the forking/sub-session pattern
@dataclass
class BlockDetectorToolMetadata(ToolMetadata):
    prev_semantic_blocks: list[IRSemanticBlockRange] = field(default_factory=list)
    full_context_blocks: list[IRBlock] = field(default_factory=list)
    context_block_buffer: list[IRBlock] = field(default_factory=list)
    context_block_start_id: int = 0
    semantic_block_buffer: list[IRSemanticBlockRange] = field(default_factory=list)
    new_semantic_blocks: list[IRSemanticBlockRange] = field(default_factory=list)
    touched_block_titles: set[str] = field(default_factory=set)


@dataclass
class BlockSummarizerToolMetadata(ToolMetadata):
    new_summary: IRSemanticBlockSummary = field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


class Tool(BaseModel, Generic[T], frozen=True):
    """An item in the tool registry."""

    model_config = ConfigDict(extra="forbid")
    name: str
    input_model: type[T]
    exec: Callable[[ToolMetadata, T], Awaitable[ToolExecutionResult]]
    category: ToolCategory


class ToolError(Exception):
    """Error during tool parsing or execution. Message is safe to return to the model."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def build_tool_metadata(
    config: SpellbookConfig,
    transcript_path: Path,
    homunculus: Homunculus | None,
    skill_manager: SkillManager | None,
    fork_config: ForkConfig | None,
) -> ToolMetadata:
    match config.session_type:
        case "main":
            return ToolMetadata(
                cwd=config.cwd,
                transcript_path=transcript_path,
                homunculus=homunculus,
                skill_manager=skill_manager,
            )
        case "custom":
            return ToolMetadata(
                cwd=config.cwd,
                transcript_path=transcript_path,
                homunculus=homunculus,
                skill_manager=skill_manager,
            )
        case "block_detector":
            assert isinstance(fork_config, BlockDetectorConfig)
            return BlockDetectorToolMetadata(
                cwd=config.cwd,
                transcript_path=transcript_path,
                homunculus=homunculus,
                prev_semantic_blocks=fork_config.prev_semantic_blocks,
                full_context_blocks=fork_config.full_context_blocks,
                context_block_buffer=fork_config.context_block_buffer,
                context_block_start_id=fork_config.context_block_start_id,
                semantic_block_buffer=fork_config.semantic_block_buffer,
                new_semantic_blocks=[],
                touched_block_titles=set(),
            )
        case "block_summarizer":
            assert isinstance(fork_config, BlockSummarizerConfig)
            return BlockSummarizerToolMetadata(
                cwd=config.cwd,
                transcript_path=transcript_path,
                homunculus=homunculus,
            )


def tool_to_record(tool: Tool) -> IRToolRecord:
    desc_path = TOOL_DESCS_DIR / f"{tool.name}.md"
    if desc_path.exists():
        description = desc_path.read_text().strip()
    else:
        description = None
    input_schema = tool.input_model.model_json_schema()
    if description is not None:
        input_schema["description"] = description
    return IRToolRecord(
        name=tool.name,
        input_schema=input_schema,
        category=tool.category,
    )
