"""
Canonical IR types for the Spellbook Core rewrite. All systems will speak this language.

`turn_id` and `event_id` are TBD on exactly when/where they will be set.

`IRUserTextBlock` has `system` as origin on footer injections.

`IRStreamEvent` uses `kind` instead of `type` for legacy compat reasons, but can be changed if it proves to be annoying.
(same deal with `Record` and `ir`)

**OPEN**: How do we want to stream tool calls/results? Right now UIs get them just at the end of each API round, but do we want to stream
them at all? Options include streaming complete calls as the come in as one ToolCallEvent, or fine-grained stream the deltas. I'm leaning
against fine-grained streaming because that's a parsing and display nightmare. All current UI code currently doesn't handle streaming of
these and sees them via their own path. Further investigation warranted before we commit either way here.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import SessionType, SpellbookConfig


class IRUserTextBlock(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["user_text"] = "user_text"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_id: str | None = None
    event_id: str | None = None
    origin: Literal["human", "conduit", "system", "memory"]

    text: str


class IRAssistantTextBlock(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["assistant_text"] = "assistant_text"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_id: str | None = None
    event_id: str | None = None
    origin: Literal["model"] = "model"

    text: str


class IRImageBase64Source(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["base64"] = "base64"
    media_type: str
    data: str


# This converts to Base64 on rehydrate
class IRImageBlobSource(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["blob"] = "blob"


class IRImageURLSource(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["url"] = "url"
    url: str


IRImageSource = Annotated[
    IRImageBase64Source | IRImageBlobSource | IRImageURLSource,
    Field(discriminator="type"),
]

IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class IRImageBlock(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image"] = "image"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_id: str | None = None
    event_id: str | None = None
    origin: Literal["human", "conduit", "system", "tool"]

    source: IRImageSource
    blob_path: str | None = None  # for the persistant blobs alongside the transcript

    @model_validator(mode="after")
    def _validate_blob_path(self) -> Self:
        if isinstance(self.source, IRImageBlobSource):
            if self.blob_path is None:
                raise ValueError(
                    "Image blocks with a blob source must have a `blob_path` set."
                )
        return self


IRInboundBlock = Annotated[IRUserTextBlock | IRImageBlock, Field(discriminator="type")]


class IRThinkingBlock(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["thinking"] = "thinking"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_id: str | None = None
    event_id: str | None = None
    origin: Literal["model"] = "model"

    text: str = ""
    signature: str = ""


class IRToolCallBlock(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_call"] = "tool_call"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_id: str | None = None
    event_id: str | None = None
    origin: Literal["model"] = "model"

    call_id: str
    tool: str
    input: dict


class IRToolTextBlock(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_text"] = "tool_text"
    origin: Literal["tool"] = "tool"

    text: str


IRToolResultContentBlock = Annotated[
    IRToolTextBlock | IRImageBlock, Field(discriminator="type")
]


class IRToolResultBlock(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["tool_result"] = "tool_result"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_id: str | None = None
    event_id: str | None = None
    origin: Literal["tool"] = "tool"

    call_id: str
    tool: str
    content: list[IRToolResultContentBlock] = Field(default_factory=list)
    display: dict = Field(default_factory=dict)
    is_error: bool = False


IRBlock = Annotated[
    IRUserTextBlock
    | IRAssistantTextBlock
    | IRImageBlock
    | IRThinkingBlock
    | IRToolCallBlock
    | IRToolResultBlock,
    Field(discriminator="type"),
]


class IRStreamThinkingStartEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["thinking_start"] = "thinking_start"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IRStreamThinkingDeltaEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["thinking_delta"] = "thinking_delta"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str


class IRStreamThinkingEndEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["thinking_end"] = "thinking_end"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IRStreamTextStartEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["text_start"] = "text_start"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IRStreamTextDeltaEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["text_delta"] = "text_delta"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str


class IRStreamTextEndEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["text_end"] = "text_end"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


IRStreamEvent = Annotated[
    IRStreamThinkingStartEvent
    | IRStreamThinkingDeltaEvent
    | IRStreamThinkingEndEvent
    | IRStreamTextStartEvent
    | IRStreamTextDeltaEvent
    | IRStreamTextEndEvent,
    Field(discriminator="kind"),
]


class IRUsage(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_create_tokens

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_create_tokens
            + self.output_tokens
        )


StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "cancelled",
    "error",
    "pause_turn",
    "refusal",
    "stop_sequence",
    "unspecified",
]

ToolResultTTLTrigger = Literal["end_turn", "seq"]
ToolResultTTLSource = Literal["auto", "manual", "repair"]


class IRGeneration(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_id: str | None = None
    model: str
    blocks: list[IRBlock]
    stop_reason: StopReason
    usage: IRUsage | None

    @property
    def tool_calls(self) -> list[IRToolCallBlock]:
        """Tool call blocks from this generation."""
        return [block for block in self.blocks if isinstance(block, IRToolCallBlock)]


class IRExecution(BaseModel, frozen=True):
    blocks: list[IRToolResultBlock]
    cancelled_early: bool = False


class IRLoopResult(BaseModel, frozen=True):
    blocks: list[IRBlock]
    generations: list[IRGeneration]
    executions: list[IRExecution]
    stop_reason: StopReason
    rounds: int  # how many rounds ran


class IRToolRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    name: str
    input_schema: dict[str, Any]  # tool.input_model.model_json_schema()
    category: str


# "turn" starts or queues a new turn. "inject" joins the active turn at the
# next round boundary, and starts a turn if the session is idle. "footer" is
# ambient context rendered by the footer controller.
InboundDelivery = Literal["turn", "inject", "footer"]


class IRInboundMessage(BaseModel, frozen=True):
    blocks: list[IRInboundBlock]
    source_metadata: dict = Field(default_factory=dict)
    delivery: InboundDelivery


# TODO: these are guesses at names / future stuff. Subject to change.
# Also not sure if we need both Type and Source? Maybe could collapse to just one?
# But both are useful...
# Add more here as needed
FooterType = Literal[
    "gas_gauge",
    "time",
    "reminder",
    "notif",
    "bg_task",
    "compaction",
    "skill_catalog_update",
    "conduit",
    "surface",
]
FooterSource = Literal[
    "telemetry",
    "planner",
    "conduit",
    "idle",
    "detector",
    "skill_manager",
    "runtime",
]


class IRFooter(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    text: str
    id: str = Field(default_factory=lambda: f"footer_{uuid4().hex}")
    type: FooterType
    source: FooterSource
    key: str  # for dedup
    priority: int = 50  # lower = higher rendering prio


PrefixCountMethod = Literal[
    "frame",
    "observed_input",
    "count_blocks",
    "repaired_count_blocks",
    "observed_generation_total",
]
RangeCountMethod = Literal[
    "empty",
    "api",
    "repaired_count_blocks",
    "chunked_count_blocks",
    "prefix_delta",
    "prefix_delta_repaired_boundary",
    "prefix_delta_approximate",
]


class IRTokenPrefixCount(BaseModel, frozen=True):
    """Token count for a half-open canonical block boundary.

    Boundary 0 means before block 0. Boundary N means after blocks[:N].
    """

    model_config = ConfigDict(extra="forbid")
    tokens: int
    method: PrefixCountMethod
    exact: bool


class IRTokenRangeCount(BaseModel, frozen=True):
    """Token count for a half-open canonical block range."""

    model_config = ConfigDict(extra="forbid")
    tokens: int
    method: RangeCountMethod
    exact: bool


class IRSemanticBlockRange(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: f"block_range_{uuid4().hex}")
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str
    start_block: int
    end_block: int
    completed: bool = False
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_order(self) -> Self:
        if self.start_block > self.end_block:
            raise ValueError(
                "Semantic block range start_block must be less than or equal to end_block."
            )
        return self


SemanticBlockMode = Literal["full", "summary"]


class IRSemanticBlockFacet(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: f"facet_{uuid4().hex}")
    title: str
    description: str
    start_block: int
    end_block: int
    resources: list[str]


class IRSemanticBlockSummary(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: f"summary_{uuid4().hex}")
    type: Literal["summary"] = "summary"
    mode: SemanticBlockMode = "summary"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    headline: str
    text: str
    facets: list[IRSemanticBlockFacet]
    open_thread: str | None
    toks: IRTokenRangeCount | None


IRSemanticBlockArtifact = Annotated[IRSemanticBlockSummary, Field(discriminator="type")]


def _default_semantic_block_modes() -> list[SemanticBlockMode]:
    return ["full"]


SemanticBlockPinKind = Literal["block", "facet"]


class IRSemanticBlockPin(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: SemanticBlockPinKind
    reason: str
    facet_id: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.kind == "facet" and self.facet_id is None:
            raise ValueError(
                'IRSemanticBlockPin with kind="facet" needs a `facet_id` that is not None.'
            )
        return self


class IRSemanticBlock(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(default_factory=lambda: f"block_{uuid4().hex}")
    idx: int
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    title: str
    range: IRSemanticBlockRange
    mode: SemanticBlockMode = "full"
    toks: IRTokenRangeCount | None
    full_toks: IRTokenRangeCount | None
    available_modes: list[SemanticBlockMode] = Field(
        default_factory=_default_semantic_block_modes
    )
    artifacts: list[IRSemanticBlockArtifact] = Field(default_factory=list)
    pin: IRSemanticBlockPin | None = None
    facet_pins: list[IRSemanticBlockPin] = Field(default_factory=list)


class IRCompactBlockIntent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["compact_block"] = "compact_block"
    block_idx: int


# TODO: add more intents, TTL probably wants to live here once it lands, as an intent

IRContextPlanIntent = Annotated[IRCompactBlockIntent, Field(discriminator="kind")]


class IRContextPlan(BaseModel, frozen=True):
    """A context plan.
    Contains a list of intents, to be applied by the homunculus/block_manager"""

    intents: list[IRContextPlanIntent]


PlannerResultKind = Literal["proposal", "action"]


class IRPlannerResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: PlannerResultKind
    plan: IRContextPlan


SkillScope = Literal["project", "user"]


class IRSkill(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    location: Path  # absolute path to SKILL.md
    directory: Path  # parent of SKILL.md (the skill's base directory)
    scope: SkillScope


class IRSkillCatalog(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    skills: dict[str, IRSkill] = Field(default_factory=dict)  # name -> skill


class IRSkillCatalogDelta(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    added: dict[str, IRSkill]
    updated: dict[str, IRSkill]
    removed: list[str]


# ALL RECORDS should start with `session_id`, `ir`, `time`, in that order
# all records that occur within a turn context should end with `turn` and `turn_id`


# TODO: replace with frame/profile once that exists
class IRSessionRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["session"] = "session"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: SpellbookConfig
    tools: list[IRToolRecord]
    skill_catalog: IRSkillCatalog


class IRTurnStartRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["turn_start"] = "turn_start"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn: int
    turn_id: str


# TODO: put UI display stuff on this
class IRTurnEndRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["turn_end"] = "turn_end"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stop_reason: StopReason | None = None
    turn: int
    turn_id: str


class IRBlockRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["event"] = "event"
    turn: int
    seq: int
    event: IRBlock


class IRToolResultTTLRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["tool_result_ttl"] = "tool_result_ttl"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    call_id: str
    replace_content: str
    ttl: int
    trigger: ToolResultTTLTrigger = "end_turn"
    delivered_turn: int
    source: ToolResultTTLSource = "auto"
    output_ref: str | None = None
    turn: int
    turn_id: str


class IRFooterQueueRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["footer_queue"] = "footer_queue"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    footer: IRFooter
    turn: int
    turn_id: str


class IRFooterDrainRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["footer_drain"] = "footer_drain"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    footers: list[IRFooter]
    turn: int
    turn_id: str


class IRBlockDetectionRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["block_detection"] = "block_detection"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed: list[IRSemanticBlockRange]
    still_buffered: list[IRSemanticBlockRange]
    turn: int
    turn_id: str


class IRSemanticBlockRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["semantic_block"] = "semantic_block"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    id: str
    idx: int
    range_id: str
    toks: IRTokenRangeCount | None
    full_toks: IRTokenRangeCount | None
    turn: int
    turn_id: str


class IRSemanticBlockArtifactRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["semantic_block_artifact"] = "semantic_block_artifact"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    block_id: str
    artifact: IRSemanticBlockArtifact
    turn: int
    turn_id: str


class IRSemanticBlockMetricsRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["semantic_block_metrics"] = "semantic_block_metrics"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    block_id: str
    toks: IRTokenRangeCount
    turn: int
    turn_id: str


class IRSemanticBlockPinRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["semantic_block_pin"] = "semantic_block_pin"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    block_id: str
    pin: IRSemanticBlockPin
    turn: int
    turn_id: str


SemanticBlockApplyModeSource = Literal["model", "planner"]


class IRSemanticBlockApplyModeRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["semantic_block_apply_mode"] = "semantic_block_apply_mode"
    block_id: str
    mode: SemanticBlockMode
    source: SemanticBlockApplyModeSource = "model"
    turn: int
    turn_id: str


class IRContextPlanProposalRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["context_plan_proposal"] = "context_plan_proposal"
    plan: IRContextPlan
    turn: int
    turn_id: str


class IRForkSummonRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["fork_summon"] = "fork_summon"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fork_id: str
    fork_type: SessionType
    child_transcript_path: str
    turn: int
    turn_id: str


class IRForkShutdownRecord(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    ir: Literal["fork_shutdown"] = "fork_shutdown"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    fork_id: str
    turn: int
    turn_id: str


class IRSkillCatalogUpdateRecord(BaseModel, frozen=True):
    session_id: str
    ir: Literal["skill_catalog_update"] = "skill_catalog_update"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delta: IRSkillCatalogDelta
    turn: int
    turn_id: str


IRRecord = Annotated[
    IRSessionRecord
    | IRTurnStartRecord
    | IRTurnEndRecord
    | IRBlockRecord
    | IRToolResultTTLRecord
    | IRFooterQueueRecord
    | IRFooterDrainRecord
    | IRBlockDetectionRecord
    | IRSemanticBlockRecord
    | IRSemanticBlockArtifactRecord
    | IRSemanticBlockMetricsRecord
    | IRSemanticBlockPinRecord
    | IRSemanticBlockApplyModeRecord
    | IRContextPlanProposalRecord
    | IRForkSummonRecord
    | IRForkShutdownRecord
    | IRSkillCatalogUpdateRecord,
    Field(discriminator="ir"),
]
