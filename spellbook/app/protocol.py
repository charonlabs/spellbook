"""Protocol types for Core App Server.

IR for transcript/runtime facts, custom protocol for server/session facts.

The invariant is:

- If it is canonical conversation or transcript truth, payload should be IR or an IR record.
- If it is transport/control/UI state, make a server protocol type.
- If it is a convenience projection, name it as a projection/snapshot so it never masquerades as truth.

That keeps the seam honest. We avoid parallel fake versions of `IRBlock`,
but we also avoid forcing clients to infer “runtime is idle” or “this turn just started” from lower-level
transcript events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spellbook.homunculus.common import AwarenessHomunculusSnapshot
from spellbook.ir_types import (
    IRBlock,
    IRInboundMessage,
    IRLoopResult,
    IRRecord,
    IRStreamEvent,
)
from spellbook.nursery import AwarenessNurserySnapshot
from spellbook.rehydrator import RehydrationResult

if TYPE_CHECKING:
    from spellbook.session_manager import SessionState

RuntimeState = Literal["idle", "running", "dreaming", "suspended"]
ConduitType = Literal["context", "message", "notification"]
ConduitAction = Literal["started_turn", "queued_as_message", "queued_as_context"]
WebSocketCatchupMode = Literal["none", "lite", "full"]


def session_to_runtime_state(state: SessionState) -> RuntimeState:
    match state:
        case "idle":
            return "idle"
        case "running":
            return "running"
        case "dreaming":
            return "dreaming"
        case "suspended":
            return "suspended"
        case _:
            raise ValueError(
                f"There is no corresponding runtime state for session state `{state}`"
            )


class StreamEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["stream"] = "stream"
    event: IRStreamEvent


class ContextBlockAddedEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["context_block_added"] = "context_block_added"
    block: IRBlock


# This one is for debug stuff mostly? Probably
# still want it even though we end up duplicating
# some other events.
class RecordWrittenEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["record_written"] = "record_written"
    record: IRRecord


# This supercedes the old "message_queued_delivered" via
# the `message` field
class TurnStartedEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["turn_started"] = "turn_started"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn: int
    turn_id: str
    message: IRInboundMessage


# Full `IRLoopResult` might be heavy here, idk
class TurnEndedEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["turn_ended"] = "turn_ended"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn: int
    turn_id: str
    result: IRLoopResult


class RuntimeStateEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["runtime_state"] = "runtime_state"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: RuntimeState


class MessageQueuedEvent(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["message_queued"] = "message_queued"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: IRInboundMessage


ServerEvent = Annotated[
    StreamEvent
    | ContextBlockAddedEvent
    | RecordWrittenEvent
    | TurnStartedEvent
    | TurnEndedEvent
    | RuntimeStateEvent
    | MessageQueuedEvent,
    Field(discriminator="kind"),
]


class AwarenessSnapshot(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    homunculus: AwarenessHomunculusSnapshot
    nursery: AwarenessNurserySnapshot
    surface: str | None
    surface_time: datetime | None


# This is v1 - probably want something lighter
# in the future
class CatchupResponse(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["catchup"] = "catchup"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rehydrated: RehydrationResult
    surface: str | None = None
    surface_time: datetime | None = None


class SubmitMessageBody(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    text: str
    metadata: dict = Field(default_factory=dict)
    inject: bool = False


class ConduitBody(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: str
    source: str = "unknown"
    content: str
    metadata: Any = Field(default_factory=dict)


class SubmitMessageResponse(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["submit_message"] = "submit_message"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started: bool
    queued: bool

    @model_validator(mode="after")
    def _validate_bools(self) -> Self:
        if self.started == self.queued:
            raise ValueError(
                "`SubmitMessageResponse` must have either `started` or `queued` set to True, and not both."
            )
        return self


class InterruptResponse(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["interrupt"] = "interrupt"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    interrupted: bool


class ConduitResponse(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["conduit"] = "conduit"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered: bool
    action: ConduitAction
    source: str


class HealthResponse(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["health"] = "health"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["ok"] = "ok"
    model: str
    state: RuntimeState
    turns: int
    gauge_input_tokens: int | None
    surface: str | None = None
    surface_time: datetime | None = None


class AwarenessResponse(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["awareness"] = "awareness"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot: AwarenessSnapshot


class CatchupLiteResponse(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["catchup_lite"] = "catchup_lite"
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    health: HealthResponse
    awareness: AwarenessResponse
