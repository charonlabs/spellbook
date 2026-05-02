"""Fork protocol and runner for derived session work.

This module defines the typed protocol for fork-scoped work and the runtime
service that executes those forks.

Right now the only supported fork type is block detection, but the design intent
is broader: `ForkRunner` is the reusable substrate for child-session work that
derives from a parent session's runtime/config while remaining isolated from the
parent's canonical transcript state.

Important invariants:

- fork input/output should be explicit typed protocol, not ad hoc kwargs/results
- a fork sees a projection of parent state, not direct mutation access to the
  parent's transcript
- the parent session decides how fork results are integrated
- `ForkRunner` owns child-session orchestration; feature-specific subsystems
  should not each reinvent session spawning
- block detector is the first specialization, not the permanent shape of the
  fork layer

If you add new fork types, keep config/result typing, child-session wiring, and
result decoding explicit and coherent together.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Coroutine, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRBlock,
    IRInboundMessage,
    IRLoopResult,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRUserTextBlock,
)
from spellbook.session_lifecycle import SessionContext, SessionLifecycle

if TYPE_CHECKING:
    from spellbook.recorder import Recorder

    from .session_manager import SessionBuilder

DEFAULT_DETECTOR_MODEL = "claude-opus-4-6"
# Summarizer default is None = inherit parent session's model.
# The mind that lived the experience compresses the experience.
DEFAULT_SUMMARIZER_MODEL: str | None = None


class BlockDetectorConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["block_detector"] = "block_detector"
    prev_semantic_blocks: list[IRSemanticBlockRange]
    full_context_blocks: list[IRBlock]
    context_block_buffer: list[IRBlock]
    context_block_start_id: (
        int  # The number of the first context block in the full context slice
    )
    semantic_block_buffer: list[IRSemanticBlockRange]
    inbound_block: IRUserTextBlock
    detector_model: str = DEFAULT_DETECTOR_MODEL


class BlockSummarizerConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["block_summarizer"] = "block_summarizer"
    inbound_block: IRUserTextBlock
    # None = inherit parent session model. Your memory should sound like you.
    summarizer_model: str | None = DEFAULT_SUMMARIZER_MODEL


ForkConfig = Annotated[
    BlockDetectorConfig | BlockSummarizerConfig, Field(discriminator="type")
]


class BlockDetectorResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["block_detector"] = "block_detector"
    completed: list[IRSemanticBlockRange]
    still_buffered: list[IRSemanticBlockRange]


class BlockSummarizerResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["block_summarizer"] = "block_summarizer"
    summary: IRSemanticBlockSummary


ForkResult = Annotated[
    BlockDetectorResult | BlockSummarizerResult, Field(discriminator="type")
]


@dataclass(frozen=True, slots=True)
class PreparedFork:
    coro: Coroutine[Any, Any, ForkResult]
    fork_id: str


class ForkSessionLifecycle(SessionLifecycle):
    def __init__(self) -> None:
        self.turn_end_event = asyncio.Event()

    async def on_turn_ended(
        self, ctx: SessionContext, result: IRLoopResult, turn_id: str
    ) -> None:
        self.turn_end_event.set()


class ForkRunner:
    """Spawns child sessions to do fork-scoped work.

    Used by subsystems that need model-driven multi-round work isolated
    from the main session. Block detection is the first consumer; Consult,
    Dreamer, and others will follow the same pattern."""

    ORIENTATION_PATH = Path(__file__).parent / "orientation" / "forks"

    def __init__(
        self,
        *,
        parent_config: SpellbookConfig,
        parent_transcript_path: Path,
        recorder: Recorder,
        session_builder: "SessionBuilder",
    ):
        self._parent_config = parent_config
        self._parent_path = parent_transcript_path
        self._recorder = recorder
        self._build_session = session_builder

    async def run_fork(self, fork_config: ForkConfig) -> PreparedFork:
        match fork_config:
            case BlockDetectorConfig():
                return await self._run_block_detector(fork_config)
            case BlockSummarizerConfig():
                return await self._run_block_summarizer(fork_config)
            case _:
                raise NotImplementedError(
                    f"Forks of type {fork_config.type} are not yet supported."
                )

    def integrate_result(self, fork_id: str) -> None:
        self._recorder.shutdown_fork(fork_id)

    async def _run_block_detector(
        self,
        fork_config: BlockDetectorConfig,
    ) -> PreparedFork:
        child_config = self._parent_config.model_copy(
            update={
                "session_type": "block_detector",
                "tool_categories": {"block_detection"},
                "model": fork_config.detector_model,
                "system_prompt": self._get_orientation(fork_config),
            }
        )
        fork_id = f"detector_{uuid4().hex}"
        child_transcript_path = self._parent_path.parent / "forks" / f"{fork_id}.jsonl"
        lifecycle = ForkSessionLifecycle()
        self._recorder.summon_fork(
            fork_id=fork_id,
            fork_type="block_detector",
            child_transcript_path=str(child_transcript_path),
        )
        fork_session = await self._build_session(
            transcript_path=child_transcript_path,
            config=child_config,
            lifecycle=lifecycle,
            fork_config=fork_config,
            session_id=fork_id,
        )

        async def _run() -> BlockDetectorResult:
            asyncio.create_task(fork_session.run())
            try:
                initial_msg = IRInboundMessage(
                    blocks=[fork_config.inbound_block],
                    delivery="turn",
                )
                await fork_session.submit_message(initial_msg)
                await lifecycle.turn_end_event.wait()
                final_meta = await fork_session.get_tool_meta()
                from spellbook.tools.common import BlockDetectorToolMetadata

                assert isinstance(final_meta, BlockDetectorToolMetadata)
                return BlockDetectorResult(
                    completed=[
                        b for b in final_meta.semantic_block_buffer if b.completed
                    ],
                    still_buffered=[
                        b for b in final_meta.semantic_block_buffer if not b.completed
                    ],
                )
            finally:
                await fork_session.shutdown()

        return PreparedFork(coro=_run(), fork_id=fork_id)

    async def _run_block_summarizer(
        self,
        fork_config: BlockSummarizerConfig,
    ) -> PreparedFork:
        # Resolve summarizer model: None means inherit parent's model.
        # Your memory should sound like you.
        summarizer_model = (
            fork_config.summarizer_model
            if fork_config.summarizer_model is not None
            else self._parent_config.model
        )
        child_config = self._parent_config.model_copy(
            update={
                "session_type": "block_summarizer",
                "tool_categories": {"block_summarization"},
                "model": summarizer_model,
                "system_prompt": self._get_orientation(fork_config),
            }
        )
        fork_id = f"summarizer_{uuid4().hex}"
        child_transcript_path = self._parent_path.parent / "forks" / f"{fork_id}.jsonl"
        lifecycle = ForkSessionLifecycle()
        self._recorder.summon_fork(
            fork_id=fork_id,
            fork_type="block_summarizer",
            child_transcript_path=str(child_transcript_path),
        )
        fork_session = await self._build_session(
            transcript_path=child_transcript_path,
            config=child_config,
            lifecycle=lifecycle,
            fork_config=fork_config,
            session_id=fork_id,
        )

        async def _run() -> BlockSummarizerResult:
            asyncio.create_task(fork_session.run())
            try:
                initial_msg = IRInboundMessage(
                    blocks=[fork_config.inbound_block],
                    delivery="turn",
                )
                # TODO: make this shutdown on the after_execute round boundary instead of the turn boundary
                await fork_session.submit_message(initial_msg)
                await lifecycle.turn_end_event.wait()
                final_meta = await fork_session.get_tool_meta()
                from spellbook.tools.common import BlockSummarizerToolMetadata

                assert isinstance(final_meta, BlockSummarizerToolMetadata)
                return BlockSummarizerResult(summary=final_meta.new_summary)
            finally:
                await fork_session.shutdown()

        return PreparedFork(coro=_run(), fork_id=fork_id)

    def _get_orientation(self, fc: ForkConfig) -> str:
        orientation_path = self.ORIENTATION_PATH / fc.type
        if orientation_path.is_file():
            return orientation_path.read_text()
        markdown_orientation_path = orientation_path.with_suffix(".md")
        if markdown_orientation_path.is_file():
            return markdown_orientation_path.read_text()
        return ""
