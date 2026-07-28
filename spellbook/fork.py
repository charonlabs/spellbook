"""Fork protocol and runner for derived session work.

This module defines the typed protocol for fork-scoped work and the runtime
service that executes those forks.

`ForkRunner` is the reusable substrate for child-session work that
derives from a parent session's runtime/config while remaining isolated from the
parent's canonical transcript state.

Important invariants:

- fork input/output should be explicit typed protocol, not ad hoc kwargs/results
- a fork sees a projection of parent state, not direct mutation access to the
  parent's transcript
- the parent session decides how fork results are integrated
- `ForkRunner` owns child-session orchestration; feature-specific subsystems
  should not each reinvent session spawning
- specializations stay explicit without turning the fork layer into
  feature-specific glue

If you add new fork types, keep config/result typing, child-session wiring, and
result decoding explicit and coherent together.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Coroutine, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from spellbook.config import SpellbookConfig
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRBlock,
    IRInboundMessage,
    IRLoopResult,
    IRRefusalBlock,
    IRRecord,
    IRSemanticBlockRange,
    IRSemanticBlockSummary,
    IRSessionRecord,
    IRUserTextBlock,
    StopReason,
)
from spellbook.profiles import QUANTUM, SessionProfile
from spellbook.session_lifecycle import SessionContext, SessionLifecycle

if TYPE_CHECKING:
    from spellbook.debug_visibility import DebugNoticeLevel, DebugEmitter
    from spellbook.recorder import Recorder

    from .session_manager import SessionBuilder

DEFAULT_DETECTOR_MODEL: str | None = None
# Detector default is None = inherit parent session's model/provider.
# The grouping pass should run in the same model family as the live mind unless
# the caller makes an explicit override.
# Summarizer default is None = inherit parent session's model.
# The mind that lived the experience compresses the experience.
DEFAULT_SUMMARIZER_MODEL: str | None = None


class BlockDetectorConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["block_detector"] = "block_detector"
    prev_semantic_blocks: list[IRSemanticBlockRange]
    full_context_blocks: list[IRBlock]
    full_context_start_id: int | None = None
    context_block_buffer: list[IRBlock]
    context_block_start_id: int
    semantic_block_buffer: list[IRSemanticBlockRange]
    inbound_block: IRUserTextBlock
    detector_model: str | None = DEFAULT_DETECTOR_MODEL


class BlockSummarizerConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["block_summarizer"] = "block_summarizer"
    inbound_block: IRUserTextBlock
    # None = inherit parent session model. Your memory should sound like you.
    summarizer_model: str | None = DEFAULT_SUMMARIZER_MODEL


class QuantumForkConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["quantum"] = "quantum"
    instruction: IRUserTextBlock
    profile: SessionProfile = QUANTUM
    submit_tool: bool = True
    fork_label: str | None = None


ForkConfig = Annotated[
    BlockDetectorConfig | BlockSummarizerConfig | QuantumForkConfig,
    Field(discriminator="type"),
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


class QuantumForkResult(BaseModel, frozen=True):
    model_config = ConfigDict(extra="forbid")
    type: Literal["quantum"] = "quantum"
    final_text: str
    submitted: JsonValue | None
    fork_transcript_path: str
    rounds: int
    stop_reason: StopReason


ForkResult = Annotated[
    BlockDetectorResult | BlockSummarizerResult | QuantumForkResult,
    Field(discriminator="type"),
]


@dataclass(frozen=True, slots=True)
class PreparedFork:
    coro: Coroutine[Any, Any, ForkResult]
    fork_id: str


class ForkSession(Protocol):
    async def run(self) -> None: ...

    async def shutdown(self) -> None: ...


class ForkSessionLifecycle(SessionLifecycle):
    def __init__(self) -> None:
        self.turn_end_event = asyncio.Event()
        self.last_result: IRLoopResult | None = None
        self.last_turn_id: str | None = None

    async def on_turn_ended(
        self, ctx: SessionContext, result: IRLoopResult, turn_id: str
    ) -> None:
        self.last_result = result
        self.last_turn_id = turn_id
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
        debug_emitter: DebugEmitter | None = None,
    ):
        self._parent_config = parent_config
        self._parent_path = parent_transcript_path
        self._recorder = recorder
        self._build_session = session_builder
        self._debug = debug_emitter

    async def run_fork(self, fork_config: ForkConfig) -> PreparedFork:
        match fork_config:
            case BlockDetectorConfig():
                return await self._run_block_detector(fork_config)
            case BlockSummarizerConfig():
                return await self._run_block_summarizer(fork_config)
            case QuantumForkConfig():
                return await self._run_quantum(fork_config)
            case _:
                raise NotImplementedError(
                    f"Forks of type {fork_config.type} are not yet supported."
                )

    def integrate_result(self, fork_id: str, error_note: str | None = None) -> None:
        self._recorder.shutdown_fork(fork_id, error_note=error_note)

    async def _run_block_detector(
        self,
        fork_config: BlockDetectorConfig,
    ) -> PreparedFork:
        detector_model = (
            fork_config.detector_model
            if fork_config.detector_model is not None
            else self._parent_config.model
        )
        child_config = self._parent_config.model_copy(
            update={
                "session_type": "block_detector",
                "tool_categories": {"block_detection"},
                "model": detector_model,
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
        try:
            fork_session = await self._build_session(
                transcript_path=child_transcript_path,
                config=child_config,
                lifecycle=lifecycle,
                fork_config=fork_config,
                session_id=fork_id,
            )
        except Exception as exc:
            self._alert_fork_failure(
                fork_id=fork_id,
                fork_type="block_detector",
                event="build_failed",
                title="Block detector fork failed to build",
                error=exc,
            )
            self.integrate_result(fork_id)
            raise

        async def _run() -> BlockDetectorResult:
            child_task = asyncio.create_task(fork_session.run())
            try:
                initial_msg = IRInboundMessage(
                    blocks=[fork_config.inbound_block],
                    delivery="turn",
                )
                await fork_session.submit_message(initial_msg)
                await self._wait_for_turn_end_or_child_exit(
                    fork_id=fork_id,
                    fork_type="block_detector",
                    lifecycle=lifecycle,
                    child_task=child_task,
                )
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
                await self._shutdown_child_session(
                    fork_id=fork_id,
                    fork_type="block_detector",
                    fork_session=fork_session,
                    child_task=child_task,
                )

        return PreparedFork(coro=_run(), fork_id=fork_id)

    async def _run_quantum(
        self,
        fork_config: QuantumForkConfig,
    ) -> PreparedFork:
        fork_id = f"quantum_{_safe_fork_label(fork_config.fork_label)}_{uuid4().hex}"
        child_transcript_path = self._prepare_quantum_snapshot(
            fork_id=fork_id,
            fork_config=fork_config,
        )
        child_config = self._parent_config.model_copy(
            update={
                "session_type": fork_config.profile.name,
                "profile": fork_config.profile,
                "tool_categories": None,
            }
        )
        lifecycle = ForkSessionLifecycle()
        self._recorder.summon_fork(
            fork_id=fork_id,
            fork_type="quantum",
            child_transcript_path=str(child_transcript_path),
        )
        try:
            fork_session = await self._build_session(
                transcript_path=child_transcript_path,
                config=child_config,
                lifecycle=lifecycle,
                fork_config=fork_config,
                session_id=fork_id,
            )
        except Exception as exc:
            self._alert_fork_failure(
                fork_id=fork_id,
                fork_type="quantum",
                event="build_failed",
                title="Quantum fork failed to build",
                error=exc,
            )
            self._shutdown_failed_fork(fork_id, exc)
            raise

        async def _run() -> QuantumForkResult:
            child_task = asyncio.create_task(fork_session.run())
            try:
                try:
                    initial_msg = IRInboundMessage(
                        blocks=[fork_config.instruction],
                        delivery="turn",
                        source_metadata={
                            "source": "quantum_fork",
                            "fork_id": fork_id,
                            "fork_label": fork_config.fork_label,
                        },
                    )
                    await fork_session.submit_message(initial_msg)
                    await self._wait_for_turn_end_or_child_exit(
                        fork_id=fork_id,
                        fork_type="quantum",
                        lifecycle=lifecycle,
                        child_task=child_task,
                    )
                    loop_result = lifecycle.last_result
                    if loop_result is None:
                        raise RuntimeError(
                            f"Quantum fork {fork_id} ended without a loop result."
                        )
                    final_meta = await fork_session.get_tool_meta()
                    from spellbook.tools.common import QuantumForkToolMetadata

                    assert isinstance(final_meta, QuantumForkToolMetadata)
                    return QuantumForkResult(
                        final_text=_last_assistant_text(loop_result),
                        submitted=(
                            final_meta.submitted if final_meta.submit_called else None
                        ),
                        fork_transcript_path=str(child_transcript_path),
                        rounds=loop_result.rounds,
                        stop_reason=loop_result.stop_reason,
                    )
                finally:
                    await self._shutdown_child_session(
                        fork_id=fork_id,
                        fork_type="quantum",
                        fork_session=fork_session,
                        child_task=child_task,
                    )
            except asyncio.CancelledError as exc:
                self._alert_fork_failure(
                    fork_id=fork_id,
                    fork_type="quantum",
                    event="cancelled",
                    title="Quantum fork was cancelled",
                    error=exc,
                    level="warning",
                )
                self._shutdown_failed_fork(fork_id, exc)
                raise
            except Exception as exc:
                self._alert_fork_failure(
                    fork_id=fork_id,
                    fork_type="quantum",
                    event="run_failed",
                    title="Quantum fork failed",
                    error=exc,
                )
                self._shutdown_failed_fork(fork_id, exc)
                raise

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
        try:
            fork_session = await self._build_session(
                transcript_path=child_transcript_path,
                config=child_config,
                lifecycle=lifecycle,
                fork_config=fork_config,
                session_id=fork_id,
            )
        except Exception as exc:
            self._alert_fork_failure(
                fork_id=fork_id,
                fork_type="block_summarizer",
                event="build_failed",
                title="Block summarizer fork failed to build",
                error=exc,
            )
            self.integrate_result(fork_id)
            raise

        async def _run() -> BlockSummarizerResult:
            child_task = asyncio.create_task(fork_session.run())
            try:
                initial_msg = IRInboundMessage(
                    blocks=[fork_config.inbound_block],
                    delivery="turn",
                )
                # TODO: make this shutdown on the after_execute round boundary instead of the turn boundary
                await fork_session.submit_message(initial_msg)
                await self._wait_for_turn_end_or_child_exit(
                    fork_id=fork_id,
                    fork_type="block_summarizer",
                    lifecycle=lifecycle,
                    child_task=child_task,
                )
                final_meta = await fork_session.get_tool_meta()
                from spellbook.tools.common import BlockSummarizerToolMetadata

                assert isinstance(final_meta, BlockSummarizerToolMetadata)
                return BlockSummarizerResult(summary=final_meta.new_summary)
            finally:
                await self._shutdown_child_session(
                    fork_id=fork_id,
                    fork_type="block_summarizer",
                    fork_session=fork_session,
                    child_task=child_task,
                )

        return PreparedFork(coro=_run(), fork_id=fork_id)

    async def _wait_for_turn_end_or_child_exit(
        self,
        *,
        fork_id: str,
        fork_type: str,
        lifecycle: ForkSessionLifecycle,
        child_task: asyncio.Task[None],
    ) -> None:
        """Wait for a fork turn to finish, but don't hide child-session failure.

        A fork child session is expected to keep running until the fork turn ends
        and the parent explicitly shuts it down. If the child session task exits
        first, the prepared fork should fail so the Nursery can harvest the error
        instead of leaving a background job waiting forever.
        """
        if child_task.done() and not lifecycle.turn_end_event.is_set():
            await self._raise_child_exit_before_turn_end(
                fork_id=fork_id,
                fork_type=fork_type,
                child_task=child_task,
            )
        if lifecycle.turn_end_event.is_set():
            return

        turn_end_task = asyncio.create_task(lifecycle.turn_end_event.wait())
        try:
            done, _ = await asyncio.wait(
                {turn_end_task, child_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if child_task in done and not lifecycle.turn_end_event.is_set():
                await self._raise_child_exit_before_turn_end(
                    fork_id=fork_id,
                    fork_type=fork_type,
                    child_task=child_task,
                )
        finally:
            if not turn_end_task.done():
                turn_end_task.cancel()
                with suppress(asyncio.CancelledError):
                    await turn_end_task

    def _prepare_quantum_snapshot(
        self, *, fork_id: str, fork_config: QuantumForkConfig
    ) -> Path:
        self._assert_parent_quiescent()
        fork_dir = self._parent_path.parent / "forks" / fork_id
        fork_dir.mkdir(parents=True, exist_ok=False)
        child_transcript_path = fork_dir / "transcript.jsonl"
        shutil.copy2(self._parent_path, child_transcript_path)
        self._rewrite_quantum_snapshot_records(
            transcript_path=child_transcript_path,
            fork_id=fork_id,
            fork_config=fork_config,
        )
        for dirname in ("blobs", "tool_outputs"):
            (fork_dir / dirname).symlink_to(
                Path("..") / ".." / dirname,
                target_is_directory=True,
            )
        return child_transcript_path

    def _assert_parent_quiescent(self) -> None:
        from spellbook.rehydrator import Rehydrator

        rehydrated = Rehydrator(self._parent_path).run()
        if rehydrated.is_unfinished_turn:
            raise RuntimeError(
                "Cannot spawn quantum fork from a transcript with an in-progress turn."
            )

    def _rewrite_quantum_snapshot_records(
        self,
        *,
        transcript_path: Path,
        fork_id: str,
        fork_config: QuantumForkConfig,
    ) -> None:
        from spellbook.tools.registry import ToolRegistry

        adapter = TypeAdapter(IRRecord)
        records: list[IRRecord] = []
        for raw_line in transcript_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            record = adapter.validate_json(line)
            update: dict[str, Any] = {"session_id": fork_id}
            if isinstance(record, IRSessionRecord):
                config = record.config.model_copy(
                    update={
                        "session_type": fork_config.profile.name,
                        "profile": fork_config.profile,
                        "tool_categories": None,
                    }
                )
                tool_registry = ToolRegistry.build(
                    None,
                    surface=fork_config.profile.tool_surface,
                    include_quantum_submit=fork_config.submit_tool,
                    body_url=config.body_url,
                    minecraft_url=config.minecraft_url,
                )
                update.update({"config": config, "tools": tool_registry.records})
            records.append(record.model_copy(update=update))
        transcript_path.write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
        )

    def _shutdown_failed_fork(self, fork_id: str, error: BaseException) -> None:
        self.integrate_result(
            fork_id,
            error_note=f"{type(error).__name__}: {error}",
        )

    async def _raise_child_exit_before_turn_end(
        self, *, fork_id: str, fork_type: str, child_task: asyncio.Task[None]
    ) -> None:
        try:
            await child_task
        except asyncio.CancelledError as exc:
            self._alert_fork_failure(
                fork_id=fork_id,
                fork_type=fork_type,
                event="child_cancelled_before_turn_end",
                title="Fork child session was cancelled before turn end",
                error=exc,
            )
            raise RuntimeError(
                f"Fork child session {fork_id} was cancelled before turn end."
            ) from exc
        except Exception as exc:
            self._alert_fork_failure(
                fork_id=fork_id,
                fork_type=fork_type,
                event="child_failed_before_turn_end",
                title="Fork child session failed before turn end",
                error=exc,
            )
            raise
        self._alert_fork_failure(
            fork_id=fork_id,
            fork_type=fork_type,
            event="child_exited_before_turn_end",
            title="Fork child session exited before turn end",
            error=None,
        )
        raise RuntimeError(f"Fork child session {fork_id} exited before turn end.")

    async def _shutdown_child_session(
        self,
        *,
        fork_id: str,
        fork_type: str,
        fork_session: ForkSession,
        child_task: asyncio.Task[None],
    ) -> None:
        try:
            await fork_session.shutdown()
        except Exception as exc:
            self._alert_fork_failure(
                fork_id=fork_id,
                fork_type=fork_type,
                event="shutdown_failed",
                title="Fork child session shutdown failed",
                error=exc,
            )
            raise
        was_done = child_task.done()
        if not child_task.done():
            await asyncio.sleep(0)
        if not child_task.done():
            child_task.cancel()
        try:
            await child_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if not was_done:
                self._alert_fork_failure(
                    fork_id=fork_id,
                    fork_type=fork_type,
                    event="child_shutdown_failed",
                    title="Fork child task failed during shutdown",
                    error=exc,
                    level="warning",
                )

    def _get_orientation(self, fc: ForkConfig) -> str:
        orientation_path = self.ORIENTATION_PATH / fc.type
        if orientation_path.is_file():
            return orientation_path.read_text()
        markdown_orientation_path = orientation_path.with_suffix(".md")
        if markdown_orientation_path.is_file():
            return markdown_orientation_path.read_text()
        return ""

    def _alert_fork_failure(
        self,
        *,
        fork_id: str,
        fork_type: str,
        event: str,
        title: str,
        error: BaseException | None,
        level: "DebugNoticeLevel" = "error",
    ) -> None:
        if self._debug is None:
            return
        self._debug.alert(
            subsystem="fork",
            event=event,
            title=title,
            content=_render_fork_failure(
                fork_id=fork_id,
                fork_type=fork_type,
                event=event,
                title=title,
                error=error,
            ),
            plaintext=_fork_failure_plaintext(
                fork_id=fork_id,
                event=event,
                title=title,
                error=error,
            ),
            metadata={
                "fork_id": fork_id,
                "fork_type": fork_type,
                "error_type": type(error).__name__ if error is not None else None,
                "error_message": str(error) if error is not None else None,
            },
            level=level,
        )


def _render_fork_failure(
    *,
    fork_id: str,
    fork_type: str,
    event: str,
    title: str,
    error: BaseException | None,
) -> str:
    error_text = "-" if error is None else f"{type(error).__name__}: {error}"
    return "\n".join(
        [
            f"# {title}",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Fork | `{fork_id}` |",
            f"| Type | `{fork_type}` |",
            f"| Event | `{event}` |",
            f"| Error | `{_escape_table(error_text)}` |",
        ]
    )


def _fork_failure_plaintext(
    *, fork_id: str, event: str, title: str, error: BaseException | None
) -> str:
    if error is None:
        return f"{title}: {fork_id} ({event})."
    return f"{title}: {fork_id} ({event}): {error}"


def _last_assistant_text(result: IRLoopResult) -> str:
    for generation in reversed(result.generations):
        for block in reversed(generation.blocks):
            if isinstance(block, IRAssistantTextBlock):
                return block.text
            if isinstance(block, IRRefusalBlock):
                return block.partial_text
    return ""


def _safe_fork_label(label: str | None) -> str:
    if label is None:
        return "fork"
    sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", label.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "fork"


def _escape_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
