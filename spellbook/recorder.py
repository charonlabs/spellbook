"""Persistance layer for a Spellbook entity."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence
from uuid import uuid4

from spellbook.round_lifecycle import RoundContext, RoundLifecycle

from .config import SessionType, SpellbookConfig
from .config_override import validate_override
from .image_blobs import persist_image_blobs_in_block
from .ir_types import (
    IRBlock,
    IRBlockDetectionRecord,
    IRBlockRecord,
    IRContextPlan,
    IRContextPlanProposalRecord,
    IRConfigOverrideRecord,
    IRExecution,
    IRFooter,
    IRFooterDrainRecord,
    IRFooterQueueRecord,
    IRForkShutdownRecord,
    IRForkSummonRecord,
    IRGeneration,
    IRRecord,
    IRRuntimeConfigRecord,
    IRSemanticBlock,
    IRSemanticBlockApplyModeRecord,
    IRSemanticBlockArtifact,
    IRSemanticBlockArtifactRecord,
    IRSemanticBlockMetricsRecord,
    IRSemanticBlockPin,
    IRSemanticBlockPinRecord,
    IRSemanticBlockRecord,
    IRSessionRecord,
    IRSkillCatalog,
    IRSkillCatalogDelta,
    IRSkillCatalogUpdateRecord,
    IRSystemResponseRecord,
    IRTokenRangeCount,
    IRToolResultTTLRecord,
    IRTurnEndRecord,
    IRTurnStartRecord,
    SemanticBlockApplyModeSource,
    SemanticBlockMode,
    StopReason,
    RuntimeConfigNamespace,
    RuntimeConfigSource,
    RuntimeConfigValue,
    ToolResultTTLSource,
    ToolResultTTLTrigger,
)
from .system_response import SystemResponse
from .tools.registry import ToolRegistry

if TYPE_CHECKING:
    from spellbook.fork import BlockDetectorResult

RecordTap = Callable[[IRRecord], None]


class RecordsPersistedError(RuntimeError):
    """A record observer failed after a complete batch reached the transcript."""


class Recorder:
    """Persists blocks and records to the transcript.

    Owns sequence numbering, only one writer at a time per session."""

    def __init__(
        self,
        config: SpellbookConfig,
        transcript_path: Path,
        session_id: str,
        tool_registry: ToolRegistry,
        record_tap: RecordTap | None = None,
    ):
        self._config = config
        self._path = transcript_path
        self._session_id = session_id
        self._registry = tool_registry
        self._record_tap = record_tap
        self._turn: int = 0
        self._seq: int = 0
        self._curr_turn_id: str = ""

    @property
    def current_turn_idx(self) -> int:
        return self._turn

    @property
    def transcript_path(self) -> Path:
        return self._path

    def _write_record(self, record: IRRecord) -> None:
        with open(self._path, "a") as f:
            f.write(record.model_dump_json() + "\n")
        if self._record_tap is not None:
            self._record_tap(record)

    def write_block(self, block: IRBlock) -> None:
        if block.turn_id is None:
            block = block.model_copy(update={"turn_id": self._curr_turn_id})
        event_id = str(uuid4())
        block = block.model_copy(update={"event_id": event_id})
        block = persist_image_blobs_in_block(block)
        record = IRBlockRecord(
            session_id=self._session_id, turn=self._turn, seq=self._seq, event=block
        )
        self._write_record(record)
        self._seq += 1

    def write_tool_result_ttl(
        self,
        *,
        call_id: str,
        replace_content: str,
        ttl: int,
        trigger: ToolResultTTLTrigger,
        delivered_turn: int | None = None,
        source: ToolResultTTLSource = "auto",
        output_ref: str | None = None,
    ) -> IRToolResultTTLRecord:
        ttl_record = IRToolResultTTLRecord(
            session_id=self._session_id,
            call_id=call_id,
            replace_content=replace_content,
            ttl=ttl,
            trigger=trigger,
            delivered_turn=delivered_turn if delivered_turn is not None else self._turn,
            source=source,
            output_ref=output_ref,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(ttl_record)
        return ttl_record

    def write_runtime_config(
        self,
        *,
        namespace: RuntimeConfigNamespace,
        updates: dict[str, RuntimeConfigValue],
        effective: dict[str, RuntimeConfigValue],
        source: RuntimeConfigSource = "model",
    ) -> IRRuntimeConfigRecord:
        record = IRRuntimeConfigRecord(
            session_id=self._session_id,
            namespace=namespace,
            updates=updates,
            effective=effective,
            source=source,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(record)
        return record

    def write_config_override(
        self,
        *,
        updates: dict[str, object],
        source: str,
        actor: str,
        note: str | None = None,
    ) -> IRConfigOverrideRecord:
        """Append a validated override that takes effect on the next resume."""

        normalized_updates = validate_override(updates)
        record = IRConfigOverrideRecord(
            session_id=self._session_id,
            source=source,
            actor=actor,
            updates=normalized_updates,
            note=note,
        )
        self._write_record(record)
        return record

    def write_system_response(self, response: SystemResponse) -> IRSystemResponseRecord:
        record = IRSystemResponseRecord(
            session_id=self._session_id,
            command=response.command,
            content=response.content,
            plaintext=response.plaintext,
            metadata=response.metadata,
        )
        self._write_record(record)
        return record

    def set_state(self, turn_id: str, turn: int, seq: int) -> None:
        self._curr_turn_id = turn_id
        self._turn = turn
        self._seq = seq

    def write_session_record(self, skill_catalog: IRSkillCatalog) -> None:
        """Creates transcript and writes initial session record."""
        if self._path.exists():
            raise ValueError(
                "You just tried to overwrite a transcript that already exists... don't do that!"
            )
        session_record = IRSessionRecord(
            config=self._config,
            session_id=self._session_id,
            tools=self._registry.records,
            skill_catalog=skill_catalog,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._write_record(session_record)

    def start_turn(self, turn_id: str, new_blocks: Sequence[IRBlock]) -> None:
        self._curr_turn_id = turn_id
        self._turn += 1
        self._seq = 0
        start_record = IRTurnStartRecord(
            turn_id=turn_id, session_id=self._session_id, turn=self._turn
        )
        self._write_record(start_record)
        for block in new_blocks:
            self.write_block(block)

    def end_turn(self, stop_reason: StopReason | None = None) -> None:
        end_record = IRTurnEndRecord(
            session_id=self._session_id,
            stop_reason=stop_reason,
            turn_id=self._curr_turn_id,
            turn=self._turn,
        )
        self._write_record(end_record)

    def queue_footer(self, footer: IRFooter) -> None:
        footer_queue_record = IRFooterQueueRecord(
            session_id=self._session_id,
            footer=footer,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(footer_queue_record)

    def drain_footers(self, footers: list[IRFooter]) -> None:
        footer_drain_record = IRFooterDrainRecord(
            session_id=self._session_id,
            footers=footers,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(footer_drain_record)

    def detect_blocks(self, result: BlockDetectorResult) -> None:
        detect_blocks_record = IRBlockDetectionRecord(
            session_id=self._session_id,
            completed=result.completed,
            still_buffered=result.still_buffered,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(detect_blocks_record)

    def write_semantic_block(self, block: IRSemanticBlock) -> None:
        semantic_block_record = IRSemanticBlockRecord(
            session_id=self._session_id,
            id=block.id,
            idx=block.idx,
            range_id=block.range.id,
            toks=block.toks,
            full_toks=block.full_toks,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(semantic_block_record)

    def write_block_artifact(
        self, artifact: IRSemanticBlockArtifact, block_id: str
    ) -> None:
        block_artifact_record = IRSemanticBlockArtifactRecord(
            session_id=self._session_id,
            block_id=block_id,
            artifact=artifact,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(block_artifact_record)

    def write_block_metrics(self, toks: IRTokenRangeCount, block_id: str) -> None:
        block_metrics_record = IRSemanticBlockMetricsRecord(
            session_id=self._session_id,
            block_id=block_id,
            toks=toks,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(block_metrics_record)

    def apply_block_pin(self, pin: IRSemanticBlockPin, block_id: str) -> None:
        block_pin_record = IRSemanticBlockPinRecord(
            session_id=self._session_id,
            block_id=block_id,
            pin=pin,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(block_pin_record)

    def apply_semantic_block_mode(
        self,
        mode: SemanticBlockMode,
        block_id: str,
        source: SemanticBlockApplyModeSource,
    ) -> None:
        self.apply_semantic_block_modes(((mode, block_id),), source=source)

    def apply_semantic_block_modes(
        self,
        applications: Sequence[tuple[SemanticBlockMode, str]],
        *,
        source: SemanticBlockApplyModeSource,
    ) -> None:
        """Append one logical batch of existing apply-mode records.

        Sleep uses this for a parent/child narrative pair. All records are
        constructed and serialized before the transcript is opened, then
        appended with one write so no Python-level failure can land only one
        half of the pair.
        """

        records = tuple(
            IRSemanticBlockApplyModeRecord(
                session_id=self._session_id,
                block_id=block_id,
                mode=mode,
                source=source,
                turn=self._turn,
                turn_id=self._curr_turn_id,
            )
            for mode, block_id in applications
        )
        if not records:
            return
        payload = "".join(
            record.model_dump_json() + "\n" for record in records
        ).encode()
        with open(self._path, "ab") as f:
            f.seek(0, 2)
            original_size = f.tell()
            try:
                written = f.write(payload)
                if written != len(payload):
                    raise OSError(
                        "Apply-mode record batch was only partially appended."
                    )
                f.flush()
            except BaseException:
                # The batch is one logical transition. Removing only its
                # incomplete tail is corruption recovery, not history editing.
                f.truncate(original_size)
                f.flush()
                raise
        if self._record_tap is not None:
            try:
                for record in records:
                    self._record_tap(record)
            except Exception as exc:
                raise RecordsPersistedError(
                    "Apply-mode records landed, but a record observer failed."
                ) from exc

    def propose_plan(self, proposal: IRContextPlan) -> None:
        propose_plan_record = IRContextPlanProposalRecord(
            session_id=self._session_id,
            plan=proposal,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(propose_plan_record)

    def summon_fork(
        self, fork_id: str, fork_type: SessionType, child_transcript_path: str
    ) -> None:
        summon_fork_record = IRForkSummonRecord(
            session_id=self._session_id,
            fork_id=fork_id,
            fork_type=fork_type,
            child_transcript_path=child_transcript_path,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(summon_fork_record)

    def shutdown_fork(self, fork_id: str, error_note: str | None = None) -> None:
        shutdown_fork_record = IRForkShutdownRecord(
            session_id=self._session_id,
            fork_id=fork_id,
            error_note=error_note,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(shutdown_fork_record)

    def update_skill_catalog(self, delta: IRSkillCatalogDelta) -> None:
        skill_update_record = IRSkillCatalogUpdateRecord(
            session_id=self._session_id,
            delta=delta,
            turn=self._turn,
            turn_id=self._curr_turn_id,
        )
        self._write_record(skill_update_record)


class RecordingRoundLifecycle(RoundLifecycle):
    def __init__(self, recorder: Recorder):
        self._recorder = recorder

    async def after_generate(
        self,
        ctx: RoundContext,
        generation: IRGeneration,
    ) -> None:
        for block in generation.blocks:
            self._recorder.write_block(block)

    async def after_execute(
        self,
        ctx: RoundContext,
        execution: IRExecution,
    ) -> None:
        for block in execution.blocks:
            self._recorder.write_block(block)
