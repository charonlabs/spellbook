"""Rehydration from transcript back into memory."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from .config import SpellbookConfig
from .image_blobs import hydrate_image_blobs_in_block
from .ir_types import (
    IRBlock,
    IRBlockDetectionRecord,
    IRBlockRecord,
    IRContextPlan,
    IRContextPlanProposalRecord,
    IRFooter,
    IRFooterDrainRecord,
    IRFooterQueueRecord,
    IRRecord,
    IRSemanticBlock,
    IRSemanticBlockApplyModeRecord,
    IRSemanticBlockArtifactRecord,
    IRSemanticBlockMetricsRecord,
    IRSemanticBlockPinRecord,
    IRSemanticBlockRange,
    IRSemanticBlockRecord,
    IRSessionRecord,
    IRSkillCatalog,
    IRSkillCatalogUpdateRecord,
    IRToolRecord,
    IRToolResultTTLRecord,
    IRTurnEndRecord,
    IRTurnStartRecord,
)
from .tools.common import tool_to_record
from .tools.registry import KNOWN_TOOL_REGISTRY

MISSING_SKILL_CATALOG_ERROR = (
    "This transcript was made before Skill support was added. "
    'Please populate the session record with an empty skill catalog: {"skills": {}}.'
)

adapter = TypeAdapter(IRRecord)


class RehydrationResult(BaseModel, frozen=True):
    """The result of a rehydration.
    When rehydrating from a transcript with a clean end of turn as the last event,
    `current_turn_id`, `last_seq` and `in_progress_turn` will both be `None`."""

    session_id: str
    records: list[IRRecord]
    blocks: list[IRBlock]
    config: SpellbookConfig
    tools: list[IRToolRecord]
    last_completed_turn: int
    pending_footers: dict[str, IRFooter]
    completed_semantic_block_ranges: list[IRSemanticBlockRange]
    buffered_semantic_block_ranges: list[IRSemanticBlockRange]
    semantic_blocks: list[IRSemanticBlock]
    plan_proposal: IRContextPlan | None
    skill_catalog: IRSkillCatalog
    tool_result_ttls: list[IRToolResultTTLRecord] = Field(default_factory=list)
    is_unfinished_turn: bool = False
    current_turn_id: str | None = None
    last_seq: int | None = None
    in_progress_turn: int | None = None


class Rehydrator:
    def __init__(self, transcript_path: Path):
        self._path = transcript_path

    def _validate_session_record_shape(self) -> None:
        with open(self._path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    first_record = json.loads(line)
                except json.JSONDecodeError:
                    return
                if not isinstance(first_record, dict):
                    return
                if (
                    first_record.get("ir") == "session"
                    and "skill_catalog" not in first_record
                ):
                    raise ValueError(MISSING_SKILL_CATALOG_ERROR)
                return

    def _read_records(self) -> list[IRRecord]:
        records: list[IRRecord] = []
        with open(self._path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                # Fails loudly. Malformed transcripts should never silently sneak through.
                records.append(adapter.validate_json(line))
        return records

    def run(self) -> RehydrationResult:
        self._validate_session_record_shape()
        records = self._read_records()
        blocks: list[IRBlock] = []
        config: SpellbookConfig | None = None
        tools: list[IRToolRecord] = []
        pending_footers: dict[str, IRFooter] = {}
        completed_semantic_block_ranges: list[IRSemanticBlockRange] = []
        buffered_semantic_block_ranges: list[IRSemanticBlockRange] = []
        semantic_blocks: list[IRSemanticBlock] = []
        plan_proposal: IRContextPlan | None = None
        skill_catalog: IRSkillCatalog | None = None
        tool_result_ttls: list[IRToolResultTTLRecord] = []
        current_turn: int = 0
        in_progress_turn: int | None = None
        current_turn_id: str | None = None
        current_seq: int | None = None
        session_id: str | None = None
        last_completed_turn = 0
        is_unfinished_turn = False
        for record in records:
            match record:
                case IRSessionRecord():
                    config = record.config
                    session_id = record.session_id
                    skill_catalog = record.skill_catalog
                    # TODO: figure out how to deal with tool refreshing. In the old version,
                    # we just always used the most recent versions of the tools. Here, should
                    # we do something like what exists now (and we make like a cli tool to refresh
                    # like we also have in the old one) or "... print warning while silenty using newer
                    # tools"? For now I'm erroring loudly
                    for tool in record.tools:
                        registered = KNOWN_TOOL_REGISTRY.get(tool.name)
                        if registered is None:
                            raise ValueError(
                                f"Tool `{tool.name}` present in frame but not registry."
                            )
                        registered_record = tool_to_record(registered)
                        # Category mismatches realistically are fine - those are config concerns
                        normalized_registered_record = registered_record.model_copy(
                            update={"category": "normalized"}
                        )
                        normalized_tool = tool.model_copy(
                            update={"category": "normalized"}
                        )
                        if not normalized_registered_record == normalized_tool:
                            raise ValueError(
                                f"Mismatch between frame and registry for tool `{tool.name}`"
                            )
                        tools.append(tool)
                case IRTurnStartRecord():
                    current_turn = record.turn
                    current_turn_id = record.turn_id
                case IRTurnEndRecord():
                    current_turn_id = None
                    current_seq = None
                    last_completed_turn = current_turn
                case IRBlockRecord():
                    current_seq = record.seq
                    blocks.append(
                        hydrate_image_blobs_in_block(record.event, self._path)
                    )
                case IRToolResultTTLRecord():
                    tool_result_ttls.append(record)
                case IRFooterQueueRecord():
                    pending_footers[record.footer.key] = record.footer
                case IRFooterDrainRecord():
                    drained_ids = {f.id for f in record.footers}
                    pending_footers = {
                        key: f
                        for key, f in pending_footers.items()
                        if f.id not in drained_ids
                    }
                case IRBlockDetectionRecord():
                    completed_semantic_block_ranges.extend(record.completed)
                    buffered_semantic_block_ranges = record.still_buffered
                case IRSemanticBlockRecord():
                    r = next(
                        b
                        for b in completed_semantic_block_ranges
                        if b.id == record.range_id
                    )
                    semantic_blocks.append(
                        IRSemanticBlock(
                            id=record.id,
                            idx=record.idx,
                            time=record.time,
                            range=r,
                            title=r.title,
                            toks=record.toks,
                            full_toks=record.full_toks,
                        )
                    )
                case IRSemanticBlockArtifactRecord():
                    block = next(b for b in semantic_blocks if b.id == record.block_id)
                    new_block = block.model_copy(
                        update={
                            "artifacts": block.artifacts + [record.artifact],
                            "available_modes": block.available_modes
                            + (
                                [record.artifact.mode]
                                if record.artifact.mode not in block.available_modes
                                else []
                            ),
                        }
                    )
                    semantic_blocks[block.idx] = new_block
                case IRSemanticBlockMetricsRecord():
                    block = next(b for b in semantic_blocks if b.id == record.block_id)
                    new_block = block.model_copy(update={"full_toks": record.toks})
                    if new_block.mode == "full":
                        new_block = new_block.model_copy(update={"toks": record.toks})
                    semantic_blocks[block.idx] = new_block
                case IRSemanticBlockPinRecord():
                    block = next(b for b in semantic_blocks if b.id == record.block_id)
                    if record.pin.kind == "facet":
                        new_block = block.model_copy(
                            update={"facet_pins": block.facet_pins + [record.pin]}
                        )
                    else:
                        new_block = block.model_copy(update={"pin": record.pin})
                    semantic_blocks[block.idx] = new_block
                    plan_proposal = None  # invalidates plan proposal
                case IRSemanticBlockApplyModeRecord():
                    block = next(b for b in semantic_blocks if b.id == record.block_id)
                    toks = (
                        block.full_toks
                        if record.mode == "full"
                        else next(
                            a for a in block.artifacts if a.mode == record.mode
                        ).toks
                    )
                    new_block = block.model_copy(
                        update={"mode": record.mode, "toks": toks}
                    )
                    semantic_blocks[block.idx] = new_block
                    plan_proposal = None  # invalidates plan proposal
                case IRContextPlanProposalRecord():
                    plan_proposal = record.plan
                case IRSkillCatalogUpdateRecord():
                    assert skill_catalog is not None
                    skills = dict(skill_catalog.skills)
                    for rm in record.delta.removed:
                        del skills[rm]
                    for new in record.delta.added.values():
                        skills[new.name] = new
                    for upd in record.delta.updated.values():
                        skills[upd.name] = upd
                    skill_catalog = IRSkillCatalog(skills=skills)
        if session_id is None or config is None:
            raise ValueError(
                "Session record broken. Either `session_id` or `config` is None"
            )
        if skill_catalog is None:
            raise ValueError(MISSING_SKILL_CATALOG_ERROR)
        if current_turn_id is not None:  # unfinished
            is_unfinished_turn = True
            in_progress_turn = current_turn
            if current_turn != 0:
                last_completed_turn = current_turn - 1
        return RehydrationResult(
            session_id=session_id,
            records=records,
            blocks=blocks,
            config=config,
            tools=tools,
            last_completed_turn=last_completed_turn,
            pending_footers=pending_footers,
            completed_semantic_block_ranges=completed_semantic_block_ranges,
            buffered_semantic_block_ranges=buffered_semantic_block_ranges,
            semantic_blocks=semantic_blocks,
            plan_proposal=plan_proposal,
            skill_catalog=skill_catalog,
            tool_result_ttls=tool_result_ttls,
            is_unfinished_turn=is_unfinished_turn,
            current_turn_id=current_turn_id,
            last_seq=current_seq,
            in_progress_turn=in_progress_turn,
        )
