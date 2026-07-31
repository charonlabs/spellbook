"""Rehydration from transcript back into memory."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .config import SpellbookConfig
from .config_override import (
    CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY,
    CONFIG_OVERRIDE_DISCLOSURE_PRIORITY,
    FROZEN_IDENTITY_FIELDS,
    ConfigOverrideValidationError,
    apply_override,
)
from .image_blobs import hydrate_image_blobs_in_block
from .ir_types import (
    IRBlock,
    IRBlockDetectionRecord,
    IRBlockRecord,
    IRConfigOverrideRecord,
    IRContextPlan,
    IRContextPlanProposalRecord,
    IRFooter,
    IRFooterDrainRecord,
    IRFooterQueueRecord,
    IRRecord,
    IRRuntimeConfigRecord,
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
    IRSystemResponseRecord,
    IRToolRecord,
    IRToolResultTTLRecord,
    IRTurnEndRecord,
    IRTurnStartRecord,
)
from .refusal import canonicalize_legacy_refusal
from .tools.common import Tool, tool_to_record
from .tools.registry import ALL_TOOLS, KNOWN_TOOL_REGISTRY, ToolRegistry

MISSING_SKILL_CATALOG_ERROR = (
    "This transcript was made before Skill support was added. "
    'Please populate the session record with an empty skill catalog: {"skills": {}}.'
)

adapter = TypeAdapter(IRRecord)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _OverrideEntry:
    position: int
    record: IRConfigOverrideRecord | None
    raw: dict[str, Any] | None = None
    parse_error: str | None = None


@dataclass(frozen=True)
class _ReadTranscript:
    records: list[IRRecord]
    override_entries: list[_OverrideEntry]
    latest_disclosure_position: int


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
    runtime_config_updates: list[IRRuntimeConfigRecord] = Field(default_factory=list)
    config_override_updates: list[IRConfigOverrideRecord] = Field(default_factory=list)
    config_override_disclosure_footer: IRFooter | None = None
    system_responses: list[IRSystemResponseRecord] = Field(default_factory=list)
    is_unfinished_turn: bool = False
    current_turn_id: str | None = None
    last_seq: int | None = None
    in_progress_turn: int | None = None


class Rehydrator:
    def __init__(
        self, transcript_path: Path, *, custom_tools: list[Tool] | None = None
    ):
        self._path = transcript_path
        self._custom_tools = custom_tools

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

    def _read_records(self) -> _ReadTranscript:
        records: list[IRRecord] = []
        override_entries: list[_OverrideEntry] = []
        latest_disclosure_position = -1
        with open(self._path, "r") as f:
            for position, raw_line in enumerate(f):
                line = raw_line.strip()
                if not line:
                    continue
                # Config overrides are the one poison-pill-safe exception to the
                # normal fail-loud transcript policy.
                try:
                    record = adapter.validate_json(line)
                except ValidationError as exc:
                    try:
                        raw_record = json.loads(line)
                    except json.JSONDecodeError:
                        raise exc
                    if not (
                        isinstance(raw_record, dict)
                        and raw_record.get("ir") == "config_override"
                    ):
                        raise
                    override_entries.append(
                        _OverrideEntry(
                            position=position,
                            record=None,
                            raw=raw_record,
                            parse_error=str(exc),
                        )
                    )
                    continue
                records.append(record)
                if isinstance(record, IRConfigOverrideRecord):
                    override_entries.append(
                        _OverrideEntry(position=position, record=record)
                    )
                elif (
                    isinstance(record, IRFooterQueueRecord)
                    and record.footer.key == CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY
                ):
                    latest_disclosure_position = position
        return _ReadTranscript(
            records=records,
            override_entries=override_entries,
            latest_disclosure_position=latest_disclosure_position,
        )

    def run(self) -> RehydrationResult:
        self._validate_session_record_shape()
        read_result = self._read_records()
        records = read_result.records
        session_record = next(
            (record for record in records if isinstance(record, IRSessionRecord)),
            None,
        )
        config: SpellbookConfig | None = None
        disclosure_lines: list[str] = []
        if session_record is not None:
            config, disclosure_lines = self._merge_config_overrides(
                session_record.config,
                read_result.override_entries,
                latest_disclosure_position=read_result.latest_disclosure_position,
            )
        refusal_turns = {
            record.turn
            for record in records
            if isinstance(record, IRTurnEndRecord) and record.stop_reason == "refusal"
        }
        blocks: list[IRBlock] = []
        tools: list[IRToolRecord] = []
        pending_footers: dict[str, IRFooter] = {}
        completed_semantic_block_ranges: list[IRSemanticBlockRange] = []
        buffered_semantic_block_ranges: list[IRSemanticBlockRange] = []
        semantic_blocks: list[IRSemanticBlock] = []
        plan_proposal: IRContextPlan | None = None
        skill_catalog: IRSkillCatalog | None = None
        tool_result_ttls: list[IRToolResultTTLRecord] = []
        runtime_config_updates: list[IRRuntimeConfigRecord] = []
        config_override_updates: list[IRConfigOverrideRecord] = []
        system_responses: list[IRSystemResponseRecord] = []
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
                    assert config is not None
                    session_id = record.session_id
                    skill_catalog = record.skill_catalog
                    # TODO: figure out how to deal with tool refreshing. In the old version,
                    # we just always used the most recent versions of the tools. Here, should
                    # we do something like what exists now (and we make like a cli tool to refresh
                    # like we also have in the old one) or "... print warning while silenty using newer
                    # tools"? For now I'm erroring loudly
                    known_registry = KNOWN_TOOL_REGISTRY
                    if config.profile.custom_surface:
                        if self._custom_tools is None:
                            raise ValueError(
                                "Tried to rehydrate a custom session without giving custom tools."
                            )
                        known_registry = ToolRegistry(
                            tools=ALL_TOOLS + self._custom_tools
                        )
                    for tool in record.tools:
                        registered = known_registry.get(tool.name)
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
                    block = hydrate_image_blobs_in_block(record.event, self._path)
                    if record.turn in refusal_turns:
                        block = canonicalize_legacy_refusal(block) or block
                    blocks.append(block)
                case IRToolResultTTLRecord():
                    tool_result_ttls.append(record)
                case IRRuntimeConfigRecord():
                    runtime_config_updates.append(record)
                case IRConfigOverrideRecord():
                    config_override_updates.append(record)
                case IRSystemResponseRecord():
                    system_responses.append(record)
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
                            a
                            for a in reversed(block.artifacts)
                            if a.mode == record.mode
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
        disclosure_footer: IRFooter | None = None
        if disclosure_lines:
            disclosure_text = "\n".join(disclosure_lines)
            existing_disclosure = pending_footers.get(
                CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY
            )
            if existing_disclosure is not None:
                disclosure_text = f"{existing_disclosure.text}\n{disclosure_text}"
            disclosure_footer = IRFooter(
                text=disclosure_text,
                id="footer_config_override_disclosure",
                type="notif",
                source="runtime",
                key=CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY,
                priority=CONFIG_OVERRIDE_DISCLOSURE_PRIORITY,
            )
            pending_footers[CONFIG_OVERRIDE_DISCLOSURE_FOOTER_KEY] = disclosure_footer
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
            runtime_config_updates=runtime_config_updates,
            config_override_updates=config_override_updates,
            config_override_disclosure_footer=disclosure_footer,
            system_responses=system_responses,
            is_unfinished_turn=is_unfinished_turn,
            current_turn_id=current_turn_id,
            last_seq=current_seq,
            in_progress_turn=in_progress_turn,
        )

    def _merge_config_overrides(
        self,
        base_config: SpellbookConfig,
        entries: list[_OverrideEntry],
        *,
        latest_disclosure_position: int,
    ) -> tuple[SpellbookConfig, list[str]]:
        config = base_config
        disclosure_lines: list[str] = []
        for entry in entries:
            record = entry.record
            if record is None:
                refusal = _malformed_override_refusal(entry)
                logger.critical(
                    "config_override.refused position=%s error=%s",
                    entry.position,
                    entry.parse_error,
                )
                if entry.position > latest_disclosure_position:
                    disclosure_lines.append(refusal)
                continue

            try:
                updated_config, effective_updates = apply_override(
                    config, record.updates
                )
            except ConfigOverrideValidationError as exc:
                refusal = _override_refusal(record, exc)
                logger.critical(
                    "config_override.refused position=%s source=%r actor=%r error=%s",
                    entry.position,
                    record.source,
                    record.actor,
                    exc,
                )
                if entry.position > latest_disclosure_position:
                    disclosure_lines.append(refusal)
                continue

            if entry.position > latest_disclosure_position:
                for field, new_value in effective_updates.items():
                    old_value = getattr(config, field)
                    disclosure_lines.append(
                        "since you last ran, your config changed: "
                        f"{field} {_format_override_value(old_value)}"
                        f"->{_format_override_value(new_value)} "
                        f"(source: {record.source}, by {record.actor})"
                    )
            config = updated_config
        return config, disclosure_lines


def _override_refusal(
    record: IRConfigOverrideRecord, error: ConfigOverrideValidationError
) -> str:
    frozen_fields = sorted(set(record.updates) & FROZEN_IDENTITY_FIELDS)
    if frozen_fields:
        reason = "attempted to change " + ", ".join(frozen_fields)
    else:
        reason = str(error)
    return (
        f"a config override record was refused: {reason} "
        f"(source: {record.source}, by {record.actor})"
    )


def _malformed_override_refusal(entry: _OverrideEntry) -> str:
    raw = entry.raw or {}
    updates = raw.get("updates")
    frozen_fields = (
        sorted(set(updates) & FROZEN_IDENTITY_FIELDS)
        if isinstance(updates, dict)
        else []
    )
    if frozen_fields:
        reason = "attempted to change " + ", ".join(frozen_fields)
    else:
        reason = "malformed record"
    attribution = _override_attribution(raw)
    return f"a config override record was refused: {reason}{attribution}"


def _override_attribution(raw: dict[str, Any]) -> str:
    source = raw.get("source")
    actor = raw.get("actor")
    if isinstance(source, str) and isinstance(actor, str) and source and actor:
        return f" (source: {source}, by {actor})"
    if isinstance(source, str) and source:
        return f" (source: {source})"
    return ""


def _format_override_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str | Path):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, set | frozenset):
        return json.dumps(sorted(value), sort_keys=True)
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)
