"""Replay a core transcript through BlockManager detection and summarization."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import rich
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from spellbook.backends.anthropic import AnthropicBackend
from spellbook.backends.model_backend import ModelBackend
from spellbook.config import SpellbookConfig
from spellbook.footer import FooterController
from spellbook.fork import ForkRunner
from spellbook.homunculus.block_manager import BlockManager
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import (
    IRBlock,
    IRBlockDetectionRecord,
    IRBlockRecord,
    IRForkShutdownRecord,
    IRForkSummonRecord,
    IRRecord,
    IRSemanticBlock,
    IRSemanticBlockArtifactRecord,
    IRSemanticBlockRange,
    IRSemanticBlockRecord,
    IRSemanticBlockSummary,
    IRSkillCatalog,
    IRTurnEndRecord,
    IRTurnStartRecord,
)
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.rehydrator import RehydrationResult, Rehydrator
from spellbook.session_manager import SessionManager
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.registry import ToolRegistry


@dataclass(frozen=True)
class ReplayTickReport:
    tick: int
    start_block: int
    end_block: int
    semantic_blocks: int
    summaries: int
    created_forks: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "start_block": self.start_block,
            "end_block": self.end_block,
            "semantic_blocks": self.semantic_blocks,
            "summaries": self.summaries,
            "created_forks": self.created_forks,
        }


@dataclass
class ReplayReport:
    source_path: Path
    output_path: Path
    source_session_id: str
    replay_session_id: str
    source_blocks: int
    replayed_blocks: int
    interval: int
    starting_semantic_blocks: int = 0
    starting_summaries: int = 0
    final_semantic_blocks: int | None = None
    final_summaries: int | None = None
    finalization_passes: int = 0
    resume_backup_path: Path | None = None
    ticks: list[ReplayTickReport] = field(default_factory=list)

    @property
    def semantic_blocks(self) -> int:
        if self.final_semantic_blocks is not None:
            return self.final_semantic_blocks
        if not self.ticks:
            return self.starting_semantic_blocks
        return self.ticks[-1].semantic_blocks

    @property
    def summaries(self) -> int:
        if self.final_summaries is not None:
            return self.final_summaries
        if not self.ticks:
            return self.starting_summaries
        return self.ticks[-1].summaries

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "output_path": str(self.output_path),
            "source_session_id": self.source_session_id,
            "replay_session_id": self.replay_session_id,
            "source_blocks": self.source_blocks,
            "replayed_blocks": self.replayed_blocks,
            "interval": self.interval,
            "starting_semantic_blocks": self.starting_semantic_blocks,
            "starting_summaries": self.starting_summaries,
            "final_semantic_blocks": self.final_semantic_blocks,
            "final_summaries": self.final_summaries,
            "finalization_passes": self.finalization_passes,
            "resume_backup_path": str(self.resume_backup_path)
            if self.resume_backup_path is not None
            else None,
            "semantic_blocks": self.semantic_blocks,
            "summaries": self.summaries,
            "ticks": [tick.to_dict() for tick in self.ticks],
        }


@dataclass
class _ReplayState:
    recorder: Recorder
    block_manager: BlockManager
    output_path: Path
    interval: int
    report: ReplayReport
    previous_forks: set[Path] = field(init=False)
    tick: int = 0
    pending_blocks: list[IRBlock] = field(default_factory=list)
    pending_start_block: int | None = None

    def __post_init__(self) -> None:
        self.previous_forks = _fork_files(self.output_path)

    async def write_block(self, block: IRBlock) -> None:
        replayed_block = _as_replayed_block(block)
        self.recorder.write_block(replayed_block)
        if self.pending_start_block is None:
            self.pending_start_block = self.report.replayed_blocks
        self.pending_blocks.append(replayed_block)
        self.report.replayed_blocks += 1
        if len(self.pending_blocks) >= self.interval:
            await self.flush()

    async def flush(self) -> None:
        if not self.pending_blocks:
            return
        start_block = self.pending_start_block
        assert start_block is not None
        blocks = list(self.pending_blocks)
        await self.block_manager.append_context_blocks(blocks)
        # Replay is an offline artifact builder, not the live low-latency loop.
        # Wait at tick boundaries so detector forks cannot overlap against stale
        # semantic-buffer state.
        await self.block_manager.check_nursery(wait_for_all=True)

        current_forks = _fork_files(self.output_path)
        created_forks = sorted(
            str(path) for path in current_forks - self.previous_forks
        )
        self.previous_forks = current_forks
        self.tick += 1
        tick_report = ReplayTickReport(
            tick=self.tick,
            start_block=start_block,
            end_block=start_block + len(blocks) - 1,
            semantic_blocks=len(self.block_manager.semantic_blocks),
            summaries=_summary_count(self.block_manager),
            created_forks=created_forks,
        )
        self.report.ticks.append(tick_report)
        rich.print(
            f"[cyan]tick {tick_report.tick}[/cyan] "
            f"blocks {tick_report.start_block}-{tick_report.end_block}: "
            f"{tick_report.semantic_blocks} semantic block(s), "
            f"{tick_report.summaries} summary artifact(s), "
            f"{len(created_forks)} fork(s)"
        )

        self.pending_blocks.clear()
        self.pending_start_block = None


@dataclass
class _ReplayRecordPrinter:
    semantic_ranges_by_id: dict[str, IRSemanticBlockRange] = field(default_factory=dict)
    semantic_blocks_by_id: dict[str, IRSemanticBlock] = field(default_factory=dict)
    printed_completed_block_ids: set[str] = field(default_factory=set)
    printed_summary_ids: set[str] = field(default_factory=set)
    last_proposed_signatures: set[tuple[str, int, int]] = field(default_factory=set)

    def __call__(self, record: IRRecord) -> None:
        self._observe(record, emit=True)

    def prime(self, records: list[IRRecord]) -> None:
        for record in records:
            self._observe(record, emit=False)

    def _observe(self, record: IRRecord, *, emit: bool) -> None:
        match record:
            case IRBlockDetectionRecord():
                self._handle_block_detection(record, emit=emit)
            case IRSemanticBlockRecord():
                self._handle_semantic_block(record, emit=emit)
            case IRSemanticBlockArtifactRecord():
                self._handle_semantic_block_artifact(record, emit=emit)
            case IRForkSummonRecord():
                if emit:
                    rich.print(
                        "[dim]"
                        f"fork summoned: {record.fork_type} {record.fork_id} -> "
                        f"{record.child_transcript_path}"
                        "[/dim]"
                    )
            case IRForkShutdownRecord():
                if emit:
                    rich.print(f"[dim]fork shutdown: {record.fork_id}[/dim]")

    def _handle_block_detection(
        self, record: IRBlockDetectionRecord, *, emit: bool
    ) -> None:
        for block in [*record.completed, *record.still_buffered]:
            self.semantic_ranges_by_id[block.id] = block

        proposed = record.still_buffered
        proposed_signatures = _semantic_range_signatures(proposed)
        if emit and proposed and proposed_signatures != self.last_proposed_signatures:
            rich.print(_render_proposed_blocks(proposed))
        self.last_proposed_signatures = proposed_signatures

    def _handle_semantic_block(
        self, record: IRSemanticBlockRecord, *, emit: bool
    ) -> None:
        semantic_range = self.semantic_ranges_by_id.get(record.range_id)
        if semantic_range is None:
            if emit:
                rich.print(
                    "[yellow]"
                    f"Semantic block {record.id} referenced unknown range "
                    f"{record.range_id}."
                    "[/yellow]"
                )
            return

        block = IRSemanticBlock(
            id=record.id,
            idx=record.idx,
            time=record.time,
            title=semantic_range.title,
            range=semantic_range,
            toks=record.toks,
            full_toks=record.full_toks,
        )
        self.semantic_blocks_by_id[block.id] = block
        if emit and block.id not in self.printed_completed_block_ids:
            rich.print(_render_completed_blocks([block]))
        self.printed_completed_block_ids.add(block.id)

    def _handle_semantic_block_artifact(
        self, record: IRSemanticBlockArtifactRecord, *, emit: bool
    ) -> None:
        artifact = record.artifact
        if artifact.type != "summary" or artifact.id in self.printed_summary_ids:
            return

        block = self.semantic_blocks_by_id.get(record.block_id)
        if emit:
            if block is None:
                rich.print(_render_summary_for_unknown_block(record.block_id, artifact))
            else:
                rich.print(_render_summary(block, artifact))
        self.printed_summary_ids.add(artifact.id)


@dataclass(frozen=True)
class _ReplayBuild:
    config: SpellbookConfig
    replay_session_id: str
    replay_interval: int
    start_block: int
    rehydrated_output: RehydrationResult | None = None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a core transcript through block detection and summarization."
    )
    parser.add_argument(
        "--transcript",
        required=True,
        type=Path,
        help="Path to a core transcript.jsonl.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the derived replay transcript.jsonl.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        help=(
            "How many source blocks to append per replay tick. Defaults to the "
            "source transcript's homunculus detect_interval."
        ),
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        help="Stop after replaying at most this many source blocks.",
    )
    parser.add_argument(
        "--session-id",
        help="Override the replay session id. Defaults to '<source-session-id>_replay'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the replay transcript and clear its forks directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing replay transcript after validating it is a source prefix.",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "After replaying all source blocks, run one EOF detector pass to "
            "flush stable buffered blocks into completed semantic blocks."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optionally write a JSON replay report.",
    )
    return parser.parse_args(argv)


async def replay_transcript(
    *,
    transcript_path: Path,
    output_path: Path,
    interval: int | None = None,
    max_blocks: int | None = None,
    session_id: str | None = None,
    force: bool = False,
    resume: bool = False,
    finalize: bool = False,
) -> ReplayReport:
    if interval is not None and interval <= 0:
        raise ValueError("Replay interval must be greater than zero.")
    if max_blocks is not None and max_blocks < 0:
        raise ValueError("max_blocks must be greater than or equal to zero.")
    transcript_path = transcript_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")
    if transcript_path == output_path:
        raise ValueError("Replay output must not overwrite the source transcript.")

    _prepare_output(output_path, force=force, resume=resume)
    source = Rehydrator(transcript_path).run()

    replay_build = _build_replay_config(
        source=source,
        output_path=output_path,
        interval=interval,
        session_id=session_id,
        max_blocks=max_blocks,
        resume=resume,
    )
    resume_backup_path = _backup_resume_output(output_path) if resume else None
    tool_registry = ToolRegistry.build(
        replay_build.config.tool_categories,
        surface=replay_build.config.session_type,
    )
    record_printer = _ReplayRecordPrinter()
    if replay_build.rehydrated_output is not None:
        record_printer.prime(replay_build.rehydrated_output.records)
    recorder = Recorder(
        config=replay_build.config,
        transcript_path=output_path,
        session_id=replay_build.replay_session_id,
        tool_registry=tool_registry,
        record_tap=record_printer,
    )
    if replay_build.rehydrated_output is None:
        recorder.write_session_record(skill_catalog=IRSkillCatalog())
    else:
        _restore_recorder_state(recorder, replay_build.rehydrated_output)
    block_manager = _build_block_manager(
        config=replay_build.config,
        transcript_path=output_path,
        recorder=recorder,
        tool_registry=tool_registry,
    )
    if replay_build.rehydrated_output is not None:
        block_manager.context_blocks = list(replay_build.rehydrated_output.blocks)
        block_manager.next_block_id = len(block_manager.context_blocks)
        block_manager.rehydrate(replay_build.rehydrated_output)
        await block_manager.generate_next_summary()
        await block_manager.check_nursery(wait_for_all=True)

    report = ReplayReport(
        source_path=transcript_path,
        output_path=output_path,
        source_session_id=source.session_id,
        replay_session_id=replay_build.replay_session_id,
        source_blocks=len(source.blocks),
        replayed_blocks=replay_build.start_block,
        interval=replay_build.replay_interval,
        starting_semantic_blocks=len(block_manager.semantic_blocks),
        starting_summaries=_summary_count(block_manager),
        resume_backup_path=resume_backup_path,
    )

    resume_backup_line = (
        f"Resume backup: {resume_backup_path}\n" if resume_backup_path else ""
    )
    rich.print(
        f"[bold]{'Resuming' if resume else 'Replaying'} core transcript[/bold]\n"
        f"Source: {transcript_path}\n"
        f"Output: {output_path}\n"
        f"{resume_backup_line}"
        f"Blocks: {report.replayed_blocks} / {len(source.blocks)}\n"
        f"Interval: {replay_build.replay_interval}"
    )

    replay_state = _ReplayState(
        recorder=recorder,
        block_manager=block_manager,
        output_path=output_path,
        interval=replay_build.replay_interval,
        report=report,
    )
    await _replay_source_records(
        records=source.records,
        state=replay_state,
        max_blocks=max_blocks,
        start_block=replay_build.start_block,
        resume_turn_id=(
            replay_build.rehydrated_output.current_turn_id
            if replay_build.rehydrated_output is not None
            and replay_build.rehydrated_output.is_unfinished_turn
            else None
        ),
    )
    await block_manager.check_nursery(wait_for_all=True)
    if finalize:
        report.finalization_passes = await _finalize_replay_context(
            block_manager=block_manager,
        )
    report.final_semantic_blocks = len(block_manager.semantic_blocks)
    report.final_summaries = _summary_count(block_manager)

    Rehydrator(output_path).run()
    rich.print(
        "\n[bold]Replay complete[/bold]\n"
        f"Semantic blocks: {report.semantic_blocks}\n"
        f"Summaries:        {report.summaries}\n"
        f"Transcript:       {output_path}"
    )
    return report


def _prepare_output(output_path: Path, *, force: bool, resume: bool = False) -> None:
    if resume:
        if force:
            raise ValueError("Cannot use --resume and --force together.")
        if not output_path.exists():
            raise FileNotFoundError(f"Replay transcript not found: {output_path}")
        return
    if output_path.exists():
        if not force:
            raise FileExistsError(f"Replay transcript already exists: {output_path}")
        output_path.unlink()
    forks_dir = output_path.parent / "forks"
    if force and forks_dir.exists():
        shutil.rmtree(forks_dir)


def _backup_resume_output(output_path: Path) -> Path:
    backup_path = _resume_backup_path(output_path)
    shutil.copy2(output_path, backup_path)
    return backup_path


def _resume_backup_path(output_path: Path) -> Path:
    candidate = output_path.with_suffix(output_path.suffix + ".resume.bak")
    if not candidate.exists():
        return candidate
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_path.with_suffix(output_path.suffix + f".resume.bak.{timestamp}")


def _build_replay_config(
    *,
    source: RehydrationResult,
    output_path: Path,
    interval: int | None,
    session_id: str | None,
    max_blocks: int | None,
    resume: bool,
) -> _ReplayBuild:
    if resume:
        output = Rehydrator(output_path).run()
        _validate_resume_prefix(source, output)
        if session_id is not None and session_id != output.session_id:
            raise ValueError(
                "Resume session id must match the existing replay transcript. "
                f"Got {session_id}, expected {output.session_id}."
            )
        replay_interval = output.config.hom_config.detect_interval
        if interval is not None and interval != replay_interval:
            raise ValueError(
                "Resume interval must match the existing replay transcript. "
                f"Got {interval}, expected {replay_interval}."
            )
        if max_blocks is not None and max_blocks < len(output.blocks):
            raise ValueError(
                "max_blocks cannot be less than the number of blocks already replayed. "
                f"Got {max_blocks}, existing replay has {len(output.blocks)}."
            )
        return _ReplayBuild(
            config=output.config,
            replay_session_id=output.session_id,
            replay_interval=replay_interval,
            start_block=len(output.blocks),
            rehydrated_output=output,
        )

    replay_session_id = session_id or f"{source.session_id}_replay"
    replay_interval = interval or source.config.hom_config.detect_interval
    config = source.config.model_copy(
        update={
            "session_type": "main",
            "tool_categories": None,
            "hom_config": source.config.hom_config.model_copy(
                update={"detect_interval": replay_interval}
            ),
        }
    )
    return _ReplayBuild(
        config=config,
        replay_session_id=replay_session_id,
        replay_interval=replay_interval,
        start_block=0,
    )


def _validate_resume_prefix(
    source: RehydrationResult,
    output: RehydrationResult,
) -> None:
    if len(output.blocks) > len(source.blocks):
        raise ValueError(
            "Existing replay transcript has more blocks than the source transcript. "
            f"Output has {len(output.blocks)}, source has {len(source.blocks)}."
        )

    for idx, output_block in enumerate(output.blocks):
        source_block = source.blocks[idx]
        if _as_replayed_block(output_block) != _as_replayed_block(source_block):
            raise ValueError(
                "Existing replay transcript is not a prefix of the source transcript. "
                f"First mismatch at block {idx}."
            )


def _restore_recorder_state(
    recorder: Recorder,
    rehydrated: RehydrationResult,
) -> None:
    recorder.set_state(
        turn_id=rehydrated.current_turn_id or "",
        turn=rehydrated.last_completed_turn
        if rehydrated.in_progress_turn is None
        else rehydrated.in_progress_turn,
        seq=(rehydrated.last_seq + 1) if rehydrated.last_seq is not None else 0,
    )


async def _replay_source_records(
    *,
    records: list[IRRecord],
    state: _ReplayState,
    max_blocks: int | None,
    start_block: int = 0,
    resume_turn_id: str | None = None,
) -> None:
    if max_blocks == 0:
        return
    if max_blocks is not None and start_block >= max_blocks:
        return

    source_block_idx = 0
    pending_turn_start: IRTurnStartRecord | None = None
    skipped_block_turn_id: str | None = None
    skipping = start_block > 0 or resume_turn_id is not None
    stop_after_turn_end = False
    for idx, record in enumerate(records):
        if stop_after_turn_end:
            if isinstance(record, IRTurnEndRecord):
                await state.flush()
                state.recorder.end_turn(record.stop_reason)
                return
            continue

        if skipping:
            match record:
                case IRTurnStartRecord():
                    pending_turn_start = record
                    continue
                case IRBlockRecord():
                    if source_block_idx < start_block:
                        skipped_block_turn_id = record.event.turn_id
                        source_block_idx += 1
                        continue
                    skipping = False
                    if resume_turn_id is not None:
                        block_turn_id = record.event.turn_id
                        if block_turn_id != resume_turn_id:
                            raise ValueError(
                                "Existing replay transcript ended inside a different "
                                "turn than the next source block. "
                                f"Replay turn={resume_turn_id}, source turn={block_turn_id}."
                            )
                    else:
                        if pending_turn_start is None:
                            raise ValueError(
                                "Could not find a source turn_start for the next "
                                f"replay block {source_block_idx}."
                            )
                        _start_source_turn(state.recorder, pending_turn_start)
                case IRTurnEndRecord():
                    if resume_turn_id is not None and record.turn_id == resume_turn_id:
                        skipping = False
                        await state.flush()
                        state.recorder.end_turn(record.stop_reason)
                    elif (
                        resume_turn_id is None
                        and source_block_idx >= start_block
                        and record.turn_id == skipped_block_turn_id
                    ):
                        skipping = False
                    elif (
                        resume_turn_id is None
                        and source_block_idx >= start_block
                        and record.turn_id != skipped_block_turn_id
                    ):
                        if (
                            pending_turn_start is None
                            or pending_turn_start.turn_id != record.turn_id
                        ):
                            raise ValueError(
                                "Could not find a source turn_start for empty turn "
                                f"{record.turn_id}."
                            )
                        skipping = False
                        _start_source_turn(state.recorder, pending_turn_start)
                        await state.flush()
                        state.recorder.end_turn(record.stop_reason)
                    continue
                case _:
                    continue

        match record:
            case IRTurnStartRecord():
                _start_source_turn(state.recorder, record)
            case IRBlockRecord():
                if (
                    max_blocks is not None
                    and state.report.replayed_blocks >= max_blocks
                ):
                    await state.flush()
                    return
                await state.write_block(record.event)
                source_block_idx += 1
                if (
                    max_blocks is not None
                    and state.report.replayed_blocks >= max_blocks
                ):
                    await state.flush()
                    if _next_source_boundary_is_turn_end(records, idx):
                        stop_after_turn_end = True
                    else:
                        return
            case IRTurnEndRecord():
                await state.flush()
                state.recorder.end_turn(record.stop_reason)

    await state.flush()


async def _finalize_replay_context(
    *,
    block_manager: BlockManager,
) -> int:
    started = await block_manager.force_detect(finalize=True)
    if not started:
        return 0
    rich.print("[cyan]finalize pass 1[/cyan]: detector fork started")
    await block_manager.check_nursery(wait_for_all=True)
    return 1


def _start_source_turn(recorder: Recorder, record: IRTurnStartRecord) -> None:
    recorder.set_state(
        turn_id="",
        turn=max(record.turn - 1, 0),
        seq=0,
    )
    recorder.start_turn(record.turn_id, [])


def _next_source_boundary_is_turn_end(records: list[IRRecord], idx: int) -> bool:
    for record in records[idx + 1 :]:
        if isinstance(record, IRTurnEndRecord):
            return True
        if isinstance(record, IRBlockRecord | IRTurnStartRecord):
            return False
    return False


def _build_block_manager(
    *,
    config: SpellbookConfig,
    transcript_path: Path,
    recorder: Recorder,
    tool_registry: ToolRegistry,
) -> BlockManager:
    backend = _build_backend(config)
    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=tool_registry,
    )
    token_counter = backend.build_token_counter(
        config=config,
        surface_builder=surface_builder,
    )
    fork_runner = ForkRunner(
        parent_config=config,
        parent_transcript_path=transcript_path,
        recorder=recorder,
        session_builder=SessionManager.build,
    )
    footer_controller = FooterController(
        inbound_queue=InboundMessageQueue(),
        recorder=recorder,
    )
    nursery = Nursery(config=config)
    return BlockManager(
        config=config.hom_config,
        fork_runner=fork_runner,
        footer_c=footer_controller,
        nursery=nursery,
        recorder=recorder,
        token_meter=TokenMeter(config=config.hom_config, tok_counter=token_counter),
    )


def _build_backend(config: SpellbookConfig) -> ModelBackend:
    match config.provider:
        case "anthropic":
            return AnthropicBackend()
        case _:
            raise NotImplementedError(
                f"Replay does not support provider `{config.provider}` yet."
            )


def _as_replayed_block(block: IRBlock) -> IRBlock:
    return block.model_copy(update={"event_id": None})


def _semantic_range_signatures(
    ranges: list[IRSemanticBlockRange],
) -> set[tuple[str, int, int]]:
    return {(block.title, block.start_block, block.end_block) for block in ranges}


def _render_proposed_blocks(blocks: list[IRSemanticBlockRange]) -> Table:
    table = Table(title="Proposed Blocks", show_lines=True)
    table.add_column("Title", style="cyan")
    table.add_column("Range", justify="right")
    for block in blocks:
        table.add_row(block.title, _format_range(block.start_block, block.end_block))
    return table


def _render_completed_blocks(blocks: list[IRSemanticBlock]) -> Table:
    table = Table(title="Completed Blocks", show_lines=True)
    table.add_column("Idx", justify="right")
    table.add_column("Title", style="green")
    table.add_column("Range", justify="right")
    table.add_column("Tokens", justify="right")
    for block in blocks:
        table.add_row(
            str(block.idx),
            block.title,
            _format_range(block.range.start_block, block.range.end_block),
            _format_tokens(block.full_toks.tokens if block.full_toks else None),
        )
    return table


def _render_summary(
    block: IRSemanticBlock,
    summary: IRSemanticBlockSummary,
) -> Panel:
    title = f'Summary: [Block {block.idx}] "{block.title}"'
    return _render_summary_panel(title, summary)


def _render_summary_for_unknown_block(
    block_id: str,
    summary: IRSemanticBlockSummary,
) -> Panel:
    return _render_summary_panel(f"Summary: {block_id}", summary)


def _render_summary_panel(title: str, summary: IRSemanticBlockSummary) -> Panel:
    return Panel(
        Markdown(_format_summary_markdown(summary)),
        title=title,
        border_style="magenta",
    )


def _format_summary_markdown(summary: IRSemanticBlockSummary) -> str:
    parts = [f"# {summary.headline}", "", summary.text]
    if summary.facets:
        parts.extend(["", "## Facets"])
        for facet in summary.facets:
            parts.append(
                f"- **{facet.title}** "
                f"({_format_range(facet.start_block, facet.end_block)})"
            )
            parts.append(f"  {facet.description}")
            if facet.resources:
                parts.append(f"  Resources: {'; '.join(facet.resources)}")
    if summary.open_thread:
        parts.extend(["", f"Open thread: {summary.open_thread}"])
    return "\n".join(parts)


def _format_range(start: int, end: int) -> str:
    return f"{start}-{end}"


def _format_tokens(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    return str(tokens)


def _fork_files(output_path: Path) -> set[Path]:
    forks_dir = output_path.parent / "forks"
    if not forks_dir.exists():
        return set()
    return set(forks_dir.glob("*.jsonl"))


def _summary_count(block_manager: BlockManager) -> int:
    return sum(
        1
        for block in block_manager.semantic_blocks
        if any(artifact.type == "summary" for artifact in block.artifacts)
    )


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = await replay_transcript(
        transcript_path=args.transcript,
        output_path=args.output,
        interval=args.interval,
        max_blocks=args.max_blocks,
        session_id=args.session_id,
        force=args.force,
        resume=args.resume,
        finalize=args.finalize,
    )
    if args.report_json is not None:
        report_path = args.report_json.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main(argv: list[str] | None = None) -> None:
    load_dotenv(Path.home() / ".chorus" / ".env")
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()
