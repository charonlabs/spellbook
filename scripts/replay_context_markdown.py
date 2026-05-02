"""Render replay-context transcript events as Markdown."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from spellbook.ir_types import (
    IRBlockDetectionRecord,
    IRBlockRecord,
    IRForkShutdownRecord,
    IRForkSummonRecord,
    IRRecord,
    IRSemanticBlock,
    IRSemanticBlockArtifactRecord,
    IRSemanticBlockFacet,
    IRSemanticBlockRange,
    IRSemanticBlockRecord,
    IRSemanticBlockSummary,
)
from spellbook.rehydrator import Rehydrator


@dataclass
class ReplayMarkdownReport:
    transcript_path: Path
    session_id: str
    model: str
    detect_interval: int
    source_blocks: int
    body_lines: list[str]
    fork_summons: int
    fork_shutdowns: int
    detection_records: int
    completed_blocks: int
    summaries: int

    def render(self) -> str:
        lines = [
            "# Replay Context Report",
            "",
            f"- Transcript: `{self.transcript_path}`",
            f"- Session: `{self.session_id}`",
            f"- Model: `{self.model}`",
            f"- Detect interval: `{self.detect_interval}`",
            f"- Context blocks: `{self.source_blocks}`",
            f"- Forks summoned: `{self.fork_summons}`",
            f"- Forks shut down: `{self.fork_shutdowns}`",
            f"- Detection records: `{self.detection_records}`",
            f"- Completed semantic blocks: `{self.completed_blocks}`",
            f"- Summary artifacts: `{self.summaries}`",
            "",
            "## Event Stream",
            "",
        ]
        if self.body_lines:
            lines.extend(self.body_lines)
        else:
            lines.append(
                "_No replay detector, semantic block, summary, or fork events found._"
            )
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class ReplayMarkdownRenderer:
    transcript_path: Path
    semantic_ranges_by_id: dict[str, IRSemanticBlockRange] = field(default_factory=dict)
    semantic_blocks_by_id: dict[str, IRSemanticBlock] = field(default_factory=dict)
    last_proposed_signatures: set[tuple[str, int, int]] = field(default_factory=set)
    body_lines: list[str] = field(default_factory=list)
    block_count: int = 0
    fork_summons: int = 0
    fork_shutdowns: int = 0
    detection_records: int = 0
    completed_blocks: int = 0
    summaries: int = 0

    def render(self) -> ReplayMarkdownReport:
        rehydrated = Rehydrator(self.transcript_path).run()
        for record in rehydrated.records:
            self._observe(record)

        return ReplayMarkdownReport(
            transcript_path=self.transcript_path,
            session_id=rehydrated.session_id,
            model=rehydrated.config.model,
            detect_interval=rehydrated.config.hom_config.detect_interval,
            source_blocks=len(rehydrated.blocks),
            body_lines=self.body_lines,
            fork_summons=self.fork_summons,
            fork_shutdowns=self.fork_shutdowns,
            detection_records=self.detection_records,
            completed_blocks=self.completed_blocks,
            summaries=self.summaries,
        )

    def _observe(self, record: IRRecord) -> None:
        match record:
            case IRBlockRecord():
                self.block_count += 1
            case IRForkSummonRecord():
                self._handle_fork_summon(record)
            case IRForkShutdownRecord():
                self._handle_fork_shutdown(record)
            case IRBlockDetectionRecord():
                self._handle_block_detection(record)
            case IRSemanticBlockRecord():
                self._handle_semantic_block(record)
            case IRSemanticBlockArtifactRecord():
                self._handle_semantic_block_artifact(record)

    def _handle_fork_summon(self, record: IRForkSummonRecord) -> None:
        self.fork_summons += 1
        self.body_lines.extend(
            [
                f"### Fork Summoned: `{record.fork_id}`",
                "",
                f"- Type: `{record.fork_type}`",
                f"- Turn: `{record.turn_id}` (`{record.turn}`)",
                f"- Child transcript: `{record.child_transcript_path}`",
                "",
            ]
        )

    def _handle_fork_shutdown(self, record: IRForkShutdownRecord) -> None:
        self.fork_shutdowns += 1
        self.body_lines.extend(
            [
                f"### Fork Shutdown: `{record.fork_id}`",
                "",
                f"- Turn: `{record.turn_id}` (`{record.turn}`)",
                "",
            ]
        )

    def _handle_block_detection(self, record: IRBlockDetectionRecord) -> None:
        self.detection_records += 1
        for block in [*record.completed, *record.still_buffered]:
            self.semantic_ranges_by_id[block.id] = block

        proposed = record.still_buffered
        proposed_signatures = _semantic_range_signatures(proposed)
        if proposed and proposed_signatures != self.last_proposed_signatures:
            self.body_lines.extend(
                [
                    f"### Proposed Blocks After Detection {self.detection_records}",
                    "",
                    f"- Turn: `{record.turn_id}` (`{record.turn}`)",
                    f"- Context blocks observed so far: `{self.block_count}`",
                    "",
                    _render_range_table(proposed),
                    "",
                ]
            )
        self.last_proposed_signatures = proposed_signatures

    def _handle_semantic_block(self, record: IRSemanticBlockRecord) -> None:
        semantic_range = self.semantic_ranges_by_id.get(record.range_id)
        if semantic_range is None:
            self.body_lines.extend(
                [
                    f"### Completed Block {record.idx}: Unknown Range",
                    "",
                    f"- Block id: `{record.id}`",
                    f"- Missing range id: `{record.range_id}`",
                    f"- Turn: `{record.turn_id}` (`{record.turn}`)",
                    "",
                ]
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
        self.completed_blocks += 1
        self.body_lines.extend(
            [
                f"### Completed Block {block.idx}: {_escape_heading(block.title)}",
                "",
                f"- Block id: `{block.id}`",
                f"- Range: `{_format_range(block.range.start_block, block.range.end_block)}`",
                f"- Tokens: `{_format_tokens(block.full_toks.tokens if block.full_toks else None)}`",
                f"- Turn: `{record.turn_id}` (`{record.turn}`)",
                "",
            ]
        )

    def _handle_semantic_block_artifact(
        self, record: IRSemanticBlockArtifactRecord
    ) -> None:
        artifact = record.artifact
        if artifact.type != "summary":
            return

        self.summaries += 1
        block = self.semantic_blocks_by_id.get(record.block_id)
        block_label = (
            f'Block {block.idx}: "{block.title}"'
            if block
            else f"Block `{record.block_id}`"
        )
        self.body_lines.extend(
            [
                f"### Summary for {block_label}",
                "",
                f"- Summary id: `{artifact.id}`",
                f"- Turn: `{record.turn_id}` (`{record.turn}`)",
                "",
                _render_summary_markdown(artifact),
                "",
            ]
        )


def render_replay_markdown(transcript_path: Path) -> str:
    path = transcript_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {path}")
    return ReplayMarkdownRenderer(path).render().render()


def _semantic_range_signatures(
    ranges: list[IRSemanticBlockRange],
) -> set[tuple[str, int, int]]:
    return {(block.title, block.start_block, block.end_block) for block in ranges}


def _render_range_table(blocks: list[IRSemanticBlockRange]) -> str:
    lines = [
        "| Title | Range | Status |",
        "| --- | ---: | --- |",
    ]
    for block in blocks:
        status = "completed" if block.completed else "proposed"
        lines.append(
            "| "
            f"{_table_cell(block.title)} | "
            f"`{_format_range(block.start_block, block.end_block)}` | "
            f"{status} |"
        )
    return "\n".join(lines)


def _render_summary_markdown(summary: IRSemanticBlockSummary) -> str:
    parts = [
        f"#### {_escape_heading(summary.headline)}",
        "",
        summary.text,
    ]
    if summary.facets:
        parts.extend(["", "##### Facets", "", _render_facets_table(summary.facets)])
    if summary.open_thread:
        parts.extend(["", f"**Open thread:** {summary.open_thread}"])
    return "\n".join(parts)


def _render_facets_table(facets: list[IRSemanticBlockFacet]) -> str:
    lines = [
        "| Title | Range | Description | Resources |",
        "| --- | ---: | --- | --- |",
    ]
    for facet in facets:
        resources = "; ".join(facet.resources) if facet.resources else ""
        lines.append(
            "| "
            f"{_table_cell(facet.title)} | "
            f"`{_format_range(facet.start_block, facet.end_block)}` | "
            f"{_table_cell(facet.description)} | "
            f"{_table_cell(resources)} |"
        )
    return "\n".join(lines)


def _format_range(start: int, end: int) -> str:
    return f"{start}-{end}"


def _format_tokens(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    return str(tokens)


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>").strip()


def _escape_heading(value: str) -> str:
    return value.replace("\n", " ").strip()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render replay detector and summarizer records as Markdown."
    )
    parser.add_argument(
        "--transcript",
        required=True,
        type=Path,
        help="Path to a replay output core transcript.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the Markdown report. Defaults to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    markdown = render_replay_markdown(args.transcript)
    if args.output is None:
        print(markdown, end="")
        return

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
