import argparse
import asyncio
from pathlib import Path

import rich
from dotenv import load_dotenv

from spellbook.config import SpellbookConfig
from spellbook.fork import BlockDetectorResult, ForkRunner
from spellbook.homunculus.block_detector import BlockDetector
from spellbook.ir_types import IRSemanticBlockRange
from spellbook.recorder import Recorder
from spellbook.rehydrator import Rehydrator
from spellbook.session_manager import SessionManager
from spellbook.tools.registry import ToolRegistry


def _format_block(block: IRSemanticBlockRange) -> str:
    status = "complete" if block.completed else "buffered"
    return f"- {block.title} [{block.start_block}-{block.end_block}] {status}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the core block detector hot path over an existing core transcript."
        )
    )
    parser.add_argument(
        "--transcript",
        required=True,
        type=Path,
        help="Path to a core transcript.jsonl written by spellbook.Recorder.",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        help=(
            "How many leading IR blocks to feed into the detector. Defaults to the "
            "transcript's homunculus detect_interval."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    transcript_path = args.transcript.expanduser().resolve()
    if not transcript_path.exists():
        raise SystemExit(f"Transcript not found: {transcript_path}")

    rehydrated = Rehydrator(transcript_path).run()
    config: SpellbookConfig = rehydrated.config
    detect_interval = config.hom_config.detect_interval
    max_blocks = args.max_blocks or detect_interval
    source_blocks = rehydrated.blocks[:max_blocks]
    if len(source_blocks) < detect_interval:
        rich.print(
            "[yellow]Not enough blocks to trigger detection.[/yellow]\n"
            f"Loaded {len(source_blocks)} block(s); "
            f"detector interval is {detect_interval}."
        )
        return

    tool_registry = ToolRegistry.build(
        config.tool_categories,
        surface=config.session_type,
    )
    recorder = Recorder(
        config=config,
        transcript_path=transcript_path,
        session_id=rehydrated.session_id,
        tool_registry=tool_registry,
    )
    fork_runner = ForkRunner(
        parent_config=config,
        parent_transcript_path=transcript_path,
        recorder=recorder,
        session_builder=SessionManager.build,
    )
    detector = BlockDetector(
        config=config.hom_config,
        fork_runner=fork_runner,
        recorder=recorder,
    )

    forks_dir = transcript_path.parent / "forks"
    before = set(forks_dir.glob("detector_*.jsonl")) if forks_dir.exists() else set()

    rich.print(
        "[bold]Running block detector[/bold]\n"
        f"Transcript: {transcript_path}\n"
        f"Blocks:     {len(source_blocks)}"
    )
    prepared = await detector.maybe_detect(source_blocks, first_block_id=0)
    assert prepared is not None
    completed = await prepared.coro
    assert isinstance(completed, BlockDetectorResult)

    after = set(forks_dir.glob("detector_*.jsonl")) if forks_dir.exists() else set()
    created = sorted(after - before)

    rich.print("\n[bold]Fork transcripts[/bold]")
    if created:
        for path in created:
            rich.print(f"- {path}")
    else:
        rich.print("- none created")

    rich.print("\n[bold]Completed This Run[/bold]")
    if completed:
        for block in completed.completed:
            rich.print(_format_block(block))
    else:
        rich.print("- none")

    rich.print("\n[bold]Detector State[/bold]")
    rich.print("[cyan]Completed blocks[/cyan]")
    if detector.completed_blocks:
        for block in detector.completed_blocks:
            rich.print(_format_block(block))
    else:
        rich.print("- none")

    rich.print("[cyan]Buffered blocks[/cyan]")
    if detector.buffered_blocks:
        for block in detector.buffered_blocks:
            rich.print(_format_block(block))
    else:
        rich.print("- none")


if __name__ == "__main__":
    load_dotenv(Path.home() / ".chorus" / ".env")
    asyncio.run(main())
