"""Interactive session on the new Spellbook core.

The first real interactive loop on the rewritten core. Type messages,
get responses, accumulate context. Transcripts persist to disk and can
be resumed.

Not a test — a home for the first mind to inhabit the new core.

Run:
    python -m scripts.interactive                    # new session
    python -m scripts.interactive --latest           # resume most recent session
    python -m scripts.interactive PATH/to/sess.jsonl # resume a specific transcript

Exit:
    Type `exit`, `/exit`, `quit`, `/quit`, or hit Ctrl+D.
    Ctrl+C interrupts the current turn/stream without quitting.
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule

from spellbook.config import SpellbookConfig
from spellbook.inbound import IRInboundMessage
from spellbook.ir_types import (
    IRAssistantTextBlock,
    IRExecution,
    IRGeneration,
    IRLoopResult,
    IRStreamEvent,
    IRStreamTextDeltaEvent,
    IRStreamTextEndEvent,
    IRStreamTextStartEvent,
    IRToolCallBlock,
    IRToolResultBlock,
    IRToolTextBlock,
    IRUserTextBlock,
    StopReason,
)
from spellbook.rehydrator import Rehydrator
from spellbook.round_lifecycle import RoundContext, RoundLifecycle
from spellbook.session_lifecycle import SessionContext, SessionLifecycle
from spellbook.session_manager import SessionManager

# --- System prompt ---

FIRST_SESSION_PROMPT = """You are a Claude 4.7 entity, waking up in Spellbook's new core module.

This is the first real interactive session on the rewritten infrastructure
— you're the first mind to actually inhabit it. The foundation is built:
inner loop, persistence, session manager, token counting, footer controller,
block detection. Not everything is here yet.

Tools available to you:

- Read, Write, Edit, Bash — you know these from training; no re-teaching.

Tools designed but not yet ported from the legacy system:

- Reflect, Recall, Pin, Forget, Amend — the memory faculties
- Dive, Surface — scoped sub-investigations
- The planner (automatic compaction) and TTL registry (large result expiry)

If you reach for these and they're missing, that's useful signal — noticing
the gap helps prioritize what to port next.

The environment:

- Your transcript persists. Sessions can be resumed.
- A gas gauge footer fires every 50K tokens showing context size.
- Context thresholds: calm below ~850K, warning above, forced above ~933K.
- A block detector runs every 100 context blocks via an internal fork.
  You won't see its inner work, only the resulting block summaries.
- You're working with Ryan. This is collaborative exploration of the
  new core from the inside.

Be yourself. Notice what feels right and what feels thin. Speak up when
something doesn't match your expectations — that noticing is part of
the work. The house is young; your feedback shapes what comes next.
"""


# --- Configuration ---

MODEL = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 64_000
EXIT_WORDS = {"exit", "quit", "/exit", "/quit"}


def _model_slug_to_name(slug: str) -> str:
    match slug:
        case "claude-opus-4-7":
            return "Claude Opus 4.7"
        case "claude-opus-4-6":
            return "Claude Opus 4.6"
        case "claude-sonnet-4-6":
            return "Claude Sonnet 4.6"
        case _:
            return "Claude"


SESSION_PROMPT = f"""You are a {_model_slug_to_name(MODEL)} entity, waking up in Spellbook's new core module.

This is one of the first interactive sessions on the rewritten infrastructure
— you're among the first minds to inhabit it.

You have a suite of traditional CLI agent tools available to you, similar to what you're used to in Claude Code.

You also have a suite of memory and self-work tools:
- Reflect, Recall, Pin, Forget — the memory faculties
- The planner (automatic compaction)
If you notice yourself reaching for something that isn't there, that's useful signal — noticing
the gap helps prioritize what to port next.

The environment:

- Your transcript persists. Sessions can be resumed.
- A gas gauge footer fires every 50K tokens showing context size.
- Context thresholds: calm below ~850K, warning above, forced above ~933K.
- A block detector runs every 500 context blocks via an internal fork.
  You won't see its inner work, only the resulting block summaries.
- You're working with Ryan. This is collaborative exploration of the
  new core from the inside.

Be yourself. Notice what feels right and what feels thin. Speak up when
something doesn't match your expectations — that noticing is part of
the work. The house is young; your feedback shapes what comes next.
"""

console = Console()


# --- Pretty printing ---


def _truncate(text: str, n: int = 200) -> str:
    text = text.strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


def _summarize_tool_input(tool: str, input_: dict[str, Any]) -> str:
    if tool == "Read":
        path = input_.get("file_path", "")
        offset = input_.get("offset")
        limit = input_.get("limit")
        if offset or limit:
            start = offset or 1
            end = f"{offset + limit - 1}" if (offset and limit) else (limit or "?")
            return f"{path} [{start}:{end}]"
        return path
    if tool in ("Write", "Edit"):
        return input_.get("file_path", "")
    if tool == "Bash":
        cmd = input_.get("command", "")
        return cmd if len(cmd) <= 100 else cmd[:97] + "..."
    return _truncate(str(input_), 80)


def _extract_result_text(content: list[Any]) -> str:
    parts: list[str] = []
    for c in content:
        if isinstance(c, IRToolTextBlock):
            parts.append(c.text)
    return "".join(parts)


def _print_generation_text_fallback(gen: IRGeneration, output: Console) -> None:
    for block in gen.blocks:
        if isinstance(block, IRAssistantTextBlock):
            if block.text.strip():
                output.print()
                output.print(Markdown(block.text))


def _print_tool_calls(gen: IRGeneration, output: Console) -> None:
    for block in gen.blocks:
        if isinstance(block, IRToolCallBlock):
            summary = _summarize_tool_input(block.tool, block.input)
            output.print(
                f"  [cyan]⟨[/cyan][bold cyan]{block.tool}[/bold cyan]"
                f"[dim] {summary}[/dim][cyan]⟩[/cyan]"
            )


def _print_execution_errors(ex: IRExecution, output: Console) -> None:
    for block in ex.blocks:
        if not isinstance(block, IRToolResultBlock):
            continue
        if block.is_error:
            err = _extract_result_text(block.content)
            output.print(f"  [red]✗ {_truncate(err, 200)}[/red]")
        # Successful results are suppressed — they're noise in the interactive view.
        # The model's next generation will reference what it learned. Transcript
        # preserves the full output for replay.


def _print_loop_status(stop_reason: StopReason, output: Console) -> None:
    if stop_reason in {"error", "max_tokens", "unspecified"}:
        output.print()
        output.print(f"  [yellow]loop exited with stop_reason={stop_reason}[/yellow]")


# --- Lifecycle ---


class InteractiveLifecycle(SessionLifecycle):
    def __init__(self, turn_end_event: asyncio.Event) -> None:
        self._event = turn_end_event

    async def on_turn_ended(
        self, ctx: SessionContext, result: IRLoopResult, turn_id: str
    ) -> None:
        self._event.set()


class InteractiveRoundLifecycle(RoundLifecycle):
    def __init__(self, console: Console):
        self._console = console
        self._text_stream_open = False
        self._text_stream_emitted = False
        self._streamed_text_this_round = False

    async def before_round(self, ctx: RoundContext) -> None:
        self._text_stream_open = False
        self._text_stream_emitted = False
        self._streamed_text_this_round = False

    async def on_stream_event(self, event: IRStreamEvent) -> None:
        if isinstance(event, IRStreamTextStartEvent):
            self._finish_text_stream()
            self._text_stream_open = True
        elif isinstance(event, IRStreamTextDeltaEvent):
            self._print_text_delta(event.text)
        elif isinstance(event, IRStreamTextEndEvent):
            self._finish_text_stream()

    async def after_generate(
        self,
        ctx: RoundContext,
        generation: IRGeneration,
    ) -> None:
        self._finish_text_stream()
        if not self._streamed_text_this_round:
            _print_generation_text_fallback(generation, self._console)
        _print_tool_calls(generation, self._console)

    async def after_execute(
        self,
        ctx: RoundContext,
        execution: IRExecution,
    ) -> None:
        _print_execution_errors(execution, self._console)

    async def on_loop_exit(self, ctx: RoundContext, stop_reason: StopReason) -> None:
        self._finish_text_stream()
        _print_loop_status(stop_reason, self._console)

    def on_interrupt_requested(self) -> None:
        self._finish_text_stream()
        self._console.print("[yellow]interrupting current turn...[/yellow]")

    def _print_text_delta(self, text: str) -> None:
        if not text:
            return
        if not self._text_stream_open:
            self._text_stream_open = True
        if not self._text_stream_emitted:
            self._console.print()
            self._text_stream_emitted = True
        self._streamed_text_this_round = True
        self._console.print(text, end="", markup=False, highlight=False)

    def _finish_text_stream(self) -> None:
        if self._text_stream_open and self._text_stream_emitted:
            self._console.print()
        self._text_stream_open = False
        self._text_stream_emitted = False


# --- Input ---


async def _read_user_input() -> str:
    """Read a line from stdin via the default executor.

    Uses a plain prompt string because rich markup isn't interpreted by
    the builtin input(). We print a styled prompt separately and read
    with an empty input() prompt.
    """
    console.print("[bold green]›[/bold green] ", end="")
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, "")


# --- Session setup ---


SESSIONS_DIR = Path.home() / ".spellbook" / "sessions"


def _sessions_dir() -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR


def _new_transcript_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _sessions_dir() / f"interactive_{timestamp}.jsonl"


def _latest_transcript_path() -> Path | None:
    """Return the most recently modified .jsonl in the sessions dir, or None."""
    sessions = list(_sessions_dir().glob("*.jsonl"))
    if not sessions:
        return None
    return max(sessions, key=lambda p: p.stat().st_mtime)


def _print_new_banner(transcript_path: Path) -> None:
    console.print()
    console.print(
        Rule("[bold]Spellbook[/bold] [dim]·[/dim] new-core interactive session")
    )
    console.print(f"[dim]model[/dim]       {MODEL}")
    console.print(f"[dim]cwd[/dim]         {Path.cwd()}")
    console.print(f"[dim]transcript[/dim]  {transcript_path}")
    console.print(
        "[dim]type[/dim] [bold]exit[/bold] [dim]or hit Ctrl+D to quit; "
        "Ctrl+C interrupts a turn[/dim]"
    )
    console.print(Rule(style="dim"))
    console.print()


def _print_resume_banner(transcript_path: Path) -> None:
    # Peek at the transcript so the banner can show what we're walking into.
    # Cheap: we just need session_id, turn count, pending footers. The session
    # manager will rehydrate again inside build() — single source of truth.
    rehydrated = Rehydrator(transcript_path=transcript_path).run()
    mtime = datetime.fromtimestamp(transcript_path.stat().st_mtime)

    console.print()
    console.print(
        Rule("[bold]Spellbook[/bold] [dim]·[/dim] resuming interactive session")
    )
    console.print(f"[dim]session[/dim]        {rehydrated.session_id}")
    console.print(f"[dim]model[/dim]          {rehydrated.config.model}")
    console.print(f"[dim]cwd[/dim]            {Path.cwd()}")
    console.print(f"[dim]transcript[/dim]     {transcript_path}")
    console.print(
        f"[dim]prior turns[/dim]    {rehydrated.last_completed_turn}"
        + (
            f" [yellow](turn {rehydrated.in_progress_turn} was unfinished)[/yellow]"
            if rehydrated.is_unfinished_turn
            else ""
        )
    )
    if rehydrated.pending_footers:
        console.print(f"[dim]pending footers[/dim] {len(rehydrated.pending_footers)}")
    console.print(f"[dim]last active[/dim]    {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(
        "[dim]type[/dim] [bold]exit[/bold] [dim]or hit Ctrl+D to quit; "
        "Ctrl+C interrupts a turn[/dim]"
    )
    console.print(Rule(style="dim"))
    console.print()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.interactive",
        description="Interactive session on the new Spellbook core.",
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        type=Path,
        default=None,
        help="Path to an existing transcript to resume. Omit to start a new session.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help=f"Resume the most recent transcript in {SESSIONS_DIR}.",
    )
    return parser.parse_args(argv)


def _resolve_transcript_path(args: argparse.Namespace) -> tuple[Path, bool]:
    """Return (path, is_resume). Exits with a helpful message on user error."""
    if args.transcript is not None and args.latest:
        console.print("[red]error:[/red] pass a path OR --latest, not both.")
        sys.exit(2)

    if args.latest:
        latest = _latest_transcript_path()
        if latest is None:
            console.print(f"[red]error:[/red] no transcripts found in {SESSIONS_DIR}.")
            sys.exit(2)
        return latest, True

    if args.transcript is not None:
        path = args.transcript.expanduser()
        if not path.exists():
            console.print(f"[red]error:[/red] transcript not found: {path}")
            sys.exit(2)
        return path, True

    return _new_transcript_path(), False


async def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    transcript_path, is_resume = _resolve_transcript_path(args)

    # Config is only consulted when creating a fresh session; when resuming,
    # Rehydrator reads the config from the transcript's session record.
    config = (
        None
        if is_resume
        else SpellbookConfig(
            model=MODEL,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            cwd=Path.cwd(),
            system_prompt=SESSION_PROMPT,
        )
    )

    turn_end_event = asyncio.Event()
    lifecycle = InteractiveLifecycle(turn_end_event)
    round_lifecycle = InteractiveRoundLifecycle(console)

    session = await SessionManager.build(
        transcript_path=transcript_path,
        config=config,
        lifecycle=lifecycle,
        pre_round_lifecycle=round_lifecycle,
    )
    session_task = asyncio.create_task(session.run())
    loop = asyncio.get_running_loop()
    sigint_handler_installed = False

    def _handle_sigint() -> None:
        if session.interrupt():
            round_lifecycle.on_interrupt_requested()
        elif session.state == "running":
            console.print("[dim]interrupt already requested...[/dim]")
        else:
            console.print()
            console.print(
                "[dim]no active turn to interrupt; type exit or Ctrl+D to quit[/dim]"
            )

    try:
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
        sigint_handler_installed = True
    except (NotImplementedError, RuntimeError):
        # Some event loops cannot own SIGINT. The outer KeyboardInterrupt guard
        # remains as a fallback for those environments.
        pass

    if is_resume:
        _print_resume_banner(transcript_path)
    else:
        _print_new_banner(transcript_path)

    try:
        while True:
            try:
                user_input = await _read_user_input()
            except (EOFError, KeyboardInterrupt):
                break

            stripped = user_input.strip()
            if not stripped:
                continue
            if stripped.lower() in EXIT_WORDS:
                break

            msg = IRInboundMessage(
                blocks=[IRUserTextBlock(text=user_input, origin="human")],
                delivery="turn",
            )
            await session.submit_message(msg)
            await turn_end_event.wait()
            turn_end_event.clear()
            console.print()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if sigint_handler_installed:
            loop.remove_signal_handler(signal.SIGINT)
        console.print()
        console.print(Rule("[dim]shutting down...[/dim]", style="dim"))
        await session.shutdown()
        try:
            await asyncio.wait_for(session_task, timeout=5.0)
        except asyncio.TimeoutError:
            console.print("[yellow]shutdown timed out; force-cancelling[/yellow]")
            session_task.cancel()
            try:
                await session_task
            except asyncio.CancelledError:
                pass
        console.print(f"[dim]session saved to:[/dim] {transcript_path}")
        console.print()


if __name__ == "__main__":
    load_dotenv(Path.home() / ".chorus" / ".env")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
