"""Run a tiny custom Spellbook app server.

This is the smallest useful pattern for downstream integrations:

- define a Pydantic input model
- define an async tool function
- wrap it in a `Tool`
- pass `CustomSurface(tools=[...])` into `create_app`
- run the app with uvicorn

Usage:
    uv run python examples/custom_surface_server.py
    uv run python examples/custom_surface_server.py ./custom-transcript.jsonl --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from spellbook.app.server import create_app
from spellbook.backends import infer_provider_for_model
from spellbook.config import (
    DEFAULT_DETECT_INTERVAL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_USER_NAME,
    HomunculusConfig,
    SpellbookConfig,
)
from spellbook.custom import CustomSurface
from spellbook.ir_types import IRToolTextBlock
from spellbook.system_prompt import build_runtime_system_prompt
from spellbook.tools.common import (
    Tool,
    ToolCategory,
    ToolExecutionResult,
    ToolMetadata,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_ENV_PATH = Path.home() / ".chorus" / ".env"
SESSIONS_DIR = Path.home() / ".spellbook" / "sessions"

CUSTOM_PROMPT = """\
You are running inside a custom Spellbook surface.

This example surface gives you one local custom tool, ReadConstellation. Use it
when a lightweight symbolic map, planning nudge, or thematic reframing would
help the conversation.
"""


class ReadConstellationInput(BaseModel):
    """Read a small deterministic constellation note for a topic."""

    topic: str = Field(
        min_length=1,
        max_length=200,
        description="The topic, project, question, or feeling to map.",
    )
    question: str | None = Field(
        default=None,
        max_length=500,
        description="Optional question to orient the note around.",
    )
    tone: Literal["clear", "curious", "steady", "tender"] = Field(
        default="curious",
        description="The flavor of the returned note.",
    )


async def read_constellation(
    meta: ToolMetadata,
    input: ReadConstellationInput,
) -> ToolExecutionResult:
    """Return a deterministic, local, no-network note."""

    topic = " ".join(input.topic.split())
    question = " ".join(input.question.split()) if input.question else None
    seed = hashlib.sha256(
        f"{topic}|{question or ''}|{input.tone}|{meta.cwd}".encode("utf-8")
    ).hexdigest()

    fixed_stars = [
        "name the real constraint",
        "protect the living thread",
        "make the next step inspectable",
        "keep the interface kind",
        "let the transcript stay truthful",
    ]
    moving_lights = [
        "a small prototype will teach more than another abstraction pass",
        "the missing test is probably a boundary test",
        "the useful simplification is hiding in ownership",
        "the surface wants fewer knobs and better defaults",
        "the next good move is to write down the invariant",
    ]
    closing_notes = [
        "Look for the part that already knows what shape it wants.",
        "Prefer the change that future-you can verify at a glance.",
        "If the tool feels magical, make the record more explicit.",
        "A quiet seam is usually better than a clever one.",
        "Leave a breadcrumb where the next mind will need it.",
    ]

    first = int(seed[0:2], 16)
    second = int(seed[2:4], 16)
    third = int(seed[4:6], 16)
    lines = [
        "<constellation-note>",
        f"topic: {topic}",
        f"tone: {input.tone}",
    ]
    if question is not None:
        lines.append(f"question: {question}")
    lines.extend(
        [
            "",
            f"fixed star: {fixed_stars[first % len(fixed_stars)]}",
            f"moving light: {moving_lights[second % len(moving_lights)]}",
            f"next glimmer: {closing_notes[third % len(closing_notes)]}",
            "</constellation-note>",
        ]
    )

    return ToolExecutionResult(
        content=[IRToolTextBlock(text="\n".join(lines))],
        display={
            "kind": "constellation_note",
            "topic": topic,
            "tone": input.tone,
        },
    )


READ_CONSTELLATION_TOOL: Tool[ReadConstellationInput] = Tool(
    name="ReadConstellation",
    input_model=ReadConstellationInput,
    exec=read_constellation,
    category="thinking",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="examples.custom_surface_server",
        description="Run a Spellbook custom surface example server.",
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a core transcript. Omit to create a new custom transcript.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model slug for a new transcript. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host/interface to bind. Defaults to {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to bind. Defaults to {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="Working directory for a newly initialized session. Defaults to cwd.",
    )
    parser.add_argument(
        "--user-name",
        default=DEFAULT_USER_NAME,
        help=f"User name for a new transcript. Defaults to {DEFAULT_USER_NAME}.",
    )
    parser.add_argument(
        "--include-memory-tools",
        action="store_true",
        help="Also expose Reflect, Forget, Pin, and Recall on the custom surface.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before startup. Defaults to {DEFAULT_ENV_PATH}.",
    )
    parser.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
        help="Spellbook app and Uvicorn log level. Defaults to info.",
    )
    return parser.parse_args(argv)


def _custom_surface(args: argparse.Namespace) -> CustomSurface:
    include: set[ToolCategory] = {"memory"} if args.include_memory_tools else set()
    return CustomSurface(
        tools=[READ_CONSTELLATION_TOOL],
        include_tool_categories=include,
    )


def _config_from_args(args: argparse.Namespace) -> SpellbookConfig:
    provider = infer_provider_for_model(args.model)
    return SpellbookConfig(
        provider=provider,
        model=args.model,
        session_type="custom",
        cwd=args.cwd.expanduser().resolve(),
        user_name=args.user_name,
        max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
        system_prompt=build_runtime_system_prompt(
            model=args.model,
            cwd=args.cwd,
            user_name=args.user_name,
            system_prompt_text=(CUSTOM_PROMPT,),
            discover_claude_md=provider == "anthropic",
        ),
        hom_config=HomunculusConfig(detect_interval=DEFAULT_DETECT_INTERVAL),
    )


def _new_transcript_path() -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SESSIONS_DIR / f"custom_surface_{timestamp}.jsonl"


def _resolve_transcript_path(args: argparse.Namespace) -> Path:
    if args.transcript is None:
        return _new_transcript_path().resolve()
    return args.transcript.expanduser().resolve()


def _log_level_from_arg(value: str) -> str:
    if value == "trace":
        return "debug"
    return value


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)

    transcript_path = _resolve_transcript_path(args)
    custom_surface = _custom_surface(args)
    app = create_app(
        transcript_path=transcript_path,
        config=None if transcript_path.exists() else _config_from_args(args),
        custom_surface=custom_surface,
        log_level=args.log_level,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=_log_level_from_arg(args.log_level),
    )


if __name__ == "__main__":
    main()
