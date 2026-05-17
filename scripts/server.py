"""Run the new core app server around a transcript.

Usage:
    python -m scripts.server /path/to/transcript.jsonl --model claude-sonnet-4-6
    python -m scripts.server --model claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import uvicorn
from dotenv import load_dotenv
from spellbook.app.server import create_app
from spellbook.backends import infer_provider_for_model
from spellbook.config import (
    DEFAULT_DETECT_INTERVAL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_USER_NAME,
    HomunculusConfig,
    SpellbookConfig,
)
from spellbook.orientation import (
    model_slug_to_name,
    orientation_filename_for_model,
)
from spellbook.system_prompt import (
    build_runtime_orientation,
    build_runtime_system_prompt,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_ENV_PATH = Path.home() / ".chorus" / ".env"
SESSIONS_DIR = Path.home() / ".spellbook" / "sessions"

LogLevel = Literal["critical", "error", "warning", "info", "debug", "trace"]


def _sessions_dir() -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR


def _new_transcript_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _sessions_dir() / f"server_{timestamp}.jsonl"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts.server",
        description="Run the new Spellbook core app server.",
    )
    parser.add_argument(
        "transcript",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Path to the core transcript to serve. Omit to create a new "
            f"transcript in {SESSIONS_DIR}. If the path does not exist, "
            "a new transcript is initialized from --model."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model slug to use when initializing a new transcript.",
    )
    parser.add_argument(
        "--system-prompt-text",
        action="append",
        default=None,
        help=(
            "Additional system prompt text to append after the core server prompt. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        action="append",
        dest="system_prompt_files",
        default=None,
        help=(
            "Path to a file containing additional system prompt text to append "
            "after --system-prompt-text. May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--no-discover-claude-md",
        action="store_true",
        help="Disable automatic CLAUDE.md discovery for newly initialized sessions.",
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
        type=str,
        default=DEFAULT_USER_NAME,
        help="The name of the user, to be included in the orientation.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Max output tokens for a new session. Defaults to {DEFAULT_MAX_OUTPUT_TOKENS}.",
    )
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=DEFAULT_DETECT_INTERVAL,
        help=f"Detector interval in context blocks. Defaults to {DEFAULT_DETECT_INTERVAL}",
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
    args = parser.parse_args(argv)
    if args.model is None and (
        args.transcript is None or not args.transcript.expanduser().exists()
    ):
        parser.error("--model is required when initializing a new transcript")
    return args


def _config_from_args(args: argparse.Namespace) -> SpellbookConfig:
    if args.model is None:
        raise ValueError("Cannot build a new SpellbookConfig without --model.")
    return SpellbookConfig(
        provider=infer_provider_for_model(args.model),
        system_prompt=_system_prompt_from_args(args),
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        cwd=args.cwd.expanduser().resolve(),
        user_name=args.user_name,
        hom_config=HomunculusConfig(detect_interval=args.detect_interval),
    )


def _system_prompt_from_args(args: argparse.Namespace) -> str:
    return build_runtime_system_prompt(
        model=args.model,
        cwd=args.cwd,
        user_name=args.user_name,
        system_prompt_text=tuple(args.system_prompt_text or ()),
        system_prompt_files=tuple(args.system_prompt_files or ()),
        discover_claude_md=_should_discover_claude_md(args),
    )


def _should_discover_claude_md(args: argparse.Namespace) -> bool:
    if args.no_discover_claude_md:
        return False
    if args.model is None:
        return False
    return infer_provider_for_model(args.model) == "anthropic"


def _log_level_from_arg(value: str) -> LogLevel:
    if value not in {"critical", "error", "warning", "info", "debug", "trace"}:
        raise ValueError(f"Unsupported log level: {value}")
    return cast(LogLevel, value)


def _model_slug_to_name(slug: str) -> str:
    return model_slug_to_name(slug)


def _orientation_filename_for_model(model: str) -> str:
    return orientation_filename_for_model(model)


def _build_system_prompt(model: str, *, cwd: Path, user_name: str) -> str:
    return build_runtime_orientation(model, cwd=cwd, user_name=user_name)


def _resolve_transcript_path(args: argparse.Namespace) -> Path:
    if args.transcript is None:
        return _new_transcript_path().resolve()
    return args.transcript.expanduser().resolve()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    transcript_path = _resolve_transcript_path(args)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)

    app = create_app(
        transcript_path=transcript_path,
        config=None if transcript_path.exists() else _config_from_args(args),
        log_level=_log_level_from_arg(args.log_level),
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=_log_level_from_arg(args.log_level),
    )


if __name__ == "__main__":
    main()
