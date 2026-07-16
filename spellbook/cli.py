"""Command-line entry point for Spellbook."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import uvicorn
from dotenv import load_dotenv

from spellbook.app.server import create_app
from spellbook.config import Provider
from spellbook.profiles import SessionType
from spellbook.serve import (
    ServeConfigError,
    ServeOverrides,
    build_spellbook_config,
    load_entity_file,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_ENV_PATH = Path.home() / ".chorus" / ".env"

LogLevel = Literal["critical", "error", "warning", "info", "debug", "trace"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spellbook",
        description="Run and tend persistent Spellbook entities.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser(
        "serve",
        help="Start an entity app server from a TOML config.",
        description=(
            "Start a fully configured Spellbook entity. A positional path is an "
            "entity TOML file unless --config is also supplied, in which case it "
            "is the transcript path."
        ),
    )
    serve.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Entity TOML file, or transcript path when used with --config.",
    )
    serve.add_argument("--config", type=Path, help="Entity TOML config file.")
    serve.add_argument(
        "--transcript",
        type=Path,
        help="Transcript to create or resume. Defaults to ~/.spellbook/sessions/.",
    )
    serve.add_argument("--model", help="Override entity.model.")
    serve.add_argument(
        "--provider",
        choices=("anthropic", "openai", "local"),
        help="Override entity.provider.",
    )
    serve.add_argument("--effort", help="Override entity.effort.")
    serve.add_argument("--cwd", type=Path, help="Override entity.cwd.")
    serve.add_argument("--max-output-tokens", type=int)
    serve.add_argument("--user-name")
    serve.add_argument("--local-timezone")
    serve.add_argument("--idle-footer-threshold-seconds", type=int)
    serve.add_argument(
        "--session-type",
        choices=("main", "custom", "block_detector", "block_summarizer", "quantum"),
    )
    serve.add_argument("--orientation", help="Override prompt.orientation.")
    role_source = serve.add_mutually_exclusive_group()
    role_source.add_argument("--role", help="Override prompt.role with inline text.")
    role_source.add_argument(
        "--role-file", type=Path, help="Override prompt.role_file."
    )
    serve.add_argument("--frame", help="Override prompt.frame; use 'none' to skip.")
    serve.add_argument(
        "--local-frame-discovery",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override discovery.local_frame_discovery.",
    )
    serve.add_argument(
        "--skill-discovery-dir",
        action="append",
        dest="skill_discovery_dirs",
        help="Override discovery.skill_discovery_dirs; repeat for multiple paths.",
    )
    serve.add_argument("--detect-interval", type=int)
    serve.add_argument("--soft-threshold", type=int)
    serve.add_argument("--medium-threshold", type=int)
    serve.add_argument("--hard-threshold", type=int)
    serve.add_argument("--max-tokens", type=int)
    serve.add_argument(
        "--ttl-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--ttl-turns", type=int)
    serve.add_argument("--ttl-char-threshold", type=int)
    serve.add_argument(
        "--hearth-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--hearth-interval-minutes", type=int)
    serve.add_argument("--hearth-quiet-hours")
    serve.add_argument(
        "--tool-category",
        action="append",
        dest="tool_categories",
        help="Override tools.categories; repeat for multiple categories.",
    )
    serve.add_argument("--body-url")
    serve.add_argument("--chorus-url")
    serve.add_argument("--chorus-entity-name")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    serve.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _serve_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    if args.config is None:
        return args.path, args.transcript
    if args.path is not None and args.transcript is not None:
        raise ServeConfigError(
            "Pass the transcript either positionally with --config or via "
            "--transcript, not both."
        )
    return args.config, args.transcript or args.path


def _new_transcript_path(config_path: Path | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = config_path.stem if config_path is not None else "server"
    return Path.home() / ".spellbook" / "sessions" / f"{stem}_{timestamp}.jsonl"


def _resolve_transcript_path(
    transcript_path: Path | None, *, config_path: Path | None
) -> Path:
    if transcript_path is None:
        return _new_transcript_path(config_path).resolve()
    return transcript_path.expanduser().resolve()


def _overrides_from_args(args: argparse.Namespace) -> ServeOverrides:
    provider = cast(Provider | None, args.provider)
    session_type = cast(SessionType | None, args.session_type)
    return ServeOverrides(
        model=args.model,
        provider=provider,
        effort=args.effort,
        cwd=args.cwd,
        max_output_tokens=args.max_output_tokens,
        user_name=args.user_name,
        local_timezone=args.local_timezone,
        idle_footer_threshold_seconds=args.idle_footer_threshold_seconds,
        session_type=session_type,
        orientation=args.orientation,
        role=args.role,
        role_file=args.role_file,
        frame=args.frame,
        local_frame_discovery=args.local_frame_discovery,
        skill_discovery_dirs=args.skill_discovery_dirs,
        detect_interval=args.detect_interval,
        soft_threshold=args.soft_threshold,
        medium_threshold=args.medium_threshold,
        hard_threshold=args.hard_threshold,
        max_tokens=args.max_tokens,
        ttl_enabled=args.ttl_enabled,
        ttl_turns=args.ttl_turns,
        ttl_char_threshold=args.ttl_char_threshold,
        hearth_enabled=args.hearth_enabled,
        hearth_interval_minutes=args.hearth_interval_minutes,
        hearth_quiet_hours=args.hearth_quiet_hours,
        tool_categories=args.tool_categories,
        body_url=args.body_url,
        chorus_url=args.chorus_url,
        chorus_entity_name=args.chorus_entity_name,
    )


def _run_serve(args: argparse.Namespace) -> None:
    config_path, requested_transcript = _serve_paths(args)
    loaded = load_entity_file(config_path)
    transcript_path = _resolve_transcript_path(
        requested_transcript, config_path=loaded.path
    )
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)

    is_resume = transcript_path.exists()
    config = None
    if not is_resume:
        config = build_spellbook_config(loaded, _overrides_from_args(args))
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    log_level = cast(LogLevel, args.log_level)
    app = create_app(
        transcript_path=transcript_path,
        config=config,
        log_level=log_level,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=log_level,
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        match args.command:
            case "serve":
                _run_serve(args)
            case _:
                parser.error(f"Unknown command: {args.command}")
    except ServeConfigError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
