"""Count tokens for a markdown file as one Anthropic user message."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from spellbook.backends.anthropic import AnthropicBackend
from spellbook.backends.model_backend import TokenCounter
from spellbook.config import SpellbookConfig
from spellbook.ir_types import IRUserTextBlock
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.registry import ToolRegistry

DEFAULT_ENV_PATH = Path.home() / ".chorus/.env"
DEFAULT_MODEL = "claude-opus-4-6"


@dataclass(frozen=True)
class MarkdownTokenReport:
    path: Path
    model: str
    tokens: int
    characters: int


async def count_markdown_tokens(
    path: Path,
    *,
    model: str = DEFAULT_MODEL,
    token_counter: TokenCounter | None = None,
) -> MarkdownTokenReport:
    markdown_path = path.expanduser().resolve()
    text = markdown_path.read_text(encoding="utf-8")
    counter = token_counter or _build_token_counter(model)
    count = await counter.count_blocks([IRUserTextBlock(text=text, origin="human")])
    if count is None:
        raise RuntimeError(f"Token counting failed for {markdown_path}.")
    return MarkdownTokenReport(
        path=markdown_path,
        model=model,
        tokens=count,
        characters=len(text),
    )


def _build_token_counter(model: str) -> TokenCounter:
    config = SpellbookConfig(
        provider="anthropic",
        model=model,
        cwd=Path.cwd(),
    )
    backend = AnthropicBackend()
    registry = ToolRegistry.build(config.tool_categories, surface=config.session_type)
    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=registry,
    )
    return backend.build_token_counter(config=config, surface_builder=surface_builder)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mdtoks",
        description="Count a markdown file as the content of one Anthropic user message.",
    )
    parser.add_argument("path", type=Path, help="Markdown file to count.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Anthropic model to count against. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help=f"Dotenv file to load before token counting. Defaults to {DEFAULT_ENV_PATH}.",
    )
    return parser.parse_args(argv)


async def _async_main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    env_path = args.env.expanduser()
    if env_path.exists():
        load_dotenv(env_path)

    report = await count_markdown_tokens(args.path, model=args.model)
    _print_report(report)


def _print_report(report: MarkdownTokenReport) -> None:
    print(f"File:   {report.path}")
    print(f"Model:  {report.model}")
    print(f"Tokens: {report.tokens:,}")
    print(f"Chars:  {report.characters:,}")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(_async_main(argv))


if __name__ == "__main__":
    main()
