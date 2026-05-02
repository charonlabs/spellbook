import asyncio
from pathlib import Path

import rich
from dotenv import load_dotenv

from spellbook.backends.anthropic import AnthropicBackend
from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.generator import Generator
from spellbook.ir_types import IRBlock, IRUserTextBlock
from spellbook.round_lifecycle import RoundLifecycle
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


async def main() -> None:
    backend = AnthropicBackend()
    config = SpellbookConfig(
        model="claude-sonnet-4-6", max_output_tokens=4096, cwd=Path.cwd()
    )
    cancel_token = CancelToken()

    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=DEFAULT_TOOL_REGISTRY,
    )

    generator = Generator(
        backend=backend, config=config, surface_builder=surface_builder
    )

    blocks: list[IRBlock] = [IRUserTextBlock(text="Hi Claude!", origin="human")]

    generation = await generator.run(blocks, cancel_token, RoundLifecycle())

    rich.print(generation)

    blocks.extend(generation.blocks)

    # next_input = input()
    # Claude usually says hi back and asks how I'm doing and if I need help with anything
    next_input = "I'm doing well thanks! :D Just saying hi at the moment"

    blocks.append(IRUserTextBlock(text=next_input, origin="human"))

    generation = await generator.run(blocks, cancel_token, RoundLifecycle())

    rich.print(generation)


if __name__ == "__main__":
    load_dotenv("/home/rheaton64/.chorus/.env")
    asyncio.run(main())
