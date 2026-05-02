import asyncio
from pathlib import Path

import rich
from dotenv import load_dotenv

from spellbook.backends.anthropic import AnthropicBackend
from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.executor import Executor
from spellbook.generator import Generator
from spellbook.ir_types import IRBlock, IRSkillCatalog, IRUserTextBlock
from spellbook.loop import run_loop
from spellbook.recorder import Recorder
from spellbook.round_lifecycle import RoundLifecycle
from spellbook.surface_builder import RequestSurfaceBuilder
from spellbook.tools.registry import DEFAULT_TOOL_REGISTRY


async def main() -> None:
    backend = AnthropicBackend()
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        max_output_tokens=4096,
        cwd=Path.cwd(),
        system_prompt="You are an entity with access to a `Bash` tool. You and the user share the environment.",
    )
    cancel_token = CancelToken()
    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=DEFAULT_TOOL_REGISTRY,
    )
    recorder = Recorder(
        config,
        Path("/home/rheaton64/code/spellbook/scripts/out.jsonl"),
        "session1",
        DEFAULT_TOOL_REGISTRY,
    )

    generator = Generator(
        backend=backend, config=config, surface_builder=surface_builder
    )

    executor = Executor(config, Path(), DEFAULT_TOOL_REGISTRY)

    initial_blocks: list[IRBlock] = [
        IRUserTextBlock(
            text="Hi Claude! Can you tell me what files you see in your current directory?",
            origin="human",
        )
    ]

    lifecycle = RoundLifecycle()
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("turn1", initial_blocks)
    loop_result = await run_loop(
        generator=generator,
        executor=executor,
        lifecycle=lifecycle,
        initial_blocks=initial_blocks,
        cancel_token=cancel_token,
    )
    for block in loop_result.blocks:
        recorder.write_block(block)
    recorder.end_turn(loop_result.stop_reason)
    rich.print(loop_result)


if __name__ == "__main__":
    load_dotenv("/home/rheaton64/.chorus/.env")
    asyncio.run(main())
