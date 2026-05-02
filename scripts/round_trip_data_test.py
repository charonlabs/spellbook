import asyncio
from pathlib import Path

from dotenv import load_dotenv

from spellbook.backends.anthropic import AnthropicBackend
from spellbook.cancel_token import CancelToken
from spellbook.config import SpellbookConfig
from spellbook.executor import Executor
from spellbook.generator import Generator
from spellbook.ir_types import IRBlock, IRSkillCatalog, IRUserTextBlock
from spellbook.loop import run_loop
from spellbook.recorder import Recorder, RecordingRoundLifecycle
from spellbook.rehydrator import Rehydrator
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

    transcript_path = Path("/tmp/out_round.jsonl")
    if transcript_path.is_file():
        transcript_path.unlink()

    recorder = Recorder(
        config,
        transcript_path,
        "session1",
        DEFAULT_TOOL_REGISTRY,
    )

    surface_builder = RequestSurfaceBuilder.from_config(
        backend=backend,
        config=config,
        tool_registry=DEFAULT_TOOL_REGISTRY,
    )

    generator = Generator(
        backend=backend, config=config, surface_builder=surface_builder
    )

    executor = Executor(config, transcript_path, DEFAULT_TOOL_REGISTRY)

    initial_blocks: list[IRBlock] = [
        IRUserTextBlock(
            text="Hi Claude! Can you tell me what files you see in your current directory?",
            origin="human",
        )
    ]

    lifecycle = RecordingRoundLifecycle(recorder)
    recorder.write_session_record(skill_catalog=IRSkillCatalog())
    recorder.start_turn("turn1", initial_blocks)
    loop_result = await run_loop(
        generator=generator,
        executor=executor,
        lifecycle=lifecycle,
        initial_blocks=initial_blocks,
        cancel_token=cancel_token,
    )
    recorder.end_turn(loop_result.stop_reason)

    rehydrator = Rehydrator(transcript_path)
    rehydrated = rehydrator.run()
    stripped_blocks = [
        rb.model_copy(update={"event_id": None, "turn_id": None})
        for rb in rehydrated.blocks
    ]
    assert stripped_blocks == loop_result.blocks
    assert rehydrated.config == config
    print("**PASSED**")
    transcript_path.unlink()


if __name__ == "__main__":
    load_dotenv("/home/rheaton64/.chorus/.env")
    asyncio.run(main())
