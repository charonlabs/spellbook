import asyncio
from pathlib import Path

import rich
from dotenv import load_dotenv

from spellbook.config import SpellbookConfig
from spellbook.inbound import IRInboundMessage
from spellbook.ir_types import IRLoopResult, IRUserTextBlock
from spellbook.session_lifecycle import SessionContext, SessionLifecycle
from spellbook.session_manager import SessionManager


class PrintLifecycle(SessionLifecycle):
    def __init__(self, turn_end_event: asyncio.Event):
        self.turn_end_event = turn_end_event

    async def on_turn_ended(
        self, ctx: SessionContext, result: IRLoopResult, turn_id: str
    ) -> None:
        rich.print("**TURN END**")
        rich.print(result.blocks[-1])
        self.turn_end_event.set()

    async def on_enter_idle(self, ctx: SessionContext) -> None:
        """Manager transitioned to idle. Hearth crackles, ambient behaviors,
        idle-time subsystems start their work."""
        rich.print("**ENTERED IDLE")

    async def on_exit_idle(self, ctx: SessionContext, reason: str) -> None:
        """About to leave idle. Reason: 'message' | 'rest' | 'shutdown'."""
        rich.print("**EXITED IDLE")

    async def on_turn_started(self, ctx: SessionContext, turn_id: str) -> None:
        """About to invoke run_loop for a new turn."""
        rich.print("**TURN START")

    async def on_shutdown(self, ctx: SessionContext) -> None:
        """Shutdown requested."""
        rich.print("**SHUTDOWN")


async def main() -> None:
    config = SpellbookConfig(
        model="claude-sonnet-4-6",
        max_output_tokens=4096,
        cwd=Path.cwd(),
        system_prompt="You are an entity with access to a `Bash` tool. You and the user share the environment.",
    )
    transcript_path = Path("/tmp/spellbook_session_test.jsonl")
    if transcript_path.is_file():
        transcript_path.unlink()
    initial_msg = IRInboundMessage(
        blocks=[
            IRUserTextBlock(
                text="Hi Claude! Can you tell me what files you see in your current directory?",
                origin="human",
            )
        ],
        delivery="turn",
    )
    turn_end_event = asyncio.Event()
    session_lifecycle = PrintLifecycle(turn_end_event)
    session_manager = await SessionManager.build(
        transcript_path=transcript_path, config=config, lifecycle=session_lifecycle
    )
    asyncio.create_task(session_manager.run())
    await session_manager.submit_message(initial_msg)
    await turn_end_event.wait()
    turn_end_event.clear()
    follow_up_msg = IRInboundMessage(
        blocks=[
            IRUserTextBlock(
                text="Thanks! Can you also look in the `spellbook/core/` dir and tell me what you see?",
                origin="human",
            )
        ],
        delivery="turn",
    )
    await session_manager.submit_message(follow_up_msg)
    await turn_end_event.wait()
    await session_manager.shutdown()
    print("**DONE**")


if __name__ == "__main__":
    load_dotenv("/home/rheaton64/.chorus/.env")
    asyncio.run(main())
