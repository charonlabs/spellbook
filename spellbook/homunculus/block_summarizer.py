from spellbook.config import HomunculusConfig
from spellbook.fork import (
    BlockSummarizerConfig,
    ForkRunner,
    PreparedFork,
)
from spellbook.homunculus.common import render_context_block, render_summary
from spellbook.ir_types import (
    IRBlock,
    IRSemanticBlock,
    IRUserTextBlock,
)
from spellbook.recorder import Recorder


class BlockSummarizer:
    def __init__(
        self, *, config: HomunculusConfig, fork_runner: ForkRunner, recorder: Recorder
    ):
        self._config = config
        self._fork_runner = fork_runner
        self._recorder = recorder

    def integrate_result(self, fork_id: str) -> None:
        self._fork_runner.integrate_result(fork_id)

    async def summarize(
        self,
        *,
        semantic_block: IRSemanticBlock,
        context_block_slice: list[IRBlock],
        prev_semantic_blocks: list[IRSemanticBlock],
    ) -> PreparedFork:
        text = "# Previous summaries\n\n"
        for s in prev_semantic_blocks:
            text += (
                render_summary(
                    s,
                ).text
                + "\n\n"
            )
        text += f'# Current block to summarize (title="{semantic_block.title}"'
        curr = semantic_block.range.start_block
        for cb in context_block_slice:
            text += render_context_block(cb, curr) + "\n"
            curr += 1

        fc = BlockSummarizerConfig(
            inbound_block=IRUserTextBlock(text=text, origin="system")
        )
        return await self._fork_runner.run_fork(fork_config=fc)
