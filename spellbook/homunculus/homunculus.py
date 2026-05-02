from typing import Any, Sequence
from uuid import uuid4

from spellbook.backends.model_backend import TokenCounter
from spellbook.config import HomunculusConfig
from spellbook.footer import FooterController
from spellbook.fork import ForkConfig, ForkRunner
from spellbook.homunculus.block_manager import BlockManager
from spellbook.homunculus.common import (
    AwarenessBudgetSnapshot,
    AwarenessHomunculusSnapshot,
    AwarenessTailSnapshot,
    render_intent,
    render_plan,
)
from spellbook.homunculus.gas_gauge import GasGauge
from spellbook.homunculus.planner import Planner
from spellbook.homunculus.token_meter import TokenMeter
from spellbook.homunculus.tool_result_ttl import (
    TTL_TRIGGER_END_TURN,
    TTL_TRIGGER_SEQ,
    ToolResultTTLRegistry,
)
from spellbook.nursery import Nursery
from spellbook.recorder import Recorder
from spellbook.round_lifecycle import RoundContext, RoundLifecycle

from ..ir_types import (
    IRBlock,
    IRCompactBlockIntent,
    IRExecution,
    IRGeneration,
    SemanticBlockApplyModeSource,
    StopReason,
)
from ..rehydrator import RehydrationResult


class Homunculus:
    def __init__(
        self,
        *,
        config: HomunculusConfig,
        footer_c: FooterController,
        recorder: Recorder,
        token_counter: TokenCounter,
        nursery: Nursery,
        fork_runner: ForkRunner,
        fork_config: ForkConfig | None = None,
    ):
        self._config = config
        self._footer_c = footer_c
        self._gas_gauge = GasGauge(config=config, footer_c=footer_c)
        self._recorder = recorder  # this is here cuz we'll need it later
        self._token_meter = TokenMeter(config=config, tok_counter=token_counter)
        self._ttl_registry = ToolResultTTLRegistry(config=config, recorder=recorder)
        self._nursery = nursery
        self._planner = Planner(config=config)
        self._fork_runner = fork_runner
        self._block_manager = BlockManager(
            config=config,
            fork_runner=fork_runner,
            footer_c=footer_c,
            nursery=nursery,
            recorder=recorder,
            token_meter=self._token_meter,
        )
        self._fork_config = fork_config
        self._should_rerender: bool = False

    async def rehydrate(self, rehydrated: RehydrationResult) -> None:
        self._block_manager.context_blocks = rehydrated.blocks
        self._block_manager.next_block_id = len(self._block_manager.context_blocks)
        self._block_manager.rehydrate(rehydrated)
        self._planner.rehydrate(rehydrated)
        self._ttl_registry.rehydrate(
            rehydrated.tool_result_ttls,
            last_completed_turn=rehydrated.last_completed_turn,
        )

    def build_awareness(self) -> AwarenessHomunculusSnapshot:
        input_tokens = self._gas_gauge.input_tokens
        budget = AwarenessBudgetSnapshot(
            max_tokens=self._config.max_tokens,
            reserve_output_tokens=self._config.max_tokens - self._config.hard_threshold,
            current_input_tokens=input_tokens,
            current_slack_tokens=(self._config.max_tokens - input_tokens)
            if input_tokens is not None
            else None,
            regime=self._gas_gauge.regime,
            warning_threshold=self._config.soft_threshold,
            forced_threshold=self._config.medium_threshold,
            critical_threshold=self._config.hard_threshold,
        )
        tail_end = len(self._block_manager.context_blocks) - 1
        if len(self._block_manager.semantic_blocks) == 0:
            tail_start = 0
        else:
            tail_start = self._block_manager.semantic_blocks[-1].range.end_block + 1
        tail = AwarenessTailSnapshot(
            tail_start=tail_start,
            tail_end=tail_end,
            toks=None,
        )
        return AwarenessHomunculusSnapshot(
            budget=budget,
            semantic_blocks=self._block_manager.semantic_blocks,
            proposed_blocks=self._block_manager.proposed_semantic_blocks,
            tail=tail,
            plan_proposal=self._planner.proposal,
        )

    async def render_context(self, new_blocks: Sequence[IRBlock]) -> list[IRBlock]:
        if len(new_blocks) > 0:
            await self._block_manager.append_context_blocks(new_blocks)
        context: list[IRBlock] = []
        for b in self._block_manager.semantic_blocks:
            context.extend(self._block_manager.render_block(semantic_block=b))

        context.extend(self._block_manager.render_tail())
        return self._ttl_registry.collapse_blocks(context)

    async def render_reflect(
        self, block_idx: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        """Render the `Reflect` tool output text.
        Returns a tuple of (rendered, display_metadata)"""
        if block_idx is not None:
            return self._block_manager.render_summary_preview(block_idx), {
                "kind": "reflect",
                "target_block": block_idx,
            }

        frame_token_count = await self._token_meter.frame_tokens()
        result = "# Your Context\n\n"
        curr_input_toks = self._gas_gauge.input_tokens
        if curr_input_toks is None:
            result += "Currently calculating rendered count... you should see the updated count in your gas gauge shortly.\n"
            token_summary = "unknown count"
        else:
            result += f"Currently rendered: {curr_input_toks // 1000}K / 1M - {self._gas_gauge.regime}.\n"
            token_summary = f"{curr_input_toks // 1000}K {self._gas_gauge.regime}"

        result += f"Frame: {self._format_token_count(frame_token_count)}.\n"
        result += "\n"
        proposal = self._planner.proposal
        if proposal is not None:
            result += "## Planner\n\n"
            result += "Pending proposal:\n"
            result += render_plan(
                proposal,
                self._block_manager.semantic_blocks,
            )
            result += "\n\n"

        block_displays: list[str] = []
        next_block_start = 0
        for b in self._block_manager.semantic_blocks:
            block_displays.append(
                (
                    f'[Block {b.idx}]: "{b.title}" ({b.range.start_block} - '
                    f"{b.range.end_block}) - toks: "
                    f"~{self._format_token_count(b.toks.tokens if b.toks else None)}"
                )
            )
            if b.pin is not None:
                block_displays.append(f"- pinned: {b.pin.reason}")
            summary = next((a for a in b.artifacts if a.type == "summary"), None)
            if summary is not None and summary.facets:
                facet_pins = {
                    pin.facet_id: pin
                    for pin in b.facet_pins
                    if pin.facet_id is not None
                }
                for facet in summary.facets:
                    line = (
                        f"- facet {facet.id}: {facet.title} "
                        f"({facet.start_block}-{facet.end_block})"
                    )
                    pin = facet_pins.get(facet.id)
                    if pin is not None:
                        line += f" [pinned: {pin.reason}]"
                    block_displays.append(line)
            next_block_start = b.range.end_block + 1
        if block_displays:
            result += "## Blocks\n\n"
            result += (
                'Format: [Block <idx>]: "<title>" '
                "(<context block range>) - toks: ~<num tokens>\n"
            )
            result += "\n".join(block_displays)
            result += "\n"
        else:
            result += "No blocks detected yet.\n"
        result += "\n"
        tail_count = await self._token_meter.count_range(
            self._block_manager.context_blocks,
            next_block_start,
            len(self._block_manager.context_blocks),
        )
        result += "## Context Tail (not blocked yet)\n\n"
        result += (
            f"{len(self._block_manager.context_blocks[next_block_start:])} content blocks: "
            f"{self._format_token_count(tail_count.tokens if tail_count else None)}."
        )

        return result, {
            "kind": "reflect",
            "block_count": len(self._block_manager.semantic_blocks),
            "token_summary": token_summary,
        }

    async def forget(
        self,
        block_idx: int,
        confirm: bool = False,
        source: SemanticBlockApplyModeSource = "model",
    ) -> None:
        self._block_manager.forget_block(block_idx, confirm, source)
        self._invalidate()

    async def pin(
        self, block_idx: int, reason: str, facet_id: str | None = None
    ) -> None:
        # returns True when the pin invalidates the prefix
        should_invalidate = (
            self._block_manager.pin_facet(block_idx, facet_id, reason)
            if facet_id is not None
            else self._block_manager.pin_block(block_idx, reason)
        )
        if should_invalidate:
            self._invalidate()
        else:
            self._planner.invalidate()

    async def recall(self, block_idx: int) -> str:
        return self._block_manager.recall_block(block_idx)

    async def integrate_generation(self, generation: IRGeneration) -> None:
        """Absorb a generation's output"""
        await self._block_manager.append_context_blocks(
            generation.blocks, usage=generation.usage
        )
        if generation.usage is not None:
            self._gas_gauge.observe(generation.usage.total_input_tokens)

    async def integrate_execution(self, execution: IRExecution) -> None:
        """Absorb an execution's output"""
        await self._block_manager.append_context_blocks(execution.blocks)
        self._ttl_registry.observe_execution(execution)

    async def maybe_rerender(self) -> list[IRBlock] | None:
        if not self._should_rerender:
            return None
        self._should_rerender = False
        return await self.render_context([])

    async def check_nursery(self) -> None:
        await self._block_manager.check_nursery()

    async def check_planner(self) -> None:
        if self._gas_gauge.input_tokens is None:
            return  # invalid, wait for next round
        result = self._planner.plan(
            self._block_manager.semantic_blocks, self._gas_gauge.input_tokens
        )
        if result is None:
            return  # no plan updates
        update_msgs: list[str] = []
        match result.kind:
            case "proposal":
                self._recorder.propose_plan(result.plan)
                for intent in result.plan.intents:
                    match intent:
                        case IRCompactBlockIntent():
                            until_medium = (
                                self._config.medium_threshold
                                - self._gas_gauge.input_tokens
                            )
                            update_msgs.append(
                                (
                                    f"new proposal - {render_intent(intent, self._block_manager.semantic_blocks)} "
                                    f"after another {until_medium} toks."
                                )
                            )
                        case _:
                            raise NotImplementedError(
                                f"`check_planner` does not yet support a proposed intent of type {type(intent)}"
                            )
            case "action":
                for intent in result.plan.intents:
                    match intent:
                        case IRCompactBlockIntent():
                            await self.forget(intent.block_idx, source="planner")
                            update_msgs.append(
                                (
                                    render_intent(
                                        intent,
                                        self._block_manager.semantic_blocks,
                                        verb="compacted",
                                    )
                                    + "."
                                )
                            )
                        case _:
                            raise NotImplementedError(
                                f"`check_planner` does not yet support an intent of type {type(intent)}"
                            )
        if len(update_msgs) > 0:
            footer_text = "Planner:\n" + "\n".join(update_msgs)
            self._footer_c.queue_footer(
                text=footer_text,
                footer_type="compaction",
                source="planner",
                key=f"compaction_{uuid4().hex}",
            )

    def tick_round_ttls(self) -> None:
        if self._ttl_registry.tick(TTL_TRIGGER_SEQ):
            self._invalidate()

    def tick_end_turn_ttls(self) -> None:
        if self._ttl_registry.tick(TTL_TRIGGER_END_TURN):
            self._invalidate_counts()

    def _format_token_count(self, count: int | None) -> str:
        if count is None:
            return "unknown tokens"
        return f"{count} tokens"

    def _invalidate(self) -> None:
        self._should_rerender = True
        self._invalidate_counts()

    def _invalidate_counts(self) -> None:
        self._token_meter.invalidate()
        self._gas_gauge.invalidate()
        self._planner.invalidate()


class HomunculusRoundLifecycle(RoundLifecycle):
    def __init__(self, homunculus: Homunculus):
        self._homunculus = homunculus

    async def after_generate(
        self,
        ctx: RoundContext,
        generation: IRGeneration,
    ) -> None:
        await self._homunculus.integrate_generation(generation)

    async def after_execute(
        self,
        ctx: RoundContext,
        execution: IRExecution,
    ) -> None:
        await self._homunculus.integrate_execution(execution)

    async def between_rounds(self, ctx: RoundContext) -> None:
        await self._homunculus.check_nursery()
        self._homunculus.tick_round_ttls()
        await self._homunculus.check_planner()
        maybe_new_blocks = await self._homunculus.maybe_rerender()
        if maybe_new_blocks is not None:
            ctx.blocks = maybe_new_blocks

    async def on_loop_exit(self, ctx: RoundContext, stop_reason: StopReason) -> None:
        await self._homunculus.check_nursery()
        if stop_reason == "end_turn":
            self._homunculus.tick_end_turn_ttls()
