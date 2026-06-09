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
    ToolResultTTLSettings,
    ToolResultTTLStatus,
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
    IRToolResultBlock,
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
            config_records=rehydrated.runtime_config_updates,
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

    def render_tool_results(
        self, *, verbose: bool = False
    ) -> tuple[str, dict[str, Any]]:
        statuses = [
            self._ttl_registry.status_for_block(
                block,
                current_turn=self._recorder.current_turn_idx,
            )
            for block in self._block_manager.context_blocks
            if isinstance(block, IRToolResultBlock)
        ]
        shown = statuses if verbose else [s for s in statuses if s.show_by_default]

        pending = sum(1 for s in statuses if s.kind == "pending")
        collapsed = sum(1 for s in statuses if s.kind == "collapsed")
        untracked = len(statuses) - pending - collapsed
        large_untracked = sum(1 for s in statuses if s.kind == "large_untracked")

        lines = ["## Tool Results", ""]
        if not statuses:
            lines.append("No tool results are currently in context.")
            return "\n".join(lines), {
                "kind": "reflect_tool_results",
                "shown": 0,
                "total": 0,
                "verbose": verbose,
            }

        lines.append(
            (
                f"Showing {len(shown)} of {len(statuses)} tool result(s) "
                f"({'verbose' if verbose else 'default'} view)."
            )
        )
        lines.append(
            (
                f"Tracked: {pending} pending, {collapsed} collapsed. "
                f"Untracked: {untracked} ({large_untracked} above threshold)."
            )
        )
        if not verbose:
            lines.append(
                "Default view shows pending TTLs and large untracked results. "
                "Pass verbose=true to inspect everything."
            )
        lines.append("")

        if not shown:
            lines.append("No token-relevant active tool results are waiting on TTL.")
        else:
            total_chars = sum(s.chars or 0 for s in shown)
            total_lines = sum(s.lines or 0 for s in shown)
            lines.append(
                (
                    f"Total shown: {total_chars:,} chars / {total_lines:,} lines "
                    f"(~{self._approx_tokens(total_chars):,} tokens)."
                )
            )
            lines.append("")
            for status in shown:
                lines.extend(self._render_tool_result_status(status))
                lines.append("")

        return "\n".join(lines).rstrip(), {
            "kind": "reflect_tool_results",
            "shown": len(shown),
            "total": len(statuses),
            "verbose": verbose,
            "pending": pending,
            "collapsed": collapsed,
            "untracked": untracked,
            "large_untracked": large_untracked,
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

    async def forget_tool_result(self, call_id: str) -> str:
        block = self._resolve_tool_result(call_id)
        state = self._ttl_registry.forget(block)
        self._invalidate()
        return (
            f"Tool result {block.call_id} successfully forgotten. "
            f"It will render as: {state.replace_content}"
        )

    def configure(
        self,
        *,
        key: str | None = None,
        value: str | int | bool | None = None,
    ) -> tuple[str, dict[str, Any]]:
        ttl_enabled: bool | None = None
        ttl_turns: int | None = None
        ttl_char_threshold: int | None = None
        if key is None and value is not None:
            raise ValueError("Configure writes require both `key` and `value`.")
        if key is not None:
            if value is None:
                raise ValueError("Configure writes require both `key` and `value`.")
            normalized_key = self._normalize_config_key(key)
            match normalized_key:
                case "enabled":
                    ttl_enabled = self._parse_bool_config_value(value, key)
                case "ttl_turns":
                    ttl_turns = self._parse_int_config_value(value, key)
                case "char_threshold":
                    ttl_char_threshold = self._parse_int_config_value(value, key)
                case _:
                    raise ValueError(f"Unknown runtime config key `{key}`.")

        old, new, updates = self._ttl_registry.configure(
            enabled=ttl_enabled,
            ttl_turns=ttl_turns,
            char_threshold=ttl_char_threshold,
        )
        if updates:
            self._recorder.write_runtime_config(
                namespace="tool_result_ttl",
                updates=updates,
                effective=new.as_record_dict(),
            )

        lines = ["## Runtime Configuration", "", "### Tool Result TTL", ""]
        if not updates:
            lines.extend(self._render_ttl_settings())
            lines.append("")
            lines.append(
                "Set `key` and `value` to update one setting. Available keys: "
                "`ttl_enabled`, `ttl_turns`, `ttl_char_threshold`."
            )
            action = "read"
        else:
            lines.append("Updated:")
            for key in updates:
                old_value = self._ttl_setting_value(old, key)
                new_value = self._ttl_setting_value(new, key)
                lines.append(
                    f"- {self._display_ttl_key(key)}: {old_value} -> {new_value}"
                )
            lines.append("")
            lines.append("Current:")
            lines.extend(self._render_ttl_settings())
            action = "update"
        return "\n".join(lines), {
            "kind": "configure",
            "action": action,
            "namespace": "tool_result_ttl",
            "updates": updates,
            "effective": new.as_record_dict(),
        }

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

    def _render_tool_result_status(self, status: ToolResultTTLStatus) -> list[str]:
        heading = f"{status.call_id} {status.tool}"
        if status.label:
            heading += f" ({status.label})"
        lines = [heading]
        if status.chars is None or status.lines is None:
            lines.append("  size: non-text output")
        else:
            lines.append(f"  size: {status.chars:,} chars / {status.lines:,} lines")
        if status.delivered_turn is None or status.age_turns is None:
            lines.append("  age: unknown")
        else:
            plural = "" if status.age_turns == 1 else "s"
            lines.append(
                (
                    f"  age: delivered turn {status.delivered_turn}, "
                    f"{status.age_turns} turn{plural} ago"
                )
            )
        lines.append(f"  status: {status.status}")
        if status.output_ref is not None:
            lines.append(f"  saved: {status.output_ref}")
        return lines

    def _approx_tokens(self, chars: int) -> int:
        return chars // 4

    def _render_ttl_settings(self) -> list[str]:
        settings = self._ttl_registry.settings
        return [
            f"- ttl_enabled: {settings.enabled}",
            f"- ttl_turns: {settings.ttl_turns}",
            f"- ttl_char_threshold: {settings.char_threshold}",
        ]

    def _ttl_setting_value(
        self, settings: ToolResultTTLSettings, key: str
    ) -> bool | int:
        match key:
            case "enabled":
                return settings.enabled
            case "ttl_turns":
                return settings.ttl_turns
            case "char_threshold":
                return settings.char_threshold
            case _:
                raise ValueError(f"Unknown tool_result_ttl config key `{key}`.")

    def _display_ttl_key(self, key: str) -> str:
        match key:
            case "enabled":
                return "ttl_enabled"
            case "ttl_turns":
                return "ttl_turns"
            case "char_threshold":
                return "ttl_char_threshold"
            case _:
                return key

    def _normalize_config_key(self, key: str) -> str:
        normalized = key.strip()
        aliases = {
            "enabled": "enabled",
            "ttl_enabled": "enabled",
            "tool_result_ttl.enabled": "enabled",
            "ttl_turns": "ttl_turns",
            "tool_result_ttl.ttl_turns": "ttl_turns",
            "ttl_char_threshold": "char_threshold",
            "char_threshold": "char_threshold",
            "tool_result_ttl.char_threshold": "char_threshold",
        }
        try:
            return aliases[normalized]
        except KeyError as e:
            raise ValueError(f"Unknown runtime config key `{key}`.") from e

    def _parse_bool_config_value(self, value: str | int | bool, key: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            match value.strip().lower():
                case "true" | "yes" | "on" | "1":
                    return True
                case "false" | "no" | "off" | "0":
                    return False
        raise ValueError(f"`{key}` expects a boolean value.")

    def _parse_int_config_value(self, value: str | int | bool, key: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"`{key}` expects a non-negative integer value.")
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, str):
            try:
                parsed = int(value.strip())
            except ValueError as e:
                raise ValueError(
                    f"`{key}` expects a non-negative integer value."
                ) from e
        else:
            raise ValueError(f"`{key}` expects a non-negative integer value.")
        if parsed < 0:
            raise ValueError(f"`{key}` expects a non-negative integer value.")
        return parsed

    def _resolve_tool_result(self, call_id: str) -> IRToolResultBlock:
        results = [
            block
            for block in self._block_manager.context_blocks
            if isinstance(block, IRToolResultBlock)
        ]
        exact = [block for block in results if block.call_id == call_id]
        if exact:
            return exact[0]
        matches = [block for block in results if block.call_id.startswith(call_id)]
        if not matches:
            raise ValueError(f"No tool result found for call_id `{call_id}`.")
        if len(matches) > 1:
            options = ", ".join(block.call_id for block in matches[:8])
            if len(matches) > 8:
                options += ", ..."
            raise ValueError(
                f"Tool result call_id `{call_id}` is ambiguous. Matches: {options}"
            )
        return matches[0]

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
