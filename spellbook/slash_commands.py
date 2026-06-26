from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from spellbook.config import SpellbookConfig
from spellbook.homunculus import Homunculus
from spellbook.homunculus.common import render_plan
from spellbook.ir_types import IRInboundMessage, IRUserTextBlock
from spellbook.recorder import Recorder
from spellbook.system_response import SystemResponse

SlashCommandFn = Callable[[str], Awaitable[SystemResponse]]


class SlashCommandSession(Protocol):
    config: SpellbookConfig
    homunculus: Homunculus
    recorder: Recorder
    state: Any


@dataclass(frozen=True)
class SlashCommandSpec:
    name: str
    description: str
    usage: str
    handler: SlashCommandFn


def parse_slash_command_message(
    message: IRInboundMessage,
) -> tuple[str, str] | None:
    if message.delivery != "turn" or not message.blocks:
        return None
    first_block = message.blocks[0]
    if not isinstance(first_block, IRUserTextBlock) or first_block.origin != "human":
        return None
    text = first_block.text.lstrip()
    if not text.startswith("/"):
        return None
    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return command, args


class SlashCommandHandler:
    def __init__(self, session: SlashCommandSession):
        self._session = session
        self._commands: dict[str, SlashCommandSpec] = {}
        self._register_defaults()

    def register(
        self,
        name: str,
        handler: SlashCommandFn,
        *,
        description: str,
        usage: str | None = None,
    ) -> None:
        command = _normalize_command_name(name)
        self._commands[command] = SlashCommandSpec(
            name=command,
            description=description,
            usage=usage or command,
            handler=handler,
        )

    async def handle(self, command: str, args: str) -> SystemResponse:
        command = _normalize_command_name(command)
        spec = self._commands.get(command)
        if spec is None:
            return SystemResponse(
                command=command,
                content=(
                    f"Unknown command: `{_escape_inline(command)}`\n\n"
                    "Run `/help` to list available commands."
                ),
                plaintext=f"Unknown command: {command}. Run /help.",
                metadata={"known_commands": sorted(self._commands)},
            )
        return await spec.handler(args)

    def _register_defaults(self) -> None:
        self.register(
            "/status",
            self._status,
            description="Show current runtime, model, context, and block status.",
            usage="/status",
        )
        self.register(
            "/reflect",
            self._reflect,
            description="Show a compact view of semantic blocks and planner state.",
            usage="/reflect",
        )
        self.register(
            "/help",
            self._help,
            description="List available slash commands.",
            usage="/help",
        )

    async def _status(self, args: str) -> SystemResponse:
        awareness = self._session.homunculus.build_awareness()
        budget = awareness.budget
        block_count = len(awareness.semantic_blocks)
        tail_size = _tail_size(awareness.tail.tail_start, awareness.tail.tail_end)
        input_tokens = _format_tokens(budget.current_input_tokens)
        slack_tokens = _format_tokens(budget.current_slack_tokens)
        content = "\n".join(
            [
                "# Status",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Model | `{_escape_table(self._session.config.model)}` |",
                f"| Runtime | `{_escape_table(self._session.state)}` |",
                f"| Turns | `{self._session.recorder.current_turn_idx}` |",
                f"| Context tokens | `{input_tokens}` |",
                f"| Slack tokens | `{slack_tokens}` |",
                f"| Regime | `{budget.regime}` |",
                f"| Semantic blocks | `{block_count}` |",
                f"| Tail blocks | `{tail_size}` |",
            ]
        )
        plaintext = (
            f"Status: {self._session.config.model}, {input_tokens} context tokens, "
            f"{budget.regime}, {block_count} blocks, {tail_size} tail blocks."
        )
        return SystemResponse(
            command="/status",
            content=content,
            plaintext=plaintext,
            metadata={
                "model": self._session.config.model,
                "state": self._session.state,
                "turns": self._session.recorder.current_turn_idx,
                "context_tokens": budget.current_input_tokens,
                "slack_tokens": budget.current_slack_tokens,
                "regime": budget.regime,
                "semantic_blocks": block_count,
                "tail_blocks": tail_size,
            },
        )

    async def _reflect(self, args: str) -> SystemResponse:
        awareness = self._session.homunculus.build_awareness()
        budget = awareness.budget
        proposal = awareness.plan_proposal
        rows = [
            "# Reflect",
            "",
            (
                f"Regime: `{budget.regime}`. "
                f"Context tokens: `{_format_tokens(budget.current_input_tokens)}`."
            ),
            "",
        ]
        if proposal is not None:
            rows.extend(
                [
                    "## Planner",
                    "",
                    render_plan(proposal, awareness.semantic_blocks),
                    "",
                ]
            )

        rows.extend(
            [
                "## Blocks",
                "",
                "| Block | Title | Fidelity | Range | Tokens | Pins |",
                "| ---: | --- | --- | ---: | ---: | --- |",
            ]
        )
        for block in awareness.semantic_blocks:
            pins = _block_pin_summary(block.pin is not None, len(block.facet_pins))
            rows.append(
                "| "
                f"{block.idx} | "
                f"{_escape_table(block.title)} | "
                f"{_escape_table(block.mode)} | "
                f"{block.range.start_block}-{block.range.end_block} | "
                f"{_format_tokens(block.toks.tokens if block.toks else None)} | "
                f"{_escape_table(pins)} |"
            )
        if not awareness.semantic_blocks:
            rows.append("| - | No semantic blocks yet | - | - | - | - |")

        proposal_text = "proposal pending" if proposal is not None else "no proposal"
        plaintext = (
            f"Reflect: {len(awareness.semantic_blocks)} blocks, "
            f"{budget.regime}, {proposal_text}."
        )
        return SystemResponse(
            command="/reflect",
            content="\n".join(rows),
            plaintext=plaintext,
            metadata={
                "block_count": len(awareness.semantic_blocks),
                "regime": budget.regime,
                "proposal_pending": proposal is not None,
            },
        )

    async def _help(self, args: str) -> SystemResponse:
        rows = [
            "# Slash Commands",
            "",
            "| Command | Description | Usage |",
            "| --- | --- | --- |",
        ]
        lines: list[str] = []
        for spec in sorted(self._commands.values(), key=lambda item: item.name):
            rows.append(
                f"| `{spec.name}` | {_escape_table(spec.description)} | "
                f"`{_escape_table(spec.usage)}` |"
            )
            lines.append(f"{spec.name}: {spec.description}")
        return SystemResponse(
            command="/help",
            content="\n".join(rows),
            plaintext="\n".join(lines),
            metadata={
                "commands": [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "usage": spec.usage,
                    }
                    for spec in sorted(
                        self._commands.values(), key=lambda item: item.name
                    )
                ]
            },
        )


def _normalize_command_name(name: str) -> str:
    stripped = name.strip().lower()
    if not stripped.startswith("/"):
        stripped = f"/{stripped}"
    return stripped


def _format_tokens(tokens: int | None) -> str:
    if tokens is None:
        return "unknown"
    return f"{tokens:,}"


def _tail_size(tail_start: int, tail_end: int) -> int:
    if tail_end < tail_start:
        return 0
    return tail_end - tail_start + 1


def _block_pin_summary(block_pinned: bool, facet_pins: int) -> str:
    parts: list[str] = []
    if block_pinned:
        parts.append("block")
    if facet_pins:
        parts.append(f"{facet_pins} facet")
    return ", ".join(parts) if parts else "-"


def _escape_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _escape_inline(value: object) -> str:
    return str(value).replace("`", "\\`")
