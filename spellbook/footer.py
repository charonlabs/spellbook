"""Footer queueing, normalization, and round-time injection.

This module owns the footer pipeline that sits between inbound footer messages,
in-memory pending footer state, transcript recording, and model-facing footer
injection.

Important invariants:

- `FooterController` owns pending footer state in memory
- pending footers are keyed and deduped last-write-wins by `key`
- inbound footer messages are normalized only when drained
- malformed footer messages should fail loudly when drained, not be silently
  repaired
- footer queueing and footer draining are explicit transcript events via
  `Recorder`
- rendered footers are injected as a system-origin `IRUserTextBlock`
- pending footer replay on resume comes from transcript rehydration, not
  inference

If you change footer behavior, keep queue semantics, transcript truth, replay,
and round-time injection coherent together.
"""

from typing import TYPE_CHECKING
from uuid import uuid4

from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import FooterSource, FooterType, IRFooter, IRUserTextBlock
from spellbook.recorder import Recorder
from spellbook.round_lifecycle import (
    ContextBlockIntegrator,
    RoundContext,
    RoundLifecycle,
)

if TYPE_CHECKING:
    from spellbook.debug_visibility import DebugEmitter


class FooterController:
    """Pending ambient-awareness payloads, drained into each next round."""

    def __init__(
        self,
        inbound_queue: InboundMessageQueue,
        recorder: Recorder,
        debug_emitter: "DebugEmitter | None" = None,
    ) -> None:
        self._inbound = inbound_queue
        self._pending: dict[str, IRFooter] = {}  # last-write-wins by key
        self._recorder = recorder
        self._debug = debug_emitter

    def bind_debug_emitter(self, debug_emitter: "DebugEmitter") -> None:
        self._debug = debug_emitter

    def _drain_inbound(self) -> None:
        msgs = self._inbound.drain_footer_messages()
        for msg in msgs:
            # fail loudly, when drained, on footer msgs that have multiple blocks or bad blocks
            if len(msg.blocks) != 1 or not isinstance(msg.blocks[0], IRUserTextBlock):
                raise ValueError(
                    f"Tried to drain malformed inbound footer message. msg={msg.model_dump_json()}"
                )
            type = msg.source_metadata.get("footer_type", "notif")
            source = msg.source_metadata.get("footer_source", "conduit")
            key = msg.source_metadata.get("footer_key", str(uuid4()))
            priority = msg.source_metadata.get("footer_priority", 50)
            self.queue_footer(
                text=msg.blocks[0].text,
                footer_type=type,
                source=source,
                key=key,
                priority=priority,
            )

    def queue_footer(
        self,
        *,
        text: str,
        footer_type: FooterType,
        source: FooterSource,
        key: str,
        priority: int = 50,
    ) -> None:
        """Queue or replace a footer. Same key → replace (dedup)."""
        f = IRFooter(
            text=text, type=footer_type, source=source, key=key, priority=priority
        )
        replaced = key in self._pending
        self._pending[key] = f
        self._recorder.queue_footer(f)
        self._debug_footer_event(
            "footer_queued",
            title=f"Footer queued: {key}",
            footers=[f],
            metadata={"key": key, "replaced": replaced},
        )

    def clear_footer(self, key: str) -> None:
        """Remove a footer from the queue by key."""
        to_clear = self._pending.get(key, None)
        if not to_clear:
            return
        del self._pending[key]
        self._recorder.drain_footers([to_clear])
        self._debug_footer_event(
            "footer_cleared",
            title=f"Footer cleared: {key}",
            footers=[to_clear],
            metadata={"key": key},
        )

    def collect_and_drain(self) -> list[IRFooter]:
        """Take all pending reminders in priority order. ALSO DRAIN THE INBOUND QUEUED FOOTERS.
        Clear the queue."""
        self._drain_inbound()
        drained = sorted(self._pending.values(), key=lambda f: f.priority)
        self._pending.clear()
        if len(drained) > 0:
            self._recorder.drain_footers(drained)
            self._debug_footer_event(
                "footers_drained",
                title=f"Footers drained: {len(drained)}",
                footers=drained,
                metadata={"count": len(drained)},
            )
        return drained

    def peek_pending(self) -> list[IRFooter]:
        """Read without clearing."""
        return sorted(self._pending.values(), key=lambda f: f.priority)

    def render_footers(self, footers: list[IRFooter]) -> str:
        body = "\n---\n".join(f.text for f in footers)
        return f"<spellbook>\n{body}\n</spellbook>"

    def rehydrate(self, footers: dict[str, IRFooter]) -> None:
        self._pending = footers

    def _debug_footer_event(
        self,
        event: str,
        *,
        title: str,
        footers: list[IRFooter],
        metadata: dict[str, object],
    ) -> None:
        if self._debug is None:
            return
        event_metadata = {
            **metadata,
            "footer_count": len(footers),
            "footer_keys": [footer.key for footer in footers],
            "footer_types": [footer.type for footer in footers],
            "footer_sources": [footer.source for footer in footers],
        }
        self._debug.debug(
            subsystem="footer",
            event=event,
            title=title,
            content=_render_footer_debug_event(title=title, footers=footers),
            plaintext=title,
            metadata=event_metadata,
        )


class FooterControllerRoundLifecycle(RoundLifecycle):
    """Weaves pending footers into the round's blocks before generate."""

    def __init__(
        self,
        controller: FooterController,
        recorder: Recorder,
        homunculus: ContextBlockIntegrator,
        debug_emitter: "DebugEmitter | None" = None,
    ):
        self._controller = controller
        self._recorder = recorder
        self._homunculus = homunculus
        self._debug = debug_emitter

    async def before_round(self, ctx: RoundContext) -> None:
        pending = self._controller.collect_and_drain()
        if not pending:
            return
        rendered = self._controller.render_footers(pending)
        self._debug_injected_footers(ctx=ctx, footers=pending, rendered=rendered)
        footer_block = IRUserTextBlock(text=rendered, origin="system")
        ctx.blocks.append(footer_block)
        ctx.blocks_this_round.append(footer_block)
        self._recorder.write_block(footer_block)
        await self._homunculus.integrate_context_blocks([footer_block])

    def _debug_injected_footers(
        self, *, ctx: RoundContext, footers: list[IRFooter], rendered: str
    ) -> None:
        if self._debug is None:
            return
        title = f"Footers injected for round {ctx.round_number}"
        self._debug.debug(
            subsystem="footer",
            event="footers_injected",
            title=title,
            content="\n".join(
                [
                    f"# {title}",
                    "",
                    "```xml",
                    rendered,
                    "```",
                ]
            ),
            plaintext=f"{title}: {len(footers)} footer(s).",
            metadata={
                "round_number": ctx.round_number,
                "footer_count": len(footers),
                "footer_keys": [footer.key for footer in footers],
                "footer_types": [footer.type for footer in footers],
                "footer_sources": [footer.source for footer in footers],
                "rendered": rendered,
            },
        )


def _render_footer_debug_event(*, title: str, footers: list[IRFooter]) -> str:
    rows = [
        f"# {title}",
        "",
        "| Key | Type | Source | Priority | Text |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for footer in footers:
        rows.append(
            "| "
            f"{_escape_table(footer.key)} | "
            f"{footer.type} | "
            f"{footer.source} | "
            f"{footer.priority} | "
            f"{_escape_table(footer.text)} |"
        )
    return "\n".join(rows)


def _escape_table(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
