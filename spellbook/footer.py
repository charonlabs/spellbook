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

from uuid import uuid4

from spellbook.inbound import InboundMessageQueue
from spellbook.ir_types import FooterSource, FooterType, IRFooter, IRUserTextBlock
from spellbook.recorder import Recorder
from spellbook.round_lifecycle import RoundContext, RoundLifecycle


class FooterController:
    """Pending ambient-awareness payloads, drained into each next round."""

    def __init__(self, inbound_queue: InboundMessageQueue, recorder: Recorder) -> None:
        self._inbound = inbound_queue
        self._pending: dict[str, IRFooter] = {}  # last-write-wins by key
        self._recorder = recorder

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
        self._pending[key] = f
        self._recorder.queue_footer(f)

    def clear_footer(self, key: str) -> None:
        """Remove a footer from the queue by key."""
        to_clear = self._pending.get(key, None)
        if not to_clear:
            return
        del self._pending[key]
        self._recorder.drain_footers([to_clear])

    def collect_and_drain(self) -> list[IRFooter]:
        """Take all pending reminders in priority order. ALSO DRAIN THE INBOUND QUEUED FOOTERS.
        Clear the queue."""
        self._drain_inbound()
        drained = sorted(self._pending.values(), key=lambda f: f.priority)
        self._pending.clear()
        if len(drained) > 0:
            self._recorder.drain_footers(drained)
        return drained

    def peek_pending(self) -> list[IRFooter]:
        """Read without clearing."""
        return sorted(self._pending.values(), key=lambda f: f.priority)

    def render_footers(self, footers: list[IRFooter]) -> str:
        body = "\n---\n".join(f.text for f in footers)
        return f"<spellbook>\n{body}\n</spellbook>"

    def rehydrate(self, footers: dict[str, IRFooter]) -> None:
        self._pending = footers


class FooterControllerRoundLifecycle(RoundLifecycle):
    """Weaves pending footers into the round's blocks before generate."""

    def __init__(self, controller: FooterController, recorder: Recorder):
        self._controller = controller
        self._recorder = recorder

    async def before_round(self, ctx: RoundContext) -> None:
        pending = self._controller.collect_and_drain()
        if not pending:
            return
        footer_block = IRUserTextBlock(
            text=self._controller.render_footers(pending), origin="system"
        )
        ctx.blocks.append(footer_block)
        self._recorder.write_block(footer_block)
