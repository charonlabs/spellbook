"""Live event bus for the core app server.

The event bus is live-only transport infrastructure. It is not durable catchup
truth. Reconnect/catchup should be built from transcript + current runtime state,
then live events should drain from a subscription queue created before catchup.

Core invariant: publishing an event must never block the session loop on a slow
WebSocket/client.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import uuid4

from spellbook.app.protocol import RecordWrittenEvent, ServerEvent
from spellbook.ir_types import IRRecord

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CloseSentinel:
    """Private queue sentinel used to unblock subscription iterators."""


_CLOSE_SENTINEL = _CloseSentinel()


@dataclass(eq=False, slots=True)
class AppEventSubscription:
    """One live event subscription.

    Each WebSocket/client gets its own subscription and bounded queue. Consumers
    should normally use `async for event in subscription` rather than reading the
    queue directly, because the queue also carries a private close sentinel.
    """

    id: str
    queue: asyncio.Queue[ServerEvent | _CloseSentinel]
    _bus: AppEventBus = field(repr=False, compare=False)
    closed: bool = False

    def __aiter__(self) -> AppEventSubscription:
        return self

    async def __anext__(self) -> ServerEvent:
        item = await self.queue.get()
        if isinstance(item, _CloseSentinel):
            raise StopAsyncIteration
        return item

    def close(self) -> None:
        """Close this subscription and unregister it from the bus.

        This is cancellation-safe from the consumer side: a websocket handler can
        call it in `finally` without caring whether the bus already closed the
        subscription as stale.
        """

        self._bus.unsubscribe(self)


class AppEventBus:
    """Single-loop live event fanout for app-server clients.

    The bus has per-subscriber queues. Publish never awaits client consumption:
    it uses `put_nowait()`, and a full queue means the subscriber is stale and
    should be disconnected.
    """

    def __init__(self, *, max_q_size: int = 1024):
        if max_q_size < 1:
            raise ValueError("`max_q_size` must be >= 1")
        self._max_q_size = max_q_size
        self._subs: set[AppEventSubscription] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False

    def publish(self, event: ServerEvent) -> None:
        """Publish an event to all current subscribers.

        Publishing must happen on the bus event loop. It never awaits websocket
        clients or queue capacity; slow subscribers are closed.
        """

        self._bind_running_loop()
        self._publish(event)

    def subscribe(self) -> AppEventSubscription:
        """Create and register a new live subscription.

        Call this before building/sending catchup so events published during
        catchup are queued for this client.
        """
        self._bind_running_loop()

        if self._closed:
            raise RuntimeError("Cannot subscribe to a closed AppEventBus")

        subscription = AppEventSubscription(
            id=f"sub_{uuid4().hex}",
            queue=asyncio.Queue(maxsize=self._max_q_size),
            _bus=self,
        )
        self._subs.add(subscription)
        logger.info(
            "event_bus.subscribed sub=%s total=%s max_q=%s",
            subscription.id,
            len(self._subs),
            self._max_q_size,
        )
        return subscription

    def unsubscribe(self, subscription: AppEventSubscription) -> None:
        """Unregister and close a subscription."""

        self._bind_running_loop()
        self._close_subscription(subscription, reason="unsubscribe")

    def close(self) -> None:
        """Close the bus and all live subscriptions."""

        self._bind_running_loop()
        self._closed = True
        logger.info("event_bus.closing subscribers=%s", len(self._subs))
        for sub in list(self._subs):
            self._close_subscription(sub, reason="bus_close")
        logger.info("event_bus.closed")

    def record_tap(self, record: IRRecord) -> None:
        """Recorder tap adapter.

        This is intentionally sync so it can be passed directly to `Recorder`.
        """
        self.publish(RecordWrittenEvent(record=record))

    @property
    def sub_count(self) -> int:
        return len(self._subs)

    @property
    def closed(self) -> bool:
        return self._closed

    def _publish(self, event: ServerEvent) -> None:
        """Deliver an event without awaiting subscribers.

        State mutations are safe because every public method is bound to one
        event loop and this method contains no await points.
        """

        if self._closed:
            return

        event_kind = getattr(event, "kind", type(event).__name__)
        for subscription in list(self._subs):
            if subscription.closed:
                self._subs.discard(subscription)
                logger.debug(
                    "event_bus.discard_closed sub=%s total=%s",
                    subscription.id,
                    len(self._subs),
                )
                continue
            try:
                subscription.queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "event_bus.stale_subscriber sub=%s event=%s qsize=%s max_q=%s total=%s",
                    subscription.id,
                    event_kind,
                    subscription.queue.qsize(),
                    self._max_q_size,
                    len(self._subs),
                )
                self._close_subscription(subscription, reason="stale_queue_full")

    def _close_subscription(
        self,
        subscription: AppEventSubscription,
        *,
        reason: str,
    ) -> None:
        """Mark a subscription closed, unregister it, and unblock its iterator."""
        if subscription.closed:
            logger.debug(
                "event_bus.close_ignored sub=%s reason=%s total=%s",
                subscription.id,
                reason,
                len(self._subs),
            )
            return

        qsize = subscription.queue.qsize()
        subscription.closed = True
        self._subs.discard(subscription)
        self._signal_closed(subscription)
        logger.info(
            "event_bus.unsubscribed sub=%s reason=%s qsize=%s total=%s",
            subscription.id,
            reason,
            qsize,
            len(self._subs),
        )

    def _signal_closed(self, subscription: AppEventSubscription) -> None:
        """Put a close sentinel into the subscription queue.

        If the queue is full, the subscriber is stale anyway, so discard queued
        events until the sentinel fits. This avoids leaving websocket loops stuck
        forever awaiting `queue.get()`.
        """

        discarded = 0
        while True:
            try:
                subscription.queue.put_nowait(_CLOSE_SENTINEL)
                if discarded:
                    logger.warning(
                        "event_bus.discarded_events_for_close sub=%s discarded=%s",
                        subscription.id,
                        discarded,
                    )
                return
            except asyncio.QueueFull:
                try:
                    subscription.queue.get_nowait()
                    discarded += 1
                except asyncio.QueueEmpty:
                    return

    def _bind_running_loop(self) -> None:
        loop = asyncio.get_running_loop()

        if self._loop is None:
            self._loop = loop
            return

        if self._loop is not loop:
            raise RuntimeError("AppEventBus async methods must run on one event loop.")
