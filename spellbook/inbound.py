import asyncio
from collections import deque

from .ir_types import IRInboundMessage
from .recorder import Recorder
from .round_lifecycle import RoundContext, RoundLifecycle


class InboundMessageQueue:
    def __init__(self) -> None:
        self._messages: deque[IRInboundMessage] = deque()
        self._cv = asyncio.Condition()
        self._shutdown = False

    def _is_turn_eligible(self, msg: IRInboundMessage) -> bool:
        return msg.delivery in {"turn", "inject"}

    async def put(self, msg: IRInboundMessage) -> None:
        async with self._cv:
            self._messages.append(msg)
            self._cv.notify_all()

    def push_back(self, msg: IRInboundMessage) -> None:
        self._messages.appendleft(msg)

    def has_pending_turn(self) -> bool:
        return any(self._is_turn_eligible(msg) for msg in self._messages)

    def drain_footer_messages(self) -> list[IRInboundMessage]:
        drained: list[IRInboundMessage] = []
        retained: deque[IRInboundMessage] = deque()

        while self._messages:
            msg = self._messages.popleft()
            if msg.delivery == "footer":
                drained.append(msg)
            else:
                retained.append(msg)

        self._messages = retained
        return drained

    def drain_injected_messages(self) -> list[IRInboundMessage]:
        drained: list[IRInboundMessage] = []
        retained: deque[IRInboundMessage] = deque()

        while self._messages:
            msg = self._messages.popleft()
            if msg.delivery == "inject":
                drained.append(msg)
            else:
                retained.append(msg)

        self._messages = retained
        return drained

    async def take_turn(self) -> IRInboundMessage | None:
        async with self._cv:
            while True:
                if self._shutdown:
                    return None
                retained: deque[IRInboundMessage] = deque()

                while self._messages:
                    msg = self._messages.popleft()
                    if self._is_turn_eligible(msg):
                        while retained:
                            self._messages.appendleft(retained.pop())
                        return msg
                    retained.append(msg)

                self._messages = retained
                await self._cv.wait()

    async def shutdown_queue(self) -> None:
        async with self._cv:
            self._shutdown = True
            self._cv.notify_all()


class InboundInjectionRoundLifecycle(RoundLifecycle):
    def __init__(self, *, inbound_queue: InboundMessageQueue, recorder: Recorder):
        self._inbound_queue = inbound_queue
        self._recorder = recorder

    async def before_round(self, ctx: RoundContext) -> None:
        for message in self._inbound_queue.drain_injected_messages():
            for block in message.blocks:
                ctx.blocks.append(block)
                ctx.blocks_this_round.append(block)
                self._recorder.write_block(block)
