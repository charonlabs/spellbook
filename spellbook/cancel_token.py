"""Cooperative cancellation for the inner loop.

A ``CancelToken`` threads through the Generator, Executor, and tool
dispatch. When ``cancel()`` fires, every subsystem that awaits or
checkpoints on the token stops at its next opportunity. Cancellation
is authoritative — the loop exits with ``stop_reason="cancelled"`` and
in-flight tools are killed rather than awaited.

A single token is created per ``run_loop`` invocation.
"""

import asyncio


class CancelToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Request cancellation. Idempotent."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    async def wait_cancelled(self) -> None:
        """Suspend until cancel() is called."""
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        """Raise CancelledError if cancelled. For synchronous check points."""
        if self.cancelled:
            raise asyncio.CancelledError
