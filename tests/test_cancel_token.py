"""Tests for CancelToken — cooperative cancellation for the inner loop."""

from __future__ import annotations

import asyncio

import pytest

from spellbook.cancel_token import CancelToken


class TestCancelToken:
    def test_not_cancelled_on_construction(self) -> None:
        token = CancelToken()
        assert token.cancelled is False

    def test_cancel_sets_cancelled(self) -> None:
        token = CancelToken()
        token.cancel()
        assert token.cancelled is True

    def test_cancel_is_idempotent(self) -> None:
        token = CancelToken()
        token.cancel()
        token.cancel()  # should not raise
        assert token.cancelled is True

    def test_raise_if_cancelled_no_op_when_not_cancelled(self) -> None:
        token = CancelToken()
        token.raise_if_cancelled()  # should not raise

    def test_raise_if_cancelled_raises_when_cancelled(self) -> None:
        token = CancelToken()
        token.cancel()
        with pytest.raises(asyncio.CancelledError):
            token.raise_if_cancelled()

    @pytest.mark.asyncio
    async def test_wait_cancelled_completes_after_cancel(self) -> None:
        token = CancelToken()

        async def cancel_after_delay() -> None:
            await asyncio.sleep(0.01)
            token.cancel()

        asyncio.create_task(cancel_after_delay())
        await token.wait_cancelled()
        assert token.cancelled is True

    @pytest.mark.asyncio
    async def test_wait_cancelled_returns_immediately_if_already_cancelled(self) -> None:
        token = CancelToken()
        token.cancel()

        # Should complete immediately without timing out
        await asyncio.wait_for(token.wait_cancelled(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_multiple_waiters_all_wake_on_cancel(self) -> None:
        """Multiple coroutines waiting on the same token all resume on cancel."""
        token = CancelToken()
        resumed = []

        async def waiter(label: str) -> None:
            await token.wait_cancelled()
            resumed.append(label)

        task_a = asyncio.create_task(waiter("a"))
        task_b = asyncio.create_task(waiter("b"))
        await asyncio.sleep(0)  # let tasks start

        token.cancel()
        await asyncio.gather(task_a, task_b)
        assert set(resumed) == {"a", "b"}

    def test_independent_tokens(self) -> None:
        """Tokens are independent — cancelling one doesn't affect another."""
        t1 = CancelToken()
        t2 = CancelToken()
        t1.cancel()
        assert t1.cancelled is True
        assert t2.cancelled is False
