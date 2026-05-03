"""Runtime owner for the core app server.

`CoreAppRuntime` is the app-server control plane around one long-lived
`SessionManager`. It owns live event fanout, starts the session loop, accepts
inbound messages, and builds simple transcript-backed snapshots.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spellbook.app.conduits import (
    conduit_key,
    conduit_priority,
    format_conduit_footer,
    frame_conduit,
    surface_for_inbound,
)
from spellbook.app.event_bus import AppEventBus
from spellbook.app.lifecycle import AppRoundLifecycle, AppSessionLifecycle
from spellbook.app.protocol import (
    AwarenessResponse,
    AwarenessSnapshot,
    CatchupResponse,
    ConduitResponse,
    HealthResponse,
    MessageQueuedEvent,
    SubmitMessageResponse,
    session_to_runtime_state,
)
from spellbook.config import SpellbookConfig
from spellbook.ir_types import IRInboundMessage, IRUserTextBlock
from spellbook.rehydrator import Rehydrator
from spellbook.session_lifecycle import SessionContext
from spellbook.session_manager import SessionBuilder, SessionManager

logger = logging.getLogger(__name__)


class CoreAppRuntime:
    """Own one core session and expose app-server operations."""

    def __init__(
        self,
        *,
        transcript_path: Path,
        config: SpellbookConfig | None = None,
        session_builder: SessionBuilder = SessionManager.build,
        bus: AppEventBus | None = None,
    ) -> None:
        self.transcript_path = transcript_path
        self.config = config
        self.bus = bus or AppEventBus()
        self._session_builder = session_builder
        self._session: SessionManager | None = None
        self._session_task: asyncio.Task[None] | None = None
        self._command_lock = asyncio.Lock()
        self._last_active_surface: str | None = None
        self._last_surface_time: datetime | None = None
        self._last_reported_surface: str | None = None

    @property
    def session(self) -> SessionManager | None:
        return self._session

    @property
    def running(self) -> bool:
        task = self._session_task
        return task is not None and not task.done()

    async def startup(self) -> None:
        """Build and start the owned session loop."""
        if self._session is not None:
            raise RuntimeError("CoreAppRuntime has already been started.")

        session = await self._session_builder(
            transcript_path=self.transcript_path,
            config=self.config,
            lifecycle=AppSessionLifecycle(
                self.bus,
                before_turn_started=self._before_turn_started,
            ),
            pre_round_lifecycle=AppRoundLifecycle(self.bus),
            record_tap=self.bus.record_tap,
        )
        self._session = session
        self._session_task = asyncio.create_task(session.run())
        self._session_task.add_done_callback(self._on_session_task_done)
        await asyncio.sleep(0)

    def build_awareness(self) -> AwarenessResponse:
        hom = self._require_session().homunculus.build_awareness()
        nursery = self._require_session().nursery.build_awareness()
        return AwarenessResponse(
            snapshot=AwarenessSnapshot(
                homunculus=hom,
                nursery=nursery,
                surface=self._last_active_surface,
                surface_time=self._last_surface_time,
            )
        )

    async def submit_message(self, message: IRInboundMessage) -> SubmitMessageResponse:
        """Submit a turn or injection message to the owned session.

        `started=True` means the message should become the next active turn.
        `queued=True` means the message was queued behind active work or queued input.
        """
        if message.delivery == "footer":
            raise ValueError("`submit_message` only accepts turn/inject messages.")

        async with self._command_lock:
            return await self._submit_message_unlocked(message)

    async def handle_conduit(
        self,
        *,
        conduit_type: str,
        source: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConduitResponse:
        """Route an inbound conduit based on type and runtime state."""
        metadata_keys = sorted(metadata.keys()) if metadata is not None else None
        logger.info(
            "runtime.conduit.start type=%s source=%s content_chars=%s metadata_keys=%s",
            conduit_type,
            source,
            len(content),
            metadata_keys,
        )
        async with self._command_lock:
            if conduit_type == "context":
                logger.info(
                    "runtime.conduit.route type=%s source=%s action=%s",
                    conduit_type,
                    source,
                    "queued_as_context",
                )
                await self._queue_conduit_footer(
                    conduit_type=conduit_type,
                    source=source,
                    content=content,
                    metadata=metadata,
                )
                response = ConduitResponse(
                    delivered=True,
                    action="queued_as_context",
                    source=source,
                )
                logger.info(
                    "runtime.conduit.complete type=%s source=%s action=%s delivered=%s",
                    conduit_type,
                    source,
                    response.action,
                    response.delivered,
                )
                return response

            if conduit_type == "message":
                logger.info(
                    "runtime.conduit.route type=%s source=%s action=%s",
                    conduit_type,
                    source,
                    "submit_inject",
                )
                response = await self._submit_message_unlocked(
                    IRInboundMessage(
                        blocks=[IRUserTextBlock(text=content, origin="conduit")],
                        source_metadata={
                            "source": source,
                            "origin": "conduit",
                            "conduit_type": conduit_type,
                            "metadata": dict(metadata or {}),
                        },
                        delivery="inject",
                    )
                )
                conduit_response = ConduitResponse(
                    delivered=True,
                    action="queued_as_message" if response.queued else "started_turn",
                    source=source,
                )
                logger.info(
                    "runtime.conduit.complete type=%s source=%s action=%s delivered=%s started=%s queued=%s",
                    conduit_type,
                    source,
                    conduit_response.action,
                    conduit_response.delivered,
                    response.started,
                    response.queued,
                )
                return conduit_response

            if conduit_type == "notification":
                session = self._require_session()
                pending_turn = session.inbound_queue.has_pending_turn()
                should_wake = session.state == "idle" and not pending_turn
                logger.info(
                    "runtime.conduit.notification_decision source=%s state=%s pending_turn=%s should_wake=%s",
                    source,
                    session.state,
                    pending_turn,
                    should_wake,
                )
                if should_wake:
                    logger.info(
                        "runtime.conduit.route type=%s source=%s action=%s",
                        conduit_type,
                        source,
                        "wake_as_message",
                    )
                    response = await self._submit_message_unlocked(
                        IRInboundMessage(
                            blocks=[
                                IRUserTextBlock(
                                    text=frame_conduit(source, content, metadata),
                                    origin="conduit",
                                )
                            ],
                            source_metadata={
                                "source": source,
                                "origin": "conduit",
                                "conduit_type": conduit_type,
                                "metadata": dict(metadata or {}),
                            },
                            delivery="inject",
                        )
                    )
                    conduit_response = ConduitResponse(
                        delivered=True,
                        action=(
                            "queued_as_message" if response.queued else "started_turn"
                        ),
                        source=source,
                    )
                    logger.info(
                        "runtime.conduit.complete type=%s source=%s action=%s delivered=%s started=%s queued=%s",
                        conduit_type,
                        source,
                        conduit_response.action,
                        conduit_response.delivered,
                        response.started,
                        response.queued,
                    )
                    return conduit_response

                logger.info(
                    "runtime.conduit.route type=%s source=%s action=%s",
                    conduit_type,
                    source,
                    "queued_as_context",
                )
                await self._queue_conduit_footer(
                    conduit_type=conduit_type,
                    source=source,
                    content=content,
                    metadata=metadata,
                )
                response = ConduitResponse(
                    delivered=True,
                    action="queued_as_context",
                    source=source,
                )
                logger.info(
                    "runtime.conduit.complete type=%s source=%s action=%s delivered=%s",
                    conduit_type,
                    source,
                    response.action,
                    response.delivered,
                )
                return response

            logger.warning(
                "runtime.conduit.invalid_type type=%s source=%s", conduit_type, source
            )
            raise ValueError(f"invalid conduit type: {conduit_type!r}")

    async def interrupt(self) -> bool:
        """Request cancellation of the active turn."""
        async with self._command_lock:
            session = self._require_session()
            return session.interrupt()

    def build_catchup(self) -> CatchupResponse:
        """Build a transcript-backed catchup snapshot."""
        self._require_session()
        return CatchupResponse(
            rehydrated=Rehydrator(self.transcript_path).run(),
            surface=self._last_active_surface,
            surface_time=self._last_surface_time,
        )

    def build_health(self) -> HealthResponse:
        """Build a small runtime health snapshot."""
        session = self._require_session()
        return HealthResponse(
            model=session.config.model,
            state=session_to_runtime_state(session.state),
            turns=session.recorder.current_turn_idx,
            gauge_input_tokens=None,
            surface=self._last_active_surface,
            surface_time=self._last_surface_time,
        )

    async def _submit_message_unlocked(
        self, message: IRInboundMessage
    ) -> SubmitMessageResponse:
        if message.delivery == "footer":
            raise ValueError("`submit_message` only accepts turn/inject messages.")
        session = self._require_session()
        if message.delivery == "inject" and session.state == "running":
            await self._note_surface_for_inbound(message)
        queued = session.state == "running" or session.inbound_queue.has_pending_turn()
        await session.submit_message(message)
        if queued:
            self.bus.publish(MessageQueuedEvent(message=message))
        return SubmitMessageResponse(started=not queued, queued=queued)

    async def _before_turn_started(
        self,
        ctx: SessionContext,
        turn_id: str,
    ) -> None:
        if ctx.inbound is None:
            logger.debug("runtime.turn_started_no_inbound turn=%s", turn_id)
            return
        logger.debug(
            "runtime.turn_started_callback turn=%s delivery=%s metadata_keys=%s",
            turn_id,
            ctx.inbound.delivery,
            sorted(ctx.inbound.source_metadata.keys()),
        )
        await self._note_surface_for_inbound(ctx.inbound)

    async def _note_surface_for_inbound(self, inbound: IRInboundMessage) -> None:
        assert self.config is not None
        surface = surface_for_inbound(inbound)
        if surface is None:
            logger.debug(
                "runtime.surface.skipped reason=no_surface delivery=%s metadata_keys=%s",
                inbound.delivery,
                sorted(inbound.source_metadata.keys()),
            )
            return
        if (
            inbound.source_metadata.get("origin") == "conduit"
            and surface == "terminal TUI"
        ):
            logger.debug(
                "runtime.surface.skipped reason=conduit_terminal_tui delivery=%s",
                inbound.delivery,
            )
            return

        previous_surface = self._last_active_surface
        self._last_active_surface = surface
        self._last_surface_time = datetime.now(timezone.utc)
        logger.debug(
            "runtime.surface.observed surface=%s previous=%s delivery=%s",
            surface,
            previous_surface,
            inbound.delivery,
        )
        if surface == self._last_reported_surface:
            logger.debug("runtime.surface.report_skipped surface=%s", surface)
            return

        if self._last_reported_surface is None:
            text = f"Current human surface: {surface}."
        else:
            text = f"{self.config.user_name} is now on {surface}."
        self._last_reported_surface = surface
        logger.info(
            "runtime.surface.transition surface=%s previous_reported=%s",
            surface,
            previous_surface,
        )
        await self._queue_footer(
            text=text,
            footer_type="surface",
            footer_source="runtime",
            key="surface_transition",
        )

    async def _queue_conduit_footer(
        self,
        *,
        conduit_type: str,
        source: str,
        content: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        key = conduit_key(conduit_type, source, content, metadata)
        priority = conduit_priority(metadata)
        logger.info(
            "runtime.conduit.footer_queue type=%s source=%s key=%s priority=%s",
            conduit_type,
            source,
            key,
            priority,
        )
        await self._queue_footer(
            text=format_conduit_footer(source, content, metadata),
            footer_type="conduit",
            footer_source="conduit",
            key=key,
            priority=priority,
        )

    async def _queue_footer(
        self,
        *,
        text: str,
        footer_type: str,
        footer_source: str,
        key: str,
        priority: int = 50,
    ) -> None:
        await self._require_session().submit_message(
            IRInboundMessage(
                blocks=[IRUserTextBlock(text=text, origin="system")],
                source_metadata={
                    "footer_type": footer_type,
                    "footer_source": footer_source,
                    "footer_key": key,
                    "footer_priority": priority,
                },
                delivery="footer",
            )
        )

    async def shutdown(self) -> None:
        """Stop the session loop and close live subscriptions."""
        session = self._session
        task = self._session_task
        self._session_task = None

        if session is not None:
            await session.shutdown()

        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                self.bus.close()
                raise

        self.bus.close()

    def _on_session_task_done(self, task: asyncio.Task[None]) -> None:
        if self._session_task is not task:
            return

        if task.cancelled():
            logger.info("Core session task was cancelled.")
            return

        exc = task.exception()
        if exc is None:
            logger.error("Core session task exited unexpectedly without an exception.")
            return

        logger.critical(
            "Core session task crashed.", exc_info=(type(exc), exc, exc.__traceback__)
        )

    def _require_session(self) -> SessionManager:
        if self._session is None:
            raise RuntimeError("CoreAppRuntime has not been started.")
        return self._session
