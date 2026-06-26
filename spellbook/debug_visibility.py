from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from spellbook.ir_types import (
    IRRuntimeConfigRecord,
    RuntimeConfigNamespace,
    RuntimeConfigValue,
)
from spellbook.session_lifecycle import SessionContext, SessionLifecycle
from spellbook.system_response import SystemResponse

if TYPE_CHECKING:
    from spellbook.recorder import Recorder

logger = logging.getLogger(__name__)

DEBUG_RUNTIME_CONFIG_NAMESPACE: RuntimeConfigNamespace = "debug_visibility"

DebugNoticeLevel = Literal["debug", "info", "warning", "error"]


@dataclass(frozen=True)
class DebugVisibilitySettings:
    enabled: bool = False

    def as_record_dict(self) -> dict[str, RuntimeConfigValue]:
        return {"enabled": self.enabled}


@dataclass(frozen=True, slots=True)
class DebugNotice:
    subsystem: str
    event: str
    level: DebugNoticeLevel
    title: str
    content: str | None = None
    plaintext: str | None = None
    metadata: dict[str, Any] | None = None
    debug_only: bool = True


def settings_from_runtime_config_records(
    records: Sequence[IRRuntimeConfigRecord],
) -> DebugVisibilitySettings:
    settings = DebugVisibilitySettings()
    for record in records:
        if record.namespace != DEBUG_RUNTIME_CONFIG_NAMESPACE:
            continue
        settings = _apply_debug_settings(settings, record.effective)
    return settings


class DebugEmitter:
    """Session-owned operator visibility emitter.

    Subsystems enqueue notices synchronously. The emitter serializes them through
    the existing SystemResponse path so live clients and transcript catchup see
    the same durable event stream.
    """

    def __init__(
        self,
        *,
        recorder: Recorder,
        session_lifecycle: SessionLifecycle,
        settings: DebugVisibilitySettings | None = None,
    ) -> None:
        self._recorder = recorder
        self._session_lifecycle = session_lifecycle
        self._settings = settings or DebugVisibilitySettings()
        self._ctx: SessionContext | None = None
        self._queue: deque[DebugNotice] = deque()
        self._flush_lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    @property
    def settings(self) -> DebugVisibilitySettings:
        return self._settings

    def bind_context(self, ctx: SessionContext) -> None:
        self._ctx = ctx

    def configure_enabled(
        self, enabled: bool
    ) -> tuple[DebugVisibilitySettings, DebugVisibilitySettings, bool]:
        old = self._settings
        if old.enabled == enabled:
            return old, old, False

        new = replace(old, enabled=enabled)
        self._settings = new
        self._recorder.write_runtime_config(
            namespace=DEBUG_RUNTIME_CONFIG_NAMESPACE,
            updates={"enabled": enabled},
            effective=new.as_record_dict(),
            source="operator",
        )
        return old, new, True

    def alert(
        self,
        *,
        subsystem: str,
        event: str,
        title: str,
        content: str | None = None,
        plaintext: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.emit(
            DebugNotice(
                subsystem=subsystem,
                event=event,
                level="error",
                title=title,
                content=content,
                plaintext=plaintext,
                metadata=metadata,
                debug_only=False,
            )
        )

    def debug(
        self,
        *,
        subsystem: str,
        event: str,
        title: str,
        content: str | None = None,
        plaintext: str | None = None,
        metadata: dict[str, Any] | None = None,
        level: DebugNoticeLevel = "debug",
    ) -> None:
        self.emit(
            DebugNotice(
                subsystem=subsystem,
                event=event,
                level=level,
                title=title,
                content=content,
                plaintext=plaintext,
                metadata=metadata,
                debug_only=True,
            )
        )

    def emit(self, notice: DebugNotice) -> None:
        if notice.debug_only and not self.enabled:
            return
        self._queue.append(notice)
        self._schedule_flush()

    async def flush(self) -> None:
        async with self._flush_lock:
            while self._queue:
                notice = self._queue.popleft()
                response = self._response_for_notice(notice)
                self._recorder.write_system_response(response)
                await self._session_lifecycle.on_system_response(
                    self._require_context(), response
                )

    async def close(self) -> None:
        self._closed = True
        await self.flush()

    def _schedule_flush(self) -> None:
        if self._closed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._flush_task is not None and not self._flush_task.done():
            return
        self._flush_task = loop.create_task(self._flush_safely())

    async def _flush_safely(self) -> None:
        try:
            await self.flush()
        except Exception:
            logger.exception("debug_emitter.flush_failed")
        finally:
            self._flush_task = None
            if self._queue and not self._closed:
                self._schedule_flush()

    def _require_context(self) -> SessionContext:
        if self._ctx is None:
            raise RuntimeError("DebugEmitter has no bound SessionContext.")
        return self._ctx

    def _response_for_notice(self, notice: DebugNotice) -> SystemResponse:
        metadata: dict[str, Any] = {
            "kind": "debug_event",
            "level": notice.level,
            "subsystem": notice.subsystem,
            "event": notice.event,
            "debug_only": notice.debug_only,
        }
        if notice.metadata is not None:
            metadata.update(notice.metadata)
        return SystemResponse(
            command="/debug",
            content=notice.content or f"## {notice.title}",
            plaintext=notice.plaintext or notice.title,
            metadata=metadata,
        )


def _apply_debug_settings(
    settings: DebugVisibilitySettings, updates: dict[str, RuntimeConfigValue]
) -> DebugVisibilitySettings:
    allowed = {"enabled"}
    unknown = sorted(set(updates) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown debug_visibility config key(s): {', '.join(unknown)}"
        )

    if "enabled" not in updates:
        return settings
    enabled = updates["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("debug_visibility.enabled must be a boolean.")
    return replace(settings, enabled=enabled)
