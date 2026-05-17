"""FastAPI transport shell for the core app runtime."""

from __future__ import annotations

import faulthandler
import logging
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import AsyncIterator, Callable, Literal, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from spellbook.app.protocol import (
    AwarenessResponse,
    CatchupResponse,
    ConduitBody,
    ConduitResponse,
    HealthResponse,
    InterruptResponse,
    SubmitMessageBody,
    SubmitMessageResponse,
    WebSocketCatchupMode,
)
from spellbook.app.runtime import CoreAppRuntime
from spellbook.config import SpellbookConfig
from spellbook.custom import CustomSurface
from spellbook.ir_types import IRInboundMessage, IRUserTextBlock

RuntimeFactory = Callable[
    [Path, SpellbookConfig | None, CustomSurface | None], CoreAppRuntime
]
AppLogLevel = Literal["critical", "error", "warning", "info", "debug", "trace"]
APP_LOGGER_NAME = "spellbook"
APP_LOG_HANDLER_NAME = "spellbook-core-app-stderr"
_PYTHON_LOG_LEVELS: dict[AppLogLevel, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
}

logger = logging.getLogger(__name__)


def _default_runtime_factory(
    transcript_path: Path,
    config: SpellbookConfig | None,
    custom_surface: CustomSurface | None = None,
) -> CoreAppRuntime:
    return CoreAppRuntime(
        transcript_path=transcript_path, config=config, custom_surface=custom_surface
    )


def _enable_faulthandler() -> None:
    if faulthandler.is_enabled():
        return
    fault_file = sys.__stderr__
    if fault_file is None:
        fault_file = sys.stderr
    try:
        faulthandler.enable(file=fault_file, all_threads=True)
    except Exception:
        logger.exception("app.faulthandler_enable_failed")
    else:
        logger.info("app.faulthandler_enabled")


def configure_app_logging(log_level: AppLogLevel | int = "info") -> None:
    """Configure Spellbook runtime logs for direct `create_app` callers."""
    level = _python_log_level(log_level)
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level)
    app_logger.propagate = False

    handler = _app_log_handler(app_logger)
    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(APP_LOG_HANDLER_NAME)
        handler.setFormatter(
            logging.Formatter("%(levelname)s:     %(name)s: %(message)s")
        )
        app_logger.addHandler(handler)
    handler.setLevel(level)


def _python_log_level(value: AppLogLevel | int) -> int:
    if isinstance(value, int):
        return value
    return _PYTHON_LOG_LEVELS[_validate_log_level(value)]


def _validate_log_level(value: str) -> AppLogLevel:
    if value not in {"critical", "error", "warning", "info", "debug", "trace"}:
        raise ValueError(f"Unsupported log level: {value}")
    return cast(AppLogLevel, value)


def _app_log_handler(logger: logging.Logger) -> logging.Handler | None:
    for handler in logger.handlers:
        if handler.get_name() == APP_LOG_HANDLER_NAME:
            return handler
    return None


def create_app(
    *,
    transcript_path: Path,
    config: SpellbookConfig | None = None,
    custom_surface: CustomSurface | None = None,
    runtime_factory: RuntimeFactory = _default_runtime_factory,
    log_level: AppLogLevel | int | None = "info",
) -> FastAPI:
    """Create a FastAPI app around one `CoreAppRuntime`."""
    if log_level is not None:
        configure_app_logging(log_level)
    _enable_faulthandler()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = runtime_factory(transcript_path, config, custom_surface)
        app.state.runtime = runtime
        logger.info("app.startup transcript=%s", transcript_path)
        await runtime.startup()
        logger.info("app.started transcript=%s", transcript_path)
        try:
            yield
        finally:
            logger.info("app.shutdown_start transcript=%s", transcript_path)
            await runtime.shutdown()
            logger.info("app.shutdown_complete transcript=%s", transcript_path)

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def handle_health(request: Request) -> HealthResponse:
        return _runtime_from_request(request).build_health()

    @app.get("/catchup")
    async def handle_catchup(request: Request) -> CatchupResponse:
        return _runtime_from_request(request).build_catchup()

    @app.get("/awareness")
    async def handle_awareness(request: Request) -> AwarenessResponse:
        return _runtime_from_request(request).build_awareness()

    @app.post("/message")
    async def handle_submit(
        request: Request,
        body: SubmitMessageBody,
    ) -> SubmitMessageResponse:
        if not body.text.strip():
            raise HTTPException(status_code=400, detail="empty message")
        delivery = "inject" if body.inject else "turn"
        logger.info(
            "message.received delivery=%s text_chars=%s metadata_keys=%s",
            delivery,
            len(body.text),
            sorted(body.metadata.keys()),
        )
        response = await _runtime_from_request(request).submit_message(
            IRInboundMessage(
                blocks=[IRUserTextBlock(text=body.text, origin="human")],
                source_metadata=body.metadata,
                delivery=delivery,
            )
        )
        logger.info(
            "message.routed delivery=%s started=%s queued=%s",
            delivery,
            response.started,
            response.queued,
        )
        return response

    @app.post("/conduit")
    async def handle_conduit(
        request: Request,
        body: ConduitBody,
    ) -> ConduitResponse:
        conduit_type = body.type.strip()
        if conduit_type not in {"context", "message", "notification"}:
            raise HTTPException(
                status_code=400,
                detail=f"invalid conduit type: {conduit_type!r}",
            )
        content = body.content.strip()
        if not content:
            raise HTTPException(status_code=400, detail="empty content")
        source = body.source.strip() or "unknown"
        metadata = body.metadata if isinstance(body.metadata, dict) else None
        logger.info(
            "conduit.received type=%s source=%s content_chars=%s metadata_keys=%s",
            conduit_type,
            source,
            len(content),
            sorted(metadata.keys()) if metadata is not None else None,
        )
        response = await _runtime_from_request(request).handle_conduit(
            conduit_type=conduit_type,
            source=source,
            content=content,
            metadata=metadata,
        )
        logger.info(
            "conduit.routed type=%s source=%s action=%s delivered=%s",
            conduit_type,
            source,
            response.action,
            response.delivered,
        )
        return response

    @app.post("/interrupt")
    async def handle_interrupt(request: Request) -> InterruptResponse:
        logger.info("interrupt.received")
        interrupted = await _runtime_from_request(request).interrupt()
        logger.info("interrupt.routed interrupted=%s", interrupted)
        return InterruptResponse(interrupted=interrupted)

    @app.websocket("/ws")
    async def handle_ws(websocket: WebSocket) -> None:
        ws_id = f"ws_{uuid4().hex[:8]}"
        client_label = _ws_client_label(websocket)
        catchup_mode = _ws_catchup_mode(websocket)
        peer = _ws_peer(websocket)
        user_agent = websocket.headers.get("user-agent", "")
        logger.info(
            "ws.connecting id=%s client=%s catchup=%s peer=%s ua=%r",
            ws_id,
            client_label,
            catchup_mode,
            peer,
            user_agent[:120],
        )
        runtime = _runtime_from_app(websocket.app)
        subscription = None
        try:
            await websocket.accept()
            logger.info(
                "ws.accepted id=%s client=%s catchup=%s",
                ws_id,
                client_label,
                catchup_mode,
            )
            subscription = runtime.bus.subscribe()
            await _send_ws_catchup(
                websocket=websocket,
                runtime=runtime,
                mode=catchup_mode,
                ws_id=ws_id,
                client_label=client_label,
            )
            async for event in subscription:
                await websocket.send_json(event.model_dump(mode="json"))
                logger.debug(
                    "ws.event_sent id=%s client=%s kind=%s",
                    ws_id,
                    client_label,
                    event.kind,
                )
        except WebSocketDisconnect as exc:
            logger.info(
                "ws.disconnected id=%s client=%s code=%s",
                ws_id,
                client_label,
                exc.code,
            )
        except Exception:
            logger.exception("ws.crashed id=%s client=%s", ws_id, client_label)
            raise
        finally:
            if subscription is not None:
                subscription.close()
                logger.info(
                    "ws.subscription_closed id=%s client=%s sub=%s",
                    ws_id,
                    client_label,
                    subscription.id,
                )
            with suppress(RuntimeError):
                await websocket.close()
            logger.info("ws.closed id=%s client=%s", ws_id, client_label)

    return app


def _runtime_from_request(request: Request) -> CoreAppRuntime:
    return _runtime_from_app(request.app)


def _runtime_from_app(app: FastAPI) -> CoreAppRuntime:
    runtime = getattr(app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="runtime is not started")
    return runtime


def _ws_client_label(websocket: WebSocket) -> str:
    raw = websocket.query_params.get("client")
    if not raw:
        return "unknown"
    label = raw.strip()
    if not label:
        return "unknown"
    return label[:80]


def _ws_catchup_mode(websocket: WebSocket) -> WebSocketCatchupMode:
    raw = websocket.query_params.get("catchup")
    if raw is None or raw == "":
        return "lite"
    normalized = raw.strip().lower()
    if normalized == "none":
        return "none"
    if normalized == "lite":
        return "lite"
    if normalized == "full":
        return "full"
    logger.warning(
        "ws.invalid_catchup_mode raw=%r fallback=%s",
        raw,
        "lite",
    )
    return "lite"


async def _send_ws_catchup(
    *,
    websocket: WebSocket,
    runtime: CoreAppRuntime,
    mode: WebSocketCatchupMode,
    ws_id: str,
    client_label: str,
) -> None:
    match mode:
        case "none":
            logger.info("ws.catchup_skipped id=%s client=%s", ws_id, client_label)
        case "lite":
            catchup = runtime.build_catchup_lite()
            await websocket.send_json(catchup.model_dump(mode="json"))
            logger.info(
                "ws.catchup_lite_sent id=%s client=%s state=%s semantic_blocks=%s",
                ws_id,
                client_label,
                catchup.health.state,
                len(catchup.awareness.snapshot.homunculus.semantic_blocks),
            )
        case "full":
            catchup = runtime.build_catchup()
            await websocket.send_json(catchup.model_dump(mode="json"))
            logger.info(
                "ws.catchup_sent id=%s client=%s records=%s blocks=%s",
                ws_id,
                client_label,
                len(catchup.rehydrated.records),
                len(catchup.rehydrated.blocks),
            )


def _ws_peer(websocket: WebSocket) -> str:
    client = websocket.client
    if client is None:
        return "unknown"
    return f"{client.host}:{client.port}"
