"""
Arbitron FastAPI application.

Endpoints:
  GET  /health
  GET  /api/universes
  POST /api/data/prices
  GET  /api/factors
  POST /api/factors/evaluate
  POST /api/backtest
  POST /api/portfolio/optimize
  POST /api/portfolio/frontier
  POST /api/risk
  POST /api/lab/run
  POST /api/agent/run         — fire-and-forget, returns session_id
  WS   /ws/agent/{session_id} — streams AgentEvent messages as JSON

Everything under /api requires an `X-API-Key` header when
`ARBITRON_REQUIRE_API_KEY=true` (see `api.security`).  `/health` is
deliberately unauthenticated so platform health checks keep working.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect

from fastapi.middleware.cors import CORSMiddleware

# Load .env for local development (no-op in production where env vars are injected)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .routers import data, factors, backtest, portfolio, risk, lab, agent as agent_router
from .schemas import HealthResponse
from .security import (
    api_key_required,
    configured_keys,
    log_startup_posture,
    require_api_key,
)
from .serialization import SafeJSONResponse, json_safe
from .sessions import SessionRegistry

log = logging.getLogger("arbitron.api")

#: Bounded registry of live agent runs (see api/sessions.py).
SESSIONS = SessionRegistry(
    max_sessions=int(os.environ.get("ARBITRON_MAX_SESSIONS", "64")),
    ttl_seconds=float(os.environ.get("ARBITRON_SESSION_TTL", "1800")),
)


def get_sessions() -> SessionRegistry:
    return SESSIONS


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=os.environ.get("ARBITRON_LOG_LEVEL", "INFO"))
    log_startup_posture()
    yield
    SESSIONS.clear()


def _cors_origins() -> tuple[list[str], bool]:
    """
    Resolve the CORS allow-list.

    A wildcard origin combined with an unauthenticated API is what let
    anyone on the internet spend the operator's Groq key, so `*` is now
    refused whenever key enforcement is off: it must be an explicit,
    deliberate configuration in both places or neither.
    """
    raw = os.environ.get("ARBITRON_CORS_ORIGIN", "").strip()

    if not raw:
        return ([
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ], True)

    if raw == "*":
        if not api_key_required():
            log.warning(
                "ARBITRON_CORS_ORIGIN='*' with API key enforcement OFF. "
                "Falling back to localhost origins. Set ARBITRON_CORS_ORIGIN "
                "to your real frontend origin."
            )
            return (["http://localhost:5173", "http://127.0.0.1:5173"], True)
        # Wildcard is tolerable only because the API key is the real gate.
        return (["*"], False)   # credentials are not allowed with a wildcard

    return ([o.strip() for o in raw.split(",") if o.strip()], True)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Arbitron API",
        description=(
            "Autonomous Quant Research Agent — REST + WebSocket API.\n\n"
            "Walk-forward backtesting, Ledoit-Wolf portfolio optimisation, "
            "causal HMM regime detection, and a statistical-defense layer "
            "(PSR / Deflated Sharpe / PBO).\n\n"
            "**Data limitation:** the built-in universes are today's index "
            "members, not point-in-time membership. Every response derived "
            "from them carries a `data_caveats` block spelling out the "
            "survivorship and selection bias."
        ),
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        default_response_class=SafeJSONResponse,
    )

    allow_origins, allow_credentials = _cors_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    # -----------------------------------------------------------------------
    # Routers — every /api route sits behind the API-key dependency
    # -----------------------------------------------------------------------
    api_guard = [Depends(require_api_key)]

    for module, tag in (
        (data,         "data"),
        (factors,      "factors"),
        (backtest,     "backtest"),
        (portfolio,    "portfolio"),
        (risk,         "risk"),
        (lab,          "research-lab"),
        (agent_router, "agent"),
    ):
        app.include_router(
            module.router, prefix="/api", tags=[tag], dependencies=api_guard
        )

    # -----------------------------------------------------------------------
    # Health — unauthenticated so platform probes keep working
    # -----------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(
            api_key_required=api_key_required(),
            **SESSIONS.stats(),
        )

    # -----------------------------------------------------------------------
    # WebSocket: stream agent events
    # -----------------------------------------------------------------------
    @app.websocket("/ws/agent/{session_id}")
    async def agent_ws(websocket: WebSocket, session_id: str):
        # Browsers cannot set headers on a WebSocket handshake, so the key
        # travels as a query parameter here.
        if api_key_required():
            token = websocket.query_params.get("api_key", "")
            if not any(token == k for k in configured_keys()):
                await websocket.close(code=4401, reason="Invalid or missing api_key")
                return

        await websocket.accept()

        session = SESSIONS.get(session_id)
        if session is None:
            await websocket.send_json({
                "event_type": "error",
                "data": {"message": "Unknown or expired session_id"},
                "session_id": session_id,
            })
            await websocket.close()
            return

        queue = session.queue
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Keep intermediaries from closing an idle connection, and
                    # surface a silently-dead worker instead of hanging forever.
                    if session.task is not None and session.task.done() and queue.empty():
                        await websocket.send_json({
                            "event_type": "error",
                            "data": {"message": "Agent worker exited without a result."},
                            "session_id": session_id,
                        })
                        break
                    await websocket.send_json({"event_type": "ping", "data": None,
                                               "session_id": session_id})
                    continue

                await websocket.send_json({
                    "event_type": event.get("event_type"),
                    "data":       json_safe(event.get("data")),
                    "session_id": session_id,
                })

                if event.get("event_type") in ("done", "error"):
                    break

        except WebSocketDisconnect:
            pass
        except Exception as exc:              # never let a stream kill the loop
            log.exception("WebSocket stream failed for session %s", session_id)
            try:
                await websocket.send_json({
                    "event_type": "error",
                    "data": {"message": f"Stream failed: {exc}"},
                    "session_id": session_id,
                })
            except Exception:
                pass
        finally:
            SESSIONS.remove(session_id)

    app.state.sessions = SESSIONS
    return app


app = create_app()
