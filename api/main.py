"""
Arbitron FastAPI application.

Endpoints:
  GET  /health
  GET  /api/factors
  POST /api/factors/compute
  POST /api/factors/evaluate
  POST /api/backtest
  POST /api/portfolio/optimize
  GET  /api/portfolio/frontier
  POST /api/risk
  POST /api/agent/run         — fire-and-forget, returns session_id
  WS   /ws/agent/{session_id} — streams AgentEvent messages as JSON
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Load .env for local development (no-op in production where env vars are injected)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .routers import data, factors, backtest, portfolio, risk, agent as agent_router
from .schemas import HealthResponse


# ---------------------------------------------------------------------------
# In-memory session store (agent tasks)
# ---------------------------------------------------------------------------

# session_id → asyncio.Queue of AgentEvent dicts
_sessions: Dict[str, asyncio.Queue] = {}


def get_sessions() -> Dict[str, asyncio.Queue]:
    return _sessions


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: nothing heavy needed
    yield
    # Shutdown: clean up sessions
    _sessions.clear()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Arbitron API",
        description=(
            "Autonomous Quant Research Agent — REST + WebSocket API.\n\n"
            "Powered by Claude tool_use, walk-forward backtesting, "
            "Ledoit-Wolf portfolio optimization, and HMM regime detection."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — dev: localhost only; prod: set ARBITRON_CORS_ORIGIN=* or comma-separated list
    _cors_env = os.environ.get("ARBITRON_CORS_ORIGIN", "")
    if _cors_env == "*":
        _allow_origins = ["*"]
        _allow_credentials = False  # credentials not allowed with wildcard origin
    elif _cors_env:
        _allow_origins = [o.strip() for o in _cors_env.split(",")]
        _allow_credentials = True
    else:
        _allow_origins = [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
        ]
        _allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allow_origins,
        allow_credentials=_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Include routers
    # -----------------------------------------------------------------------
    app.include_router(data.router,      prefix="/api",       tags=["data"])
    app.include_router(factors.router,   prefix="/api",       tags=["factors"])
    app.include_router(backtest.router,  prefix="/api",       tags=["backtest"])
    app.include_router(portfolio.router, prefix="/api",       tags=["portfolio"])
    app.include_router(risk.router,      prefix="/api",       tags=["risk"])
    app.include_router(
        agent_router.router,
        prefix="/api",
        tags=["agent"],
        # Pass session store to router via app state
    )

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse()

    # -----------------------------------------------------------------------
    # WebSocket: stream agent events
    # -----------------------------------------------------------------------
    @app.websocket("/ws/agent/{session_id}")
    async def agent_ws(websocket: WebSocket, session_id: str):
        await websocket.accept()

        if session_id not in _sessions:
            await websocket.send_json({"error": "Unknown session_id"})
            await websocket.close()
            return

        queue = _sessions[session_id]
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    await websocket.send_json({"event_type": "ping"})
                    continue

                payload = {
                    "event_type": event.get("event_type"),
                    "data":       _json_safe(event.get("data")),
                    "session_id": session_id,
                }
                await websocket.send_json(payload)

                if event.get("event_type") in ("done", "error"):
                    break

        except WebSocketDisconnect:
            pass
        finally:
            _sessions.pop(session_id, None)

    # Store session registry on app state so routers can access it
    app.state.sessions = _sessions

    return app


def _json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


app = create_app()
