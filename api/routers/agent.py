"""
Agent API router — background task launch and WebSocket streaming.

The previous implementation built the agent twice (a `_blocking_stream`
closure that was never called, plus a `_stream_with_queue` that was), leaked
two nested `ThreadPoolExecutor`s per request, and had an error path that
queued no terminal event — so a client whose run crashed waited on the
WebSocket forever.

This version runs exactly one worker thread per session, forwards its events
to the session queue, and guarantees a terminal `done`/`error` event on every
exit path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request

from ..schemas import AgentRunRequest, AgentRunResponse
from ..security import rate_limit_expensive
from ..serialization import json_safe

router = APIRouter()
log = logging.getLogger("arbitron.agent")

CACHE_DIR = os.environ.get("ARBITRON_CACHE_DIR", ".cache/arbitron")

#: Hard wall-clock ceiling for one agent run.  Without it a wedged tool call
#: pins a worker thread for the life of the process.
AGENT_TIMEOUT_SECONDS = float(os.environ.get("ARBITRON_AGENT_TIMEOUT", "900"))


@router.post(
    "/agent/run",
    response_model=AgentRunResponse,
    dependencies=[Depends(rate_limit_expensive)],
)
async def run_agent(req: AgentRunRequest, request: Request) -> AgentRunResponse:
    """
    Launch an autonomous agent session.

    Returns a session_id. Connect to /ws/agent/{session_id} to receive
    streaming events from the research loop.
    """
    registry = request.app.state.sessions
    session_id = str(uuid.uuid4())
    session = registry.create(session_id)

    task = asyncio.create_task(
        _run_agent_background(
            task=req.task,
            queue=session.queue,
            max_iterations=req.max_iterations,
            session_id=session_id,
        )
    )
    # Strong reference — `create_task` alone keeps only a weak one, so an
    # in-flight run could be garbage-collected mid-request.
    registry.track_task(session, task)

    return AgentRunResponse(
        session_id=session_id,
        status="running",
        message=(
            f"Agent started. Connect to /ws/agent/{session_id} to stream events."
        ),
    )


async def _run_agent_background(
    task: str,
    queue: asyncio.Queue,
    max_iterations: int,
    session_id: str,
) -> None:
    """
    Drive one ArbitronAgent run on a worker thread, forwarding its events.

    The agent loop is synchronous and does blocking network + CPU work, so it
    must not touch the event loop; it runs in a thread and hands events back
    through a loop-owned queue.
    """
    from arbitron.agent import ArbitronAgent

    loop = asyncio.get_running_loop()
    relay: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker() -> None:
        """Runs on a worker thread. Must never raise into the executor."""
        try:
            agent = ArbitronAgent(cache_dir=CACHE_DIR, max_iterations=max_iterations)
            for event in agent.stream(task):
                asyncio.run_coroutine_threadsafe(
                    relay.put({"event_type": event.event_type, "data": event.data}),
                    loop,
                ).result(timeout=30)
        except Exception as exc:                       # noqa: BLE001
            log.exception("Agent session %s failed", session_id)
            try:
                asyncio.run_coroutine_threadsafe(
                    relay.put({
                        "event_type": "error",
                        "data": {"message": f"{type(exc).__name__}: {exc}"},
                    }),
                    loop,
                ).result(timeout=10)
            except Exception:
                pass
        finally:
            try:
                asyncio.run_coroutine_threadsafe(relay.put(_SENTINEL), loop).result(timeout=10)
            except Exception:
                pass

    terminal_sent = False
    try:
        worker = loop.run_in_executor(None, _worker)

        async def _pump() -> None:
            nonlocal terminal_sent
            while True:
                event = await relay.get()
                if event is _SENTINEL:
                    return
                await queue.put({
                    "event_type": event["event_type"],
                    "data": json_safe(event["data"]),
                })
                if event["event_type"] in ("done", "error"):
                    terminal_sent = True

        await asyncio.wait_for(_pump(), timeout=AGENT_TIMEOUT_SECONDS)
        await worker

    except asyncio.TimeoutError:
        await _put_terminal(queue, {
            "event_type": "error",
            "data": {"message": f"Agent run exceeded {AGENT_TIMEOUT_SECONDS:.0f}s and was abandoned."},
        })
        terminal_sent = True
    except asyncio.CancelledError:
        await _put_terminal(queue, {
            "event_type": "error",
            "data": {"message": "Agent run cancelled by the server."},
        })
        terminal_sent = True
        raise
    except Exception as exc:                            # noqa: BLE001
        log.exception("Agent supervisor failed for session %s", session_id)
        await _put_terminal(queue, {
            "event_type": "error",
            "data": {"message": f"{type(exc).__name__}: {exc}"},
        })
        terminal_sent = True
    finally:
        # A client must always see an ending, whatever happened above.
        if not terminal_sent:
            await _put_terminal(queue, {
                "event_type": "done",
                "data": {"status": "complete"},
            })


async def _put_terminal(queue: asyncio.Queue, event: dict[str, Any]) -> None:
    """Enqueue a terminal event, dropping the oldest item if the queue is full."""
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()
            queue.put_nowait(event)
        except Exception:
            pass
