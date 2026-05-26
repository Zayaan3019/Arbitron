"""Agent API router — async task launch and WebSocket streaming."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from ..schemas import AgentRunRequest, AgentRunResponse

router = APIRouter()
CACHE_DIR = os.environ.get("ARBITRON_CACHE_DIR", ".cache/arbitron")


@router.post("/agent/run", response_model=AgentRunResponse)
async def run_agent(req: AgentRunRequest, request: Request):
    """
    Launch an autonomous agent session.

    Returns a session_id. Connect to /ws/agent/{session_id} to receive
    streaming events from the research loop.
    """
    sessions: dict = request.app.state.sessions
    session_id = str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    sessions[session_id] = queue

    # Run agent in background task
    asyncio.create_task(
        _run_agent_background(
            task=req.task,
            queue=queue,
            max_iterations=req.max_iterations,
        )
    )

    return AgentRunResponse(
        session_id=session_id,
        status="running",
        message=(
            f"Agent started. Connect to /ws/agent/{session_id} "
            "to stream events."
        ),
    )


async def _run_agent_background(
    task: str,
    queue: asyncio.Queue,
    max_iterations: int,
) -> None:
    """Run ArbitronAgent in a thread pool to avoid blocking the event loop."""
    from arbitron.agent import ArbitronAgent, AgentEvent
    import concurrent.futures

    loop = asyncio.get_running_loop()

    def _blocking_stream():
        """Collect events synchronously — runs in thread pool."""
        events = []
        agent = ArbitronAgent(
            cache_dir=CACHE_DIR,
            max_iterations=max_iterations,
        )
        for event in agent.stream(task):
            events.append(event)
        return events

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = loop.run_in_executor(pool, _blocking_stream)

        # Poll the future until done, draining a local queue
        # We use a helper thread that puts events into an intermediate queue
        local_q: asyncio.Queue = asyncio.Queue()

        def _stream_with_queue():
            agent = ArbitronAgent(
                cache_dir=CACHE_DIR,
                max_iterations=max_iterations,
            )
            for event in agent.stream(task):
                asyncio.run_coroutine_threadsafe(
                    local_q.put(event), loop
                ).result(timeout=5)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool2:
            fut = loop.run_in_executor(pool2, _stream_with_queue)

            # Forward local_q → WebSocket queue until done/error
            while True:
                try:
                    event = await asyncio.wait_for(local_q.get(), timeout=5.0)
                    await queue.put({
                        "event_type": event.event_type,
                        "data":       _serialisable(event.data),
                    })
                    if event.event_type in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    if fut.done():
                        break
                    # keep waiting
            await asyncio.wrap_future(fut)

    # Ensure done event is in queue
    try:
        queue.put_nowait({"event_type": "done", "data": {"status": "complete"}})
    except asyncio.QueueFull:
        pass


def _serialisable(obj: Any) -> Any:
    """Convert to JSON-safe types."""
    import math
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(x) for x in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj
