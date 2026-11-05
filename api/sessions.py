"""
Bounded agent-session registry.

The previous implementation was a bare module-level ``dict`` that only ever
had entries removed when a WebSocket client connected *and* disconnected
cleanly.  Two consequences:

* **Leak** — every `/api/agent/run` whose client never connected left a queue
  (and its accumulated events) resident for the lifetime of the process.
* **Trivial DoS** — a loop of unauthenticated POSTs filled memory with no
  bound at all.

This registry caps the number of live sessions, evicts by age, and hands out
a terminal event to anything it evicts so no client is left hanging.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Session:
    """One agent run and the queue feeding its WebSocket."""
    session_id: str
    queue: asyncio.Queue
    created_at: float = field(default_factory=time.time)
    task: Optional[asyncio.Task] = None
    finished: bool = False

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class SessionRegistry:
    """
    Bounded, TTL-expiring session store.

    Not thread-safe by design — it is only touched from the event loop.
    """

    def __init__(self, max_sessions: int = 64, ttl_seconds: float = 1800.0) -> None:
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, Session] = {}
        #: Strong references to in-flight tasks.  `asyncio.create_task` only
        #: holds a weak reference, so a task with no other referent can be
        #: garbage-collected mid-flight and simply stop running.
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------

    def create(self, session_id: str, maxsize: int = 1000) -> Session:
        """Register a new session, evicting old ones to stay within bounds."""
        self._evict_expired()
        while len(self._sessions) >= self.max_sessions:
            oldest = min(self._sessions.values(), key=lambda s: s.created_at)
            self._drop(oldest.session_id, reason="evicted: session limit reached")

        session = Session(session_id=session_id, queue=asyncio.Queue(maxsize=maxsize))
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        self._evict_expired()
        return self._sessions.get(session_id)

    def track_task(self, session: Session, task: asyncio.Task) -> None:
        """Keep a strong reference so the background run cannot be GC'd."""
        session.task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def finish(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.finished = True

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear(self) -> None:
        for sid in list(self._sessions):
            self._drop(sid, reason="server shutting down")
        for task in list(self._tasks):
            task.cancel()
        self._tasks.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "live_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "running_tasks": sum(1 for t in self._tasks if not t.done()),
        }

    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        for sid, session in list(self._sessions.items()):
            if session.age_seconds > self.ttl_seconds:
                self._drop(sid, reason="expired")

    def _drop(self, session_id: str, reason: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        # Always leave a terminal event behind: a client that reconnects must
        # see an ending rather than block on an empty queue forever.
        try:
            session.queue.put_nowait({"event_type": "error", "data": {"message": reason}})
        except asyncio.QueueFull:
            pass
        if session.task is not None and not session.task.done():
            session.task.cancel()
