"""
API-key gate and request throttling.

`/api/agent/run` starts an LLM loop that spends the operator's Groq quota,
and `/api/lab/run` burns minutes of CPU per call.  Before this module both
were reachable by anyone who found the URL, with `ARBITRON_CORS_ORIGIN="*"`
advertising that fact.  (CORS was never the control anyway — it constrains
browsers, not `curl`.)

Configuration
-------------
``ARBITRON_API_KEY``
    Shared secret.  Clients send it as the ``X-API-Key`` header.  Comma
    separate to allow several keys (e.g. during rotation).
``ARBITRON_REQUIRE_API_KEY``
    ``true`` (default in `render.yaml`) makes a missing/incorrect key a 401.
    When it is true but no key is configured the service fails closed with a
    503 rather than silently serving the world.
    Unset/false leaves the API open — intended for local development only,
    and logged loudly at startup.
``ARBITRON_RATE_LIMIT_PER_MIN``
    Per-client-IP request budget for the expensive endpoints (default 10).
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

from fastapi import Header, HTTPException, Request, status

log = logging.getLogger("arbitron.security")

API_KEY_HEADER = "X-API-Key"


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def configured_keys() -> list[str]:
    raw = os.environ.get("ARBITRON_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def api_key_required() -> bool:
    return _truthy(os.environ.get("ARBITRON_REQUIRE_API_KEY"))


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """
    FastAPI dependency enforcing the shared-secret header.

    Constant-time comparison, so the endpoint does not leak the key one
    character at a time through response timing.
    """
    keys = configured_keys()

    if not api_key_required():
        return  # development mode; startup already warned

    if not keys:
        # Configured to require a key but none set: fail closed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Server misconfiguration: ARBITRON_REQUIRE_API_KEY is set but "
                "ARBITRON_API_KEY is empty. Refusing to serve unauthenticated."
            ),
        )

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {API_KEY_HEADER} header.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    if not any(hmac.compare_digest(x_api_key, k) for k in keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid {API_KEY_HEADER}.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class SlidingWindowLimiter:
    """
    In-process sliding-window limiter keyed by client IP.

    Deliberately simple: one Render instance, one process.  It is a spend
    guard, not a distributed rate limiter — behind multiple replicas the
    effective limit multiplies by the replica count.  Documented rather than
    over-engineered.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and now - bucket[0] > self.window:
                bucket.popleft()

            # Opportunistic sweep so idle clients cannot grow the dict forever.
            if len(self._hits) > 4096:
                for k in [k for k, v in self._hits.items() if not v]:
                    self._hits.pop(k, None)

            if len(bucket) >= self.max_requests:
                return False, max(0.0, self.window - (now - bucket[0]))
            bucket.append(now)
            return True, 0.0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def _limit_from_env() -> int:
    try:
        return max(1, int(os.environ.get("ARBITRON_RATE_LIMIT_PER_MIN", "10")))
    except ValueError:
        return 10


EXPENSIVE_LIMITER = SlidingWindowLimiter(_limit_from_env())


def _client_ip(request: Request) -> str:
    # Render terminates TLS upstream, so the socket peer is the proxy.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit_expensive(request: Request) -> None:
    """
    FastAPI dependency for endpoints that cost real money or minutes of CPU
    (`/api/agent/run`, `/api/lab/run`).
    """
    allowed, retry_after = EXPENSIVE_LIMITER.check(_client_ip(request))
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded ({EXPENSIVE_LIMITER.max_requests} requests "
                f"per {int(EXPENSIVE_LIMITER.window)}s). Retry in "
                f"{retry_after:.0f}s."
            ),
            headers={"Retry-After": str(int(retry_after) + 1)},
        )


def log_startup_posture() -> None:
    """Make the security posture visible in the deploy log, not a surprise."""
    if api_key_required():
        n = len(configured_keys())
        log.info("API key enforcement: ON (%d key(s) configured)", n)
    else:
        log.warning(
            "API key enforcement: OFF. Every endpoint including /api/agent/run "
            "is open. Set ARBITRON_REQUIRE_API_KEY=true and ARBITRON_API_KEY "
            "before exposing this service to the internet."
        )
