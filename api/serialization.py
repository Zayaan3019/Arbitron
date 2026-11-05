"""
JSON safety at the response boundary.

Quant code produces NaN and Inf routinely and legitimately: a Sharpe on two
observations, a payoff ratio with no losing days, a Calmar with no drawdown.
`metrics.compute_metrics` returns NaN for all of those.

NaN is **not valid JSON**.  Python's `json` emits the bare token `NaN` unless
told otherwise, which every strict client rejects, and a NaN reaching a
non-optional Pydantic `float` field surfaces as an unhandled 500.  Everything
leaving this service therefore passes through `json_safe`, which maps
NaN/Inf to `null` recursively and normalises numpy/pandas scalars.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse


def json_safe(value: Any) -> Any:
    """
    Recursively convert a value into something `json.dumps(..., allow_nan=False)`
    will accept.

    * NaN / +Inf / -Inf  -> ``None``
    * numpy scalars      -> Python int / float / bool
    * numpy arrays, pandas Series / Index -> list
    * pandas DataFrame   -> list of row dicts
    * Timestamp / datetime / date -> ISO-8601 string
    * NaT / pd.NA        -> ``None``
    * dict keys          -> ``str``
    * anything unknown   -> ``str(value)`` (never raises)
    """
    # Scalars that are their own special cases first.
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, (float, np.floating)):
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        return value

    if value is pd.NaT or (value is not None and value is pd.NA):
        return None

    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, pd.DataFrame):
        return [json_safe(row) for row in value.to_dict(orient="records")]
    if isinstance(value, (pd.Series, pd.Index)):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in value]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def safe_float(value: Any, default: float | None = None) -> float | None:
    """Coerce one value to a finite float, or `default` (None) if it is not."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class SafeJSONResponse(JSONResponse):
    """
    JSONResponse that cannot emit invalid JSON.

    Installed as the app-wide `default_response_class`, so this holds for
    every endpoint including ones added later and FastAPI's own error
    responses.
    """

    def render(self, content: Any) -> bytes:
        import json

        return json.dumps(
            json_safe(content),
            ensure_ascii=False,
            allow_nan=False,          # belt and braces: raise rather than emit NaN
            separators=(",", ":"),
        ).encode("utf-8")
