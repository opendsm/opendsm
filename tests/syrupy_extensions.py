"""Syrupy snapshot extension with float-tolerance comparison.

JSON-backed; the matcher parses both sides and walks structures, applying
math.isclose at numeric leaves so snapshots survive cross-machine FP/BLAS
rounding skew.
"""

import json
import math
from typing import Any, Iterator

from syrupy.extensions.json import JSONSnapshotExtension

FLOAT_ATOL = 1e-7
# 5e-6 is the empirical floor: it covers the BLAS-FMA drift observed between
# Linux (OpenBLAS) and macOS (Apple Accelerate) on the hourly model regression
# path, where the deepest stack (ElasticNet coordinate descent + KMeans inertia
# + wavelet PCA SVD) accumulates ~2e-6 of platform-ordering noise per
# prediction. 1e-6 was tight enough on Linux alone but failed cross-platform.
# 5e-6 keeps the snapshots informative for genuine algorithmic regressions
# (which shift outputs by >>1e-5 relative) while accommodating the floor.
FLOAT_RTOL = 5e-6


class TolerantJSONSnapshotExtension(JSONSnapshotExtension):
    """JSON snapshot extension that compares floats with math.isclose tolerance.

    Default behavior on numeric leaves: ``math.isclose(actual, expected,
    abs_tol={atol}, rel_tol={rtol})``. NaN compares equal to NaN. Lists,
    dicts, and other structures compare element-wise; structural mismatch is
    a hard fail.

    On failure, replaces syrupy's per-line unified diff with a compact summary
    of the largest numeric divergences so long arrays produce short output.
    """

    def matches(self, *, serialized_data, snapshot_data) -> bool:
        if serialized_data == snapshot_data:
            return True
        try:
            actual = json.loads(serialized_data)
            expected = json.loads(snapshot_data)
        except (json.JSONDecodeError, TypeError):
            return False

        return _deep_close(actual, expected, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)

    def diff_lines(self, serialized_data, snapshot_data) -> Iterator[str]:
        try:
            actual = json.loads(serialized_data)
            expected = json.loads(snapshot_data)
        except (json.JSONDecodeError, TypeError):
            yield from super().diff_lines(serialized_data, snapshot_data)
            return

        diffs: list[tuple[str, float, float]] = []
        _collect_diffs(actual, expected, "", diffs, atol=FLOAT_ATOL, rtol=FLOAT_RTOL)
        if not diffs:
            yield from super().diff_lines(serialized_data, snapshot_data)
            return

        diffs.sort(key=lambda d: -_score(d[1], d[2]))
        yield f"{len(diffs)} value(s) outside tolerance (atol={FLOAT_ATOL}, rtol={FLOAT_RTOL})"
        for path, a, e in diffs[:10]:
            yield f"  {path}: received {a!r}, snapshot {e!r}  (abs={abs(a - e):.3g}, rel={_rel(a, e):.3g})"
        if len(diffs) > 10:
            yield f"  ... {len(diffs) - 10} more"


def _deep_close(a: Any, b: Any, *, atol: float, rtol: float) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False

        return all(_deep_close(a[k], b[k], atol=atol, rtol=rtol) for k in a)

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False

        return all(_deep_close(a[i], b[i], atol=atol, rtol=rtol) for i in range(len(a)))

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        a_nan = isinstance(a, float) and math.isnan(a)
        b_nan = isinstance(b, float) and math.isnan(b)
        if a_nan and b_nan:
            return True
        if a_nan or b_nan:
            return False

        return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)

    return a == b


def _collect_diffs(
    a: Any,
    b: Any,
    path: str,
    diffs: list,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Walk structures and record (path, actual, expected) for numeric leaves outside tolerance."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            _collect_diffs(a.get(k), b.get(k), f"{path}.{k}" if path else k, diffs, atol=atol, rtol=rtol)
        return

    if isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            ai = a[i] if i < len(a) else None
            bi = b[i] if i < len(b) else None
            _collect_diffs(ai, bi, f"{path}[{i}]", diffs, atol=atol, rtol=rtol)
        return

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        a_f = float(a)
        b_f = float(b)
        if math.isnan(a_f) and math.isnan(b_f):
            return
        if not math.isclose(a_f, b_f, abs_tol=atol, rel_tol=rtol):
            diffs.append((path, a_f, b_f))
        return

    if a != b:
        diffs.append((path, a, b))


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))

    return abs(a - b) / scale if scale > 0 else 0.0


def _score(a, b) -> float:
    try:
        return max(abs(float(a) - float(b)), _rel(float(a), float(b)))
    except (TypeError, ValueError):
        return 0.0
