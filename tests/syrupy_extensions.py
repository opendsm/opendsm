"""Syrupy snapshot extension with float-tolerance comparison.

JSON-backed; the matcher parses both sides and walks structures, applying
math.isclose at numeric leaves so snapshots survive cross-machine FP/BLAS
rounding skew.
"""

import json
import math
from types import MappingProxyType
from typing import (
    Any,
    Iterator,
    Mapping,
)

from syrupy.extensions.json import JSONSnapshotExtension

# 5e-6 atol covers bin-mean residuals near zero where rtol can't help; 1e-5
# rtol covers the ~5-7e-6 relative drift observed on larger-magnitude bin
# means and stds across (a) Linux OpenBLAS vs macOS Accelerate and (b)
# numpy/sklearn version differences between py3.10/3.11/3.12 under tox.
# Bin residuals have smaller magnitudes than the underlying predictions,
# so their relative noise floor is higher than per-element prediction noise.
# Snapshots remain informative for genuine algorithmic regressions (which
# shift outputs by >> 1e-4 relative; PR 5's clustering reorg shifted
# per-hour spread by 3-12%, four orders of magnitude above this floor).
FLOAT_ATOL = 5e-6
FLOAT_RTOL = 1e-5


class TolerantJSONSnapshotExtension(JSONSnapshotExtension):
    """JSON snapshot extension that compares numeric leaves within a tolerance.

    Two integers compare exactly (JSON booleans count as integers). Any other numeric pair
    compares with ``math.isclose(abs_tol=atol, rel_tol=rtol)``, and NaN equals NaN but no
    number. Every other leaf compares with ``==``. A dict key or list index present on one side
    only is a difference.

    ``atol`` is chosen per top-level dict key: ``atol`` under ``atol_keys``, ``FLOAT_ATOL`` under
    the rest. A numeric leaf whose own key (the last dict key on its path; list elements take
    their list's key) is in ``leaf_atol`` uses the mapped atol instead. A snapshot whose top level
    is not a dict compares at ``FLOAT_ATOL`` throughout, with ``leaf_atol`` still applying.
    ``rtol`` applies everywhere.

    On failure, replaces syrupy's per-line unified diff with a compact summary of the largest
    divergences so long arrays produce short output.
    """

    atol = FLOAT_ATOL
    rtol = FLOAT_RTOL
    atol_keys: frozenset[str] = frozenset()
    leaf_atol: Mapping[str, float] = MappingProxyType({})

    @classmethod
    def with_tolerance(
        cls,
        *,
        atol: float,
        rtol: float,
        atol_keys: frozenset[str],
        leaf_atol: Mapping[str, float],
    ) -> type["TolerantJSONSnapshotExtension"]:
        """Return a subclass of this extension that compares with these tolerances."""
        # syrupy finds an extension by module and name, and type() alone would set the module to abc
        attrs = {
            "atol": atol,
            "rtol": rtol,
            "atol_keys": atol_keys,
            "leaf_atol": MappingProxyType(dict(leaf_atol)),
            "__module__": cls.__module__,
        }

        return type(cls.__name__, (cls,), attrs)

    def _atol_for(self, key: str) -> float:
        return self.atol if key in self.atol_keys else FLOAT_ATOL

    def _tolerance_text(self) -> str:
        if not self.atol_keys and not self.leaf_atol:
            return f"atol={self.atol:g}, rtol={self.rtol:g}"

        parts = []
        if self.atol_keys:
            parts.append(f"atol={self.atol:g} under {', '.join(sorted(self.atol_keys))}")

        if self.leaf_atol:
            leaf_items = sorted(self.leaf_atol.items())
            leaves = ", ".join(f"{name}={value:g}" for name, value in leaf_items)
            parts.append(f"atol by leaf name {leaves}")

        parts.append(f"atol={FLOAT_ATOL:g} elsewhere")
        parts.append(f"rtol={self.rtol:g}")
        text = "; ".join(parts)

        return text

    def matches(self, *, serialized_data, snapshot_data) -> bool:
        if serialized_data == snapshot_data:
            return True
        try:
            actual = json.loads(serialized_data)
            expected = json.loads(snapshot_data)
        except (json.JSONDecodeError, TypeError):
            return False

        return not self._differences(actual, expected)

    def diff_lines(self, serialized_data, snapshot_data) -> Iterator[str]:
        try:
            actual = json.loads(serialized_data)
            expected = json.loads(snapshot_data)
        except (json.JSONDecodeError, TypeError):
            yield from super().diff_lines(serialized_data, snapshot_data)
            return

        diffs = self._differences(actual, expected)
        if not diffs:
            yield from super().diff_lines(serialized_data, snapshot_data)
            return

        diffs.sort(key=lambda d: -_score(d[1], d[2]))
        yield f"{len(diffs)} value(s) outside tolerance ({self._tolerance_text()})"
        for path, a, e in diffs[:10]:
            line = f"  {path}: received {a!r}, snapshot {e!r}"
            if _is_number(a) and _is_number(e):
                line += f"  (abs={abs(a - e):.3g}, rel={_rel(a, e):.3g})"

            yield line

        if len(diffs) > 10:
            yield f"  ... {len(diffs) - 10} more"

    def _differences(self, actual: Any, expected: Any) -> list[tuple[str, Any, Any]]:
        """Return (path, actual, expected) for every leaf outside tolerance or present on one
        side only."""
        diffs: list[tuple[str, Any, Any]] = []
        self._collect_diffs(actual, expected, "", None, FLOAT_ATOL, 0, diffs)

        return diffs

    def _collect_diffs(
        self,
        a: Any,
        b: Any,
        path: str,
        key: str | None,
        atol: float,
        depth: int,
        diffs: list,
    ) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for k in a.keys() | b.keys():
                child_atol = atol
                if depth == 0:
                    child_atol = self._atol_for(k)

                child_path = f"{path}.{k}" if path else k
                ak = a.get(k, _MISSING)
                bk = b.get(k, _MISSING)
                self._collect_diffs(ak, bk, child_path, k, child_atol, depth + 1, diffs)

            return

        if isinstance(a, list) and isinstance(b, list):
            for i in range(max(len(a), len(b))):
                ai = a[i] if i < len(a) else _MISSING
                bi = b[i] if i < len(b) else _MISSING
                self._collect_diffs(ai, bi, f"{path}[{i}]", key, atol, depth + 1, diffs)

            return

        if isinstance(a, int) and isinstance(b, int):
            if a != b:
                diffs.append((path, a, b))

            return

        if _is_number(a) and _is_number(b):
            a_nan = math.isnan(a)
            b_nan = math.isnan(b)
            if a_nan and b_nan:
                return

            leaf_atol = self.leaf_atol.get(key, atol)
            if a_nan or b_nan or not math.isclose(a, b, abs_tol=leaf_atol, rel_tol=self.rtol):
                diffs.append((path, a, b))

            return

        if a != b:
            diffs.append((path, a, b))


class _Missing:
    """Stands in for a dict key or list index absent on one side; equals no JSON value."""

    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _Missing()


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _rel(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))

    return abs(a - b) / scale if scale > 0 else 0.0


def _score(a, b) -> float:
    try:
        return max(abs(float(a) - float(b)), _rel(float(a), float(b)))
    except (TypeError, ValueError):
        return 0.0
