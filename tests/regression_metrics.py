#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Regression-test metric helpers.

Produces compact, BLAS-stable snapshots that catch model regressions while
surviving cross-platform floating-point noise. Three orthogonal lenses:

  * scalar aggregates (sum/mean/std/min/max + p01/p50/p99) on the predicted
    series and on residuals, plus pearson_r and n_finite
  * canonical time lens (frequency-specific): season x hour-of-week for
    hourly, season x day-of-week for daily, season for billing
  * driver lenses: residual mean+std per temperature bin (always) and per
    GHI bin (solar only)

When `observed` is absent (e.g. modeled_savings derivative tests), the time
and driver lenses bin the predicted series itself rather than residuals.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


_MONTH_TO_SEASON = {
    1: "winter", 2: "winter",
    3: "shoulder", 4: "shoulder", 5: "shoulder",
    6: "summer", 7: "summer", 8: "summer", 9: "summer",
    10: "shoulder",
    11: "winter", 12: "winter",
}

# Fixed Fahrenheit edges; refined around the comfort range where most
# commercial load action happens.
TEMP_BIN_EDGES_F = [-math.inf, 30, 40, 50, 60, 65, 70, 75, 80, 90, math.inf]

# Fixed W/m^2 edges spanning the typical operating range.
GHI_BIN_EDGES = [-math.inf, 100, 200, 300, 400, 500, 600, 700, 800, 900, math.inf]

_QUANTILES = (0.01, 0.50, 0.99)


def _aggregates(series: pd.Series) -> dict:
    """Sum/mean/std/min/max + p01/p50/p99 on the finite subset of a series."""
    s = series.dropna()
    qs = s.quantile(list(_QUANTILES))

    return {
        "sum": float(s.sum()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "p01": float(qs.loc[0.01]),
        "p50": float(qs.loc[0.50]),
        "p99": float(qs.loc[0.99]),
    }


def _pearson_r(observed: pd.Series, predicted: pd.Series) -> float:
    """Pearson r on paired finite values; None when either side is degenerate (constant)."""
    paired = pd.concat([observed, predicted], axis=1).dropna()
    if len(paired) < 2:
        return None
    x = paired.iloc[:, 0].to_numpy()
    y = paired.iloc[:, 1].to_numpy()
    if x.std() == 0 or y.std() == 0:
        return None

    return float(pearsonr(x, y)[0])


def _residual_aggregates(residuals: pd.Series) -> dict:
    """Residual mean/std/max_abs; orthogonal to predicted aggregates."""
    r = residuals.dropna()

    return {
        "mean": float(r.mean()),
        "std": float(r.std(ddof=0)),
        "max_abs": float(r.abs().max()),
    }


def _bin_stats(values: pd.Series, groups, *, with_std: bool) -> dict:
    """Group `values` by `groups` and return {label: {mean, std?}}."""
    grouped = values.dropna().groupby(groups, observed=True)
    out = {}
    for key, vals in grouped:
        label = _format_key(key)
        entry = {"mean": float(vals.mean())}
        if with_std:
            entry["std"] = float(vals.std(ddof=0)) if len(vals) > 1 else 0.0
        out[label] = entry

    return dict(sorted(out.items()))


def _format_key(key) -> str:
    """Stable string form of a groupby key (scalar or tuple)."""
    if isinstance(key, tuple):
        return "_".join(_atom(k) for k in key)

    return _atom(key)


def _atom(k) -> str:
    """Render one component of a groupby key. pandas Intervals get a compact form."""
    if isinstance(k, pd.Interval):
        return f"[{_edge(k.left)},{_edge(k.right)}]"

    return str(k)


def _edge(x: float) -> str:
    if x == -math.inf:
        return "-inf"
    if x == math.inf:
        return "inf"

    return str(int(x)) if float(x).is_integer() else str(x)


def _season_series(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(index.month.map(_MONTH_TO_SEASON), index=index, name="season")


def _time_groups(index: pd.DatetimeIndex, freq: str):
    """Return groupby keys for the canonical time lens at the given frequency."""
    season = _season_series(index)

    if freq == "hourly":
        # zero-pad how so cell labels sort numerically in the snapshot
        how = pd.Series(
            [f"how{(d * 24 + h):03d}" for d, h in zip(index.dayofweek, index.hour)],
            index=index, name="how",
        )

        return [season, how]

    if freq == "daily":
        dow = pd.Series(
            [f"dow{d}" for d in index.dayofweek], index=index, name="dow",
        )

        return [season, dow]

    if freq == "billing":
        return [season]

    raise ValueError(f"Unknown freq: {freq!r}; expected hourly/daily/billing")


def _time_bins(values: pd.Series, freq: str) -> dict:
    """Apply the canonical time lens; billing uses mean-only (n per cell too small for std)."""
    groups = _time_groups(values.index, freq)
    with_std = freq != "billing"

    return _bin_stats(values, groups, with_std=with_std)


def _temp_bins(values: pd.Series, temperature: pd.Series) -> dict:
    """Mean+std of values per fixed-edge temperature bin."""
    bins = pd.cut(temperature, bins=TEMP_BIN_EDGES_F, include_lowest=True)

    return _bin_stats(values, [bins], with_std=True)


def _ghi_bins(values: pd.Series, ghi: pd.Series) -> dict:
    """Mean+std of values per fixed-edge GHI bin."""
    bins = pd.cut(ghi, bins=GHI_BIN_EDGES, include_lowest=True)

    return _bin_stats(values, [bins], with_std=True)


def regression_block(
    df: pd.DataFrame,
    freq: str,
    *,
    value_col: str = "predicted",
) -> dict:
    """Compact regression snapshot for a model prediction series.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain `value_col`. Optional columns: `observed`, `temperature`, `ghi`.
        Columns absent or all-NaN are skipped automatically.
    freq : {"hourly", "daily", "billing"}
        Selects the canonical time lens.
    value_col : str
        Column to snapshot. Defaults to "predicted".

    Returns
    -------
    dict
        JSON-serializable nested dict suitable for syrupy snapshot comparison.
        Top-level keys:
          - "predicted": aggregates on `value_col`
          - "n_finite_predicted": count of finite entries
          - "residuals", "pearson_r": present iff observed is available
          - "time_bins": residual stats per (season, time) cell if observed
                         is available, else stats of `value_col` itself
          - "temp_bins": same dispatch, per temperature bin
          - "ghi_bins": only if `ghi` is present and not all-NaN
    """
    predicted = df[value_col]
    block = {
        "predicted": _aggregates(predicted),
        "n_finite_predicted": int(predicted.notna().sum()),
    }

    has_observed = "observed" in df.columns and df["observed"].notna().any()
    if has_observed:
        observed = df["observed"]
        residuals = observed - predicted
        block["residuals"] = _residual_aggregates(residuals)
        block["pearson_r"] = _pearson_r(observed, predicted)
        bin_values = residuals
    else:
        bin_values = predicted

    block["time_bins"] = _time_bins(bin_values, freq)

    if "temperature" in df.columns and df["temperature"].notna().any():
        block["temp_bins"] = _temp_bins(bin_values, df["temperature"])

    if "ghi" in df.columns and df["ghi"].notna().any():
        block["ghi_bins"] = _ghi_bins(bin_values, df["ghi"])

    return block
