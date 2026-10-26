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

from __future__ import annotations

import json
from enum import Enum
from io import StringIO

import numpy as np
import pandas as pd

from opendsm.common.const import default_season_def



SCHEMA_VERSION = 1

_OUTPUT_COLUMNS = [
    "id",
    "period",
    "observed",
    "corrected",
    "savings",
    "savings_unc",
    "pct_savings",
    "coverage",
]


class SavingsAggregation(str, Enum):
    """Periods ``compute_savings`` sums over: each native timestep, month, season,
    year, or the whole window."""

    NATIVE = "native"
    MONTHLY = "monthly"
    SEASONAL = "seasonal"
    ANNUAL = "annual"
    TOTAL = "total"


def _period_labels(datetime, aggregation, season_def):
    if aggregation == SavingsAggregation.NATIVE:
        labels = datetime
    elif aggregation == SavingsAggregation.MONTHLY:
        labels = datetime.dt.strftime("%Y-%m")
    elif aggregation == SavingsAggregation.SEASONAL:
        season = datetime.dt.month_name().map(season_def)
        labels = datetime.dt.year.astype(str) + "-" + season
    elif aggregation == SavingsAggregation.ANNUAL:
        labels = datetime.dt.year.astype(str)
    elif aggregation == SavingsAggregation.TOTAL:
        labels = pd.Series("total", index=datetime.index)
    else:
        raise ValueError(f"unsupported aggregation {aggregation!r}.")

    return labels


def _align_observed_unc(native, observed_unc, meter_id):
    """Per-row ``observed_unc`` values for ``native`` from an explicit override."""
    if isinstance(observed_unc, pd.DataFrame):
        if observed_unc.duplicated(subset=["id", "datetime"]).any():
            raise ValueError("observed_unc frame has duplicate (id, datetime) rows.")

        if not (observed_unc["id"] == meter_id).any():
            raise ValueError(f"observed_unc frame has no rows for meter {meter_id!r}.")

        merged = native[["id", "datetime"]].merge(observed_unc, on=["id", "datetime"], how="left")
        values = merged["observed_unc"].fillna(0.0).to_numpy(dtype=np.float64)

        return values

    if meter_id not in observed_unc:
        raise ValueError(f"observed_unc has no entry for meter {meter_id!r}.")

    values = native["id"].map(lambda mid: float(observed_unc[mid])).to_numpy(dtype=np.float64)

    return values


def _expand_read_periods(native, periods):
    """Spread each billing read period's totals and savings variance uniformly
    over its calendar days, so calendar rollups sum whole days into buckets.

    Variance is spread the same way, so a period's days sum back to its read
    variance additively and a bucket fed by several reads combines those reads'
    variances in quadrature. Coverage is measured at day grain: every expanded
    day inherits its read's finite flag.
    """
    period_ends = dict(periods)
    frames = []

    for row in native.itertuples(index=False):
        start = row.datetime

        if start not in period_ends:
            raise ValueError(f"no read period starts at {start}.")

        days = pd.date_range(start, period_ends[start], freq="D")
        n = len(days)

        frame = pd.DataFrame(
            {
                "id": row.id,
                "datetime": days,
                "observed": row.observed / n,
                "corrected": row.corrected / n,
                "savings": row.savings / n,
                "savings_var": row.savings_var / n,
                "finite": row.finite,
            }
        )
        frames.append(frame)

    if not frames:
        return native

    expanded = pd.concat(frames, ignore_index=True)

    return expanded


def _aggregate(native):
    """Roll native meter-timesteps up to ``(id, period)``; a covered timestep
    whose uncertainty is non-finite makes its period's band NaN."""
    masked = ["masked_observed", "masked_corrected", "masked_savings", "masked_savings_var"]
    rows = native.astype({column: np.float64 for column in masked})
    grouped = rows.groupby(["id", "period"], sort=True)
    agg = grouped.agg(
        observed=("masked_observed", "sum"),
        corrected=("masked_corrected", "sum"),
        savings=("masked_savings", "sum"),
        # pandas' "sum" skips NaN.
        savings_var=("masked_savings_var", lambda s: s.to_numpy().sum()),
        coverage=("finite", "mean"),
    ).reset_index()

    no_coverage = agg["coverage"].to_numpy(dtype=np.float64) == 0.0
    agg.loc[no_coverage, ["observed", "corrected", "savings", "savings_var"]] = np.nan

    agg["savings_unc"] = np.sqrt(agg["savings_var"].to_numpy(dtype=np.float64))
    agg = agg.drop(columns=["savings_var"])

    denom = agg["corrected"].to_numpy(dtype=np.float64)
    pct = np.full(len(agg), np.nan)
    np.divide(agg["savings"].to_numpy(dtype=np.float64), denom, out=pct, where=denom != 0.0)
    agg["pct_savings"] = pct
    savings = agg[_OUTPUT_COLUMNS]

    return savings


def compute_savings(correction, observed_unc=None, aggregation="native", season_def=None):
    """Avoided energy and its uncertainty for one treatment meter, from a
    ``CorrectionResult``.

    Per meter-timestep: ``savings = corrected - observed``; ``savings_unc`` is
    ``corrected_unc`` combined in quadrature with ``observed_unc``.
    ``pct_savings`` is each period's ``savings / corrected`` (NaN where
    corrected is 0).

    Every energy column is a SUM (total energy) at its aggregation (native,
    monthly, seasonal, annual or total), never an average. The native cadence equals the
    correction cadence, so a billing treatment's native rows are its read
    periods. Calendar rollups (monthly and coarser) of a billing meter spread
    each read period's savings and variance uniformly over its calendar days
    and sum whole days into buckets: exact for a read that lands inside one
    bucket, pro-rated by day count for a read that straddles a boundary (an
    approximation). Rollups of hourly and daily meters group native rows directly.

    Time roll-up sums float64; uncertainty aggregates in quadrature (variance
    sums, then a single final square root).

    Args:
        correction: ``CorrectionResult`` to compute savings from.
        observed_unc: optional explicit observed uncertainty for this meter,
            replacing the ``observed_unc`` column of ``correction.corrected``
            (used when omitted). It affects the savings uncertainty only. Either
            a dict or Series keyed by id, whose value for this meter is broadcast
            across its timestamps (a NaN value stays NaN), or an
            ``[id, datetime, observed_unc]`` frame matched on ``(id, datetime)``,
            where a timestep of this meter with no row, or with a NaN value,
            contributes 0.
        aggregation: ``"native" | "monthly" | "seasonal" | "annual" | "total"``.
        season_def: month-name to season-label mapping; defaults to
            ``opendsm.common.const.default_season_def``.

    Returns:
        ``SavingsResult`` with a ``savings`` frame for this meter.

    Raises:
        ValueError: ``aggregation`` is not a ``SavingsAggregation`` value;
            ``observed_unc`` has no entry (mapping or Series) or no rows (frame)
            for this meter; or an ``observed_unc`` frame has duplicate
            ``(id, datetime)`` rows.
    """
    aggregation = SavingsAggregation(aggregation)

    if season_def is None:
        season_def = default_season_def

    native = correction.corrected[
        ["id", "datetime", "observed", "corrected", "corrected_unc", "observed_unc"]
    ].copy()
    native["datetime"] = native["datetime"].dt.as_unit("ns")

    if observed_unc is not None:
        native["observed_unc"] = _align_observed_unc(native, observed_unc, correction.meter_id)

    native["savings"] = native["corrected"] - native["observed"]
    native["savings_var"] = native["corrected_unc"] ** 2 + native["observed_unc"] ** 2
    corrected_values = native["corrected"].to_numpy(dtype=np.float64)
    observed_values = native["observed"].to_numpy(dtype=np.float64)
    native["finite"] = np.isfinite(corrected_values) & np.isfinite(observed_values)

    calendar_rollup = aggregation != SavingsAggregation.NATIVE

    if correction.model_frequency == "billing" and calendar_rollup:
        rows = _expand_read_periods(native, correction.correction_periods)
    else:
        rows = native

    rows["masked_observed"] = np.where(rows["finite"], rows["observed"], 0.0)
    rows["masked_corrected"] = np.where(rows["finite"], rows["corrected"], 0.0)
    rows["masked_savings"] = np.where(rows["finite"], rows["savings"], 0.0)
    rows["masked_savings_var"] = np.where(rows["finite"], rows["savings_var"], 0.0)
    rows["period"] = _period_labels(rows["datetime"], aggregation, season_def)

    savings = _aggregate(rows)

    result = SavingsResult(
        meter_id=correction.meter_id,
        savings=savings,
        aggregation=aggregation.value,
        tz=correction.tz,
    )

    return result


class SavingsResult:
    """Avoided energy for one treatment meter, at ``aggregation`` grain.

    ``meter_id`` is the treatment meter this result belongs to.

    ``savings`` (columns ``id``, ``period``, ``observed``, ``corrected``,
    ``savings``, ``savings_unc``, ``pct_savings``, ``coverage``) is float64.
    ``coverage`` is the fraction of native timesteps in that period with a
    finite corrected and observed value; uncovered timesteps leave every period
    sum, the band included. Periods with zero coverage report NaN rather than a
    zero built from no data.

    ``savings_unc`` quadrature-sums the per-timestep ``savings_unc``
    (``corrected_unc`` with ``observed_unc``) over the period's finite
    timesteps. At hourly cadence this reconstructs the ASHRAE aggregate
    uncertainty for the period exactly; at daily/billing cadence, where
    ``corrected_unc`` is itself a t-scaled prediction-interval band, it is a
    quadrature-summed PI band rather than an ASHRAE aggregate share. This is
    not a calibrated (1 - alpha) interval.
    """

    def __init__(self, meter_id, savings, aggregation, tz):
        self.meter_id = meter_id
        self.savings = savings
        self.aggregation = aggregation
        self.tz = tz

    @property
    def tables(self):
        """Flat frames (id as a column) for tabular storage."""
        frames = {
            "savings": self.savings.copy(),
        }

        return frames

    def to_json(self):
        """Serialize to JSON.

        Returns:
            A JSON string holding the header and the savings table.
        """
        payload = {
            "header": {
                "schema_version": SCHEMA_VERSION,
                "meter_id": self.meter_id,
                "tz": self.tz,
                "aggregation": self.aggregation,
            },
            "savings": self.savings.to_json(orient="table", double_precision=15),
        }

        return json.dumps(payload, allow_nan=False)

    @classmethod
    def from_json(cls, s):
        """Rebuild from ``to_json()`` output.

        Args:
            s: JSON string written by ``to_json()``.

        Returns:
            The rebuilt ``SavingsResult``.

        Raises:
            ValueError: The payload's ``schema_version`` is not supported.
        """
        payload = json.loads(s)
        header = payload["header"]

        if header["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {header['schema_version']}; expected {SCHEMA_VERSION}."
            )

        meter_id = header["meter_id"]
        tz = header["tz"]
        aggregation = header["aggregation"]
        savings = _read_frame(payload["savings"], tz, aggregation)

        result = cls(
            meter_id=meter_id,
            savings=savings,
            aggregation=aggregation,
            tz=tz,
        )

        return result


def _read_frame(payload, tz, aggregation):
    frame = pd.read_json(StringIO(payload), orient="table")
    frame["id"] = frame["id"].astype(str)

    if aggregation == SavingsAggregation.NATIVE.value:
        frame["period"] = pd.to_datetime(frame["period"], utc=True).dt.tz_convert(tz).dt.as_unit("ns")
    else:
        frame["period"] = frame["period"].astype(str)

    return frame
