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
    "period",
    "observed",
    "corrected",
    "savings",
    "savings_unc",
    "pct_savings",
    "coverage",
]


class SavingsAggregation(str, Enum):
    NATIVE = "native"
    MONTHLY = "monthly"
    SEASONAL = "seasonal"
    ANNUAL = "annual"
    TOTAL = "total"


def _period_labels(datetime, aggregation, season_def):
    if aggregation == SavingsAggregation.NATIVE:
        return datetime

    if aggregation == SavingsAggregation.MONTHLY:
        return datetime.dt.strftime("%Y-%m")

    if aggregation == SavingsAggregation.SEASONAL:
        season = datetime.dt.month_name().map(season_def)

        return datetime.dt.year.astype(str) + "-" + season

    if aggregation == SavingsAggregation.ANNUAL:
        return datetime.dt.year.astype(str)

    if aggregation == SavingsAggregation.TOTAL:
        return pd.Series("total", index=datetime.index)

    raise ValueError(f"unsupported aggregation {aggregation!r}.")


def _align_observed_unc(native, observed_unc):
    """An explicit override of the corrected frame's ``observed_unc`` column, as
    per meter-timestep values to quadrature-combine with the corrected
    uncertainty. A dict/Series keyed by id is a scalar-per-meter value broadcast
    across timestamps (missing ids default to 0; an id present but mapped to NaN
    stays NaN). A ``[id, datetime, observed_unc]`` frame supplies per-timestep
    values (unmatched rows default to 0)."""
    if isinstance(observed_unc, pd.DataFrame):
        if observed_unc.duplicated(subset=["id", "datetime"]).any():
            raise ValueError("observed_unc frame has duplicate (id, datetime) rows.")

        merged = native[["id", "datetime"]].merge(observed_unc, on=["id", "datetime"], how="left")

        return merged["observed_unc"].fillna(0.0).to_numpy(dtype=np.float64)

    values = native["id"].map(lambda mid: float(observed_unc.get(mid, 0.0)))

    return values.to_numpy(dtype=np.float64)


def _expand_read_periods(native, periods):
    """Spread each billing read period's totals and savings variance uniformly
    over its calendar days, so calendar rollups sum whole days into buckets.

    A read landing inside a single bucket contributes its total exactly; a read
    straddling a bucket boundary is pro-rated by day count into each bucket (a
    documented approximation — the daily shape within a read is unknown).
    Variance is spread the same way, so a period's days sum back to its read
    variance additively and a bucket fed by several reads combines those reads'
    variances in quadrature. Coverage is measured at day grain: every expanded
    day inherits its read's finite flag.
    """
    frames = []

    for row in native.itertuples(index=False):
        start = row.datetime
        end = _read_period_end(periods, start)
        days = pd.date_range(start, end, freq="D")
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


def _read_period_end(periods, start):
    for period_start, period_end in periods:
        if period_start == start:
            return period_end

    raise ValueError(f"no read period starts at {start}.")


def _aggregate(native, keys):
    """Roll native meter-timesteps up to ``keys``.

    ``observed``/``corrected``/``savings`` sum only over finite timesteps (those
    with a finite corrected AND observed value); ``coverage`` reports the finite
    fraction. ``savings_unc`` is the quadrature over those same finite timesteps,
    so the band describes the covered fraction the point sums describe; a finite
    timestep whose uncertainty is non-finite still propagates NaN to the
    period's band. A period with zero coverage has no valid data and reports
    NaN throughout rather than a misleading zero.
    """
    grouped = native.groupby(keys, sort=True)
    agg = grouped.agg(
        observed=("masked_observed", lambda s: np.sum(s.to_numpy(dtype=np.float64))),
        corrected=("masked_corrected", lambda s: np.sum(s.to_numpy(dtype=np.float64))),
        savings=("masked_savings", lambda s: np.sum(s.to_numpy(dtype=np.float64))),
        savings_var=("masked_savings_var", lambda s: np.sum(s.to_numpy(dtype=np.float64))),
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

    return agg[keys + _OUTPUT_COLUMNS[1:]]


def compute_savings(correction, observed_unc=None, aggregation="native", season_def=None):
    """Avoided energy and its uncertainty for one treatment meter, from a
    ``CorrectionResult``.

    Per meter-timestep: ``savings = corrected - observed``; ``savings_unc`` is
    ``corrected_unc`` combined in quadrature with ``observed_unc`` (0 when not
    supplied); ``pct_savings = savings / corrected`` (NaN where corrected is 0,
    guarding the divide-by-zero rather than raising under
    ``filterwarnings=error``). A timestep is covered when both its corrected
    and observed values are finite; uncovered timesteps leave every period sum,
    the band included, and ``coverage`` reports their share.

    Every output is a SUM (total energy) at its aggregation — native, monthly,
    seasonal, annual, total — never an average. The native cadence equals the
    correction cadence, so a billing treatment's native rows are its read
    periods. Calendar rollups (monthly and coarser) of a billing meter spread
    each read period's savings and variance uniformly over its calendar days
    and sum whole days into buckets: exact for a read that lands inside one
    bucket, pro-rated by day count for a read that straddles a boundary (a
    documented approximation). Hourly and daily rollups group native rows
    directly. The month-length normalization of billing baseline loadshapes is
    a selection feature only and never touches a savings number.

    Time roll-up sums float64; uncertainty aggregates in quadrature (variance
    sums, then a single final square root).

    Whether the time roll-up is exact depends on the treatment's granularity.
    At hourly cadence, ``corrected_unc`` already carries the per-hour ASHRAE
    aggregate band, so quadrature-summing it across a period reconstructs that
    aggregate exactly. At daily/billing cadence, ``corrected_unc`` is itself
    only a t-scaled prediction-interval band for that single read period, so a
    period's ``savings_unc`` is a quadrature-summed PI band rather than an
    ASHRAE aggregate share, an approximation distinct from the hourly path's
    exact reconstruction (unifying the two constructions is future work; see
    ``ROADMAP.md``).

    Args:
        correction: ``CorrectionResult`` to compute savings from.
        observed_unc: optional explicit observed uncertainty for this meter —
            a dict/Series keyed by id (scalar, broadcast across timestamps) or a
            ``[id, datetime, observed_unc]`` frame (per-timestep). Two-tier
            precedence: when omitted, the ``observed_unc`` column carried on
            ``correction.corrected`` is used (the treatment meter's observed
            uncertainty threaded through the correction, 0 where unset); when
            supplied, this kwarg overrides that column outright. Either way it
            affects the savings uncertainty only and never enters the correction
            math.
        aggregation: ``"native" | "monthly" | "seasonal" | "annual" | "total"``.
        season_def: month-name to season-label mapping; defaults to
            ``opendsm.common.const.default_season_def``.

    Returns:
        ``SavingsResult`` with a ``savings`` frame for this meter.
    """
    aggregation = SavingsAggregation(aggregation)

    if season_def is None:
        season_def = default_season_def

    native = correction.corrected[
        ["id", "datetime", "observed", "corrected", "corrected_unc", "observed_unc"]
    ].copy()
    native["datetime"] = native["datetime"].dt.as_unit("ns")

    if observed_unc is not None:
        native["observed_unc"] = _align_observed_unc(native, observed_unc)

    native["savings"] = native["corrected"] - native["observed"]
    native["savings_var"] = native["corrected_unc"] ** 2 + native["observed_unc"] ** 2
    corrected_values = native["corrected"].to_numpy(dtype=np.float64)
    observed_values = native["observed"].to_numpy(dtype=np.float64)
    native["finite"] = np.isfinite(corrected_values) & np.isfinite(observed_values)

    calendar_rollup = aggregation != SavingsAggregation.NATIVE

    if correction.granularity == "billing" and calendar_rollup:
        rows = _expand_read_periods(native, correction.correction_periods)
    else:
        rows = native

    rows["masked_observed"] = np.where(rows["finite"], rows["observed"], 0.0)
    rows["masked_corrected"] = np.where(rows["finite"], rows["corrected"], 0.0)
    rows["masked_savings"] = np.where(rows["finite"], rows["savings"], 0.0)
    rows["masked_savings_var"] = np.where(rows["finite"], rows["savings_var"], 0.0)
    rows["period"] = _period_labels(rows["datetime"], aggregation, season_def)

    savings = _aggregate(rows, ["id", "period"])

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
    finite corrected and observed value; periods with zero coverage report NaN
    rather than a zero built from no data.

    ``savings_unc`` quadrature-sums the per-timestep ``corrected_unc`` over
    the period's finite timesteps. At hourly cadence this reconstructs the ASHRAE aggregate
    uncertainty for the period exactly; at daily/billing cadence, where
    ``corrected_unc`` is itself only a t-scaled prediction-interval band, the
    period's ``savings_unc`` is a quadrature-summed PI band rather than an
    ASHRAE aggregate share, and is not the hourly path's exact reconstruction.
    This is not a calibrated (1 - alpha) interval.
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
