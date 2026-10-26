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
from io import StringIO

import numpy as np
import pandas as pd

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.selection import (
    COVERAGE_FREQ,
    FINENESS,
    check_model_frequency_fineness,
    coerce_window,
    window_coverage,
    window_to_json,
)
from opendsm.comparison_groups.savings.model_correction import (
    MIN_CG_METERS,
    MIN_CLUSTER_METERS,
    model_correction_matrix,
)
from opendsm.comparison_groups.savings.read_periods import (
    infer_read_periods,
    periods_from_boundaries,
)
from opendsm.comparison_groups.savings.settings import CGCorrectionSettings
from opendsm.eemeter.common.exceptions import EEMeterError



SCHEMA_VERSION = 2

_DAY_NS = 86_400_000_000_000


def _ns(index):
    """Nanosecond integer timestamps of a datetime index, whatever resolution
    it carries."""
    i8 = index.as_unit("ns").asi8

    return i8


def _column_correlations(observed, modeled):
    """Per-column Pearson correlation between observed and modeled over finite
    pairs. Columns with fewer than two finite pairs or zero variance yield 0.0.
    Emits no warnings (the correction runs under ``filterwarnings=error``)."""
    n_meters = observed.shape[1]
    corr = np.zeros(n_meters, dtype=np.float64)

    for m in range(n_meters):
        x = observed[:, m]
        y = modeled[:, m]
        finite = np.isfinite(x) & np.isfinite(y)

        if int(finite.sum()) < 2:
            continue

        xf = x[finite].astype(np.float64)
        yf = y[finite].astype(np.float64)
        xf = xf - xf.mean()
        yf = yf - yf.mean()
        sx = np.sqrt(np.dot(xf, xf))
        sy = np.sqrt(np.dot(yf, yf))

        if sx == 0.0 or sy == 0.0:
            continue

        corr[m] = float(np.dot(xf, yf) / (sx * sy))

    return corr


def _clusters_used(mask, cg_label):
    """Per timestep, the count of distinct clusters that survived (at least
    ``MIN_CLUSTER_METERS`` finite retained meters per ``mask``) out of the
    clusters in ``cg_label``."""
    labels = np.unique(cg_label)
    labels = labels[labels >= 0]
    used = np.zeros(mask.shape[0], dtype=int)

    for label in labels:
        used += mask[:, cg_label == label].sum(axis=1) >= MIN_CLUSTER_METERS

    return used


def _active_clusters(weights_row):
    """Map a treatment meter's ``pct_cluster_*`` weights to ``{cluster_label:
    weight}`` for the clusters it actually draws on (weight > 0)."""
    active = {}

    for column, weight in weights_row.items():
        value = float(weight)
        if value > 0.0:
            label = int(str(column).rsplit("_", 1)[1])
            active[label] = value

    return active


def _cg_membership(selection, treatment_id):
    """One treatment meter's raw comparison group as ordered ``(pool_id,
    cluster_label)`` pairs, plus the ``{cluster_label: weight}`` map of the
    clusters it draws on.

    The walk goes from the meter's weights row to the clusters it draws on
    (weight > 0) to every member of those clusters, in sorted-label order, and a
    pool meter reached through a duplicated selection row is kept once. No
    usability filter is applied, so the membership covers meters the correction
    later drops. A meter absent from the selection, or with all-NaN weights, has
    no comparison group and no active clusters.
    """
    weights = selection.treatment_weights
    members = []
    active = {}

    if treatment_id not in weights.index:
        return members, active

    weights_row = weights.loc[treatment_id]
    if weights_row.isna().all():
        return members, active

    clusters = selection.clusters
    active = _active_clusters(weights_row)
    seen = set()

    for label in sorted(active):
        for member in clusters.index[clusters["cluster"] == label]:
            mid = str(member)
            if mid in seen:
                continue

            seen.add(mid)
            members.append((mid, label))

    return members, active


def cg_member_ids(selection, treatment_id):
    """The pool ids in one treatment meter's raw comparison group, unfiltered by
    usability.

    Args:
        selection: The ``ComparisonGroupSelection`` holding the comparison groups.
        treatment_id: The treatment meter's id.

    Returns:
        A list of pool ids.
    """
    members, _ = _cg_membership(selection, treatment_id)
    ids = [mid for mid, _ in members]

    return ids


def _cluster_weights(active, labels):
    """Per-cluster weights aligned with the sorted labels the kernel expects.

    A selected cluster with no surviving comparison-group column contributes no
    label, so the weights are renormalized over the clusters that ARE present to
    keep ``sum(t_weight) == 1``, the kernel's contract for its normalized point
    vs. raw-weight uncertainty.
    """
    present = sorted(set(labels))
    t_weight = np.array([active[label] for label in present], dtype=np.float64)
    total = t_weight.sum()
    if total > 0.0:
        t_weight = t_weight / total

    return t_weight


def _align_to_index(arr, row_map, present, n_rows):
    """The rows of ``arr`` at ``row_map`` as float64, NaN where not
    ``present``; ``None`` passes through."""
    if arr is None:
        return None

    out = np.full((n_rows, arr.shape[1]), np.nan, dtype=np.float64)
    out[present] = arr[row_map[present]]

    return out


def _period_codes(source_i8, edges_i8):
    """Assign each source row (nanosecond timestamps ``source_i8``) to the
    period it falls in, given ``edges_i8`` (``n_periods + 1`` strictly
    increasing edges). Row ``t`` maps to period ``p`` when ``edges_i8[p] <=
    source_i8[t] < edges_i8[p + 1]``; rows outside ``[edges_i8[0],
    edges_i8[-1])`` map to ``-1`` (dropped)."""
    codes = np.searchsorted(edges_i8, source_i8, side="right") - 1
    n_periods = len(edges_i8) - 1
    codes = np.where((codes < 0) | (codes >= n_periods), -1, codes)

    return codes


def _restrict_to_days(codes, index, day_i8):
    """Drop (code -1) the rows of ``index`` whose local calendar day is not among
    ``day_i8`` (local-midnight nanosecond timestamps), so a pool aggregated to a
    treatment read period covers exactly the days the treatment covers."""
    row_days = _ns(index.normalize())
    keep = np.isin(row_days, day_i8)
    restricted = np.where(keep, codes, -1)

    return restricted


def _sum_to_periods(values, codes, n_periods):
    """Sum native rows to periods per the ``codes`` map (each row's period
    index, -1 for none). Each period-column sums the finite native rows assigned
    to it; a period with no finite native row in a column is NaN, never 0."""
    x = np.asarray(values, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[:, None]

    in_range = codes >= 0
    finite = np.isfinite(x) & in_range[:, None]
    contrib = np.where(finite, x, 0.0)
    pid = np.where(in_range, codes, 0)

    n_cols = x.shape[1]
    sums = np.zeros((n_periods, n_cols), dtype=np.float64)
    counts = np.zeros((n_periods, n_cols), dtype=np.float64)
    np.add.at(sums, pid, contrib)
    np.add.at(counts, pid, finite.astype(np.float64))

    out = np.where(counts >= 1.0, sums, np.nan)
    if single:
        out = out[:, 0]

    return out


def _quadrature_to_periods(values, codes, n_periods):
    """Combine per-row uncertainties to periods in quadrature: the root of
    ``_sum_to_periods`` over the squared rows, NaN where that sum is NaN."""
    x = np.asarray(values, dtype=np.float64)
    combined = np.sqrt(_sum_to_periods(x * x, codes, n_periods))

    return combined


def _reduce_pool(p_obs, p_mod, p_mod_unc, p_obs_unc, codes, n_periods):
    """The pool's observed and modeled matrices summed to periods and its
    uncertainty matrices combined in quadrature, returned as ``(oCGr, mCGr,
    mCGr_unc, oCGr_unc)``; ``oCGr_unc`` is None when ``p_obs_unc`` is."""
    oCGr = _sum_to_periods(p_obs, codes, n_periods)
    mCGr = _sum_to_periods(p_mod, codes, n_periods)
    mCGr_unc = _quadrature_to_periods(p_mod_unc, codes, n_periods)
    oCGr_unc = None
    if p_obs_unc is not None:
        oCGr_unc = _quadrature_to_periods(p_obs_unc, codes, n_periods)

    return oCGr, mCGr, mCGr_unc, oCGr_unc


def _cg_correlations(p_obs_unc, oCGr, mCGr):
    """Per-member correlation between the comparison group's observed and
    modeled rows as the kernel receives them, or None when no pool meter
    carries observed uncertainty (the kernel reads the correlation only
    then)."""
    if p_obs_unc is None or not np.any(p_obs_unc != 0.0):
        return None

    corr = _column_correlations(oCGr, mCGr)

    return corr


def _native_edges(index):
    """Period edges for an identity (per-native-row) structure: one period per
    row, the final period closed one native cadence past the last row so that
    row's finer-pool contributions bin into it."""
    i8 = _ns(index)
    if len(i8) >= 2:
        tail = i8[-1] + (i8[-1] - i8[-2])
    else:
        tail = i8[-1] + _DAY_NS
    edges = np.append(i8, tail)

    return edges


def _billing_edges(periods):
    """Period edges (nanosecond timestamps) for read periods, closing the final
    read one day past its last daily-substrate row."""
    starts = [start.value for start, _ in periods]
    tail = periods[-1][1].value + _DAY_NS
    edges = np.array(starts + [tail], dtype=np.int64)

    return edges


def _localize(value, tz):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(tz)
    else:
        stamp = stamp.tz_convert(tz)

    return stamp


def _period_mask(datetimes, period, tz):
    if period is None:
        return np.ones(len(datetimes), dtype=bool)

    start, end = period
    mask = np.ones(len(datetimes), dtype=bool)

    if start is not None:
        mask &= (datetimes >= _localize(start, tz)).to_numpy()

    if end is not None:
        mask &= (datetimes <= _localize(end, tz)).to_numpy()

    return mask


def _check_prior(new_corrected, prior, model_frequency):
    """Point corrections are slice-invariant, so any timestep the prior actually
    covered (finite corrected value) must reproduce that value exactly. Timesteps
    the prior left uncovered (NaN, e.g. pool data not yet present) are treated as
    newly covered, not as mismatches. A billing extension may re-draw the prior's
    trailing read period (partial then, or extended now), so that trailing period
    is excluded while every earlier period must match exactly."""
    prior_df = prior.corrected
    if prior_df.empty or new_corrected.empty:
        return

    merged = new_corrected.merge(
        prior_df[["datetime", "corrected"]],
        on="datetime",
        suffixes=("", "_prior"),
    )

    if model_frequency == "billing":
        # each read period is labelled by its start datetime, so the prior's last
        # read is the trailing period an extension may re-draw; a prior reported
        # under a period filter still knows its full read calendar
        if prior.correction_periods:
            trailing_start = prior.correction_periods[-1][0]
        else:
            trailing_start = prior_df["datetime"].max()

        merged = merged[merged["datetime"] < trailing_start]

    covered = merged[np.isfinite(merged["corrected_prior"].to_numpy(dtype=np.float64))]
    if covered.empty:
        return

    current = covered["corrected"].to_numpy(dtype=np.float64)
    previous = covered["corrected_prior"].to_numpy(dtype=np.float64)
    equal = current == previous

    if not equal.all():
        mismatched = covered.loc[~equal, "datetime"].tolist()
        raise ValueError(
            "prior correction mismatch on overlapping timesteps (point corrections must be "
            f"slice-invariant); offending datetimes: {mismatched}"
        )


def _prune_reporting(population, ids, ledger, failure):
    """Keep the ``ids`` of ``population`` for which ``failure(reporting_data)``
    returns None, in order, and ledger every other id at the correction stage
    with the ``(origin, reason, detail)`` it returns. Meters absent from
    ``population`` or without reporting data are kept for the
    prediction-usability guard. Returns the kept ids and the ledger."""
    kept = []

    for mid in ids:
        rec = population.meter(mid)
        if rec is None or rec.reporting_data is None:
            kept.append(mid)
            continue

        found = failure(rec.reporting_data)
        if found is None:
            kept.append(mid)
            continue

        origin, reason, detail = found
        ledger = exclusions.append(ledger, [mid], "correction", origin, reason, detail=detail)

    return kept, ledger


def _observed_dq_exclusions(population, ids, ledger):
    """Prune the ``ids`` of ``population`` whose reporting data carries an
    eemeter observed or joint (observed-and-temperature co-missing)
    disqualification (e.g. non-zero identical reads failing the unique-values
    check, or too many co-missing days), ledgering the warnings verbatim. The
    prune runs before any prediction, since the warning alone would leave the
    meter corrected. Returns the kept ids and the ledger."""

    def failure(reporting_data):
        observed_dq = [
            entry
            for entry in reporting_data.compliance.disqualification
            if "observed" in entry.qualified_name or "joint" in entry.qualified_name
        ]
        if not observed_dq:
            return None

        found = (
            "reporting_data",
            "reporting observed data disqualified",
            exclusions.format_warnings(observed_dq),
        )

        return found

    kept, ledger = _prune_reporting(population, ids, ledger, failure)

    return kept, ledger


def _coverage_exclusions(population, ids, ledger, window, min_reporting_coverage):
    """Prune the ``ids`` of ``population`` whose finite observed-and-temperature
    coverage over ``window``, the treatment meter's reporting span, falls below
    ``min_reporting_coverage``, ledgering each with its coverage in ``detail``
    (a pool member's construction-time checks know nothing of the treatment
    span). Returns the kept ids and the ledger."""
    freq = COVERAGE_FREQ[population.model_frequency]

    def failure(reporting_data):
        coverage = window_coverage(reporting_data, window, freq)
        if coverage >= min_reporting_coverage:
            return None

        found = (
            "reporting_coverage",
            "reporting coverage below minimum over the treatment meter's reporting span",
            f"reporting span coverage {coverage:.4f} < {min_reporting_coverage}",
        )

        return found

    kept, ledger = _prune_reporting(population, ids, ledger, failure)

    return kept, ledger


def _reporting_span(rec):
    """The extent ``(start, end)`` of a meter's reporting index (a billing
    meter's daily substrate), or None when it carries no reporting data."""
    if rec.reporting_data is None:
        return None

    index = rec.reporting_data.df.index
    span = (index.min(), index.max())

    return span


def _usable_reporting_ids(population, ids, ledger, missing_origin):
    """Which of ``ids`` are in ``population``, have reporting data and predict
    over it without an eemeter error, in the given order; meters that cannot are
    recorded on the correction ledger and dropped."""
    usable = []

    for mid in ids:
        rec = population.meter(mid)
        if rec is None:
            ledger = exclusions.append(
                ledger,
                [mid],
                "correction",
                "pool_membership",
                "comparison-group member absent from the pool population",
            )
            continue

        if rec.reporting_data is None:
            ledger = exclusions.append(
                ledger, [mid], "correction", missing_origin, "missing reporting data"
            )
            continue

        try:
            population.prediction_frames("reporting", ids=[mid])
        except EEMeterError as exc:
            ledger = exclusions.append(
                ledger,
                [mid],
                "correction",
                "model",
                "model prediction failed",
                detail=str(exc),
            )
            continue

        usable.append(mid)

    return usable, ledger


def _meter_error(ledger, treatment_id):
    """A ``MeterCorrectionError`` carrying the whole correction ledger and
    summarizing the rows in it that name ``treatment_id``."""
    rows = ledger[ledger["id"] == treatment_id]
    reasons = []

    for row in rows.itertuples():
        if row.detail:
            reasons.append(f"{row.reason} ({row.detail})")
        else:
            reasons.append(row.reason)

    message = f"treatment meter {treatment_id} cannot be corrected: {'; '.join(reasons)}"

    return exclusions.MeterCorrectionError(message, ledger)


class CorrectionResult:
    """Corrected reporting-period series for one treatment meter.

    ``meter_id`` is the treatment meter this result belongs to.

    ``corrected`` is a long frame with columns ``id``, ``datetime``,
    ``observed``, ``modeled``, ``modeled_unc``, ``corrected``, ``corrected_unc``,
    ``observed_unc``. ``observed_unc`` carries the treatment meter's
    observed uncertainty (0 when it carries none), threaded from the treatment
    population for ``compute_savings`` to combine into the savings band. At hourly
    cadence the per-timestep model uncertainty already reconstructs the ASHRAE
    hourly aggregate band; at daily/billing cadence it is a t-scaled
    prediction-interval band treated as sigma-like by the kernel for combination.
    Either way, ``modeled_unc``/``corrected_unc`` quadrature-combine that
    per-timestep band across the comparison-group meters used at each timestep,
    treating them as independent, and are heuristic bands rather than calibrated
    (1 - alpha) intervals.

    ``cg_ids`` is the ordered list of comparison-group meter ids the correction
    used: the members that predicted over the reporting period. Passing this
    result as ``prior`` to a later call reuses exactly these members.

    ``cg_usage`` reports, per timestep, the fraction of the meter's
    comparison-group meters actually used in the correction (finite, retained,
    and in a cluster that survived the timestep; ``cg_usage_fraction``) and the
    count of clusters that survived with at least ``MIN_CLUSTER_METERS`` finite
    retained meters (``clusters_used``). ``exclusions`` is the
    correction-stage disqualification ledger (``[id, stage, origin, reason,
    detail]``): the pool meters dropped while correcting this meter, and why.

    ``correction_periods`` is the ordered list of ``(start, end)`` periods the
    correction was aggregated over. It is empty for hourly/daily, whose
    corrections run at the native cadence; for billing these are the meter's
    inferred read periods.
    """

    def __init__(
        self,
        meter_id,
        corrected,
        cg_usage,
        exclusions,
        settings,
        covered_window,
        tz,
        model_frequency,
        fingerprint,
        correction_periods,
        cg_ids,
    ):
        self.meter_id = meter_id
        self.cg_ids = [str(mid) for mid in cg_ids]
        self.corrected = corrected
        self.cg_usage = cg_usage
        self.exclusions = exclusions
        self.settings = settings
        self.covered_window = covered_window
        self.tz = tz
        self.model_frequency = model_frequency
        self.fingerprint = fingerprint
        self.correction_periods = correction_periods

    @property
    def tables(self):
        """Flat frames (id as a column) for tabular storage."""
        frames = {
            "corrected": self.corrected.copy(),
            "cg_usage": self.cg_usage.copy(),
            "exclusions": self.exclusions.copy(),
        }

        return frames

    def _header(self):
        header = {
            "schema_version": SCHEMA_VERSION,
            "meter_id": self.meter_id,
            "tz": self.tz,
            "model_frequency": self.model_frequency,
            "fingerprint": self.fingerprint,
            "covered_window": window_to_json(self.covered_window),
            "settings": self.settings,
            "cg_ids": self.cg_ids,
        }

        return header

    def to_json(self):
        """Serialize to JSON.

        Returns:
            A JSON string holding the header, the corrected and usage tables, the ledger and the correction periods.
        """
        payload = {
            "header": self._header(),
            "corrected": self.corrected.to_json(orient="table", double_precision=15),
            "cg_usage": self.cg_usage.to_json(orient="table", double_precision=15),
            "exclusions": self.exclusions.to_json(orient="table"),
            "correction_periods": _correction_periods_to_json(self.correction_periods),
        }

        return json.dumps(payload, allow_nan=False)

    @classmethod
    def from_json(cls, s):
        """Rebuild from ``to_json()`` output.

        Args:
            s: JSON string written by ``to_json()``.

        Returns:
            The rebuilt ``CorrectionResult``.

        Raises:
            ValueError: The payload's ``schema_version`` is not supported.
        """
        payload = json.loads(s)
        header = payload["header"]

        if header["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {header['schema_version']}; expected {SCHEMA_VERSION}."
            )

        tz = header["tz"]
        corrected = _read_table(payload["corrected"], tz)
        cg_usage = _read_table(payload["cg_usage"], tz)
        ledger = exclusions.read_table_json(payload["exclusions"])

        result = cls(
            meter_id=header["meter_id"],
            corrected=corrected,
            cg_usage=cg_usage,
            exclusions=ledger,
            settings=header["settings"],
            covered_window=coerce_window(header["covered_window"], tz),
            tz=tz,
            model_frequency=header["model_frequency"],
            fingerprint=header["fingerprint"],
            correction_periods=_correction_periods_from_json(payload["correction_periods"], tz),
            cg_ids=header["cg_ids"],
        )

        return result


def _read_table(payload, tz):
    frame = pd.read_json(StringIO(payload), orient="table")
    frame["id"] = frame["id"].astype(str)

    if "datetime" in frame.columns:
        frame["datetime"] = (
            pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert(tz).dt.as_unit("ns")
        )

    return frame


def _correction_periods_to_json(correction_periods):
    payload = [[start.isoformat(), end.isoformat()] for start, end in correction_periods]

    return payload


def _correction_periods_from_json(payload, tz):
    correction_periods = [
        (pd.Timestamp(start).tz_convert(tz), pd.Timestamp(end).tz_convert(tz))
        for start, end in payload
    ]

    return correction_periods


def correct_reporting(
    selection,
    treatment,
    pool,
    treatment_id,
    settings=None,
    period=None,
    prior=None,
    read_boundaries=None,
):
    """Correct one treatment meter's reporting-period model with its selected
    comparison group.

    The meter is corrected at its RECEIVED cadence: an hourly or daily treatment
    at its native rows, a billing treatment at its inferred read periods. A
    billing meter rides a contiguous daily substrate; the substrate is
    partitioned into read periods and the meter's own rows plus the
    comparison-pool matrices are aggregated to those periods (sum observed and
    modeled, quadrature uncertainty) before the difference-in-differences kernel
    runs. Within a read the treatment and the pool are summed over the same
    days, so a day where the treatment lacks an observed or modeled value
    leaves every sum. A same-or-finer comparison pool is aggregated up to the
    treatment's cadence the same way.

    ``treatment`` and ``pool`` are whole populations (they hold the data, the
    timezone and model frequency validation, and the prediction cache), while
    ``treatment_id`` singles out the meter to correct. The comparison group is
    the raw cluster membership behind that meter's weights row, pruned to the
    pool meters that predict over the reporting period; only those columns are
    predicted and only the meter's own rows are emitted. Reporting coverage, for
    the meter and its comparison-group members alike, is judged over the
    meter's own reporting span (the extent of its reporting index), so neither
    the verdict nor the membership depends on other treatment meters.

    The correction always recomputes over the full cumulative reporting window:
    past point corrections reproduce identically, while correlation and
    uncertainty reflect the full period. ``period`` only filters the emitted
    rows, so a timestep's corrected value never depends on it. A
    ``prior`` result freezes the comparison group: its members are reused as
    they are, with no coverage prune (observed data-quality enforcement still
    applies) and no new entrants, so a pool meter whose reporting lags the
    treatment's growing window cannot change the group and move points already
    reported. Overlapping timesteps must match the prior's
    point corrections exactly, and a billing extension also freezes the prior's
    confirmed read periods so only the trailing period can be re-drawn.

    Treatment observed uncertainty never enters the correction math.

    A comparison-group member absent from ``pool``, and a pool meter whose
    reporting prediction raises, are dropped and ledgered in every call that
    selects them.

    Per-timestep failures never raise. Where the correction kernel's mask kills a
    timestep, ``corrected`` and ``corrected_unc`` are NaN and the row is still
    emitted; ``compute_savings`` reports it through ``coverage``.

    Args:
        selection: ``ComparisonGroupSelection`` for this treatment/pool pair.
        treatment: ``TreatmentGroup`` with reporting data attached.
        pool: ``ComparisonPool`` with reporting data attached; its model frequency
            must be same-or-finer than the treatment's.
        treatment_id: id of the treatment meter to correct.
        settings: ``CGCorrectionSettings``; defaults to the package defaults.
        period: optional ``(start, end)`` bounds filtering the emitted rows.
        prior: optional ``CorrectionResult`` for this same meter and selection
            from a smaller reporting window.
        read_boundaries: optional list of period start datetimes overriding
            read-period inference for a billing meter.

    Raises:
        MeterCorrectionError: the meter cannot be corrected at all: absent from
            the treatment population or from the selection, all-NaN cluster
            weights, fewer than ``MIN_CG_METERS`` usable comparison-group meters,
            missing reporting data, a failed reporting prediction, reporting
            coverage below ``min_reporting_coverage`` over its own reporting span, an
            eemeter observed disqualification on the reporting data, an
            unrecoverable billing read cadence, or a frozen comparison-group
            member that no longer predicts or whose reporting observed data is
            disqualified. The ledger rows recording the
            drop, including the pool-side rows behind a short comparison group,
            ride on the exception's ``exclusions`` attribute.
        ValueError: an analysis-level problem: a timezone or fingerprint
            mismatch, a comparison pool coarser than the treatment, an invalid
            period, a ``prior`` belonging to a different meter, made against a
            different selection, or carrying no comparison group, or a ``prior``
            whose overlapping point corrections disagree.

    Returns:
        ``CorrectionResult``.
    """
    if settings is None:
        settings = CGCorrectionSettings()

    if treatment.tz != pool.tz:
        raise ValueError(
            f"treatment timezone {treatment.tz} does not match pool timezone {pool.tz}."
        )

    if selection.tz != treatment.tz:
        raise ValueError(
            f"selection timezone {selection.tz} does not match treatment timezone {treatment.tz}."
        )

    check_model_frequency_fineness(treatment, pool)

    selection.verify_fingerprint()

    tz = treatment.tz
    model_frequency = treatment.model_frequency
    treatment_id = str(treatment_id)

    if period is not None:
        start, end = period
        if start is not None and end is not None and _localize(start, tz) > _localize(end, tz):
            raise ValueError(f"invalid period: start {start} is after end {end}.")

    if prior is not None and prior.meter_id != treatment_id:
        raise ValueError(
            f"prior correction belongs to meter {prior.meter_id}, not {treatment_id}."
        )

    if prior is not None and prior.fingerprint != selection.fingerprint:
        raise ValueError(
            "prior correction was made against a different selection (fingerprint "
            f"{prior.fingerprint}, selection {selection.fingerprint})."
        )

    if prior is not None and not prior.cg_ids:
        raise ValueError("prior correction carries no comparison group to reuse.")

    ledger = exclusions.empty_ledger()

    if treatment.meter(treatment_id) is None:
        ledger = exclusions.append(
            ledger,
            [treatment_id],
            "correction",
            "correction_guard",
            "treatment meter absent from the treatment population",
        )

        raise _meter_error(ledger, treatment_id)

    weights = selection.treatment_weights

    if treatment_id not in weights.index:
        ledger = exclusions.append(
            ledger,
            [treatment_id],
            "correction",
            "correction_guard",
            "treatment meter absent from selection",
        )

        raise _meter_error(ledger, treatment_id)

    weights_row = weights.loc[treatment_id]

    if weights_row.isna().all():
        ledger = exclusions.append(
            ledger, [treatment_id], "correction", "correction_guard", "all-NaN cluster weights"
        )

        raise _meter_error(ledger, treatment_id)

    window = _reporting_span(treatment.meter(treatment_id))

    t_kept, ledger = _observed_dq_exclusions(treatment, [treatment_id], ledger)
    t_kept, ledger = _coverage_exclusions(
        treatment, t_kept, ledger, window, settings.min_reporting_coverage
    )
    t_usable, ledger = _usable_reporting_ids(treatment, t_kept, ledger, "correction_guard")

    if not t_usable:
        raise _meter_error(ledger, treatment_id)

    membership, active = _cg_membership(selection, treatment_id)

    if prior is not None:
        frozen = set(prior.cg_ids)
        membership = [(mid, label) for mid, label in membership if mid in frozen]

    member_ids = [mid for mid, _ in membership]
    p_kept, ledger = _observed_dq_exclusions(pool, member_ids, ledger)

    if prior is None:
        p_kept, ledger = _coverage_exclusions(
            pool, p_kept, ledger, window, settings.min_reporting_coverage
        )

    p_usable, ledger = _usable_reporting_ids(pool, p_kept, ledger, "reporting_data")

    if prior is not None and len(p_usable) < len(member_ids):
        lost = [mid for mid in member_ids if mid not in set(p_usable)]
        ledger = exclusions.append(
            ledger,
            [treatment_id],
            "correction",
            "correction_guard",
            "frozen comparison group cannot be reproduced",
            detail=f"prior members no longer available: {lost}",
        )

        raise _meter_error(ledger, treatment_id)

    # the kernel's within-cluster sums are order-sensitive at the last ulp, so the
    # comparison-group columns keep their membership-walk order throughout
    usable = set(p_usable)
    selected = [(mid, label) for mid, label in membership if mid in usable]
    cg_ids = [mid for mid, _ in selected]

    if len(cg_ids) < MIN_CG_METERS:
        ledger = exclusions.append(
            ledger,
            [treatment_id],
            "correction",
            "correction_guard",
            f"fewer than {MIN_CG_METERS} comparison-group meters available",
            detail=f"{len(cg_ids)} comparison-group meters with reporting predictions",
        )

        raise _meter_error(ledger, treatment_id)

    labels = [label for _, label in selected]
    cg_label = np.array(labels, dtype=np.float64)
    t_weight = _cluster_weights(active, labels)

    p_index, _, p_obs, p_mod, p_mod_unc, p_obs_unc = pool.prediction_matrices(
        "reporting", ids=cg_ids
    )
    finer_pool = FINENESS[pool.model_frequency] > FINENESS[model_frequency]

    own = treatment.prediction_frames("reporting", ids=[treatment_id])[treatment_id]
    own_obs_unc = treatment.observed_unc_at(treatment_id, own.index)

    correction_periods = []

    if model_frequency in ("hourly", "daily"):
        out_index = own.index.as_unit("ns")
        n_rows = len(out_index)
        oTr = own["observed"].to_numpy(dtype=np.float64)
        mTr = own["modeled"].to_numpy(dtype=np.float64)
        mTr_unc = own["modeled_unc"].to_numpy(dtype=np.float64)
        oTr_unc_out = own_obs_unc

        if finer_pool:
            codes = _period_codes(_ns(p_index), _native_edges(out_index))
            oCGr, mCGr, mCGr_unc, oCGr_unc = _reduce_pool(
                p_obs, p_mod, p_mod_unc, p_obs_unc, codes, n_rows
            )
        else:
            row_map = p_index.get_indexer(out_index)
            present = row_map >= 0
            oCGr = _align_to_index(p_obs, row_map, present, n_rows)
            mCGr = _align_to_index(p_mod, row_map, present, n_rows)
            mCGr_unc = _align_to_index(p_mod_unc, row_map, present, n_rows)
            oCGr_unc = _align_to_index(p_obs_unc, row_map, present, n_rows)
    elif model_frequency == "billing":
        prior_periods = None
        if prior is not None:
            prior_periods = prior.correction_periods

        if read_boundaries is not None:
            rp = periods_from_boundaries(own.index, read_boundaries)
        else:
            rp = infer_read_periods(own[["observed"]], prior_periods)

        if rp.periods is None:
            ledger = exclusions.append(
                ledger, [treatment_id], "correction", "reporting_data", rp.ledger_reason
            )

            raise _meter_error(ledger, treatment_id)

        correction_periods = rp.periods
        edges = _billing_edges(correction_periods)
        n_periods = len(correction_periods)
        out_index = pd.DatetimeIndex([start for start, _ in correction_periods]).as_unit("ns")

        own_obs = own["observed"].to_numpy(dtype=np.float64)
        own_mod = own["modeled"].to_numpy(dtype=np.float64)
        own_finite = np.isfinite(own_obs) & np.isfinite(own_mod)
        own_codes = np.where(own_finite, _period_codes(_ns(own.index), edges), -1)
        oTr = _sum_to_periods(own_obs, own_codes, n_periods)
        mTr = _sum_to_periods(own_mod, own_codes, n_periods)
        mTr_unc = _quadrature_to_periods(own["modeled_unc"].to_numpy(), own_codes, n_periods)
        oTr_unc_out = _quadrature_to_periods(own_obs_unc, own_codes, n_periods)

        own_days = _ns(own.index.normalize())[own_finite]
        pool_codes = _restrict_to_days(_period_codes(_ns(p_index), edges), p_index, own_days)
        oCGr, mCGr, mCGr_unc, oCGr_unc = _reduce_pool(
            p_obs, p_mod, p_mod_unc, p_obs_unc, pool_codes, n_periods
        )
    else:
        raise ValueError(f"Unknown model_frequency {model_frequency!r}.")

    cg_corr = _cg_correlations(p_obs_unc, oCGr, mCGr)

    mTrc, mTrc_unc, mask = model_correction_matrix(
        mTr,
        oCGr,
        mCGr,
        mTr_unc,
        oCGr_unc,
        mCGr_unc,
        cg_corr,
        cg_label,
        t_weight,
        settings,
    )

    corrected = pd.DataFrame(
        {
            "id": treatment_id,
            "datetime": out_index,
            "observed": oTr,
            "modeled": mTr,
            "modeled_unc": mTr_unc,
            "corrected": mTrc,
            "corrected_unc": mTrc_unc,
            "observed_unc": oTr_unc_out,
        }
    )

    cg_usage = pd.DataFrame(
        {
            "id": treatment_id,
            "datetime": out_index,
            "cg_usage_fraction": mask.sum(axis=1) / len(cg_ids),
            "clusters_used": _clusters_used(mask, cg_label),
        }
    )

    if prior is not None:
        _check_prior(corrected, prior, model_frequency)

    keep = _period_mask(corrected["datetime"], period, tz)
    corrected = corrected[keep].reset_index(drop=True)
    cg_usage = cg_usage[keep].reset_index(drop=True)

    if corrected.empty:
        covered_window = None
    else:
        covered_window = (corrected["datetime"].min(), corrected["datetime"].max())

    result = CorrectionResult(
        meter_id=treatment_id,
        corrected=corrected,
        cg_usage=cg_usage,
        exclusions=ledger,
        settings=settings.model_dump(mode="json"),
        covered_window=covered_window,
        tz=tz,
        model_frequency=model_frequency,
        fingerprint=selection.fingerprint,
        correction_periods=correction_periods,
        cg_ids=cg_ids,
    )

    return result
