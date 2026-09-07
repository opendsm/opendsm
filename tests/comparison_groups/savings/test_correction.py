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

import copy
import os

import numpy as np
import pandas as pd
import pytest

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.population import ComparisonPool, TreatmentGroup
from opendsm.comparison_groups.selection import select_comparison_group
from opendsm.comparison_groups.random_sampling.settings import Settings as RandomSamplingSettings
from opendsm.comparison_groups.savings.correction import (
    CorrectionResult,
    _billing_edges,
    cg_member_ids,
    _cg_membership,
    _check_prior,
    _cluster_weights,
    _column_correlations,
    _native_edges,
    _period_codes,
    _reduce_to_periods,
    _restrict_to_days,
    correct_reporting,
)
from opendsm.comparison_groups.savings.savings import compute_savings
from opendsm.comparison_groups.savings.settings import CGCorrectionSettings
from opendsm.eemeter.common.warnings import EEMeterWarning
from .equivalence_env import (
    VARIANT_NAMES,
    _manual_clustering_selection,
    _mixed_selection,
    load_equivalence_snapshot,
    write_models_fixture,
)



_CORRECTED_COLUMNS = [
    "id",
    "datetime",
    "observed",
    "modeled",
    "modeled_unc",
    "corrected",
    "corrected_unc",
    "observed_unc",
]

_LEDGER_COLUMNS = ["id", "stage", "origin", "reason", "detail"]

# DST fall-back day in America/Chicago for the 2019 ComStock reporting year.
_DST_FALLBACK_DATE = pd.Timestamp("2019-11-03").date()

# Split point for the incremental (Jan–Jun then full-year) tests.
_MIDYEAR = pd.Timestamp("2019-07-01", tz="America/Chicago")

# Variants whose treatment reads its prediction frames at float64 in both the
# pinned batch output and the per-meter path, so equivalence holds to 1e-9
# relative: the pinned models pass through a JSON round trip, and library
# versions differ in their summation and t-quantile kernels in the last few
# digits, so bit equality is not portable while anything larger would be a
# real change.
_EXACT_RTOL = 1e-9

_EXACT_EQUIVALENCE_VARIANTS = (
    "billing_cg_clustering",
    "billing_pool_observed_unc",
    "billing_imm",
)

# Absolute floor for the hourly/daily equivalence comparison, in kWh: those
# variants round the treatment through float32 buffers in the pinned batch
# output, and a purely relative tolerance is meaningless on a near-zero row.
_EQUIVALENCE_ATOL = 1e-6


# ── builders ─────────────────────────────────────────────────────────────────


def _reporting_map(df_r, ids, cutoff=None):
    reporting = {}

    for mid in ids:
        raw = df_r.xs(mid, level="id")
        if cutoff is not None:
            raw = raw[raw.index < cutoff]
        reporting[str(mid)] = raw.reset_index()

    return reporting


def _population_meters(comstock, population, reporting=True):
    """A ``from_fit_models`` mapping reusing a population's already-fitted models
    with the ComStock frames behind them, so a variant population is rebuilt
    without refitting."""
    granularity = population.granularity
    meters = {}

    for mid, rec in population._meters.items():
        entry = {"model": rec.model, "baseline_df": comstock.baseline(granularity, mid)}
        if reporting:
            entry["reporting_df"] = comstock.reporting(granularity, mid)
        meters[mid] = entry

    return meters


def _disqualified_copy(model):
    """A copy of a fitted model carrying a constructed disqualification, so its
    prediction fails."""
    disqualified = copy.deepcopy(model)
    disqualified.disqualification = [
        EEMeterWarning(
            qualified_name="eemeter.model_fit_metrics",
            description="constructed disqualification",
            data=None,
        )
    ]

    return disqualified


def _rewrap_population(comstock, population, df_r, cutoff=None):
    """``_population_meters`` with a (possibly truncated) reporting slice taken
    straight from ``df_r``."""
    meters = _population_meters(comstock, population, reporting=False)

    for mid, entry in meters.items():
        raw = df_r.xs(int(mid), level="id")
        if cutoff is not None:
            raw = raw[raw.index < cutoff]
        entry["reporting_df"] = raw.reset_index()

    return meters


def _largest_cluster_ids(membership):
    """The pool ids of the largest cluster in a comparison-group membership
    walk."""
    labels = np.array([label for _, label in membership])
    unique, counts = np.unique(labels, return_counts=True)
    big_label = unique[np.argmax(counts)]
    ids = [mid for mid, label in membership if label == big_label]

    return ids


def _hand_aggregate_over_periods(series, periods):
    """Independent per-period sum (finite reads only, NaN when none), stepping
    period boundaries from each read's own start up to the next read's start
    (or one day past the final read's end)."""
    values = []
    for i, (start, end) in enumerate(periods):
        if i + 1 < len(periods):
            stop = periods[i + 1][0]
        else:
            stop = end + pd.Timedelta(days=1)

        span = series[(series.index >= start) & (series.index < stop)].dropna()
        if len(span):
            values.append(span.sum())
        else:
            values.append(np.nan)

    return np.array(values)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("GENERATE_FIXTURES"),
    reason="regenerates the committed equivalence models fixture; run manually",
)
def test_generate_equivalence_models(_comstock_daily_all, _comstock_monthly_all, _comstock_hourly_all):
    """Refit every equivalence variant and rewrite the pinned models fixture.

    GENERATE_FIXTURES=1 pytest -k generate_equivalence_models

    The equivalence snapshot pins outputs of these models, so regenerate it
    alongside whenever this runs.
    """
    write_models_fixture(_comstock_daily_all, _comstock_monthly_all, _comstock_hourly_all)


@pytest.fixture(scope="module")
def daily_data(comstock, model_bank):
    df_b, df_r = comstock.frames("daily")
    ids = sorted(df_b.index.get_level_values("id").unique())
    bundle = {
        "df_b": df_b,
        "df_r": df_r,
        "treatment_ids": ids[:8],
        "pool_ids": ids[8:68],
    }

    return bundle


@pytest.fixture(scope="module")
def daily_env(variant_env):
    """Real ComStock daily treatment (8) / pool (60) populations with a clustering
    selection. Reporting data is the full 2019 year."""
    return variant_env("daily_cg_clustering")


@pytest.fixture(scope="module")
def daily_results(daily_env):
    """One ``CorrectionResult`` per treatment meter — the caller-owned loop the
    per-meter API expects."""
    results = {
        tid: correct_reporting(
            daily_env["selection"], daily_env["treatment"], daily_env["pool"], tid
        )
        for tid in daily_env["treatment"].ids
    }

    return results


@pytest.fixture(scope="module")
def billing_env(variant_env):
    """Real ComStock monthly treatment (6) / pool (40) populations with a
    clustering selection."""
    return variant_env("billing_cg_clustering")


@pytest.fixture(scope="module")
def mixed_hourly_daily_env(variant_env):
    """A finer (hourly) comparison pool against a coarser (daily) treatment:
    the same underlying ComStock meters, disjoint ids, each fit at its own
    native granularity."""
    env = variant_env("hourly_pool_daily_treatment")
    results = {
        tid: correct_reporting(env["selection"], env["treatment"], env["pool"], tid)
        for tid in env["treatment"].ids
    }
    bundle = dict(env, results=results)

    return bundle


@pytest.fixture(scope="module")
def mixed_hourly_billing_env(comstock, model_bank):
    """A finer (hourly) comparison pool against a coarser (billing) treatment,
    corrected at the treatment's inferred read-period cadence."""
    df_bh, df_rh = comstock.frames("hourly")
    df_bm, df_rm = comstock.frames("billing")
    ids = sorted(df_bm.index.get_level_values("id").unique())

    treatment = TreatmentGroup.from_fit_models(comstock.meters(model_bank, "billing", ids[:2]))
    pool = ComparisonPool.from_fit_models(comstock.meters(model_bank, "hourly", ids[2:8]))
    selection = _mixed_selection(treatment, pool)
    results = {
        tid: correct_reporting(selection, treatment, pool, tid) for tid in treatment.ids
    }
    bundle = {
        "treatment": treatment,
        "pool": pool,
        "selection": selection,
        "results": results,
        "df_rh": df_rh,
    }

    return bundle


@pytest.fixture(scope="module")
def mixed_daily_billing_env(comstock, model_bank):
    """A finer (daily) comparison pool against a coarser (billing) treatment,
    corrected at the treatment's inferred read-period cadence."""
    df_bd, df_rd = comstock.frames("daily")
    df_bm, df_rm = comstock.frames("billing")
    ids = sorted(df_bm.index.get_level_values("id").unique())

    treatment = TreatmentGroup.from_fit_models(comstock.meters(model_bank, "billing", ids[:2]))
    pool = ComparisonPool.from_fit_models(comstock.meters(model_bank, "daily", ids[2:8]))
    selection = _mixed_selection(treatment, pool)
    results = {
        tid: correct_reporting(selection, treatment, pool, tid) for tid in treatment.ids
    }
    bundle = {
        "treatment": treatment,
        "pool": pool,
        "selection": selection,
        "results": results,
    }

    return bundle


@pytest.fixture(scope="module")
def guard_env(comstock, model_bank):
    """A small, cheap billing population pair plus a hand-built valid selection
    (all pool meters in one cluster) for guard-path tests."""
    df_b, df_r = comstock.frames("billing")
    ids = sorted(df_b.index.get_level_values("id").unique())
    treatment = TreatmentGroup.from_fit_models(comstock.meters(model_bank, "billing", ids[:2]))
    pool = ComparisonPool.from_fit_models(comstock.meters(model_bank, "billing", ids[2:10]))

    cluster_of = {p: 0 for p in pool.ids}
    weights = {t: {"pct_cluster_0": 1.0} for t in treatment.ids}
    selection = _manual_clustering_selection(
        treatment.ids, pool.ids, cluster_of, weights, treatment.tz
    )
    bundle = {
        "treatment": treatment,
        "pool": pool,
        "selection": selection,
        "cluster_of": cluster_of,
        "comstock": comstock,
        "df_b": df_b,
        "df_r": df_r,
        "ids": ids,
    }

    return bundle


@pytest.fixture(scope="module")
def daily_dq_env(comstock, model_bank):
    """A small real ComStock daily population (2 treatment, 6 pool) with a
    hand-built one-cluster selection, for correction-stage reporting DQ tests.
    Pinned models and the raw reporting frame are exposed so variant reporting
    (truncated, flat, reduced) can be swapped in without refitting."""
    df_b, df_r = comstock.frames("daily")
    ids = comstock.ids("daily")[:8]
    treatment_ids = [str(i) for i in ids[:2]]
    pool_ids = [str(i) for i in ids[2:8]]
    records = comstock.meters(model_bank, "daily", ids)

    cluster_of = {p: 0 for p in pool_ids}
    weights = {t: {"pct_cluster_0": 1.0} for t in treatment_ids}
    selection = _manual_clustering_selection(
        treatment_ids, pool_ids, cluster_of, weights, "America/Chicago"
    )
    bundle = {
        "records": records,
        "df_r": df_r,
        "treatment_ids": treatment_ids,
        "pool_ids": pool_ids,
        "selection": selection,
    }

    return bundle


def _meters_with_overrides(records, ids, overrides):
    """The pinned-model records for ``ids``, with the reporting frame replaced
    where ``overrides`` supplies one."""
    meters = {}

    for mid in ids:
        entry = dict(records[mid])
        if mid in overrides:
            entry["reporting_df"] = overrides[mid]
        meters[mid] = entry

    return meters


def _build_daily_dq(daily_dq_env, treatment_overrides=None, pool_overrides=None):
    records = daily_dq_env["records"]
    treatment = TreatmentGroup.from_fit_models(
        _meters_with_overrides(records, daily_dq_env["treatment_ids"], treatment_overrides or {}
        )
    )
    pool = ComparisonPool.from_fit_models(
        _meters_with_overrides(records, daily_dq_env["pool_ids"], pool_overrides or {})
    )

    return treatment, pool


def _reporting_override(df_r, mid, transform):
    """One meter's reporting frame, put through ``transform``."""
    raw = df_r.xs(int(mid), level="id").reset_index()
    reporting = transform(raw)

    return reporting


@pytest.fixture(scope="module")
def billing_dq_env(comstock, model_bank):
    """A small real ComStock billing population (2 treatment, 6 pool) with a
    hand-built one-cluster selection, mirroring ``daily_dq_env`` at billing
    granularity for the same correction-stage reporting DQ tests."""
    df_b, df_r = comstock.frames("billing")
    ids = sorted(df_b.index.get_level_values("id").unique())[:8]
    treatment_ids = [str(i) for i in ids[:2]]
    pool_ids = [str(i) for i in ids[2:8]]

    records = comstock.meters(model_bank, "billing", ids)

    cluster_of = {p: 0 for p in pool_ids}
    weights = {t: {"pct_cluster_0": 1.0} for t in treatment_ids}
    selection = _manual_clustering_selection(
        treatment_ids, pool_ids, cluster_of, weights, "America/Chicago"
    )
    bundle = {
        "records": records,
        "df_r": df_r,
        "treatment_ids": treatment_ids,
        "pool_ids": pool_ids,
        "selection": selection,
    }

    return bundle


def _build_billing_dq(billing_dq_env, treatment_overrides=None, pool_overrides=None):
    records = billing_dq_env["records"]
    treatment = TreatmentGroup.from_fit_models(
        _meters_with_overrides(records, billing_dq_env["treatment_ids"], treatment_overrides or {}
        )
    )
    pool = ComparisonPool.from_fit_models(
        _meters_with_overrides(records, billing_dq_env["pool_ids"], pool_overrides or {}
        )
    )

    return treatment, pool


@pytest.fixture(scope="module")
def hourly_dq_env(comstock, model_bank):
    """A small real ComStock hourly population (2 treatment, 6 pool) with a
    hand-built one-cluster selection, mirroring ``daily_dq_env`` at hourly
    granularity for the same correction-stage reporting DQ tests."""
    df_b, df_r = comstock.frames("hourly")
    ids = sorted(df_b.index.get_level_values("id").unique())[:8]
    treatment_ids = [str(i) for i in ids[:2]]
    pool_ids = [str(i) for i in ids[2:8]]

    records = comstock.meters(model_bank, "hourly", ids)

    cluster_of = {p: 0 for p in pool_ids}
    weights = {t: {"pct_cluster_0": 1.0} for t in treatment_ids}
    selection = _manual_clustering_selection(
        treatment_ids, pool_ids, cluster_of, weights, "America/Chicago"
    )
    bundle = {
        "records": records,
        "df_r": df_r,
        "treatment_ids": treatment_ids,
        "pool_ids": pool_ids,
        "selection": selection,
    }

    return bundle


def _build_hourly_dq(hourly_dq_env, treatment_overrides=None, pool_overrides=None):
    records = hourly_dq_env["records"]
    treatment = TreatmentGroup.from_fit_models(
        _meters_with_overrides(records, hourly_dq_env["treatment_ids"], treatment_overrides or {}
        )
    )
    pool = ComparisonPool.from_fit_models(
        _meters_with_overrides(records, hourly_dq_env["pool_ids"], pool_overrides or {}
        )
    )

    return treatment, pool


# ── daily end-to-end (slow) ──────────────────────────────────────────────────


def test_daily_end_to_end_schema(daily_results, daily_env):
    """One call per treatment meter, one result per meter: each frame carries
    that meter's id only, over its own reporting year."""
    assert set(daily_results) == set(daily_env["treatment"].ids)

    for tid, result in daily_results.items():
        corrected = result.corrected

        assert result.meter_id == tid
        assert list(corrected.columns) == _CORRECTED_COLUMNS
        assert set(corrected["id"].unique()) == {tid}
        assert corrected["datetime"].dt.tz is not None
        assert result.tz == "America/Chicago"
        assert result.granularity == "daily"
        assert len(corrected) == 365


def test_daily_corrected_is_finite_and_covers_full_year(daily_results):
    for result in daily_results.values():
        corrected = result.corrected

        assert np.isfinite(corrected["corrected"].to_numpy()).all()
        start, end = result.covered_window
        assert start == pd.Timestamp("2019-01-01", tz="America/Chicago")
        assert end == pd.Timestamp("2019-12-31", tz="America/Chicago")


@pytest.mark.regression
def test_daily_correction_regression_totals(daily_results):
    """Point corrections are deterministic given the fitted models and clustering
    settings, so the totals over the caller's loop are pinned."""
    corrected = pd.concat(
        [result.corrected for result in daily_results.values()], ignore_index=True
    )

    total_corrected = float(np.nansum(corrected["corrected"].to_numpy()))
    total_modeled = float(np.nansum(corrected["modeled"].to_numpy()))

    assert total_corrected == pytest.approx(3386073.02, rel=1e-3)
    assert total_modeled == pytest.approx(5297171.84, rel=1e-3)


def test_daily_cg_usage_fraction_reflects_finite_mask(daily_results, daily_env, daily_data, comstock):
    """``cg_usage_fraction`` is the fraction of a treatment meter's comparison
    meters retained (finite, no outlier rejection). The fixture's comparison
    data is fully finite, so every meter is used (fraction 1.0); NaN-ing two of
    one treatment meter's comparison meters at three timesteps drops those
    timesteps to ``(n_cg - 2) / n_cg`` while the rest stay at 1.0."""
    for result in daily_results.values():
        fractions = result.cg_usage["cg_usage_fraction"].to_numpy()

        assert np.isfinite(fractions).all()
        assert (fractions >= 0.0).all()
        assert (fractions <= 1.0).all()
        assert set(result.cg_usage["id"].unique()) == {result.meter_id}
        # fully-finite comparison data means every comparison meter is used
        assert np.allclose(fractions, 1.0)

    treatment = daily_env["treatment"]
    pool = daily_env["pool"]
    selection = daily_env["selection"]
    tid = treatment.ids[0]

    # pick two meters from the meter's largest contributing cluster so the
    # cluster keeps at least three finite meters (no degradation) after corruption
    membership = _cg_membership(selection, tid)
    n_cg = len(membership)
    big_ids = _largest_cluster_ids(membership)
    assert len(big_ids) >= 5
    corrupt_ids = big_ids[:2]

    injected_pool = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, pool, daily_data["df_r"])
    )
    pred = injected_pool._ensure_pred("reporting")
    corrupt_dates = list(pred[corrupt_ids[0]].index[10:13])

    for cid in corrupt_ids:
        pred[cid].loc[corrupt_dates, "observed"] = np.nan

    injected = correct_reporting(selection, treatment, injected_pool, tid)

    usage = injected.cg_usage.set_index("datetime")["cg_usage_fraction"]
    expected = (n_cg - 2) / n_cg

    for date in corrupt_dates:
        assert usage.loc[date] == pytest.approx(expected)

    uncorrupted = [d for d in usage.index if d not in corrupt_dates]
    assert np.allclose(usage.loc[uncorrupted].to_numpy(), 1.0)


@pytest.fixture(scope="module")
def degraded_daily(daily_env, daily_data, comstock):
    """One treatment meter corrected against a pool whose largest contributing
    cluster is NaN-ed down to two finite meters on three days, so the matrix
    kernel degrades exactly those timesteps."""
    treatment = daily_env["treatment"]
    selection = daily_env["selection"]
    tid = treatment.ids[0]

    membership = _cg_membership(selection, tid)
    big_ids = _largest_cluster_ids(membership)
    assert len(big_ids) >= 5
    n_corrupt = len(big_ids) - 2
    corrupt_ids = big_ids[:n_corrupt]

    injected_pool = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, daily_env["pool"], daily_data["df_r"])
    )
    pred = injected_pool._ensure_pred("reporting")
    corrupt_dates = list(pred[corrupt_ids[0]].index[20:23])

    for cid in corrupt_ids:
        pred[cid].loc[corrupt_dates, "observed"] = np.nan

    result = correct_reporting(selection, treatment, injected_pool, tid)
    bundle = {"result": result, "corrupt_dates": corrupt_dates}

    return bundle


def test_daily_degraded_timestep_nans_corrected_and_zeroes_usage(degraded_daily):
    """When a treatment meter's cluster drops below three finite comparison
    meters at a timestep, the matrix kernel degrades that timestep: ``corrected``
    and ``corrected_unc`` go NaN there (finite everywhere else), and because no
    meter entered the point, ``cg_usage_fraction`` there is zero rather than the
    surviving finite fraction."""
    result = degraded_daily["result"]
    corrupt_dates = degraded_daily["corrupt_dates"]

    rows = result.corrected.set_index("datetime")
    corrected_series = rows["corrected"]
    unc_series = rows["corrected_unc"]

    for date in corrupt_dates:
        assert np.isnan(corrected_series.loc[date])
        assert np.isnan(unc_series.loc[date])

    finite_dates = [d for d in corrected_series.index if d not in corrupt_dates]
    assert np.isfinite(corrected_series.loc[finite_dates].to_numpy()).all()

    usage = result.cg_usage.set_index("datetime")["cg_usage_fraction"]
    for date in corrupt_dates:
        assert usage.loc[date] == 0.0

    uncorrupted = [d for d in usage.index if d not in corrupt_dates]
    assert np.allclose(usage.loc[uncorrupted].to_numpy(), 1.0)


def test_degraded_timestep_is_returned_and_lowers_savings_coverage(degraded_daily):
    """A per-timestep failure never raises: the NaN row is RETURNED with the rest
    of the meter's series, and ``compute_savings`` reports it through
    ``coverage`` — 0 at the degraded native timestep, and below 1 for the month
    holding it while every other month stays at 1."""
    result = degraded_daily["result"]
    corrupt_dates = degraded_daily["corrupt_dates"]

    assert len(result.corrected) == 365

    native = compute_savings(result).savings.set_index("period")
    monthly = compute_savings(result, aggregation="monthly").savings.set_index("period")

    assert (native.loc[corrupt_dates, "coverage"].to_numpy() == 0.0).all()

    month = corrupt_dates[0].strftime("%Y-%m")
    days_in_month = corrupt_dates[0].days_in_month
    other_months = [period for period in monthly.index if period != month]

    assert monthly.loc[month, "coverage"] == pytest.approx(
        (days_in_month - len(corrupt_dates)) / days_in_month
    )
    assert (monthly.loc[other_months, "coverage"].to_numpy() == 1.0).all()


def test_daily_json_roundtrip_preserves_frames_and_tz(daily_results):
    result = daily_results[next(iter(daily_results))]

    rebuilt = CorrectionResult.from_json(result.to_json())

    pd.testing.assert_frame_equal(result.corrected, rebuilt.corrected)
    pd.testing.assert_frame_equal(result.cg_usage, rebuilt.cg_usage)
    pd.testing.assert_frame_equal(result.exclusions, rebuilt.exclusions)
    assert rebuilt.meter_id == result.meter_id
    assert rebuilt.tz == result.tz
    assert rebuilt.granularity == result.granularity
    assert rebuilt.fingerprint == result.fingerprint
    assert rebuilt.covered_window == result.covered_window
    assert rebuilt.cg_ids == result.cg_ids


# ── billing end-to-end (slow) ────────────────────────────────────────────────


@pytest.mark.regression
def test_billing_end_to_end_monthly_timesteps(billing_env):
    results = {
        tid: correct_reporting(
            billing_env["selection"], billing_env["treatment"], billing_env["pool"], tid
        )
        for tid in billing_env["treatment"].ids
    }
    corrected = pd.concat([result.corrected for result in results.values()], ignore_index=True)

    for result in results.values():
        assert list(result.corrected.columns) == _CORRECTED_COLUMNS
        assert result.granularity == "billing"
        # 12 monthly rows per treatment meter
        assert len(result.corrected) == 12

    # Inferred boundaries land within a day or two of the calendar month start;
    # a near-identical daily rate across a month boundary can shift detection
    # by a day.
    assert (corrected["datetime"].dt.day <= 2).all()
    assert np.isfinite(corrected["corrected"].to_numpy()).all()

    total_corrected = float(np.nansum(corrected["corrected"].to_numpy()))
    assert total_corrected == pytest.approx(2263284.60, rel=1e-3)


# ── mixed-granularity e2e (slow, ComStock) ───────────────────────────────────


def test_hourly_pool_daily_treatment_cadence_matches_received(mixed_hourly_daily_env):
    """A daily treatment corrects at its received (daily) cadence regardless of
    the comparison pool's finer (hourly) granularity."""
    for result in mixed_hourly_daily_env["results"].values():
        assert result.granularity == "daily"
        assert len(result.corrected) == 365
        assert np.isfinite(result.corrected["corrected"].to_numpy()).all()


def test_hourly_pool_aggregates_to_daily_treatment_periods(mixed_hourly_daily_env):
    """The hourly pool's own observed rows, reduced to the treatment's daily
    periods via the same reduction machinery ``correct_reporting`` uses, match
    an independent pandas day-resample of the pool meter's raw hourly
    predictions (min_count 1: a day with no finite hourly reads is NaN, never
    0)."""
    treatment = mixed_hourly_daily_env["treatment"]
    pool = mixed_hourly_daily_env["pool"]
    tid = treatment.ids[0]

    t_index = treatment._ensure_pred("reporting", ids=[tid])[tid].index
    cg_ids = cg_member_ids(mixed_hourly_daily_env["selection"], tid)
    p_index, _, p_obs, *_ = pool._prediction_matrices("reporting", ids=cg_ids)

    codes = _period_codes(p_index.as_unit("ns").asi8, _native_edges(t_index))
    aggregated = _reduce_to_periods(p_obs, codes, len(t_index), False)

    col = 0
    pool_series = pd.Series(p_obs[:, col].astype(np.float64), index=p_index)
    hand = pool_series.resample("D").sum(min_count=1).reindex(t_index)

    np.testing.assert_allclose(aggregated[:, col], hand.to_numpy(), equal_nan=True)


def test_hourly_pool_billing_treatment_cadence_matches_received(mixed_hourly_billing_env):
    """A billing treatment corrects at its own inferred read periods regardless
    of the comparison pool's finer (hourly) granularity: one row per period,
    labelled by the period start."""
    for result in mixed_hourly_billing_env["results"].values():
        assert result.granularity == "billing"
        assert len(result.corrected) == len(result.correction_periods)
        assert list(result.corrected["datetime"]) == [
            start for start, _ in result.correction_periods
        ]


def test_hourly_pool_aggregates_to_billing_treatment_read_periods(mixed_hourly_billing_env):
    """The hourly pool's own observed rows, reduced to a billing treatment's
    inferred read periods via the same reduction machinery ``correct_reporting``
    uses, match an independent per-period pandas sum over the pool meter's raw
    hourly predictions."""
    treatment = mixed_hourly_billing_env["treatment"]
    pool = mixed_hourly_billing_env["pool"]
    tid = treatment.ids[0]
    periods = mixed_hourly_billing_env["results"][tid].correction_periods

    cg_ids = cg_member_ids(mixed_hourly_billing_env["selection"], tid)
    p_index, _, p_obs, *_ = pool._prediction_matrices("reporting", ids=cg_ids)
    codes = _period_codes(p_index.as_unit("ns").asi8, _billing_edges(periods))
    aggregated = _reduce_to_periods(p_obs, codes, len(periods), False)

    col = 0
    pool_series = pd.Series(p_obs[:, col].astype(np.float64), index=p_index)
    hand = _hand_aggregate_over_periods(pool_series, periods)

    np.testing.assert_allclose(aggregated[:, col], hand, equal_nan=True)


@pytest.mark.slow
def test_hourly_pool_missing_read_period_degrades_to_nan_not_zero(mixed_hourly_billing_env, comstock):
    """A finer (hourly) pool NaN-ed across an entire read period for every
    member of the treatment's single cluster aggregates to NaN there (min_count
    1), leaving fewer than three finite comparison meters and degrading that
    period's ``corrected`` value to NaN rather than a summed-to-zero finite
    value."""
    treatment = mixed_hourly_billing_env["treatment"]
    pool = mixed_hourly_billing_env["pool"]
    selection = mixed_hourly_billing_env["selection"]
    df_rh = mixed_hourly_billing_env["df_rh"]
    tid = treatment.ids[0]
    periods = mixed_hourly_billing_env["results"][tid].correction_periods
    target_start, target_end = periods[len(periods) // 2]

    injected_pool = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, pool, df_rh)
    )
    pred = injected_pool._ensure_pred("reporting")
    for pid in injected_pool.ids:
        hours = pred[pid].index
        target = hours[(hours >= target_start) & (hours < target_end + pd.Timedelta(days=1))]
        pred[pid].loc[target, "observed"] = np.nan

    result = correct_reporting(selection, treatment, injected_pool, tid)

    rows = result.corrected.set_index("datetime")
    assert np.isnan(rows.loc[target_start, "corrected"])
    other = [d for d in rows.index if d != target_start]
    assert np.isfinite(rows.loc[other, "corrected"].to_numpy()).all()


def test_daily_pool_billing_treatment_cadence_matches_received(mixed_daily_billing_env):
    """A billing treatment corrects at its own inferred read periods regardless
    of the comparison pool's finer (daily) granularity."""
    for result in mixed_daily_billing_env["results"].values():
        assert result.granularity == "billing"
        assert len(result.corrected) == len(result.correction_periods)
        assert list(result.corrected["datetime"]) == [
            start for start, _ in result.correction_periods
        ]


def test_daily_pool_aggregates_to_billing_treatment_read_periods(mixed_daily_billing_env):
    """The daily pool's own observed rows, reduced to a billing treatment's
    inferred read periods via the same reduction machinery ``correct_reporting``
    uses, match an independent per-period pandas sum over the pool meter's raw
    daily predictions."""
    treatment = mixed_daily_billing_env["treatment"]
    pool = mixed_daily_billing_env["pool"]
    tid = treatment.ids[0]
    periods = mixed_daily_billing_env["results"][tid].correction_periods

    cg_ids = cg_member_ids(mixed_daily_billing_env["selection"], tid)
    p_index, _, p_obs, *_ = pool._prediction_matrices("reporting", ids=cg_ids)
    codes = _period_codes(p_index.as_unit("ns").asi8, _billing_edges(periods))
    aggregated = _reduce_to_periods(p_obs, codes, len(periods), False)

    col = 0
    pool_series = pd.Series(p_obs[:, col].astype(np.float64), index=p_index)
    hand = _hand_aggregate_over_periods(pool_series, periods)

    np.testing.assert_allclose(aggregated[:, col], hand, equal_nan=True)


# ── output schema (fast) ─────────────────────────────────────────────────────


def test_output_schema_and_tables(guard_env):
    tid = guard_env["treatment"].ids[0]

    result = correct_reporting(
        guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid
    )

    assert result.meter_id == tid
    assert list(result.corrected.columns) == _CORRECTED_COLUMNS
    assert list(result.cg_usage.columns) == ["id", "datetime", "cg_usage_fraction", "clusters_used"]
    assert list(result.exclusions.columns) == _LEDGER_COLUMNS
    assert set(result.tables) == {"corrected", "cg_usage", "exclusions"}
    assert result.corrected["id"].map(type).eq(str).all()
    assert set(result.corrected["id"].unique()) == {tid}
    # every pool meter sits in the one cluster and predicts, in selection order
    assert result.cg_ids == guard_env["pool"].ids


# ── comparison-group membership and weights (fast) ───────────────────────────


def test_cluster_weights_renormalize_when_a_selected_cluster_has_no_columns():
    """A treatment meter's active cluster with no usable comparison column
    contributes no label, so ``t_weight`` renormalizes to sum 1 over the clusters
    that ARE present, keeping the kernel's normalized point consistent with its
    raw-weight uncertainty (rather than passing a raw Sigma < 1)."""
    active = {0: 0.6, 1: 0.4}

    # only cluster 0 survived the usability walk, contributing two columns
    t_weight = _cluster_weights(active, [0, 0])

    assert t_weight.sum() == pytest.approx(1.0)
    assert np.allclose(t_weight, [1.0])


def test_column_correlations_over_a_subset_equal_the_full_pool_restriction(guard_env):
    """The correction computes the comparison-group correlation over one meter's
    columns rather than the whole pool. ``_column_correlations`` is per column
    over finite pairs, so a subset computation is identical to the full-pool one
    restricted to those columns — the per-meter restriction changes no number."""
    pool = guard_env["pool"]
    _, p_ids, observed, modeled, *_ = pool._prediction_matrices("reporting")

    full = _column_correlations(
        np.asarray(observed, dtype=np.float64), np.asarray(modeled, dtype=np.float64)
    )

    subset = [1, 3, 4]
    _, _, s_obs, s_mod, *_ = pool._prediction_matrices("reporting", ids=[p_ids[c] for c in subset])
    restricted = _column_correlations(
        np.asarray(s_obs, dtype=np.float64), np.asarray(s_mod, dtype=np.float64)
    )

    np.testing.assert_array_equal(restricted, full[subset])


def test_clusters_used_reflects_per_cluster_drop(guard_env, comstock):
    """``clusters_used`` counts, per read period, the clusters that retain at
    least three finite meters. Billing correction runs at read-period cadence, so
    degradation fires per read period. NaN-ing a whole read period of one
    3-meter cluster's meters drops that cluster below three finite pool meters
    there only, so ``clusters_used`` falls from 2 to 1 at that period and stays 2
    elsewhere."""
    treatment = guard_env["treatment"]
    pool = guard_env["pool"]
    pool_ids = pool.ids
    tid = treatment.ids[0]

    # cluster 1 gets the first 3 pool meters (bare minimum to survive); cluster
    # 0 gets the remaining 5 (never drops below 3 under a single corruption)
    cluster_of = {p: 0 for p in pool_ids}
    for p in pool_ids[:3]:
        cluster_of[p] = 1
    weights = {
        tid: {"pct_cluster_0": 0.6, "pct_cluster_1": 0.4},
        treatment.ids[1]: {"pct_cluster_0": 1.0},
    }
    selection = _manual_clustering_selection(
        treatment.ids, pool_ids, cluster_of, weights, treatment.tz
    )

    injected_pool = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, pool, guard_env["df_r"])
    )
    pred = injected_pool._ensure_pred("reporting")
    # NaN the whole March read period of one cluster-1 meter: its March period
    # aggregates to NaN (min_count 1), leaving cluster 1 with two finite meters
    substrate = pred[pool_ids[0]].index
    march = substrate[substrate.month == 3]
    pred[pool_ids[0]].loc[march, "observed"] = np.nan

    result = correct_reporting(selection, treatment, injected_pool, tid)

    usage = result.cg_usage.set_index("datetime")["clusters_used"]
    march_start = pd.Timestamp("2019-03-01", tz=treatment.tz)

    assert usage.loc[march_start] == 1
    other_dates = [d for d in usage.index if d != march_start]
    assert (usage.loc[other_dates] == 2).all()


def test_period_bounds_restrict_reported_window(guard_env):
    """``period`` bounds trim the reported rows (computation still spans the full
    reporting window); the covered window reflects the trimmed output."""
    start = pd.Timestamp("2019-04-01", tz="America/Chicago")
    end = pd.Timestamp("2019-06-30", tz="America/Chicago")

    result = correct_reporting(
        guard_env["selection"],
        guard_env["treatment"],
        guard_env["pool"],
        guard_env["treatment"].ids[0],
        period=(start, end),
    )

    datetimes = result.corrected["datetime"]
    assert (datetimes >= start).all()
    assert (datetimes <= end).all()
    assert result.covered_window[0] >= start
    assert result.covered_window[1] <= end
    # April–June inclusive is three monthly billing rows
    assert len(result.corrected) == 3


# ── received-cadence reduction (fast) ────────────────────────────────────────


def test_reduce_to_periods_sums_finite_rows_with_min_count_one():
    """The reduction map sums a period's finite native rows per column. min_count
    is 1 everywhere: a period whose column has no finite native rows aggregates
    to NaN, never 0. Rows mapped to -1 (outside every period) contribute
    nowhere."""
    values = np.array([[2.0, 10.0], [3.0, 20.0], [np.nan, np.nan]])
    codes = np.array([0, 0, 1])

    summed = _reduce_to_periods(values, codes, 2, False)

    assert np.array_equal(summed[0], [5.0, 30.0])
    # second period's only native row is NaN -> NaN, not 0.0
    assert np.isnan(summed[1]).all()

    dropped = _reduce_to_periods(np.array([[5.0], [7.0]]), np.array([-1, 0]), 1, False)
    assert dropped[0, 0] == 7.0


def test_reduce_to_periods_quadrature_sums_uncertainty():
    """With ``square`` the reduction quadrature-sums per-row uncertainties
    (root of summed squares); an empty period is still NaN."""
    unc = np.array([[3.0], [4.0], [np.nan]])
    codes = np.array([0, 0, 1])

    quad = _reduce_to_periods(unc, codes, 2, True)

    assert quad[0, 0] == pytest.approx(5.0)
    assert np.isnan(quad[1, 0])


def test_native_edges_bin_finer_rows_into_coarse_periods():
    """A finer pool aggregates up to a coarser treatment's identity structure:
    ``_native_edges`` opens one period per treatment row, closed one cadence past
    the last, and ``_period_codes`` assigns each finer row to the period covering
    it. Rows outside the span map to -1 (dropped)."""
    daily = pd.date_range("2019-01-01", periods=3, freq="D", tz="America/Chicago")
    hourly = pd.date_range("2019-01-01", periods=72, freq="h", tz="America/Chicago")

    edges = _native_edges(daily)
    codes = _period_codes(hourly.as_unit("ns").asi8, edges)

    assert np.array_equal(np.bincount(codes, minlength=3), [24, 24, 24])

    outside = pd.DatetimeIndex([daily[0] - pd.Timedelta(hours=1), daily[-1] + pd.Timedelta(days=1)])
    assert np.array_equal(_period_codes(outside.as_unit("ns").asi8, edges), [-1, -1])


def test_restrict_to_days_drops_finer_rows_outside_the_treatment_days():
    """Pool rows whose local day is not among the treatment's finite days leave
    the reduction (code -1), so a read sums both sides over the same days."""
    index = pd.date_range("2019-01-01", periods=48, freq="h", tz="America/Chicago")
    codes = np.zeros(48, dtype=int)
    day_i8 = pd.DatetimeIndex([pd.Timestamp("2019-01-01", tz="America/Chicago")]).as_unit("ns").asi8

    restricted = _restrict_to_days(codes, index, day_i8)

    assert (restricted[:24] == 0).all()
    assert (restricted[24:] == -1).all()


def test_billing_correction_bins_treatment_to_read_periods(guard_env):
    """A billing treatment corrects at its inferred read periods: one corrected
    row per period, labelled by the period start, and a period's ``observed``
    equals the hand-summed daily-substrate observed over that period (the
    reduction-map aggregation)."""
    tid = guard_env["treatment"].ids[0]

    result = correct_reporting(
        guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid
    )

    periods = result.correction_periods
    assert periods

    own = guard_env["treatment"]._ensure_pred("reporting", ids=[tid])[tid]
    rows = result.corrected.set_index("datetime")

    assert list(rows.index) == [start for start, _ in periods]

    start, end = periods[len(periods) // 2]
    span = own["observed"][(own.index >= start) & (own.index <= end)]
    expected = float(np.nansum(span.to_numpy()))

    assert rows.loc[start, "observed"] == pytest.approx(expected)


def test_billing_read_with_a_missing_day_sums_observed_and_modeled_over_the_same_days(
    guard_env, comstock
):
    """A day missing observed inside a read leaves that read's modeled total and
    uncertainty too, so the read compares like with like rather than a partial
    observed total against a full modeled one."""
    tid = guard_env["treatment"].ids[0]
    treatment = TreatmentGroup.from_fit_models(
        _rewrap_population(comstock, guard_env["treatment"], guard_env["df_r"])
    )
    clean = correct_reporting(guard_env["selection"], treatment, guard_env["pool"], tid)
    own = treatment._ensure_pred("reporting", ids=[tid])[tid]
    period_start, period_end = clean.correction_periods[3]
    in_period = (own.index >= period_start) & (own.index <= period_end)
    hole = own.index[in_period][5]
    own.loc[hole, "observed"] = np.nan

    holed = correct_reporting(guard_env["selection"], treatment, guard_env["pool"], tid)

    assert holed.correction_periods == clean.correction_periods
    kept = in_period & (own.index != hole)
    row = holed.corrected.set_index("datetime").loc[period_start]
    assert row["observed"] == pytest.approx(own.loc[kept, "observed"].sum())
    assert row["modeled"] == pytest.approx(own.loc[kept, "modeled"].sum())
    assert row["modeled_unc"] == pytest.approx(np.sqrt((own.loc[kept, "modeled_unc"] ** 2).sum()))
    assert row["modeled"] < clean.corrected.set_index("datetime").loc[period_start, "modeled"]


def test_billing_missing_pool_period_degrades_to_nan_not_zero(guard_env, comstock):
    """A read period with no finite pool observed aggregates to NaN, never 0
    (min_count 1). NaN-ing a whole read period across every pool meter of the
    single cluster leaves fewer than three finite meters there, so ``corrected``
    is NaN at that period and finite elsewhere — a summed-to-zero period would
    instead keep the cluster alive and the row finite."""
    treatment = guard_env["treatment"]
    tid = treatment.ids[0]

    injected_pool = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, guard_env["pool"], guard_env["df_r"])
    )
    pred = injected_pool._ensure_pred("reporting")
    for pid in injected_pool.ids:
        substrate = pred[pid].index
        march = substrate[substrate.month == 3]
        pred[pid].loc[march, "observed"] = np.nan

    result = correct_reporting(guard_env["selection"], treatment, injected_pool, tid)

    rows = result.corrected.set_index("datetime")
    march_start = pd.Timestamp("2019-03-01", tz=treatment.tz)

    assert np.isnan(rows.loc[march_start, "corrected"])
    other = [d for d in rows.index if d != march_start]
    assert np.isfinite(rows.loc[other, "corrected"].to_numpy()).all()


# ── meter-level guards (fast) ────────────────────────────────────────────────


def _guard_absent_from_population(guard_env):
    """A treatment id the population never held."""
    selection = guard_env["selection"]
    treatment = guard_env["treatment"]
    pool = guard_env["pool"]
    tid = "not-a-real-meter"

    return selection, treatment, pool, tid


def _guard_absent_from_selection(guard_env):
    """A valid selection that only mentions the first treatment meter."""
    treatment = guard_env["treatment"]
    pool = guard_env["pool"]
    tids = treatment.ids
    weights = {tids[0]: {"pct_cluster_0": 1.0}}
    selection = _manual_clustering_selection(
        [tids[0]], pool.ids, guard_env["cluster_of"], weights, treatment.tz
    )
    tid = tids[1]

    return selection, treatment, pool, tid


def _guard_nan_weights(guard_env):
    treatment = guard_env["treatment"]
    pool = guard_env["pool"]
    tids = treatment.ids
    weights = {t: {"pct_cluster_0": 1.0} for t in tids}
    weights[tids[0]] = {"pct_cluster_0": np.nan}
    selection = _manual_clustering_selection(
        tids, pool.ids, guard_env["cluster_of"], weights, treatment.tz
    )
    tid = tids[0]

    return selection, treatment, pool, tid


def _guard_too_few_cg_meters(guard_env):
    """The meter draws only on a three-member cluster, below the kernel's floor
    of five comparison meters."""
    treatment = guard_env["treatment"]
    pool = guard_env["pool"]
    tids = treatment.ids
    pool_ids = pool.ids
    cluster_of = {p: 0 for p in pool_ids}
    for p in pool_ids[:3]:
        cluster_of[p] = 1
    weights = {
        tids[0]: {"pct_cluster_0": 1.0, "pct_cluster_1": 0.0},
        tids[1]: {"pct_cluster_0": 0.0, "pct_cluster_1": 1.0},
    }
    selection = _manual_clustering_selection(tids, pool_ids, cluster_of, weights, treatment.tz)
    tid = tids[1]

    return selection, treatment, pool, tid


def _guard_missing_reporting(guard_env):
    selection = guard_env["selection"]
    pool = guard_env["pool"]
    tids = guard_env["treatment"].ids
    tid = tids[0]
    meters = _population_meters(guard_env["comstock"], guard_env["treatment"])
    del meters[tid]["reporting_df"]

    treatment = TreatmentGroup.from_fit_models(meters)

    return selection, treatment, pool, tid


def _guard_prediction_failure(guard_env):
    selection = guard_env["selection"]
    pool = guard_env["pool"]
    tids = guard_env["treatment"].ids
    tid = tids[0]
    meters = _population_meters(guard_env["comstock"], guard_env["treatment"])
    meters[tid]["model"] = _disqualified_copy(meters[tid]["model"])

    with pytest.warns(UserWarning, match="disqualified"):
        treatment = TreatmentGroup.from_fit_models(meters)

    return selection, treatment, pool, tid


@pytest.mark.parametrize(
    "build_case, origin, reason",
    [
        (
            _guard_absent_from_population,
            "correction_guard",
            "treatment meter absent from the treatment population",
        ),
        (
            _guard_absent_from_selection,
            "correction_guard",
            "treatment meter absent from selection",
        ),
        (_guard_nan_weights, "correction_guard", "all-NaN cluster weights"),
        (
            _guard_too_few_cg_meters,
            "correction_guard",
            "fewer than 5 comparison-group meters available",
        ),
        (_guard_missing_reporting, "correction_guard", "missing reporting data"),
        (_guard_prediction_failure, "model", "model prediction failed"),
    ],
    ids=[
        "meter_absent_from_population",
        "meter_absent_from_selection",
        "all_nan_cluster_weights",
        "fewer_than_five_cg_meters",
        "missing_reporting_data",
        "prediction_failure",
    ],
)
def test_meter_guard_raises_with_its_ledger_row(guard_env, build_case, origin, reason):
    """Every meter-level guard is all-or-nothing: it raises
    ``MeterCorrectionError`` rather than returning a result missing the meter,
    and the exception carries the ledger row naming what broke."""
    selection, treatment, pool, tid = build_case(guard_env)

    with pytest.raises(exclusions.MeterCorrectionError) as excinfo:
        correct_reporting(selection, treatment, pool, tid)

    ledger = excinfo.value.exclusions
    assert list(ledger.columns) == _LEDGER_COLUMNS

    rows = ledger[ledger["id"] == tid]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["stage"] == "correction"
    assert row["origin"] == origin
    assert row["reason"] == reason
    assert reason in str(excinfo.value)


def test_guarded_meter_does_not_stop_its_neighbour(guard_env):
    """The guards are per meter: a treatment meter with all-NaN weights raises
    while the caller's next meter corrects normally off the same selection."""
    tids = guard_env["treatment"].ids
    selection, treatment, pool, victim = _guard_nan_weights(guard_env)
    survivor = tids[1]

    with pytest.raises(exclusions.MeterCorrectionError):
        correct_reporting(selection, treatment, pool, victim)

    result = correct_reporting(selection, treatment, pool, survivor)

    assert result.meter_id == survivor
    assert not result.corrected.empty


def test_billing_read_cadence_unrecoverable_raises(billing_dq_env):
    """Billing correction never runs the eemeter unique-values check (its
    cadence is read-period inference, not native rows): a non-zero identical
    reporting billing meter instead has no recoverable read cadence, so the
    correction raises with that reason and the other treatment meter is
    unaffected."""
    victim = billing_dq_env["treatment_ids"][1]
    survivor = billing_dq_env["treatment_ids"][0]
    flat = _reporting_override(
        billing_dq_env["df_r"], victim, lambda raw: raw.assign(observed=123.0)
    )
    treatment, pool = _build_billing_dq(billing_dq_env, treatment_overrides={victim: flat})

    with pytest.raises(exclusions.MeterCorrectionError) as excinfo:
        correct_reporting(billing_dq_env["selection"], treatment, pool, victim)

    row = excinfo.value.exclusions.set_index("id").loc[victim]
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_data"
    assert "no recoverable read" in row["reason"]
    # the flat meter's documented path is explicit ``read_boundaries``
    assert "read_boundaries" in row["reason"]

    result = correct_reporting(billing_dq_env["selection"], treatment, pool, survivor)
    assert result.correction_periods


def test_invalid_period_raises(guard_env):
    start = pd.Timestamp("2019-06-30", tz="America/Chicago")
    end = pd.Timestamp("2019-04-01", tz="America/Chicago")

    with pytest.raises(ValueError, match="invalid period"):
        correct_reporting(
            guard_env["selection"],
            guard_env["treatment"],
            guard_env["pool"],
            guard_env["treatment"].ids[0],
            period=(start, end),
        )


def test_fingerprint_mismatch_raises(guard_env):
    selection = copy.deepcopy(guard_env["selection"])
    label_col = selection.clusters.columns.get_loc("cluster")
    selection.clusters.iloc[0, label_col] = 99

    with pytest.raises(ValueError, match="fingerprint"):
        correct_reporting(
            selection, guard_env["treatment"], guard_env["pool"], guard_env["treatment"].ids[0]
        )


def test_timezone_mismatch_raises(guard_env):
    mismatched_pool = copy.copy(guard_env["pool"])
    mismatched_pool.tz = "America/New_York"

    with pytest.raises(ValueError, match="timezone"):
        correct_reporting(
            guard_env["selection"],
            guard_env["treatment"],
            mismatched_pool,
            guard_env["treatment"].ids[0],
        )


def test_coarser_pool_than_treatment_raises(daily_dq_env, comstock, model_bank):
    """The cross-population fineness rule at correction: a comparison pool
    coarser than the treatment (billing pool, daily treatment) is a validation
    error naming both granularities."""
    df_b, df_r = comstock.frames("billing")
    ids = sorted(df_b.index.get_level_values("id").unique())
    treatment, _ = _build_daily_dq(daily_dq_env)
    pool = ComparisonPool.from_fit_models(comstock.meters(model_bank, "billing", ids[:6]))

    with pytest.raises(ValueError, match="billing.*coarser.*daily"):
        correct_reporting(
            daily_dq_env["selection"], treatment, pool, daily_dq_env["treatment_ids"][0]
        )


# ── pool-side drops (fast) ───────────────────────────────────────────────────


def _ledger_row(result, mid):
    return result.exclusions.set_index("id").loc[mid]


@pytest.fixture(scope="module")
def pool_missing_reporting(guard_env):
    """One treatment meter corrected against a pool whose first two meters carry
    no reporting data at all."""
    meters = _population_meters(guard_env["comstock"], guard_env["pool"])
    missing_ids = list(meters)[:2]

    for mid in missing_ids:
        del meters[mid]["reporting_df"]

    pool = ComparisonPool.from_fit_models(meters)
    tid = guard_env["treatment"].ids[0]
    result = correct_reporting(guard_env["selection"], guard_env["treatment"], pool, tid)
    bundle = {"result": result, "missing_ids": missing_ids, "tid": tid}

    return bundle


def test_pool_meters_missing_reporting_are_excluded(pool_missing_reporting):
    """Pool meters without reporting data drop out of the matrices and are
    recorded on this meter's ledger; the six remaining comparison meters keep the
    correction whole."""
    result = pool_missing_reporting["result"]

    recorded = result.exclusions[result.exclusions["reason"] == "missing reporting data"]
    assert set(recorded["id"]) == set(pool_missing_reporting["missing_ids"])
    assert (recorded["stage"] == "correction").all()
    assert (recorded["origin"] == "reporting_data").all()
    assert not result.corrected.empty


def test_json_roundtrip_with_nan_uncertainty(pool_missing_reporting):
    """A NaN ``corrected_unc`` must survive the table roundtrip (NaN -> null ->
    NaN), alongside a non-empty ledger."""
    result = copy.deepcopy(pool_missing_reporting["result"])
    result.corrected.loc[0, "corrected_unc"] = np.nan

    rebuilt = CorrectionResult.from_json(result.to_json())

    assert np.isnan(rebuilt.corrected["corrected_unc"].iloc[0])
    pd.testing.assert_frame_equal(result.corrected, rebuilt.corrected)
    pd.testing.assert_frame_equal(result.exclusions, rebuilt.exclusions)
    assert not result.exclusions.empty


def test_failed_pool_meter_is_ledgered_in_every_call_that_selects_it(guard_env):
    """``_ensure_pred`` caches successes only, so a pool meter whose reporting
    prediction raises is re-attempted — and re-ledgered — in every per-meter call
    that selects it; the row is not deduped away."""
    victim = guard_env["pool"].ids[0]
    tids = guard_env["treatment"].ids
    meters = _population_meters(guard_env["comstock"], guard_env["pool"])
    meters[victim]["model"] = _disqualified_copy(meters[victim]["model"])

    with pytest.warns(UserWarning, match="disqualified"):
        pool = ComparisonPool.from_fit_models(meters)

    results = [
        correct_reporting(guard_env["selection"], guard_env["treatment"], pool, tids[0]),
        correct_reporting(guard_env["selection"], guard_env["treatment"], pool, tids[1]),
        correct_reporting(guard_env["selection"], guard_env["treatment"], pool, tids[0]),
    ]

    for result in results:
        row = _ledger_row(result, victim)
        assert row["stage"] == "correction"
        assert row["origin"] == "model"
        assert row["reason"] == "model prediction failed"
        # the remaining seven comparison meters keep the correction whole
        assert not result.corrected.empty

    assert victim not in pool._reporting_pred


# ── reporting coverage / observed-DQ at correction (fast, daily) ─────────────


def test_reporting_gross_hole_pool_meter_pruned_before_predict(daily_dq_env):
    """A pool meter whose reporting covers only part of the reporting group
    window (a gross hole) fails coverage at correction: it is ledgered with its
    coverage in ``detail`` and never predicted, while the remaining five
    comparison meters keep the treatment meter corrected."""
    victim = daily_dq_env["pool_ids"][0]
    df_r = daily_dq_env["df_r"]
    cutoff = df_r.xs(int(victim), level="id").index.min() + pd.Timedelta(days=180)
    truncated = _reporting_override(
        df_r, victim, lambda raw: raw[raw["datetime"] < cutoff]
    )
    treatment, pool = _build_daily_dq(daily_dq_env, pool_overrides={victim: truncated})

    result = correct_reporting(
        daily_dq_env["selection"], treatment, pool, daily_dq_env["treatment_ids"][0]
    )

    row = _ledger_row(result, victim)
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_coverage"
    assert row["reason"] == "reporting coverage below minimum over the group window"
    assert "reporting window coverage" in row["detail"]
    coverage = float(row["detail"].split()[3])
    assert coverage < 0.9
    # pruned before prediction: the failed meter never enters the reporting cache
    assert victim not in pool._reporting_pred
    assert not result.corrected.empty


def test_reporting_gross_hole_treatment_meter_raises_below_coverage(daily_dq_env):
    """The same coverage floor applied to the treatment meter is all-or-nothing:
    a meter covering only half the reporting group window raises rather than
    returning a half-corrected series, with its coverage in the ledger detail."""
    victim = daily_dq_env["treatment_ids"][1]
    survivor = daily_dq_env["treatment_ids"][0]
    df_r = daily_dq_env["df_r"]
    cutoff = df_r.xs(int(victim), level="id").index.min() + pd.Timedelta(days=180)
    truncated = _reporting_override(
        df_r, victim, lambda raw: raw[raw["datetime"] < cutoff]
    )
    treatment, pool = _build_daily_dq(daily_dq_env, treatment_overrides={victim: truncated})

    with pytest.raises(exclusions.MeterCorrectionError) as excinfo:
        correct_reporting(daily_dq_env["selection"], treatment, pool, victim)

    row = excinfo.value.exclusions.set_index("id").loc[victim]
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_coverage"
    assert row["reason"] == "reporting coverage below minimum over the group window"
    assert float(row["detail"].split()[3]) < 0.9
    assert victim not in treatment._reporting_pred

    result = correct_reporting(daily_dq_env["selection"], treatment, pool, survivor)
    assert not result.corrected.empty


def test_reporting_flat_treatment_meter_disqualified_via_unique_values(daily_dq_env):
    """A treatment meter with non-zero identical reporting reads has full
    coverage but fails the eemeter unique-values check at construction. The
    correction stage enforces that observed disqualification: the meter is
    ledgered verbatim, pruned before predicting, and the call raises."""
    victim = daily_dq_env["treatment_ids"][1]
    flat = _reporting_override(
        daily_dq_env["df_r"], victim, lambda raw: raw.assign(observed=123.0)
    )
    treatment, pool = _build_daily_dq(daily_dq_env, treatment_overrides={victim: flat})

    with pytest.raises(exclusions.MeterCorrectionError) as excinfo:
        correct_reporting(daily_dq_env["selection"], treatment, pool, victim)

    row = excinfo.value.exclusions.set_index("id").loc[victim]
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_data"
    assert row["reason"] == "reporting observed data disqualified"
    assert "insufficient_unique_observed_values" in row["detail"]
    assert victim not in treatment._reporting_pred


def test_reporting_joint_comissing_treatment_meter_disqualified(daily_dq_env):
    """A treatment meter whose observed and temperature go missing on DISJOINT
    day sets — neither column individually below the daily coverage floor, but
    their union (the joint observed-and-temperature valid days) below it — trips
    ONLY the eemeter joint valid-days check. The correction stage enforces that
    joint disqualification the same as an observed one."""
    victim = daily_dq_env["treatment_ids"][1]

    def _joint_comissing(raw):
        raw = raw.copy()
        # 25 observed-missing days and temperature-missing days on disjoint rows:
        # each column keeps > 90% daily coverage, but the joint coverage (both
        # present) falls below the 90% floor. Temperature-missing days are spread
        # ~15 apart so no single month's temperature coverage dips below 90%.
        obs_rows = raw.index[5:30]
        temp_rows = raw.index[40:365:15]
        raw.loc[obs_rows, "observed"] = np.nan
        raw.loc[temp_rows, "temperature"] = np.nan

        return raw

    holed = _reporting_override(daily_dq_env["df_r"], victim, _joint_comissing)
    treatment, pool = _build_daily_dq(daily_dq_env, treatment_overrides={victim: holed})

    with pytest.raises(exclusions.MeterCorrectionError) as excinfo:
        correct_reporting(daily_dq_env["selection"], treatment, pool, victim)

    row = excinfo.value.exclusions.set_index("id").loc[victim]
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_data"
    assert row["reason"] == "reporting observed data disqualified"
    assert "too_many_days_with_missing_joint_data" in row["detail"]
    # the joint failure alone triggered it: neither the observed-only nor the
    # temperature-only valid-days disqualification is present
    assert "too_many_days_with_missing_observed_data" not in row["detail"]
    assert "too_many_days_with_missing_temperature_data" not in row["detail"]
    assert victim not in treatment._reporting_pred


def test_reporting_large_savings_meter_is_not_disqualified(daily_dq_env):
    """A treatment meter whose reporting consumption dropped sharply (large
    savings) still varies and fully covers the window, so no observed
    disqualification or coverage failure fires — the correction stage runs no
    extreme-value screen on reporting observed, which would otherwise flag a
    meter precisely BECAUSE its savings are large."""
    saver = daily_dq_env["treatment_ids"][1]
    reduced = _reporting_override(
        daily_dq_env["df_r"], saver, lambda raw: raw.assign(observed=raw["observed"] * 0.3)
    )
    treatment, pool = _build_daily_dq(daily_dq_env, treatment_overrides={saver: reduced})

    result = correct_reporting(daily_dq_env["selection"], treatment, pool, saver)

    assert result.meter_id == saver
    assert saver not in set(result.exclusions["id"])


def test_reporting_flat_treatment_meter_disqualified_via_unique_values_hourly(hourly_dq_env):
    """The unique-values enforcement at correction is granularity-agnostic: an
    hourly treatment meter with non-zero identical reporting reads is ledgered
    and pruned before predicting, same as daily."""
    victim = hourly_dq_env["treatment_ids"][1]
    flat = _reporting_override(
        hourly_dq_env["df_r"], victim, lambda raw: raw.assign(observed=123.0)
    )
    treatment, pool = _build_hourly_dq(hourly_dq_env, treatment_overrides={victim: flat})

    with pytest.raises(exclusions.MeterCorrectionError) as excinfo:
        correct_reporting(hourly_dq_env["selection"], treatment, pool, victim)

    row = excinfo.value.exclusions.set_index("id").loc[victim]
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_data"
    assert row["reason"] == "reporting observed data disqualified"
    assert "insufficient_unique_observed_values" in row["detail"]
    assert victim not in treatment._reporting_pred


def test_interior_reporting_hole_below_coverage_passes_e_and_degrades_via_r4(daily_dq_env):
    """A pool meter's reporting carries a real 25-day interior NaN hole — not
    injected post-fit into the prediction cache. Individually it still covers
    340/365 = 0.9315 of the reporting group window, above the 0.9 default
    floor, so it clears the group-window coverage DQ (E) outright: no
    ``reporting_coverage`` ledger row. Its cluster (the minimum 3-meter size)
    then has only two finite meters during the hole, degrading that period
    under R4; the other 3-meter cluster keeps `corrected` finite there, and
    `clusters_used` falls from 2 to 1 exactly over the hole and nowhere else —
    the residual-gap visibility E's coverage gate alone cannot provide."""
    treatment_ids = daily_dq_env["treatment_ids"]
    pool_ids = daily_dq_env["pool_ids"]
    victim = pool_ids[0]
    df_r = daily_dq_env["df_r"]

    victim_index = df_r.xs(int(victim), level="id").index
    hole_start = victim_index.min() + pd.Timedelta(days=150)
    hole_end = hole_start + pd.Timedelta(days=25)
    hole_dates = victim_index[(victim_index >= hole_start) & (victim_index < hole_end)]
    holed = _reporting_override(
        df_r,
        victim,
        lambda raw: raw.assign(
            observed=np.where(
                (raw["datetime"] >= hole_start) & (raw["datetime"] < hole_end),
                np.nan,
                raw["observed"],
            )
        ),
    )
    treatment, pool = _build_daily_dq(daily_dq_env, pool_overrides={victim: holed})

    # two 3-member clusters: cluster 1 holds the victim (minimum size); cluster
    # 0 the other three, never dropping below three finite under this single
    # meter's hole
    cluster_of = {p: 0 for p in pool_ids}
    for p in pool_ids[:3]:
        cluster_of[p] = 1
    weights = {tid: {"pct_cluster_0": 0.5, "pct_cluster_1": 0.5} for tid in treatment_ids}
    selection = _manual_clustering_selection(
        treatment_ids, pool_ids, cluster_of, weights, "America/Chicago"
    )

    tid = treatment_ids[0]
    result = correct_reporting(selection, treatment, pool, tid)

    assert result.exclusions[result.exclusions["origin"] == "reporting_coverage"].empty

    rows = result.corrected.set_index("datetime")
    usage = result.cg_usage.set_index("datetime")["clusters_used"]
    other_dates = [d for d in rows.index if d not in hole_dates]

    assert (usage.loc[hole_dates] == 1).all()
    assert (usage.loc[other_dates] == 2).all()
    assert np.isfinite(rows.loc[hole_dates, "corrected"].to_numpy()).all()


# ── heterogeneous reporting spans (fast, daily) ──────────────────────────────


@pytest.fixture(scope="module")
def heterogeneous_span_results(daily_dq_env):
    """Two daily treatment meters over different reporting spans: the second
    carries only the first 300 days of its real ComStock reporting. The group
    window still spans the longer meter's full year, so the coverage floor is
    lowered for the truncated meter to be corrected at all."""
    long_id, short_id = daily_dq_env["treatment_ids"]
    truncated = _reporting_override(
        daily_dq_env["df_r"], short_id, lambda raw: raw.iloc[:300]
    )
    treatment, pool = _build_daily_dq(daily_dq_env, treatment_overrides={short_id: truncated})
    settings = CGCorrectionSettings(min_window_coverage=0.5)

    results = {
        tid: correct_reporting(daily_dq_env["selection"], treatment, pool, tid, settings=settings)
        for tid in (long_id, short_id)
    }
    bundle = {"long_id": long_id, "short_id": short_id, "results": results}

    return bundle


def test_heterogeneous_spans_emit_each_meters_own_index(heterogeneous_span_results):
    """Each meter's frame spans its own prediction index and nothing else: the
    truncated meter emits 300 rows, not the 365-row union padded with NaN."""
    long_result = heterogeneous_span_results["results"][heterogeneous_span_results["long_id"]]
    short_result = heterogeneous_span_results["results"][heterogeneous_span_results["short_id"]]

    assert len(long_result.corrected) == 365
    assert len(short_result.corrected) == 300
    assert np.isfinite(short_result.corrected["corrected"].to_numpy()).all()
    assert short_result.covered_window[1] < long_result.covered_window[1]


def test_heterogeneous_spans_keep_the_boundary_period_whole(heterogeneous_span_results):
    """Under own-index semantics the truncated meter's last (partial) month is a
    complete period over the rows it has: coverage stays 1 and its
    ``savings_unc`` is finite. Padding to the union index would put NaN rows in
    that month, dropping its coverage and poisoning the quadrature sum."""
    long_id = heterogeneous_span_results["long_id"]
    short_id = heterogeneous_span_results["short_id"]

    long_monthly = compute_savings(
        heterogeneous_span_results["results"][long_id], aggregation="monthly"
    ).savings
    short_monthly = compute_savings(
        heterogeneous_span_results["results"][short_id], aggregation="monthly"
    ).savings

    assert set(short_monthly["period"]) < set(long_monthly["period"])
    assert len(long_monthly) == 12
    assert (short_monthly["coverage"].to_numpy() == 1.0).all()

    boundary = short_monthly.iloc[-1]
    assert np.isfinite(boundary["savings_unc"])
    assert np.isfinite(boundary["savings"])


# ── individual meter matching (slow) ─────────────────────────────────────────


@pytest.mark.slow
def test_imm_correction_uses_the_shared_cluster_union_not_the_matched_rows(
    variant_env, comstock
):
    """IMM's correction deliberately draws on the shared cluster-0 union, with
    duplicate selection rows collapsed — NOT the treatment meter's own matched
    rows (its rows in the clusters table). NaN-ing a read period of a union
    member this meter did not match moves that period's ``cg_usage_fraction``;
    columns sourced from the matched rows alone would leave it untouched."""
    env = variant_env("billing_imm")
    selection = env["selection"]
    treatment = env["treatment"]
    tid = treatment.ids[0]

    members = cg_member_ids(selection, tid)
    clusters = selection.clusters
    matched = sorted(set(clusters.index[clusters["treatment"] == tid].astype(str)))
    union = sorted(set(selection.clusters.index.astype(str)))

    assert sorted(members) == union
    assert len(members) == len(set(members))
    assert set(matched) < set(members)

    victim = next(mid for mid in members if mid not in set(matched))
    _, df_r = comstock.frames("billing")
    injected_pool = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, env["pool"], df_r)
    )
    pred = injected_pool._ensure_pred("reporting", ids=[victim])
    substrate = pred[victim].index
    march = substrate[substrate.month == 3]
    pred[victim].loc[march, "observed"] = np.nan

    result = correct_reporting(selection, treatment, injected_pool, tid)

    usage = result.cg_usage.set_index("datetime")["cg_usage_fraction"]
    march_start = pd.Timestamp("2019-03-01", tz=treatment.tz)

    assert usage.loc[march_start] == pytest.approx((len(members) - 1) / len(members))
    other = [d for d in usage.index if d != march_start]
    assert np.allclose(usage.loc[other].to_numpy(), 1.0)


# ── prior semantics (fast, constructed) ──────────────────────────────────────


def _prior_from_corrected(meter_id, corrected, granularity, tz="America/Chicago"):
    """Wrap a constructed ``[id, datetime, corrected]`` frame as a prior
    ``CorrectionResult`` for direct ``_check_prior`` exercise."""
    result = CorrectionResult(
        meter_id=meter_id,
        corrected=corrected,
        cg_usage=pd.DataFrame(columns=["id", "datetime", "cg_usage_fraction"]),
        exclusions=exclusions.empty_ledger(),
        settings={},
        covered_window=None,
        tz=tz,
        granularity=granularity,
        fingerprint="test-fingerprint",
    )

    return result


def _corrected_frame(datetimes, values):
    frame = pd.DataFrame(
        {
            "id": ["m1"] * len(datetimes),
            "datetime": datetimes,
            "corrected": np.asarray(values, dtype=np.float64),
        }
    )

    return frame


def test_check_prior_billing_excludes_trailing_month_mismatch():
    """At billing granularity the trailing (possibly partial) read period is
    re-drawn on extension, so a mismatch there is excluded from the
    slice-invariance check and does not raise."""
    months = pd.date_range("2019-01-01", periods=3, freq="MS", tz="America/Chicago")
    prior = _prior_from_corrected("m1", _corrected_frame(months, [10.0, 20.0, 30.0]), "billing")
    new_corrected = _corrected_frame(months, [10.0, 20.0, 999.0])

    _check_prior(new_corrected, prior, "billing")


def test_check_prior_billing_earlier_month_mismatch_raises():
    """A mismatch in a non-trailing month is a genuine slice-invariance
    violation and raises even at billing granularity."""
    months = pd.date_range("2019-01-01", periods=3, freq="MS", tz="America/Chicago")
    prior = _prior_from_corrected("m1", _corrected_frame(months, [10.0, 20.0, 30.0]), "billing")
    new_corrected = _corrected_frame(months, [10.0, 999.0, 30.0])

    with pytest.raises(ValueError, match="prior correction mismatch"):
        _check_prior(new_corrected, prior, "billing")


def test_check_prior_billing_trailing_period_comes_from_the_read_calendar():
    """A prior reported under a period filter ends its frame before its last
    read; the trailing period is the last read in ``correction_periods``, so the
    frame's final row is still confirmed and a mismatch there raises."""
    tz = "America/Chicago"
    datetimes = pd.DatetimeIndex([pd.Timestamp(f"2019-0{m}-01", tz=tz) for m in (1, 2, 3)])
    prior = _prior_from_corrected("m1", _corrected_frame(datetimes, [1.0, 2.0, 3.0]), "billing")
    april = pd.Timestamp("2019-04-01", tz=tz)
    reads = list(datetimes) + [april]
    prior.correction_periods = [(start, start + pd.Timedelta(days=27)) for start in reads]
    new = _corrected_frame(datetimes, [1.0, 2.0, 30.0])

    with pytest.raises(ValueError, match="prior correction mismatch"):
        _check_prior(new, prior, "billing")


def test_check_prior_daily_compares_full_overlap_including_last():
    """Daily granularity has no trailing-period carve-out, so a mismatch on the
    last overlapping timestep raises."""
    days = pd.date_range("2019-01-01", periods=3, freq="D", tz="America/Chicago")
    prior = _prior_from_corrected("m1", _corrected_frame(days, [10.0, 20.0, 30.0]), "daily")
    new_corrected = _corrected_frame(days, [10.0, 20.0, 999.0])

    with pytest.raises(ValueError, match="prior correction mismatch"):
        _check_prior(new_corrected, prior, "daily")


class _UnreadablePeriods(list):
    """Stands in for ``prior.correction_periods``: reading it at all is a test
    failure."""

    def __iter__(self):
        raise AssertionError("prior.correction_periods was read for a mismatched meter")

    def __len__(self):
        raise AssertionError("prior.correction_periods was read for a mismatched meter")


def test_prior_belonging_to_another_meter_raises_before_its_periods_are_read(guard_env):
    """A prior carrying a different ``meter_id`` is an analysis-level error
    (``ValueError``, not ``MeterCorrectionError``), and the identity guard fires
    before anything reads the prior's read calendar — freezing the wrong meter's
    read periods would otherwise pass silently."""
    tids = guard_env["treatment"].ids
    prior = _prior_from_corrected(
        tids[1], _corrected_frame(pd.DatetimeIndex([]), []), "billing"
    )
    prior.correction_periods = _UnreadablePeriods()

    with pytest.raises(ValueError, match="prior correction belongs to meter"):
        correct_reporting(
            guard_env["selection"],
            guard_env["treatment"],
            guard_env["pool"],
            tids[0],
            prior=prior,
        )


# ── incremental hybrid semantics (slow) ──────────────────────────────────────


def test_incremental_treatment_overlap_is_slice_invariant(daily_env, daily_data, comstock):
    df_r = daily_data["df_r"]
    ids = daily_data["treatment_ids"]
    selection = daily_env["selection"]
    pool = daily_env["pool"]
    tid = str(ids[0])

    half = TreatmentGroup.from_fit_models(
        _rewrap_population(comstock, daily_env["treatment"], df_r, cutoff=_MIDYEAR)
    )
    result_half = correct_reporting(selection, half, pool, tid)

    half.add_reporting_data(_reporting_map(df_r, ids))
    result_full = correct_reporting(selection, half, pool, tid, prior=result_half)

    assert result_half.covered_window[1] < _MIDYEAR
    assert result_full.covered_window[1] == pd.Timestamp("2019-12-31", tz="America/Chicago")

    overlap = result_full.corrected.merge(
        result_half.corrected[["id", "datetime", "corrected"]],
        on=["id", "datetime"],
        suffixes=("", "_prior"),
    )
    assert len(overlap) == len(result_half.corrected)
    assert np.array_equal(
        overlap["corrected"].to_numpy(), overlap["corrected_prior"].to_numpy(), equal_nan=True
    )


def test_incremental_prior_mismatch_raises(daily_env, daily_data, comstock):
    df_r = daily_data["df_r"]
    ids = daily_data["treatment_ids"]
    selection = daily_env["selection"]
    pool = daily_env["pool"]
    tid = str(ids[0])

    half = TreatmentGroup.from_fit_models(
        _rewrap_population(comstock, daily_env["treatment"], df_r, cutoff=_MIDYEAR)
    )
    result_half = correct_reporting(selection, half, pool, tid)

    tampered = copy.deepcopy(result_half)
    tampered.corrected["corrected"] = tampered.corrected["corrected"] + 1.0

    half.add_reporting_data(_reporting_map(df_r, ids))

    with pytest.raises(ValueError, match="prior correction mismatch"):
        correct_reporting(selection, half, pool, tid, prior=tampered)


@pytest.mark.slow
def test_growing_pool_reporting_keeps_prior_points(daily_env, daily_data, comstock):
    df_r = daily_data["df_r"]
    pool_ids = daily_data["pool_ids"]
    treatment = daily_env["treatment"]
    selection = daily_env["selection"]
    tid = treatment.ids[0]

    # the half-year pool covers only half the full-year treatment window; lower
    # the reporting-coverage floor so it is not pruned, exercising the growing
    # pool rather than the coverage DQ
    settings = CGCorrectionSettings(min_window_coverage=0.4)

    pool_half = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, daily_env["pool"], df_r, cutoff=_MIDYEAR)
    )
    result_half = correct_reporting(selection, treatment, pool_half, tid, settings=settings)

    late = result_half.corrected[result_half.corrected["datetime"] >= _MIDYEAR]
    assert not np.isfinite(late["corrected"].to_numpy()).any()

    pool_half.add_reporting_data(_reporting_map(df_r, pool_ids))
    result_full = correct_reporting(
        selection, treatment, pool_half, tid, settings=settings, prior=result_half
    )

    assert np.isfinite(result_full.corrected["corrected"].to_numpy()).all()

    early = result_full.corrected[result_full.corrected["datetime"] < _MIDYEAR]
    early_prior = result_half.corrected[result_half.corrected["datetime"] < _MIDYEAR]
    merged = early.merge(
        early_prior[["id", "datetime", "corrected"]],
        on=["id", "datetime"],
        suffixes=("", "_prior"),
    )
    assert np.array_equal(merged["corrected"].to_numpy(), merged["corrected_prior"].to_numpy())


# ── frozen comparison group under a prior ────────────────────────────────────


def test_prior_from_another_selection_raises(guard_env):
    """A prior froze a comparison group chosen by one selection; reusing it
    under a selection with a different fingerprint is an analysis-level error."""
    tid = guard_env["treatment"].ids[0]
    prior = correct_reporting(guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid)
    other_selection, treatment, pool, _ = _guard_too_few_cg_meters(guard_env)

    with pytest.raises(ValueError, match="different selection"):
        correct_reporting(other_selection, treatment, pool, tid, prior=prior)


def test_prior_freezes_the_comparison_group_against_new_entrants(guard_env):
    """Pool meters that lacked reporting data when the prior was made do not
    join the group later: the prior's members are reused as they are, so the
    overlapping points reproduce exactly."""
    tid = guard_env["treatment"].ids[0]
    missing = guard_env["pool"].ids[:2]
    meters = _population_meters(guard_env["comstock"], guard_env["pool"])

    for mid in missing:
        del meters[mid]["reporting_df"]

    reduced_pool = ComparisonPool.from_fit_models(meters)
    prior = correct_reporting(guard_env["selection"], guard_env["treatment"], reduced_pool, tid)
    assert set(prior.cg_ids) == set(guard_env["pool"].ids) - set(missing)

    result = correct_reporting(
        guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid, prior=prior
    )

    assert result.cg_ids == prior.cg_ids
    assert not result.exclusions["id"].isin(missing).any()
    pd.testing.assert_frame_equal(result.corrected, prior.corrected)


def test_prior_member_that_no_longer_predicts_raises_with_ledger_rows(guard_env):
    """A frozen group is all-or-nothing: when one member can no longer predict
    the group cannot be reproduced, so the meter raises, carrying the member's
    own row and a guard row naming it."""
    tid = guard_env["treatment"].ids[0]
    victim = guard_env["pool"].ids[0]
    prior = correct_reporting(guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid)
    meters = _population_meters(guard_env["comstock"], guard_env["pool"])
    meters[victim]["model"] = _disqualified_copy(meters[victim]["model"])

    with pytest.warns(UserWarning, match="disqualified"):
        pool = ComparisonPool.from_fit_models(meters)

    with pytest.raises(exclusions.MeterCorrectionError, match="cannot be reproduced") as excinfo:
        correct_reporting(guard_env["selection"], guard_env["treatment"], pool, tid, prior=prior)

    ledger = excinfo.value.exclusions
    victim_row = ledger[ledger["id"] == victim].iloc[0]
    assert victim_row["reason"] == "model prediction failed"
    guard_row = ledger[ledger["id"] == tid].iloc[0]
    assert guard_row["reason"] == "frozen comparison group cannot be reproduced"
    assert victim in guard_row["detail"]


def test_prior_member_with_disqualified_reporting_raises_with_ledger_rows(daily_dq_env):
    """Freezing the group keeps its members regardless of coverage, but not
    regardless of data quality: a frozen member whose new reporting data fails
    the observed checks is ledgered verbatim, never predicted, and the group
    cannot be reproduced."""
    tid = daily_dq_env["treatment_ids"][0]
    victim = daily_dq_env["pool_ids"][0]
    treatment, pool = _build_daily_dq(daily_dq_env)
    prior = correct_reporting(daily_dq_env["selection"], treatment, pool, tid)
    assert victim in prior.cg_ids

    flat = _reporting_override(
        daily_dq_env["df_r"], victim, lambda raw: raw.assign(observed=123.0)
    )
    _, flat_pool = _build_daily_dq(daily_dq_env, pool_overrides={victim: flat})

    with pytest.raises(exclusions.MeterCorrectionError, match="cannot be reproduced") as excinfo:
        correct_reporting(daily_dq_env["selection"], treatment, flat_pool, tid, prior=prior)

    ledger = excinfo.value.exclusions
    victim_row = ledger[ledger["id"] == victim].iloc[0]
    assert victim_row["origin"] == "reporting_data"
    assert victim_row["reason"] == "reporting observed data disqualified"
    assert "insufficient_unique_observed_values" in victim_row["detail"]
    assert victim not in flat_pool._reporting_pred


def test_prior_without_a_comparison_group_raises(guard_env):
    """A prior that carries no comparison group cannot freeze one; it is an
    analysis-level error rather than a short-group guard."""
    tid = guard_env["treatment"].ids[0]
    prior = correct_reporting(guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid)
    hollow = copy.deepcopy(prior)
    hollow.cg_ids = []

    with pytest.raises(ValueError, match="no comparison group"):
        correct_reporting(
            guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid, prior=hollow
        )


@pytest.mark.slow
def test_lagging_pool_reporting_keeps_the_frozen_group_under_a_prior(
    daily_env, daily_data, comstock
):
    """At the default coverage floor a pool meter whose reporting lags the
    treatment's growing window would be pruned, changing the group under every
    past point. A prior freezes the group instead: the lagging members stay, the
    late half comes back NaN until their data arrives, and the early points are
    unchanged."""
    df_r = daily_data["df_r"]
    selection = daily_env["selection"]
    tid = daily_env["treatment"].ids[0]

    treatment_half = TreatmentGroup.from_fit_models(
        _rewrap_population(comstock, daily_env["treatment"], df_r, cutoff=_MIDYEAR)
    )
    pool_half = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, daily_env["pool"], df_r, cutoff=_MIDYEAR)
    )
    prior = correct_reporting(selection, treatment_half, pool_half, tid)
    assert np.isfinite(prior.corrected["corrected"].to_numpy()).all()

    treatment_half.add_reporting_data(
        _reporting_map(df_r, daily_data["treatment_ids"])
    )

    # without the prior, the half-year pool fails coverage against the full-year window
    with pytest.raises(exclusions.MeterCorrectionError):
        correct_reporting(selection, treatment_half, pool_half, tid)

    lagging = correct_reporting(selection, treatment_half, pool_half, tid, prior=prior)
    assert lagging.cg_ids == prior.cg_ids
    late = lagging.corrected[lagging.corrected["datetime"] >= _MIDYEAR]
    assert not np.isfinite(late["corrected"].to_numpy()).any()

    pool_half.add_reporting_data(_reporting_map(df_r, daily_data["pool_ids"]))
    full = correct_reporting(selection, treatment_half, pool_half, tid, prior=prior)
    assert np.isfinite(full.corrected["corrected"].to_numpy()).all()

    early = full.corrected[full.corrected["datetime"] < _MIDYEAR]
    merged = early.merge(
        prior.corrected[["id", "datetime", "corrected"]],
        on=["id", "datetime"],
        suffixes=("", "_prior"),
    )
    assert len(merged) == len(prior.corrected)
    assert np.array_equal(merged["corrected"].to_numpy(), merged["corrected_prior"].to_numpy())


# ── DST fall-back (slow, hourly) ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def hourly_dst_result(comstock, model_bank):
    """Hourly correction across the America/Chicago DST fall-back day. Random
    sampling avoids the clustering minimum-size floor so a tiny (1 treatment,
    6 pool) population suffices."""
    df_b, df_r = comstock.frames("hourly")
    ids = sorted(df_b.index.get_level_values("id").unique())

    treatment = TreatmentGroup.from_fit_models(comstock.meters(model_bank, "hourly", ids[:1]))
    pool = ComparisonPool.from_fit_models(comstock.meters(model_bank, "hourly", ids[1:7]))
    selection = select_comparison_group(
        treatment,
        pool,
        method="random_sampling",
        method_settings=RandomSamplingSettings(
            n_meters_total=6, n_meters_per_treatment=None, seed=0
        ),
    )
    result = correct_reporting(selection, treatment, pool, treatment.ids[0])

    return result


def test_dst_fallback_day_has_no_duplicate_or_missing_hours(hourly_dst_result):
    corrected = hourly_dst_result.corrected

    assert not corrected.duplicated(["id", "datetime"]).any()

    day = corrected[corrected["datetime"].dt.date == _DST_FALLBACK_DATE]
    # the fall-back day has 25 local hours (the 01:00 hour occurs twice)
    assert len(day) == 25
    assert np.isfinite(day["corrected"].to_numpy()).all()


# ── equivalence snapshot artifact (fast) ─────────────────────────────────────


@pytest.fixture(scope="module")
def equivalence_snapshot():
    """The pinned per-meter correction and savings output the reshaped
    per-meter API is compared against."""
    return load_equivalence_snapshot()


def test_equivalence_snapshot_covers_every_variant(equivalence_snapshot):
    """Every variant is present with its populations and selection method
    recorded, and every pinned meter belongs to that variant's treatment."""
    variants = equivalence_snapshot["variants"]

    assert set(variants) == set(VARIANT_NAMES)
    assert variants["billing_imm"]["method"] == "individual_meter_matching"
    assert variants["hourly_pool_daily_treatment"]["granularity"] == "daily"

    for name, variant in variants.items():
        assert variant["treatment_ids"], f"{name}: no treatment ids recorded"
        assert variant["pool_ids"], f"{name}: no pool ids recorded"
        assert variant["meters"], f"{name}: no per-meter payloads recorded"
        assert set(variant["meters"]) <= set(variant["treatment_ids"]), f"{name}: stray meter id"


def test_equivalence_snapshot_tables_carry_the_correction_schema(equivalence_snapshot):
    """Each pinned meter's frames round-trip with the correction and savings
    columns, one native savings row per corrected row, and — for a billing
    treatment — one corrected row per recorded read period."""
    for name, variant in equivalence_snapshot["variants"].items():
        for mid, meter in variant["meters"].items():
            label = f"{name}/{mid}"
            corrected = meter["corrected"]

            assert list(corrected.columns) == _CORRECTED_COLUMNS, label
            assert list(meter["cg_usage"].columns) == [
                "id",
                "datetime",
                "cg_usage_fraction",
                "clusters_used",
            ], label
            assert list(meter["exclusions"].columns) == _LEDGER_COLUMNS, label
            assert set(meter["savings"]) == {"native", "monthly"}, label

            assert len(corrected) > 0, label
            assert set(corrected["id"].unique()) == {mid}, label
            assert len(meter["cg_usage"]) == len(corrected), label
            assert len(meter["savings"]["native"]) == len(corrected), label

            if variant["granularity"] == "billing":
                assert len(meter["correction_periods"]) == len(corrected), label
            else:
                assert meter["correction_periods"] == [], label


def test_equivalence_snapshot_comparison_groups_are_live(equivalence_snapshot):
    """Every pinned meter cleared the five-comparison-meter guard: it carries no
    exclusion row, and at every timestep at least one cluster survived with a
    non-empty used fraction."""
    for name, variant in equivalence_snapshot["variants"].items():
        for mid, meter in variant["meters"].items():
            label = f"{name}/{mid}"
            usage = meter["cg_usage"]
            fraction = usage["cg_usage_fraction"].to_numpy(dtype=float)

            assert meter["exclusions"].empty, label
            assert (fraction > 0.0).all(), label
            assert (fraction <= 1.0).all(), label
            assert (usage["clusters_used"].to_numpy(dtype=int) >= 1).all(), label


def test_equivalence_snapshot_coverage_matches_corrected_nan_positions(equivalence_snapshot):
    """NaN positions survive the JSON artifact: at native aggregation a period's
    ``coverage`` is 1 exactly where that meter's ``corrected`` value is finite
    and 0 where it is NaN."""
    for name, variant in equivalence_snapshot["variants"].items():
        for mid, meter in variant["meters"].items():
            corrected = meter["corrected"].set_index("datetime")["corrected"]
            coverage = meter["savings"]["native"].set_index("period")["coverage"]
            finite = np.isfinite(corrected.to_numpy(dtype=float)).astype(float)

            np.testing.assert_array_equal(
                coverage.reindex(corrected.index).to_numpy(dtype=float),
                finite,
                err_msg=f"{name}/{mid}: coverage disagrees with corrected NaN positions",
            )


@pytest.mark.regression
def test_equivalence_snapshot_reproduces_pinned_correction_totals(equivalence_snapshot):
    """The artifact is real pipeline output rather than hand-authored values:
    its daily and billing clustering variants sum to the correction totals the
    end-to-end tests already pin over the same populations."""
    totals = {}

    for name in ("daily_cg_clustering", "billing_cg_clustering"):
        meters = equivalence_snapshot["variants"][name]["meters"].values()
        frames = [meter["corrected"] for meter in meters]
        values = pd.concat(frames, ignore_index=True)["corrected"].to_numpy(dtype=float)
        totals[name] = float(np.nansum(values))

    assert totals["daily_cg_clustering"] == pytest.approx(3386073.02, rel=1e-3)
    assert totals["billing_cg_clustering"] == pytest.approx(2263284.60, rel=1e-3)


# ── correction equivalence with the pinned batch output (slow) ───────────────


def _assert_values_match(actual, expected, exact, label):
    """Every numeric column, compared over the meter's own span with NaN
    positions required to line up."""
    for column in actual.columns:
        if column in ("id", "datetime"):
            continue

        left = actual[column].to_numpy(dtype=np.float64)
        right = expected[column].to_numpy(dtype=np.float64)
        message = f"{label}: {column}"

        if exact:
            np.testing.assert_allclose(
                left, right, rtol=_EXACT_RTOL, atol=0.0, equal_nan=True, err_msg=message
            )
        else:
            np.testing.assert_allclose(
                left,
                right,
                rtol=1e-6,
                atol=_EQUIVALENCE_ATOL,
                equal_nan=True,
                err_msg=message,
            )


@pytest.mark.regression
@pytest.mark.parametrize("variant_name", VARIANT_NAMES)
def test_per_meter_correction_reproduces_the_batch_snapshot(
    variant_name, variant_env, equivalence_snapshot
):
    """The per-meter correction reproduces the pinned output meter by meter:
    to 1e-9 relative for the billing variants, and at float32 precision for
    the hourly/daily ones, where the batch-pinned treatment values carry a float32
    rounding the per-meter path's float64 prediction frames do not. The
    hourly-pool variant is pinned from the per-meter path itself and shares
    that tolerance, which leaves room for platform-level differences in the
    hourly model fits."""
    env = variant_env(variant_name)
    pinned = equivalence_snapshot["variants"][variant_name]
    exact = variant_name in _EXACT_EQUIVALENCE_VARIANTS

    for mid, expected in pinned["meters"].items():
        result = correct_reporting(env["selection"], env["treatment"], env["pool"], mid)
        label = f"{variant_name}/{mid}"

        assert result.meter_id == mid, label
        assert list(result.corrected["datetime"]) == list(expected["corrected"]["datetime"]), label
        _assert_values_match(result.corrected, expected["corrected"], exact, label)

        # the usage mask is finiteness-driven, so the float32 round-trip cannot
        # move it: cg_usage matches exactly for every variant
        _assert_values_match(result.cg_usage, expected["cg_usage"], True, label)

        assert result.correction_periods == expected["correction_periods"], label
        pd.testing.assert_frame_equal(
            result.exclusions.reset_index(drop=True),
            expected["exclusions"].reset_index(drop=True),
            check_dtype=False,
            obj=label,
        )
