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
import re

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
    _quadrature_to_periods,
    _restrict_to_days,
    _sum_to_periods,
    correct_reporting,
)
from opendsm.comparison_groups.savings.settings import (
    CGCorrectionSettings,
    WeightClusterAggChoice,
)
from opendsm.eemeter.common.compliance import Compliance
from opendsm.eemeter.common.warnings import EEMeterWarning
from .equivalence_env import (
    POOL_OBSERVED_UNC,
    VARIANT_NAMES,
    _manual_clustering_selection,
    _mixed_selection,
    load_equivalence_snapshot,
    shorten_baselines,
)



# the pinned real-data outputs were produced under model-magnitude weighting
_PINNED_SETTINGS = CGCorrectionSettings(weight_cluster_aggregation=WeightClusterAggChoice.MODEL)

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

# Variants whose treatment prediction frames are float64 in both the snapshot and the
# per-meter path, so equivalence holds to 1e-9 relative: the pinned models pass through a
# JSON round trip and library summation and t-quantile kernels differ in the last digits.
_EXACT_RTOL = 1e-9

_EXACT_EQUIVALENCE_VARIANTS = (
    "billing_cg_clustering",
    "billing_pool_observed_unc",
    "billing_imm",
)

# Tolerances for the hourly/daily equivalence comparison: their snapshot treatment values
# carry float32 rounding, and the absolute floor (kWh) covers near-zero rows, where a
# relative tolerance is meaningless.
_FLOAT32_RTOL = 1e-6
_EQUIVALENCE_ATOL = 1e-6

_COVERAGE_DETAIL = re.compile(r"reporting span coverage (\d+\.\d+) <")


# ── builders ─────────────────────────────────────────────────────────────────


def _reporting_map(df_r, ids, cutoff=None):
    reporting = {}

    for mid in ids:
        raw = df_r.xs(mid, level="id")
        if cutoff is not None:
            raw = raw[raw.index < cutoff]
        reporting[str(mid)] = raw.reset_index()

    return reporting


def _disqualified(model):
    """A copy of ``model`` carrying a disqualification, so its prediction fails. Uses
    the ``settings`` member rather than ``model``: a population's own predictions
    never enforce ``model`` (a fit-disqualified meter stays admitted and
    predictable), so only a still-enforced member reproduces a genuine prediction
    failure here."""
    model = copy.deepcopy(model)
    model.compliance = Compliance(
        all=[
            EEMeterWarning(
                qualified_name="eemeter.settings.nonstandard",
                description="constructed disqualification",
                severity="disqualified",
                member="settings",
                data=None,
            )
        ]
    )

    return model


def _rewrap_population(comstock, population, df_r, cutoff=None):
    """``population``'s meters mapping with its fitted models and the reporting
    frame taken from ``df_r``, truncated before ``cutoff`` when given."""
    meters = comstock.population_meters(population)

    for mid, entry in meters.items():
        raw = df_r.xs(int(mid), level="id")
        if cutoff is not None:
            raw = raw[raw.index < cutoff]
        entry["reporting_df"] = raw.reset_index()

    return meters


def _reload_with_reporting(population, reporting, baseline=None):
    """Rebuild ``population`` from its JSON with ``reporting`` (and ``baseline``,
    when given) attached, as an incremental run reloads its saved population
    with cumulative reporting."""
    reloaded = type(population).from_json(
        population.to_json(), baseline=baseline, reporting=reporting
    )

    return reloaded


def _largest_cluster_ids(membership):
    """The pool ids of the largest cluster in a comparison-group membership
    walk."""
    labels = np.array([label for _, label in membership])
    unique, counts = np.unique(labels, return_counts=True)
    big_label = unique[np.argmax(counts)]
    ids = [mid for mid, label in membership if label == big_label]

    return ids


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def daily_data(comstock, model_bank):
    _, df_r = comstock.frames("daily")
    ids = comstock.ids("daily")
    bundle = {
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
    """One ``CorrectionResult`` per treatment meter: the caller-owned loop the
    per-meter API expects."""
    results = {
        tid: correct_reporting(
            daily_env["selection"],
            daily_env["treatment"],
            daily_env["pool"],
            tid,
            settings=_PINNED_SETTINGS,
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
def billing_results(billing_env):
    results = {
        tid: correct_reporting(
            billing_env["selection"],
            billing_env["treatment"],
            billing_env["pool"],
            tid,
            settings=_PINNED_SETTINGS,
        )
        for tid in billing_env["treatment"].ids
    }

    return results


@pytest.fixture(scope="module")
def mixed_hourly_daily_env(variant_env):
    """A finer (hourly) comparison pool against a coarser (daily) treatment:
    the same underlying ComStock meters, disjoint ids, each fit at its own
    native model frequency."""
    env = variant_env("hourly_pool_daily_treatment")
    results = {
        tid: correct_reporting(env["selection"], env["treatment"], env["pool"], tid)
        for tid in env["treatment"].ids
    }
    bundle = dict(env, results=results)

    return bundle


@pytest.fixture(scope="module", params=[pytest.param("hourly", marks=pytest.mark.slow), "daily"])
def mixed_billing_results(request, comstock, model_bank):
    """A finer (hourly or daily) comparison pool against a coarser (billing)
    treatment, corrected at the treatment's inferred read-period cadence: one
    result per treatment meter."""
    ids = comstock.ids("billing")

    treatment = TreatmentGroup.from_fit_models(comstock.meters(model_bank, "billing", ids[:2]))
    pool = ComparisonPool.from_fit_models(comstock.meters(model_bank, request.param, ids[2:8]))
    selection = _mixed_selection(treatment, pool)
    results = {
        tid: correct_reporting(selection, treatment, pool, tid) for tid in treatment.ids
    }

    return results


@pytest.fixture(scope="module")
def guard_env(comstock, model_bank):
    """A small, cheap billing population pair plus a hand-built valid selection
    (all pool meters in one cluster) for guard-path tests."""
    _, df_r = comstock.frames("billing")
    ids = comstock.ids("billing")
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
        "df_r": df_r,
        "ids": ids,
    }

    return bundle


def _dq_env(comstock, model_bank, model_frequency):
    """A small real ComStock population pair (2 treatment, 6 pool) at
    ``model_frequency`` with a hand-built one-cluster selection, for
    correction-stage reporting DQ tests. The pinned-model records and the raw
    reporting frame are exposed so variant reporting (truncated, flat, reduced)
    can be swapped in without refitting."""
    _, df_r = comstock.frames(model_frequency)
    ids = comstock.ids(model_frequency)[:8]
    treatment_ids = [str(i) for i in ids[:2]]
    pool_ids = [str(i) for i in ids[2:8]]
    records = comstock.meters(model_bank, model_frequency, ids)

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


@pytest.fixture(scope="module")
def daily_dq_env(comstock, model_bank):
    return _dq_env(comstock, model_bank, "daily")


@pytest.fixture(scope="module")
def billing_dq_env(comstock, model_bank):
    return _dq_env(comstock, model_bank, "billing")


@pytest.fixture(scope="module")
def hourly_dq_env(comstock, model_bank):
    return _dq_env(comstock, model_bank, "hourly")


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


def _build_dq(dq_env, treatment_overrides=None, pool_overrides=None):
    records = dq_env["records"]
    treatment = TreatmentGroup.from_fit_models(
        _meters_with_overrides(records, dq_env["treatment_ids"], treatment_overrides or {})
    )
    pool = ComparisonPool.from_fit_models(
        _meters_with_overrides(records, dq_env["pool_ids"], pool_overrides or {})
    )

    return treatment, pool


def _reporting_override(dq_env, mid, transform):
    """One meter's reporting frame, put through ``transform``."""
    raw = dq_env["df_r"].xs(int(mid), level="id").reset_index()
    reporting = transform(raw)

    return reporting


# ── daily end-to-end ─────────────────────────────────────────────────────────


@pytest.mark.slow
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
        assert result.model_frequency == "daily"
        assert len(corrected) == 365


@pytest.mark.slow
def test_daily_corrected_is_finite_and_covers_full_year(daily_results):
    for result in daily_results.values():
        corrected = result.corrected

        assert np.isfinite(corrected["corrected"].to_numpy()).all()
        start, end = result.covered_window
        assert start == pd.Timestamp("2019-01-01", tz="America/Chicago")
        assert end == pd.Timestamp("2019-12-31", tz="America/Chicago")


@pytest.mark.slow
def test_daily_cg_usage_fraction_is_one_over_finite_comparison_data(daily_results):
    """``cg_usage_fraction`` is the fraction of a treatment meter's comparison
    meters retained (finite, no outlier rejection); over fully finite comparison
    data every meter is used at every timestep."""
    for result in daily_results.values():
        fractions = result.cg_usage["cg_usage_fraction"].to_numpy()

        assert set(result.cg_usage["id"].unique()) == {result.meter_id}
        assert np.allclose(fractions, 1.0)


def test_daily_cg_usage_fraction_drops_where_comparison_meters_are_nan(
    daily_env, daily_data, comstock
):
    """NaN-ing two of a treatment meter's comparison meters at three timesteps
    drops those timesteps' ``cg_usage_fraction`` to ``(n_cg - 2) / n_cg`` while
    the rest stay at 1.0."""
    treatment = daily_env["treatment"]
    pool = daily_env["pool"]
    selection = daily_env["selection"]
    tid = treatment.ids[0]

    # pick two meters from the meter's largest contributing cluster so the
    # cluster keeps at least three finite meters (no degradation) after corruption
    membership, _ = _cg_membership(selection, tid)
    n_cg = len(membership)
    big_ids = _largest_cluster_ids(membership)
    assert len(big_ids) >= 5
    corrupt_ids = big_ids[:2]

    injected_pool = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, pool, daily_data["df_r"])
    )
    pred = injected_pool.prediction_frames("reporting")
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

    membership, _ = _cg_membership(selection, tid)
    big_ids = _largest_cluster_ids(membership)
    assert len(big_ids) >= 5
    n_corrupt = len(big_ids) - 2
    corrupt_ids = big_ids[:n_corrupt]

    injected_pool = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, daily_env["pool"], daily_data["df_r"])
    )
    pred = injected_pool.prediction_frames("reporting")
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


def test_daily_degraded_timestep_is_returned_not_dropped(degraded_daily):
    """A per-timestep failure never raises: the NaN rows are returned with the
    rest of the meter's reporting year."""
    result = degraded_daily["result"]

    assert len(result.corrected) == 365


@pytest.mark.slow
def test_daily_json_roundtrip_preserves_frames_and_tz(daily_results):
    result = daily_results[next(iter(daily_results))]

    rebuilt = CorrectionResult.from_json(result.to_json())

    pd.testing.assert_frame_equal(result.corrected, rebuilt.corrected)
    pd.testing.assert_frame_equal(result.cg_usage, rebuilt.cg_usage)
    pd.testing.assert_frame_equal(result.exclusions, rebuilt.exclusions)
    assert rebuilt.meter_id == result.meter_id
    assert rebuilt.tz == result.tz
    assert rebuilt.model_frequency == result.model_frequency
    assert rebuilt.fingerprint == result.fingerprint
    assert rebuilt.covered_window == result.covered_window
    assert rebuilt.cg_ids == result.cg_ids


def test_correction_after_reload_without_baseline_matches_reload_with_it(daily_env, comstock):
    """Correction reads no baseline data: a meter corrected on populations
    reloaded with reporting only matches the same reload with baseline attached."""
    tid = daily_env["treatment"].ids[0]
    reloads = {"with": [], "without": []}
    for population in (daily_env["treatment"], daily_env["pool"]):
        frames = comstock.population_meters(population)
        baseline = {mid: entry["baseline_df"] for mid, entry in frames.items()}
        reporting = {mid: entry["reporting_df"] for mid, entry in frames.items()}
        reloads["with"].append(_reload_with_reporting(population, reporting, baseline=baseline))
        reloads["without"].append(_reload_with_reporting(population, reporting))

    results = {
        key: correct_reporting(
            daily_env["selection"], treatment, pool, tid, settings=_PINNED_SETTINGS
        )
        for key, (treatment, pool) in reloads.items()
    }

    pd.testing.assert_frame_equal(results["without"].corrected, results["with"].corrected)
    pd.testing.assert_frame_equal(results["without"].cg_usage, results["with"].cg_usage)
    pd.testing.assert_frame_equal(results["without"].exclusions, results["with"].exclusions)


# ── billing end-to-end ───────────────────────────────────────────────────────


@pytest.mark.slow
def test_billing_end_to_end_schema(billing_results):
    for result in billing_results.values():
        assert list(result.corrected.columns) == _CORRECTED_COLUMNS
        assert result.model_frequency == "billing"


@pytest.mark.slow
def test_billing_end_to_end_corrects_twelve_monthly_reads(billing_results):
    """Each treatment meter corrects one row per monthly read over the reporting
    year. Inferred boundaries land within a day or two of the calendar month
    start: a near-identical daily rate across a month boundary can shift
    detection by a day."""
    for result in billing_results.values():
        assert len(result.corrected) == 12
        assert (result.corrected["datetime"].dt.day <= 2).all()


@pytest.mark.slow
def test_billing_end_to_end_corrected_is_finite(billing_results):
    for result in billing_results.values():
        assert np.isfinite(result.corrected["corrected"].to_numpy()).all()


# ── mixed-frequency end-to-end ───────────────────────────────────────────────


@pytest.mark.slow
def test_hourly_pool_daily_treatment_cadence_matches_received(mixed_hourly_daily_env):
    """A daily treatment corrects at its received (daily) cadence regardless of
    the comparison pool's finer (hourly) model frequency."""
    for result in mixed_hourly_daily_env["results"].values():
        assert result.model_frequency == "daily"
        assert len(result.corrected) == 365
        assert np.isfinite(result.corrected["corrected"].to_numpy()).all()


def test_finer_pool_billing_treatment_cadence_matches_received(mixed_billing_results):
    """A billing treatment corrects at its own inferred read periods regardless
    of the comparison pool's finer (hourly or daily) model frequency: one row
    per period, labelled by the period start."""
    for result in mixed_billing_results.values():
        assert result.model_frequency == "billing"
        assert list(result.corrected["datetime"]) == [
            start for start, _ in result.correction_periods
        ]


# ── output schema ────────────────────────────────────────────────────────────


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


# ── comparison-group membership and weights ──────────────────────────────────


def test_cluster_weights_renormalize_when_a_selected_cluster_has_no_columns():
    """A treatment meter's active cluster with no usable comparison column
    contributes no label, so ``t_weight`` renormalizes to sum 1 over the clusters
    that ARE present, keeping the kernel's normalized point consistent with its
    raw-weight uncertainty (rather than passing a raw Sigma < 1)."""
    active = {0: 0.5, 1: 0.3, 2: 0.2}

    # cluster 1 has no usable column; clusters 0 and 2 contribute one each
    t_weight = _cluster_weights(active, [0, 2])

    np.testing.assert_allclose(t_weight, [5 / 7, 2 / 7])


def test_column_correlations_over_a_subset_equal_the_full_pool_restriction(guard_env):
    """Correlations over a subset of pool columns equal the full-pool correlations
    restricted to those columns, since ``_column_correlations`` works per column over
    finite pairs."""
    pool = guard_env["pool"]
    _, p_ids, observed, modeled, *_ = pool.prediction_matrices("reporting")

    full = _column_correlations(
        np.asarray(observed, dtype=np.float64), np.asarray(modeled, dtype=np.float64)
    )

    subset = [1, 3, 4]
    _, _, s_obs, s_mod, *_ = pool.prediction_matrices("reporting", ids=[p_ids[c] for c in subset])
    restricted = _column_correlations(
        np.asarray(s_obs, dtype=np.float64), np.asarray(s_mod, dtype=np.float64)
    )

    np.testing.assert_array_equal(restricted, full[subset])


def test_clusters_used_reflects_per_cluster_drop(guard_env, comstock):
    """``clusters_used`` counts, per read period, the clusters that retain at
    least three finite meters. Billing correction runs at read-period cadence, so
    degradation fires per read period. NaN-ing a whole read period of one meter
    in a 3-meter cluster drops that cluster below three finite pool meters there
    only, so ``clusters_used`` falls from 2 to 1 at that period and stays 2
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
    pred = injected_pool.prediction_frames("reporting")
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


def test_period_bounds_filter_emitted_rows(guard_env):
    """``period`` bounds filter the emitted rows; the covered window reflects the
    filtered output."""
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


def test_period_correction_matches_full_correction_on_its_rows(daily_dq_env):
    """A timestep's corrected value never depends on ``period``: the rows a
    sub-period correction emits equal the full correction's rows exactly, in
    both ``corrected`` and ``cg_usage``. Reporting coverage is judged over the
    treatment's full reporting span, so a pool meter with a half-month outage
    inside ``period`` (above the coverage floor over the year, below it over the
    quarter) stays in the comparison group under both calls. The outage is built
    on a daily meter: a billing meter loses whole reads, never half a month."""
    start = pd.Timestamp("2019-04-01", tz="America/Chicago")
    end = pd.Timestamp("2019-06-30", tz="America/Chicago")
    tid = daily_dq_env["treatment_ids"][0]
    hole_id = daily_dq_env["pool_ids"][0]

    def blank_half_may(raw):
        raw = raw.copy()
        in_hole = (raw["datetime"].dt.month == 5) & (raw["datetime"].dt.day <= 15)
        raw.loc[in_hole, "observed"] = np.nan

        return raw

    holed = _reporting_override(daily_dq_env, hole_id, blank_half_may)
    treatment, pool = _build_dq(daily_dq_env, pool_overrides={hole_id: holed})
    args = (daily_dq_env["selection"], treatment, pool)

    full = correct_reporting(*args, tid)
    sub = correct_reporting(*args, tid, period=(start, end))

    assert hole_id in full.cg_ids, f"{hole_id} pruned over the full span: {full.cg_ids}"
    assert sub.cg_ids == full.cg_ids

    for frame in ("corrected", "cg_usage"):
        full_frame = getattr(full, frame)
        in_period = full_frame["datetime"].between(start, end)
        expected = full_frame[in_period].reset_index(drop=True)
        pd.testing.assert_frame_equal(getattr(sub, frame), expected, check_exact=True)


# ── received-cadence reduction ───────────────────────────────────────────────


def test_sum_to_periods_sums_finite_rows_with_min_count_one():
    """The reduction map sums a period's finite native rows per column. min_count
    is 1 everywhere: a period whose column has no finite native rows aggregates
    to NaN, never 0. Rows mapped to -1 (outside every period) contribute
    nowhere."""
    values = np.array([[2.0, 10.0], [3.0, 20.0], [np.nan, np.nan]])
    codes = np.array([0, 0, 1])

    summed = _sum_to_periods(values, codes, 2)

    assert np.array_equal(summed[0], [5.0, 30.0])
    # second period's only native row is NaN -> NaN, not 0.0
    assert np.isnan(summed[1]).all()

    dropped = _sum_to_periods(np.array([[5.0], [7.0]]), np.array([-1, 0]), 1)
    assert dropped[0, 0] == 7.0


def test_quadrature_to_periods_combines_uncertainty():
    """Per-row uncertainties combine to a period as the root of their summed
    squares; a period with no finite row is NaN."""
    unc = np.array([[3.0], [4.0], [np.nan]])
    codes = np.array([0, 0, 1])

    quad = _quadrature_to_periods(unc, codes, 2)

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

    outside = pd.DatetimeIndex(
        [daily[0] - pd.Timedelta(hours=1), daily[-1] + pd.Timedelta(days=1)]
    )
    assert np.array_equal(_period_codes(outside.as_unit("ns").asi8, edges), [-1, -1])


@pytest.mark.parametrize(
    "start, counts",
    [("2019-11-02", [24, 25, 24, 24]), ("2019-03-09", [24, 23, 24, 24])],
    ids=["fall_back", "spring_forward"],
)
def test_native_edges_bin_a_dst_day_by_its_local_hours(start, counts):
    """Daily periods are local days: the hourly rows of a DST transition day
    (25 on fall-back, 23 on spring-forward) all bin into that day and no other."""
    daily = pd.date_range(start, periods=4, freq="D", tz="America/Chicago")
    hourly = pd.date_range(daily[0], periods=sum(counts), freq="h")

    codes = _period_codes(hourly.as_unit("ns").asi8, _native_edges(daily))

    np.testing.assert_array_equal(np.bincount(codes, minlength=4), counts)


def test_billing_edges_bin_hourly_rows_by_local_read_days_across_dst():
    """Read periods are inclusive local-day spans: a read holding the
    spring-forward day collects one hour fewer than its days times 24, one
    holding the fall-back day one hour more, and every hourly row lands in
    exactly one read."""
    tz = "America/Chicago"
    periods = [
        (pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz))
        for start, end in [
            ("2019-03-01", "2019-03-31"),
            ("2019-04-01", "2019-10-31"),
            ("2019-11-01", "2019-11-30"),
        ]
    ]
    hourly = pd.date_range(
        periods[0][0], pd.Timestamp("2019-12-01", tz=tz), freq="h", inclusive="left"
    )

    codes = _period_codes(hourly.as_unit("ns").asi8, _billing_edges(periods))

    expected = [31 * 24 - 1, 214 * 24, 30 * 24 + 1]
    np.testing.assert_array_equal(np.bincount(codes, minlength=3), expected)


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

    own = guard_env["treatment"].prediction_frames("reporting", ids=[tid])[tid]
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
    own = treatment.prediction_frames("reporting", ids=[tid])[tid]
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
    is NaN at that period and finite elsewhere; a summed-to-zero period would
    instead keep the cluster alive and the row finite."""
    treatment = guard_env["treatment"]
    tid = treatment.ids[0]

    injected_pool = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, guard_env["pool"], guard_env["df_r"])
    )
    pred = injected_pool.prediction_frames("reporting")
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


# ── meter-level guards ───────────────────────────────────────────────────────


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


def _guard_short_baseline_excluded_at_selection(guard_env):
    """A real selection over a treatment meter with a single baseline month,
    which the ``MONTH`` loadshape check leaves out of ``treatment_weights``."""
    pool = guard_env["pool"]
    tid = guard_env["treatment"].ids[1]
    treatment = TreatmentGroup.from_fit_models(
        guard_env["comstock"].population_meters(guard_env["treatment"])
    )
    shorten_baselines(treatment, [tid])
    selection = select_comparison_group(
        treatment,
        pool,
        method="random_sampling",
        method_settings=RandomSamplingSettings(
            n_meters_total=6, n_meters_per_treatment=None, seed=0
        ),
    )

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
    treatment = TreatmentGroup.from_fit_models(
        guard_env["comstock"].population_meters(guard_env["treatment"], drop_reporting=(tid,))
    )

    return selection, treatment, pool, tid


def _guard_prediction_failure(guard_env):
    selection = guard_env["selection"]
    pool = guard_env["pool"]
    tids = guard_env["treatment"].ids
    tid = tids[0]
    models = {tid: _disqualified(guard_env["treatment"]._meters[tid].model)}
    meters = guard_env["comstock"].population_meters(guard_env["treatment"], models=models)

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
        (
            _guard_short_baseline_excluded_at_selection,
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
        "short_baseline_excluded_at_selection",
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
    correction raises with that reason."""
    victim = billing_dq_env["treatment_ids"][1]
    flat = _reporting_override(billing_dq_env, victim, lambda raw: raw.assign(observed=123.0))
    treatment, pool = _build_dq(billing_dq_env, treatment_overrides={victim: flat})

    with pytest.raises(exclusions.MeterCorrectionError) as excinfo:
        correct_reporting(billing_dq_env["selection"], treatment, pool, victim)

    row = excinfo.value.exclusions.set_index("id").loc[victim]
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_data"
    assert "no recoverable read" in row["reason"]
    # the flat meter's documented path is explicit ``read_boundaries``
    assert "read_boundaries" in row["reason"]


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
    error naming both model frequencies."""
    ids = comstock.ids("billing")
    treatment, _ = _build_dq(daily_dq_env)
    pool = ComparisonPool.from_fit_models(comstock.meters(model_bank, "billing", ids[:6]))

    with pytest.raises(ValueError, match="billing.*coarser.*daily"):
        correct_reporting(
            daily_dq_env["selection"], treatment, pool, daily_dq_env["treatment_ids"][0]
        )


# ── pool-side drops ──────────────────────────────────────────────────────────


_SPAN_COVERAGE_REASON = "reporting coverage below minimum over the treatment meter's reporting span"


def _ledger_row(result, mid):
    return result.exclusions.set_index("id").loc[mid]


def _coverage_from_detail(row):
    """The coverage a ``reporting_coverage`` ledger row records in its detail."""
    match = _COVERAGE_DETAIL.search(row["detail"])
    assert match, f"no reporting span coverage in detail {row['detail']!r}"
    coverage = float(match.group(1))

    return coverage


@pytest.fixture(scope="module")
def pool_missing_reporting(guard_env):
    """One treatment meter corrected against a pool whose first two meters carry
    no reporting data at all."""
    missing_ids = guard_env["pool"].ids[:2]
    pool = ComparisonPool.from_fit_models(
        guard_env["comstock"].population_meters(guard_env["pool"], drop_reporting=missing_ids)
    )
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


def test_cg_member_absent_from_pool_is_recorded(guard_env, comstock, model_bank):
    """A comparison-group member the given pool lacks is recorded on this
    meter's ledger as absent from the pool population, and the seven members
    the pool holds keep the correction going."""
    ids = guard_env["ids"]
    absent = str(ids[2])
    pool = ComparisonPool.from_fit_models(comstock.meters(model_bank, "billing", ids[3:10]))
    tid = guard_env["treatment"].ids[0]

    result = correct_reporting(guard_env["selection"], guard_env["treatment"], pool, tid)

    row = _ledger_row(result, absent)
    assert row["stage"] == "correction"
    assert row["origin"] == "pool_membership"
    assert row["reason"] == "comparison-group member absent from the pool population"
    assert sorted(result.cg_ids) == sorted(str(mid) for mid in ids[3:10]), (
        f"expected the seven pool-held members, got {result.cg_ids}"
    )
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
    """``prediction_frames`` caches successes only, so a pool meter whose reporting
    prediction raises is re-attempted, and re-ledgered, in every per-meter call
    that selects it; the row is not deduped away."""
    victim = guard_env["pool"].ids[0]
    tids = guard_env["treatment"].ids
    models = {victim: _disqualified(guard_env["pool"]._meters[victim].model)}
    meters = guard_env["comstock"].population_meters(guard_env["pool"], models=models)

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


# ── reporting coverage and observed DQ at correction ─────────────────────────


def test_reporting_gross_hole_pool_meter_pruned_before_predict(daily_dq_env):
    """A pool meter whose reporting covers only part of the treatment meter's
    reporting span (a gross hole) fails coverage at correction: it is ledgered
    with its coverage in ``detail`` and never predicted, while the remaining five
    comparison meters keep the treatment meter corrected."""
    victim = daily_dq_env["pool_ids"][0]
    df_r = daily_dq_env["df_r"]
    cutoff = df_r.xs(int(victim), level="id").index.min() + pd.Timedelta(days=180)
    truncated = _reporting_override(daily_dq_env, victim, lambda raw: raw[raw["datetime"] < cutoff])
    treatment, pool = _build_dq(daily_dq_env, pool_overrides={victim: truncated})

    result = correct_reporting(
        daily_dq_env["selection"], treatment, pool, daily_dq_env["treatment_ids"][0]
    )

    row = _ledger_row(result, victim)
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_coverage"
    assert row["reason"] == _SPAN_COVERAGE_REASON
    assert _coverage_from_detail(row) < 0.9
    # pruned before prediction: the failed meter never enters the reporting cache
    assert victim not in pool._reporting_pred
    assert not result.corrected.empty


def test_reporting_hole_treatment_meter_raises_below_coverage(daily_dq_env):
    """The coverage floor applied to the treatment meter is all-or-nothing: a
    meter whose 25-day interior NaN hole leaves it covering 340/365 of its own
    reporting span raises under a 0.95 floor rather than returning a partly
    corrected series, with its coverage in the ledger detail."""
    victim = daily_dq_env["treatment_ids"][1]
    df_r = daily_dq_env["df_r"]
    hole_start = df_r.xs(int(victim), level="id").index.min() + pd.Timedelta(days=150)
    hole_end = hole_start + pd.Timedelta(days=25)
    holed = _reporting_override(
        daily_dq_env,
        victim,
        lambda raw: raw.assign(
            observed=raw["observed"].where(
                (raw["datetime"] < hole_start) | (raw["datetime"] >= hole_end)
            )
        ),
    )
    treatment, pool = _build_dq(daily_dq_env, treatment_overrides={victim: holed})
    settings = CGCorrectionSettings(min_reporting_coverage=0.95)

    with pytest.raises(exclusions.MeterCorrectionError) as excinfo:
        correct_reporting(daily_dq_env["selection"], treatment, pool, victim, settings=settings)

    row = excinfo.value.exclusions.set_index("id").loc[victim]
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_coverage"
    assert row["reason"] == _SPAN_COVERAGE_REASON
    assert _coverage_from_detail(row) == pytest.approx(340 / 365, abs=1e-4)
    assert victim not in treatment._reporting_pred


@pytest.mark.parametrize("dq_env_name", ["daily_dq_env", "hourly_dq_env"])
def test_reporting_flat_treatment_meter_disqualified_via_unique_values(request, dq_env_name):
    """A treatment meter with non-zero identical reporting reads has full
    coverage but fails the eemeter unique-values check at construction. The
    correction stage enforces that observed disqualification at any native model
    frequency: the meter is ledgered verbatim, pruned before predicting, and the
    call raises."""
    dq_env = request.getfixturevalue(dq_env_name)
    victim = dq_env["treatment_ids"][1]
    flat = _reporting_override(dq_env, victim, lambda raw: raw.assign(observed=123.0))
    treatment, pool = _build_dq(dq_env, treatment_overrides={victim: flat})

    with pytest.raises(exclusions.MeterCorrectionError) as excinfo:
        correct_reporting(dq_env["selection"], treatment, pool, victim)

    row = excinfo.value.exclusions.set_index("id").loc[victim]
    assert row["stage"] == "correction"
    assert row["origin"] == "reporting_data"
    assert row["reason"] == "reporting observed data disqualified"
    assert "insufficient_unique_observed_values" in row["detail"]
    assert victim not in treatment._reporting_pred


def test_reporting_joint_comissing_treatment_meter_disqualified(daily_dq_env):
    """A treatment meter whose observed and temperature go missing on DISJOINT
    day sets trips ONLY the eemeter joint valid-days check. The correction stage enforces that
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

    holed = _reporting_override(daily_dq_env, victim, _joint_comissing)
    treatment, pool = _build_dq(daily_dq_env, treatment_overrides={victim: holed})

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


def test_reporting_temperature_only_disqualification_keeps_treatment_meter(daily_dq_env):
    """A treatment meter whose reporting data carries only a temperature
    disqualification (one month's temperature coverage below 90%, while its
    observed-and-temperature coverage of the reporting span stays above the
    floor) is corrected: the correction stage prunes observed and joint
    disqualifications only."""
    victim = daily_dq_env["treatment_ids"][1]

    def _temperature_gap(raw):
        raw = raw.copy()
        # 8 consecutive days: at least 4 fall in one month, so that month's
        # temperature coverage is at most 27/31
        raw.loc[raw.index[40:48], "temperature"] = np.nan

        return raw

    holed = _reporting_override(daily_dq_env, victim, _temperature_gap)
    treatment, pool = _build_dq(daily_dq_env, treatment_overrides={victim: holed})
    dq_names = [w.qualified_name for w in treatment.meter(victim).reporting_data.compliance.disqualification]

    result = correct_reporting(daily_dq_env["selection"], treatment, pool, victim)

    assert dq_names == ["eemeter.sufficiency_criteria.missing_monthly_temperature_data"], (
        f"expected a lone monthly temperature disqualification, got {dq_names}"
    )
    assert result.meter_id == victim
    assert victim not in set(result.exclusions["id"])


def test_reporting_large_savings_meter_is_not_disqualified(daily_dq_env):
    """A treatment meter whose reporting consumption dropped sharply (large
    savings) still varies and fully covers the window, so no observed
    disqualification or coverage failure fires: the correction stage runs no
    extreme-value screen on reporting observed, which would otherwise flag a
    meter precisely BECAUSE its savings are large."""
    saver = daily_dq_env["treatment_ids"][1]
    reduced = _reporting_override(
        daily_dq_env, saver, lambda raw: raw.assign(observed=raw["observed"] * 0.3)
    )
    treatment, pool = _build_dq(daily_dq_env, treatment_overrides={saver: reduced})

    result = correct_reporting(daily_dq_env["selection"], treatment, pool, saver)

    assert result.meter_id == saver
    assert saver not in set(result.exclusions["id"])


def test_interior_reporting_hole_clears_coverage_and_degrades_its_cluster(daily_dq_env):
    """A pool meter's reporting carries a real 25-day interior NaN hole. It covers
    340/365 = 0.9315 of the treatment meter's reporting span, above the 0.9 default floor, so
    no ``reporting_coverage`` row is written. Its 3-meter cluster has only two finite
    meters during the hole and drops from those timesteps; the other 3-meter cluster
    keeps ``corrected`` finite there, and ``clusters_used`` falls from 2 to 1 exactly
    over the hole and nowhere else."""
    treatment_ids = daily_dq_env["treatment_ids"]
    pool_ids = daily_dq_env["pool_ids"]
    victim = pool_ids[0]
    df_r = daily_dq_env["df_r"]

    victim_index = df_r.xs(int(victim), level="id").index
    hole_start = victim_index.min() + pd.Timedelta(days=150)
    hole_end = hole_start + pd.Timedelta(days=25)
    hole_dates = victim_index[(victim_index >= hole_start) & (victim_index < hole_end)]
    holed = _reporting_override(
        daily_dq_env,
        victim,
        lambda raw: raw.assign(
            observed=np.where(
                (raw["datetime"] >= hole_start) & (raw["datetime"] < hole_end),
                np.nan,
                raw["observed"],
            )
        ),
    )
    treatment, pool = _build_dq(daily_dq_env, pool_overrides={victim: holed})

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


# ── heterogeneous reporting spans ────────────────────────────────────────────


@pytest.fixture(scope="module")
def heterogeneous_span_results(daily_dq_env):
    """Two daily treatment meters over different reporting spans: the second
    carries only the first 300 days of its real ComStock reporting. Each is
    judged over its own span, so both clear the default coverage floor."""
    long_id, short_id = daily_dq_env["treatment_ids"]
    truncated = _reporting_override(daily_dq_env, short_id, lambda raw: raw.iloc[:300])
    treatment, pool = _build_dq(daily_dq_env, treatment_overrides={short_id: truncated})

    results = {
        tid: correct_reporting(daily_dq_env["selection"], treatment, pool, tid)
        for tid in (long_id, short_id)
    }
    bundle = {"long_id": long_id, "short_id": short_id, "results": results}

    return bundle


def test_heterogeneous_spans_emit_each_meters_own_index(heterogeneous_span_results):
    """Each meter's frame spans its own prediction index: the truncated meter emits 300 rows."""
    long_result = heterogeneous_span_results["results"][heterogeneous_span_results["long_id"]]
    short_result = heterogeneous_span_results["results"][heterogeneous_span_results["short_id"]]

    assert len(long_result.corrected) == 365
    assert len(short_result.corrected) == 300
    assert np.isfinite(short_result.corrected["corrected"].to_numpy()).all()
    assert short_result.covered_window[1] < long_result.covered_window[1]


def _short_span_case(daily_dq_env, n_days=300):
    """The second treatment meter and the first pool meter, both cut to the first
    ``n_days`` of their real ComStock reporting, so the two share one short span."""
    short_id = daily_dq_env["treatment_ids"][1]
    short_pool_id = daily_dq_env["pool_ids"][0]
    overrides = {
        mid: _reporting_override(daily_dq_env, mid, lambda raw: raw.iloc[:n_days])
        for mid in (short_id, short_pool_id)
    }

    return short_id, short_pool_id, overrides


def test_coverage_ignores_other_treatment_meters_spans(daily_dq_env):
    """Holding the selection fixed, a short-span treatment meter's coverage
    verdict, comparison group and corrected values are the same whether it is
    corrected alone or alongside a full-year treatment meter: the other meter's
    span never enters the coverage check."""
    records = daily_dq_env["records"]
    short_id, short_pool_id, overrides = _short_span_case(daily_dq_env)
    pool = ComparisonPool.from_fit_models(
        _meters_with_overrides(records, daily_dq_env["pool_ids"], overrides)
    )
    alone = TreatmentGroup.from_fit_models(
        _meters_with_overrides(records, [short_id], overrides)
    )
    joined = TreatmentGroup.from_fit_models(
        _meters_with_overrides(records, daily_dq_env["treatment_ids"], overrides)
    )

    results = {
        name: correct_reporting(daily_dq_env["selection"], treatment, pool, short_id)
        for name, treatment in (("alone", alone), ("joined", joined))
    }

    for name, result in results.items():
        coverage_rows = result.exclusions[result.exclusions["origin"] == "reporting_coverage"]
        assert coverage_rows.empty, f"{name}: unexpected coverage rows {list(coverage_rows['id'])}"
        assert short_pool_id in result.cg_ids, f"{name}: short pool meter missing from the group"

    assert results["alone"].cg_ids == results["joined"].cg_ids
    pd.testing.assert_frame_equal(
        results["alone"].corrected, results["joined"].corrected, check_exact=True
    )


def test_pool_member_is_judged_over_the_treatment_meters_span(daily_dq_env):
    """A pool member cut to 300 days covers a 300-day treatment meter fully and
    stays in its comparison group, but covers only 300/365 of a full-year
    treatment meter and is pruned from that meter's group."""
    long_id = daily_dq_env["treatment_ids"][0]
    short_id, short_pool_id, overrides = _short_span_case(daily_dq_env)
    treatment, pool = _build_dq(
        daily_dq_env, treatment_overrides=overrides, pool_overrides=overrides
    )

    short_result = correct_reporting(daily_dq_env["selection"], treatment, pool, short_id)
    long_result = correct_reporting(daily_dq_env["selection"], treatment, pool, long_id)

    assert short_pool_id in short_result.cg_ids
    assert short_pool_id not in long_result.cg_ids
    row = _ledger_row(long_result, short_pool_id)
    assert row["origin"] == "reporting_coverage"
    assert row["reason"] == _SPAN_COVERAGE_REASON
    assert _coverage_from_detail(row) == pytest.approx(300 / 365, abs=1e-4)


def test_pool_reporting_past_the_treatment_span_leaves_corrected_unc_unchanged(daily_dq_env):
    """The comparison-group correlation behind ``corrected_unc`` (read when pool
    meters carry observed uncertainty) spans the pool rows on the treatment meter's
    reporting index, so pool reporting past its last timestamp changes nothing."""
    tid = daily_dq_env["treatment_ids"][1]
    pool_ids = daily_dq_env["pool_ids"]
    short = _reporting_override(daily_dq_env, tid, lambda raw: raw.iloc[:300])
    treatment = TreatmentGroup.from_fit_models(
        _meters_with_overrides(daily_dq_env["records"], [tid], {tid: short})
    )
    end = treatment.meter(tid).reporting_data.df.index.max()

    pool_records = {
        mid: dict(daily_dq_env["records"][mid], observed_unc=POOL_OBSERVED_UNC)
        for mid in pool_ids
    }
    cut = {
        mid: _reporting_override(daily_dq_env, mid, lambda raw: raw[raw["datetime"] <= end])
        for mid in pool_ids
    }
    pools = {
        "extended": ComparisonPool.from_fit_models(
            _meters_with_overrides(pool_records, pool_ids, {})
        ),
        "truncated": ComparisonPool.from_fit_models(
            _meters_with_overrides(pool_records, pool_ids, cut)
        ),
    }
    pool_ends = {
        name: max(pool.meter(mid).reporting_data.df.index.max() for mid in pool_ids)
        for name, pool in pools.items()
    }
    assert pool_ends["extended"] > end, f"pool ends {pool_ends['extended']}, treatment {end}"
    assert pool_ends["truncated"] == end, f"pool ends {pool_ends['truncated']}, treatment {end}"

    results = {
        name: correct_reporting(daily_dq_env["selection"], treatment, pool, tid)
        for name, pool in pools.items()
    }

    assert results["extended"].cg_ids == results["truncated"].cg_ids == pool_ids
    extended_unc = results["extended"].corrected["corrected_unc"].to_numpy()
    truncated_unc = results["truncated"].corrected["corrected_unc"].to_numpy()
    assert np.isfinite(truncated_unc).all(), "truncated-pool corrected_unc has non-finite rows"
    np.testing.assert_allclose(extended_unc, truncated_unc, rtol=1e-12, atol=0.0)


# ── individual meter matching ────────────────────────────────────────────────


@pytest.mark.slow
def test_imm_correction_uses_the_shared_cluster_union_not_the_matched_rows(
    variant_env, comstock
):
    """IMM's correction deliberately draws on the shared cluster-0 union, with
    duplicate selection rows collapsed, NOT the treatment meter's own matched
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
    pred = injected_pool.prediction_frames("reporting", ids=[victim])
    substrate = pred[victim].index
    march = substrate[substrate.month == 3]
    pred[victim].loc[march, "observed"] = np.nan

    result = correct_reporting(selection, treatment, injected_pool, tid)

    usage = result.cg_usage.set_index("datetime")["cg_usage_fraction"]
    march_start = pd.Timestamp("2019-03-01", tz=treatment.tz)

    assert usage.loc[march_start] == pytest.approx((len(members) - 1) / len(members))
    other = [d for d in usage.index if d != march_start]
    assert np.allclose(usage.loc[other].to_numpy(), 1.0)


# ── prior semantics ──────────────────────────────────────────────────────────


def _prior_from_corrected(meter_id, corrected, model_frequency, tz="America/Chicago"):
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
        model_frequency=model_frequency,
        fingerprint="test-fingerprint",
        correction_periods=[],
        cg_ids=[],
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
    """At billing model frequency the trailing (possibly partial) read period is
    re-drawn on extension, so a mismatch there is excluded from the
    slice-invariance check and does not raise."""
    months = pd.date_range("2019-01-01", periods=3, freq="MS", tz="America/Chicago")
    prior = _prior_from_corrected("m1", _corrected_frame(months, [10.0, 20.0, 30.0]), "billing")
    new_corrected = _corrected_frame(months, [10.0, 20.0, 999.0])

    _check_prior(new_corrected, prior, "billing")


def test_check_prior_billing_earlier_month_mismatch_raises():
    """A mismatch in a non-trailing month is a genuine slice-invariance
    violation and raises even at billing model frequency."""
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
    """Daily model frequency has no trailing-period carve-out, so a mismatch on the
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
    before anything reads the prior's read calendar: freezing the wrong meter's
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


# ── incremental reporting windows ────────────────────────────────────────────


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

    full = TreatmentGroup.from_fit_models(
        _rewrap_population(comstock, daily_env["treatment"], df_r)
    )
    result_full = correct_reporting(selection, full, pool, tid, prior=result_half)

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

    full = TreatmentGroup.from_fit_models(
        _rewrap_population(comstock, daily_env["treatment"], df_r)
    )

    with pytest.raises(ValueError, match="prior correction mismatch"):
        correct_reporting(selection, full, pool, tid, prior=tampered)


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
    settings = CGCorrectionSettings(min_reporting_coverage=0.4)

    pool_half = ComparisonPool.from_fit_models(
        _rewrap_population(comstock, daily_env["pool"], df_r, cutoff=_MIDYEAR)
    )
    result_half = correct_reporting(selection, treatment, pool_half, tid, settings=settings)

    late = result_half.corrected[result_half.corrected["datetime"] >= _MIDYEAR]
    assert not np.isfinite(late["corrected"].to_numpy()).any()

    pool_full = _reload_with_reporting(pool_half, _reporting_map(df_r, pool_ids))
    result_full = correct_reporting(
        selection, treatment, pool_full, tid, settings=settings, prior=result_half
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
    prior = correct_reporting(
        guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid
    )
    other_selection, treatment, pool, _ = _guard_too_few_cg_meters(guard_env)

    with pytest.raises(ValueError, match="different selection"):
        correct_reporting(other_selection, treatment, pool, tid, prior=prior)


def test_prior_freezes_the_comparison_group_against_new_entrants(guard_env):
    """Pool meters that lacked reporting data when the prior was made do not
    join the group later: the prior's members are reused as they are, so the
    overlapping points reproduce exactly."""
    tid = guard_env["treatment"].ids[0]
    missing = guard_env["pool"].ids[:2]
    reduced_pool = ComparisonPool.from_fit_models(
        guard_env["comstock"].population_meters(guard_env["pool"], drop_reporting=missing)
    )
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
    prior = correct_reporting(
        guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid
    )
    models = {victim: _disqualified(guard_env["pool"]._meters[victim].model)}
    meters = guard_env["comstock"].population_meters(guard_env["pool"], models=models)

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
    treatment, pool = _build_dq(daily_dq_env)
    prior = correct_reporting(daily_dq_env["selection"], treatment, pool, tid)
    assert victim in prior.cg_ids

    flat = _reporting_override(daily_dq_env, victim, lambda raw: raw.assign(observed=123.0))
    _, flat_pool = _build_dq(daily_dq_env, pool_overrides={victim: flat})

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
    prior = correct_reporting(
        guard_env["selection"], guard_env["treatment"], guard_env["pool"], tid
    )
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

    treatment = TreatmentGroup.from_fit_models(
        _rewrap_population(comstock, daily_env["treatment"], df_r)
    )

    # without the prior, the half-year pool fails coverage against the full-year window
    with pytest.raises(exclusions.MeterCorrectionError):
        correct_reporting(selection, treatment, pool_half, tid)

    lagging = correct_reporting(selection, treatment, pool_half, tid, prior=prior)
    assert lagging.cg_ids == prior.cg_ids
    late = lagging.corrected[lagging.corrected["datetime"] >= _MIDYEAR]
    assert not np.isfinite(late["corrected"].to_numpy()).any()

    pool_full = _reload_with_reporting(pool_half, _reporting_map(df_r, daily_data["pool_ids"]))
    full = correct_reporting(selection, treatment, pool_full, tid, prior=prior)
    assert np.isfinite(full.corrected["corrected"].to_numpy()).all()

    early = full.corrected[full.corrected["datetime"] < _MIDYEAR]
    merged = early.merge(
        prior.corrected[["id", "datetime", "corrected"]],
        on=["id", "datetime"],
        suffixes=("", "_prior"),
    )
    assert len(merged) == len(prior.corrected)
    assert np.array_equal(merged["corrected"].to_numpy(), merged["corrected_prior"].to_numpy())


# ── hourly DST fall-back ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hourly_dst_result(comstock, model_bank):
    """Hourly correction across the America/Chicago DST fall-back day. Random
    sampling avoids the clustering minimum-size floor so a tiny (1 treatment,
    6 pool) population suffices."""
    ids = comstock.ids("hourly")

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


@pytest.mark.slow
def test_dst_fallback_day_has_no_duplicate_or_missing_hours(hourly_dst_result):
    corrected = hourly_dst_result.corrected

    assert not corrected.duplicated(["id", "datetime"]).any()

    day = corrected[corrected["datetime"].dt.date == _DST_FALLBACK_DATE]
    # the fall-back day has 25 local hours (the 01:00 hour occurs twice)
    assert len(day) == 25
    assert np.isfinite(day["corrected"].to_numpy()).all()


# ── equivalence snapshot artifact ────────────────────────────────────────────


@pytest.fixture(scope="module")
def equivalence_snapshot():
    """The pinned correction and savings output the per-meter API is checked against."""
    return load_equivalence_snapshot()


def test_equivalence_snapshot_loads_every_variant_as_correction_tables(equivalence_snapshot):
    """The snapshot loads every variant, each pinned meter's tables restored with
    the correction columns over datetimes in the variant's timezone."""
    variants = equivalence_snapshot["variants"]

    assert set(variants) == set(VARIANT_NAMES)

    for name, variant in variants.items():
        assert variant["meters"], f"{name}: no per-meter payloads recorded"

        for mid, meter in variant["meters"].items():
            label = f"{name}/{mid}"

            assert list(meter["corrected"].columns) == _CORRECTED_COLUMNS, label
            assert list(meter["exclusions"].columns) == _LEDGER_COLUMNS, label
            assert str(meter["corrected"]["datetime"].dt.tz) == variant["tz"], label


# ── correction equivalence with the pinned snapshot ──────────────────────────


def _assert_values_match(actual, expected, rtol, atol, label):
    """Every numeric column, compared over the meter's own span with NaN
    positions required to line up."""
    for column in actual.columns:
        if column in ("id", "datetime"):
            continue

        np.testing.assert_allclose(
            actual[column].to_numpy(dtype=np.float64),
            expected[column].to_numpy(dtype=np.float64),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
            err_msg=f"{label}: {column}",
        )


@pytest.mark.regression
@pytest.mark.parametrize("variant_name", VARIANT_NAMES)
def test_per_meter_correction_reproduces_the_pinned_snapshot(
    variant_name, variant_env, equivalence_snapshot
):
    """The per-meter correction reproduces the pinned output meter by meter: to 1e-9
    relative for the billing variants, and at float32 precision for the hourly/daily
    ones, whose pinned treatment values carry a float32 rounding the per-meter path's
    float64 prediction frames do not. The hourly-pool variant shares that tolerance,
    which leaves room for platform-level differences in the hourly model fits."""
    env = variant_env(variant_name)
    pinned = equivalence_snapshot["variants"][variant_name]

    if variant_name in _EXACT_EQUIVALENCE_VARIANTS:
        rtol, atol = _EXACT_RTOL, 0.0
    else:
        rtol, atol = _FLOAT32_RTOL, _EQUIVALENCE_ATOL

    for mid, expected in pinned["meters"].items():
        result = correct_reporting(
            env["selection"], env["treatment"], env["pool"], mid, settings=_PINNED_SETTINGS
        )
        label = f"{variant_name}/{mid}"
        expected_datetimes = list(expected["corrected"]["datetime"])

        assert result.meter_id == mid, label
        assert list(result.corrected["datetime"]) == expected_datetimes, label
        _assert_values_match(result.corrected, expected["corrected"], rtol, atol, label)

        # the usage mask is finiteness-driven, so the float32 round-trip cannot
        # move it: cg_usage matches to the exact tolerance for every variant
        _assert_values_match(result.cg_usage, expected["cg_usage"], _EXACT_RTOL, 0.0, label)

        assert result.correction_periods == expected["correction_periods"], label
        pd.testing.assert_frame_equal(
            result.exclusions.reset_index(drop=True),
            expected["exclusions"].reset_index(drop=True),
            check_dtype=False,
            obj=label,
        )
