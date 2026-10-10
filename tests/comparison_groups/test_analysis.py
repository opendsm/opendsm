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

import warnings

import numpy as np
import pandas as pd
import pytest

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.analysis import MeterAnalysis
from opendsm.comparison_groups.population import ComparisonPool, TreatmentGroup
from opendsm.comparison_groups.random_sampling.settings import Settings as RS_Settings
from opendsm.comparison_groups.savings.settings import CGCorrectionSettings
from opendsm.comparison_groups.selection import (
    ComparisonGroupSelection,
    SelectionMethod,
    _finalize_clusters,
    _normalize_treatment_weights,
    select_comparison_group,
)



# ── builders ─────────────────────────────────────────────────────────────────


def _rewrap_meters(comstock, population):
    """A meters dict reusing a population's fitted models with the ComStock
    frames behind them (no refits)."""
    granularity = population.granularity
    meters = {}

    for mid, rec in population._meters.items():
        meters[mid] = {
            "model": rec.model,
            "baseline_df": comstock.baseline(granularity, mid),
            "reporting_df": comstock.reporting(granularity, mid),
        }

    return meters


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def billing_env(variant_env):
    """Real ComStock monthly treatment (6) / pool (40) populations with a
    clustering selection, shared with the correction tests."""
    return variant_env("billing_cg_clustering")


def _analysis(env, treatment=None, treatment_id=None, selection=None):
    """A MeterAnalysis over the env's first treatment meter unless told otherwise."""
    treatment = treatment or env["treatment"]
    selection = selection or env["selection"]
    treatment_id = treatment_id or treatment.ids[0]
    built = MeterAnalysis(selection, treatment, env["pool"], treatment_id)

    return built


# ── correction_settings.min_window_coverage across both stages ──────────────


@pytest.fixture(scope="module")
def coverage_env(comstock, model_bank):
    """Real ComStock daily population (2 treatment, 8 pool) on pinned models.
    Individual baseline/reporting frames are swapped per test to create a
    coverage hole without refitting."""
    df_b, df_r = comstock.frames("daily")
    ids = comstock.ids("daily")[:10]
    treatment_ids = [str(i) for i in ids[:2]]
    pool_ids = [str(i) for i in ids[2:10]]
    records = comstock.meters(model_bank, "daily", ids, reporting=False)
    env = {
        "records": records,
        "df_b": df_b,
        "df_r": df_r,
        "treatment_ids": treatment_ids,
        "pool_ids": pool_ids,
    }

    return env


def _daily_reporting_for(df_r, mid):
    return df_r.xs(int(mid), level="id").reset_index()


def _truncated_baseline(df_b, mid, n_days):
    """The first ``n_days`` of ``mid``'s real baseline, ~0.92 of the full
    365-day span; below a 0.95 coverage floor, above the 0.9 default."""
    raw = df_b.xs(int(mid), level="id").reset_index().iloc[:n_days]

    return raw


def _truncated_reporting(df_r, mid, n_days):
    return df_r.xs(int(mid), level="id").reset_index().iloc[:n_days]


def _coverage_population(
    cls, records, df_r, ids, baseline_overrides=None, reporting_overrides=None, **kwargs
):
    baseline_overrides = baseline_overrides or {}
    reporting_overrides = reporting_overrides or {}
    meters = {}

    for mid in ids:
        meters[mid] = {
            "model": records[mid]["model"],
            "baseline_df": baseline_overrides.get(mid, records[mid]["baseline_df"]),
            "reporting_df": reporting_overrides.get(mid, _daily_reporting_for(df_r, mid)),
        }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Population includes disqualified")
        population = cls.from_fit_models(meters, **kwargs)

    return population


def _build_coverage_case(coverage_env, min_window_coverage):
    """Treatment/pool with one baseline-hole and one reporting-hole pool meter,
    plus the selection produced at ``min_window_coverage``."""
    records = coverage_env["records"]
    df_b, df_r = coverage_env["df_b"], coverage_env["df_r"]
    treatment_ids = coverage_env["treatment_ids"]
    pool_ids = coverage_env["pool_ids"]
    base_hole_id, report_hole_id = pool_ids[0], pool_ids[1]

    treatment = _coverage_population(TreatmentGroup, records, df_r, treatment_ids)
    pool = _coverage_population(
        ComparisonPool,
        records,
        df_r,
        pool_ids,
        baseline_overrides={base_hole_id: _truncated_baseline(df_b, base_hole_id, 336)},
        reporting_overrides={report_hole_id: _truncated_reporting(df_r, report_hole_id, 336)},
    )
    method_settings = RS_Settings(n_meters_total=len(pool_ids), n_meters_per_treatment=None)
    selection = select_comparison_group(
        treatment,
        pool,
        method="random_sampling",
        method_settings=method_settings,
        basis="observed",
        min_window_coverage=min_window_coverage,
    )
    settings = CGCorrectionSettings(min_window_coverage=min_window_coverage)
    case = {
        "treatment": treatment,
        "pool": pool,
        "selection": selection,
        "settings": settings,
        "base_hole_id": base_hole_id,
        "report_hole_id": report_hole_id,
    }

    return case


def test_default_coverage_survives_a_partial_meter(coverage_env):
    """At the CGCorrectionSettings default (0.9), a pool meter covering ~0.92 of
    the group window at both baseline and reporting survives selection and
    correction."""
    case = _build_coverage_case(coverage_env, min_window_coverage=0.9)

    assert case["base_hole_id"] in case["selection"].clusters.index
    assert case["selection"].exclusions[
        case["selection"].exclusions["origin"] == "baseline_coverage"
    ].empty

    analysis = MeterAnalysis(
        case["selection"],
        case["treatment"],
        case["pool"],
        case["treatment"].ids[0],
        correction_settings=case["settings"],
    )
    analysis.correct()

    assert analysis.correction.exclusions[
        analysis.correction.exclusions["origin"] == "reporting_coverage"
    ].empty


def test_min_window_coverage_reaches_selection_and_correction_prune(coverage_env):
    """A single ``min_window_coverage`` reaches both stage inputs: raised to 0.95
    it prunes the ~0.92-coverage pool meter at selection (baseline) AND a
    different ~0.92-coverage pool meter at correction (reporting), neither of
    which is pruned at the 0.9 default."""
    case = _build_coverage_case(coverage_env, min_window_coverage=0.95)

    assert case["base_hole_id"] not in case["selection"].clusters.index
    selection_rows = case["selection"].exclusions[
        case["selection"].exclusions["origin"] == "baseline_coverage"
    ]
    assert list(selection_rows["id"]) == [case["base_hole_id"]]

    analysis = MeterAnalysis(
        case["selection"],
        case["treatment"],
        case["pool"],
        case["treatment"].ids[0],
        correction_settings=case["settings"],
    )
    analysis.correct()

    correction_rows = analysis.correction.exclusions[
        analysis.correction.exclusions["origin"] == "reporting_coverage"
    ]
    assert list(correction_rows["id"]) == [case["report_hole_id"]]


def test_pruned_cg_member_row_is_in_this_meters_log(coverage_env):
    """A pool meter pruned at correction is a member of this meter's comparison
    group, so its reporting_coverage row appears in the meter's own log."""
    case = _build_coverage_case(coverage_env, min_window_coverage=0.95)
    analysis = MeterAnalysis(
        case["selection"],
        case["treatment"],
        case["pool"],
        case["treatment"].ids[0],
        correction_settings=case["settings"],
    )
    analysis.correct()

    log = analysis.meter_log()
    pruned = log[log["origin"] == "reporting_coverage"]

    assert list(pruned["id"]) == [case["report_hole_id"]]


def test_explicit_reporting_window_governs_coverage(coverage_env):
    """Correction reads its group window from ``treatment.reporting_window``. An
    explicit window narrowed to one pool meter's heavily truncated reporting span
    gives that meter full coverage over the group window, so it survives
    correction even though treatment's own attached span is the full year."""
    records = coverage_env["records"]
    df_r = coverage_env["df_r"]
    treatment_ids = coverage_env["treatment_ids"]
    pool_ids = coverage_env["pool_ids"]
    victim = pool_ids[0]

    victim_reporting = _truncated_reporting(df_r, victim, 200)
    narrow_window = (victim_reporting["datetime"].min(), victim_reporting["datetime"].max())

    treatment = _coverage_population(
        TreatmentGroup, records, df_r, treatment_ids, reporting_window=narrow_window
    )
    pool = _coverage_population(
        ComparisonPool, records, df_r, pool_ids, reporting_overrides={victim: victim_reporting}
    )
    method_settings = RS_Settings(n_meters_total=len(pool_ids), n_meters_per_treatment=None)
    selection = select_comparison_group(
        treatment, pool, method="random_sampling", method_settings=method_settings, basis="observed"
    )

    assert treatment.reporting_window == narrow_window

    analysis = MeterAnalysis(selection, treatment, pool, treatment.ids[0])
    analysis.correct()

    assert analysis.correction.exclusions[
        analysis.correction.exclusions["origin"] == "reporting_coverage"
    ].empty


# ── staged == run ────────────────────────────────────────────────────────────


def test_staged_pipeline_equals_run(billing_env):
    """Calling the stages individually yields the same savings as ``run()``."""
    staged = _analysis(billing_env)
    staged.correct().savings()

    combined = _analysis(billing_env).run()

    pd.testing.assert_frame_equal(staged.correction.corrected, combined.correction.corrected)
    pd.testing.assert_frame_equal(
        staged.savings_result.savings, combined.savings_result.savings
    )


def test_stages_return_self_for_chaining(billing_env):
    analysis = _analysis(billing_env)

    assert analysis.correct() is analysis
    assert analysis.savings() is analysis
    assert analysis.correction is not None
    assert analysis.savings_result is not None


def test_analysis_corrects_only_its_own_meter(billing_env):
    """The per-meter API corrects exactly the treatment meter it was given, not
    the whole treatment population."""
    treatment_id = billing_env["treatment"].ids[1]
    analysis = _analysis(billing_env, treatment_id=treatment_id).correct()

    assert set(analysis.correction.corrected["id"].unique()) == {treatment_id}


# ── treatment observed_unc through the corrected column ──────────────────────


def _treatment_with_observed_unc(comstock, billing_env, unc):
    """Rebuild the billing treatment reusing its fitted models, tagging every
    meter with a scalar observed uncertainty."""
    meters = _rewrap_meters(comstock, billing_env["treatment"])

    for entry in meters.values():
        entry["observed_unc"] = unc

    treatment = TreatmentGroup.from_fit_models(meters)

    return treatment


def test_population_observed_unc_reaches_savings_via_corrected_column(billing_env, comstock):
    """A treatment meter carrying observed_unc is threaded onto the correction's
    ``observed_unc`` column and, with no explicit kwarg, combined into
    savings_unc as the quadrature of the corrected band with that column."""
    treatment = _treatment_with_observed_unc(comstock, billing_env, unc=4.0)
    analysis = _analysis(billing_env, treatment=treatment)
    analysis.correct().savings()

    corrected = analysis.correction.corrected

    assert "observed_unc" in corrected.columns
    assert (corrected["observed_unc"].to_numpy(dtype=float) > 0).any()

    expected = corrected[["id", "datetime", "corrected_unc", "observed_unc"]].copy()
    expected["expected_unc"] = np.sqrt(
        expected["corrected_unc"].to_numpy(dtype=float) ** 2
        + expected["observed_unc"].to_numpy(dtype=float) ** 2
    )

    savings_rows = analysis.savings_result.savings
    merged = savings_rows.merge(expected, left_on=["id", "period"], right_on=["id", "datetime"])

    assert len(merged) == len(savings_rows)
    assert np.isfinite(merged["expected_unc"].to_numpy()).any()
    np.testing.assert_allclose(
        merged["savings_unc"].to_numpy(dtype=float),
        merged["expected_unc"].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_unset_observed_unc_leaves_corrected_column_zero(billing_env):
    """A treatment carrying no observed_unc still gets a zero-filled
    ``observed_unc`` column, so savings_unc equals the corrected band alone."""
    analysis = _analysis(billing_env).correct()

    corrected = analysis.correction.corrected

    assert "observed_unc" in corrected.columns
    np.testing.assert_array_equal(
        corrected["observed_unc"].to_numpy(dtype=float),
        np.zeros(len(corrected)),
    )


def test_explicit_observed_unc_kwarg_overrides_corrected_column(billing_env, comstock):
    """An explicit observed_unc passed to savings() wins over the population's
    threaded column (the 4.0 treatment value is ignored in favor of the 3.0
    kwarg)."""
    treatment = _treatment_with_observed_unc(comstock, billing_env, unc=4.0)
    analysis = _analysis(billing_env, treatment=treatment)
    analysis.correct()

    explicit = {mid: 3.0 for mid in treatment.ids}
    analysis.savings(observed_unc=explicit)

    corrected = analysis.correction.corrected
    expected = corrected[["id", "datetime", "corrected_unc"]].copy()
    expected["expected_unc"] = np.sqrt(
        expected["corrected_unc"].to_numpy(dtype=float) ** 2 + 3.0**2
    )

    savings_rows = analysis.savings_result.savings
    merged = savings_rows.merge(expected, left_on=["id", "period"], right_on=["id", "datetime"])

    assert np.isfinite(merged["expected_unc"].to_numpy()).any()
    np.testing.assert_allclose(
        merged["savings_unc"].to_numpy(dtype=float),
        merged["expected_unc"].to_numpy(dtype=float),
        equal_nan=True,
    )


# ── stage-order and whole-meter guards ───────────────────────────────────────


def test_savings_before_correct_raises(billing_env):
    analysis = _analysis(billing_env)

    with pytest.raises(RuntimeError, match="run correct"):
        analysis.savings()


def test_correct_without_reporting_data_raises_meter_error(comstock, model_bank):
    """A population built from baseline only can still be selected on, but
    correcting a meter with no reporting data raises MeterCorrectionError."""
    df_b, df_r = comstock.frames("billing")
    ids = sorted(df_b.index.get_level_values("id").unique())

    treatment = TreatmentGroup.from_fit_models(
        comstock.meters(model_bank, "billing", ids[:6], reporting=False)
    )
    pool = ComparisonPool.from_fit_models(
        comstock.meters(model_bank, "billing", ids[6:46], reporting=False)
    )
    selection = select_comparison_group(treatment, pool)
    analysis = MeterAnalysis(selection, treatment, pool, treatment.ids[0])

    with pytest.raises(exclusions.MeterCorrectionError):
        analysis.correct()


def test_meter_absent_from_selection_raises_with_ledger_row(billing_env):
    """A treatment meter the selection never saw cannot be corrected, and the
    error carries the guard row explaining it."""
    analysis = MeterAnalysis(
        billing_env["selection"], billing_env["treatment"], billing_env["pool"], "not-a-meter"
    )

    with pytest.raises(exclusions.MeterCorrectionError) as caught:
        analysis.correct()

    rows = caught.value.exclusions

    assert list(rows["id"]) == ["not-a-meter"]
    assert list(rows["stage"]) == ["correction"]


# ── meter_log ────────────────────────────────────────────────────────────────


def _manual_selection(treatment, pool, nan_weight_id):
    """A single-cluster clustering selection over the pool with one treatment
    meter's weights all-NaN, carrying the matching selection-stage ledger row."""
    clusters_flat = pd.DataFrame(
        {
            "id": [str(p) for p in pool.ids],
            "cluster": 0,
            "weight": 1.0,
            "treatment": pd.NA,
        }
    )
    clusters = _finalize_clusters(clusters_flat, SelectionMethod.CG_CLUSTERING)

    weights = {t: {"pct_cluster_0": 1.0} for t in treatment.ids}
    weights[nan_weight_id] = {"pct_cluster_0": np.nan}
    weights_frame = pd.DataFrame(weights).T
    weights_frame.index.name = "id"
    treatment_weights = _normalize_treatment_weights(weights_frame)

    ledger = exclusions.append(
        exclusions.empty_ledger(),
        [nan_weight_id],
        "selection",
        "treatment_fit",
        "treatment loadshape invalid (all-NaN cluster weights)",
    )
    selection = ComparisonGroupSelection(
        clusters=clusters,
        treatment_weights=treatment_weights,
        method="cg_clustering",
        basis="error",
        method_settings={},
        data_settings=None,
        exclusions=ledger,
        treatment_ids=treatment.ids,
        pool_ids=pool.ids,
        tz=treatment.tz,
    )

    return selection


def test_meter_log_empty_case_returns_typed_empty_frame(billing_env):
    analysis = _analysis(billing_env)

    log = analysis.meter_log()

    assert list(log.columns) == ["id", "stage", "origin", "reason", "detail"]
    assert log.empty


def test_meter_log_carries_guard_rows_after_a_raising_correction(billing_env):
    """After correct() raises, the log still explains the failure: the selection
    row for the all-NaN-weights meter and the correction guard row it raised on,
    in stage order."""
    treatment = billing_env["treatment"]
    pool = billing_env["pool"]
    nan_id = treatment.ids[0]
    selection = _manual_selection(treatment, pool, nan_id)
    analysis = MeterAnalysis(selection, treatment, pool, nan_id)

    with pytest.raises(exclusions.MeterCorrectionError):
        analysis.correct()

    log = analysis.meter_log()

    assert list(log.columns) == ["id", "stage", "origin", "reason", "detail"]
    assert list(log["stage"]) == ["selection", "correction"]
    assert list(log["origin"]) == ["treatment_fit", "correction_guard"]
    assert set(log["id"]) == {nan_id}


def test_failed_correct_clears_the_previous_results(billing_env):
    """A correct() that raises leaves no stale result behind: the earlier run's
    correction and savings are cleared, so savings() cannot silently recompute
    from a superseded correction."""
    treatment = billing_env["treatment"]
    pool = billing_env["pool"]
    tid = treatment.ids[0]
    analysis = MeterAnalysis(billing_env["selection"], treatment, pool, tid).run()
    assert analysis.correction is not None
    assert analysis.savings_result is not None

    analysis.selection = _manual_selection(treatment, pool, tid)

    with pytest.raises(exclusions.MeterCorrectionError):
        analysis.correct()

    assert analysis.correction is None
    assert analysis.savings_result is None

    with pytest.raises(RuntimeError, match="run correct"):
        analysis.savings()


def test_meter_log_excludes_pool_meters_outside_this_comparison_group(billing_env, comstock):
    """Population-stage rows for pool meters trimmed before selection belong to
    no comparison group, so they stay out of this meter's log even though they
    are on the pool population's own ledger."""
    treatment = billing_env["treatment"]
    pool_meters = _rewrap_meters(comstock, billing_env["pool"])
    pool = ComparisonPool.from_fit_models(pool_meters, max_pool_size=len(pool_meters) - 2, seed=1)

    trimmed = pool.exclusions

    assert len(trimmed) == 2
    assert (trimmed["origin"] == "pool_trim").all()

    selection = _manual_selection(treatment, pool, treatment.ids[0])
    analysis = MeterAnalysis(selection, treatment, pool, treatment.ids[1])
    log = analysis.meter_log()

    assert set(log["id"]).isdisjoint(set(trimmed["id"]))
