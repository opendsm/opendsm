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
import warnings

import numpy as np
import pandas as pd
import pytest

from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const
from opendsm.comparison_groups.population import ComparisonPool, TreatmentGroup
from opendsm.comparison_groups.selection import (
    ComparisonGroupSelection,
    SelectionMethod,
    _build_data,
    table_fingerprint,
    _normalize_clusters,
    _stage_exclusions,
    window_coverage,
    select_comparison_group,
)
from opendsm.comparison_groups.cg_clustering.settings import CG_Clustering_Settings
from opendsm.comparison_groups.individual_meter_matching.settings import Settings as IMM_Settings
from opendsm.comparison_groups.random_sampling.settings import Settings as RS_Settings
from opendsm.comparison_groups.stratified_sampling.create_comparison_groups import Stratified_Sampling
from opendsm.comparison_groups.stratified_sampling.settings import (
    DistanceStratifiedSamplingSettings,
    DSS_StratificationColumnSettings,
)
from opendsm.eemeter.models import DailyModel
from opendsm.eemeter.models.daily.data import DailyBaselineData



# Individual meter matching with duplicate matches allowed and more matches than
# distinct pool meters forces at least one pool meter to appear for multiple
# treatments, so the normalized clusters carry a genuine duplicate id index.
_IMM_DUPLICATE_SETTINGS = dict(allow_duplicate_matches=True, n_matches_per_treatment=8)


@pytest.fixture(scope="module")
def billing_meters(comstock, model_bank):
    """Pinned BillingModels for real ComStock monthly meters: 6 treatment and 40
    pool, baseline only. The same pinned fits serve the correction tests, so
    the selections are reproducible across machines."""
    ids = comstock.ids("billing")
    bundle = {
        "treatment": comstock.meters(model_bank, "billing", ids[:6], reporting=False),
        "pool": comstock.meters(model_bank, "billing", ids[6:46], reporting=False),
    }

    return bundle


@pytest.fixture(scope="module")
def treatment(billing_meters):
    return TreatmentGroup.from_fit_models(billing_meters["treatment"])


@pytest.fixture(scope="module")
def pool(billing_meters):
    return ComparisonPool.from_fit_models(billing_meters["pool"])


def _pool_with_features(billing_meters):
    """A pool carrying synthetic stratification features on the real meter ids.
    No real dataset pairs load shapes with stratification features, so the
    features are constructed (repo precedent); the load shapes remain real."""
    pool_meters = billing_meters["pool"]
    ids = list(pool_meters)
    features = pd.DataFrame(
        {
            "summer_usage": np.linspace(3000, 6000, len(ids)),
            "winter_usage": np.linspace(6000, 3000, len(ids)),
        },
        index=ids,
    )

    return ComparisonPool.from_fit_models(pool_meters, features=features)


def _select(treatment, pool, method, **kwargs):
    return select_comparison_group(treatment, pool, method=method, **kwargs)


# -- all four methods --------------------------------------------------------


def test_clustering_selection_has_expected_fields(treatment, pool):
    selection = _select(treatment, pool, "cg_clustering")

    assert selection.method == SelectionMethod.CG_CLUSTERING
    assert selection.basis == "error"
    assert selection.tz == treatment.tz
    assert set(selection.treatment_ids) == set(treatment.ids)
    assert set(selection.pool_ids) == set(pool.ids)
    assert len(selection.fingerprint) == 64


def test_random_selection_default_basis_is_modeled(treatment, pool):
    selection = _select(treatment, pool, "random_sampling")

    assert selection.method == SelectionMethod.RANDOM_SAMPLING
    assert selection.basis == "modeled"


# -- normalized clusters schema per method -----------------------------------


def test_clustering_clusters_schema(treatment, pool):
    selection = _select(treatment, pool, "cg_clustering")
    clusters = selection.clusters

    assert list(clusters.columns) == ["cluster", "weight", "treatment"]
    assert clusters.index.name == "id"
    assert pd.api.types.is_string_dtype(clusters.index)
    assert clusters["cluster"].dtype == np.int64
    assert clusters["weight"].dtype == np.float64
    assert str(clusters["treatment"].dtype) == "string"
    assert (clusters["weight"] == 1.0).all()
    assert clusters["treatment"].isna().all()
    assert not clusters.index.duplicated().any()
    assert sorted(clusters.index) == sorted(str(x) for x in pool.ids)


def test_imm_clusters_schema_has_duplicate_index(treatment, pool):
    selection = _select(
        treatment, pool, "individual_meter_matching", method_settings=IMM_Settings(**_IMM_DUPLICATE_SETTINGS)
    )
    clusters = selection.clusters

    assert list(clusters.columns) == ["cluster", "weight", "treatment", "distance", "duplicated"]
    assert clusters["cluster"].dtype == np.int64
    assert clusters["distance"].dtype == np.float64
    assert clusters["duplicated"].dtype == bool
    assert str(clusters["treatment"].dtype) == "string"
    # more match slots than distinct pool meters => at least one duplicate id
    assert clusters.index.duplicated().any()
    assert set(clusters["treatment"]) <= set(selection.treatment_ids)


def test_random_clusters_schema(treatment, pool):
    selection = _select(treatment, pool, "random_sampling", method_settings=RS_Settings())
    clusters = selection.clusters

    assert list(clusters.columns) == ["cluster", "weight", "treatment"]
    assert (clusters["cluster"] == 0).all()
    assert (clusters["weight"] == 1.0).all()
    assert clusters["treatment"].isna().all()


def test_stratified_clusters_schema(stratified_feature_loadshape_data):
    """The stratified DSS path runs on the synthetic feature+loadshape fixture
    (no real dataset pairs both); the adapter must map its output to the
    cluster-0 / weight-1 normalized schema."""
    treatment_data, pool_data = stratified_feature_loadshape_data
    columns = [
        DSS_StratificationColumnSettings(column_name="summer_usage", min_value_allowed=0, max_value_allowed=10000),
        DSS_StratificationColumnSettings(column_name="winter_usage", min_value_allowed=0, max_value_allowed=10000),
    ]
    settings = DistanceStratifiedSamplingSettings(
        seed=42,
        n_samples_approx=100,
        relax_n_samples_approx_constraint=True,
        min_n_sampled_to_n_treatment_ratio=0,
        min_n_bins=1,
        max_n_bins=3,
        stratification_column=columns,
    )
    clusters_raw, _ = Stratified_Sampling(settings).get_comparison_group(treatment_data, pool_data)

    clusters = _normalize_clusters(SelectionMethod.STRATIFIED_SAMPLING, clusters_raw)

    assert list(clusters.columns) == ["cluster", "weight", "treatment"]
    assert (clusters["cluster"] == 0).all()
    assert (clusters["weight"] == 1.0).all()
    assert clusters["treatment"].isna().all()
    assert not clusters.empty


# -- pre-mutation settings capture + determinism -----------------------------


def test_clustering_settings_captured_before_mutation(treatment, pool):
    """CG_Clustering.get_labels rebuilds the n_cluster bounds on self.settings;
    the artifact must record the caller's pre-mutation settings."""
    selection = _select(treatment, pool, "cg_clustering", method_settings=CG_Clustering_Settings())
    default_bounds = CG_Clustering_Settings().model_dump(mode="json")["bisecting_kmeans"]["n_cluster"]

    assert selection.method_settings["bisecting_kmeans"]["n_cluster"] == default_bounds


def test_clustering_rerun_from_deserialized_settings_is_deterministic(treatment, pool):
    selection = _select(treatment, pool, "cg_clustering")
    rebuilt_settings = CG_Clustering_Settings(**selection.method_settings)

    rerun = _select(treatment, pool, "cg_clustering", method_settings=rebuilt_settings)

    pd.testing.assert_frame_equal(selection.clusters, rerun.clusters)
    pd.testing.assert_frame_equal(selection.treatment_weights, rerun.treatment_weights)
    assert selection.fingerprint == rerun.fingerprint


# -- stratified feature requirements -----------------------------------------


def test_stratified_without_features_raises(treatment, pool):
    with pytest.raises(ValueError, match="requires stratification features"):
        _select(treatment, pool, "stratified_sampling")


def test_stratified_build_data_carries_loadshapes_and_features(billing_meters):
    pool = _pool_with_features(billing_meters)

    data = _build_data(pool, SelectionMethod.STRATIFIED_SAMPLING, "modeled", None)

    assert isinstance(data, Data)
    assert data.loadshape is not None
    assert data.features is not None
    assert set(data.features.columns) == {"summer_usage", "winter_usage"}


# -- serialization roundtrip -------------------------------------------------


@pytest.mark.parametrize(
    "method, settings_kwargs",
    [
        ("cg_clustering", None),
        ("individual_meter_matching", _IMM_DUPLICATE_SETTINGS),
        ("random_sampling", {}),
    ],
)
def test_json_roundtrip_is_exact(treatment, pool, method, settings_kwargs):
    if method == "individual_meter_matching":
        settings = IMM_Settings(**settings_kwargs)
    elif method == "random_sampling":
        settings = RS_Settings(**settings_kwargs)
    else:
        settings = None

    selection = _select(treatment, pool, method, method_settings=settings)
    rebuilt = ComparisonGroupSelection.from_json(selection.to_json())

    pd.testing.assert_frame_equal(selection.clusters, rebuilt.clusters)
    pd.testing.assert_frame_equal(selection.treatment_weights, rebuilt.treatment_weights)
    pd.testing.assert_frame_equal(selection.exclusions, rebuilt.exclusions)
    assert rebuilt.method == selection.method
    assert rebuilt.basis == selection.basis
    assert rebuilt.tz == selection.tz
    assert rebuilt.treatment_ids == selection.treatment_ids
    assert rebuilt.pool_ids == selection.pool_ids
    assert rebuilt.method_settings == selection.method_settings
    assert rebuilt.data_settings == selection.data_settings


def test_fingerprint_is_stable_across_roundtrip(treatment, pool):
    selection = _select(
        treatment, pool, "individual_meter_matching", method_settings=IMM_Settings(**_IMM_DUPLICATE_SETTINGS)
    )
    rebuilt = ComparisonGroupSelection.from_json(selection.to_json())

    assert rebuilt.fingerprint == selection.fingerprint
    # a corrupted table must not deserialize under the stored fingerprint
    payload = selection.to_json().replace(selection.fingerprint, "0" * 64)

    with pytest.raises(ValueError, match="fingerprint"):
        ComparisonGroupSelection.from_json(payload)


# -- timezone guard ----------------------------------------------------------


def test_timezone_mismatch_raises(treatment, pool):
    mismatched = copy.copy(pool)
    mismatched.tz = "America/New_York"

    with pytest.raises(ValueError, match="timezone"):
        _select(treatment, mismatched, "cg_clustering")


# -- cross-population granularity fineness -----------------------------------


@pytest.fixture(scope="module")
def daily_and_billing_meters(comstock, model_bank):
    """Two real ComStock daily meters and two billing meters (same underlying
    dataset, same timezone) on pinned models, for exercising the
    selection-stage granularity fineness rule."""
    daily = comstock.meters(model_bank, "daily", comstock.ids("daily")[:2], reporting=False)
    billing = comstock.meters(model_bank, "billing", comstock.ids("billing")[:2], reporting=False)

    return {"daily": daily, "billing": billing}


def test_coarser_pool_raises_fineness_error(daily_and_billing_meters):
    """A daily (finer) treatment against a billing (coarser) pool violates
    fineness(pool) >= fineness(treatment) and must raise, naming both
    granularities."""
    treatment = TreatmentGroup.from_fit_models(daily_and_billing_meters["daily"])
    pool = ComparisonPool.from_fit_models(daily_and_billing_meters["billing"])

    with pytest.raises(ValueError, match="daily.*billing|billing.*daily"):
        select_comparison_group(treatment, pool, method="random_sampling")


def test_finer_pool_builds_both_loadshapes_at_treatment_time_period(daily_and_billing_meters):
    """A billing (coarser) treatment against a daily (finer) pool is allowed;
    both loadshapes build at the treatment's MONTH time period, so they come
    out the same length (the v1 bug let the daily pool default to its own,
    finer, time period)."""
    treatment = TreatmentGroup.from_fit_models(daily_and_billing_meters["billing"])
    pool = ComparisonPool.from_fit_models(daily_and_billing_meters["daily"])
    settings = RS_Settings(n_meters_total=len(pool.ids), n_meters_per_treatment=None)

    selection = select_comparison_group(
        treatment, pool, method="random_sampling", method_settings=settings, basis="modeled"
    )
    expected_settings = Data_Settings(time_period=_const.TimePeriod.MONTH).model_dump(mode="json")
    assert selection.data_settings == expected_settings  # the defaulted settings are stored

    treatment_loadshape = _build_data(
        treatment,
        SelectionMethod.RANDOM_SAMPLING,
        "modeled",
        Data_Settings(time_period=_const.TimePeriod.MONTH),
    ).loadshape
    pool_loadshape = _build_data(
        pool,
        SelectionMethod.RANDOM_SAMPLING,
        "modeled",
        Data_Settings(time_period=_const.TimePeriod.MONTH),
    ).loadshape

    assert treatment_loadshape.shape[1] == pool_loadshape.shape[1] == 12


# -- exclusions ledger -------------------------------------------------------


def test_data_validation_exclusion_recorded_and_pool_trim_not_duplicated(
    billing_meters, treatment
):
    """A meter that ``Data`` drops during validation is recorded at the
    selection stage with the verbatim ``Data`` reason string; pool-trim rows
    stay on the pool population's ledger and are not duplicated here."""
    pool = ComparisonPool.from_fit_models(billing_meters["pool"], max_pool_size=35, seed=0)
    trimmed_ids = set(pool.exclusions["id"])

    # NaN one surviving pool meter's cached modeled baseline so its monthly
    # loadshape is entirely missing and Data drops it during validation
    victim = pool.ids[0]
    pred = pool._ensure_pred("baseline")
    pred[victim].loc[:, "modeled"] = np.nan

    selection = _select(treatment, pool, "random_sampling", method_settings=RS_Settings())
    ledger = selection.exclusions

    row = ledger.set_index("id").loc[victim]
    assert row["stage"] == "selection"
    assert row["origin"] == "data_validation"
    assert row["reason"] == "missing minimum number of values in loadshape_df"
    assert victim not in set(selection.clusters.index.astype(str))

    assert (ledger["origin"] != "pool_trim").all()
    assert set(ledger["id"]).isdisjoint(trimmed_ids)
    assert len(pool.exclusions) == 5

    # the fingerprint covers clusters and treatment weights only, never the ledger
    fingerprinted = {k: v for k, v in selection.tables.items() if k != "exclusions"}
    assert set(fingerprinted) == {"clusters", "treatment_weights"}
    assert selection.fingerprint == table_fingerprint(fingerprinted)


def test_stage_exclusions_records_nan_weight_treatments_as_treatment_fit():
    class _DataStub:
        excluded_ids = pd.DataFrame(columns=["id", "reason"])

    weights = pd.DataFrame(
        {"pct_cluster_0": [1.0, np.nan], "pct_cluster_1": [0.0, np.nan]},
        index=pd.Index(["t-ok", "t-nan"], name="id"),
    )

    ledger = _stage_exclusions(_DataStub(), _DataStub(), weights)

    assert list(ledger["id"]) == ["t-nan"]
    row = ledger.iloc[0]
    assert row["stage"] == "selection"
    assert row["origin"] == "treatment_fit"
    assert row["reason"] == "treatment loadshape invalid (all-NaN cluster weights)"


# -- snapshot ----------------------------------------------------------------


def test_selection_tables_summary_snapshot(treatment, pool, snapshot):
    """Permutation-invariant summary of the clustering and IMM selections on
    real ComStock meters, pinned across runs."""
    clustering = _select(treatment, pool, "cg_clustering")
    imm = _select(
        treatment, pool, "individual_meter_matching", method_settings=IMM_Settings(**_IMM_DUPLICATE_SETTINGS)
    )

    cl = clustering.clusters
    sizes = cl[cl["cluster"] >= 0]["cluster"].value_counts().sort_values()
    imm_clusters = imm.clusters

    summary = {
        "clustering": {
            "n_clusters": int(cl.loc[cl["cluster"] >= 0, "cluster"].nunique()),
            "sorted_cluster_sizes": sorted(sizes.tolist()),
            "dominant_weight_per_treatment": sorted(
                np.round(clustering.treatment_weights.to_numpy().max(axis=1), 4).tolist()
            ),
            "n_excluded": int(len(clustering.exclusions)),
        },
        "imm": {
            "n_rows": int(len(imm_clusters)),
            "n_unique_cg": int(imm_clusters.index.nunique()),
            "n_duplicated_rows": int(imm_clusters["duplicated"].sum()),
            "sorted_distances": sorted(np.round(imm_clusters["distance"].to_numpy(), 4).tolist()),
        },
    }

    assert summary == snapshot


# -- baseline-coverage DQ (selection stage) ----------------------------------


_COVERAGE_TZ = "America/New_York"


def _daily_baseline(seed=0, hole=None, span=365, start="2020-01-01"):
    """A constructed daily baseline frame with a smooth weather-driven load.
    ``hole`` is a ``(start, stop)`` positional slice of observed rows set to NaN;
    ``span`` and ``start`` shift the meter's coverage relative to a group
    window."""
    index = pd.date_range(start, periods=span, freq="D", tz=_COVERAGE_TZ)
    rng = np.random.default_rng(seed)
    temperature = 50 + 25 * np.sin(np.arange(span) / 365 * 2 * np.pi) + rng.normal(0, 2, span)
    observed = 20 + 0.8 * np.abs(temperature - 62) + rng.normal(0, 1, span)
    frame = pd.DataFrame({"observed": observed, "temperature": temperature}, index=index)

    if hole is not None:
        lo, hi = hole
        frame.iloc[lo:hi, frame.columns.get_loc("observed")] = np.nan

    baseline = frame.reset_index().rename(columns={"index": "datetime"})

    return baseline


def _daily_baseline_data(**kwargs):
    """``_daily_baseline`` as a data object, for the coverage helper that reads
    one directly rather than through a population."""
    return DailyBaselineData(_daily_baseline(**kwargs), is_electricity_data=True)


def _daily_population(cls, specs, model, **kwargs):
    """Build a daily population from ``{id: baseline_df}``, reusing one fitted
    model (the coverage DQ reads only baseline data). The disqualified-meter
    warning is expected for gross-hole meters and suppressed here."""
    meters = {mid: {"model": model, "baseline_df": df} for mid, df in specs.items()}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Population includes disqualified")
        population = cls.from_fit_models(meters, granularity="daily", **kwargs)

    return population


@pytest.fixture(scope="module")
def daily_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DailyModel().fit(
            _daily_baseline(seed=0), is_electricity_data=True, ignore_disqualification=True
        )

    return model


def _random_selection(treatment, pool):
    settings = RS_Settings(n_meters_total=len(pool.ids), n_meters_per_treatment=None)
    selection = select_comparison_group(
        treatment, pool, method="random_sampling", method_settings=settings, basis="observed"
    )

    return selection


def test_window_coverage_is_finite_fraction_over_window_length():
    baseline = _daily_baseline_data(hole=(100, 160))  # 60 of 365 observed days NaN
    window = (baseline.df.index.min(), baseline.df.index.max())

    coverage = window_coverage(baseline, window, "D")

    assert len(pd.date_range(window[0], window[1], freq="D")) == 365
    assert coverage == pytest.approx(305 / 365)


def test_window_coverage_full_clean_meter_is_one():
    baseline = _daily_baseline_data()
    window = (baseline.df.index.min(), baseline.df.index.max())

    coverage = window_coverage(baseline, window, "D")

    assert coverage == pytest.approx(1.0)


def test_window_coverage_penalizes_a_span_short_of_the_window():
    baseline = _daily_baseline_data(span=300, start="2020-02-01")  # 300 finite days inside 2020
    window = (
        pd.Timestamp("2020-01-01", tz=_COVERAGE_TZ),
        pd.Timestamp("2020-12-30", tz=_COVERAGE_TZ),
    )
    n_days = len(pd.date_range(window[0], window[1], freq="D"))

    coverage = window_coverage(baseline, window, "D")

    assert n_days == 365
    assert coverage == pytest.approx(300 / 365)


def test_baseline_coverage_prunes_gross_hole_pool_meter(daily_model):
    pool_specs = {
        "p1": _daily_baseline(seed=1),
        "p2": _daily_baseline(seed=2),
        "p3": _daily_baseline(seed=3),
        "p-hole": _daily_baseline(seed=9, hole=(100, 200)),  # 265/365 = 0.7260
    }
    treatment_specs = {"t0": _daily_baseline(seed=100), "t1": _daily_baseline(seed=101)}
    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)

    selection = _random_selection(treatment, pool)

    assert "p-hole" not in selection.clusters.index
    assert set(selection.clusters.index) == {"p1", "p2", "p3"}

    coverage_rows = selection.exclusions[selection.exclusions["origin"] == "baseline_coverage"]
    assert list(coverage_rows["id"]) == ["p-hole"]
    row = coverage_rows.iloc[0]
    assert row["stage"] == "selection"
    assert row["reason"] == "baseline coverage below minimum over the group window"
    assert "baseline window coverage 0.7260 < 0.9" in row["detail"]
    assert "too_many_days_with_missing_observed_data" in row["detail"]


def test_baseline_coverage_prunes_treatment_meter(daily_model):
    treatment_specs = {
        "t-ok": _daily_baseline(seed=1),
        "t-hole": _daily_baseline(seed=9, hole=(100, 200)),
    }
    pool_specs = {f"p{i}": _daily_baseline(seed=20 + i) for i in range(4)}
    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)

    selection = _random_selection(treatment, pool)

    assert list(selection.treatment_weights.index) == ["t-ok"]
    assert "t-hole" not in selection.treatment_weights.index

    coverage_rows = selection.exclusions[selection.exclusions["origin"] == "baseline_coverage"]
    assert list(coverage_rows["id"]) == ["t-hole"]
    assert "baseline window coverage 0.7260 < 0.9" in coverage_rows.iloc[0]["detail"]


def test_offset_meter_within_coverage_survives(daily_model):
    pool_specs = {
        "p-offset": _daily_baseline(seed=5, start="2020-01-21"),  # 345/365 = 0.945
        "p1": _daily_baseline(seed=1),
        "p2": _daily_baseline(seed=2),
    }
    treatment_specs = {"t0": _daily_baseline(seed=100), "t1": _daily_baseline(seed=101)}
    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)

    selection = _random_selection(treatment, pool)

    assert "p-offset" in selection.clusters.index
    assert selection.exclusions[selection.exclusions["origin"] == "baseline_coverage"].empty


def test_explicit_baseline_window_changes_coverage_verdict(daily_model):
    pool_specs = {
        "p-partial": _daily_baseline(seed=7, hole=(250, 330)),  # 285/365 = 0.7808
        "p1": _daily_baseline(seed=1),
        "p2": _daily_baseline(seed=2),
    }
    treatment_specs = {"t0": _daily_baseline(seed=100), "t1": _daily_baseline(seed=101)}
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)

    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    inferred = _random_selection(treatment, pool)

    assert "p-partial" not in inferred.clusters.index

    # A narrower explicit window falls entirely before the hole, so the same
    # meter now covers all of it and survives.
    narrow = (
        pd.Timestamp("2020-01-01", tz=_COVERAGE_TZ),
        pd.Timestamp("2020-08-01", tz=_COVERAGE_TZ),
    )
    treatment_narrow = _daily_population(
        TreatmentGroup, treatment_specs, daily_model, baseline_window=narrow
    )
    narrowed = _random_selection(treatment_narrow, pool)

    assert "p-partial" in narrowed.clusters.index
    assert narrowed.exclusions[narrowed.exclusions["origin"] == "baseline_coverage"].empty


def test_json_and_ndjson_roundtrip_preserve_explicit_window_coverage_verdict(daily_model):
    """A narrow explicit ``baseline_window`` that falls entirely before a pool
    meter's data hole makes that meter survive selection coverage. If a
    JSON/NDJSON round trip recomputed the window from the rebuilt treatment
    population's own attached-data span (the full year, hole included) instead
    of restoring the explicit override, the same meter would flip back to
    pruned. It doesn't, through either serialization path."""
    pool_specs = {
        "p-partial": _daily_baseline(seed=7, hole=(250, 330)),  # 285/365 = 0.7808
        "p1": _daily_baseline(seed=1),
        "p2": _daily_baseline(seed=2),
    }
    treatment_specs = {"t0": _daily_baseline(seed=100), "t1": _daily_baseline(seed=101)}
    narrow = (
        pd.Timestamp("2020-01-01", tz=_COVERAGE_TZ),
        pd.Timestamp("2020-08-01", tz=_COVERAGE_TZ),
    )
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)
    treatment = _daily_population(
        TreatmentGroup, treatment_specs, daily_model, baseline_window=narrow
    )

    rebuilt_json = TreatmentGroup.from_json(treatment.to_json(), baseline=treatment_specs)
    rebuilt_ndjson = TreatmentGroup.from_ndjson(treatment.to_ndjson(), baseline=treatment_specs)

    assert rebuilt_json.baseline_window == narrow
    assert rebuilt_ndjson.baseline_window == narrow

    for rebuilt in (rebuilt_json, rebuilt_ndjson):
        selection = _random_selection(rebuilt, pool)

        assert "p-partial" in selection.clusters.index
        assert selection.exclusions[selection.exclusions["origin"] == "baseline_coverage"].empty
