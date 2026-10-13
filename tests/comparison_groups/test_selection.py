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
    _ALGORITHM,
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
    StratificationColumnSettings,
    StratifiedSamplingSettings,
)
from opendsm.eemeter.models import (
    DailyBaselineData,
    DailyModel,
)



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


def _synthetic_features(ids):
    """Synthetic stratification features on real meter ids. No real dataset
    pairs load shapes with stratification features, so the features are
    constructed; the load shapes remain real."""
    features = pd.DataFrame(
        {
            "summer_usage": np.linspace(3000, 6000, len(ids)),
            "winter_usage": np.linspace(6000, 3000, len(ids)),
        },
        index=list(ids),
    )

    return features


def _pool_with_features(billing_meters):
    pool_meters = billing_meters["pool"]

    return ComparisonPool.from_fit_models(pool_meters, features=_synthetic_features(pool_meters))


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

    data = _build_data(pool, SelectionMethod.STRATIFIED_SAMPLING, "modeled", None, ())

    assert isinstance(data, Data)
    assert data.loadshape is not None
    assert data.features is not None
    assert set(data.features.columns) == {"summer_usage", "winter_usage"}


def test_stratified_build_data_disables_the_data_trim(billing_meters):
    """The stratified ``Data`` never trims the population: its cap is the
    population size and its seed the fixed loadshape seed."""
    pool = _pool_with_features(billing_meters)

    settings = _build_data(pool, SelectionMethod.STRATIFIED_SAMPLING, "modeled", None, ()).settings

    assert settings.max_pool_size == len(pool)
    assert settings.seed == 0


class _StubPopulation:
    role = "pool"

    def __init__(self, loadshape_data, features):
        self._loadshape_data = loadshape_data
        self.features = features

    def __len__(self):
        return len(self.features)

    def loadshape_data(self, basis, data_settings, exclude):
        return self._loadshape_data


def test_stratified_build_data_carries_forward_loadshape_exclusions():
    """A meter the loadshape ``Data`` drops (too few times, interpolation
    off) must still reach the stratified ``Data``'s exclusion ledger, and must
    not remain selectable through its surviving features row."""
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
        interpolate_missing=False,
    )
    loadshape_data = Data(
        time_series_df=_hourly_meters({"p0": 48, "p1": 20, "p2": 48}), settings=settings
    )
    features = pd.DataFrame({"summer_usage": [1.0, 2.0, 3.0]}, index=["p0", "p1", "p2"])
    population = _StubPopulation(loadshape_data, features)

    data = _build_data(population, SelectionMethod.STRATIFIED_SAMPLING, "observed", settings, ())

    assert "p1" in set(data.excluded_ids["id"])
    assert "p1" not in set(data.features.index.astype(str))


def test_stratified_build_data_drops_excluded_ids_from_features():
    """An id selection excludes loses its features row too, so stratified
    sampling cannot draw it."""
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )
    loadshape_data = Data(
        time_series_df=_hourly_meters({"p0": 48, "p2": 48}), settings=settings
    )
    features = pd.DataFrame({"summer_usage": [1.0, 2.0, 3.0]}, index=["p0", "p1", "p2"])
    population = _StubPopulation(loadshape_data, features)

    data = _build_data(population, SelectionMethod.STRATIFIED_SAMPLING, "observed", settings, ["p1"])

    assert sorted(data.features.index.astype(str)) == ["p0", "p2"]


def test_stratified_build_data_drops_features_without_a_meter():
    """A features row whose id the population does not hold (a pool-trimmed
    meter) is dropped, so stratified sampling cannot draw it."""
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )
    loadshape_data = Data(
        time_series_df=_hourly_meters({"p0": 48, "p2": 48}), settings=settings
    )
    features = pd.DataFrame({"summer_usage": [1.0, 2.0, 3.0]}, index=["p0", "trimmed", "p2"])
    population = _StubPopulation(loadshape_data, features)

    data = _build_data(population, SelectionMethod.STRATIFIED_SAMPLING, "observed", settings, ())

    assert sorted(data.features.index.astype(str)) == ["p0", "p2"]


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


# -- baseline guard ----------------------------------------------------------


def test_selection_without_baseline_raises_naming_every_meter(treatment, pool, billing_meters):
    """Selection refuses populations reloaded without baseline data, naming each
    meter that lacks it in either population."""
    kept = pool.ids[0]
    reloaded_treatment = TreatmentGroup.from_json(treatment.to_json())
    reloaded_pool = ComparisonPool.from_json(
        pool.to_json(), baseline={kept: billing_meters["pool"][kept]["baseline_data"]}
    )
    missing = {"treatment": treatment.ids, "pool": pool.ids[1:]}

    with pytest.raises(ValueError, match="Selection requires baseline data") as raised:
        _select(reloaded_treatment, reloaded_pool, "random_sampling")

    assert str(missing) in str(raised.value)


# -- cross-population model frequency fineness -----------------------------------


@pytest.fixture(scope="module")
def daily_and_billing_meters(comstock, model_bank):
    """Two real ComStock daily meters and two billing meters (same underlying
    dataset, same timezone, disjoint ids so selection's pool-overlap exclusion
    does not apply) on pinned models, for exercising the selection-stage model
    frequency fineness rule."""
    daily = comstock.meters(model_bank, "daily", comstock.ids("daily")[2:4], reporting=False)
    billing = comstock.meters(model_bank, "billing", comstock.ids("billing")[:2], reporting=False)

    return {"daily": daily, "billing": billing}


def test_coarser_pool_raises_fineness_error(daily_and_billing_meters):
    """A daily (finer) treatment against a billing (coarser) pool violates
    fineness(pool) >= fineness(treatment) and must raise, naming both
    model frequencies."""
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
        exclude=(),
    ).loadshape
    pool_loadshape = _build_data(
        pool,
        SelectionMethod.RANDOM_SAMPLING,
        "modeled",
        Data_Settings(time_period=_const.TimePeriod.MONTH),
        exclude=(),
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


def _hourly_meters(lengths):
    frames = [
        pd.DataFrame(
            {
                "id": meter_id,
                "datetime": pd.date_range("2024-01-01", periods=n_hours, freq="h"),
                "observed": np.arange(n_hours, dtype=float) + i,
            }
        )
        for i, (meter_id, n_hours) in enumerate(lengths.items())
    ]

    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize(
    "treatment_lengths, pool_lengths, incomplete_id",
    [
        ({"t0": 48, "t1": 48}, {"p0": 48, "p1": 20, "p2": 48, "p3": 48}, "p1"),
        ({"t0": 48, "t1": 20, "t2": 48}, {"p0": 48, "p1": 48, "p2": 48}, "t1"),
    ],
    ids=["pool_meter", "treatment_meter"],
)
def test_incomplete_meter_without_interpolation_is_ledgered_and_selection_proceeds(
    treatment_lengths, pool_lengths, incomplete_id
):
    """With interpolation off, one incomplete meter in either population is excluded,
    recorded once at the selection stage, and selection runs on the rest."""
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
        interpolate_missing=False,
    )
    treatment_data = Data(time_series_df=_hourly_meters(treatment_lengths), settings=settings)
    pool_data = Data(time_series_df=_hourly_meters(pool_lengths), settings=settings)
    algorithm = _ALGORITHM[SelectionMethod.RANDOM_SAMPLING](
        RS_Settings(n_meters_total=2, n_meters_per_treatment=None, seed=0)
    )

    clusters, treatment_weights = algorithm.get_comparison_group(treatment_data, pool_data)
    ledger = _stage_exclusions(treatment_data, pool_data, treatment_weights)

    assert ledger[["id", "stage", "origin", "reason"]].to_dict("records") == [
        {
            "id": incomplete_id,
            "stage": "selection",
            "origin": "data_validation",
            "reason": "Unique time counts per id don't have the minimum time counts required",
        }
    ]
    assert incomplete_id not in set(clusters.index.astype(str))
    assert incomplete_id not in set(treatment_weights.index.astype(str))
    assert len(set(clusters.index.astype(str))) == 2


# -- short-baseline exclusion (MONTH loadshapes) -----------------------------


def _with_short_baselines(meters, short_ids):
    """``meters`` with the baselines of ``short_ids`` cut to their first six
    daily rows, so each spans a single month."""
    shortened = {}

    for mid, entry in meters.items():
        entry = dict(entry)
        if mid in short_ids:
            baseline_data = copy.deepcopy(entry["baseline_data"])
            baseline_data._df = baseline_data._df.head(6)
            entry["baseline_data"] = baseline_data
        shortened[mid] = entry

    return shortened


@pytest.fixture
def short_baseline_env(billing_meters):
    """Billing populations, both carrying stratification features, in which one
    treatment meter and one pool meter have a single baseline month."""
    short_treatment = list(billing_meters["treatment"])[1]
    short_pool = list(billing_meters["pool"])[3]
    treatment_meters = _with_short_baselines(billing_meters["treatment"], {short_treatment})
    pool_meters = _with_short_baselines(billing_meters["pool"], {short_pool})
    env = {
        "treatment": TreatmentGroup.from_fit_models(
            treatment_meters, features=_synthetic_features(treatment_meters)
        ),
        "pool": ComparisonPool.from_fit_models(
            pool_meters, features=_synthetic_features(pool_meters)
        ),
        "short_treatment": short_treatment,
        "short_pool": short_pool,
    }

    return env


# Fixed bins: the distance-binning search needs more treatment meters than the five here.
_STRATIFIED_SETTINGS = StratifiedSamplingSettings(
    seed=42,
    n_samples_approx=100,
    relax_n_samples_approx_constraint=True,
    min_n_sampled_to_n_treatment_ratio=0,
    stratification_column=[
        StratificationColumnSettings(
            column_name="summer_usage", n_bins=2, min_value_allowed=0, max_value_allowed=10000
        ),
    ],
)


@pytest.mark.parametrize(
    "method, method_settings",
    [("random_sampling", RS_Settings()), ("stratified_sampling", _STRATIFIED_SETTINGS)],
    ids=["random_sampling", "stratified_sampling"],
)
def test_month_selection_leaves_populations_unchanged(short_baseline_env, method, method_settings):
    """Excluding short-baseline meters at selection never mutates either
    population: ids, ledgers and features are as they were."""
    populations = (short_baseline_env["treatment"], short_baseline_env["pool"])
    before = [(p.ids, p.exclusions.copy(), p.features.copy()) for p in populations]

    _select(*populations, method, method_settings=method_settings)

    for population, (ids, ledger, features) in zip(populations, before):
        assert population.ids == ids
        pd.testing.assert_frame_equal(population.exclusions, ledger)
        pd.testing.assert_frame_equal(population.features, features)


def test_short_baseline_meters_ledgered_at_selection_with_month_counts(short_baseline_env):
    """Each short-baseline meter is one selection-stage ``baseline_data`` row
    carrying its month count, and is absent from the selection's tables."""
    short_ids = [short_baseline_env["short_treatment"], short_baseline_env["short_pool"]]

    selection = _select(
        short_baseline_env["treatment"],
        short_baseline_env["pool"],
        "random_sampling",
        method_settings=RS_Settings(),
    )

    rows = selection.exclusions[selection.exclusions["origin"] == "baseline_data"]
    assert rows[["id", "stage", "detail"]].to_dict("records") == [
        {"id": str(mid), "stage": "selection", "detail": "baseline months: 1"} for mid in short_ids
    ]
    assert str(short_ids[0]) not in set(selection.treatment_weights.index)
    assert str(short_ids[1]) not in set(selection.clusters.index)


def test_stratified_sampling_never_draws_short_baseline_pool_meter(short_baseline_env):
    """With features on every pool meter, stratified sampling draws from the
    pool minus the short-baseline meter, never the meter itself."""
    pool = short_baseline_env["pool"]
    short_pool = str(short_baseline_env["short_pool"])

    selection = _select(
        short_baseline_env["treatment"],
        pool,
        "stratified_sampling",
        method_settings=_STRATIFIED_SETTINGS,
    )

    drawn = set(selection.clusters.index)
    assert drawn, "stratified sampling drew no pool meters"
    assert short_pool not in drawn


def test_daily_loadshape_skips_the_baseline_month_check(short_baseline_env):
    """Only a ``MONTH`` loadshape needs 12 baseline months: under any other time
    period no meter is ledgered as ``baseline_data``, and a single-month treatment
    meter is pruned by baseline coverage instead."""
    settings = Data_Settings(time_period=_const.TimePeriod.DAY_OF_WEEK)

    selection = _select(
        short_baseline_env["treatment"],
        short_baseline_env["pool"],
        "random_sampling",
        method_settings=RS_Settings(),
        data_settings=settings,
    )

    assert (selection.exclusions["origin"] != "baseline_data").all()
    short_treatment = str(short_baseline_env["short_treatment"])
    origin = selection.exclusions.set_index("id").loc[short_treatment, "origin"]
    assert origin == "baseline_coverage", f"{short_treatment} ledgered as {origin}"


# -- treatment/pool overlap --------------------------------------------------


@pytest.fixture(scope="module")
def overlap_env(billing_meters):
    """Billing populations with features in which the first two treatment
    meters also sit at the end of the pool; the pool copy of the second has a
    single baseline month. ``plain_pool`` is the same pool without them."""
    treatment_meters = billing_meters["treatment"]
    plain_meters = billing_meters["pool"]
    overlap = list(treatment_meters)[:2]
    overlap_meters = _with_short_baselines(
        {mid: treatment_meters[mid] for mid in overlap}, {overlap[1]}
    )
    plain_features = _synthetic_features(plain_meters)
    overlap_features = pd.concat([plain_features, _synthetic_features(overlap_meters)])
    env = {
        "treatment": TreatmentGroup.from_fit_models(
            treatment_meters, features=_synthetic_features(treatment_meters)
        ),
        "pool": ComparisonPool.from_fit_models(
            {**plain_meters, **overlap_meters}, features=overlap_features
        ),
        "plain_pool": ComparisonPool.from_fit_models(plain_meters, features=plain_features),
        "overlap": [str(mid) for mid in overlap],
    }

    return env


_OVERLAP_METHOD_SETTINGS = {
    "cg_clustering": CG_Clustering_Settings(),
    "individual_meter_matching": IMM_Settings(),
    "random_sampling": RS_Settings(seed=42),
    "stratified_sampling": _STRATIFIED_SETTINGS,
}


@pytest.fixture(scope="module", params=list(_OVERLAP_METHOD_SETTINGS))
def overlap_runs(request, overlap_env):
    """Per method: the selection on the overlapping pool, the selection on the
    plain pool, and each population's (ids, ledger, features) before and after
    the overlapping run."""
    method = request.param
    settings = _OVERLAP_METHOD_SETTINGS[method]
    populations = (overlap_env["treatment"], overlap_env["pool"])

    def state():
        return [(p.ids, p.exclusions.copy(), p.features.copy()) for p in populations]

    before = state()
    overlapping = _select(*populations, method, method_settings=settings)
    after = state()
    plain = _select(
        overlap_env["treatment"], overlap_env["plain_pool"], method, method_settings=settings
    )
    runs = {"overlapping": overlapping, "plain": plain, "before": before, "after": after}

    return runs


def test_pool_fully_excluded_by_overlap_raises_naming_the_pool(overlap_env):
    """A pool whose every meter is also in the treatment group fails with an
    error naming the pool, not a bare concatenation error."""
    treatment = overlap_env["treatment"]

    with pytest.raises(ValueError, match="No comparison pool meters remain"):
        _select(treatment, treatment, "random_sampling", method_settings=RS_Settings())


def test_overlapping_ids_never_selected_into_clusters(overlap_env, overlap_runs):
    """A meter in both populations is never drawn as its own comparison."""
    drawn = set(overlap_runs["overlapping"].clusters.index)

    assert drawn.isdisjoint(overlap_env["overlap"]), drawn & set(overlap_env["overlap"])


def test_overlapping_ids_ledgered_once_as_pool_overlap(overlap_env, overlap_runs):
    """Each overlapping id has exactly one selection-stage ledger row, origin
    ``pool_overlap``, naming the pool copy, even when that copy is also short
    of 12 baseline months."""
    ledger = overlap_runs["overlapping"].exclusions
    rows = ledger[ledger["id"].isin(overlap_env["overlap"])]

    assert rows[["id", "stage", "origin", "reason"]].to_dict("records") == [
        {
            "id": mid,
            "stage": "selection",
            "origin": "pool_overlap",
            "reason": "pool copy excluded: meter is in the treatment group",
        }
        for mid in overlap_env["overlap"]
    ]


def test_overlapping_id_short_in_both_copies_has_one_row_per_copy(billing_meters):
    """A meter in both populations whose copies each span fewer than 12 baseline
    months keeps a ``baseline_data`` row for its treatment copy and a
    ``pool_overlap`` row for its pool copy."""
    treatment_meters = billing_meters["treatment"]
    mid = list(treatment_meters)[0]
    shortened = _with_short_baselines(treatment_meters, {mid})
    treatment = TreatmentGroup.from_fit_models(shortened)
    pool = ComparisonPool.from_fit_models({**billing_meters["pool"], mid: shortened[mid]})

    selection = _select(treatment, pool, "random_sampling", method_settings=RS_Settings(seed=42))

    ledger = selection.exclusions
    origins = sorted(ledger.loc[ledger["id"] == str(mid), "origin"])
    assert origins == ["baseline_data", "pool_overlap"], ledger[ledger["id"] == str(mid)]


def test_overlapping_ids_keep_their_treatment_copy(overlap_env, overlap_runs):
    """The treatment copy of an overlapping meter stays in selection."""
    weighted = set(overlap_runs["overlapping"].treatment_weights.index)

    assert set(overlap_env["overlap"]) <= weighted, set(overlap_env["overlap"]) - weighted


def test_overlap_selection_equals_selection_on_pool_without_overlap(overlap_runs):
    """Excluding the pool copies yields the selection a pool built without them
    gives: same clusters, treatment weights and fingerprint."""
    overlapping = overlap_runs["overlapping"]
    plain = overlap_runs["plain"]

    pd.testing.assert_frame_equal(overlapping.clusters, plain.clusters)
    pd.testing.assert_frame_equal(overlapping.treatment_weights, plain.treatment_weights)
    assert overlapping.fingerprint == plain.fingerprint


def test_overlap_selection_leaves_populations_unchanged(overlap_runs):
    """Excluding overlapping pool copies mutates neither population: ids,
    ledgers and features are as they were."""
    for (ids, ledger, features), (ids_after, ledger_after, features_after) in zip(
        overlap_runs["before"], overlap_runs["after"]
    ):
        assert ids_after == ids
        pd.testing.assert_frame_equal(ledger_after, ledger)
        pd.testing.assert_frame_equal(features_after, features)


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


def _daily_baseline(seed=0, hole=None, span=365, start="2020-01-01", nan_where=None):
    """A constructed DailyBaselineData with a smooth weather-driven load. ``hole``
    is a ``(start, stop)`` positional slice of observed rows set to NaN, and
    ``nan_where`` a predicate on the index selecting further NaN days; ``span``
    and ``start`` shift the meter's coverage relative to a group window."""
    index = pd.date_range(start, periods=span, freq="D", tz=_COVERAGE_TZ)
    rng = np.random.default_rng(seed)
    temperature = 50 + 25 * np.sin(np.arange(span) / 365 * 2 * np.pi) + rng.normal(0, 2, span)
    observed = 20 + 0.8 * np.abs(temperature - 62) + rng.normal(0, 1, span)
    frame = pd.DataFrame({"observed": observed, "temperature": temperature}, index=index)

    if hole is not None:
        lo, hi = hole
        frame.iloc[lo:hi, frame.columns.get_loc("observed")] = np.nan

    if nan_where is not None:
        frame.loc[nan_where(index), "observed"] = np.nan

    baseline = DailyBaselineData(
        frame.reset_index().rename(columns={"index": "datetime"}),
        is_electricity_data=True,
    )

    return baseline


def _daily_population(cls, specs, model, **kwargs):
    """Build a daily population from ``{id: baseline_data}``, reusing one fitted
    model (the coverage DQ reads only baseline data). The disqualified-meter
    warning is expected for gross-hole meters and suppressed here."""
    meters = {mid: {"model": model, "baseline_data": bd} for mid, bd in specs.items()}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Population includes disqualified")
        population = cls.from_fit_models(meters, model_frequency="daily", **kwargs)

    return population


@pytest.fixture(scope="module")
def daily_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DailyModel().fit(_daily_baseline(seed=0), ignore_disqualification=True)

    return model


def _random_selection(treatment, pool, **kwargs):
    settings = RS_Settings(n_meters_total=len(pool.ids), n_meters_per_treatment=None)
    selection = select_comparison_group(
        treatment,
        pool,
        method="random_sampling",
        method_settings=settings,
        basis="observed",
        **kwargs,
    )

    return selection


@pytest.mark.parametrize(
    "interpolate_missing, expected_excluded",
    [(True, []), (False, [("p-gap", "null values in features_df")])],
    ids=["interpolate_on", "interpolate_off"],
)
def test_selection_data_settings_interpolate_missing_reaches_data(
    daily_model, interpolate_missing, expected_excluded
):
    """A pool meter missing every summer Monday lacks one loadshape time: it is filled
    with interpolation on and excluded at data validation with it off, so the caller's
    flag reaches ``Data``."""
    pool_specs = {
        "p1": _daily_baseline(seed=1),
        "p2": _daily_baseline(seed=2),
        # 18 summer Mondays missing, 347/365 days clears baseline coverage
        "p-gap": _daily_baseline(
            seed=3,
            nan_where=lambda index: (index.dayofweek == 0) & index.month.isin([6, 7, 8, 9]),
        ),
    }
    treatment_specs = {"t0": _daily_baseline(seed=100), "t1": _daily_baseline(seed=101)}
    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)
    data_settings = Data_Settings(
        time_period=_const.TimePeriod.SEASONAL_DAY_OF_WEEK,
        interpolate_missing=interpolate_missing,
    )

    selection = select_comparison_group(
        treatment,
        pool,
        method="random_sampling",
        method_settings=RS_Settings(n_meters_total=2, n_meters_per_treatment=None, seed=0),
        basis="observed",
        data_settings=data_settings,
    )

    validation = selection.exclusions[selection.exclusions["origin"] == "data_validation"]
    assert list(zip(validation["id"], validation["reason"])) == expected_excluded
    assert len(selection.clusters.index.unique()) == 2
    assert selection.data_settings["interpolate_missing"] is interpolate_missing


def test_window_coverage_is_finite_fraction_over_window_length():
    baseline = _daily_baseline(hole=(100, 160))  # 60 of 365 observed days NaN
    window = (baseline.df.index.min(), baseline.df.index.max())

    coverage = window_coverage(baseline, window, "D")

    assert len(pd.date_range(window[0], window[1], freq="D")) == 365
    assert coverage == pytest.approx(305 / 365)


def test_window_coverage_full_clean_meter_is_one():
    baseline = _daily_baseline()
    window = (baseline.df.index.min(), baseline.df.index.max())

    coverage = window_coverage(baseline, window, "D")

    assert coverage == pytest.approx(1.0)


def test_window_coverage_penalizes_a_span_short_of_the_window():
    baseline = _daily_baseline(span=300, start="2020-02-01")  # 300 finite days inside 2020
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


_NARROW_WINDOW = (
    pd.Timestamp("2020-01-01", tz=_COVERAGE_TZ),
    pd.Timestamp("2020-08-01", tz=_COVERAGE_TZ),
)


@pytest.fixture(scope="module")
def partial_hole_env(daily_model):
    """Two full-2020 treatment meters and a pool whose ``p-partial`` meter has a
    data hole after August (285/365 = 0.7808 coverage over the full year)."""
    pool_specs = {
        "p-partial": _daily_baseline(seed=7, hole=(250, 330)),
        "p1": _daily_baseline(seed=1),
        "p2": _daily_baseline(seed=2),
    }
    treatment_specs = {"t0": _daily_baseline(seed=100), "t1": _daily_baseline(seed=101)}
    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)

    return treatment, pool


def test_default_baseline_window_is_treatment_baseline_union(partial_hole_env):
    """Without ``baseline_window`` the coverage check judges the treatment
    meters' baseline union, which the partial pool meter fails."""
    treatment, pool = partial_hole_env
    expected = (
        pd.Timestamp("2020-01-01", tz=_COVERAGE_TZ),
        pd.Timestamp("2020-12-30", tz=_COVERAGE_TZ),
    )

    selection = _random_selection(treatment, pool)

    assert "p-partial" not in selection.clusters.index
    assert selection.baseline_window == expected


@pytest.mark.parametrize("naive", [False, True], ids=["tz_aware", "naive"])
def test_explicit_baseline_window_changes_coverage_verdict(partial_hole_env, naive):
    """A window ending before the pool meter's hole lets it survive, and the
    window is recorded on the selection in the treatment timezone whether its
    bounds arrive naive or tz-aware."""
    treatment, pool = partial_hole_env
    window = _NARROW_WINDOW
    if naive:
        window = tuple(bound.tz_localize(None) for bound in _NARROW_WINDOW)

    selection = _random_selection(treatment, pool, baseline_window=window)

    assert "p-partial" in selection.clusters.index
    assert selection.exclusions[selection.exclusions["origin"] == "baseline_coverage"].empty
    assert selection.baseline_window == _NARROW_WINDOW
    assert all(str(bound.tz) == _COVERAGE_TZ for bound in selection.baseline_window)


def test_json_roundtrip_restores_baseline_window(partial_hole_env):
    """A selection JSON round trip restores the recorded window as tz-aware
    bounds and leaves the fingerprint unchanged."""
    treatment, pool = partial_hole_env
    selection = _random_selection(treatment, pool, baseline_window=_NARROW_WINDOW)

    rebuilt = ComparisonGroupSelection.from_json(selection.to_json())

    assert rebuilt.baseline_window == _NARROW_WINDOW
    assert all(str(bound.tz) == _COVERAGE_TZ for bound in rebuilt.baseline_window)
    assert rebuilt.fingerprint == selection.fingerprint


def test_default_baseline_window_includes_short_treatment_meter(daily_model):
    """A treatment meter with a short, earlier baseline widens the default union
    even though selection then excludes it; its peers and the pool still pass
    coverage over the widened window (365/396 = 0.9217)."""
    treatment_specs = {
        "t0": _daily_baseline(seed=100),
        "t1": _daily_baseline(seed=101),
        "t-short": _daily_baseline(seed=102, span=100, start="2019-12-01"),
    }
    pool_specs = {f"p{i}": _daily_baseline(seed=20 + i) for i in range(3)}
    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)
    expected = (
        pd.Timestamp("2019-12-01", tz=_COVERAGE_TZ),
        pd.Timestamp("2020-12-30", tz=_COVERAGE_TZ),
    )

    selection = _random_selection(treatment, pool)

    assert selection.baseline_window == expected
    assert list(selection.exclusions["id"]) == ["t-short"]
    assert sorted(selection.treatment_weights.index) == ["t0", "t1"]
    assert sorted(selection.clusters.index) == ["p0", "p1", "p2"]


def test_pinned_bank_selections_load(pinned_models):
    """Every stored selection in the pinned model bank rebuilds; real-run
    selections record the treatment baseline union of the 2018 fixture year and
    hand-built ones record no window."""
    expected = (
        pd.Timestamp("2018-01-01T00:00:00-06:00"),
        pd.Timestamp("2018-12-31T00:00:00-06:00"),
    )

    for name, stored in pinned_models["selections"].items():
        selection = ComparisonGroupSelection.from_json(stored)

        if selection.data_settings is None:
            assert selection.baseline_window is None, name
        else:
            assert selection.baseline_window == expected, name
