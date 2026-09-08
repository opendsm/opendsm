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
import json
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import pytest

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const
from opendsm.comparison_groups.population import ComparisonPool, TreatmentGroup
from opendsm.comparison_groups.selection import (
    ComparisonGroupSelection,
    SelectionMethod,
    _build_data,
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
from opendsm.eemeter.models import DailyModel
from opendsm.eemeter.models.daily.data import DailyBaselineData

from savings.equivalence_env import VARIANT_NAMES



# Individual meter matching with duplicate matches allowed and more matches than
# distinct pool meters forces at least one pool meter to appear for multiple
# treatments, so the normalized clusters carry a genuine duplicate id index.
_IMM_DUPLICATE_SETTINGS = dict(allow_duplicate_matches=True, n_matches_per_treatment=8)

# Fixed bins: the distance-binning search needs more treatment meters than the
# five of ``short_baseline_env``.
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

_METHOD_SETTINGS = {
    "cg_clustering": CG_Clustering_Settings(),
    "individual_meter_matching": IMM_Settings(**_IMM_DUPLICATE_SETTINGS),
    "random_sampling": RS_Settings(seed=42),
    "stratified_sampling": _STRATIFIED_SETTINGS,
}


@pytest.fixture(scope="module")
def billing_meters(comstock, model_bank):
    """Pinned BillingModels for real ComStock monthly meters: 6 treatment and 40
    pool, baseline only; the pinned fits keep the selections reproducible
    across machines."""
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


@pytest.fixture(scope="module")
def featured(billing_meters):
    """The billing treatment and pool, each carrying synthetic stratification
    features, so every method can run on them."""
    treatment_meters = billing_meters["treatment"]
    populations = (
        TreatmentGroup.from_fit_models(
            treatment_meters, features=_synthetic_features(treatment_meters)
        ),
        _pool_with_features(billing_meters),
    )

    return populations


def _hand_built_selection(clusters, treatment_weights, method="random_sampling", **fields):
    """A selection over ``clusters`` and ``treatment_weights`` as given; ``fields``
    override the empty settings, window and ledger."""
    arguments = {
        "basis": "modeled",
        "method_settings": {},
        "data_settings": None,
        "baseline_window": None,
        "exclusions": exclusions.empty_ledger(),
        "treatment_ids": list(treatment_weights.index),
        "pool_ids": list(dict.fromkeys(clusters.index)),
        "tz": "UTC",
        **fields,
    }
    selection = ComparisonGroupSelection(
        clusters=clusters, treatment_weights=treatment_weights, method=method, **arguments
    )

    return selection


def _int_id_selection(**fields):
    """A one-cluster selection whose meter ids are ints."""
    clusters = pd.DataFrame(
        {"cluster": [0, 0], "weight": [1.0, 1.0]}, index=pd.Index([101, 102], name="id")
    )
    weights = pd.DataFrame({"pct_cluster_0": [1.0]}, index=pd.Index([7], name="id"))
    selection = _hand_built_selection(clusters, weights, **fields)

    return selection


# -- all four methods --------------------------------------------------------


def test_clustering_selection_has_expected_fields(treatment, pool):
    selection = select_comparison_group(treatment, pool, "cg_clustering")

    assert selection.method == SelectionMethod.CG_CLUSTERING
    assert selection.basis == "error"
    assert selection.tz == treatment.tz
    assert set(selection.treatment_ids) == set(treatment.ids)
    assert set(selection.pool_ids) == set(pool.ids)
    assert len(selection.fingerprint) == 64


def test_random_selection_default_basis_is_modeled(treatment, pool):
    selection = select_comparison_group(treatment, pool, "random_sampling")

    assert selection.method == SelectionMethod.RANDOM_SAMPLING
    assert selection.basis == "modeled"


# -- normalized clusters schema per method -----------------------------------


def test_clustering_clusters_schema(treatment, pool):
    selection = select_comparison_group(treatment, pool, "cg_clustering")
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
    selection = select_comparison_group(
        treatment,
        pool,
        "individual_meter_matching",
        method_settings=IMM_Settings(**_IMM_DUPLICATE_SETTINGS),
    )
    clusters = selection.clusters

    assert list(clusters.columns) == ["cluster", "weight", "treatment", "distance", "duplicated"]
    assert clusters["cluster"].dtype == np.int64
    assert clusters["distance"].dtype == np.float64
    assert clusters["duplicated"].dtype == bool
    assert str(clusters["treatment"].dtype) == "string"
    assert clusters.index.duplicated().any()
    assert set(clusters["treatment"]) <= set(selection.treatment_ids)


def test_random_clusters_schema(treatment, pool):
    selection = select_comparison_group(
        treatment, pool, "random_sampling", method_settings=RS_Settings()
    )
    clusters = selection.clusters

    assert list(clusters.columns) == ["cluster", "weight", "treatment"]
    assert (clusters["cluster"] == 0).all()
    assert (clusters["weight"] == 1.0).all()
    assert clusters["treatment"].isna().all()


def test_stratified_clusters_schema(stratified_feature_loadshape_data):
    """The stratified DSS output maps to the cluster-0 / weight-1 normalized schema."""
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
    clusters_raw, weights_raw = Stratified_Sampling(settings).get_comparison_group(
        treatment_data, pool_data
    )

    clusters = _hand_built_selection(clusters_raw, weights_raw, "stratified_sampling").clusters

    assert list(clusters.columns) == ["cluster", "weight", "treatment"]
    assert (clusters["cluster"] == 0).all()
    assert (clusters["weight"] == 1.0).all()
    assert clusters["treatment"].isna().all()
    assert not clusters.empty


def test_clusters_lacking_a_method_column_raise():
    """An individual-meter-matching clusters frame without ``distance`` and
    ``duplicated`` is rejected, naming both."""
    clusters = pd.DataFrame({"cluster": [0], "weight": [1.0]}, index=pd.Index(["p0"], name="id"))
    weights = pd.DataFrame({"pct_cluster_0": [1.0]}, index=pd.Index(["t0"], name="id"))

    with pytest.raises(ValueError, match=r"\['distance', 'duplicated'\]"):
        _hand_built_selection(clusters, weights, "individual_meter_matching")


# -- pre-mutation settings capture + determinism -----------------------------


def test_clustering_settings_captured_before_mutation(treatment, pool):
    """CG_Clustering.get_labels rebuilds the n_cluster bounds on self.settings;
    the artifact must record the caller's pre-mutation settings."""
    selection = select_comparison_group(
        treatment, pool, "cg_clustering", method_settings=CG_Clustering_Settings()
    )
    default_bounds = CG_Clustering_Settings().model_dump(mode="json")["bisecting_kmeans"]["n_cluster"]

    assert selection.method_settings["bisecting_kmeans"]["n_cluster"] == default_bounds


def test_clustering_rerun_from_deserialized_settings_is_deterministic(treatment, pool):
    selection = select_comparison_group(treatment, pool, "cg_clustering")
    rebuilt_settings = CG_Clustering_Settings(**selection.method_settings)

    rerun = select_comparison_group(
        treatment, pool, "cg_clustering", method_settings=rebuilt_settings
    )

    pd.testing.assert_frame_equal(selection.clusters, rerun.clusters)
    pd.testing.assert_frame_equal(selection.treatment_weights, rerun.treatment_weights)
    assert selection.fingerprint == rerun.fingerprint


# -- stratified feature requirements -----------------------------------------


def test_stratified_without_features_raises(treatment, pool):
    with pytest.raises(ValueError, match="requires stratification features"):
        select_comparison_group(treatment, pool, "stratified_sampling")


def test_stratified_build_data_carries_loadshapes_and_features(billing_meters):
    pool = _pool_with_features(billing_meters)

    data = _build_data(pool, SelectionMethod.STRATIFIED_SAMPLING, "modeled", None, ())

    assert isinstance(data, Data)
    assert data.loadshape is not None
    assert data.features is not None
    assert set(data.features.columns) == {"summer_usage", "winter_usage"}


def test_stratified_build_data_trims_no_meter(billing_meters):
    """The population owns the pool trim: a caller's ``max_pool_size`` below the
    population size leaves every meter in the stratified ``Data``."""
    pool = _pool_with_features(billing_meters)
    settings = Data_Settings(time_period=_const.TimePeriod.MONTH, max_pool_size=5)

    data = _build_data(pool, SelectionMethod.STRATIFIED_SAMPLING, "modeled", settings, ())

    assert sorted(data.ids) == sorted(pool.ids)
    assert sorted(data.features.index) == sorted(pool.ids)


def test_stratified_build_data_drops_features_of_a_meter_the_loadshape_build_drops(
    billing_meters,
):
    """A meter the loadshape build drops (all-NaN modeled baseline) is on the
    stratified ``Data``'s exclusion ledger and has no selectable features row."""
    pool = _pool_with_features(billing_meters)
    victim = pool.ids[0]
    pool.prediction_frames("baseline")[victim].loc[:, "modeled"] = np.nan

    data = _build_data(pool, SelectionMethod.STRATIFIED_SAMPLING, "modeled", None, ())

    assert victim in set(data.excluded_ids["id"])
    assert victim not in set(data.features.index)


def test_stratified_build_data_drops_excluded_ids_from_features(billing_meters):
    """An id selection excludes loses its features row too, so stratified
    sampling cannot draw it."""
    pool = _pool_with_features(billing_meters)
    excluded = pool.ids[0]

    data = _build_data(pool, SelectionMethod.STRATIFIED_SAMPLING, "modeled", None, [excluded])

    assert sorted(data.features.index) == sorted(pool.ids[1:])


def test_stratified_build_data_drops_features_without_a_meter(billing_meters):
    """A features row whose id the population does not hold is dropped, so
    stratified sampling cannot draw it."""
    pool_meters = billing_meters["pool"]
    features = _synthetic_features([*pool_meters, "absent"])
    pool = ComparisonPool.from_fit_models(pool_meters, features=features)

    data = _build_data(pool, SelectionMethod.STRATIFIED_SAMPLING, "modeled", None, ())

    assert sorted(data.features.index) == sorted(pool.ids)


# -- serialization roundtrip -------------------------------------------------


@pytest.mark.parametrize("method", list(_METHOD_SETTINGS))
def test_json_roundtrip_is_exact(featured, method):
    selection = select_comparison_group(
        *featured, method, method_settings=_METHOD_SETTINGS[method]
    )

    rebuilt = ComparisonGroupSelection.from_json(selection.to_json())

    pd.testing.assert_frame_equal(selection.clusters, rebuilt.clusters)
    pd.testing.assert_frame_equal(selection.treatment_weights, rebuilt.treatment_weights)
    pd.testing.assert_frame_equal(selection.exclusions, rebuilt.exclusions)
    assert rebuilt.fingerprint == selection.fingerprint
    assert rebuilt.method == selection.method
    assert rebuilt.basis == selection.basis
    assert rebuilt.tz == selection.tz
    assert rebuilt.treatment_ids == selection.treatment_ids
    assert rebuilt.pool_ids == selection.pool_ids
    assert rebuilt.method_settings == selection.method_settings
    assert rebuilt.data_settings == selection.data_settings


def test_int_id_selection_roundtrips_with_str_ids():
    """A selection built with int meter ids holds them as strings and rebuilds
    from its own JSON with equal tables and fingerprint."""
    selection = _int_id_selection()

    rebuilt = ComparisonGroupSelection.from_json(selection.to_json())

    assert list(selection.clusters.index) == ["101", "102"]
    assert list(selection.treatment_weights.index) == ["7"]
    pd.testing.assert_frame_equal(selection.clusters, rebuilt.clusters)
    pd.testing.assert_frame_equal(selection.treatment_weights, rebuilt.treatment_weights)
    assert rebuilt.fingerprint == selection.fingerprint


def _tamper_stored_fingerprint(payload):
    payload["fingerprint"] = "0" * 64


def _tamper_treatment_weight(payload):
    tables = payload["tables"]
    weights = pd.read_json(StringIO(tables["treatment_weights"]), orient="table")
    weights["pct_cluster_0"] = 0.5
    tables["treatment_weights"] = weights.to_json(orient="table", double_precision=15)


@pytest.mark.parametrize(
    "tamper",
    [_tamper_stored_fingerprint, _tamper_treatment_weight],
    ids=["stored_fingerprint", "treatment_weight"],
)
def test_payload_inconsistent_with_its_fingerprint_does_not_deserialize(tamper):
    """A payload whose stored fingerprint does not match its tables raises,
    whether the fingerprint or a table weight was edited."""
    payload = json.loads(_int_id_selection().to_json())
    tamper(payload)

    with pytest.raises(ValueError, match="fingerprint does not match the stored value"):
        ComparisonGroupSelection.from_json(json.dumps(payload))


def test_verify_fingerprint_accepts_an_unchanged_selection():
    _int_id_selection().verify_fingerprint()


def test_verify_fingerprint_raises_after_a_table_edit():
    selection = _int_id_selection()
    selection.treatment_weights.iloc[0, 0] = 0.5

    with pytest.raises(ValueError, match="selection fingerprint does not match its tables"):
        selection.verify_fingerprint()


def test_fingerprint_ignores_the_exclusions_ledger():
    """The fingerprint covers clusters and treatment weights only: selections
    differing only in their exclusions ledger share it."""
    ledger = exclusions.append(
        exclusions.empty_ledger(), ["9"], "selection", "pool_overlap", "overlap"
    )

    with_ledger = _int_id_selection(exclusions=ledger)

    assert with_ledger.fingerprint == _int_id_selection().fingerprint


# -- timezone guard ----------------------------------------------------------


def test_timezone_mismatch_raises(treatment, pool):
    mismatched = copy.copy(pool)
    mismatched.tz = "America/New_York"

    with pytest.raises(ValueError, match="timezone"):
        select_comparison_group(treatment, mismatched, "cg_clustering")


# -- baseline guard ----------------------------------------------------------


def test_selection_without_baseline_raises_naming_every_meter(treatment, pool, billing_meters):
    """Selection refuses populations reloaded without baseline data, naming each
    meter that lacks it in either population."""
    kept = pool.ids[0]
    reloaded_treatment = TreatmentGroup.from_json(treatment.to_json())
    reloaded_pool = ComparisonPool.from_json(
        pool.to_json(), baseline={kept: billing_meters["pool"][kept]["baseline_df"]}
    )
    missing = {"treatment": treatment.ids, "pool": pool.ids[1:]}

    with pytest.raises(ValueError, match="Selection requires baseline data") as raised:
        select_comparison_group(reloaded_treatment, reloaded_pool, "random_sampling")

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


def test_finer_pool_selects_at_the_treatment_time_period(daily_and_billing_meters):
    """A billing (coarser) treatment against a daily (finer) pool is allowed:
    both loadshapes build at the treatment's MONTH time period, so individual
    meter matching, which needs loadshapes of equal length, matches every
    treatment meter, and the defaulted settings are stored."""
    treatment = TreatmentGroup.from_fit_models(daily_and_billing_meters["billing"])
    pool = ComparisonPool.from_fit_models(daily_and_billing_meters["daily"])
    expected_settings = Data_Settings(time_period=_const.TimePeriod.MONTH).model_dump(mode="json")

    selection = select_comparison_group(
        treatment,
        pool,
        method="individual_meter_matching",
        method_settings=IMM_Settings(n_matches_per_treatment=1),
        basis="modeled",
    )

    assert selection.data_settings == expected_settings
    assert sorted(selection.clusters["treatment"]) == sorted(treatment.ids)


# -- exclusions ledger -------------------------------------------------------


@pytest.fixture(scope="module")
def trimmed_pool_run(billing_meters, treatment):
    """A random selection on a pool trimmed to 35 meters, one of whose survivors
    has an all-NaN modeled baseline, so its monthly loadshape is entirely
    missing and ``Data`` drops it during validation."""
    pool = ComparisonPool.from_fit_models(billing_meters["pool"], max_pool_size=35, seed=0)
    victim = pool.ids[0]
    pool.prediction_frames("baseline")[victim].loc[:, "modeled"] = np.nan

    selection = select_comparison_group(
        treatment, pool, "random_sampling", method_settings=RS_Settings()
    )

    return {"pool": pool, "victim": victim, "selection": selection}


def test_data_validation_exclusion_recorded_with_the_data_reason(trimmed_pool_run):
    """A meter that ``Data`` drops during validation is recorded at the
    selection stage with the verbatim ``Data`` reason string and is absent from
    the clusters."""
    selection = trimmed_pool_run["selection"]
    victim = trimmed_pool_run["victim"]

    row = selection.exclusions.set_index("id").loc[victim]

    assert row[["stage", "origin", "reason"]].to_dict() == {
        "stage": "selection",
        "origin": "data_validation",
        "reason": "fewer loadshape times present than min_data_pct_required",
    }
    assert victim not in set(selection.clusters.index)


def test_pool_trim_rows_stay_on_the_pool_ledger(trimmed_pool_run):
    """Pool-trim rows live on the pool population's ledger and never reach the
    selection ledger."""
    trimmed_ids = set(trimmed_pool_run["pool"].exclusions["id"])
    ledger = trimmed_pool_run["selection"].exclusions

    assert len(trimmed_ids) == 5
    assert set(ledger["id"]).isdisjoint(trimmed_ids)


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
def test_stage_exclusions_record_an_incomplete_meter_once(
    treatment_lengths, pool_lengths, incomplete_id
):
    """With interpolation off, ``Data`` drops one incomplete meter in either
    population, and the selection-stage ledger records it once with ``Data``'s
    reason."""
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
        interpolate_missing=False,
    )
    treatment_data = Data(time_series_df=_hourly_meters(treatment_lengths), settings=settings)
    pool_data = Data(time_series_df=_hourly_meters(pool_lengths), settings=settings)
    treatment_weights = pd.DataFrame(
        {"pct_cluster_0": 1.0}, index=pd.Index(treatment_data.ids, name="id")
    )

    ledger = _stage_exclusions(treatment_data, pool_data, treatment_weights)

    assert ledger[["id", "stage", "origin", "reason"]].to_dict("records") == [
        {
            "id": incomplete_id,
            "stage": "selection",
            "origin": "data_validation",
            "reason": "missing loadshape times with interpolate_missing off",
        }
    ]


# -- short-baseline exclusion (MONTH loadshapes) -----------------------------


def _shorten_baselines(population, short_ids):
    """Cut the attached baselines of ``short_ids`` to their first six daily rows,
    so each spans a single month; done after construction so the cut is not
    itself a construction-time sufficiency failure."""
    for mid in short_ids:
        baseline_data = population._meters[mid].baseline_data
        baseline_data._df = baseline_data._df.head(6)


@pytest.fixture
def short_baseline_env(billing_meters):
    """Billing populations, both carrying stratification features, in which one
    treatment meter and one pool meter have a single baseline month."""
    short_treatment = list(billing_meters["treatment"])[1]
    short_pool = list(billing_meters["pool"])[3]
    treatment = TreatmentGroup.from_fit_models(
        billing_meters["treatment"], features=_synthetic_features(billing_meters["treatment"])
    )
    pool = ComparisonPool.from_fit_models(
        billing_meters["pool"], features=_synthetic_features(billing_meters["pool"])
    )
    _shorten_baselines(treatment, {short_treatment})
    _shorten_baselines(pool, {short_pool})
    env = {
        "treatment": treatment,
        "pool": pool,
        "short_treatment": short_treatment,
        "short_pool": short_pool,
    }

    return env


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

    select_comparison_group(*populations, method, method_settings=method_settings)

    for population, (ids, ledger, features) in zip(populations, before):
        assert population.ids == ids
        pd.testing.assert_frame_equal(population.exclusions, ledger)
        pd.testing.assert_frame_equal(population.features, features)


def test_short_baseline_meters_ledgered_at_selection_with_month_counts(short_baseline_env):
    """Each short-baseline meter is one selection-stage ``baseline_data`` row
    carrying its month count, and is absent from the selection's tables."""
    short_ids = [short_baseline_env["short_treatment"], short_baseline_env["short_pool"]]

    selection = select_comparison_group(
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

    selection = select_comparison_group(
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
    meter is excluded for baseline coverage instead."""
    settings = Data_Settings(time_period=_const.TimePeriod.DAY_OF_WEEK)

    selection = select_comparison_group(
        short_baseline_env["treatment"],
        short_baseline_env["pool"],
        "random_sampling",
        method_settings=RS_Settings(),
        data_settings=settings,
    )

    assert (selection.exclusions["origin"] != "baseline_data").all()
    short_treatment = str(short_baseline_env["short_treatment"])
    row = selection.exclusions.set_index("id").loc[short_treatment, ["origin", "reason"]]
    assert row.to_dict() == {
        "origin": "baseline_coverage",
        "reason": "baseline coverage below minimum over the group window",
    }


def test_select_comparison_group_raises_when_every_treatment_meter_is_short_of_12_months(
    billing_meters, pool
):
    """Selection rejects a treatment population in which every meter is short of
    12 baseline months, naming each meter, and leaves the population unchanged."""
    treatment_meters = billing_meters["treatment"]
    group = TreatmentGroup.from_fit_models(treatment_meters)
    _shorten_baselines(group, group.ids)

    with pytest.raises(ValueError, match="at least 12 baseline months") as excinfo:
        select_comparison_group(group, pool, method="random_sampling")

    for mid in treatment_meters:
        assert f"{mid}: baseline months: 1" in str(excinfo.value)
    assert group.ids == list(treatment_meters)
    assert group.exclusions.empty


# -- treatment/pool overlap --------------------------------------------------


@pytest.fixture(scope="module")
def overlap_env(billing_meters):
    """Billing populations with features in which the first two treatment
    meters also sit at the end of the pool; the pool copy of the second has a
    single baseline month. ``plain_pool`` is the same pool without them."""
    treatment_meters = billing_meters["treatment"]
    plain_meters = billing_meters["pool"]
    overlap = list(treatment_meters)[:2]
    overlap_meters = {mid: treatment_meters[mid] for mid in overlap}
    plain_features = _synthetic_features(plain_meters)
    overlap_features = pd.concat([plain_features, _synthetic_features(overlap_meters)])
    pool = ComparisonPool.from_fit_models(
        {**plain_meters, **overlap_meters}, features=overlap_features
    )
    _shorten_baselines(pool, {overlap[1]})
    env = {
        "treatment": TreatmentGroup.from_fit_models(
            treatment_meters, features=_synthetic_features(treatment_meters)
        ),
        "pool": pool,
        "plain_pool": ComparisonPool.from_fit_models(plain_meters, features=plain_features),
        "overlap": [str(mid) for mid in overlap],
    }

    return env


@pytest.fixture(scope="module", params=list(_METHOD_SETTINGS))
def overlap_runs(request, overlap_env):
    """Per method: the selection on the overlapping pool, the selection on the
    plain pool, and each population's (ids, ledger, features) before and after
    the overlapping run."""
    method = request.param
    settings = _METHOD_SETTINGS[method]
    populations = (overlap_env["treatment"], overlap_env["pool"])

    def state():
        return [(p.ids, p.exclusions.copy(), p.features.copy()) for p in populations]

    before = state()
    overlapping = select_comparison_group(*populations, method, method_settings=settings)
    after = state()
    plain = select_comparison_group(
        overlap_env["treatment"], overlap_env["plain_pool"], method, method_settings=settings
    )
    runs = {"overlapping": overlapping, "plain": plain, "before": before, "after": after}

    return runs


def test_pool_fully_excluded_by_overlap_raises_naming_the_pool(overlap_env):
    """A pool whose every meter is also in the treatment group fails with an
    error naming the pool, not a bare concatenation error."""
    treatment = overlap_env["treatment"]

    with pytest.raises(ValueError, match="No comparison pool meters remain"):
        select_comparison_group(
            treatment, treatment, "random_sampling", method_settings=RS_Settings()
        )


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
    treatment = TreatmentGroup.from_fit_models(treatment_meters)
    pool = ComparisonPool.from_fit_models({**billing_meters["pool"], mid: treatment_meters[mid]})
    _shorten_baselines(treatment, {mid})
    _shorten_baselines(pool, {mid})

    selection = select_comparison_group(
        treatment, pool, "random_sampling", method_settings=RS_Settings(seed=42)
    )

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
    clustering = select_comparison_group(treatment, pool, "cg_clustering")
    imm = select_comparison_group(
        treatment,
        pool,
        "individual_meter_matching",
        method_settings=IMM_Settings(**_IMM_DUPLICATE_SETTINGS),
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
    """A constructed daily baseline frame with a smooth weather-driven load.
    ``hole`` is a ``(start, stop)`` positional slice of observed rows set to NaN,
    and ``nan_where`` a predicate on the index selecting further NaN days;
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

    if nan_where is not None:
        frame.loc[nan_where(index), "observed"] = np.nan

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
        population = cls.from_fit_models(meters, model_frequency="daily", **kwargs)

    return population


@pytest.fixture(scope="module")
def daily_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DailyModel().fit(
            _daily_baseline(seed=0), is_electricity_data=True, enforce_compliance=False
        )

    return model


def _random_selection(treatment, pool, n_meters_total, **kwargs):
    settings = RS_Settings(n_meters_total=n_meters_total, n_meters_per_treatment=None)
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
    [(True, []), (False, [("p-gap", "only null readings in a loadshape time")])],
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
            nan_where=lambda index: (index.day_of_week == 0) & index.month.isin([6, 7, 8, 9]),
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


@pytest.fixture(scope="module")
def gross_hole_env(daily_model):
    """Two treatment meters and a pool of three meters spanning the 365 days from
    2020-01-01, plus ``p-hole``, whose 100-day observed hole leaves 265/365 = 0.7260
    coverage."""
    pool_specs = {
        "p1": _daily_baseline(seed=1),
        "p2": _daily_baseline(seed=2),
        "p3": _daily_baseline(seed=3),
        "p-hole": _daily_baseline(seed=9, hole=(100, 200)),
    }
    treatment_specs = {"t0": _daily_baseline(seed=100), "t1": _daily_baseline(seed=101)}
    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)

    return treatment, pool


def test_gross_hole_pool_meter_ledgered_with_its_coverage(gross_hole_env):
    selection = _random_selection(*gross_hole_env, n_meters_total=3)

    coverage_rows = selection.exclusions[selection.exclusions["origin"] == "baseline_coverage"]
    assert list(coverage_rows["id"]) == ["p-hole"]
    row = coverage_rows.iloc[0]
    assert row["stage"] == "selection"
    assert row["reason"] == "baseline coverage below minimum over the group window"
    assert "baseline coverage 0.7260 < min_baseline_coverage 0.9" in row["detail"]
    assert "too_many_days_with_missing_observed_data" in row["detail"]


def test_random_sampling_draws_n_meters_total_from_meters_clearing_coverage(gross_hole_env):
    """Coverage failures leave the pool before sampling, so random sampling draws
    exactly ``n_meters_total`` meters, none of them below the floor."""
    selection = _random_selection(*gross_hole_env, n_meters_total=3)

    assert sorted(selection.clusters.index) == ["p1", "p2", "p3"]


def test_imm_treatment_meter_keeps_a_full_match_set_around_a_coverage_failure(gross_hole_env):
    """Coverage failures leave the pool before matching, so every treatment meter
    keeps ``n_matches_per_treatment`` matches, none of them below the floor."""
    settings = IMM_Settings(allow_duplicate_matches=True, n_matches_per_treatment=3)

    selection = select_comparison_group(
        *gross_hole_env,
        "individual_meter_matching",
        method_settings=settings,
        basis="observed",
    )

    matches = selection.clusters.groupby("treatment").size().to_dict()
    assert matches == {"t0": 3, "t1": 3}
    assert "p-hole" not in selection.clusters.index


@pytest.mark.parametrize(
    "holed_role, message",
    [("treatment", "No treatment meters remain"), ("pool", "No comparison pool meters remain")],
)
def test_coverage_excluding_every_meter_of_a_population_raises(daily_model, holed_role, message):
    specs = {
        "treatment": {"t0": _daily_baseline(seed=100), "t1": _daily_baseline(seed=101)},
        "pool": {"p1": _daily_baseline(seed=1), "p2": _daily_baseline(seed=2)},
    }
    specs[holed_role] = {
        mid: _daily_baseline(seed=9, hole=(100, 200)) for mid in specs[holed_role]
    }
    treatment = _daily_population(TreatmentGroup, specs["treatment"], daily_model)
    pool = _daily_population(ComparisonPool, specs["pool"], daily_model)

    with pytest.raises(ValueError, match=message):
        _random_selection(treatment, pool, n_meters_total=1)


def _hole_baselines(population, hole_ids):
    """NaN the attached observed baselines of ``hole_ids`` after their first 30
    days, far below the default coverage floor; done after construction, on the
    data objects the population built from its frames."""
    for mid in hole_ids:
        baseline_data = population._meters[mid].baseline_data
        baseline_data._df.iloc[30:, baseline_data._df.columns.get_loc("observed")] = np.nan


@pytest.fixture(scope="module")
def coverage_hole_clustering(billing_meters, treatment):
    """CG clustering on the billing pool with its last meter's observed baseline
    holed, and on the same pool without that meter."""
    pool_meters = billing_meters["pool"]
    holed_id = list(pool_meters)[-1]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Population includes disqualified")
        holed_pool = ComparisonPool.from_fit_models(pool_meters)
    _hole_baselines(holed_pool, {holed_id})

    plain_pool = ComparisonPool.from_fit_models(
        {mid: entry for mid, entry in pool_meters.items() if mid != holed_id}
    )
    runs = {
        "holed_id": str(holed_id),
        "holed": select_comparison_group(treatment, holed_pool, "cg_clustering"),
        "plain": select_comparison_group(treatment, plain_pool, "cg_clustering"),
    }

    return runs


def test_pool_meter_failing_coverage_never_reaches_the_clustering(coverage_hole_clustering):
    """A pool meter below the coverage floor is left out before clustering: the
    clusters and treatment weights equal those of the pool without it."""
    holed = coverage_hole_clustering["holed"]
    plain = coverage_hole_clustering["plain"]

    coverage_rows = holed.exclusions[holed.exclusions["origin"] == "baseline_coverage"]
    assert list(coverage_rows["id"]) == [coverage_hole_clustering["holed_id"]]
    pd.testing.assert_frame_equal(holed.clusters, plain.clusters)
    pd.testing.assert_frame_equal(holed.treatment_weights, plain.treatment_weights)


def test_baseline_coverage_excludes_treatment_meter(daily_model):
    treatment_specs = {
        "t-ok": _daily_baseline(seed=1),
        "t-hole": _daily_baseline(seed=9, hole=(100, 200)),
    }
    pool_specs = {f"p{i}": _daily_baseline(seed=20 + i) for i in range(4)}
    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)

    selection = _random_selection(treatment, pool, n_meters_total=4)

    assert list(selection.treatment_weights.index) == ["t-ok"]

    coverage_rows = selection.exclusions[selection.exclusions["origin"] == "baseline_coverage"]
    assert list(coverage_rows["id"]) == ["t-hole"]
    assert "baseline coverage 0.7260 < min_baseline_coverage 0.9" in coverage_rows.iloc[0]["detail"]


def test_offset_meter_within_coverage_survives(daily_model):
    pool_specs = {
        "p-offset": _daily_baseline(seed=5, start="2020-01-21"),  # 345/365 = 0.945
        "p1": _daily_baseline(seed=1),
        "p2": _daily_baseline(seed=2),
    }
    treatment_specs = {"t0": _daily_baseline(seed=100), "t1": _daily_baseline(seed=101)}
    treatment = _daily_population(TreatmentGroup, treatment_specs, daily_model)
    pool = _daily_population(ComparisonPool, pool_specs, daily_model)

    selection = _random_selection(treatment, pool, n_meters_total=3)

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

    selection = _random_selection(treatment, pool, n_meters_total=2)

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

    selection = _random_selection(treatment, pool, n_meters_total=3, baseline_window=window)

    assert "p-partial" in selection.clusters.index
    assert selection.exclusions[selection.exclusions["origin"] == "baseline_coverage"].empty
    assert selection.baseline_window == _NARROW_WINDOW
    assert all(str(bound.tz) == _COVERAGE_TZ for bound in selection.baseline_window)


def test_json_roundtrip_restores_baseline_window(partial_hole_env):
    """A selection JSON round trip restores the recorded window as tz-aware
    bounds and leaves the fingerprint unchanged."""
    treatment, pool = partial_hole_env
    selection = _random_selection(
        treatment, pool, n_meters_total=3, baseline_window=_NARROW_WINDOW
    )

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

    selection = _random_selection(treatment, pool, n_meters_total=3)

    assert selection.baseline_window == expected
    assert list(selection.exclusions["id"]) == ["t-short"]
    assert sorted(selection.treatment_weights.index) == ["t0", "t1"]
    assert sorted(selection.clusters.index) == ["p0", "p1", "p2"]


# Each pinned selection's recorded window: real runs record the treatment
# baseline union of the 2018 fixture year, the hand-built mixed-frequency
# selection none.
_PINNED_WINDOWS = {
    name: (
        pd.Timestamp("2018-01-01T00:00:00-06:00"),
        pd.Timestamp("2018-12-31T00:00:00-06:00"),
    )
    for name in VARIANT_NAMES
}
_PINNED_WINDOWS["hourly_pool_daily_treatment"] = None


@pytest.mark.parametrize("name", VARIANT_NAMES)
def test_pinned_bank_selection_loads_with_its_window(pinned_models, name):
    selection = ComparisonGroupSelection.from_json(pinned_models["selections"][name])

    assert selection.baseline_window == _PINNED_WINDOWS[name]
