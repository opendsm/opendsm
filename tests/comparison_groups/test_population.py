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
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups import population as population_module
from opendsm.comparison_groups.common import Data_Settings
from opendsm.comparison_groups.common import const as _const
from opendsm.comparison_groups.population import (
    ComparisonPool,
    TreatmentGroup,
    default_data_settings,
)
from opendsm.comparison_groups.warnings import ComparisonGroupWarning
from opendsm.eemeter.common.compliance import Compliance
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
    InfeasibleDataError,
    ModelFitError,
    NonstandardSettingsError,
)
from opendsm.eemeter.common.warnings import EEMeterWarning
from opendsm.eemeter.models import DailyModel
from opendsm.eemeter.models.daily.data import DailyBaselineData

from savings.equivalence_env import shorten_baselines



def _subset(meters, n):
    keys = list(meters)[:n]
    sub = {k: meters[k] for k in keys}

    return sub


def _baseline_map(meters):
    return {mid: entry["baseline_df"] for mid, entry in meters.items()}


def _reporting_map(meters):
    return {mid: entry["reporting_df"] for mid, entry in meters.items()}


@pytest.fixture(scope="module")
def daily_meters(comstock, model_bank):
    """Four real ComStock daily meters on pinned models, each with baseline and
    reporting data."""
    return comstock.meters(model_bank, "daily", comstock.ids("daily")[:4])


@pytest.fixture(scope="module")
def billing_pair(comstock, model_bank):
    """Two real ComStock monthly meters on pinned models, baseline only."""
    return comstock.meters(model_bank, "billing", comstock.ids("billing")[:2], reporting=False)


@pytest.fixture(scope="module")
def hourly_fit(comstock, model_bank):
    """Two real ComStock hourly meters on pinned models: baseline frames and full
    plus first-half reporting frames (for extension tests)."""
    _, df_r = comstock.frames("hourly")
    ids = comstock.ids("hourly")[:2]
    records = comstock.meters(model_bank, "hourly", ids)
    models = {}
    baseline = {}
    full_reporting = {}
    half_reporting = {}

    for mid in ids:
        key = str(mid)
        models[key] = records[key]["model"]
        baseline[key] = records[key]["baseline_df"]
        full_reporting[key] = records[key]["reporting_df"]
        raw = df_r.xs(mid, level="id")
        cutoff = raw.index.min() + pd.Timedelta(days=180)
        half_reporting[key] = raw[raw.index < cutoff].reset_index()

    bundle = {
        "models": models,
        "baseline": baseline,
        "full_reporting": full_reporting,
        "half_reporting": half_reporting,
    }

    return bundle


# -- model frequency dispatch ----------------------------------------------------


def test_model_frequency_dispatch_daily_not_billing(daily_meters):
    group = TreatmentGroup.from_fit_models(_subset(daily_meters, 2))

    assert group.model_frequency == "daily"


def test_model_frequency_dispatch_billing_not_daily(billing_pair):
    """A BillingModel, though a DailyModel subclass, resolves to billing."""
    group = TreatmentGroup.from_fit_models(billing_pair)

    assert group.model_frequency == "billing"


def _payload(meters, mid, serialize="to_dict"):
    """One meter whose model is a ``to_dict``/``to_json`` payload."""
    payload = {
        mid: {
            "model": getattr(meters[mid]["model"], serialize)(),
            "baseline_df": meters[mid]["baseline_df"],
        }
    }

    return payload


def _untagged_payload(meters, mid):
    """A ``to_dict`` payload in the layout written before ``model_choice``."""
    payload = _payload(meters, mid)
    for key in ("schema_version", "compliance", "model_choice"):
        del payload[mid]["model"][key]

    return payload


@pytest.mark.parametrize("serialize", ["to_dict", "to_json"])
def test_from_fit_models_infers_model_frequency_from_payload_model_choice(daily_meters, serialize):
    mid = list(daily_meters)[0]

    rebuilt = TreatmentGroup.from_fit_models(_payload(daily_meters, mid, serialize))

    assert rebuilt.model_frequency == "daily"


def test_from_fit_models_infers_billing_from_payload_model_choice(billing_pair):
    # billing and daily payloads share one layout; model_choice tells them apart
    mid = list(billing_pair)[0]

    rebuilt = TreatmentGroup.from_fit_models(_payload(billing_pair, mid))

    assert rebuilt.model_frequency == "billing"


def test_from_fit_models_requires_model_frequency_for_an_untagged_payload(daily_meters):
    payload = _untagged_payload(daily_meters, list(daily_meters)[0])

    with pytest.raises(ValueError, match="model_frequency is required"):
        TreatmentGroup.from_fit_models(payload)


def test_from_fit_models_builds_an_untagged_payload_at_the_given_model_frequency(daily_meters):
    payload = _untagged_payload(daily_meters, list(daily_meters)[0])

    rebuilt = TreatmentGroup.from_fit_models(payload, model_frequency="daily")

    assert rebuilt.model_frequency == "daily"


def test_from_fit_models_rejects_payload_model_choice_mismatch(daily_meters):
    mid = list(daily_meters)[0]

    with pytest.raises(TypeError, match="does not match"):
        TreatmentGroup.from_fit_models(_payload(daily_meters, mid), model_frequency="billing")


def test_from_fit_models_rejects_unsupported_payload_model_choice(daily_meters):
    mid = list(daily_meters)[0]
    payload = _payload(daily_meters, mid)
    payload[mid]["model"]["model_choice"] = "hourly_caltrack"

    with pytest.raises(ValueError, match="not supported"):
        TreatmentGroup.from_fit_models(payload)


def test_from_fit_models_rejects_mismatched_declared_model_frequency(billing_pair):
    with pytest.raises(TypeError, match="does not match requested"):
        TreatmentGroup.from_fit_models(billing_pair, model_frequency="daily")


# -- population validation ---------------------------------------------------


def test_uniform_fuel_validation_error(daily_meters):
    """The model carries the fuel flag the data object is built with, so two
    models disagreeing on it cannot form one population."""
    ids = list(daily_meters)[:2]
    gas_model = copy.deepcopy(daily_meters[ids[0]]["model"])
    gas_model._is_electricity_data = False
    meters = {
        ids[0]: {"model": gas_model, "baseline_df": daily_meters[ids[0]]["baseline_df"]},
        ids[1]: {
            "model": daily_meters[ids[1]]["model"],
            "baseline_df": daily_meters[ids[1]]["baseline_df"],
        },
    }

    with pytest.raises(ValueError, match="is_electricity_data"):
        TreatmentGroup.from_fit_models(meters, model_frequency="daily")


def _new_york_meter(comstock, daily_meters, mid):
    """A pinned daily meter whose baseline frame is converted to New York time."""
    df_b, _ = comstock.frames("daily")
    raw = df_b.xs(int(mid), level="id")
    raw.index = raw.index.tz_convert("America/New_York")
    meter = {"model": daily_meters[mid]["model"], "baseline_df": raw.reset_index()}

    return meter


def test_uniform_timezone_validation_error(comstock, daily_meters):
    """Meters whose fits and data sit in two timezones cannot form one population."""
    ids = list(daily_meters)[:2]
    new_york = _new_york_meter(comstock, daily_meters, ids[1])
    new_york["model"] = copy.deepcopy(new_york["model"])
    new_york["model"].baseline_timezone = "America/New_York"
    meters = {ids[0]: daily_meters[ids[0]], ids[1]: new_york}

    with pytest.raises(ValueError, match="share one timezone"):
        TreatmentGroup.from_fit_models(meters, model_frequency="daily")


def test_model_and_baseline_data_timezone_mismatch_raises(comstock, daily_meters):
    """A model fit in one timezone paired with baseline data in another names
    the meter and both zones."""
    mid = list(daily_meters)[0]
    meters = {mid: _new_york_meter(comstock, daily_meters, mid)}

    with pytest.raises(ValueError, match=f"Meter {mid}: model timezone America/Chicago"):
        TreatmentGroup.from_fit_models(meters, model_frequency="daily")


def test_disqualified_meter_warns(daily_meters):
    ids = list(daily_meters)[:2]
    flagged = copy.deepcopy(daily_meters[ids[0]]["model"])
    flagged.compliance.record(
        EEMeterWarning(
            qualified_name="eemeter.test",
            description="flagged",
            severity="disqualified",
            member="model",
            data=None,
        )
    )
    meters = {
        ids[0]: {"model": flagged, "baseline_df": daily_meters[ids[0]]["baseline_df"]},
        ids[1]: {
            "model": daily_meters[ids[1]]["model"],
            "baseline_df": daily_meters[ids[1]]["baseline_df"],
        },
    }

    with pytest.warns(ComparisonGroupWarning, match="disqualified meters or baselines"):
        TreatmentGroup.from_fit_models(meters, model_frequency="daily")


def _without_ghi(model):
    """A copy of a pinned hourly model that does not use ghi, standing in for one
    fit with ``train_features`` omitting it. Its settings omit ghi too, so a
    JSON reload keeps it non-solar."""
    non_solar = copy.deepcopy(model)
    non_solar._ts_features = [name for name in non_solar._ts_features if name != "ghi"]
    train_features = [name for name in non_solar.settings.train_features if name != "ghi"]
    non_solar.settings = non_solar.settings.model_copy(update={"train_features": train_features})

    return non_solar


def _hourly_meters(hourly_fit, reporting=None, model_of=lambda model: model):
    """The pinned hourly meters with baseline frames, plus ``reporting`` (a mapping
    of id to reporting frame) when given; ``model_of`` maps each model."""
    meters = {}

    for mid, model in hourly_fit["models"].items():
        meters[mid] = {"model": model_of(model), "baseline_df": hourly_fit["baseline"][mid]}
        if reporting is not None:
            meters[mid]["reporting_df"] = reporting[mid]

    return meters


@pytest.fixture(scope="module")
def hourly_meters(hourly_fit):
    """The pinned hourly meters with baseline data only."""
    return _hourly_meters(hourly_fit)


def test_solar_mix_raises_for_a_model_ignoring_its_ghi_data(hourly_fit):
    """Solar is whether a model uses ghi: a model that ignores the ghi column its
    baseline data carries does not pool with models that use ghi."""
    meters = _hourly_meters(hourly_fit)
    mid = next(iter(meters))
    meters[mid]["model"] = _without_ghi(meters[mid]["model"])

    assert "ghi" in meters[mid]["baseline_df"].columns

    with pytest.raises(ValueError, match="mixes solar"):
        TreatmentGroup.from_fit_models(meters, model_frequency="hourly")


def test_partial_reload_of_models_not_using_ghi_is_not_a_solar_mix(hourly_fit):
    """Hourly meters whose models omit ghi reload as one non-solar population
    whether or not their ghi-carrying baseline data is re-attached."""
    meters = _hourly_meters(hourly_fit, model_of=_without_ghi)
    payload = TreatmentGroup.from_fit_models(meters, model_frequency="hourly").to_json()
    mid = next(iter(meters))

    rebuilt = TreatmentGroup.from_json(payload, baseline={mid: meters[mid]["baseline_df"]})

    assert rebuilt.ids == list(meters)
    assert rebuilt.ids_without_baseline() == list(meters)[1:]


# -- predictions -------------------------------------------------------------


def _assert_matrices_equal(actual, expected):
    """Two ``prediction_matrices`` results hold the same index, ids and arrays."""
    index, ids, *arrays = actual
    expected_index, expected_ids, *expected_arrays = expected

    pd.testing.assert_index_equal(index, expected_index)
    assert ids == expected_ids
    for name, array, expected_array in zip(
        ["observed", "modeled", "modeled_unc", "observed_unc"], arrays, expected_arrays
    ):
        np.testing.assert_array_equal(array, expected_array, err_msg=name)


def test_prediction_frames_schema(daily_meters):
    """Every meter has a frame of ``observed``, ``modeled`` and ``modeled_unc``
    on a tz-aware ``datetime`` index."""
    group = TreatmentGroup.from_fit_models(daily_meters)

    frames = group.prediction_frames("baseline")

    assert list(frames) == group.ids
    for mid, frame in frames.items():
        assert list(frame.columns) == ["observed", "modeled", "modeled_unc"], mid
        assert frame.index.name == "datetime"
        assert str(frame.index.tz) == group.tz


def test_observed_unc_defaults_to_zero(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    mid = group.ids[0]
    index = group.prediction_frames("baseline", ids=[mid])[mid].index

    values = group.observed_unc_at(mid, index)

    np.testing.assert_array_equal(values, np.zeros(len(index)))


def test_prediction_matrices_match_prediction_frames(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    frames = group.prediction_frames("baseline")

    index, ids, observed, modeled, modeled_unc, observed_unc = group.prediction_matrices("baseline")

    assert observed.dtype == np.float32
    assert modeled.dtype == np.float32
    assert modeled_unc.dtype == np.float32
    assert observed_unc is None

    for col, mid in enumerate(ids):
        frame = frames[mid]
        rows = index.get_indexer(frame.index)
        np.testing.assert_allclose(
            observed[rows, col], frame["observed"].to_numpy(dtype=np.float32), rtol=1e-5
        )
        np.testing.assert_allclose(
            modeled[rows, col], frame["modeled"].to_numpy(dtype=np.float32), rtol=1e-5
        )
        np.testing.assert_allclose(
            modeled_unc[rows, col], frame["modeled_unc"].to_numpy(dtype=np.float32), rtol=1e-5
        )


def test_billing_predictions_ride_daily_substrate(billing_pair):
    group = TreatmentGroup.from_fit_models(billing_pair)
    frames = group.prediction_frames("baseline")

    for mid in group.ids:
        substrate = group._meters[mid].baseline_data.df
        n_rows = len(frames[mid])
        n_days = len(substrate)
        n_months = substrate.index.tz_localize(None).to_period("M").nunique()

        assert n_rows == n_days
        assert n_rows > n_months


def test_prediction_matrices_observed_unc_present_when_set(daily_meters):
    meters = {}
    for mid, entry in _subset(daily_meters, 2).items():
        meters[mid] = {**entry, "observed_unc": 3.0}
    group = TreatmentGroup.from_fit_models(meters, model_frequency="daily")

    _, _, _, _, _, observed_unc = group.prediction_matrices("baseline")

    assert observed_unc is not None
    assert observed_unc.dtype == np.float32
    assert np.allclose(observed_unc, 3.0)


def test_observed_unc_series_not_covering_window_raises(daily_meters):
    meters = dict(_subset(daily_meters, 2))
    mid = next(iter(meters))
    baseline_index = pd.DatetimeIndex(meters[mid]["baseline_df"]["datetime"])
    unc_series = pd.Series(2.0, index=baseline_index[: len(baseline_index) // 2])
    meters[mid] = {**meters[mid], "observed_unc": unc_series}
    group = TreatmentGroup.from_fit_models(meters, model_frequency="daily")

    with pytest.raises(ValueError, match="does not cover the requested window"):
        group.prediction_matrices("baseline")


# -- reporting attach --------------------------------------------------------


@pytest.mark.slow
def test_reload_with_cumulative_reporting_matches_full_repredict(hourly_fit):
    """A population saved with partial reporting and reloaded with cumulative
    reporting predicts exactly what one built on the full reporting does."""
    half = TreatmentGroup.from_fit_models(
        _hourly_meters(hourly_fit, hourly_fit["half_reporting"]), model_frequency="hourly"
    )
    half_index = half.prediction_matrices("reporting")[0]

    extended = TreatmentGroup.from_json(
        half.to_json(), baseline=hourly_fit["baseline"], reporting=hourly_fit["full_reporting"]
    )
    extended_predictions = extended.prediction_matrices("reporting")

    fresh = TreatmentGroup.from_fit_models(
        _hourly_meters(hourly_fit, hourly_fit["full_reporting"]), model_frequency="hourly"
    )
    fresh_predictions = fresh.prediction_matrices("reporting")

    assert len(half_index) < len(extended_predictions[0])
    _assert_matrices_equal(extended_predictions, fresh_predictions)


def _new_york_reporting(comstock, mid):
    """A daily meter's reporting frame converted to New York time."""
    _, df_r = comstock.frames("daily")
    raw = df_r.xs(int(mid), level="id")
    raw.index = raw.index.tz_convert("America/New_York")
    frame = raw.reset_index()

    return frame


def test_from_fit_models_reporting_timezone_mismatch_raises(comstock, daily_meters):
    """Reporting data in a zone other than the meter's baseline is rejected at
    construction, naming the meter and both zones."""
    mid = list(daily_meters)[0]
    meters = {mid: {**daily_meters[mid], "reporting_df": _new_york_reporting(comstock, mid)}}

    with pytest.raises(
        ValueError, match=f"Meter {mid}: reporting data timezone America/New_York"
    ):
        TreatmentGroup.from_fit_models(meters, model_frequency="daily")


def test_from_json_reporting_timezone_mismatch_raises(comstock, daily_meters):
    """A reload validates re-attached reporting data like any construction."""
    meters = _subset(daily_meters, 2)
    mid = list(meters)[0]
    reporting = {**_reporting_map(meters), mid: _new_york_reporting(comstock, mid)}
    payload = TreatmentGroup.from_fit_models(meters).to_json()

    with pytest.raises(
        ValueError, match=f"Meter {mid}: reporting data timezone America/New_York"
    ):
        TreatmentGroup.from_json(payload, baseline=_baseline_map(meters), reporting=reporting)


def test_reporting_frames_attached_on_reload_predict(daily_meters):
    """A population built from baseline frames alone predicts reporting once
    reloaded with reporting frames: the model builds each reporting data object
    from its frame."""
    subset = _subset(daily_meters, 2)
    meters = {
        mid: {"model": entry["model"], "baseline_df": entry["baseline_df"]}
        for mid, entry in subset.items()
    }
    group = TreatmentGroup.from_fit_models(meters, model_frequency="daily")

    with pytest.raises(ValueError, match="no reporting data attached"):
        group.prediction_frames("reporting")

    reloaded = TreatmentGroup.from_json(group.to_json(), reporting=_reporting_map(subset))
    frames = reloaded.prediction_frames("reporting")

    assert list(frames) == reloaded.ids
    for frame in frames.values():
        assert np.isfinite(frame["modeled"].to_numpy()).all()


def test_from_fit_models_rejects_a_data_object_where_a_frame_belongs(daily_meters):
    """Populations build the data objects themselves, so a prebuilt data object
    passed as ``baseline_df`` is a TypeError naming the meter."""
    mid = list(daily_meters)[0]
    data = DailyBaselineData(daily_meters[mid]["baseline_df"], is_electricity_data=True)
    meters = {mid: {"model": daily_meters[mid]["model"], "baseline_df": data}}

    with pytest.raises(TypeError, match=f"Meter {mid}: baseline_df must be a pandas DataFrame"):
        TreatmentGroup.from_fit_models(meters, model_frequency="daily")


@pytest.mark.parametrize(
    "value,type_name",
    [(None, "NoneType"), ({"observed": [1.0]}, "dict"), (pd.Series([1.0]), "Series")],
    ids=["none", "dict", "series"],
)
def test_from_data_rejects_a_non_frame_baseline_naming_the_meter(value, type_name):
    """A baseline entry that is not a DataFrame is a TypeError naming the meter and the type."""
    with pytest.raises(
        TypeError, match=f"Meter m1: baseline data must be a pandas DataFrame, got {type_name}"
    ):
        TreatmentGroup.from_data({"m1": value}, "daily", is_electricity_data=True)


# -- loadshape data ----------------------------------------------------------


@pytest.mark.parametrize(
    "meters_fixture, time_period",
    [
        ("daily_meters", _const.TimePeriod.SEASONAL_DAY_OF_WEEK),
        ("billing_pair", _const.TimePeriod.MONTH),
        ("hourly_meters", _const.TimePeriod.SEASONAL_HOURLY_DAY_OF_WEEK),
    ],
    ids=["daily", "billing", "hourly"],
)
def test_loadshape_default_time_period(request, meters_fixture, time_period):
    """Without data settings, loadshapes use the model frequency's time period."""
    group = TreatmentGroup.from_fit_models(request.getfixturevalue(meters_fixture))

    data = group.loadshape_data("modeled")

    assert data.settings.time_period == time_period


def test_default_data_settings_leave_interpolation_off():
    """The default loadshape settings take ``Data_Settings``'s own defaults
    beyond the time period, so a meter missing a time is not filled."""
    settings = default_data_settings("daily")

    assert settings.interpolate_missing is False
    assert settings.model_dump() == Data_Settings(
        time_period=_const.TimePeriod.SEASONAL_DAY_OF_WEEK
    ).model_dump()


def test_default_data_settings_unknown_model_frequency_raises():
    with pytest.raises(ValueError, match="model_frequency must be one of"):
        default_data_settings("weekly")


def test_loadshape_data_carries_caller_data_settings(daily_meters):
    """Interpolation, minimum data fraction, season and weekday/weekend reach ``Data``;
    the loadshape type, pool size and seed stay the population's own."""
    group = TreatmentGroup.from_fit_models(daily_meters)
    caller = Data_Settings(
        agg_type=_const.AggType.MEDIAN,
        time_period=_const.TimePeriod.SEASONAL_DAY_OF_WEEK,
        interpolate_missing=True,
        min_data_pct_required=0.6,
        season={"june": "shoulder", "september": "shoulder"},
        weekday_weekend={"friday": "weekend"},
        max_pool_size=1,
        seed=123,
    )

    settings = group.loadshape_data("modeled", caller).settings

    assert settings.agg_type == _const.AggType.MEDIAN
    assert settings.interpolate_missing is True
    assert settings.min_data_pct_required == 0.6
    assert settings.season.model_dump() == caller.season.model_dump()
    assert settings.weekday_weekend.model_dump() == caller.weekday_weekend.model_dump()
    assert settings.loadshape_type == _const.LoadshapeType.MODELED
    assert settings.max_pool_size == len(group.ids)
    assert settings.seed == 0


def test_loadshape_data_carries_min_data_pct_required(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    caller = Data_Settings(
        time_period=_const.TimePeriod.SEASONAL_DAY_OF_WEEK, min_data_pct_required=0.5
    )

    settings = group.loadshape_data("modeled", caller).settings

    assert settings.min_data_pct_required == 0.5


def test_loadshape_observed_basis_skips_predict(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)

    assert group._baseline_pred == {}
    data = group.loadshape_data("observed")

    assert group._baseline_pred == {}
    assert data.settings.loadshape_type == _const.LoadshapeType.OBSERVED


def test_loadshape_error_basis_guards_zero_modeled(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    group.prediction_frames("baseline")
    first = list(group._baseline_pred)[0]
    modeled_col = group._baseline_pred[first].columns.get_loc("modeled")
    group._baseline_pred[first].iloc[0, modeled_col] = 0.0

    data = group.loadshape_data("error")

    assert not np.isinf(data.loadshape.to_numpy()).any()


def test_short_baseline_months_reports_only_the_short_meter(billing_pair):
    """A meter spanning one baseline month is reported with its month count; a
    full-year meter is not, and the pool keeps both meters and an empty ledger."""
    good_mid, bad_mid = list(billing_pair)
    pool = ComparisonPool.from_fit_models(billing_pair, model_frequency="billing")
    shorten_baselines(pool, [bad_mid])

    short = pool.short_baseline_months()

    assert short == {bad_mid: 1}
    assert pool.ids == [good_mid, bad_mid]
    assert pool.exclusions.empty


def test_loadshape_data_exclude_leaves_the_meter_out_of_loadshapes_and_pool_size(billing_pair):
    good_mid, bad_mid = list(billing_pair)
    pool = ComparisonPool.from_fit_models(billing_pair, model_frequency="billing")

    data = pool.loadshape_data("modeled", exclude=[bad_mid])

    assert list(data.loadshape.index.astype(str)) == [str(good_mid)]
    assert data.settings.max_pool_size == 1
    assert pool.ids == [good_mid, bad_mid]


# -- serialization -----------------------------------------------------------


def test_json_roundtrip_reattaches_data_and_preserves_tz(daily_meters):
    meters = {}
    for mid, entry in daily_meters.items():
        meters[mid] = {**entry, "observed_unc": 1.5}
    group = TreatmentGroup.from_fit_models(meters, model_frequency="daily")

    payload = group.to_json()
    rebuilt = TreatmentGroup.from_json(
        payload,
        baseline=_baseline_map(daily_meters),
        reporting=_reporting_map(daily_meters),
    )

    assert rebuilt.model_frequency == "daily"
    assert rebuilt.tz == group.tz
    assert rebuilt.ids == group.ids
    rebuilt_predictions = rebuilt.prediction_matrices("baseline")
    _, _, _, modeled, _, observed_unc = rebuilt_predictions
    assert (observed_unc[np.isfinite(modeled)] == 1.5).all()
    _assert_matrices_equal(rebuilt_predictions, group.prediction_matrices("baseline"))


def test_json_header_carries_no_window_fields(daily_meters):
    """A population with baseline and reporting attached serializes no coverage
    window: windows belong to selection and correction, never to the population."""
    group = TreatmentGroup.from_fit_models(daily_meters)

    header = json.loads(group.to_json().splitlines()[0])["header"]

    window_keys = [key for key in header if "window" in key]
    assert window_keys == [], f"header carries window fields {window_keys}"


@pytest.mark.parametrize("stored", ["absent", "null"])
def test_from_json_rebuilds_model_payload_missing_fuel_flag(daily_meters, stored):
    """A model payload whose fuel flag is absent or null rebuilds using the population
    header's is_electricity_data, and the rebuilt model records that."""
    subset = _subset(daily_meters, 2)
    group = TreatmentGroup.from_fit_models(subset, model_frequency="daily")
    payload = json.loads(group.to_json())
    for meter in payload["meters"]:
        model_payload = json.loads(meter["model"])
        if stored == "absent":
            del model_payload["info"]["is_electricity_data"]
        else:
            model_payload["info"]["is_electricity_data"] = None
        meter["model"] = json.dumps(model_payload)

    rebuilt = TreatmentGroup.from_json(json.dumps(payload), baseline=_baseline_map(subset))

    assert rebuilt.is_electricity_data == group.is_electricity_data
    for mid in rebuilt.ids:
        names = {w.qualified_name for w in rebuilt._meters[mid].model.compliance.warnings}
        assert "eemeter.serialization.prior_format" in names


def test_ndjson_roundtrip(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)

    text = group.to_ndjson()
    rebuilt = TreatmentGroup.from_ndjson(
        text,
        baseline=_baseline_map(daily_meters),
        reporting=_reporting_map(daily_meters),
    )

    assert len(text.splitlines()) == len(group.ids) + 1
    _assert_matrices_equal(
        rebuilt.prediction_matrices("reporting"), group.prediction_matrices("reporting")
    )


def test_observed_unc_series_roundtrip_preserves_index_tz(daily_meters):
    mid = list(daily_meters)[0]
    baseline_df = daily_meters[mid]["baseline_df"]
    index = pd.DatetimeIndex(baseline_df["datetime"])
    unc = pd.Series(np.arange(len(index), dtype=float), index=index)
    group = TreatmentGroup.from_fit_models(
        {
            mid: {
                "model": daily_meters[mid]["model"],
                "baseline_df": baseline_df,
                "observed_unc": unc,
            }
        },
        model_frequency="daily",
    )

    rebuilt = TreatmentGroup.from_json(group.to_json(), baseline={mid: baseline_df})
    restored = rebuilt._meters[mid].observed_unc

    assert isinstance(restored, pd.Series)
    assert str(restored.index.tz) == group.tz
    np.testing.assert_allclose(restored.reindex(index).to_numpy(), unc.to_numpy())


def test_to_json_refuses_above_ceiling(daily_meters, monkeypatch):
    group = TreatmentGroup.from_fit_models(daily_meters)
    monkeypatch.setattr("opendsm.comparison_groups.population._MAX_JSON_MODELS", 1)

    with pytest.raises(ValueError, match="ceiling"):
        group.to_json()


def test_json_roundtrip_preserves_relaxed_enforce_compliance(daily_meters):
    """A rebuilt population carries the enforcement policy it was built with
    and still predicts on its meters."""
    group = TreatmentGroup.from_fit_models(daily_meters, enforce_compliance=["settings"])

    rebuilt = TreatmentGroup.from_json(
        group.to_json(),
        baseline=_baseline_map(daily_meters),
        reporting=_reporting_map(daily_meters),
    )

    assert rebuilt.enforce_compliance == ["settings"]
    _assert_matrices_equal(
        rebuilt.prediction_matrices("baseline"), group.prediction_matrices("baseline")
    )


@pytest.mark.parametrize(
    "enforce_compliance, error",
    [(["not_a_member"], ValueError), ("settings", TypeError)],
    ids=["unknown_member", "bare_string"],
)
def test_constructor_rejects_a_bad_enforce_compliance(daily_meters, enforce_compliance, error):
    """An ``enforce_compliance`` that names an unknown member or is not a bool,
    list or tuple fails at construction."""
    with pytest.raises(error, match="enforce_compliance"):
        TreatmentGroup.from_fit_models(daily_meters, enforce_compliance=enforce_compliance)


def test_json_roundtrip_keeps_full_enforce_compliance_as_true(daily_meters):
    """``enforce_compliance=True`` is serialized as ``true``, not as a member list, so a
    reloaded population enforces every member known when it is reloaded."""
    group = TreatmentGroup.from_fit_models(daily_meters, enforce_compliance=True)

    rebuilt = TreatmentGroup.from_json(
        group.to_json(),
        baseline=_baseline_map(daily_meters),
        reporting=_reporting_map(daily_meters),
    )

    assert rebuilt.enforce_compliance is True, (
        f"expected True, got {rebuilt.enforce_compliance!r}"
    )


def test_to_ndjson_has_no_model_ceiling(daily_meters, monkeypatch):
    group = TreatmentGroup.from_fit_models(daily_meters)
    monkeypatch.setattr("opendsm.comparison_groups.population._MAX_JSON_MODELS", 1)

    text = group.to_ndjson()

    assert len(text.splitlines()) == len(group.ids) + 1


def test_from_json_rejects_role_mismatch(daily_meters):
    pool_payload = ComparisonPool.from_fit_models(daily_meters).to_json()

    with pytest.raises(ValueError, match="role"):
        TreatmentGroup.from_json(pool_payload, baseline=_baseline_map(daily_meters))


# -- reload without baseline -------------------------------------------------


def test_reload_without_baseline_predicts_reporting_as_with_it(daily_meters):
    """A reload with reporting data only keeps the payload's timezone and fuel
    flag and predicts the reporting period exactly as the full population does."""
    group = TreatmentGroup.from_fit_models(daily_meters)

    rebuilt = TreatmentGroup.from_json(group.to_json(), reporting=_reporting_map(daily_meters))

    assert (rebuilt.tz, rebuilt.is_electricity_data) == (group.tz, group.is_electricity_data)
    _assert_matrices_equal(
        rebuilt.prediction_matrices("reporting"), group.prediction_matrices("reporting")
    )


@pytest.mark.parametrize(
    "work, message",
    [
        (lambda population: population.prediction_frames("baseline"), "no baseline data attached"),
        (lambda population: population.loadshape_data("modeled"), "Loadshapes require baseline"),
    ],
    ids=["baseline_predictions", "loadshapes"],
)
def test_reload_without_baseline_refuses_baseline_work(daily_meters, work, message):
    """Baseline predictions and loadshapes need the baseline data a reload
    may omit, and say so rather than failing on a missing frame."""
    group = TreatmentGroup.from_fit_models(daily_meters)
    rebuilt = TreatmentGroup.from_ndjson(group.to_ndjson())

    with pytest.raises(ValueError, match=message):
        work(rebuilt)


def test_from_fit_models_requires_baseline(daily_meters):
    """``from_fit_models`` rejects a meter without baseline data, with an error naming the
    meter."""
    mid = next(iter(daily_meters))
    meters = {mid: {"model": daily_meters[mid]["model"]}}

    with pytest.raises(ValueError, match=f"Meter {mid}: baseline_df is required"):
        TreatmentGroup.from_fit_models(meters)


def test_from_fit_models_leaves_caller_model_unchanged(daily_meters):
    """Building a meter's data objects from a caller's fitted model leaves every
    attribute of that model as it was."""
    mid = next(iter(daily_meters))
    model = copy.deepcopy(daily_meters[mid]["model"])
    before = dict(vars(model))
    meters = {
        mid: {
            "model": model,
            "baseline_df": daily_meters[mid]["baseline_df"],
            "reporting_df": daily_meters[mid]["reporting_df"],
        }
    }

    TreatmentGroup.from_fit_models(meters)

    after = vars(model)
    assert after.keys() == before.keys(), (
        f"attributes added {sorted(after.keys() - before.keys())}, "
        f"removed {sorted(before.keys() - after.keys())}"
    )
    rebound = sorted(name for name, value in before.items() if after[name] is not value)
    assert rebound == [], f"attributes rebound: {rebound}"


@pytest.mark.parametrize("fmt", ["json", "ndjson"])
def test_reserialize_after_reload_without_baseline_is_lossless(daily_meters, fmt):
    """Models, header, ledger and observed uncertainty survive a reload without
    baseline data unchanged, and the features given to the reload are kept."""
    meters = {}
    for mid, entry in daily_meters.items():
        index = pd.DatetimeIndex(entry["baseline_df"]["datetime"])
        unc = pd.Series(np.arange(len(index), dtype=float), index=index)
        meters[mid] = {**entry, "observed_unc": unc}
    features = pd.DataFrame({"area": np.arange(len(meters), dtype=float)}, index=list(meters))
    pool = ComparisonPool.from_fit_models(meters, features=features, max_pool_size=3, seed=42)
    payload = getattr(pool, f"to_{fmt}")()

    rebuilt = getattr(ComparisonPool, f"from_{fmt}")(payload, features=pool.features)

    assert getattr(rebuilt, f"to_{fmt}")() == payload
    pd.testing.assert_frame_equal(rebuilt.features, pool.features)


# -- comparison pool trim ----------------------------------------------------


def test_comparison_pool_trim_records_excluded(daily_meters):
    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)

    assert len(pool) == 2
    assert list(pool.exclusions.columns) == ["id", "stage", "origin", "reason", "detail"]
    assert len(pool.exclusions) == len(daily_meters) - 2
    assert set(pool.exclusions["id"]).isdisjoint(pool.ids)
    assert (pool.exclusions["stage"] == "population").all()
    assert (pool.exclusions["origin"] == "pool_trim").all()
    assert (
        pool.exclusions["reason"] == "outside the max_pool_size lowest seeded id hashes"
    ).all()


def test_comparison_pool_trim_keeps_the_lowest_seeded_sha256_ids(daily_meters):
    """For seed 42 the pool keeps the two ids with the smallest SHA-256 of
    ``"42:<id>"``, a ranking that is the same in every process."""
    ranked = sorted(daily_meters, key=lambda mid: hashlib.sha256(f"42:{mid}".encode()).digest())

    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)

    assert set(pool.ids) == set(ranked[:2])
    assert set(pool.exclusions["id"]) == set(ranked[2:])


def test_comparison_pool_trim_ignores_input_order(daily_meters):
    """The kept meters depend on the seed, not the input order, and ``ids``
    keeps the input order."""
    forward = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=7)
    reversed_meters = dict(reversed(list(daily_meters.items())))

    backward = ComparisonPool.from_fit_models(reversed_meters, max_pool_size=2, seed=7)

    assert backward.ids == list(reversed(forward.ids))


def test_comparison_pool_trim_ignores_removing_a_trimmed_meter(daily_meters):
    """Dropping a meter the trim excludes leaves the kept meters unchanged."""
    full = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=0)
    dropped = full.exclusions["id"].iloc[0]
    remaining = {mid: entry for mid, entry in daily_meters.items() if mid != dropped}

    smaller = ComparisonPool.from_fit_models(remaining, max_pool_size=2, seed=0)

    assert smaller.ids == full.ids


def test_comparison_pool_omitted_cap_is_the_data_settings_default(daily_meters):
    pool = ComparisonPool.from_fit_models(daily_meters)

    assert pool.max_pool_size == 10000
    assert pool.ids == list(daily_meters)


def test_comparison_pool_cap_none_keeps_every_meter(daily_meters):
    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=None)

    assert pool.ids == list(daily_meters)
    assert pool.exclusions.empty
    assert pool.max_pool_size is None


@pytest.mark.parametrize("fmt", ["json", "ndjson"])
def test_reload_restores_the_saved_meters_ledger_cap_and_seed(daily_meters, fmt):
    """A reload of a trimmed pool keeps every saved meter, the recorded trim
    rows, the cap and the seed."""
    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=3, seed=5)
    payload = getattr(pool, f"to_{fmt}")()

    rebuilt = getattr(ComparisonPool, f"from_{fmt}")(payload)

    assert rebuilt.ids == pool.ids
    pd.testing.assert_frame_equal(rebuilt.exclusions, pool.exclusions)
    assert (rebuilt.max_pool_size, rebuilt.seed) == (3, 5)


def test_constructor_ledger_precedes_the_trim_rows(daily_meters):
    """A ledger given to the constructor is kept, with the pool's trim rows
    appended after it."""
    prior = exclusions.append(
        exclusions.empty_ledger(), ["m0"], "population", "baseline_data", "too short"
    )

    pool = ComparisonPool.from_fit_models(daily_meters, ledger=prior, max_pool_size=2, seed=0)

    assert list(pool.exclusions["origin"]) == ["baseline_data", "pool_trim", "pool_trim"]
    assert pool.exclusions["id"].iloc[0] == "m0"


def test_exclusions_roundtrip_through_json(daily_meters):
    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)

    rebuilt = ComparisonPool.from_json(pool.to_json(), baseline=_baseline_map(daily_meters))

    pd.testing.assert_frame_equal(rebuilt.exclusions, pool.exclusions)
    assert not rebuilt.exclusions.empty


# -- from_data serial fit ----------------------------------------------------


def test_from_data_records_the_baseline_data_fit_built(daily_meters, monkeypatch):
    """Each meter's record carries the baseline data object the real ``fit``
    built, and the baseline data is built once per meter."""
    subset = _subset(daily_meters, 2)
    built = []
    build_baseline_data = DailyModel._build_baseline_data

    def counting_build_baseline_data(self, df, is_electricity_data):
        result = build_baseline_data(self, df, is_electricity_data)
        built.append(result[0])

        return result

    monkeypatch.setattr(DailyModel, "_build_baseline_data", counting_build_baseline_data)
    monkeypatch.setattr(DailyModel, "_fit", lambda self, meter_data: None)
    monkeypatch.setattr(DailyModel, "_check_model_fit", lambda self: None)

    group = TreatmentGroup.from_data(_baseline_map(subset), "daily", is_electricity_data=True)

    assert len(built) == len(subset), f"built {len(built)} baselines for {len(subset)} meters"
    for mid, data in zip(group.ids, built):
        assert group.meter(mid).baseline_data is data, f"meter {mid} carries a rebuilt baseline"


@pytest.mark.slow
def test_from_data_excludes_meter_failing_baseline_sufficiency(_comstock_daily_all):
    """A meter whose baseline is too short to pass sufficiency is excluded and
    recorded with the verbatim eemeter qualified name while the other meters
    fit normally."""
    df_b, _ = _comstock_daily_all
    ids = sorted(df_b.index.get_level_values("id").unique())[:3]
    baseline = {}

    for mid in ids[:2]:
        baseline[str(mid)] = df_b.xs(mid, level="id").reset_index()

    short = df_b.xs(ids[2], level="id").iloc[:60]
    baseline[str(ids[2])] = short.reset_index()

    group = TreatmentGroup.from_data(baseline, "daily", is_electricity_data=True)

    assert set(group.ids) == {str(mid) for mid in ids[:2]}
    row = group.exclusions.set_index("id").loc[str(ids[2])]
    assert row["stage"] == "population"
    assert row["origin"] == "baseline_data"
    assert row["reason"] == "insufficient baseline data"
    assert "eemeter.sufficiency_criteria" in row["detail"]


@pytest.mark.slow
def test_from_data_attaches_reporting_and_ignores_unfit_meters(_comstock_daily_all):
    """Reporting passed to ``from_data`` lands on the fitted meters; reporting
    for a meter excluded at fit is dropped with it."""
    df_b, df_r = _comstock_daily_all
    ids = sorted(df_b.index.get_level_values("id").unique())[:3]
    baseline = {}
    reporting = {}

    for mid in ids:
        raw_b = df_b.xs(mid, level="id")
        if mid == ids[2]:
            raw_b = raw_b.iloc[:60]
        baseline[str(mid)] = raw_b.reset_index()
        reporting[str(mid)] = df_r.xs(mid, level="id").reset_index()

    group = TreatmentGroup.from_data(
        baseline, "daily", is_electricity_data=True, reporting=reporting
    )

    assert set(group.ids) == {str(mid) for mid in ids[:2]}
    for mid in group.ids:
        assert len(group._meters[mid].reporting_data.df) == len(reporting[mid])
    assert list(group.prediction_frames("reporting")) == group.ids


def _pinned_fit(monkeypatch, meters, fail=(), disqualify=()):
    """Replace ``DailyModel._fit_baseline`` with a copy of each meter's pinned model, so
    ``from_data`` builds real baseline data but runs without fitting. Returns the ids fit, in
    call order; meters in ``fail`` raise ``DataSufficiencyError``, and meters in
    ``disqualify`` raise ``DisqualifiedModelError`` carrying one disqualification record."""
    by_baseline = {id(entry["baseline_df"]): mid for mid, entry in meters.items()}
    by_data = {}
    fitted = []
    build = DailyModel._build_baseline_data

    def build_baseline_data(self, df, is_electricity_data):
        built = build(self, df, is_electricity_data)
        by_data[id(built[0])] = by_baseline[id(df)]

        return built

    def fit_baseline(
        self, baseline, n_leading, n_trailing, is_electricity_data, enforce_compliance
    ):
        mid = by_data[id(baseline)]
        fitted.append(mid)
        if mid in fail:
            raise DataSufficiencyError("baseline too short")

        vars(self).update(vars(meters[mid]["model"]))

        if mid in disqualify:
            flagged = EEMeterWarning(
                qualified_name="eemeter.test",
                description="flagged",
                severity="disqualified",
                member="model",
                data=None,
            )
            raise DisqualifiedModelError("model disqualified", entries=[flagged])

    monkeypatch.setattr(DailyModel, "_build_baseline_data", build_baseline_data)
    monkeypatch.setattr(DailyModel, "_fit_baseline", fit_baseline)

    return fitted


def test_from_data_unknown_reporting_id_raises_before_fitting(daily_meters, monkeypatch):
    fitted = _pinned_fit(monkeypatch, daily_meters)
    stray = next(iter(daily_meters.values()))["reporting_df"]

    with pytest.raises(KeyError, match="not in baseline: \\['does-not-exist'\\]"):
        TreatmentGroup.from_data(
            _baseline_map(daily_meters),
            "daily",
            is_electricity_data=True,
            reporting={"does-not-exist": stray},
        )

    assert fitted == []


def test_from_data_reporting_timezone_mismatch_raises(comstock, daily_meters, monkeypatch):
    _pinned_fit(monkeypatch, daily_meters)
    mid = list(daily_meters)[0]
    reporting = {mid: _new_york_reporting(comstock, mid)}

    with pytest.raises(
        ValueError, match=f"Meter {mid}: reporting data timezone America/New_York"
    ):
        TreatmentGroup.from_data(
            _baseline_map(daily_meters), "daily", is_electricity_data=True, reporting=reporting
        )


def test_from_data_ledgers_a_disqualified_model_and_keeps_the_rest(daily_meters, monkeypatch):
    """A meter whose fit raises ``DisqualifiedModelError`` is one population-stage
    ``model`` row carrying the error's records; the other meters fit."""
    disqualified = list(daily_meters)[1]
    _pinned_fit(monkeypatch, daily_meters, disqualify={disqualified})

    group = TreatmentGroup.from_data(_baseline_map(daily_meters), "daily", is_electricity_data=True)

    assert group.ids == [mid for mid in daily_meters if mid != disqualified]
    assert group.exclusions.to_dict("records") == [
        {
            "id": disqualified,
            "stage": "population",
            "origin": "model",
            "reason": "model disqualified",
            "detail": "eemeter.test: flagged",
        }
    ]


@pytest.mark.parametrize("max_pool_size, n_fit", [(2, 2), (4, 4), (10, 4)])
def test_pool_from_data_fits_only_the_kept_meters(daily_meters, monkeypatch, max_pool_size, n_fit):
    """``from_data`` fits ``min(max_pool_size, len(baseline))`` meters, the ones
    the pool keeps, and ledgers the rest as ``pool_trim`` unfit."""
    fitted = _pinned_fit(monkeypatch, daily_meters)

    pool = ComparisonPool.from_data(
        _baseline_map(daily_meters),
        "daily",
        is_electricity_data=True,
        max_pool_size=max_pool_size,
        seed=0,
    )

    assert len(fitted) == n_fit, f"fit {fitted}, expected {n_fit} fits"
    assert fitted == pool.ids
    assert sorted(pool.exclusions["id"]) == sorted(set(daily_meters) - set(fitted))
    assert (pool.exclusions["origin"] == "pool_trim").all()


def test_pool_from_data_keeps_the_ids_from_fit_models_keeps(daily_meters, monkeypatch):
    _pinned_fit(monkeypatch, daily_meters)
    reference = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=3)

    pool = ComparisonPool.from_data(
        _baseline_map(daily_meters), "daily", is_electricity_data=True, max_pool_size=2, seed=3
    )

    assert pool.ids == reference.ids
    pd.testing.assert_frame_equal(pool.exclusions, reference.exclusions)


def test_pool_from_data_does_not_replace_a_kept_meter_failing_to_fit(daily_meters, monkeypatch):
    """A kept meter that fails to fit is ledgered and not replaced by a trimmed
    one, so the pool ends below the cap."""
    kept = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=0).ids
    fitted = _pinned_fit(monkeypatch, daily_meters, fail={kept[0]})

    pool = ComparisonPool.from_data(
        _baseline_map(daily_meters), "daily", is_electricity_data=True, max_pool_size=2, seed=0
    )

    assert fitted == kept
    assert pool.ids == kept[1:]
    origins = pool.exclusions.set_index("id")["origin"]
    assert origins[kept[0]] == "baseline_data"
    assert sorted(origins[origins == "pool_trim"].index) == sorted(set(daily_meters) - set(kept))


def test_pool_from_data_omitted_cap_is_the_data_settings_default(daily_meters, monkeypatch):
    fitted = _pinned_fit(monkeypatch, daily_meters)

    pool = ComparisonPool.from_data(_baseline_map(daily_meters), "daily", is_electricity_data=True)

    assert pool.max_pool_size == 10000
    assert fitted == list(daily_meters)


def test_pool_from_data_drops_reporting_of_trimmed_meters(daily_meters, monkeypatch):
    """Reporting for a trimmed meter is dropped with it rather than raising as
    an id outside ``baseline``."""
    _pinned_fit(monkeypatch, daily_meters)

    pool = ComparisonPool.from_data(
        _baseline_map(daily_meters),
        "daily",
        is_electricity_data=True,
        max_pool_size=2,
        seed=0,
        reporting=_reporting_map(daily_meters),
    )

    for mid in pool.ids:
        reporting = pool._meters[mid].reporting_data
        assert len(reporting.df) == len(daily_meters[mid]["reporting_df"])


def test_from_data_all_meters_failing_raises(_comstock_daily_all):
    df_b, _ = _comstock_daily_all
    mid = sorted(df_b.index.get_level_values("id").unique())[0]
    short = df_b.xs(mid, level="id").iloc[:60]
    baseline = {str(mid): short.reset_index()}

    with pytest.raises(ValueError, match="All meters failed to fit"):
        TreatmentGroup.from_data(baseline, "daily", is_electricity_data=True)


# -- fit-failure ledger reasons ----------------------------------------------

_STUB_TZ = "America/Chicago"

_LEDGER_ENTRY = EEMeterWarning(
    qualified_name="eemeter.test.check",
    description="ledger detail",
    severity="advisory",
    data=None,
)

_FAILING_ROWS = 5


def _stub_frame(n_rows):
    index = pd.date_range("2023-01-01", periods=n_rows, freq="D", tz=_STUB_TZ)
    frame = pd.DataFrame({"datetime": index, "temperature": 50.0, "observed": 10.0})

    return frame


class _StubData:
    """The minimum a built data object owes a population."""

    def __init__(self, df):
        self.df = df.set_index("datetime")
        self.tz = self.df.index.tz
        self.compliance = Compliance()


def _stub_model_cls(failure):
    """A daily-model stand-in whose fit raises ``failure`` on the short frame and
    succeeds on any other, so one meter lands on the ledger and one survives."""

    class StubModel:
        def __init__(self, settings=None):
            self.compliance = Compliance()
            self.baseline_timezone = _STUB_TZ
            self.is_electricity_data = None

        def _build_baseline_data(self, df, is_electricity_data):
            return _StubData(df), 0, 0

        def _fit_baseline(
            self, baseline, n_leading, n_trailing, is_electricity_data, enforce_compliance
        ):
            if len(baseline.df) == _FAILING_ROWS:
                raise failure

            self.is_electricity_data = bool(is_electricity_data)

    return StubModel


def _fit_one_failing_meter(monkeypatch, failure):
    monkeypatch.setitem(population_module._MODEL_CLASSES, "daily", _stub_model_cls(failure))
    baseline = {"ok": _stub_frame(400), "bad": _stub_frame(_FAILING_ROWS)}

    group = TreatmentGroup.from_data(baseline, "daily", is_electricity_data=True)

    return group


@pytest.mark.parametrize(
    "failure, origin, reason",
    [
        (
            DataSufficiencyError("insufficient", entries=[_LEDGER_ENTRY]),
            "baseline_data",
            "insufficient baseline data",
        ),
        (
            InfeasibleDataError("no model fits", entries=[_LEDGER_ENTRY]),
            "baseline_data",
            "no feasible model for the baseline data",
        ),
        (
            DisqualifiedModelError("disqualified", entries=[_LEDGER_ENTRY]),
            "model",
            "model disqualified",
        ),
        (
            NonstandardSettingsError("nonstandard", entries=[_LEDGER_ENTRY]),
            "settings",
            "nonstandard settings",
        ),
    ],
    ids=["sufficiency", "infeasible", "disqualified_model", "nonstandard_settings"],
)
def test_from_data_ledger_names_the_compliance_failure(monkeypatch, failure, origin, reason):
    """A compliance failure at fit is one population-stage row whose origin and
    reason name the failing part, with the error's records as detail."""
    group = _fit_one_failing_meter(monkeypatch, failure)

    assert group.ids == ["ok"]
    row = group.exclusions.set_index("id").loc["bad"]
    assert row["stage"] == "population"
    assert row["origin"] == origin
    assert row["reason"] == reason
    assert row["detail"] == "eemeter.test.check: ledger detail"


def test_from_data_ledger_records_a_model_fit_error(monkeypatch):
    """A ModelFitError carries no compliance entries, so its message is the detail."""
    failure = ModelFitError("solver did not converge")

    group = _fit_one_failing_meter(monkeypatch, failure)

    row = group.exclusions.set_index("id").loc["bad"]
    assert row["origin"] == "model"
    assert row["reason"] == "model fit failed"
    assert row["detail"] == "solver did not converge"


# -- relaxed enforcement -----------------------------------------------------


def _disqualified_meter(daily_meters, member="model"):
    mid = list(daily_meters)[0]
    flagged = copy.deepcopy(daily_meters[mid]["model"])
    flagged.compliance.record(
        EEMeterWarning(
            qualified_name="eemeter.test.flagged",
            description="flagged",
            severity="disqualified",
            member=member,
            data=None,
        )
    )
    meters = {mid: {"model": flagged, "baseline_df": daily_meters[mid]["baseline_df"]}}

    return meters


def _population_on_disqualified_meter(meters, enforce_compliance):
    # construction warns about the disqualified meter, which the suite would
    # otherwise raise
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ComparisonGroupWarning)
        group = TreatmentGroup.from_fit_models(
            meters, model_frequency="daily", enforce_compliance=enforce_compliance
        )

    return group


def test_relaxed_enforcement_predicts_on_a_disqualified_meter(daily_meters):
    """A population whose enforcement list leaves out the member a meter is
    disqualified on predicts on that meter."""
    meters = _disqualified_meter(daily_meters, member="data")

    group = _population_on_disqualified_meter(meters, ["settings"])

    frames = group.prediction_frames("baseline")
    assert all(len(frame) > 0 for frame in frames.values())


def test_full_enforcement_refuses_to_predict_on_a_disqualified_meter(daily_meters):
    """A meter disqualified on a member other than ``model`` still blocks prediction
    under full enforcement."""
    meters = _disqualified_meter(daily_meters, member="data")

    group = _population_on_disqualified_meter(meters, True)

    with pytest.raises(DisqualifiedModelError):
        group.prediction_frames("baseline")


def test_model_fit_metrics_disqualification_does_not_block_prediction(daily_meters):
    """A meter admitted despite a failing ``model`` fit-metrics compliance record
    (`_validate_meters` warns and keeps it) still predicts under the default,
    fully-enforced ``enforce_compliance``."""
    meters = _disqualified_meter(daily_meters, member="model")

    group = _population_on_disqualified_meter(meters, True)

    frames = group.prediction_frames("baseline")
    assert all(len(frame) > 0 for frame in frames.values())
