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

import numpy as np
import pandas as pd
import pytest

from opendsm.comparison_groups.common import const as _const
from opendsm.comparison_groups.population import (
    ComparisonPool,
    TreatmentGroup,
)
from opendsm.comparison_groups.selection import select_comparison_group
from opendsm.eemeter.common.exceptions import DataSufficiencyError
from opendsm.eemeter.common.warnings import EEMeterWarning
from opendsm.eemeter.models import (
    DailyBaselineData,
    DailyModel,
    DailyReportingData,
    HourlyReportingData,
)



def _subset(meters, n):
    keys = list(meters)[:n]
    sub = {k: meters[k] for k in keys}

    return sub


def _baseline_map(meters):
    return {mid: entry["baseline_data"] for mid, entry in meters.items()}


def _reporting_map(meters):
    return {mid: entry["reporting_data"] for mid, entry in meters.items()}


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
    """Two real ComStock hourly meters on pinned models: baseline data and full
    plus first-half reporting data (for extension tests)."""
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
        baseline[key] = records[key]["baseline_data"]
        full_reporting[key] = records[key]["reporting_data"]
        raw = df_r.xs(mid, level="id")
        cutoff = raw.index.min() + pd.Timedelta(days=180)
        half = raw[raw.index < cutoff]
        half_reporting[key] = HourlyReportingData(half.reset_index(), is_electricity_data=True)

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
    # BillingModel subclasses DailyModel, so isinstance would misclassify; the
    # exact-type check must resolve to billing when the model frequency comes
    # from the instances.
    group = TreatmentGroup.from_fit_models(billing_pair)

    assert group.model_frequency == "billing"


@pytest.mark.parametrize("serialize", ["to_dict", "to_json"])
def test_from_fit_models_requires_model_frequency_for_payloads(daily_meters, serialize):
    mid = list(daily_meters)[0]
    payload = {
        mid: {
            "model": getattr(daily_meters[mid]["model"], serialize)(),
            "baseline_data": daily_meters[mid]["baseline_data"],
        }
    }

    with pytest.raises(ValueError, match="model_frequency is required"):
        TreatmentGroup.from_fit_models(payload)

    rebuilt = TreatmentGroup.from_fit_models(payload, model_frequency="daily")

    assert rebuilt.model_frequency == "daily"


def test_from_fit_models_rejects_mismatched_declared_model_frequency(billing_pair):
    with pytest.raises(TypeError, match="does not match requested"):
        TreatmentGroup.from_fit_models(billing_pair, model_frequency="daily")


# -- population validation ---------------------------------------------------


def test_uniform_fuel_validation_error(daily_meters):
    ids = list(daily_meters)[:2]
    flipped = copy.deepcopy(daily_meters[ids[0]]["baseline_data"])
    flipped.is_electricity_data = False
    meters = {
        ids[0]: {"model": daily_meters[ids[0]]["model"], "baseline_data": flipped},
        ids[1]: {
            "model": daily_meters[ids[1]]["model"],
            "baseline_data": daily_meters[ids[1]]["baseline_data"],
        },
    }

    with pytest.raises(ValueError, match="is_electricity_data"):
        TreatmentGroup.from_fit_models(meters, model_frequency="daily")


def _new_york_meter(comstock, daily_meters, mid):
    """A pinned daily meter whose baseline data is converted to New York time."""
    df_b, _ = comstock.frames("daily")
    raw = df_b.xs(int(mid), level="id")
    raw.index = raw.index.tz_convert("America/New_York")
    data = DailyBaselineData(raw.reset_index(), is_electricity_data=True)
    meter = {"model": daily_meters[mid]["model"], "baseline_data": data}

    return meter


def test_uniform_timezone_validation_error(comstock, daily_meters):
    """Meters whose fits and data sit in two timezones cannot form one population."""
    ids = list(daily_meters)[:2]
    new_york = _new_york_meter(comstock, daily_meters, ids[1])
    new_york["model"] = copy.deepcopy(new_york["model"])
    new_york["model"].baseline_timezone = new_york["baseline_data"].tz
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
    flagged = copy.deepcopy(daily_meters[ids[0]]["baseline_data"])
    flagged.disqualification = [
        EEMeterWarning(qualified_name="eemeter.test", description="flagged", data=None)
    ]
    meters = {
        ids[0]: {"model": daily_meters[ids[0]]["model"], "baseline_data": flagged},
        ids[1]: {
            "model": daily_meters[ids[1]]["model"],
            "baseline_data": daily_meters[ids[1]]["baseline_data"],
        },
    }

    with pytest.warns(UserWarning, match="disqualified"):
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


def _hourly_baseline_meters(hourly_fit, model_of=lambda model: model):
    meters = {
        mid: {"model": model_of(model), "baseline_data": hourly_fit["baseline"][mid]}
        for mid, model in hourly_fit["models"].items()
    }

    return meters


def test_solar_mix_raises_for_a_model_ignoring_its_ghi_data(hourly_fit):
    """Solar is whether a model uses ghi: a model that ignores the ghi column its
    baseline data carries does not pool with models that use ghi."""
    meters = _hourly_baseline_meters(hourly_fit)
    mid = next(iter(meters))
    meters[mid]["model"] = _without_ghi(meters[mid]["model"])

    assert "ghi" in meters[mid]["baseline_data"].df.columns

    with pytest.raises(ValueError, match="mixes solar"):
        TreatmentGroup.from_fit_models(meters, model_frequency="hourly")


def test_partial_reload_of_models_not_using_ghi_is_not_a_solar_mix(hourly_fit):
    """Hourly meters whose models omit ghi reload as one non-solar population
    whether or not their ghi-carrying baseline data is re-attached."""
    meters = _hourly_baseline_meters(hourly_fit, model_of=_without_ghi)
    payload = TreatmentGroup.from_fit_models(meters, model_frequency="hourly").to_json()
    mid = next(iter(meters))

    rebuilt = TreatmentGroup.from_json(payload, baseline={mid: meters[mid]["baseline_data"]})

    assert rebuilt.ids == list(meters)
    assert rebuilt._ids_without_baseline() == list(meters)[1:]


# -- predictions -------------------------------------------------------------


def test_predictions_long_frame_schema(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    frame = group.predictions("baseline")

    assert list(frame.columns) == [
        "id",
        "datetime",
        "observed",
        "modeled",
        "modeled_unc",
        "observed_unc",
    ]
    assert set(frame["id"].unique()) == set(group.ids)
    assert (frame["observed_unc"] == 0.0).all()


def test_predictions_matches_prediction_matrices(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    long = group.predictions("baseline")
    index, ids, observed, modeled, modeled_unc, observed_unc = group._prediction_matrices("baseline")

    assert observed.dtype == np.float32
    assert modeled.dtype == np.float32
    assert modeled_unc.dtype == np.float32
    assert observed_unc is None

    for col, mid in enumerate(ids):
        sub = long[long["id"] == mid].set_index("datetime")
        rows = index.get_indexer(sub.index)
        np.testing.assert_allclose(
            observed[rows, col], sub["observed"].to_numpy(dtype=np.float32), rtol=1e-5
        )
        np.testing.assert_allclose(
            modeled[rows, col], sub["modeled"].to_numpy(dtype=np.float32), rtol=1e-5
        )
        np.testing.assert_allclose(
            modeled_unc[rows, col], sub["modeled_unc"].to_numpy(dtype=np.float32), rtol=1e-5
        )


def test_billing_predictions_ride_daily_substrate(billing_pair):
    group = TreatmentGroup.from_fit_models(billing_pair)
    frame = group.predictions("baseline")

    for mid in group.ids:
        n_rows = len(frame[frame["id"] == mid])
        n_days = len(billing_pair[mid]["baseline_data"].df)
        n_months = billing_pair[mid]["baseline_data"].df.index.tz_localize(None).to_period(
            "M"
        ).nunique()

        assert n_rows == n_days
        assert n_rows > n_months


def test_prediction_matrices_observed_unc_present_when_set(daily_meters):
    meters = {}
    for mid, entry in _subset(daily_meters, 2).items():
        meters[mid] = {**entry, "observed_unc": 3.0}
    group = TreatmentGroup.from_fit_models(meters, model_frequency="daily")

    _, _, _, _, _, observed_unc = group._prediction_matrices("baseline")

    assert observed_unc is not None
    assert observed_unc.dtype == np.float32
    assert np.allclose(observed_unc, 3.0)


def test_observed_unc_series_not_covering_window_raises(daily_meters):
    meters = dict(_subset(daily_meters, 2))
    mid = next(iter(meters))
    baseline_index = meters[mid]["baseline_data"].df.index
    unc_series = pd.Series(2.0, index=baseline_index[: len(baseline_index) // 2])
    meters[mid] = {**meters[mid], "observed_unc": unc_series}
    group = TreatmentGroup.from_fit_models(meters, model_frequency="daily")

    with pytest.raises(ValueError, match="does not cover the requested window"):
        group._prediction_matrices("baseline")


# -- reporting attach --------------------------------------------------------


def _hourly_meters(hourly_fit, reporting):
    meters = {
        mid: {
            "model": model,
            "baseline_data": hourly_fit["baseline"][mid],
            "reporting_data": reporting[mid],
        }
        for mid, model in hourly_fit["models"].items()
    }

    return meters


@pytest.mark.slow
def test_reload_with_cumulative_reporting_matches_full_repredict(hourly_fit):
    """A population saved with partial reporting and reloaded with cumulative
    reporting predicts exactly what one built on the full reporting does."""
    half = TreatmentGroup.from_fit_models(
        _hourly_meters(hourly_fit, hourly_fit["half_reporting"]), model_frequency="hourly"
    )
    half_predictions = half.predictions("reporting")

    extended = TreatmentGroup.from_json(
        half.to_json(), baseline=hourly_fit["baseline"], reporting=hourly_fit["full_reporting"]
    )
    extended_predictions = extended.predictions("reporting")

    fresh = TreatmentGroup.from_fit_models(
        _hourly_meters(hourly_fit, hourly_fit["full_reporting"]), model_frequency="hourly"
    )
    fresh_predictions = fresh.predictions("reporting")

    assert len(half_predictions) < len(extended_predictions)
    pd.testing.assert_frame_equal(extended_predictions, fresh_predictions)


def _new_york_reporting(comstock, mid):
    """A daily meter's reporting data converted to New York time."""
    _, df_r = comstock.frames("daily")
    raw = df_r.xs(int(mid), level="id")
    raw.index = raw.index.tz_convert("America/New_York")
    data = DailyReportingData(raw.reset_index(), is_electricity_data=True)

    return data


def test_from_fit_models_reporting_timezone_mismatch_raises(comstock, daily_meters):
    """Reporting data in a zone other than the meter's baseline is rejected at
    construction, naming the meter and both zones."""
    mid = list(daily_meters)[0]
    meters = {mid: {**daily_meters[mid], "reporting_data": _new_york_reporting(comstock, mid)}}

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


# -- loadshape data ----------------------------------------------------------


def test_loadshape_default_time_period_daily(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    data = group.loadshape_data("modeled")

    assert data.settings.time_period == _const.TimePeriod.SEASONAL_DAY_OF_WEEK
    assert data.settings.loadshape_type == _const.LoadshapeType.MODELED
    assert data.settings.max_pool_size == len(group.ids)


def test_loadshape_default_time_period_billing(billing_pair):
    group = TreatmentGroup.from_fit_models(billing_pair)
    data = group.loadshape_data("modeled")

    assert data.settings.time_period == _const.TimePeriod.MONTH


def test_loadshape_default_time_period_hourly(hourly_fit):
    meters = {
        mid: {"model": model, "baseline_data": hourly_fit["baseline"][mid]}
        for mid, model in hourly_fit["models"].items()
    }
    group = TreatmentGroup.from_fit_models(meters, model_frequency="hourly")
    data = group.loadshape_data("modeled")

    assert data.settings.time_period == _const.TimePeriod.SEASONAL_HOURLY_DAY_OF_WEEK


def test_loadshape_observed_basis_skips_predict(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)

    assert group._baseline_pred == {}
    data = group.loadshape_data("observed")

    assert group._baseline_pred == {}
    assert data.settings.loadshape_type == _const.LoadshapeType.OBSERVED


def test_loadshape_error_basis_guards_zero_modeled(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    group._ensure_pred("baseline")
    first = list(group._baseline_pred)[0]
    modeled_col = group._baseline_pred[first].columns.get_loc("modeled")
    group._baseline_pred[first].iloc[0, modeled_col] = 0.0

    data = group.loadshape_data("error")

    assert not np.isinf(data.loadshape.to_numpy()).any()


def _short_baseline(entry):
    """A copy of ``entry`` whose baseline keeps only its first six daily rows,
    so it spans a single month."""
    baseline_data = copy.deepcopy(entry["baseline_data"])
    baseline_data._df = baseline_data._df.head(6)
    short = {"model": entry["model"], "baseline_data": baseline_data}

    return short


def test_billing_loadshape_requires_full_year(billing_pair):
    """Selection, not ``loadshape_data``, rejects a treatment population in which
    every meter is short of 12 baseline months, naming each meter; the
    population is left unchanged."""
    truncated = {mid: _short_baseline(entry) for mid, entry in billing_pair.items()}
    group = TreatmentGroup.from_fit_models(truncated, model_frequency="billing")
    pool = ComparisonPool.from_fit_models(billing_pair, model_frequency="billing")

    with pytest.raises(ValueError, match="at least 12 baseline months") as excinfo:
        select_comparison_group(group, pool, method="random_sampling")

    for mid in billing_pair:
        assert f"{mid}: baseline months: 1" in str(excinfo.value)
    assert group.ids == list(billing_pair)
    assert group.exclusions.empty


def test_billing_pool_short_baseline_excluded_not_raised(billing_pair):
    """The short-baseline helper reports the offender and its month count
    without touching the pool, and ``loadshape_data(exclude=)`` leaves the
    offender out of the loadshapes and the pool size."""
    good_mid, good_entry = list(billing_pair.items())[0]
    bad_mid, bad_entry = list(billing_pair.items())[1]
    meters = {good_mid: good_entry, bad_mid: _short_baseline(bad_entry)}
    pool = ComparisonPool.from_fit_models(meters, model_frequency="billing")

    short = pool._short_baseline_months()
    data = pool.loadshape_data("modeled", exclude=list(short))

    assert short == {bad_mid: 1}
    assert pool.ids == [good_mid, bad_mid]
    assert pool.exclusions.empty
    assert list(data.loadshape.index.astype(str)) == [str(good_mid)]
    assert data.settings.time_period == _const.TimePeriod.MONTH
    assert data.settings.max_pool_size == 1


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
    assert (rebuilt.predictions("baseline")["observed_unc"] == 1.5).all()
    pd.testing.assert_frame_equal(group.predictions("baseline"), rebuilt.predictions("baseline"))


def test_json_header_carries_no_window_fields(daily_meters):
    """A population with baseline and reporting attached serializes no coverage
    window: windows belong to selection and correction, never to the population."""
    group = TreatmentGroup.from_fit_models(daily_meters)

    header = json.loads(group.to_json().splitlines()[0])["header"]

    window_keys = [key for key in header if "window" in key]
    assert window_keys == [], f"header carries window fields {window_keys}"


def test_ndjson_roundtrip(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)

    text = group.to_ndjson()
    rebuilt = TreatmentGroup.from_ndjson(
        text,
        baseline=_baseline_map(daily_meters),
        reporting=_reporting_map(daily_meters),
    )

    assert len(text.splitlines()) == len(group.ids) + 1
    pd.testing.assert_frame_equal(group.predictions("reporting"), rebuilt.predictions("reporting"))


def test_observed_unc_series_roundtrip_preserves_index_tz(daily_meters):
    mid = list(daily_meters)[0]
    baseline_data = daily_meters[mid]["baseline_data"]
    index = baseline_data.df.index
    unc = pd.Series(np.arange(len(index), dtype=float), index=index)
    group = TreatmentGroup.from_fit_models(
        {mid: {"model": daily_meters[mid]["model"], "baseline_data": baseline_data, "observed_unc": unc}},
        model_frequency="daily",
    )

    rebuilt = TreatmentGroup.from_json(group.to_json(), baseline={mid: baseline_data})
    restored = rebuilt._meters[mid].observed_unc

    assert isinstance(restored, pd.Series)
    assert str(restored.index.tz) == group.tz
    np.testing.assert_allclose(restored.reindex(index).to_numpy(), unc.to_numpy())


def test_to_json_refuses_above_ceiling(daily_meters, monkeypatch):
    group = TreatmentGroup.from_fit_models(daily_meters)
    monkeypatch.setattr("opendsm.comparison_groups.population._MAX_JSON_MODELS", 1)

    with pytest.raises(ValueError, match="ceiling"):
        group.to_json()

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
    pd.testing.assert_frame_equal(rebuilt.predictions("reporting"), group.predictions("reporting"))


def test_reload_without_baseline_refuses_baseline_work(daily_meters):
    """Baseline predictions and loadshapes need the baseline data a reload
    may omit, and say so rather than failing on a missing frame."""
    group = TreatmentGroup.from_fit_models(daily_meters)
    rebuilt = TreatmentGroup.from_ndjson(group.to_ndjson())

    with pytest.raises(ValueError, match="no baseline data attached"):
        rebuilt.predictions("baseline")

    with pytest.raises(ValueError, match="Loadshapes require baseline data"):
        rebuilt.loadshape_data("modeled")


def test_from_fit_models_requires_baseline(daily_meters):
    """Only a reload may omit baseline data."""
    mid = next(iter(daily_meters))
    meters = {mid: {"model": daily_meters[mid]["model"]}}

    with pytest.raises(ValueError, match=f"Meter {mid}: baseline_data is required"):
        TreatmentGroup.from_fit_models(meters)


@pytest.mark.parametrize("fmt", ["json", "ndjson"])
def test_reserialize_after_reload_without_baseline_is_lossless(daily_meters, fmt):
    """Models, header, ledger and observed uncertainty survive a reload without
    baseline data unchanged, and the features given to the reload are kept."""
    meters = {}
    for mid, entry in daily_meters.items():
        index = entry["baseline_data"].df.index
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


def test_comparison_pool_trim_is_deterministic(daily_meters):
    first = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)
    second = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)

    assert first.ids == second.ids
    assert list(first.exclusions["id"]) == list(second.exclusions["id"])


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


def test_comparison_pool_omitted_cap_reads_the_module_default(daily_meters, monkeypatch):
    monkeypatch.setattr("opendsm.comparison_groups.population._MAX_POOL_SIZE", 2)

    pool = ComparisonPool.from_fit_models(daily_meters)

    assert len(pool) == 2
    assert pool.max_pool_size == 2


def test_comparison_pool_cap_none_keeps_every_meter(daily_meters, monkeypatch):
    monkeypatch.setattr("opendsm.comparison_groups.population._MAX_POOL_SIZE", 2)

    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=None)

    assert pool.ids == list(daily_meters)
    assert pool.exclusions.empty
    assert pool.max_pool_size is None


@pytest.mark.parametrize("fmt", ["json", "ndjson"])
def test_reload_above_the_cap_keeps_every_meter_and_the_recorded_trim(
    daily_meters, monkeypatch, fmt
):
    """A reload restores the saved meters, ledger, cap and seed and never trims,
    even when the saved pool exceeds the current default cap."""
    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=3, seed=5)
    payload = getattr(pool, f"to_{fmt}")()
    monkeypatch.setattr("opendsm.comparison_groups.population._MAX_POOL_SIZE", 1)

    rebuilt = getattr(ComparisonPool, f"from_{fmt}")(payload)

    assert rebuilt.ids == pool.ids
    pd.testing.assert_frame_equal(rebuilt.exclusions, pool.exclusions)
    assert (rebuilt.max_pool_size, rebuilt.seed) == (3, 5)


@pytest.mark.parametrize("fmt", ["json", "ndjson"])
@pytest.mark.parametrize("kwarg", [{"max_pool_size": 2}, {"seed": 1}])
def test_reload_rejects_trim_arguments(daily_meters, fmt, kwarg):
    payload = getattr(ComparisonPool.from_fit_models(daily_meters), f"to_{fmt}")()

    with pytest.raises(TypeError, match=next(iter(kwarg))):
        getattr(ComparisonPool, f"from_{fmt}")(payload, **kwarg)


def test_exclusions_roundtrip_through_json(daily_meters):
    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)

    rebuilt = ComparisonPool.from_json(pool.to_json(), baseline=_baseline_map(daily_meters))

    pd.testing.assert_frame_equal(rebuilt.exclusions, pool.exclusions)
    assert not rebuilt.exclusions.empty


# -- from_data serial fit ----------------------------------------------------


@pytest.mark.slow
def test_from_data_fits_serially(_comstock_daily_all):
    df_b, _ = _comstock_daily_all
    ids = sorted(df_b.index.get_level_values("id").unique())[:2]
    baseline = {
        str(mid): DailyBaselineData(
            df_b.xs(mid, level="id").reset_index(), is_electricity_data=True
        )
        for mid in ids
    }

    group = TreatmentGroup.from_data(baseline, "daily")

    assert group.model_frequency == "daily"
    assert set(group.ids) == set(baseline)
    assert len(group.predictions("baseline")) > 0


@pytest.mark.slow
def test_from_data_excludes_meter_failing_baseline_sufficiency(_comstock_daily_all):
    """A meter whose baseline is too short to pass sufficiency is excluded and
    recorded with the verbatim eemeter qualified name while the other meters
    fit normally."""
    df_b, _ = _comstock_daily_all
    ids = sorted(df_b.index.get_level_values("id").unique())[:3]
    baseline = {}

    for mid in ids[:2]:
        baseline[str(mid)] = DailyBaselineData(
            df_b.xs(mid, level="id").reset_index(), is_electricity_data=True
        )

    short = df_b.xs(ids[2], level="id").iloc[:60]
    baseline[str(ids[2])] = DailyBaselineData(short.reset_index(), is_electricity_data=True)

    group = TreatmentGroup.from_data(baseline, "daily")

    assert set(group.ids) == {str(mid) for mid in ids[:2]}
    row = group.exclusions.set_index("id").loc[str(ids[2])]
    assert row["stage"] == "population"
    assert row["origin"] == "baseline_data"
    assert row["reason"] == "baseline data insufficient to fit a model"
    assert "eemeter.sufficiency_criteria" in row["detail"]
    assert len(group.predictions("baseline")) > 0


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
        baseline[str(mid)] = DailyBaselineData(raw_b.reset_index(), is_electricity_data=True)
        reporting[str(mid)] = DailyReportingData(
            df_r.xs(mid, level="id").reset_index(), is_electricity_data=True
        )

    group = TreatmentGroup.from_data(baseline, "daily", reporting=reporting)

    assert set(group.ids) == {str(mid) for mid in ids[:2]}
    for mid in group.ids:
        assert group._meters[mid].reporting_data is reporting[mid]
    predicted = group.predictions("reporting")
    assert set(predicted["id"]) == set(group.ids)


def test_from_data_unknown_reporting_id_raises_before_fitting(_comstock_daily_all):
    df_b, df_r = _comstock_daily_all
    mid = sorted(df_b.index.get_level_values("id").unique())[0]
    baseline = {
        str(mid): DailyBaselineData(df_b.xs(mid, level="id").reset_index(), is_electricity_data=True)
    }
    stray = DailyReportingData(df_r.xs(mid, level="id").reset_index(), is_electricity_data=True)

    with pytest.raises(KeyError, match="not in baseline: \\['does-not-exist'\\]"):
        TreatmentGroup.from_data(baseline, "daily", reporting={"does-not-exist": stray})


@pytest.mark.slow
def test_from_data_reporting_timezone_mismatch_raises(comstock, _comstock_daily_all):
    df_b, _ = _comstock_daily_all
    mid = sorted(df_b.index.get_level_values("id").unique())[0]
    baseline = {
        str(mid): DailyBaselineData(df_b.xs(mid, level="id").reset_index(), is_electricity_data=True)
    }
    reporting = {str(mid): _new_york_reporting(comstock, mid)}

    with pytest.raises(
        ValueError, match=f"Meter {mid}: reporting data timezone America/New_York"
    ):
        TreatmentGroup.from_data(baseline, "daily", reporting=reporting)


def _pinned_fit(monkeypatch, meters, fail=()):
    """Replace ``DailyModel.fit`` with a copy of each meter's pinned model, so
    ``from_data`` runs without fitting. Returns the ids fit, in call order;
    meters in ``fail`` raise ``DataSufficiencyError``."""
    by_baseline = {id(entry["baseline_data"]): mid for mid, entry in meters.items()}
    fitted = []

    def fit(self, baseline_data, ignore_disqualification=False):
        mid = by_baseline[id(baseline_data)]
        fitted.append(mid)
        if mid in fail:
            raise DataSufficiencyError("baseline too short")

        vars(self).update(vars(meters[mid]["model"]))

        return self

    monkeypatch.setattr(DailyModel, "fit", fit)

    return fitted


@pytest.mark.parametrize("max_pool_size, n_fit", [(2, 2), (4, 4), (10, 4)])
def test_pool_from_data_fits_only_the_kept_meters(daily_meters, monkeypatch, max_pool_size, n_fit):
    """``from_data`` fits ``min(max_pool_size, len(baseline))`` meters, the ones
    the pool keeps, and ledgers the rest as ``pool_trim`` unfit."""
    fitted = _pinned_fit(monkeypatch, daily_meters)

    pool = ComparisonPool.from_data(
        _baseline_map(daily_meters), "daily", max_pool_size=max_pool_size, seed=0
    )

    assert len(fitted) == n_fit, f"fit {fitted}, expected {n_fit} fits"
    assert fitted == pool.ids
    assert sorted(pool.exclusions["id"]) == sorted(set(daily_meters) - set(fitted))
    assert (pool.exclusions["origin"] == "pool_trim").all()


def test_pool_from_data_keeps_the_ids_from_fit_models_keeps(daily_meters, monkeypatch):
    _pinned_fit(monkeypatch, daily_meters)
    reference = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=3)

    pool = ComparisonPool.from_data(
        _baseline_map(daily_meters), "daily", max_pool_size=2, seed=3
    )

    assert pool.ids == reference.ids
    pd.testing.assert_frame_equal(pool.exclusions, reference.exclusions)


def test_pool_from_data_does_not_replace_a_kept_meter_failing_to_fit(daily_meters, monkeypatch):
    """A kept meter that fails to fit is ledgered and not replaced by a trimmed
    one, so the pool ends below the cap."""
    kept = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=0).ids
    fitted = _pinned_fit(monkeypatch, daily_meters, fail={kept[0]})

    pool = ComparisonPool.from_data(_baseline_map(daily_meters), "daily", max_pool_size=2, seed=0)

    assert fitted == kept
    assert pool.ids == kept[1:]
    origins = pool.exclusions.set_index("id")["origin"]
    assert origins[kept[0]] == "baseline_data"
    assert sorted(origins[origins == "pool_trim"].index) == sorted(set(daily_meters) - set(kept))


def test_pool_from_data_omitted_cap_reads_the_module_default(daily_meters, monkeypatch):
    fitted = _pinned_fit(monkeypatch, daily_meters)
    monkeypatch.setattr("opendsm.comparison_groups.population._MAX_POOL_SIZE", 2)

    pool = ComparisonPool.from_data(_baseline_map(daily_meters), "daily")

    assert len(fitted) == 2
    assert pool.max_pool_size == 2


def test_pool_from_data_drops_reporting_of_trimmed_meters(daily_meters, monkeypatch):
    """Reporting for a trimmed meter is dropped with it rather than raising as
    an id outside ``baseline``."""
    _pinned_fit(monkeypatch, daily_meters)

    pool = ComparisonPool.from_data(
        _baseline_map(daily_meters),
        "daily",
        max_pool_size=2,
        seed=0,
        reporting=_reporting_map(daily_meters),
    )

    for mid in pool.ids:
        assert pool._meters[mid].reporting_data is daily_meters[mid]["reporting_data"]


def test_from_data_all_meters_failing_raises(_comstock_daily_all):
    df_b, _ = _comstock_daily_all
    mid = sorted(df_b.index.get_level_values("id").unique())[0]
    short = df_b.xs(mid, level="id").iloc[:60]
    baseline = {str(mid): DailyBaselineData(short.reset_index(), is_electricity_data=True)}

    with pytest.raises(ValueError, match="All meters failed to fit"):
        TreatmentGroup.from_data(baseline, "daily")
