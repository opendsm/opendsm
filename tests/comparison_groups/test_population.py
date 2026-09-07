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
from opendsm.eemeter.common.warnings import EEMeterWarning
from opendsm.eemeter.models.daily.data import DailyBaselineData



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


# -- granularity dispatch ----------------------------------------------------


def test_granularity_dispatch_daily_not_billing(daily_meters):
    group = TreatmentGroup.from_fit_models(_subset(daily_meters, 2))

    assert group.granularity == "daily"


def test_granularity_dispatch_billing_not_daily(billing_pair):
    # BillingModel subclasses DailyModel, so isinstance would misclassify;
    # exact-type dispatch (billing first) must resolve to billing even when
    # granularity is inferred from the instances.
    group = TreatmentGroup.from_fit_models(billing_pair)

    assert group.granularity == "billing"


def _dict_payload(meters, mid):
    payload = {
        mid: {
            "model": meters[mid]["model"].to_dict(),
            "baseline_df": meters[mid]["baseline_df"],
        }
    }

    return payload


def test_from_fit_models_infers_granularity_from_tagged_payload(daily_meters):
    mid = list(daily_meters)[0]
    payload = _dict_payload(daily_meters, mid)

    rebuilt = TreatmentGroup.from_fit_models(payload)

    assert rebuilt.granularity == "daily"


def test_from_fit_models_infers_granularity_from_tagged_json_payload(daily_meters):
    mid = list(daily_meters)[0]
    payload = {
        mid: {
            "model": daily_meters[mid]["model"].to_json(),
            "baseline_df": daily_meters[mid]["baseline_df"],
        }
    }

    rebuilt = TreatmentGroup.from_fit_models(payload)

    assert rebuilt.granularity == "daily"


def test_from_fit_models_infers_billing_from_tagged_payload(billing_pair):
    # A billing payload is structurally identical to a daily one; the model_type
    # tag is what distinguishes them without an explicit granularity.
    mid = list(billing_pair)[0]
    payload = _dict_payload(billing_pair, mid)

    rebuilt = TreatmentGroup.from_fit_models(payload)

    assert rebuilt.granularity == "billing"


def test_from_fit_models_requires_granularity_for_untagged_payload(daily_meters):
    mid = list(daily_meters)[0]
    payload = _dict_payload(daily_meters, mid)
    del payload[mid]["model"]["model_type"]

    with pytest.raises(ValueError, match="granularity is required"):
        TreatmentGroup.from_fit_models(payload)

    rebuilt = TreatmentGroup.from_fit_models(payload, granularity="daily")

    assert rebuilt.granularity == "daily"


def test_from_fit_models_rejects_tag_granularity_mismatch(daily_meters):
    mid = list(daily_meters)[0]
    payload = _dict_payload(daily_meters, mid)

    with pytest.raises(TypeError, match="does not match requested"):
        TreatmentGroup.from_fit_models(payload, granularity="billing")


def test_from_fit_models_rejects_mismatched_declared_granularity(billing_pair):
    with pytest.raises(TypeError, match="does not match requested"):
        TreatmentGroup.from_fit_models(billing_pair, granularity="daily")


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
        TreatmentGroup.from_fit_models(meters, granularity="daily")


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
        TreatmentGroup.from_fit_models(meters, granularity="daily")


def test_model_and_baseline_data_timezone_mismatch_raises(comstock, daily_meters):
    """A model fit in one timezone paired with baseline data in another names
    the meter and both zones."""
    mid = list(daily_meters)[0]
    meters = {mid: _new_york_meter(comstock, daily_meters, mid)}

    with pytest.raises(ValueError, match=f"Meter {mid}: model timezone America/Chicago"):
        TreatmentGroup.from_fit_models(meters, granularity="daily")


def test_disqualified_meter_warns(daily_meters):
    ids = list(daily_meters)[:2]
    flagged = copy.deepcopy(daily_meters[ids[0]]["model"])
    flagged.disqualification = [
        EEMeterWarning(qualified_name="eemeter.test", description="flagged", data=None)
    ]
    meters = {
        ids[0]: {"model": flagged, "baseline_df": daily_meters[ids[0]]["baseline_df"]},
        ids[1]: {
            "model": daily_meters[ids[1]]["model"],
            "baseline_df": daily_meters[ids[1]]["baseline_df"],
        },
    }

    with pytest.warns(UserWarning, match="disqualified"):
        TreatmentGroup.from_fit_models(meters, granularity="daily")


def test_solar_mix_raises(hourly_fit):
    """One hourly meter whose baseline frame carries no GHI column alongside one
    that does is a solar mix the population refuses."""
    ids = list(hourly_fit["models"])[:2]
    non_solar = hourly_fit["baseline"][ids[0]].drop(columns="ghi")
    meters = {
        ids[0]: {"model": hourly_fit["models"][ids[0]], "baseline_df": non_solar},
        ids[1]: {
            "model": hourly_fit["models"][ids[1]],
            "baseline_df": hourly_fit["baseline"][ids[1]],
        },
    }

    with pytest.raises(ValueError, match="solar"):
        TreatmentGroup.from_fit_models(meters, granularity="hourly")


# -- windows -----------------------------------------------------------------


def test_baseline_window_spans_meters(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    start, end = group.baseline_window
    first_index = group._meters[list(daily_meters)[0]].baseline_data.df.index

    assert start == first_index.min()
    assert end == first_index.max()


def test_reporting_window_is_union_across_differing_spans(daily_meters, _comstock_daily_all):
    _, df_r = _comstock_daily_all
    ids = sorted(df_r.index.get_level_values("id").unique())[:4]
    reporting = {}

    for position, mid in enumerate(ids):
        raw = df_r.xs(mid, level="id")
        if position == 0:
            cutoff = raw.index.max() - pd.Timedelta(days=45)
            raw = raw[raw.index <= cutoff]
        reporting[str(mid)] = raw.reset_index()

    group = TreatmentGroup.from_fit_models(daily_meters)
    group.add_reporting_data(reporting)

    starts = [frame["datetime"].min() for frame in reporting.values()]
    ends = [frame["datetime"].max() for frame in reporting.values()]

    assert min(ends) < max(ends)
    assert group.reporting_window == (min(starts), max(ends))


def test_explicit_window_override_supersedes_data_span(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    span_baseline = group.baseline_window
    span_reporting = group.reporting_window
    baseline_window = (
        span_baseline[0] - pd.Timedelta(days=5),
        span_baseline[1] + pd.Timedelta(days=5),
    )
    reporting_window = (
        span_reporting[0] - pd.Timedelta(days=5),
        span_reporting[1] + pd.Timedelta(days=5),
    )

    overridden = TreatmentGroup.from_fit_models(
        daily_meters, baseline_window=baseline_window, reporting_window=reporting_window
    )

    assert overridden.baseline_window == baseline_window
    assert overridden.reporting_window == reporting_window
    assert overridden.baseline_window != span_baseline
    assert overridden.reporting_window != span_reporting


def test_naive_window_override_is_localized_to_the_population_timezone(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    start, end = group.baseline_window
    naive = (start.tz_localize(None), end.tz_localize(None))

    overridden = TreatmentGroup.from_fit_models(daily_meters, baseline_window=naive)

    assert overridden.baseline_window == (start, end)
    assert str(overridden.baseline_window[0].tz) == group.tz


def test_reporting_window_override_stays_none_without_reporting_data(daily_meters):
    meters = {
        mid: {"model": entry["model"], "baseline_df": entry["baseline_df"]}
        for mid, entry in daily_meters.items()
    }
    reporting_window = (
        pd.Timestamp("2020-01-01", tz="UTC"),
        pd.Timestamp("2020-12-31", tz="UTC"),
    )
    group = TreatmentGroup.from_fit_models(
        meters, granularity="daily", reporting_window=reporting_window
    )

    assert group.reporting_window is None


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
        substrate = group._meters[mid].baseline_data.df
        n_rows = len(frame[frame["id"] == mid])
        n_days = len(substrate)
        n_months = substrate.index.tz_localize(None).to_period("M").nunique()

        assert n_rows == n_days
        assert n_rows > n_months


def test_prediction_matrices_observed_unc_present_when_set(daily_meters):
    meters = {}
    for mid, entry in _subset(daily_meters, 2).items():
        meters[mid] = {**entry, "observed_unc": 3.0}
    group = TreatmentGroup.from_fit_models(meters, granularity="daily")

    _, _, _, _, _, observed_unc = group._prediction_matrices("baseline")

    assert observed_unc is not None
    assert observed_unc.dtype == np.float32
    assert np.allclose(observed_unc, 3.0)


def test_observed_unc_series_not_covering_window_raises(daily_meters):
    meters = dict(_subset(daily_meters, 2))
    mid = next(iter(meters))
    baseline_index = pd.DatetimeIndex(meters[mid]["baseline_df"]["datetime"])
    unc_series = pd.Series(2.0, index=baseline_index[: len(baseline_index) // 2])
    meters[mid] = {**meters[mid], "observed_unc": unc_series}
    group = TreatmentGroup.from_fit_models(meters, granularity="daily")

    with pytest.raises(ValueError, match="does not cover the requested window"):
        group._prediction_matrices("baseline")


# -- reporting attach + cache invalidation -----------------------------------


@pytest.mark.slow
def test_reporting_extension_matches_full_repredict(hourly_fit):
    def _meters(reporting=None):
        built = {}
        for mid, model in hourly_fit["models"].items():
            entry = {"model": model, "baseline_df": hourly_fit["baseline"][mid]}
            if reporting is not None:
                entry["reporting_df"] = reporting[mid]
            built[mid] = entry

        return built

    extended = TreatmentGroup.from_fit_models(_meters(), granularity="hourly")
    extended.add_reporting_data(hourly_fit["half_reporting"])
    half_predictions = extended.predictions("reporting")
    extended.add_reporting_data(hourly_fit["full_reporting"])
    extended_predictions = extended.predictions("reporting")

    fresh = TreatmentGroup.from_fit_models(
        _meters(hourly_fit["full_reporting"]), granularity="hourly"
    )
    fresh_predictions = fresh.predictions("reporting")

    assert len(half_predictions) < len(extended_predictions)
    pd.testing.assert_frame_equal(extended_predictions, fresh_predictions)


def test_add_reporting_data_unknown_meter_raises(daily_meters):
    group = TreatmentGroup.from_fit_models(_subset(daily_meters, 2))
    stray = daily_meters[list(daily_meters)[-1]]["reporting_df"]

    with pytest.raises(KeyError, match="not part of this population"):
        group.add_reporting_data({"does-not-exist": stray})


def test_add_reporting_data_tz_mismatch_leaves_record_and_cache_untouched(daily_meters):
    """A reporting frame is validated before it replaces the record, so a
    rejected meter's reporting_data and the population's reporting_window
    survive the call unchanged."""
    meters = {
        mid: {"model": entry["model"], "baseline_df": entry["baseline_df"]}
        for mid, entry in _subset(daily_meters, 2).items()
    }
    ids = list(meters)
    group = TreatmentGroup.from_fit_models(meters, granularity="daily")
    group.add_reporting_data({ids[1]: daily_meters[ids[1]]["reporting_df"]})
    stale_reporting_data = group._meters[ids[1]].reporting_data
    stale_window = group.reporting_window

    good_reporting = daily_meters[ids[0]]["reporting_df"]
    bad_reporting = daily_meters[ids[1]]["reporting_df"].copy()
    bad_reporting = bad_reporting.set_index("datetime")
    bad_reporting.index = bad_reporting.index.tz_convert("America/New_York")
    bad_reporting = bad_reporting.reset_index()

    with pytest.raises(ValueError, match="does not match"):
        group.add_reporting_data({ids[0]: good_reporting, ids[1]: bad_reporting})

    assert group._meters[ids[0]].reporting_data is not None
    assert group._meters[ids[1]].reporting_data is stale_reporting_data
    assert group.reporting_window == stale_window


def test_reporting_frame_attached_later_predicts(daily_meters):
    """A population built from baseline frames alone predicts reporting once the
    reporting frames arrive: the model builds each data object from the frame."""
    meters = {
        mid: {"model": entry["model"], "baseline_df": entry["baseline_df"]}
        for mid, entry in _subset(daily_meters, 2).items()
    }
    group = TreatmentGroup.from_fit_models(meters, granularity="daily")

    assert group.reporting_window is None

    group.add_reporting_data(_reporting_map(_subset(daily_meters, 2)))
    frame = group.predictions("reporting")

    assert group.reporting_window is not None
    assert set(frame["id"].unique()) == set(group.ids)
    assert np.isfinite(frame["modeled"].to_numpy()).all()


def test_from_fit_models_rejects_a_data_object_where_a_frame_belongs(daily_meters):
    """Populations build the data objects themselves, so a prebuilt data object
    passed as ``baseline_df`` is a TypeError naming the meter."""
    mid = list(daily_meters)[0]
    data = DailyBaselineData(daily_meters[mid]["baseline_df"], is_electricity_data=True)
    meters = {mid: {"model": daily_meters[mid]["model"], "baseline_df": data}}

    with pytest.raises(TypeError, match=f"Meter {mid}: baseline_df must be a pandas DataFrame"):
        TreatmentGroup.from_fit_models(meters, granularity="daily")


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
        mid: {"model": model, "baseline_df": hourly_fit["baseline"][mid]}
        for mid, model in hourly_fit["models"].items()
    }
    group = TreatmentGroup.from_fit_models(meters, granularity="hourly")
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


def _truncate_baseline_months(population, ids, n_months=6):
    """Cut each named meter's attached billing baseline down to ``n_months``,
    after construction so the truncation is not itself a sufficiency failure."""
    for mid in ids:
        baseline_data = population._meters[mid].baseline_data
        baseline_data._df = baseline_data._df.head(n_months)


def test_billing_loadshape_requires_full_year(billing_pair):
    group = TreatmentGroup.from_fit_models(billing_pair, granularity="billing")
    _truncate_baseline_months(group, group.ids)

    with pytest.raises(ValueError, match="at least 12 baseline months"):
        group.loadshape_data("modeled")


def test_billing_pool_short_baseline_excluded_not_raised(billing_pair):
    good_mid, bad_mid = list(billing_pair)

    pool = ComparisonPool.from_fit_models(billing_pair, granularity="billing")
    _truncate_baseline_months(pool, [bad_mid])

    data = pool.loadshape_data("modeled")

    assert pool.ids == [good_mid]
    assert list(pool.exclusions["id"]) == [bad_mid]
    assert (pool.exclusions["stage"] == "population").all()
    assert (pool.exclusions["origin"] == "baseline_data").all()
    assert data.settings.time_period == _const.TimePeriod.MONTH
    assert data.settings.max_pool_size == 1


def test_billing_treatment_all_excluded_raises_named_ids(billing_pair):
    group = TreatmentGroup.from_fit_models(billing_pair, granularity="billing")
    _truncate_baseline_months(group, group.ids)

    with pytest.raises(ValueError, match="all treatment meters excluded"):
        group.loadshape_data("modeled")

    assert group.exclusions.empty


# -- serialization -----------------------------------------------------------


def test_json_roundtrip_reattaches_data_and_preserves_tz(daily_meters):
    meters = {}
    for mid, entry in daily_meters.items():
        meters[mid] = {**entry, "observed_unc": 1.5}
    group = TreatmentGroup.from_fit_models(meters, granularity="daily")

    payload = group.to_json()
    rebuilt = TreatmentGroup.from_json(
        payload,
        baseline=_baseline_map(daily_meters),
        reporting=_reporting_map(daily_meters),
    )

    assert rebuilt.granularity == "daily"
    assert rebuilt.tz == group.tz
    assert rebuilt.ids == group.ids
    assert (rebuilt.predictions("baseline")["observed_unc"] == 1.5).all()
    pd.testing.assert_frame_equal(group.predictions("baseline"), rebuilt.predictions("baseline"))


def test_json_roundtrip_restores_explicit_windows_not_recomputed(daily_meters):
    group = TreatmentGroup.from_fit_models(daily_meters)
    span_baseline = group.baseline_window
    span_reporting = group.reporting_window
    baseline_window = (
        span_baseline[0] - pd.Timedelta(days=7),
        span_baseline[1] + pd.Timedelta(days=7),
    )
    reporting_window = (
        span_reporting[0] - pd.Timedelta(days=7),
        span_reporting[1] + pd.Timedelta(days=7),
    )
    group = TreatmentGroup.from_fit_models(
        daily_meters, baseline_window=baseline_window, reporting_window=reporting_window
    )

    rebuilt = TreatmentGroup.from_json(
        group.to_json(),
        baseline=_baseline_map(daily_meters),
        reporting=_reporting_map(daily_meters),
    )

    assert rebuilt.baseline_window == baseline_window
    assert rebuilt.reporting_window == reporting_window
    assert rebuilt.baseline_window != span_baseline
    assert rebuilt.reporting_window != span_reporting


def test_from_json_rebuilds_model_payload_missing_fuel_flag(daily_meters):
    """A model payload written before the fuel flag was stored rebuilds using the
    population header's is_electricity_data, and the rebuilt model records that."""
    subset = _subset(daily_meters, 2)
    group = TreatmentGroup.from_fit_models(subset, granularity="daily")
    payload = json.loads(group.to_json())
    for meter in payload["meters"]:
        model_payload = json.loads(meter["model"])
        del model_payload["info"]["is_electricity_data"]
        meter["model"] = json.dumps(model_payload)

    rebuilt = TreatmentGroup.from_json(json.dumps(payload), baseline=_baseline_map(subset))

    assert rebuilt.is_electricity_data == group.is_electricity_data
    for mid in rebuilt.ids:
        names = {w.qualified_name for w in rebuilt._meters[mid].model.warnings}
        assert "eemeter.serialization.prior_format" in names


def test_from_json_pool_trim_is_recorded_alongside_the_serialized_ledger(daily_meters):
    """A trim requested on the rebuild lands on the ledger after the serialized
    rows instead of replacing them."""
    pool = ComparisonPool.from_fit_models(daily_meters)

    rebuilt = ComparisonPool.from_json(
        pool.to_json(), baseline=_baseline_map(daily_meters), max_pool_size=2, seed=0
    )

    assert len(rebuilt) == 2
    assert list(rebuilt.exclusions["origin"]) == ["pool_trim", "pool_trim"]
    assert set(rebuilt.exclusions["id"]) == set(daily_meters) - set(rebuilt.ids)


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
        granularity="daily",
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

    text = group.to_ndjson()

    assert len(text.splitlines()) == len(group.ids) + 1


def test_from_json_rejects_role_mismatch(daily_meters):
    pool_payload = ComparisonPool.from_fit_models(daily_meters).to_json()

    with pytest.raises(ValueError, match="role"):
        TreatmentGroup.from_json(pool_payload, baseline=_baseline_map(daily_meters))


# -- comparison pool trim ----------------------------------------------------


def test_comparison_pool_trim_records_excluded(daily_meters):
    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)

    assert len(pool) == 2
    assert list(pool.exclusions.columns) == ["id", "stage", "origin", "reason", "detail"]
    assert len(pool.exclusions) == len(daily_meters) - 2
    assert set(pool.exclusions["id"]).isdisjoint(pool.ids)
    assert (pool.exclusions["stage"] == "population").all()
    assert (pool.exclusions["origin"] == "pool_trim").all()
    assert (pool.exclusions["reason"] == "randomly selected to reduce pool size").all()


def test_comparison_pool_trim_is_deterministic(daily_meters):
    first = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)
    second = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)

    assert first.ids == second.ids
    assert list(first.exclusions["id"]) == list(second.exclusions["id"])


def test_exclusions_roundtrip_through_json(daily_meters):
    pool = ComparisonPool.from_fit_models(daily_meters, max_pool_size=2, seed=42)

    rebuilt = ComparisonPool.from_json(pool.to_json(), baseline=_baseline_map(daily_meters))

    pd.testing.assert_frame_equal(rebuilt.exclusions, pool.exclusions)
    assert not rebuilt.exclusions.empty


# -- from_data serial fit ----------------------------------------------------


def test_from_data_requires_is_electricity_data_as_keyword():
    """is_electricity_data is keyword-only, so a positional third argument
    (such as a settings object) cannot silently bind to it instead."""
    with pytest.raises(TypeError):
        TreatmentGroup.from_data({}, "daily", True)


@pytest.mark.slow
def test_from_data_fits_serially(_comstock_daily_all):
    df_b, _ = _comstock_daily_all
    ids = sorted(df_b.index.get_level_values("id").unique())[:2]
    baseline = {str(mid): df_b.xs(mid, level="id").reset_index() for mid in ids}

    group = TreatmentGroup.from_data(baseline, "daily", is_electricity_data=True)

    assert group.granularity == "daily"
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
        baseline[str(mid)] = df_b.xs(mid, level="id").reset_index()

    short = df_b.xs(ids[2], level="id").iloc[:60]
    baseline[str(ids[2])] = short.reset_index()

    group = TreatmentGroup.from_data(baseline, "daily", is_electricity_data=True)

    assert set(group.ids) == {str(mid) for mid in ids[:2]}
    row = group.exclusions.set_index("id").loc[str(ids[2])]
    assert row["stage"] == "population"
    assert row["origin"] == "baseline_data"
    assert row["reason"] == "baseline data insufficient to fit a model"
    assert "eemeter.sufficiency_criteria" in row["detail"]
    assert len(group.predictions("baseline")) > 0


def test_from_data_all_meters_failing_raises(_comstock_daily_all):
    df_b, _ = _comstock_daily_all
    mid = sorted(df_b.index.get_level_values("id").unique())[0]
    short = df_b.xs(mid, level="id").iloc[:60]
    baseline = {str(mid): short.reset_index()}

    with pytest.raises(ValueError, match="All meters failed to fit"):
        TreatmentGroup.from_data(baseline, "daily", is_electricity_data=True)
