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

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const
from opendsm.comparison_groups.common.data import (
    _keep_smallest_abs,
    fill_missing,
)



_FOUR_SEASONS = {
    "january": "winter",
    "february": "winter",
    "march": "spring",
    "april": "spring",
    "may": "spring",
    "june": "summer",
    "july": "summer",
    "august": "summer",
    "september": "fall",
    "october": "fall",
    "november": "fall",
    "december": "winter",
    "options": ["summer", "fall", "winter", "spring"],
}

_YEAR_HOURS = pd.date_range("2023-01-01", periods=8760, freq="h")


def _long_loadshape_df(n_ids=3, n_time=24):
    rows = [
        {"id": f"m{i}", "time": t, "loadshape": float(t + i)}
        for i in range(n_ids)
        for t in range(1, n_time + 1)
    ]

    return pd.DataFrame(rows)


def _hourly_frame(meter_id, n_hours):
    timestamps = pd.date_range("2024-01-01", periods=n_hours, freq="h")

    return pd.DataFrame(
        {"id": meter_id, "datetime": timestamps, "observed": np.arange(n_hours, dtype=float)}
    )


def _treatment_time_series(comstock_hourly_all, n_ids=3):
    df_baseline, _ = comstock_hourly_all
    ids = sorted(df_baseline.index.get_level_values("id").unique())[:n_ids]
    time_series = df_baseline.loc[ids].reset_index()[["id", "datetime", "observed"]]

    return time_series, ids


def test_hour_time_period_keys_present():
    """TimePeriod.HOUR is a valid key in the data interval dict."""
    assert _const.max_data_interval_minutes_per_time_period[_const.TimePeriod.HOUR] == 60


def test_time_series_ingestion_hourly(_comstock_hourly_all):
    """Building Data from a time series reads loadshape_type and time_period and produces a
    24-column loadshape."""
    time_series, ids = _treatment_time_series(_comstock_hourly_all)
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )

    data = Data(time_series_df=time_series, settings=settings)

    assert data.loadshape is not None
    assert data.loadshape.shape[1] == 24
    assert set(data.loadshape.index).issubset(set(ids))
    assert np.isfinite(data.loadshape.to_numpy()).all()


def test_pool_trim_reproducible_with_seed(_comstock_hourly_all):
    """Comparison-pool trimming is reproducible when a seed is set."""
    time_series, _ = _treatment_time_series(_comstock_hourly_all, n_ids=8)

    def _build():
        settings = Data_Settings(
            agg_type=_const.AggType.MEAN,
            loadshape_type=_const.LoadshapeType.OBSERVED,
            time_period=_const.TimePeriod.HOUR,
            max_pool_size=4,
            seed=7,
        )

        return Data(time_series_df=time_series, settings=settings)

    data1 = _build()
    data2 = _build()

    assert len(data1.loadshape) == 4
    assert sorted(data1.loadshape.index) == sorted(data2.loadshape.index)


def test_loadshape_df_ingestion():
    """A long-format loadshape (id, time, loadshape) pivots to one row per id."""
    data = Data(loadshape_df=_long_loadshape_df(n_ids=3, n_time=24))

    assert data.loadshape.shape == (3, 24)


def test_features_df_ingestion():
    """A features frame is indexed by id with one column per feature."""
    features = pd.DataFrame(
        {
            "id": [f"m{i}" for i in range(5)],
            "summer_usage": [3000.0, 4000.0, 5000.0, 3500.0, 4500.0],
            "winter_usage": [5000.0, 5500.0, 6000.0, 5200.0, 5800.0],
        }
    )
    settings = Data_Settings(agg_type=None, loadshape_type=None, time_period=None)

    data = Data(features_df=features, settings=settings)

    assert data.features.shape == (5, 2)
    assert data.loadshape is None


def test_no_input_raises():
    with pytest.raises(ValueError):
        Data()


def test_both_loadshape_and_time_series_raises():
    loadshape = _long_loadshape_df()
    time_series = _hourly_frame("m0", 48)

    with pytest.raises(ValueError):
        Data(loadshape_df=loadshape, time_series_df=time_series)


def test_time_series_missing_loadshape_column_raises():
    time_series = _hourly_frame("m0", 48).drop(columns="observed")
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )

    with pytest.raises(ValueError):
        Data(time_series_df=time_series, settings=settings)


def test_time_series_string_datetime_raises():
    """The datetime column must be a datetime dtype, not ISO strings."""
    time_series = pd.DataFrame(
        {
            "id": ["m0"] * 3,
            "datetime": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "observed": [1.0, 2.0, 3.0],
        }
    )
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )

    with pytest.raises(ValueError):
        Data(time_series_df=time_series, settings=settings)


def test_insufficient_data_meter_is_excluded():
    """A meter with far fewer than the required hourly observations is dropped
    and recorded in excluded_ids rather than crashing the build."""
    time_series = pd.concat(
        [_hourly_frame("good", 48), _hourly_frame("bad", 5)], ignore_index=True
    )
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )

    data = Data(time_series_df=time_series, settings=settings)

    assert "good" in data.loadshape.index
    assert "bad" not in data.loadshape.index
    assert "bad" in data.excluded_ids["id"].values


def test_meter_coarser_than_time_period_is_excluded():
    """A meter read every 2 hours cannot fill an hourly time period, so it is excluded
    with the data interval reason even when its rows arrive out of time order."""
    rng = np.random.default_rng(0)
    two_hourly = pd.DataFrame(
        {
            "id": "two_hourly",
            "datetime": _YEAR_HOURS[::2],
            "observed": rng.uniform(1.0, 2.0, len(_YEAR_HOURS[::2])),
        }
    )
    hourly = _year_frame("hourly", rng.uniform(1.0, 2.0, len(_YEAR_HOURS)))
    time_series = pd.concat([two_hourly, hourly], ignore_index=True).sample(frac=1, random_state=0)
    settings = _observed_settings(time_period=_const.TimePeriod.SEASONAL_HOURLY_DAY_OF_WEEK)

    data = Data(time_series_df=time_series, settings=settings)

    assert list(data.loadshape.index) == ["hourly"]
    ledger = data.excluded_ids.to_dict("records")
    assert ledger == [
        {"id": "two_hourly", "reason": "Minimum time interval is more than the specified TimePeriod"}
    ], f"expected one data interval row for 'two_hourly', got {ledger}"


_INCOMPLETE_REASON = "missing loadshape times with interpolate_missing off"
_BELOW_MINIMUM_REASON = "fewer loadshape times present than min_data_pct_required"


def _hour_settings(interpolate_missing):
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
        interpolate_missing=interpolate_missing,
    )

    return settings


def _loadshape_settings(interpolate_missing):
    settings = Data_Settings(
        agg_type=None,
        loadshape_type=None,
        time_period=None,
        interpolate_missing=interpolate_missing,
    )

    return settings


def test_interpolation_off_ledgers_below_minimum_and_incomplete_time_series():
    """With interpolation off, a meter below min_data_pct_required gets the below-minimum
    row, a meter at or above it that misses a time gets the incomplete row, and only the
    complete meter is kept."""
    time_series = pd.concat(
        [_hourly_frame("full", 48), _hourly_frame("low", 10), _hourly_frame("short", 20)],
        ignore_index=True,
    )

    data = Data(time_series_df=time_series, settings=_hour_settings(interpolate_missing=False))

    assert list(data.loadshape.index) == ["full"]
    ledger = data.excluded_ids.to_dict("records")
    assert ledger == [
        {"id": "low", "reason": _BELOW_MINIMUM_REASON},
        {"id": "short", "reason": _INCOMPLETE_REASON},
    ], f"expected one below-minimum row for 'low' and one incomplete row for 'short', got {ledger}"


def test_interpolation_off_ledgers_below_minimum_and_incomplete_loadshape_df():
    """With interpolation off, a long loadshape meter with 3 of 24 times gets the
    below-minimum row, one with 23 of 24 gets the incomplete row, and only the complete
    meter is kept."""
    long_df = _long_loadshape_df(n_ids=3, n_time=24)
    missing = ((long_df["id"] == "m1") & (long_df["time"] == 5)) | (
        (long_df["id"] == "m2") & (long_df["time"] > 3)
    )

    data = Data(
        loadshape_df=long_df[~missing], settings=_loadshape_settings(interpolate_missing=False)
    )

    assert list(data.loadshape.index) == ["m0"]
    ledger = data.excluded_ids.to_dict("records")
    assert ledger == [
        {"id": "m2", "reason": _BELOW_MINIMUM_REASON},
        {"id": "m1", "reason": _INCOMPLETE_REASON},
    ], f"expected one below-minimum row for 'm2' and one incomplete row for 'm1', got {ledger}"


def test_interpolation_off_excludes_meter_missing_times():
    """With interpolation off, a meter missing hours is dropped with one ledger row
    and the complete meters keep their loadshapes."""
    time_series = pd.concat(
        [_hourly_frame("a", 48), _hourly_frame("short", 20), _hourly_frame("b", 48)],
        ignore_index=True,
    )

    data = Data(time_series_df=time_series, settings=_hour_settings(interpolate_missing=False))

    assert list(data.loadshape.index) == ["a", "b"]
    assert data.loadshape.shape == (2, 24)
    np.testing.assert_array_equal(data.loadshape.loc["a"].to_numpy(), np.arange(24) + 12.0)
    ledger = data.excluded_ids
    assert ledger.to_dict("records") == [{"id": "short", "reason": _INCOMPLETE_REASON}], (
        f"expected one ledger row for 'short', got {ledger.to_dict('records')}"
    )


def test_meter_without_datetimes_is_recorded():
    """A meter whose datetimes are all NaT has no reading in any time and is dropped with
    one ledger row."""
    no_times = _hourly_frame("no_times", 48)
    no_times["datetime"] = pd.NaT
    time_series = pd.concat([_hourly_frame("a", 48), no_times], ignore_index=True)

    data = Data(time_series_df=time_series, settings=_hour_settings(interpolate_missing=True))

    assert list(data.loadshape.index) == ["a"]
    ledger = data.excluded_ids.to_dict("records")
    assert ledger == [{"id": "no_times", "reason": "no readings in any loadshape time"}], (
        f"expected one no-readings row for 'no_times', got {ledger}"
    )


def test_interpolation_on_keeps_meter_missing_some_times():
    """The null control: with interpolation on, a meter missing a few times is filled, not
    excluded."""
    time_series = pd.concat(
        [_hourly_frame("a", 48), _hourly_frame("short", 20)], ignore_index=True
    )

    data = Data(time_series_df=time_series, settings=_hour_settings(interpolate_missing=True))

    assert list(data.loadshape.index) == ["a", "short"]
    assert data.excluded_ids.empty


def test_all_meters_excluded_from_time_series_raises_with_reasons():
    """With every meter below the minimum or incomplete, building raises listing both reasons."""
    time_series = pd.concat(
        [_hourly_frame("x", 20), _hourly_frame("y", 10)], ignore_index=True
    )

    with pytest.raises(ValueError, match="No meters remain") as excinfo:
        Data(time_series_df=time_series, settings=_hour_settings(interpolate_missing=False))

    assert _INCOMPLETE_REASON in str(excinfo.value)
    assert _BELOW_MINIMUM_REASON in str(excinfo.value)


def test_all_meters_excluded_from_loadshape_df_raises_with_reasons():
    """A long loadshape with every meter below the minimum or incomplete raises listing both
    reasons."""
    loadshape_df = _long_loadshape_df(n_ids=2, n_time=24)
    loadshape_df = loadshape_df[loadshape_df["time"] != 5]
    loadshape_df = pd.concat(
        [loadshape_df, pd.DataFrame([{"id": "m2", "time": 24, "loadshape": 1.0}])]
    )

    with pytest.raises(ValueError, match="No meters remain") as excinfo:
        Data(loadshape_df=loadshape_df, settings=_loadshape_settings(interpolate_missing=False))

    assert _INCOMPLETE_REASON in str(excinfo.value)
    assert _BELOW_MINIMUM_REASON in str(excinfo.value)


def test_extend_concatenates_loadshapes():
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )
    data_a = Data(time_series_df=_hourly_frame("a", 48), settings=settings)
    data_b = Data(time_series_df=_hourly_frame("b", 48), settings=settings)

    data_a.extend(data_b)

    assert sorted(data_a.loadshape.index) == ["a", "b"]


def test_month_time_period_ingestion(_comstock_monthly_all):
    """The month period groups into 12 columns via its dedicated branch."""
    df_baseline, _ = _comstock_monthly_all
    ids = sorted(df_baseline.index.get_level_values("id").unique())[:3]
    time_series = df_baseline.loc[ids].reset_index()[["id", "datetime", "observed"]]
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.MONTH,
    )

    data = Data(time_series_df=time_series, settings=settings)

    assert data.loadshape.shape[1] == 12


def test_unstacked_loadshape_ingestion():
    """A wide loadshape (id + integer time columns) ingests via the unstacked path."""
    wide = pd.DataFrame({"id": ["a", "b"]})
    for t in range(1, 25):
        wide[t] = [float(t), float(t + 1)]

    data = Data(loadshape_df=wide)

    assert data.loadshape.shape == (2, 24)


def test_extend_rejects_mismatched_time_period(_comstock_hourly_all):
    df_baseline, _ = _comstock_hourly_all
    hour_data = Data(time_series_df=_hourly_frame("a", 48), settings=Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    ))
    loadshape_only = Data(loadshape_df=_long_loadshape_df(n_ids=1))

    with pytest.raises(ValueError):
        hour_data.extend(loadshape_only)


def test_features_with_nan_row_excluded():
    features = pd.DataFrame({"id": ["a", "b", "c"], "x": [1.0, np.nan, 3.0]})
    settings = Data_Settings(agg_type=None, loadshape_type=None, time_period=None)

    data = Data(features_df=features, settings=settings)

    assert sorted(data.features.index) == ["a", "c"]


def test_all_features_excluded_raises_with_reason():
    features = pd.DataFrame({"id": ["a", "b"], "x": [np.nan, np.nan]})
    settings = Data_Settings(agg_type=None, loadshape_type=None, time_period=None)

    with pytest.raises(ValueError, match="No meters remain in the features") as excinfo:
        Data(features_df=features, settings=settings)

    assert "null values in features_df" in str(excinfo.value)


def test_partial_loadshape_settings_raise():
    """If any of agg_type/loadshape_type/time_period is set, all must be set."""
    with pytest.raises(ValueError):
        Data_Settings(agg_type=None)  # loadshape_type/time_period keep non-None defaults


def test_interpolate_missing_defaults_to_false():
    assert Data_Settings().interpolate_missing is False


@pytest.mark.parametrize("interpolate_missing", [True, False])
def test_min_data_pct_required_default_kept_in_both_modes(interpolate_missing):
    settings = Data_Settings(interpolate_missing=interpolate_missing)

    assert settings.min_data_pct_required == 0.8


@pytest.mark.parametrize("interpolate_missing", [True, False])
@pytest.mark.parametrize("given, expected", [(None, 0.8), (0.5, 0.5), (1.0, 1.0)])
def test_min_data_pct_required_kept_in_both_modes(interpolate_missing, given, expected):
    """A given min_data_pct_required is kept whatever interpolate_missing is; None selects
    the default."""
    settings = Data_Settings(interpolate_missing=interpolate_missing, min_data_pct_required=given)

    assert settings.min_data_pct_required == expected, (
        f"interpolate_missing={interpolate_missing}, given {given}: "
        f"expected {expected}, got {settings.min_data_pct_required}"
    )


@pytest.mark.parametrize("value", [0, -0.1, 1.5])
def test_min_data_pct_required_outside_unit_interval_raises(value):
    with pytest.raises(ValueError, match="min_data_pct_required must be in"):
        Data_Settings(min_data_pct_required=value)


@pytest.mark.parametrize(
    "min_data_pct_required, expected_ids", [(0.5, ["a", "sparse"]), (0.8, ["a"])]
)
def test_min_data_pct_required_sets_which_meters_survive(min_data_pct_required, expected_ids):
    """A meter with 14 of 24 hours (58%) is filled at 0.5 and excluded at 0.8."""
    time_series = pd.concat(
        [_hourly_frame("a", 48), _hourly_frame("sparse", 14)], ignore_index=True
    )
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
        interpolate_missing=True,
        min_data_pct_required=min_data_pct_required,
    )

    data = Data(time_series_df=time_series, settings=settings)

    assert list(data.loadshape.index) == expected_ids
    assert list(data.excluded_ids["id"]) == sorted({"a", "sparse"} - set(expected_ids))


def test_season_dict_is_converted_to_definition():
    """The default season dict is coerced into a Season_Definition on validation."""
    assert not isinstance(Data_Settings().season, dict)


_TIME_LAYOUT_TABLE = [
    (_const.TimePeriod.HOUR, [("hour", 24)], 24),
    (_const.TimePeriod.DAY_OF_WEEK, [("day_of_week", 7)], 7),
    (_const.TimePeriod.HOURLY_DAY_OF_WEEK, [("day_of_week", 7), ("hour", 24)], 168),
    (_const.TimePeriod.WEEKDAY_WEEKEND, [("weekday_weekend", 2)], 2),
    (_const.TimePeriod.HOURLY_WEEKDAY_WEEKEND, [("weekday_weekend", 2), ("hour", 24)], 48),
    (_const.TimePeriod.MONTH, [("month", 12)], 12),
    (_const.TimePeriod.HOURLY_MONTH, [("month", 12), ("hour", 24)], 288),
    (_const.TimePeriod.SEASONAL_DAY_OF_WEEK, [("season", 3), ("day_of_week", 7)], 21),
    (
        _const.TimePeriod.SEASONAL_HOURLY_DAY_OF_WEEK,
        [("season", 3), ("day_of_week", 7), ("hour", 24)],
        504,
    ),
    (
        _const.TimePeriod.SEASONAL_WEEKDAY_WEEKEND,
        [("season", 3), ("weekday_weekend", 2)],
        6,
    ),
    (
        _const.TimePeriod.SEASONAL_HOURLY_WEEKDAY_WEEKEND,
        [("season", 3), ("weekday_weekend", 2), ("hour", 24)],
        144,
    ),
]


@pytest.mark.parametrize("time_period, expected_layout, expected_n_times", _TIME_LAYOUT_TABLE)
def test_time_layout_default_options(time_period, expected_layout, expected_n_times):
    """time_layout lists grouping columns in unique_time_periods order with default
    season/weekday_weekend cardinalities, and n_times is their product."""
    settings = Data_Settings(time_period=time_period)

    assert settings.time_layout == expected_layout, (
        f"{time_period.value}: expected {expected_layout}, got {settings.time_layout}"
    )
    assert settings.n_times == expected_n_times, (
        f"{time_period.value}: expected {expected_n_times}, got {settings.n_times}"
    )


def test_time_layout_covers_every_time_period():
    """The time layout table has one row per TimePeriod, each with a non-empty layout."""
    table_periods = [row[0] for row in _TIME_LAYOUT_TABLE]

    assert len(table_periods) == len(set(table_periods)), f"duplicated rows: {table_periods}"
    assert set(table_periods) == set(_const.TimePeriod), (
        f"missing from the table: {set(_const.TimePeriod) - set(table_periods)}"
    )
    assert all(row[1] for row in _TIME_LAYOUT_TABLE)


def test_day_of_year_time_period_raises():
    """day_of_year is not a TimePeriod, so settings reject it."""
    with pytest.raises(ValidationError, match="time_period"):
        Data_Settings(time_period="day_of_year")


def test_time_layout_four_season_definition():
    """A 4-option season definition sets the season cardinality to 4."""
    settings = Data_Settings(
        time_period=_const.TimePeriod.SEASONAL_HOURLY_DAY_OF_WEEK, season=_FOUR_SEASONS
    )

    assert settings.time_layout == [("season", 4), ("day_of_week", 7), ("hour", 24)]
    assert settings.n_times == 672


def test_time_layout_three_option_weekday_weekend():
    """A 3-option weekday_weekend definition sets its cardinality to 3."""
    weekday_weekend = {
        "monday": "weekday",
        "tuesday": "weekday",
        "wednesday": "weekday",
        "thursday": "weekday",
        "friday": "friday",
        "saturday": "weekend",
        "sunday": "weekend",
        "options": ["weekday", "friday", "weekend"],
    }
    settings = Data_Settings(
        time_period=_const.TimePeriod.HOURLY_WEEKDAY_WEEKEND, weekday_weekend=weekday_weekend
    )

    assert settings.time_layout == [("weekday_weekend", 3), ("hour", 24)]
    assert settings.n_times == 72


def test_time_layout_without_time_period_is_empty():
    settings = Data_Settings(agg_type=None, loadshape_type=None, time_period=None)

    assert settings.time_layout == []


def test_n_times_without_time_period_is_none():
    """Settings without a time_period define no loadshape time count."""
    settings = Data_Settings(agg_type=None, loadshape_type=None, time_period=None)

    assert settings.n_times is None, f"expected None, got {settings.n_times}"


def _year_frame(meter_id, values):
    """One meter's hourly observations over the non-leap year 2023."""
    return pd.DataFrame({"id": meter_id, "datetime": _YEAR_HOURS, "observed": values})


def _season_ordinal(datetimes, settings):
    labels = datetimes.dt.month.map(settings.season._num_dict)

    return labels.map(settings.season.options.index)


def _time_digits(datetimes, column, settings):
    """0-based time digit of each timestamp for one grouping column."""
    if column == "season":
        keys = _season_ordinal(datetimes, settings)
    elif column == "month":
        keys = datetimes.dt.month - 1
    elif column == "day_of_week":
        keys = datetimes.dt.day_of_week
    elif column == "weekday_weekend":
        labels = datetimes.dt.day_of_week.map(settings.weekday_weekend._num_dict)
        keys = labels.map(settings.weekday_weekend.options.index)
    elif column == "hour":
        keys = datetimes.dt.hour
    else:
        raise ValueError(f"Unknown time column {column!r}")

    return keys.rename(column)


def _brute_force_time_matrix(time_series, settings):
    """Reference loadshape matrix: each meter aggregated alone with a plain pandas groupby,
    its times laid out in lexicographic order of the grouping columns, missing times NaN."""
    columns = [column for column, _ in settings.time_layout]
    all_times = pd.MultiIndex.from_product(
        [range(cardinality) for _, cardinality in settings.time_layout], names=columns
    )
    rows = {}
    for meter_id, meter in time_series.groupby("id"):
        keys = [_time_digits(meter["datetime"], column, settings) for column in columns]
        aggregated = meter.groupby(keys)[settings.loadshape_type.value].agg(
            settings.agg_type.value
        )
        rows[meter_id] = aggregated.reindex(all_times).to_numpy()

    reference = pd.DataFrame.from_dict(
        rows, orient="index", columns=range(1, settings.n_times + 1)
    ).sort_index()

    return reference


def _observed_settings(**kwargs):
    return Data_Settings(loadshape_type=_const.LoadshapeType.OBSERVED, **kwargs)


def _drop_time(frame, settings, season, day_of_week, hour):
    """Remove every reading of one (season, day_of_week, hour) time."""
    has_time = (
        (_season_ordinal(frame["datetime"], settings) == season)
        & (frame["datetime"].dt.day_of_week == day_of_week)
        & (frame["datetime"].dt.hour == hour)
    )

    return frame[~has_time]


@pytest.mark.parametrize("drop_times", [False, True], ids=["complete", "missing_times"])
def test_constant_loads_keep_their_own_ids(drop_times):
    """With several meters, each keeps its own loadshape through interpolation: constant
    loads come back unchanged under their own id."""
    settings = _observed_settings(interpolate_missing=True)
    frames = []
    for offset, (meter_id, load) in enumerate([("c", 100.0), ("a", 300.0), ("b", 200.0)]):
        frame = _year_frame(meter_id, load)
        if drop_times:
            for season in range(3):
                frame = _drop_time(frame, settings, season, day_of_week=2, hour=offset + 3)

        frames.append(frame)

    data = Data(time_series_df=pd.concat(frames, ignore_index=True), settings=settings)

    loadshape = data.loadshape
    assert list(loadshape.index) == ["a", "b", "c"]
    assert loadshape.shape == (3, 504)
    for meter_id, load in [("a", 300.0), ("b", 200.0), ("c", 100.0)]:
        np.testing.assert_array_equal(
            loadshape.loc[meter_id].to_numpy(),
            np.full(504, load),
            err_msg=f"meter {meter_id} should be constant {load}",
        )


def test_fill_does_not_borrow_across_season_boundary():
    """A time missing at the edge of a season block takes its own season's edge value,
    never a value interpolated toward the neighbouring season."""
    settings = _observed_settings(interpolate_missing=True)
    season = _season_ordinal(pd.Series(_YEAR_HOURS), settings)
    frame = _year_frame("a", (season + 1).astype(float).to_numpy())
    # last time of season 0 (Sunday 23h) and first time of season 2 (Monday 0h)
    frame = _drop_time(frame, settings, season=0, day_of_week=6, hour=23)
    frame = _drop_time(frame, settings, season=2, day_of_week=0, hour=0)

    data = Data(time_series_df=frame, settings=settings)

    row = data.loadshape.loc["a"]
    assert row[168] == 1.0, f"season 0 edge time: expected 1.0, got {row[168]}"
    assert row[337] == 3.0, f"season 2 edge time: expected 3.0, got {row[337]}"


def test_meter_without_a_season_is_excluded():
    """A meter with no readings in a whole season falls below the minimum-data rule and
    is recorded in the ledger under the minimum-data reason."""
    settings = _observed_settings(interpolate_missing=True)
    gap = _year_frame("gap", 1.0)
    gap = gap[_season_ordinal(gap["datetime"], settings) != 1]
    time_series = pd.concat([_year_frame("full", 2.0), gap], ignore_index=True)

    data = Data(time_series_df=time_series, settings=settings)

    assert list(data.loadshape.index) == ["full"]
    ledger = data.excluded_ids.set_index("id")["reason"]
    assert ledger["gap"] == _BELOW_MINIMUM_REASON


def test_meter_without_a_season_is_excluded_below_minimum_data_rule():
    """A meter missing a whole season clears a low min_data_pct_required, so the empty
    season block left NaN by the fill is what excludes it."""
    settings = _observed_settings(interpolate_missing=True, min_data_pct_required=0.5)
    gap = _year_frame("gap", 1.0)
    gap = gap[_season_ordinal(gap["datetime"], settings) != 1]
    time_series = pd.concat([_year_frame("full", 2.0), gap], ignore_index=True)

    data = Data(time_series_df=time_series, settings=settings)

    assert list(data.loadshape.index) == ["full"]
    assert not data.loadshape.isna().any().any()
    assert data.excluded_ids.to_dict("records") == [
        {"id": "gap", "reason": "a season block has no readings"}
    ]


def test_long_loadshape_fills_flat():
    """With settings that carry no season, a long loadshape row is interpolated as one
    block, with the edge time taking its nearest reading."""
    long_df = _long_loadshape_df(n_ids=3, n_time=24)
    dropped = (long_df["id"] == "m1") & long_df["time"].isin([1, 5])
    long_df = long_df[~dropped]

    data = Data(loadshape_df=long_df, settings=_loadshape_settings(interpolate_missing=True))

    expected = pd.DataFrame(
        [[float(t + i) for t in range(1, 25)] for i in range(3)],
        index=pd.Index(["m0", "m1", "m2"], name="id"),
        columns=range(1, 25),
    )
    expected.loc["m1", 1] = 3.0
    pd.testing.assert_frame_equal(data.loadshape, expected, check_column_type=False)


def _zero_based_long_loadshape_without_time_0_for_m1():
    long_df = _long_loadshape_df(n_ids=2, n_time=24)
    long_df["time"] -= 1

    return long_df[~((long_df["id"] == "m1") & (long_df["time"] == 0))]


def _zero_based_settings(interpolate_missing):
    return Data_Settings(
        agg_type=None,
        loadshape_type=None,
        time_period=None,
        interpolate_missing=interpolate_missing,
    )


def test_long_loadshape_zero_based_time_fills_missing_time_0():
    """A long loadshape numbered from 0 keeps time 0 and fills a meter's missing time 0 flat."""
    long_df = _zero_based_long_loadshape_without_time_0_for_m1()

    data = Data(loadshape_df=long_df, settings=_zero_based_settings(interpolate_missing=True))

    expected = pd.DataFrame(
        [[float(t + 1 + i) for t in range(24)] for i in range(2)],
        index=pd.Index(["m0", "m1"], name="id"),
        columns=range(24),
    )
    expected.loc["m1", 0] = 3.0
    pd.testing.assert_frame_equal(data.loadshape, expected, check_column_type=False)
    assert data.excluded_ids.empty, data.excluded_ids


def test_long_loadshape_zero_based_time_interpolation_off_excludes_meter_lacking_time_0():
    """With interpolation off, a meter lacking time 0 is recorded as incomplete and the
    complete meter keeps its 0-based columns."""
    long_df = _zero_based_long_loadshape_without_time_0_for_m1()

    data = Data(loadshape_df=long_df, settings=_zero_based_settings(interpolate_missing=False))

    assert list(data.loadshape.index) == ["m0"]
    assert list(data.loadshape.columns) == list(range(24))
    assert data.excluded_ids.to_dict("records") == [{"id": "m1", "reason": _INCOMPLETE_REASON}]


def _long_loadshape_without_time_1():
    long_df = _long_loadshape_df(n_ids=2, n_time=24)

    return long_df[long_df["time"] != 1]


def test_long_loadshape_fills_time_1_every_meter_lacks():
    """A 1-based long loadshape whose time 1 every meter lacks keeps time 1, filled flat."""
    long_df = _long_loadshape_without_time_1()

    data = Data(loadshape_df=long_df, settings=_loadshape_settings(interpolate_missing=True))

    expected = pd.DataFrame(
        [[float(max(t, 2) + i) for t in range(1, 25)] for i in range(2)],
        index=pd.Index(["m0", "m1"], name="id"),
        columns=range(1, 25),
    )
    pd.testing.assert_frame_equal(data.loadshape, expected, check_column_type=False)
    assert data.excluded_ids.empty, data.excluded_ids


def test_long_loadshape_interpolation_off_excludes_meters_lacking_time_1():
    """With interpolation off, every meter lacking time 1 is recorded as incomplete."""
    long_df = _long_loadshape_without_time_1()

    with pytest.raises(ValueError, match=_INCOMPLETE_REASON):
        Data(loadshape_df=long_df, settings=_loadshape_settings(interpolate_missing=False))


def test_long_loadshape_null_time_raises():
    long_df = _long_loadshape_df(n_ids=2, n_time=24)
    long_df.loc[0, "time"] = np.nan

    with pytest.raises(ValueError, match="Loadshape times must not be null"):
        Data(loadshape_df=long_df, settings=_loadshape_settings(interpolate_missing=True))


def test_wide_loadshape_missing_column_is_added_and_filled():
    wide = pd.DataFrame({"id": ["b", "a"]})
    for t in range(1, 25):
        if t != 7:
            wide[t] = [2.0 * t, float(t)]

    data = Data(loadshape_df=wide, settings=_loadshape_settings(interpolate_missing=True))

    loadshape = data.loadshape
    assert loadshape.shape == (2, 24)
    assert list(loadshape.columns) == list(range(1, 25))
    assert list(loadshape.index) == ["a", "b"]
    assert loadshape.loc["a", 7] == 7.0
    assert loadshape.loc["b", 7] == 14.0


def test_wide_loadshape_keeps_zero_based_time_columns():
    """A 0-based hour-of-week loadshape keeps hour 0, with id as the index."""
    hours = range(168)
    wide = pd.DataFrame(
        [[300.0 + h for h in hours], [100.0 + h for h in hours]],
        index=pd.Index(["a", "b"], name="id"),
        columns=hours,
    )

    data = Data(loadshape_df=wide)

    pd.testing.assert_frame_equal(data.loadshape, wide, check_column_type=False)


def _wide_loadshape_df():
    wide = pd.DataFrame({"id": ["a", "b"]})
    for t in range(1, 25):
        wide[t] = [float(t), 2.0 * t]

    return wide


def test_wide_loadshape_interpolation_off_excludes_meter_missing_time():
    wide = _wide_loadshape_df()
    wide.loc[wide["id"] == "a", 7] = np.nan
    settings = Data_Settings(
        agg_type=None, loadshape_type=None, time_period=None, interpolate_missing=False
    )

    data = Data(loadshape_df=wide, settings=settings)

    assert list(data.loadshape.index) == ["b"]
    assert data.excluded_ids.to_dict("records") == [{"id": "a", "reason": _INCOMPLETE_REASON}]


def test_wide_loadshape_interpolation_off_missing_column_raises():
    wide = _wide_loadshape_df().drop(columns=7)
    settings = Data_Settings(
        agg_type=None, loadshape_type=None, time_period=None, interpolate_missing=False
    )

    with pytest.raises(ValueError, match=_INCOMPLETE_REASON):
        Data(loadshape_df=wide, settings=settings)


def test_wide_loadshape_empty_meter_is_excluded():
    """With interpolation on, the minimum-data rule applies to wide input too."""
    wide = _wide_loadshape_df()
    wide.loc[wide["id"] == "b", list(range(1, 25))] = np.nan

    data = Data(loadshape_df=wide, settings=_loadshape_settings(interpolate_missing=True))

    assert list(data.loadshape.index) == ["a"]
    assert data.excluded_ids.to_dict("records") == [{"id": "b", "reason": _BELOW_MINIMUM_REASON}]


def test_four_season_definition_has_672_times():
    settings = _observed_settings(season=_FOUR_SEASONS)

    data = Data(time_series_df=_year_frame("a", 1.0), settings=settings)

    assert data.loadshape.shape == (1, 4 * 7 * 24)


@pytest.mark.parametrize(
    "season_kwargs", [{}, {"season": _FOUR_SEASONS}], ids=["three_seasons", "four_seasons"]
)
@pytest.mark.parametrize("agg_type", list(_const.AggType))
def test_loadshape_matches_per_meter_brute_force(agg_type, season_kwargs):
    """Each meter's times equal a plain per-meter groupby, including a meter missing a
    whole time, which is filled in place rather than shifting the later times."""
    settings = _observed_settings(agg_type=agg_type, interpolate_missing=True, **season_kwargs)
    rng = np.random.default_rng(0)
    frames = []
    for offset, meter_id in enumerate(["c", "a", "b"]):
        frame = _year_frame(meter_id, rng.normal(10.0 * (offset + 1), 3.0, size=8760))
        frames.append(frame.sample(frac=0.97, random_state=offset).sort_values("datetime"))

    frames[2] = _drop_time(frames[2], settings, season=0, day_of_week=0, hour=5)
    time_series = pd.concat(frames, ignore_index=True)

    data = Data(time_series_df=time_series, settings=settings)

    expected = _brute_force_time_matrix(time_series, settings)
    assert np.isnan(expected.loc["b", 6]), "the dropped time should be missing in the reference"
    assert list(data.loadshape.index) == list(expected.index)
    assert list(data.loadshape.columns) == list(expected.columns)
    expected_filled = fill_missing(expected.to_numpy(), settings.time_layout)
    np.testing.assert_allclose(
        data.loadshape.to_numpy(), expected_filled, rtol=0, atol=1e-12
    )


def _smallest_abs_reading(time_series, column):
    """Reference dedupe: the smallest |value| of each (id, datetime), NaN last."""
    deduped = time_series.sort_values(by=column, key=abs, kind="stable").drop_duplicates(
        subset=["id", "datetime"], keep="first"
    )

    return deduped


@pytest.mark.parametrize(
    "season_kwargs", [{}, {"season": _FOUR_SEASONS}], ids=["three_seasons", "four_seasons"]
)
@pytest.mark.parametrize("agg_type", list(_const.AggType))
def test_shuffled_duplicated_readings_match_per_meter_brute_force(agg_type, season_kwargs):
    """Shuffled readings with NaNs, a missing time, a meter without a season and duplicated
    (id, datetime) rows (an exact tie, a larger value, a NaN) aggregate to the per-meter
    reference of the smallest-|value| readings, and only the seasonless meter is excluded."""
    settings = _observed_settings(agg_type=agg_type, interpolate_missing=True, **season_kwargs)
    rng = np.random.default_rng(1)
    frames = []
    for offset, meter_id in enumerate(["c", "a", "b"]):
        frame = _year_frame(meter_id, rng.normal(10.0 * (offset + 1), 3.0, size=8760))
        frame.loc[frame.sample(frac=0.01, random_state=offset).index, "observed"] = np.nan
        frames.append(frame)

    frames[2] = _drop_time(frames[2], settings, season=0, day_of_week=0, hour=5)
    gap = _year_frame("gap", 1.0)
    frames.append(gap[_season_ordinal(gap["datetime"], settings) != 1])
    readings = pd.concat(frames, ignore_index=True)
    duplicates = readings.sample(frac=0.02, random_state=0)
    time_series = pd.concat(
        [
            readings,
            duplicates,
            duplicates.assign(observed=duplicates["observed"] + 5.0),
            duplicates.assign(observed=np.nan),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=0)

    data = Data(time_series_df=time_series, settings=settings)

    deduped = _smallest_abs_reading(time_series, "observed")
    expected = _brute_force_time_matrix(deduped[deduped["id"] != "gap"], settings)
    assert list(data.loadshape.index) == ["a", "b", "c"]
    assert list(data.loadshape.columns) == list(expected.columns)
    expected_filled = fill_missing(expected.to_numpy(), settings.time_layout)
    np.testing.assert_allclose(
        data.loadshape.to_numpy(), expected_filled, rtol=0, atol=1e-12
    )
    assert data.excluded_ids.to_dict("records") == [
        {"id": "gap", "reason": _BELOW_MINIMUM_REASON}
    ]


def test_observed_basis_duplicate_readings_keep_the_smallest_abs_value():
    """Duplicated (id, datetime) readings of the observed basis collapse to the reading with
    the smallest |value| before aggregation: an exact tie and a larger value leave the
    reading as is, a NaN never wins over a finite reading, and the sign is kept."""
    frame = _hourly_frame("m", 24)
    frame.loc[7, "observed"] = np.nan
    duplicates = pd.DataFrame(
        {
            "id": "m",
            "datetime": frame["datetime"].iloc[[5, 5, 5, 7, 9]].to_numpy(),
            "observed": [5.0, 10.0, np.nan, 4.0, -2.0],
        }
    )
    time_series = pd.concat([frame, duplicates], ignore_index=True)

    settings = _hour_settings(interpolate_missing=False)

    loadshape = Data(time_series_df=time_series, settings=settings).loadshape

    for hour, expected in [(5, 5.0), (7, 4.0), (9, -2.0), (0, 0.0)]:
        actual = loadshape.loc["m", hour + 1]
        assert actual == expected, f"hour {hour}: expected {expected}, got {actual}"


def test_interpolation_off_ledger_has_one_null_row_per_id():
    """With interpolation off, a meter with two all-NaN times gets one null-readings row and
    no incomplete row, a meter missing a time gets one incomplete row, and the complete
    meter is kept."""
    nan_meter = _hourly_frame("nan", 48)
    nan_meter.loc[nan_meter["datetime"].dt.hour.isin([0, 1]), "observed"] = np.nan
    short = _hourly_frame("short", 48)
    short = short[short["datetime"].dt.hour != 5]
    time_series = pd.concat(
        [_hourly_frame("a", 48), nan_meter, short], ignore_index=True
    ).sample(frac=1.0, random_state=0)

    data = Data(time_series_df=time_series, settings=_hour_settings(interpolate_missing=False))

    assert list(data.loadshape.index) == ["a"]
    assert data.excluded_ids.to_dict("records") == [
        {"id": "nan", "reason": "only null readings in a loadshape time"},
        {"id": "short", "reason": _INCOMPLETE_REASON},
    ]


def test_fill_missing_within_blocks():
    """Edges take the nearest reading, interior gaps are linear, an empty block and an
    empty row stay NaN, and the input is not modified."""
    nan = np.nan
    matrix = np.array(
        [
            [nan, 1.0, nan, 3.0, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan],
        ]
    )
    original = matrix.copy()

    filled = fill_missing(matrix, [("season", 2), ("hour", 4)])

    expected = np.array(
        [
            [1.0, 1.0, 2.0, 3.0, nan, nan, nan, nan],
            [nan, nan, nan, nan, nan, nan, nan, nan],
        ]
    )
    np.testing.assert_array_equal(filled, expected)
    np.testing.assert_array_equal(matrix, original)


def test_fill_missing_without_season_is_flat():
    matrix = np.array([[np.nan, 2.0, np.nan, np.nan, 8.0, np.nan]])

    filled = fill_missing(matrix, [])

    np.testing.assert_array_equal(filled, [[2.0, 2.0, 4.0, 6.0, 8.0, 8.0]])


def test_keep_smallest_abs_per_duplicate_group():
    """Each duplicated (id, time) group keeps its smallest |value| row, NaN sorts last, a
    tie keeps the earliest row, and unduplicated rows pass through."""
    df = pd.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "c", "c", "d"],
            "time": [1, 1, 1, 1, 1, 1, 1, 1],
            "loadshape": [-5.0, 2.0, np.nan, np.nan, 7.0, 3.0, -3.0, 9.0],
            "row": [0, 1, 2, 3, 4, 5, 6, 7],
        }
    )

    kept = _keep_smallest_abs(df, "loadshape", ["id", "time"])

    assert sorted(kept["row"]) == [1, 4, 5, 7], f"kept rows {sorted(kept['row'])}"


def test_error_basis_duplicate_reading_keeps_the_smallest_error():
    """For the error basis computed from observed and modeled, two readings of the same
    meter-hour are duplicates even when their observed values differ: the one with the
    smaller |error| is kept rather than both being averaged into the cell."""
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.ERROR,
        time_period=_const.TimePeriod.HOUR,
    )
    frame = _hourly_frame("m", 24)
    frame["observed"] = 1.0
    frame["modeled"] = 2.0  # error = 1 - 1/2 = 0.5 in every hour
    duplicate = frame.iloc[[5]].assign(observed=1.9)  # error = 0.05 for the same hour
    time_series = pd.concat([frame, duplicate], ignore_index=True)

    loadshape = Data(time_series_df=time_series, settings=settings).loadshape

    assert loadshape.loc["m", 6] == pytest.approx(0.05), (
        f"hour 5 should keep the smaller |error| 0.05, got {loadshape.loc['m', 6]} "
        "(0.275 means both readings were averaged)"
    )
    assert loadshape.loc["m", 1] == pytest.approx(0.5)
