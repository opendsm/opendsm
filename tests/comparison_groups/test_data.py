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

from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const



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
    """Regression: TimePeriod.HOUR must be a valid key in the row-count and
    granularity dicts. The dicts previously keyed the standalone hour period as
    'hourly', mismatching the enum value 'hour', so any lookup raised KeyError."""
    assert _const.time_period_row_counts[_const.TimePeriod.HOUR] == 24
    assert _const.min_granularity_per_time_period[_const.TimePeriod.HOUR] == 60


def test_time_series_ingestion_hourly(_comstock_hourly_all):
    """Regression: building Data from a time series exercises loadshape_type /
    time_period reads (previously uppercased -> AttributeError) and the HOUR
    row-count lookup (previously KeyError). Should produce a 24-column loadshape."""
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
    """Regression: comparison-pool trimming is reproducible when a seed is set.
    It previously used an unseeded global np.random.choice."""
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


def test_partial_loadshape_settings_raise():
    """If any of agg_type/loadshape_type/time_period is set, all must be set."""
    with pytest.raises(ValueError):
        Data_Settings(agg_type=None)  # loadshape_type/time_period keep non-None defaults


def test_interpolate_missing_controls_min_data_pct():
    assert Data_Settings(interpolate_missing=True).min_data_pct_required is not None
    assert Data_Settings(interpolate_missing=False).min_data_pct_required is None


def test_season_dict_is_converted_to_definition():
    """The default season dict is coerced into a Season_Definition on validation."""
    assert not isinstance(Data_Settings().season, dict)
