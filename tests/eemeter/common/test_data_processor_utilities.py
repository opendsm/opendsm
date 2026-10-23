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

from pandas.tseries.offsets import Day

from opendsm.eemeter.common.data_processor_utilities import (
    as_freq,
    compute_data_frequency,
    frequency_duration,
    require_dataframe,
    resolve_is_electricity_data,
    trim_edge_rows,
)



# day_counts, remove_duplicates and as_freq are tested in test_transform.py.

@pytest.mark.parametrize(
    "freq,periods,expected",
    [
        ("h", 48, "hourly"),
        ("D", 60, "daily"),
        ("MS", 12, "billing_monthly"),
        ("2MS", 6, "billing_bimonthly"),
    ],
)
def test_compute_minimum_data_frequency_from_regular_index(freq, periods, expected):
    """A regularly-spaced index resolves to the matching data frequency label."""
    index = pd.date_range("2020-01-01", periods=periods, freq=freq)

    assert compute_data_frequency(index.copy(), default_data_frequency="daily") == expected


def test_compute_minimum_data_frequency_single_point_returns_default():
    """A length-1 index has no spacing, so the default data frequency is returned."""
    index = pd.date_range("2020-01-01", periods=1, freq="D")

    assert compute_data_frequency(index, default_data_frequency="billing_monthly") == "billing_monthly"


def test_compute_minimum_data_frequency_irregular_index_uses_median_spacing():
    """An index with no inferrable frequency falls back to the median day spacing."""
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-05"])

    assert compute_data_frequency(index, default_data_frequency="daily") == "daily"


def test_compute_minimum_data_frequency_irregular_monthly_spacing():
    """Irregular reads averaging ~monthly spacing resolve to billing_monthly."""
    index = pd.to_datetime(
        ["2020-01-01", "2020-02-03", "2020-03-01", "2020-04-05", "2020-04-28"]
    )

    assert compute_data_frequency(index, default_data_frequency="daily") == "billing_monthly"


@pytest.mark.parametrize(
    "freq,expected",
    [
        ("15min", pd.Timedelta(minutes=15)),
        ("h", pd.Timedelta(hours=1)),
        ("D", pd.Timedelta(days=1)),
        ("2D", pd.Timedelta(days=2)),
        (Day(3), pd.Timedelta(days=3)),
    ],
)
def test_frequency_duration_of_fixed_frequencies(freq, expected):
    assert frequency_duration(freq) == expected, f"{freq!r} -> {frequency_duration(freq)}"


@pytest.mark.parametrize("freq", ["MS", "ME", "2MS", "QS", "YS"])
def test_frequency_duration_rejects_calendar_offsets(freq):
    with pytest.raises(TypeError):
        frequency_duration(freq)


def test_as_freq_daily_bins_start_at_the_first_observation():
    index = pd.date_range("2024-01-01 05:00", periods=72, freq="h", tz="America/Los_Angeles")
    series = pd.Series(1.0, index=index)

    resampled = as_freq(series, "D", series_type="instantaneous")

    assert list(resampled.index.hour) == [5, 5, 5], f"bin starts: {resampled.index}"
    assert resampled.tolist() == [1.0, 1.0, 1.0]


def test_as_freq_daily_bins_follow_wall_clock_across_dst():
    """Days anchored at 02:00 stay at 02:00 wall-clock time through a daylight-saving
    transition; the one label that would fall on the nonexistent 02:00 is pushed to
    03:00, and the days around the transition are 23 and 24 hours long."""
    # 95 hourly readings from 02:00 on the day before the transition end at 01:00 on the
    # fourth day; the final reading has no following interval and contributes nothing
    index = pd.date_range("2020-03-07 02:00", periods=95, freq="h", tz="America/New_York")
    series = pd.Series(1.0, index=index)

    resampled = as_freq(series, "D", atomic_freq="1h")

    assert list(resampled.index.hour) == [2, 3, 2, 2], f"bin starts: {resampled.index}"
    assert resampled.tolist() == [24.0, 23.0, 24.0, 23.0], f"daily sums: {resampled.tolist()}"



def _edge_frame(n_periods=6):
    index = pd.date_range("2020-01-01", periods=n_periods, freq="D", tz="UTC")

    return pd.DataFrame(
        {"observed": 1.0, "temperature": 50.0, "ghi": 200.0}, index=index
    )


def test_trim_edge_rows_drops_incomplete_leading_and_trailing_rows():
    """Leading rows missing observed and trailing rows missing a trailing column are dropped."""
    df = _edge_frame()
    df.iloc[0, df.columns.get_loc("observed")] = float("nan")
    df.iloc[-1, df.columns.get_loc("ghi")] = float("nan")

    trimmed, n_leading, n_trailing = trim_edge_rows(df, ("observed", "temperature", "ghi"))

    assert (n_leading, n_trailing) == (1, 1), f"expected (1, 1), got ({n_leading}, {n_trailing})"
    assert trimmed.index.equals(df.index[1:-1]), f"unexpected trimmed index {trimmed.index}"


def test_trim_edge_rows_keeps_interior_missing_rows():
    """Interior rows missing every value survive trimming."""
    df = _edge_frame()
    df.iloc[2:4, :] = float("nan")

    trimmed, n_leading, n_trailing = trim_edge_rows(df, ("observed", "temperature", "ghi"))

    assert (n_leading, n_trailing) == (0, 0), f"expected (0, 0), got ({n_leading}, {n_trailing})"
    assert trimmed.index.equals(df.index), "interior missing rows were dropped"


def test_trim_edge_rows_keeps_interior_missing_rows_of_unsorted_index():
    """Edges are found in time order, so an interior missing row stored first is kept."""
    df = _edge_frame()
    df.iloc[2, df.columns.get_loc("observed")] = float("nan")
    shuffled = df.iloc[[2, 0, 5, 4, 1, 3]]

    trimmed, n_leading, n_trailing = trim_edge_rows(shuffled, ("observed", "temperature"))

    assert (n_leading, n_trailing) == (0, 0), f"expected (0, 0), got ({n_leading}, {n_trailing})"
    assert trimmed.index.equals(df.index), f"expected the time-sorted index, got {trimmed.index}"


def test_trim_edge_rows_trims_time_edges_of_unsorted_datetime_column():
    """With a 'datetime' column, the earliest and latest times are the edges, not row order."""
    df = _edge_frame()
    df.iloc[0, df.columns.get_loc("observed")] = float("nan")
    df.iloc[-1, df.columns.get_loc("ghi")] = float("nan")
    shuffled = df.iloc[[5, 2, 0, 3, 1, 4]].rename_axis("datetime").reset_index()

    trimmed, n_leading, n_trailing = trim_edge_rows(shuffled, ("observed", "temperature", "ghi"))

    assert (n_leading, n_trailing) == (1, 1), f"expected (1, 1), got ({n_leading}, {n_trailing})"
    assert list(trimmed["datetime"]) == list(df.index[1:-1]), (
        f"unexpected trimmed times {list(trimmed['datetime'])}"
    )


def test_trim_edge_rows_returns_frame_untouched_when_no_row_qualifies():
    """A frame with no usable row is passed through for the caller's sufficiency check."""
    df = _edge_frame()
    df["observed"] = float("nan")

    trimmed, n_leading, n_trailing = trim_edge_rows(df, ("observed", "temperature", "ghi"))

    assert (n_leading, n_trailing) == (0, 0), f"expected (0, 0), got ({n_leading}, {n_trailing})"
    assert trimmed.index.equals(df.index), "an unusable frame was trimmed to an empty frame"


def test_trim_edge_rows_returns_frame_untouched_when_a_column_is_absent():
    """A frame missing a trimmable column is passed through rather than raising KeyError."""
    df = _edge_frame().drop(columns=["observed"])

    trimmed, n_leading, n_trailing = trim_edge_rows(df, ("observed", "temperature", "ghi"))

    assert (n_leading, n_trailing) == (0, 0), f"expected (0, 0), got ({n_leading}, {n_trailing})"
    assert trimmed.index.equals(df.index), "a frame missing a column was modified"


def test_require_dataframe_accepts_a_dataframe():
    """A DataFrame passes the check without raising."""
    require_dataframe(pd.DataFrame({"a": [1.0]}), "df")


@pytest.mark.parametrize(
    "value,type_name",
    [
        (None, "NoneType"),
        ({"a": [1.0]}, "dict"),
        (pd.Series([1.0]), "Series"),
        (np.array([1.0]), "ndarray"),
    ],
    ids=["none", "dict", "series", "ndarray"],
)
def test_require_dataframe_rejects_non_frames(value, type_name):
    """Anything other than a DataFrame raises TypeError naming the argument and the type."""
    with pytest.raises(TypeError, match=f"^df must be a pandas DataFrame, got {type_name}\\.$"):
        require_dataframe(value, "df")


@pytest.mark.parametrize("stored", [True, False])
def test_resolve_is_electricity_data_uses_the_stored_flag(stored):
    """A payload's stored flag is used as is, with no warning."""
    value, warning = resolve_is_electricity_data(stored, None)

    assert value is stored
    assert warning is None


@pytest.mark.parametrize("supplied", [True, False])
def test_resolve_is_electricity_data_uses_the_supplied_flag(supplied):
    """A payload storing no flag takes the supplied one."""
    value, _ = resolve_is_electricity_data(None, supplied)

    assert value is supplied


def test_resolve_is_electricity_data_warns_when_the_supplied_flag_is_used():
    """Using the supplied flag yields a prior_format warning carrying that flag."""
    _, warning = resolve_is_electricity_data(None, False)

    assert warning.qualified_name == "eemeter.serialization.prior_format"
    assert warning.description == (
        "Model payload does not store 'is_electricity_data', so the supplied value was used."
    )
    assert warning.data == {"is_electricity_data": False}


def test_resolve_is_electricity_data_rejects_both():
    """A supplied flag is rejected when the payload stores one."""
    with pytest.raises(ValueError, match="This model stores 'is_electricity_data'"):
        resolve_is_electricity_data(True, True)


def test_resolve_is_electricity_data_rejects_neither():
    """A payload storing no flag requires a supplied one."""
    with pytest.raises(ValueError, match="This model does not store 'is_electricity_data'"):
        resolve_is_electricity_data(None, None)
