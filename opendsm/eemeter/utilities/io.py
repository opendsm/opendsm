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

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
from pandas._typing import (
    CompressionOptions,
    CSVEngine,
    DtypeArg,
    DtypeBackend,
    FilePath,
    IndexLabel,
    ReadCsvBuffer,
    StorageOptions,
    WriteBuffer,
)

__all__ = (
    "meter_data_from_csv",
    "meter_data_from_json",
    "meter_data_to_csv",
    "temperature_data_from_csv",
    "temperature_data_from_json",
    "temperature_data_to_csv",
)


def meter_data_from_csv(
    filepath_or_buffer: str | FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    tz: str | datetime.tzinfo | None = None,
    start_col: str = "start",
    value_col: str = "value",
    gzipped: bool = False,
    freq: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load meter data from a CSV file and convert to a dataframe.

    Note: This is an example of the default csv structure assumed.
        ```python
        start,value
        2017-01-01T00:00:00+00:00,0.31
        2017-01-02T00:00:00+00:00,0.4
        2017-01-03T00:00:00+00:00,0.58
        ```

    Args:
        filepath_or_buffer: File path or object.
        tz: Timezone represented in the meter data. Ex: `UTC` or `US/Pacific`
        start_col: Date period start column.
        value_col: Value column, can be in any unit.
        gzipped: Whether file is gzipped.
        freq: If given, apply frequency to data using `pandas.DataFrame.resample`. One of `['hourly', 'daily']`.
        **kwargs: Extra keyword arguments to pass to `pandas.read_csv`, such as `sep='|'`.
    """

    read_csv_kwargs = {
        "usecols": [start_col, value_col],
        "dtype": {value_col: np.float64},
        "parse_dates": [start_col],
        "index_col": start_col,
    }

    if gzipped:
        read_csv_kwargs.update({"compression": "gzip"})

    # allow passing extra kwargs
    read_csv_kwargs.update(kwargs)

    df = pd.read_csv(filepath_or_buffer, **read_csv_kwargs)
    df.index = pd.to_datetime(df.index, utc=True)

    # for pandas<0.24, which doesn't localize even with utc=True
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")  # pragma: no cover

    if tz is not None:
        df = df.tz_convert(tz)

    if freq == "hourly":
        df = df.resample("h").sum(min_count=1)
    elif freq == "daily":
        df = df.resample("D").sum(min_count=1)

    return df


def temperature_data_from_csv(
    filepath_or_buffer: str | FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    tz: str | datetime.tzinfo | None = None,
    date_col: str = "dt",
    temp_col: str = "tempF",
    gzipped: bool = False,
    freq: str | None = None,
    **kwargs,
):
    """Load meter data from a CSV file and convert to a dataframe. Farenheit is assumed for building models.

    Note: This is an example of the default csv structure assumed.
        ```python
        dt,tempF
        2017-01-01T00:00:00+00:00,21
        2017-01-01T01:00:00+00:00,22.5
        2017-01-01T02:00:00+00:00,23.5
        ```

    Args:
        filepath_or_buffer: File path or object.
        tz: Timezone represented in the meter data. Ex: `UTC` or `US/Pacific`
        date_col: Date period start column.
        temp_col: Temperature column.
        gzipped: Whether file is gzipped.
        freq: If given, apply frequency to data using `pandas.DataFrame.resample`. One of `['hourly', 'daily']`.
        **kwargs: Extra keyword arguments to pass to `pandas.read_csv`, such as `sep='|'`.
    """
    read_csv_kwargs = {
        "usecols": [date_col, temp_col],
        "dtype": {temp_col: np.float64},
        "parse_dates": [date_col],
        "index_col": date_col,
    }

    if gzipped:
        read_csv_kwargs.update({"compression": "gzip"})

    # allow passing extra kwargs
    read_csv_kwargs.update(kwargs)

    df = pd.read_csv(filepath_or_buffer, **read_csv_kwargs)
    df.index = pd.to_datetime(df.index, utc=True)

    # for pandas<0.24, which doesn't localize even with utc=True
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")  # pragma: no cover

    if tz is not None:
        df = df.tz_convert(tz)

    if freq == "hourly":
        df = df.resample("h").sum(min_count=1)

    return df[temp_col]


def meter_data_from_json(data: list, orient: str = "list") -> pd.DataFrame:
    """Load meter data from a list of dictionary objects or a list of lists.

    Args:
        data: A list of meter data, with each row representing a single record.
        orient: Format of `data` parameter. Must be one of `['list', 'records']`.
            `'list'` is a list of lists, with the first element as start date and the second element as meter usage. `'records'` is a list of dicts.

    Note: This is an example of the default `list` structure.
        ```python
        [
            ['2017-01-01T00:00:00+00:00', 3.5],
            ['2017-02-01T00:00:00+00:00', 0.4],
            ['2017-03-01T00:00:00+00:00', 0.46],
        ]
        ```

    Note: This is an example of the `records` structure.
        ```python
        [
            {'start': '2017-01-01T00:00:00+00:00', 'value': 3.5},
            {'start': '2017-02-01T00:00:00+00:00', 'value': 0.4},
            {'start': '2017-03-01T00:00:00+00:00', 'value': 0.46},
        ]
        ```

    Returns:
        DataFrame with a single column (``'value'``) and a `pandas.DatetimeIndex`. A second column (``'estimated'``) may also be included if the input data contained an estimated boolean flag.
    """

    def _empty_meter_data_dataframe():
        return pd.DataFrame(
            {"value": []}, index=pd.DatetimeIndex([], tz="UTC", name="start")
        )

    if data is None:
        return _empty_meter_data_dataframe()

    if orient == "list":
        df = pd.DataFrame(data, columns=["start", "value"])
        df["start"] = pd.to_datetime(df.start, utc=True)
        df = df.set_index("start")
        return df
    elif orient == "records":

        def _noneify_meter_data_row(row):
            value = row["value"]
            if value is not None:
                try:
                    value = float(value)
                except ValueError:
                    value = None
            out_row = {"start": row["start"], "value": value}
            if "estimated" in row:
                estimated = row.get("estimated")
                out_row["estimated"] = estimated in [True, "true", "True", 1, "1"]
            return out_row

        noneified_data = [_noneify_meter_data_row(row) for row in data]
        df = pd.DataFrame(noneified_data)
        if df.empty:
            return _empty_meter_data_dataframe()
        df["start"] = pd.to_datetime(df.start, utc=True)
        df = df.set_index("start")
        df["value"] = df["value"].astype(float)
        if "estimated" in df.columns:
            df["estimated"] = (
                df["estimated"].where(df["estimated"].notna(), False).astype(bool)
            )
        return df
    else:
        raise ValueError("orientation not recognized.")


def temperature_data_from_json(data: list, orient: str = "list") -> pd.Series:
    """Load temperature data from json to a Series. Farenheit is assumed for building models.

    Args:
        data: A list of temperature data, with each row representing a single record.
        orient: Format of `data` parameter. Must be `'list'`.
            `'list'` is a list of lists, with the first element as start date and the second element as temperature.

    Note: This is an example of the default `list` structure.
        ```python
        [
            ['2017-01-01T00:00:00+00:00', 3.5],
            ['2017-01-01T01:00:00+00:00', 5.4],
            ['2017-01-01T02:00:00+00:00', 7.4],
        ]
        ```

    Returns:
        DataFrame with a single column (``'tempF'``) and a `pandas.DatetimeIndex`.

    Raises:
        ValueError: If `orient` is not `'list'`.
    """
    if orient == "list":
        df = pd.DataFrame(data, columns=["dt", "tempF"])
        series = df.tempF
        series.index = pd.to_datetime(df.dt, utc=True)
        return series
    else:
        raise ValueError("orientation not recognized.")


def meter_data_to_csv(
    meter_data: pd.DataFrame | pd.Series,
    path_or_buf: str | FilePath | WriteBuffer[bytes] | WriteBuffer[str],
) -> None:
    """Write meter data from a DataFrame or Series to a CSV. See also `pandas.DataFrame.to_csv`.

    Args:
        meter_data: DataFrame or Series with a ``'value'`` column and a `pandas.DatetimeIndex`.
        path_or_buf: Path or file handle.
    """
    if meter_data.index.name is None:
        meter_data.index.name = "start"

    return meter_data.to_csv(path_or_buf, index=True)


def temperature_data_to_csv(
    temperature_data: pd.Series,
    path_or_buf: str | FilePath | WriteBuffer[bytes] | WriteBuffer[str],
) -> None:
    """Write temperature data to CSV. See also :any:`pandas.DataFrame.to_csv`.

    Args:
        temperature_data: Temperature data series with :any:`pandas.DatetimeIndex`.
        path_or_buf: Path or file handle.
    """
    if temperature_data.index.name is None:
        temperature_data.index.name = "dt"
    if temperature_data.name is None:
        temperature_data.name = "temperature"

    return temperature_data.to_frame().to_csv(path_or_buf, index=True)
