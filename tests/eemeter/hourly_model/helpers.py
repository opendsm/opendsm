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

"""Synthetic hourly data builders and assertion helpers for the hourly-model tests."""

import numpy as np
import pandas as pd



def create_hourly_dataframe(
    start="2019-01-01",
    end="2019-12-31",
    tz="America/New_York",
    include_ghi=False,
    observed_mean=5.0,
    temperature_mean=60.0,
    add_noise=True,
    seed=0,
):
    """Hourly ``observed`` and ``temperature`` (plus ``ghi`` when ``include_ghi``) from ``start``
    through ``end`` 00:00 inclusive; ``seed`` fixes the noise."""
    rng = np.random.default_rng(seed)
    start_dt = pd.Timestamp(start, tz=tz)
    end_dt = pd.Timestamp(end, tz=tz)
    index = pd.date_range(start_dt, end_dt, freq="h")

    n = len(index)

    day_of_year = index.day_of_year
    hour_of_day = index.hour
    seasonal = 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    daily = 10 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    noise = rng.normal(0, 3, n) if add_noise else 0
    temperature = temperature_mean + seasonal + daily + noise

    base_load = observed_mean
    temp_correlation = 0.02 * (temperature - temperature_mean)
    obs_noise = rng.normal(0, 0.5, n) if add_noise else 0
    observed = np.maximum(0, base_load + temp_correlation + obs_noise)

    df = pd.DataFrame(
        {"observed": observed, "temperature": temperature}, index=index
    )

    if include_ghi:
        is_daytime = (hour_of_day >= 6) & (hour_of_day <= 18)
        sun_angle = np.sin(np.pi * (hour_of_day - 6) / 12)
        seasonal_ghi = 300 + 200 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        ghi = np.where(is_daytime, seasonal_ghi * sun_angle, 0)
        df["ghi"] = ghi

    return df


def create_repeated_values(df, repeat_pct):
    """Copy of ``df`` whose first ``repeat_pct`` fraction of rows (0.75 for 75%) take the first
    ``observed`` value."""
    df = df.copy()
    n_repeat = int(len(df) * repeat_pct)
    repeated_value = df["observed"].iloc[0]
    df.loc[df.index[:n_repeat], "observed"] = repeated_value

    return df


def create_extreme_values(df, n_extreme, column="observed", seed=0):
    """Copy of ``df`` with ``n_extreme`` rows of ``column``, chosen by ``seed``, overwritten with
    twice the Q3 + 3*IQR fence."""
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr

    extreme_indices = np.random.default_rng(seed).choice(len(df), n_extreme, replace=False)
    df.loc[df.index[extreme_indices], column] = upper_bound * 2

    return df


def assert_has_disqualification(data, qualified_name):
    names = {dq.qualified_name for dq in data.disqualification}
    assert qualified_name in names, f"{qualified_name} not in {names}"


def assert_has_warning(data, qualified_name):
    names = {w.qualified_name for w in data.warnings}
    assert qualified_name in names, f"{qualified_name} not in {names}"


def assert_column_exists(df, column):
    assert column in df.columns, f"Column {column} not found in {list(df.columns)}"


def assert_all_hours_present(df):
    """Assert the index steps by exactly one hour between consecutive rows, so it has no gap and
    no repeated instant."""
    steps = df.index[1:] - df.index[:-1]
    off = np.flatnonzero(steps != pd.Timedelta("1h"))
    assert len(off) == 0, (
        f"{len(off)} index steps are not one hour, first at row {off[0]}: {steps[off[0]]}"
    )
