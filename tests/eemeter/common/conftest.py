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

import pandas as pd
import pytest



def _utc_meter(df):
    out = df[["observed"]].rename(columns={"observed": "value"}).copy()
    out.index = out.index.tz_convert("UTC")

    return out


def _utc_temperature(df):
    out = df["temperature"].copy()
    out.index = out.index.tz_convert("UTC")

    return out


@pytest.fixture
def monthly_meter(comstock_monthly):
    df_b, df_r = comstock_monthly
    return _utc_meter(pd.concat([df_b, df_r]).dropna(subset=["observed"]))


@pytest.fixture
def monthly_temperature(comstock_hourly):
    df_b, df_r = comstock_hourly
    return _utc_temperature(pd.concat([df_b, df_r]).asfreq("h"))


@pytest.fixture
def daily_meter(comstock_daily):
    df_b, df_r = comstock_daily
    return _utc_meter(pd.concat([df_b, df_r]))


@pytest.fixture
def daily_temperature(comstock_hourly):
    df_b, df_r = comstock_hourly
    return _utc_temperature(pd.concat([df_b, df_r]).asfreq("h"))


@pytest.fixture
def hourly_meter(comstock_hourly):
    df_b, df_r = comstock_hourly
    return _utc_meter(pd.concat([df_b, df_r]).asfreq("h"))


@pytest.fixture
def hourly_temperature(comstock_hourly):
    df_b, df_r = comstock_hourly
    return _utc_temperature(pd.concat([df_b, df_r]).asfreq("h"))
