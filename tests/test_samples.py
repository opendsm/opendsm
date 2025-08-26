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
import datetime

import pytest
import pytz

from opendsm.eemeter.samples import samples, load_sample


def test_samples():
    assert samples() == [
        "il-electricity-cdd-hdd-billing_bimonthly",
        "il-electricity-cdd-hdd-billing_monthly",
        "il-electricity-cdd-hdd-daily",
        "il-electricity-cdd-hdd-hourly",
        "il-electricity-cdd-only-billing_bimonthly",
        "il-electricity-cdd-only-billing_monthly",
        "il-electricity-cdd-only-daily",
        "il-electricity-cdd-only-hourly",
        "il-gas-hdd-only-billing_bimonthly",
        "il-gas-hdd-only-billing_monthly",
        "il-gas-hdd-only-daily",
        "il-gas-hdd-only-hourly",
        "il-gas-intercept-only-billing_bimonthly",
        "il-gas-intercept-only-billing_monthly",
        "il-gas-intercept-only-daily",
        "il-gas-intercept-only-hourly",
        "uk-electricity-hdd-only-hourly-sample-0",
        "uk-electricity-hdd-only-hourly-sample-1",
        "uk-electricity-hdd-only-hourly-sample-2",
        "uk-gas-hdd-only-hourly-sample-0",
    ]


def test_load_sample_hourly():
    meter_data, temperature_data, metadata = load_sample(
        "il-electricity-cdd-hdd-hourly"
    )

    assert meter_data.shape == (19417, 1)
    assert meter_data.index.freq == "h"
    assert temperature_data.shape == (19417,)
    assert temperature_data.index.freq == "h"
    assert metadata == {
        "annual_baseline_base_load": 2000.0,
        "annual_baseline_cooling_load": 4000.0,
        "annual_baseline_heating_load": 4000.0,
        "annual_baseline_total_load": 10000,
        "annual_reporting_base_load": 1800.0,
        "annual_reporting_cooling_load": 3600.0,
        "annual_reporting_heating_load": 3600.0,
        "annual_reporting_total_load": 9000.0,
        "baseline_cooling_balance_point": 65,
        "baseline_heating_balance_point": 60,
        "blackout_end_date": datetime.datetime(2017, 1, 4, 0, 0, tzinfo=pytz.UTC),
        "blackout_start_date": datetime.datetime(2016, 12, 26, 0, 0, tzinfo=pytz.UTC),
        "freq": "hourly",
        "id": "il-electricity-cdd-hdd-hourly",
        "interpretation": "electricity",
        "meter_data_filename": "il-electricity-cdd-hdd-hourly.csv.gz",
        "reporting_cooling_balance_point": 65,
        "reporting_heating_balance_point": 60,
        "temperature_filename": "il-tempF.csv.gz",
        "unit": "kWh",
        "usaf_id": "724390",
    }


def test_load_sample_daily():
    meter_data, temperature_data, metadata = load_sample("il-electricity-cdd-hdd-daily")

    assert meter_data.shape == (810, 1)
    assert meter_data.index.freq == "D"
    assert temperature_data.shape == (19417,)
    assert temperature_data.index.freq == "h"
    assert metadata is not None


def test_load_sample_billing_monthly():
    meter_data, temperature_data, metadata = load_sample(
        "il-electricity-cdd-hdd-billing_monthly"
    )

    assert meter_data.shape == (27, 1)
    assert meter_data.index.freq is None
    assert temperature_data.shape == (19417,)
    assert temperature_data.index.freq == "h"
    assert metadata is not None


def test_load_sample_unknown():
    with pytest.raises(ValueError):
        load_sample("unknown")
