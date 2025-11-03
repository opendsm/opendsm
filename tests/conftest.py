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

import json
import importlib.resources

import pytest

from opendsm.eemeter.samples import load_sample


@pytest.fixture
def sample_metadata():
    with importlib.resources.files("opendsm.eemeter.samples").joinpath(
        "metadata.json"
    ).open("rb") as f:
        metadata = json.loads(f.read().decode("utf-8"))
    return metadata


def _from_sample(sample, tempF=True):
    meter_data, temperature_data, metadata = load_sample(sample, tempF=tempF)
    return {
        "meter_data": meter_data,
        "temperature_data": temperature_data,
        "blackout_start_date": metadata["blackout_start_date"],
        "blackout_end_date": metadata["blackout_end_date"],
    }


@pytest.fixture
def il_electricity_cdd_hdd_hourly():
    return _from_sample("il-electricity-cdd-hdd-hourly")


@pytest.fixture
def il_electricity_cdd_hdd_daily():
    return _from_sample("il-electricity-cdd-hdd-daily")


@pytest.fixture
def il_electricity_cdd_hdd_billing_monthly():
    return _from_sample("il-electricity-cdd-hdd-billing_monthly")


@pytest.fixture
def il_electricity_cdd_hdd_billing_bimonthly():
    return _from_sample("il-electricity-cdd-hdd-billing_bimonthly")


@pytest.fixture
def il_gas_hdd_only_hourly():
    return _from_sample("il-gas-hdd-only-hourly")


@pytest.fixture
def uk_electricity_hdd_only_hourly_sample_1():
    return _from_sample("uk-electricity-hdd-only-hourly-sample-1", tempF=False)


@pytest.fixture
def uk_electricity_hdd_only_hourly_sample_2():
    return _from_sample("uk-electricity-hdd-only-hourly-sample-2", tempF=False)
