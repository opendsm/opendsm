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

import pytest

from opendsm.common.test_data import load_test_data

_TEST_METER = 110596


@pytest.fixture
def hourly_data():
    baseline, reporting = load_test_data("hourly_treatment_data")
    return baseline.loc[_TEST_METER], reporting.loc[_TEST_METER]


@pytest.fixture
def baseline(hourly_data):
    baseline, _ = hourly_data
    baseline.loc[baseline["observed"] > 513, "observed"] = (
        0  # quick extreme value removal
    )
    return baseline


@pytest.fixture
def reporting(hourly_data):
    _, reporting = hourly_data
    return reporting