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
import pytest

from opendsm.common.test_data import load_test_data

from .helpers import create_hourly_dataframe



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


@pytest.fixture
def synthetic_hourly_data():
    """Clean 365 days of hourly data, America/New_York timezone"""
    return create_hourly_dataframe()


@pytest.fixture
def synthetic_hourly_with_ghi():
    """Data including GHI column"""
    return create_hourly_dataframe(include_ghi=True)


@pytest.fixture
def baseline_with_supplemental_features(baseline):
    """Baseline data with custom supplemental features for testing"""
    baseline = baseline.copy()
    rng = np.random.default_rng(42)
    baseline["custom_ts_feature"] = rng.standard_normal(len(baseline))
    # numeric codes: a categorical column reaches the design matrix uncast, so
    # string labels raise on the float conversion
    baseline["custom_cat_feature"] = rng.integers(0, 3, len(baseline))

    return baseline


@pytest.fixture
def reporting_with_supplemental_features(reporting):
    """Reporting data with the same custom supplemental features as the baseline"""
    reporting = reporting.copy()
    rng = np.random.default_rng(42)
    reporting["custom_ts_feature"] = rng.standard_normal(len(reporting))
    reporting["custom_cat_feature"] = rng.integers(0, 3, len(reporting))

    return reporting


@pytest.fixture
def dst_transition_baseline():
    """Baseline data spanning DST transitions for testing DST handling"""
    return create_hourly_dataframe(start="2019-01-01", end="2019-12-31", tz="America/New_York")


@pytest.fixture
def dst_transition_reporting():
    """Reporting data spanning DST transitions for testing DST handling"""
    return create_hourly_dataframe(start="2020-01-01", end="2020-12-31", tz="America/New_York")
