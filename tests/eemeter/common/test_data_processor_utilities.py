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

from opendsm.eemeter.common.data_processor_utilities import compute_minimum_granularity



# day_counts / remove_duplicates / as_freq are exercised in test_transform.py;
# compute_minimum_granularity has no direct coverage.

@pytest.mark.parametrize(
    "freq,periods,expected",
    [
        ("h", 48, "hourly"),
        ("D", 60, "daily"),
        ("MS", 12, "billing_monthly"),
        ("2MS", 6, "billing_bimonthly"),
    ],
)
def test_compute_minimum_granularity_from_regular_index(freq, periods, expected):
    """A regularly-spaced index resolves to the matching granularity label."""
    index = pd.date_range("2020-01-01", periods=periods, freq=freq)

    assert compute_minimum_granularity(index.copy(), default_granularity="daily") == expected


def test_compute_minimum_granularity_single_point_returns_default():
    """A length-1 index has no spacing, so the default granularity is returned."""
    index = pd.date_range("2020-01-01", periods=1, freq="D")

    assert compute_minimum_granularity(index, default_granularity="billing_monthly") == "billing_monthly"


def test_compute_minimum_granularity_irregular_index_uses_median_spacing():
    """An index with no inferrable frequency falls back to the median day spacing."""
    index = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-05"])

    assert compute_minimum_granularity(index, default_granularity="daily") == "daily"


def test_compute_minimum_granularity_irregular_monthly_spacing():
    """Irregular reads averaging ~monthly spacing resolve to billing_monthly."""
    index = pd.to_datetime(
        ["2020-01-01", "2020-02-03", "2020-03-01", "2020-04-05", "2020-04-28"]
    )

    assert compute_minimum_granularity(index, default_granularity="daily") == "billing_monthly"
