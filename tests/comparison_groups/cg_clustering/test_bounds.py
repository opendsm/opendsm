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

from opendsm.comparison_groups.cg_clustering.bounds import (
    get_cluster_bounds,
    _get_num_cluster_min,
    _get_num_cluster_max,
)


@pytest.mark.parametrize("data_size", [30, 100, 1000, 10000])
def test_get_cluster_bounds_lower_below_upper(data_size):
    lower, upper = get_cluster_bounds(
        data_size=data_size, min_cluster_size=15, num_cluster_bound_lower=8, num_cluster_bound_upper=1500
    )

    assert lower < upper
    assert lower >= 2


def test_scaling_minimum_caps_at_lower_bound_near_calibration_point():
    """The exponential minimum-cluster curve is calibrated to ~8 clusters at
    1000 meters, where the configured lower bound caps it."""
    lower = _get_num_cluster_min(data_size=1000, min_cluster_size=15, num_cluster_bound_lower=8)

    assert lower == 8


def test_num_cluster_max_at_least_two():
    upper = _get_num_cluster_max(data_size=100, min_cluster_size=15, num_cluster_bound_upper=1500)

    assert upper >= 2
