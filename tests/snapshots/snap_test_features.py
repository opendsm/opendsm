#!/usr/bin/env python
# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc

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
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots["test_compute_temperature_features_hourly_hourly_degree_days values"] = [
    5.25,
    5.72,
    4.73,
    4.33,
    1.0,
    0.0,
]

snapshots[
    "test_compute_temperature_features_hourly_hourly_degree_days_use_mean_false values"
] = [0.22, 0.24, 0.2, 0.18, 1.0, 0.0]

snapshots["test_compute_temperature_features_daily_daily_degree_days values"] = [
    11.05,
    11.61,
    3.61,
    3.25,
    1.0,
    0.0,
]

snapshots[
    "test_compute_temperature_features_daily_daily_degree_days_use_mean_false values"
] = [11.05, 11.61, 3.61, 3.25, 1.0, 0.0]

snapshots[
    "test_compute_temperature_features_billing_monthly_daily_degree_days values"
] = [10.83, 11.39, 3.68, 3.31, 30.38, 0.0]

snapshots[
    "test_compute_temperature_features_billing_monthly_daily_degree_days_use_mean_false values"
] = [324.38, 341.38, 112.59, 101.33, 30.38, 0.0]

snapshots[
    "test_compute_temperature_features_billing_bimonthly_daily_degree_days values"
] = [10.94, 11.51, 3.65, 3.28, 61.62, 0.0]

snapshots[
    "test_compute_temperature_features_daily_hourly_degree_days_use_mean_false values"
] = [11.43, 12.01, 4.05, 3.7, 23.99, 0.0]

snapshots[
    "test_compute_temperature_features_billing_monthly_hourly_degree_days values"
] = [11.22, 11.79, 4.11, 3.76, 729.23, 0.0]

snapshots[
    "test_compute_temperature_features_billing_monthly_hourly_degree_days_use_mean_false values"
] = [336.54, 353.64, 125.96, 115.09, 729.23, 0.0]

snapshots[
    "test_compute_temperature_features_billing_bimonthly_hourly_degree_days values"
] = [11.33, 11.9, 4.07, 3.72, 1478.77, 0.0]

snapshots["test_compute_temperature_features_daily_hourly_degree_days values"] = [
    11.44,
    12.02,
    4.05,
    3.7,
    23.99,
    0.0,
]
