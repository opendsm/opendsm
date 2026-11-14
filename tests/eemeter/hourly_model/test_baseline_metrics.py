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

"""Hourly baseline parameter counts for a meter with a zero-usage hour."""

import numpy as np
import pytest

from opendsm.common.test_data import load_test_data
from opendsm.eemeter import HourlyBaselineData, HourlyModel

_TEST_METER = 110596
_ZERO_HOUR = 3


@pytest.fixture(scope="module")
def zero_hour_model():
    """Fit a gas meter whose usage is zero at 3 AM on every baseline day."""
    baseline, _ = load_test_data("hourly_treatment_data")
    baseline = baseline.loc[_TEST_METER].copy()
    baseline.loc[baseline["observed"] > 513, "observed"] = 0
    baseline.loc[baseline.index.hour == _ZERO_HOUR, "observed"] = 0

    baseline_data = HourlyBaselineData(baseline, is_electricity_data=False)
    return HourlyModel().fit(baseline_data)


def test_zero_usage_hour_fits_as_intercept_only(zero_hour_model):
    assert np.count_nonzero(zero_hour_model._model.coef_[_ZERO_HOUR]) == 0

    hour_metrics = zero_hour_model.baseline_hour_metrics[_ZERO_HOUR]
    assert hour_metrics is not None
    assert hour_metrics.num_model_params == 1


def test_global_param_count_is_sum_of_hourly_counts(zero_hour_model):
    hour_counts = [
        metrics.num_model_params
        for metrics in zero_hour_model.baseline_hour_metrics.values()
    ]

    assert len(hour_counts) == 24
    assert zero_hour_model.baseline_metrics.num_model_params == sum(hour_counts)
