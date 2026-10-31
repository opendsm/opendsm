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

"""Hourly parameter counts include each fitted intercept, even with no active slopes."""

import numpy as np
import pandas as pd
import pytest

from opendsm.eemeter import HourlyModel


@pytest.mark.parametrize("with_design_matrix", [False, True])
@pytest.mark.parametrize(
    "fit_intercept, active_slopes", [(True, 0), (True, 2), (False, 0), (False, 2)]
)
def test_baseline_metrics_count_hourly_intercepts(
    fit_intercept, active_slopes, with_design_matrix
):
    model = HourlyModel(
        settings={
            "elasticnet": {"fit_intercept": fit_intercept, "alpha": 0},
            "adaptive_weights": {"enabled": False},
        }
    )
    # Supply learned slopes directly to isolate metrics from the fitting pipeline.
    model._model.coef_ = np.zeros((24, 2))
    model._model.coef_[:, :active_slopes] = 1
    index = pd.date_range("2024-01-01", periods=60 * 24, freq="h", tz="UTC")
    df = pd.DataFrame({"observed": 10.0, "predicted": 10.0}, index=index)
    X_fit = np.random.default_rng(42).normal(size=(60, 2))

    model._set_baseline_metrics(df, X_fit=X_fit if with_design_matrix else None)

    expected_per_hour = active_slopes + int(fit_intercept)
    assert model.baseline_metrics.num_model_params == 24 * expected_per_hour
    assert model.baseline_metrics.ddof == len(df) - 24 * expected_per_hour
    assert len(model.baseline_hour_metrics) == 24
    for metrics in model.baseline_hour_metrics.values():
        assert metrics.num_model_params == expected_per_hour
        assert metrics.ddof == 60 - expected_per_hour


@pytest.mark.parametrize("fit_intercept", [False, True])
def test_hour_edf_without_active_slopes_counts_only_fitted_intercept(fit_intercept):
    model = HourlyModel(settings={"elasticnet": {"fit_intercept": fit_intercept}})

    edf = model._compute_hour_edf(
        np.ones((60, 2)),
        hour=0,
        hour_coef=np.zeros(2),
        lambda_2=1,
        adaptive_weights=None,
    )

    assert edf == int(fit_intercept)
