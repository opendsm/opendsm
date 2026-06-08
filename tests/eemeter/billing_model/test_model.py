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

from opendsm.eemeter.common.exceptions import DisqualifiedModelError
from opendsm.eemeter.models.billing.data import (
    BillingBaselineData,
    BillingReportingData,
)
from opendsm.eemeter.models.billing.model import BillingModel



@pytest.fixture(scope="session")
def baseline_data(comstock_monthly):
    df_b, _ = comstock_monthly

    return BillingBaselineData(df=df_b.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def fitted_model(baseline_data):
    return BillingModel().fit(baseline_data, ignore_disqualification=True)


# ---------------------------------------------------------------------------
# failure paths (no fit required for the unfitted case)
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises(baseline_data):
    """Predicting before fitting raises RuntimeError."""
    with pytest.raises(RuntimeError, match="must be fit"):
        BillingModel().predict(baseline_data)


def test_predict_wrong_type_raises(fitted_model):
    """A non-Billing data object raises TypeError."""
    with pytest.raises(TypeError, match="BillingBaselineData or BillingReportingData"):
        fitted_model.predict("not a data object")


def test_predict_bad_aggregation_raises(fitted_model, baseline_data):
    """An unsupported aggregation level raises ValueError."""
    with pytest.raises(ValueError, match="aggregation must be one of"):
        fitted_model.predict(baseline_data, aggregation="weekly")


def test_to_dict_sets_developer_mode(fitted_model):
    """Serialising the model marks the settings developer_mode flag."""
    model_dict = fitted_model.to_dict()

    assert model_dict["settings"]["developer_mode"] is True


# ---------------------------------------------------------------------------
# aggregation arithmetic identity
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_monthly_aggregation_reducers(fitted_model, baseline_data):
    """Monthly aggregation sums energy and combines uncertainty in quadrature.

    Each aggregated column must use the correct reducer: predicted/observed are
    summed, predicted_unc is root-sum-square, temperature is averaged. Comparing
    the 'monthly' result to a hand-rolled resample of the unaggregated result
    pins those reducers (a swap of sum<->mean<->quadrature would fail).
    """
    native = fitted_model.predict(baseline_data, aggregation=None)
    monthly = fitted_model.predict(baseline_data, aggregation="monthly")

    expected_predicted = native["predicted"].resample("MS").sum()
    expected_observed = native["observed"].resample("MS").sum()
    expected_unc = native["predicted_unc"].resample("MS").apply(
        lambda x: np.sqrt(np.sum(np.square(x)))
    )
    expected_temp = native["temperature"].resample("MS").mean()

    assert np.allclose(monthly["predicted"], expected_predicted, equal_nan=True)
    assert np.allclose(monthly["observed"], expected_observed, equal_nan=True)
    assert np.allclose(monthly["predicted_unc"], expected_unc, equal_nan=True)
    assert np.allclose(monthly["temperature"], expected_temp, equal_nan=True)


@pytest.mark.slow
def test_uncertainty_quadrature_below_linear_sum(fitted_model, baseline_data):
    """Quadrature uncertainty is no larger than a naive linear sum (sub-additive)."""
    native = fitted_model.predict(baseline_data, aggregation=None)
    monthly = fitted_model.predict(baseline_data, aggregation="monthly")

    linear_sum = native["predicted_unc"].resample("MS").sum()
    finite = monthly["predicted_unc"].notna()

    assert (monthly["predicted_unc"][finite] <= linear_sum[finite] + 1e-9).all()
