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
import pandas as pd
import pytest

from opendsm.eemeter.models.billing import BillingWeightedModel



@pytest.fixture(scope="module")
def weighted_model(baseline_data_billing):
    model = BillingWeightedModel().fit(
        baseline_data_billing, is_electricity_data=True, enforce_compliance=False
    )

    return model


def _predict(model, df, **kwargs):
    return model.predict(df, enforce_compliance=False, **kwargs)


def test_predict_returns_finite_predictions_with_the_load_columns(
    weighted_model, baseline_data_billing
):
    result = _predict(weighted_model, baseline_data_billing)

    assert 0 < len(result) <= len(baseline_data_billing)
    assert {"observed", "predicted", "predicted_unc", "heating_load", "cooling_load"} <= set(
        result.columns
    )
    assert np.isfinite(result["predicted"]).all()


def test_predict_treats_the_string_none_as_no_aggregation(weighted_model, baseline_data_billing):
    native = _predict(weighted_model, baseline_data_billing)

    result = _predict(weighted_model, baseline_data_billing, aggregation="none")

    pd.testing.assert_frame_equal(result, native)


@pytest.mark.parametrize("aggregation, freq", [("monthly", "MS"), ("bimonthly", "2MS")])
def test_predict_aggregates_energy_by_sum_and_uncertainty_in_quadrature(
    weighted_model, baseline_data_billing, aggregation, freq
):
    native = _predict(weighted_model, baseline_data_billing)

    aggregated = _predict(weighted_model, baseline_data_billing, aggregation=aggregation)

    assert aggregated.index.freqstr == freq
    assert aggregated["predicted"].sum() == pytest.approx(native["predicted"].sum())
    assert aggregated["observed"].sum() == pytest.approx(native["observed"].sum())
    assert aggregated["predicted_unc"].pow(2).sum() == pytest.approx(
        native["predicted_unc"].pow(2).sum()
    )


def test_predict_rejects_an_unknown_aggregation(weighted_model, baseline_data_billing):
    with pytest.raises(ValueError, match="aggregation must be one of"):
        _predict(weighted_model, baseline_data_billing, aggregation="weekly")



def test_predict_before_fit_raises(baseline_data_billing):
    with pytest.raises(RuntimeError, match="fit"):
        BillingWeightedModel().predict(baseline_data_billing)


