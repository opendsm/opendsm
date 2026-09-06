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

from opendsm.eemeter import BillingBaselineData
from opendsm.eemeter.models.billing import BillingWeightedModel



@pytest.fixture(scope="module")
def weighted_data(comstock_monthly):
    """A data object of the weighted model's own: a fit keeps a reference to the data's
    disqualification list, so two models must not share one."""
    df_b, _ = comstock_monthly
    data = BillingBaselineData(df=df_b.reset_index().copy(), is_electricity_data=True)

    return data


@pytest.fixture(scope="module")
def weighted_model(weighted_data):
    model = BillingWeightedModel().fit(weighted_data, ignore_disqualification=True)

    return model


def _predict(model, data, **kwargs):
    return model.predict(data, ignore_disqualification=True, **kwargs)


def test_predict_returns_one_finite_prediction_per_input_period(weighted_model, weighted_data):
    result = _predict(weighted_model, weighted_data)

    assert len(result) == len(weighted_data.billing_df)
    assert {"observed", "predicted", "predicted_unc", "heating_load", "cooling_load"} <= set(
        result.columns
    )
    assert np.isfinite(result["predicted"]).all()


def test_predict_treats_the_string_none_as_no_aggregation(weighted_model, weighted_data):
    native = _predict(weighted_model, weighted_data)

    result = _predict(weighted_model, weighted_data, aggregation="none")

    pd.testing.assert_frame_equal(result, native)


@pytest.mark.parametrize("aggregation, freq", [("monthly", "MS"), ("bimonthly", "2MS")])
def test_predict_aggregates_energy_by_sum_and_uncertainty_in_quadrature(
    weighted_model, weighted_data, aggregation, freq
):
    native = _predict(weighted_model, weighted_data)

    aggregated = _predict(weighted_model, weighted_data, aggregation=aggregation)

    assert aggregated.index.freqstr == freq
    assert aggregated["predicted"].sum() == pytest.approx(native["predicted"].sum())
    assert aggregated["observed"].sum() == pytest.approx(native["observed"].sum())
    assert aggregated["predicted_unc"].pow(2).sum() == pytest.approx(
        native["predicted_unc"].pow(2).sum()
    )


def test_predict_rejects_an_unknown_aggregation(weighted_model, weighted_data):
    with pytest.raises(ValueError, match="aggregation must be one of"):
        _predict(weighted_model, weighted_data, aggregation="weekly")


def test_predict_rejects_daily_data(weighted_model, baseline_data_daily):
    with pytest.raises(TypeError, match="BillingBaselineData"):
        _predict(weighted_model, baseline_data_daily)


def test_predict_before_fit_raises(weighted_data):
    with pytest.raises(RuntimeError, match="fit"):
        BillingWeightedModel().predict(weighted_data)


