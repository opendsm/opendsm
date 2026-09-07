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
from opendsm.eemeter.common.warnings import nonstandard_settings_warning
from opendsm.eemeter.models.billing.data import (
    BillingBaselineData,
    BillingReportingData,
)
from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.billing.weighted_model import BillingWeightedModel
from opendsm.eemeter.models.daily.utilities.settings import DailySettings



@pytest.fixture(scope="session")
def baseline_data(comstock_monthly):
    df_b, _ = comstock_monthly

    return BillingBaselineData(df=df_b.reset_index(), is_electricity_data=True)


@pytest.fixture(scope="session")
def fitted_model(baseline_data):
    return BillingModel().fit(baseline_data, ignore_disqualification=True)


def _has_nonstandard_settings_warning(warnings):
    return any(w.qualified_name == "eemeter.settings.nonstandard" for w in warnings)


def test_default_settings_fit_has_no_nonstandard_settings_warning(fitted_model):
    """BillingModel's own defaults never trigger a nonstandard settings warning."""
    assert not _has_nonstandard_settings_warning(fitted_model.warnings)


def test_nonstandard_setting_fit_carries_deviation_warning(baseline_data):
    """A caller-supplied deviation from BillingModel's defaults warns exactly once."""
    baseline_model = BillingModel(settings={"segment_minimum_count": 4}).fit(
        baseline_data, ignore_disqualification=True
    )

    matching = [
        w
        for w in baseline_model.warnings
        if w.qualified_name == "eemeter.settings.nonstandard"
    ]
    assert len(matching) == 1


def test_different_preset_is_judged_against_the_billing_reference():
    """BillingModel is judged against the legacy preset, so another preset deviates on
    every field it moved, and 'preset' itself is not listed."""
    model = BillingModel(settings={"preset": "current"})

    warning = nonstandard_settings_warning(model.settings, model._reference_settings())

    deviations = warning.data["deviations"]
    assert "segment_minimum_count" in deviations
    assert "allow_smooth_model" in deviations
    assert "preset" not in deviations


@pytest.mark.parametrize(
    "model_class, segment_minimum_count",
    [(BillingModel, 10), (BillingWeightedModel, 3)],
)
def test_default_settings_use_legacy_preset_with_own_segment_minimum(
    model_class, segment_minimum_count
):
    """Both billing models default to the legacy preset; the weighted model lowers the
    minimum segment count as its own default, and neither warns on its defaults."""
    model = model_class()

    assert model.settings.preset == "legacy"
    assert model.settings.segment_minimum_count == segment_minimum_count
    assert nonstandard_settings_warning(model.settings, model._reference_settings()) is None


@pytest.mark.parametrize("model_class", [BillingModel, BillingWeightedModel])
def test_settings_instance_is_used_as_given(model_class):
    """A DailySettings instance passes through the billing constructors untouched."""
    settings = DailySettings(preset="current", segment_minimum_count=7)

    assert model_class(settings=settings).settings is settings


def test_billing_weighted_model_construction_prints_nothing(capsys):
    """Constructing BillingWeightedModel prints nothing."""
    BillingWeightedModel()

    assert capsys.readouterr().out == ""


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


def test_to_dict_tags_model_type_billing(fitted_model):
    """to_dict tags a billing payload with model_type='billing', overriding the
    inherited daily tag."""
    model_dict = fitted_model.to_dict()

    assert model_dict["model_type"] == "billing"


def test_from_dict_round_trips_tagged_billing_payload(fitted_model):
    """A billing-tagged payload round-trips through BillingModel.from_dict: the
    tag is tolerated and the rebuilt model re-serialises with the same tag."""
    model_dict = fitted_model.to_dict()

    rebuilt = BillingModel.from_dict(model_dict)

    assert isinstance(rebuilt, BillingModel)
    assert rebuilt.to_dict()["model_type"] == "billing"


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


def test_json_billing(comstock_monthly):
    df_b, df_r = comstock_monthly
    baseline_data = BillingBaselineData(df=df_b.reset_index(), is_electricity_data=True)
    baseline_model = BillingModel().fit(baseline_data, ignore_disqualification=True)

    reporting_data = BillingReportingData(df=df_r.reset_index(), is_electricity_data=True)
    metered_savings_dataframe = baseline_model.predict(reporting_data)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    json_str = baseline_model.to_json()
    loaded_model = BillingModel.from_json(json_str)
    prediction_json = loaded_model.predict(reporting_data)
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    assert total_metered_savings == total_metered_savings_loaded
