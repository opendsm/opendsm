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

from opendsm.eemeter.common.data_settings import BillingDataSettings
from opendsm.eemeter.common.warnings import nonstandard_settings_warning
from opendsm.eemeter.models.billing.data import BillingBaselineData
from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.billing.settings import BillingSettings
from opendsm.eemeter.models.billing.weighted_model import BillingWeightedModel
from opendsm.eemeter.models.daily.utilities.settings import DailySettings



@pytest.fixture(scope="session")
def baseline_df(comstock_monthly):
    df_b, _ = comstock_monthly

    return df_b.reset_index()


@pytest.fixture(scope="session")
def fitted_model(baseline_df):
    model = BillingModel().fit(
        baseline_df, is_electricity_data=True, ignore_disqualification=True
    )

    return model


@pytest.fixture(scope="session")
def fitted_weighted_model(baseline_df):
    model = BillingWeightedModel().fit(
        baseline_df, is_electricity_data=True, ignore_disqualification=True
    )

    return model


def _has_nonstandard_settings_warning(warnings):
    return any(w.qualified_name == "eemeter.settings.nonstandard" for w in warnings)


def test_default_settings_fit_has_no_nonstandard_settings_warning(fitted_model):
    """BillingModel's own defaults never trigger a nonstandard settings warning."""
    assert not _has_nonstandard_settings_warning(fitted_model.warnings)


def test_default_settings_fit_has_no_nonstandard_settings_warning_weighted(
    fitted_weighted_model,
):
    """BillingWeightedModel's own defaults never trigger a nonstandard settings warning."""
    assert not _has_nonstandard_settings_warning(fitted_weighted_model.warnings)


@pytest.mark.parametrize("model_class", [BillingModel, BillingWeightedModel])
def test_default_settings_data_carries_billing_sufficiency_settings(model_class):
    """Both billing models default settings.data to BillingDataSettings."""
    model = model_class()

    assert isinstance(model.settings.data, BillingDataSettings)


@pytest.mark.parametrize("model_class", [BillingModel, BillingWeightedModel])
def test_settings_dump_keeps_billing_sufficiency_settings(model_class):
    """Serialized settings carry the billing data block, so a rebuilt model keeps it."""
    model = model_class(settings={"data": {"sufficiency": {"min_baseline_length": 200}}})

    rebuilt = model_class(settings=model.settings.model_dump())

    assert isinstance(rebuilt.settings.data, BillingDataSettings)
    assert rebuilt.settings.data.sufficiency.min_baseline_length == 200


def test_nonstandard_setting_fit_carries_deviation_warning(baseline_df):
    """A caller-supplied deviation from BillingModel's defaults warns exactly once."""
    baseline_model = BillingModel(settings={"segment_minimum_count": 4}).fit(
        baseline_df, is_electricity_data=True, ignore_disqualification=True
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
def test_billing_settings_instance_is_used_as_given(model_class):
    """A BillingSettings instance passes through the billing constructors untouched."""
    settings = BillingSettings(segment_minimum_count=7)

    assert model_class(settings=settings).settings is settings


@pytest.mark.parametrize("model_class", [BillingModel, BillingWeightedModel])
def test_daily_settings_instance_is_rejected(model_class):
    """A plain DailySettings instance carries the daily data block, so billing rejects it."""
    with pytest.raises(TypeError):
        model_class(settings=DailySettings(preset="legacy"))


def test_billing_weighted_model_construction_prints_nothing(capsys):
    """Constructing BillingWeightedModel prints nothing."""
    BillingWeightedModel()

    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# failure paths (no fit required for the unfitted case)
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises(baseline_df):
    """Predicting before fitting raises RuntimeError."""
    with pytest.raises(RuntimeError, match="must be fit"):
        BillingModel().predict(baseline_df)


def test_predict_wrong_type_raises(fitted_model, baseline_df):
    """Passing a data-class instance where predict expects a dataframe raises TypeError."""
    stale_data_object = BillingBaselineData(baseline_df, is_electricity_data=True)

    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        fitted_model.predict(stale_data_object)


def test_predict_bad_aggregation_raises(fitted_model, baseline_df):
    """An unsupported aggregation level raises ValueError."""
    with pytest.raises(ValueError, match="aggregation must be one of"):
        fitted_model.predict(baseline_df, aggregation="weekly")


def test_fit_rejects_positional_is_electricity_data(baseline_df):
    """is_electricity_data must be passed as a keyword; a positional call is rejected."""
    with pytest.raises(TypeError):
        BillingModel().fit(baseline_df, True)


def test_fit_with_keyword_is_electricity_data_succeeds(baseline_df):
    """Passing is_electricity_data as a keyword fits the model."""
    model = BillingModel().fit(
        baseline_df, is_electricity_data=True, ignore_disqualification=True
    )

    assert model.is_fitted


# ---------------------------------------------------------------------------
# baseline_df
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_class", [BillingModel, BillingWeightedModel])
def test_baseline_df_is_none_before_fit(model_class):
    """baseline_df is None before the model has been fit."""
    assert model_class().baseline_df is None


def test_baseline_df_after_fit_is_the_daily_frame(baseline_df, fitted_model):
    """BillingModel.baseline_df is the daily-resolution frame ('df') the model fit on."""
    expected = BillingBaselineData(baseline_df, is_electricity_data=True).df

    pd.testing.assert_frame_equal(fitted_model.baseline_df, expected)


def test_baseline_df_after_fit_is_the_billing_frame(baseline_df, fitted_weighted_model):
    """BillingWeightedModel.baseline_df is the period-level frame ('billing_df') the
    model fit on."""
    expected = BillingBaselineData(baseline_df, is_electricity_data=True).billing_df

    pd.testing.assert_frame_equal(fitted_weighted_model.baseline_df, expected)


# ---------------------------------------------------------------------------
# trimming
# ---------------------------------------------------------------------------

def test_padded_baseline_trims_to_the_same_fit(comstock_monthly):
    """A baseline frame padded with an unusable leading row fits to the same coefficients as
    the unpadded frame, and exactly one edge_rows_trimmed warning reports the count. The
    trailing edge is never trimmed for billing, since its last row closes the final period."""
    df_b, _ = comstock_monthly
    df = df_b.reset_index()

    unpadded_model = BillingModel().fit(
        df, is_electricity_data=True, ignore_disqualification=True
    )

    leading_row = df.iloc[[0]].copy()
    leading_row["datetime"] -= pd.Timedelta(days=1)
    leading_row[["observed", "temperature"]] = np.nan

    padded_df = pd.concat([leading_row, df], ignore_index=True)

    padded_model = BillingModel().fit(
        padded_df, is_electricity_data=True, ignore_disqualification=True
    )

    trim_warnings = [
        w
        for w in padded_model.warnings
        if w.qualified_name == "eemeter.data_quality.edge_rows_trimmed"
    ]
    assert len(trim_warnings) == 1
    assert trim_warnings[0].data == {"leading": 1, "trailing": 0}
    assert padded_model.to_dict()["submodels"] == unpadded_model.to_dict()["submodels"]


def test_trailing_period_closing_row_survives_trimming(comstock_monthly):
    """A trailing row carrying a temperature but no observed value closes the final
    billing period, so billing trimming keeps it and nothing is reported as trimmed."""
    df_b, _ = comstock_monthly
    df = df_b.reset_index()

    closing_row = df.iloc[[-1]].copy()
    closing_row["datetime"] += pd.Timedelta(days=30)
    closing_row["observed"] = np.nan

    padded_df = pd.concat([df, closing_row], ignore_index=True)

    model = BillingModel().fit(
        padded_df, is_electricity_data=True, ignore_disqualification=True
    )

    trim_warnings = [
        w
        for w in model.warnings
        if w.qualified_name == "eemeter.data_quality.edge_rows_trimmed"
    ]
    assert trim_warnings == []


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
def test_monthly_aggregation_reducers(fitted_model, baseline_df):
    """Monthly aggregation sums energy and combines uncertainty in quadrature.

    Each aggregated column must use the correct reducer: predicted/observed are
    summed, predicted_unc is root-sum-square, temperature is averaged. Comparing
    the 'monthly' result to a hand-rolled resample of the unaggregated result
    pins those reducers (a swap of sum<->mean<->quadrature would fail).
    """
    native = fitted_model.predict(baseline_df, aggregation=None)
    monthly = fitted_model.predict(baseline_df, aggregation="monthly")

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
def test_uncertainty_quadrature_below_linear_sum(fitted_model, baseline_df):
    """Quadrature uncertainty is no larger than a naive linear sum (sub-additive)."""
    native = fitted_model.predict(baseline_df, aggregation=None)
    monthly = fitted_model.predict(baseline_df, aggregation="monthly")

    linear_sum = native["predicted_unc"].resample("MS").sum()
    finite = monthly["predicted_unc"].notna()

    assert (monthly["predicted_unc"][finite] <= linear_sum[finite] + 1e-9).all()


def test_predict_aggregation_keyword_reproduces_native_totals(fitted_model, baseline_df):
    """predict(df, aggregation='monthly') sums to the same totals as the unaggregated
    predict(df, aggregation=None) on the same fixture."""
    native = fitted_model.predict(baseline_df, aggregation=None)
    monthly = fitted_model.predict(baseline_df, aggregation="monthly")

    assert monthly["predicted"].sum() == pytest.approx(native["predicted"].sum(), rel=1e-9)
    assert monthly["observed"].sum() == pytest.approx(native["observed"].sum(), rel=1e-9)


def test_json_billing(comstock_monthly):
    df_b, df_r = comstock_monthly
    baseline_model = BillingModel().fit(
        df_b.reset_index(), is_electricity_data=True, ignore_disqualification=True
    )

    reporting_df = df_r.reset_index()
    metered_savings_dataframe = baseline_model.predict(reporting_df)
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    json_str = baseline_model.to_json()
    loaded_model = BillingModel.from_json(json_str)
    prediction_json = loaded_model.predict(reporting_df)
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    assert total_metered_savings == total_metered_savings_loaded
