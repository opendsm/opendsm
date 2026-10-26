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
from opendsm.eemeter.models.billing.data import BillingBaselineData
from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.billing.settings import BillingSettings
from opendsm.eemeter.models.billing.weighted_model import BillingWeightedModel
from opendsm.eemeter.models.daily.utilities.settings import DailySettings

from ..helpers import nonstandard_settings_warnings



@pytest.fixture(scope="session")
def baseline_df(comstock_monthly):
    df_b, _ = comstock_monthly

    return df_b.reset_index()


@pytest.fixture(scope="session")
def fitted_model(baseline_df):
    model = BillingModel().fit(
        baseline_df, is_electricity_data=True, enforce_compliance=False
    )

    return model


@pytest.fixture(scope="session")
def fitted_weighted_model(baseline_df):
    model = BillingWeightedModel().fit(
        baseline_df, is_electricity_data=True, enforce_compliance=False
    )

    return model


@pytest.mark.slow
def test_default_settings_fit_has_no_nonstandard_settings_warning(fitted_model):
    """BillingModel's own defaults never trigger a nonstandard settings warning."""
    assert nonstandard_settings_warnings(fitted_model.warnings) == []


def test_default_settings_fit_has_no_nonstandard_settings_warning_weighted(
    fitted_weighted_model,
):
    """BillingWeightedModel's own defaults never trigger a nonstandard settings warning."""
    assert nonstandard_settings_warnings(fitted_weighted_model.warnings) == []


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


@pytest.mark.slow
def test_nonstandard_setting_fit_carries_deviation_warning(baseline_df):
    """A caller-supplied deviation from BillingModel's defaults warns exactly once."""
    baseline_model = BillingModel(settings={"segment_minimum_count": 4}).fit(
        baseline_df, is_electricity_data=True, enforce_compliance=False
    )

    matching = nonstandard_settings_warnings(baseline_model.warnings)
    assert len(matching) == 1


def test_different_preset_is_judged_against_the_billing_reference():
    """BillingModel judges current-preset settings against the legacy preset:
    `segment_minimum_count` and `allow_smooth_model` deviate, and `preset` itself is not
    listed."""
    model = BillingModel(settings={"preset": "current"})

    deviations = model.settings_deviations

    assert "segment_minimum_count" in deviations
    assert "allow_smooth_model" in deviations
    assert "preset" not in deviations


@pytest.mark.parametrize("model_class", [BillingModel, BillingWeightedModel])
def test_default_settings_use_legacy_preset(model_class):
    """Both billing models default to the legacy preset."""
    assert model_class().settings.preset == "legacy"


@pytest.mark.parametrize(
    "model_class, segment_minimum_count",
    [(BillingModel, 10), (BillingWeightedModel, 3)],
)
def test_default_segment_minimum_count(model_class, segment_minimum_count):
    """The weighted model lowers the legacy preset's minimum segment count as its own
    default."""
    assert model_class().settings.segment_minimum_count == segment_minimum_count


@pytest.mark.parametrize("model_class", [BillingModel, BillingWeightedModel])
def test_default_settings_have_no_deviations(model_class):
    """Neither billing model reports deviations on its own defaults."""
    assert model_class().settings_deviations == {}


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


# ---------------------------------------------------------------------------
# failure paths
# ---------------------------------------------------------------------------

def test_predict_before_fit_raises(baseline_df):
    """Predicting before fitting raises RuntimeError."""
    with pytest.raises(RuntimeError, match="must be fit"):
        BillingModel().predict(baseline_df)


@pytest.mark.parametrize(
    "make_value,type_name",
    [
        (
            lambda df: BillingBaselineData(df, is_electricity_data=True),
            "BillingBaselineData",
        ),
        (lambda df: None, "NoneType"),
        (lambda df: {"observed": [1.0]}, "dict"),
        (lambda df: df["observed"], "Series"),
    ],
    ids=["data_object", "none", "dict", "series"],
)
@pytest.mark.parametrize("method", ["fit", "predict"])
@pytest.mark.parametrize("model_class", [BillingModel, BillingWeightedModel])
def test_non_dataframe_input_raises_type_error(
    model_class, method, make_value, type_name, baseline_df
):
    """fit and predict on an unfitted billing model reject anything that is not a DataFrame."""
    kwargs = {"fit": {"is_electricity_data": True}, "predict": {}}[method]

    with pytest.raises(TypeError, match=f"df must be a pandas DataFrame, got {type_name}"):
        getattr(model_class(), method)(make_value(baseline_df), **kwargs)


def test_predict_bad_aggregation_raises(fitted_model, baseline_df):
    """An unsupported aggregation level raises ValueError."""
    with pytest.raises(ValueError, match="aggregation must be one of"):
        fitted_model.predict(baseline_df, aggregation="weekly")


@pytest.mark.slow
@pytest.mark.parametrize("model_fixture", ["fitted_model", "fitted_weighted_model"])
def test_predict_enforce_compliance_never_binds_to_aggregation(
    model_fixture, baseline_df, request
):
    """On both billing models, predict(df, enforce_compliance=False) returns the same
    unaggregated prediction as predict(df)."""
    model = request.getfixturevalue(model_fixture)

    native = model.predict(baseline_df)
    relaxed = model.predict(baseline_df, enforce_compliance=False)

    assert len(relaxed) == len(native), (
        f"enforce_compliance changed the row count: {len(relaxed)} vs {len(native)}"
    )
    assert relaxed["predicted"].sum() == pytest.approx(
        native["predicted"].sum(), rel=1e-12
    )


def test_fit_records_is_electricity_data_and_baseline_metrics(fitted_model):
    """A fit records the is_electricity_data it was given and the fitted combination's
    baseline error metrics."""
    assert fitted_model.is_fitted
    assert fitted_model.is_electricity_data is True
    assert np.isfinite(fitted_model.baseline_metrics.cvrmse_adj)


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
        df, is_electricity_data=True, enforce_compliance=False
    )

    leading_row = df.iloc[[0]].copy()
    leading_row["datetime"] -= pd.Timedelta(days=1)
    leading_row[["observed", "temperature"]] = np.nan

    padded_df = pd.concat([leading_row, df], ignore_index=True)

    padded_model = BillingModel().fit(
        padded_df, is_electricity_data=True, enforce_compliance=False
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
        padded_df, is_electricity_data=True, enforce_compliance=False
    )

    trim_warnings = [
        w
        for w in model.warnings
        if w.qualified_name == "eemeter.data_quality.edge_rows_trimmed"
    ]
    assert trim_warnings == []


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
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
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


def test_one_settings_object_is_judged_against_each_billing_model_reference():
    """The reference belongs to the model, not to the settings: the same settings are
    standard for the weighted model and nonstandard for the plain one."""
    settings = BillingSettings(segment_minimum_count=3)

    assert BillingWeightedModel(settings=settings).settings_deviations == {}
    assert BillingModel(settings=settings).settings_deviations == {
        "segment_minimum_count": {"value": 3, "default": 10}
    }
