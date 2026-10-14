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

import warnings

import numpy as np
import pandas as pd
import pytest

from opendsm.eemeter.common.data_settings import BillingDataSettings
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
    InfeasibleDataError,
    ModelLoadError,
    NonstandardSettingsError,
)
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
        baseline_df, is_electricity_data=True, enforce_compliance=False
    )

    return model


@pytest.fixture(scope="session")
def fitted_weighted_model(baseline_df):
    model = BillingWeightedModel().fit(
        baseline_df, is_electricity_data=True, enforce_compliance=False
    )

    return model


def _has_nonstandard_settings_warning(warnings):
    return any(w.qualified_name == "eemeter.settings.nonstandard" for w in warnings)


def test_default_settings_fit_has_no_nonstandard_settings_warning(fitted_model):
    """BillingModel's own defaults never trigger a nonstandard settings warning."""
    assert not _has_nonstandard_settings_warning(fitted_model.compliance.warnings)


def test_default_settings_fit_has_no_nonstandard_settings_warning_weighted(
    fitted_weighted_model,
):
    """BillingWeightedModel's own defaults never trigger a nonstandard settings warning."""
    assert not _has_nonstandard_settings_warning(fitted_weighted_model.compliance.warnings)


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
        baseline_df, is_electricity_data=True, enforce_compliance=False
    )

    matching = [
        w
        for w in baseline_model.compliance.disqualification
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


def test_weighted_model_predicts_per_read_period_not_per_day(
    fitted_model, fitted_weighted_model, baseline_df
):
    """The two billing models predict at different cadences on the same input.

    BillingModel spreads its prediction to one row per day, while
    BillingWeightedModel fits and predicts on the read periods themselves, so it
    returns one row per billing period. Both index into the same span.
    """
    unweighted_prediction = fitted_model.predict(baseline_df)
    weighted_prediction = fitted_weighted_model.predict(baseline_df)

    assert len(weighted_prediction) < len(unweighted_prediction)
    assert weighted_prediction.index.isin(unweighted_prediction.index).all(), (
        "weighted read-period timestamps fall outside the daily prediction index"
    )


def test_weighted_model_from_dict_round_trip_predicts_identically(
    fitted_weighted_model, baseline_df
):
    """A BillingWeightedModel round-tripped through to_dict/from_dict predicts the
    same values as the original model."""
    original_prediction = fitted_weighted_model.predict(baseline_df)

    rebuilt = BillingWeightedModel.from_dict(fitted_weighted_model.to_dict())
    rebuilt_prediction = rebuilt.predict(baseline_df)

    pd.testing.assert_frame_equal(rebuilt_prediction, original_prediction, rtol=1e-12)


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


@pytest.mark.parametrize("model_fixture", ["fitted_model", "fitted_weighted_model"])
def test_predict_enforce_compliance_never_binds_to_aggregation(
    model_fixture, baseline_df, request
):
    """Both billing models forward enforce_compliance past their aggregation parameter.

    Billing is the only family whose predict takes `aggregation` between the frame and
    `enforce_compliance`. The flag must travel by keyword; bound positionally it lands
    in `aggregation` and predict raises.
    """
    model = request.getfixturevalue(model_fixture)

    native = model.predict(baseline_df)
    relaxed = model.predict(baseline_df, enforce_compliance=False)

    assert len(relaxed) == len(native), (
        f"enforce_compliance changed the row count: {len(relaxed)} vs {len(native)}"
    )
    assert relaxed["predicted"].sum() == pytest.approx(
        native["predicted"].sum(), rel=1e-12
    )


def test_fit_rejects_positional_is_electricity_data(baseline_df):
    """is_electricity_data must be passed as a keyword; a positional call is rejected."""
    with pytest.raises(TypeError):
        BillingModel().fit(baseline_df, True)


def test_fit_with_keyword_is_electricity_data_succeeds(baseline_df):
    """Passing is_electricity_data as a keyword fits the model."""
    model = BillingModel().fit(
        baseline_df, is_electricity_data=True, enforce_compliance=False
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
        for w in padded_model.compliance.warnings
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
        for w in model.compliance.warnings
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


# ---------------------------------------------------------------------------
# compliance
# ---------------------------------------------------------------------------

@pytest.fixture
def disqualified_billing_df(baseline_df):
    """The baseline period truncated well below min_baseline_length, but with enough
    days remaining to still fit a model."""
    return baseline_df.iloc[:-90].copy()


@pytest.fixture(scope="module")
def poor_fit_billing_model(baseline_df):
    """A fitted model whose fit metrics disqualify it on the model member."""
    df = baseline_df.copy()
    df.loc[:5, "observed"] += df["observed"].median() * 50
    model = BillingModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    return model


@pytest.fixture
def compliant_billing_df():
    """A full year of daily temperature with monthly billing reads: within the
    baseline length bounds, positive observed values that vary with a varying
    temperature, and no gaps in the required daily temperature coverage."""
    daily_idx = pd.date_range(
        "2021-01-01", "2022-01-01", freq="D", tz="America/Los_Angeles"
    )
    day_of_year = (daily_idx - daily_idx[0]).days.to_numpy(dtype=float)
    daily_temperature = 55 + 25 * np.sin(2 * np.pi * day_of_year / 365)

    period_idx = pd.date_range("2021-01-01", periods=13, freq="MS", tz="America/Los_Angeles")
    period_temperature = daily_temperature[np.searchsorted(daily_idx, period_idx)]
    rng = np.random.default_rng(0)
    observed = np.full(len(daily_idx), np.nan)
    period_positions = np.searchsorted(daily_idx, period_idx)
    observed[period_positions] = (
        500 + 0.8 * (period_temperature - 55) ** 2 + rng.normal(0, 5, len(period_idx))
    )
    observed[period_positions[-1]] = np.nan  # closing row

    return pd.DataFrame(
        {"observed": observed, "temperature": daily_temperature}, index=daily_idx
    )


@pytest.mark.slow
def test_fit_prioritizes_nonstandard_settings_error_over_data_sufficiency_error(
    disqualified_billing_df,
):
    """Settings compliance is checked before the baseline object is built, so nonstandard
    settings win the race against a data sufficiency failure on the same fit."""
    with pytest.raises(NonstandardSettingsError):
        BillingModel(settings={"segment_minimum_count": 4}).fit(
            disqualified_billing_df, is_electricity_data=True
        )


@pytest.mark.slow
def test_fit_raises_data_sufficiency_error_when_settings_excluded_from_enforced_list(
    disqualified_billing_df,
):
    """With settings excluded from the enforced list, the same fit falls through to the
    data sufficiency check instead."""
    with pytest.raises(DataSufficiencyError):
        BillingModel(settings={"segment_minimum_count": 4}).fit(
            disqualified_billing_df, is_electricity_data=True, enforce_compliance=["data"]
        )


@pytest.mark.parametrize(
    "enforce_compliance", [True, False, ["data"], ["model"], ["settings"]]
)
def test_fit_raises_infeasible_data_error_regardless_of_enforce_compliance(
    enforce_compliance,
):
    """No member of the enforced list bypasses an infeasibility raise."""
    idx = pd.date_range("2020-01-01", periods=1, freq="D", tz="America/Los_Angeles")
    df = pd.DataFrame({"observed": [10.0], "temperature": [50.0]}, index=idx)

    with pytest.raises(InfeasibleDataError):
        BillingModel().fit(df, is_electricity_data=True, enforce_compliance=enforce_compliance)


def test_predict_raises_disqualified_model_error_when_model_member_enforced(
    poor_fit_billing_model, baseline_df
):
    with pytest.raises(DisqualifiedModelError):
        poor_fit_billing_model.predict(baseline_df, enforce_compliance=["model"])


def test_predict_succeeds_when_disqualified_member_excluded_from_enforced_list(
    poor_fit_billing_model, baseline_df
):
    result = poor_fit_billing_model.predict(
        baseline_df, enforce_compliance=["data", "settings"]
    )

    assert "predicted" in result.columns


def test_predict_raises_data_sufficiency_error_for_disqualified_reporting_data(
    fitted_model, comstock_monthly
):
    """A reporting frame with no temperature is disqualified on the data member, and with
    data enforced, predict refuses on it."""
    _, df_r = comstock_monthly
    no_temperature = df_r.copy()
    no_temperature["temperature"] = np.nan

    with pytest.raises(DataSufficiencyError):
        fitted_model.predict(no_temperature.reset_index())


@pytest.mark.slow
@pytest.mark.parametrize(
    "enforce_compliance,should_raise",
    [
        (True, True),
        (False, False),
        (["data"], True),
        (["model"], False),
        (["settings"], False),
    ],
)
def test_fit_data_disqualification_respects_enforce_compliance_list(
    disqualified_billing_df, enforce_compliance, should_raise
):
    if should_raise:
        with pytest.raises(DataSufficiencyError):
            BillingModel().fit(
                disqualified_billing_df,
                is_electricity_data=True,
                enforce_compliance=enforce_compliance,
            )
    else:
        model = BillingModel().fit(
            disqualified_billing_df,
            is_electricity_data=True,
            enforce_compliance=enforce_compliance,
        )
        assert model.is_fitted


def test_fit_rejects_unknown_enforce_compliance_member_name(baseline_df):
    with pytest.raises(ValueError, match="bogus"):
        BillingModel().fit(baseline_df, is_electricity_data=True, enforce_compliance=["bogus"])


def test_clean_fit_is_qualified_with_no_disqualification(baseline_df):
    """A standard billing baseline carries no disqualification under the default
    enforcement."""
    model = BillingModel().fit(baseline_df, is_electricity_data=True)

    assert model.compliance.disqualification == []
    assert model.compliance.qualified is True


def test_enforce_compliance_true_and_false_yield_identical_parameters(compliant_billing_df):
    """Relaxing enforce_compliance changes only whether a refusal is raised, never the
    numbers a successful fit produces."""
    model_true = BillingModel().fit(
        compliant_billing_df, is_electricity_data=True, enforce_compliance=True
    )
    model_false = BillingModel().fit(
        compliant_billing_df, is_electricity_data=True, enforce_compliance=False
    )

    assert model_true.params.submodels.keys() == model_false.params.submodels.keys()
    for key, submodel in model_true.params.submodels.items():
        expected = submodel.coefficients.to_np_array()
        actual = model_false.params.submodels[key].coefficients.to_np_array()
        np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_nonstandard_settings_deviation_is_recorded_without_emitting(baseline_df):
    """The settings-deviation entry lands in compliance.status, but is never emitted on
    the warnings channel."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = BillingModel(settings={"segment_minimum_count": 4}).fit(
            baseline_df, is_electricity_data=True, enforce_compliance=False
        )

    assert not any(
        getattr(w.message, "qualified_name", None) == "eemeter.settings.nonstandard"
        for w in caught
    )
    assert "settings" in model.compliance.status


def test_nonstandard_settings_enforced_raises(baseline_df):
    with pytest.raises(NonstandardSettingsError):
        BillingModel(settings={"segment_minimum_count": 4}).fit(
            baseline_df, is_electricity_data=True
        )


# ---------------------------------------------------------------------------
# edge cases in predict
# ---------------------------------------------------------------------------

def test_predict_non_frame_raises_type_error(fitted_model):
    """A non-dataframe argument is rejected by type, not by an attribute error
    raised deeper in the prediction path."""
    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        fitted_model.predict("not a frame")


@pytest.mark.parametrize("aggregation", ["monthly", "bimonthly"])
def test_predict_aggregation_on_temperature_only_frame(
    fitted_model, baseline_df, aggregation
):
    """Aggregating a temperature-only prediction keeps the observed column.

    A reporting frame with no observed values yields a prediction without that
    column, so the aggregate supplies an all-NaN one and the output carries the
    same columns whichever way the caller supplies the frame.
    """
    temperature_only = baseline_df.drop(columns=["observed"])

    result = fitted_model.predict(
        temperature_only, aggregation=aggregation, enforce_compliance=False
    )

    with_observed = fitted_model.predict(
        baseline_df, aggregation=aggregation, enforce_compliance=False
    )

    assert list(result.columns) == list(with_observed.columns)
    assert result["observed"].isna().all()
    assert result["predicted"].notna().any()


def test_to_dict_writes_billing_model_choice(fitted_model):
    assert fitted_model.to_dict()["model_choice"] == "billing"


def test_from_dict_rejects_a_daily_payload(fitted_model):
    model_dict = fitted_model.to_dict()
    model_dict["model_choice"] = "daily"

    with pytest.raises(ModelLoadError, match="'daily' model; this class loads 'billing'"):
        BillingModel.from_dict(model_dict)
