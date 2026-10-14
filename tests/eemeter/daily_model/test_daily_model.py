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
import json
import warnings

import pytest

import numpy as np
import pandas as pd

from opendsm.eemeter import DailyModel
from opendsm.eemeter.models.daily.data import DailyBaselineData, DailyReportingData
from opendsm.eemeter.common.data_settings import DailyDataSettings
from opendsm.eemeter.models.daily.utilities.settings import DailySettings
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
    InfeasibleDataError,
    ModelLoadError,
    NonstandardSettingsError,
)



def _daily_frame(meter, temperature):
    """Combine a meter frame (column 'value') and a temperature series into a model input frame."""
    meter = meter.rename(columns={"value": "observed"})
    temperature = temperature.tz_convert(meter.index.tz).rename("temperature")

    return pd.concat([meter, temperature], axis=1, sort=True)


@pytest.fixture
def daily_series(comstock_daily):
    """(meter_df, temperature_series) extracted from ComStock daily baseline."""
    df_b, _ = comstock_daily
    meter = df_b[["observed"]].rename(columns={"observed": "value"}).copy()
    meter.index = meter.index.tz_convert("America/Los_Angeles")
    temp = df_b["temperature"].copy()
    temp.index = temp.index.tz_convert("UTC")

    return meter, temp


@pytest.fixture
def bad_daily_series(daily_series):
    meter, temp = daily_series
    meter.iloc[:50] += meter["value"].median() * 50

    return meter, temp


@pytest.fixture
def missing_daily_data(bad_daily_series) -> pd.DataFrame:
    meter, temp = bad_daily_series
    meter = meter[:-90]

    return _daily_frame(meter, temp)


@pytest.fixture
def bad_daily_data(bad_daily_series) -> pd.DataFrame:
    meter, temp = bad_daily_series

    return _daily_frame(meter, temp)


@pytest.mark.slow
def test_disqualified_data_error(missing_daily_data, bad_daily_data):
    with pytest.raises(DataSufficiencyError):
        DailyModel().fit(missing_daily_data, is_electricity_data=True)
    model = DailyModel().fit(
        missing_daily_data, is_electricity_data=True, enforce_compliance=False
    )
    with pytest.raises(DisqualifiedModelError):
        model.predict(bad_daily_data)
    model.predict(missing_daily_data, enforce_compliance=False)


def test_model_cvrmse_error(bad_daily_data):
    model = DailyModel().fit(bad_daily_data, is_electricity_data=True)
    with pytest.raises(DisqualifiedModelError):
        model.predict(bad_daily_data)
    model.predict(bad_daily_data, enforce_compliance=False)


def test_timezone_behavior(daily_series):
    # TODO probably move some of this to dataclass tests
    meter, temp = daily_series
    # ensure that meter is using local tz
    assert str(meter.index.tz) == "America/Los_Angeles"
    assert str(temp.index.tz) == "UTC"

    baseline_df = _daily_frame(meter, temp)
    model = DailyModel().fit(baseline_df, is_electricity_data=True)

    # a timezone other than the one fitted on is a data disqualification, so the
    # default refuses and a caller who drops "data" predicts anyway
    reporting_no_meter_utc = pd.DataFrame({"temperature": temp})
    assert str(model.baseline_timezone) != str(reporting_no_meter_utc.index.tz)
    with pytest.raises(DataSufficiencyError) as exc_info:
        model.predict(reporting_no_meter_utc)

    assert any(
        entry.qualified_name == "eemeter.data_quality.timezone_mismatch"
        for entry in exc_info.value.entries
    )
    model.predict(reporting_no_meter_utc, enforce_compliance=["model", "settings"])

    reporting_df = _daily_frame(meter, temp)
    res1 = model.predict(reporting_df)
    reporting_no_meter = pd.DataFrame({"temperature": temp.tz_convert(meter.index.tz)})
    res2 = model.predict(reporting_no_meter, enforce_compliance=["model", "settings"])
    assert round((res1["temperature"] - res2["temperature"]).sum(), 2) == 0
    assert round((res1["predicted"] - res2["predicted"]).sum(), 2) == 0


def test_predict_df_matches_input_index(daily_series):
    meter, temp = daily_series
    baseline_df = _daily_frame(meter, temp)
    model = DailyModel().fit(baseline_df, is_electricity_data=True)

    temp = temp.copy()
    temp[temp.index.day > 20] = np.nan
    reporting_df = _daily_frame(meter, temp)
    res = model.predict(reporting_df, enforce_compliance=["model", "settings"])
    assert len(res) == len(reporting_df)


def test_daily_predict_before_fit_raises(daily_series):
    """Predicting on an unfitted DailyModel raises RuntimeError, not AttributeError."""
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    with pytest.raises(RuntimeError, match="must be fit"):
        DailyModel().predict(df)


def test_fit_with_dataframe_and_is_electricity_data_fits(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    model = DailyModel().fit(df, is_electricity_data=True)

    assert model.is_fitted


def test_fit_rejects_data_class_instance(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)
    data = DailyBaselineData(df, is_electricity_data=True)

    with pytest.raises(TypeError, match="Expected a pandas DataFrame"):
        DailyModel().fit(data, is_electricity_data=True)


def test_fit_rejects_positional_is_electricity_data(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    with pytest.raises(TypeError):
        DailyModel().fit(df, True)


def test_predict_frame_matches_predict_on_built_reporting_data(daily_series):
    """predict(df) reproduces predicting on a separately constructed reporting data object."""
    meter, temp = daily_series
    baseline_df = _daily_frame(meter, temp)
    model = DailyModel().fit(baseline_df, is_electricity_data=True)

    reporting_df = _daily_frame(meter, temp)
    reporting_data = DailyReportingData(
        reporting_df, True, settings=DailyDataSettings()
    )
    expected = model._predict_data(reporting_data)

    result = model.predict(reporting_df)

    np.testing.assert_allclose(
        result["predicted"].values, expected["predicted"].values, atol=1e-9
    )


def test_predict_without_observed_column_works(daily_series):
    meter, temp = daily_series
    baseline_df = _daily_frame(meter, temp)
    model = DailyModel().fit(baseline_df, is_electricity_data=True)

    reporting_df = pd.DataFrame({"temperature": temp.tz_convert(meter.index.tz)})
    result = model.predict(reporting_df, enforce_compliance=["model", "settings"])

    assert "predicted" in result.columns
    assert "observed" not in result.columns or result["observed"].isna().all()


def test_predict_does_not_trim_edge_nan_rows(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)
    model = DailyModel().fit(df, is_electricity_data=True)

    padded_temp = temp.tz_convert(meter.index.tz)
    padded_temp.iloc[:3] = np.nan
    padded_temp.iloc[-3:] = np.nan
    reporting_df = pd.DataFrame({"temperature": padded_temp})

    result = model.predict(reporting_df, enforce_compliance=["model", "settings"])

    assert len(result) == len(reporting_df)


def test_edge_trimmed_frame_fits_same_coefficients_with_warning(daily_series):
    """A baseline frame padded with all-NaN edge rows fits identically and logs one trim warning."""
    meter, temp = daily_series
    df = _daily_frame(meter, temp)
    unpadded_model = DailyModel().fit(df, is_electricity_data=True)

    n_leading, n_trailing = 4, 6
    leading = pd.DataFrame(
        {"observed": np.nan, "temperature": np.nan},
        index=df.index[0] - pd.to_timedelta(np.arange(n_leading, 0, -1), unit="D"),
    )
    trailing = pd.DataFrame(
        {"observed": np.nan, "temperature": np.nan},
        index=df.index[-1] + pd.to_timedelta(np.arange(1, n_trailing + 1), unit="D"),
    )
    padded_df = pd.concat([leading, df, trailing])

    padded_model = DailyModel().fit(padded_df, is_electricity_data=True)

    trim_warnings = [
        w
        for w in padded_model.compliance.warnings
        if w.qualified_name == "eemeter.data_quality.edge_rows_trimmed"
    ]
    assert len(trim_warnings) == 1
    assert trim_warnings[0].data == {"leading": n_leading, "trailing": n_trailing}

    for key, submodel in unpadded_model.params.submodels.items():
        padded_submodel = padded_model.params.submodels[key]
        assert padded_submodel.coefficients.intercept == pytest.approx(
            submodel.coefficients.intercept
        )


def test_fit_disqualified_frame_raises_with_matching_disqualification_list():
    """A disqualified baseline frame raises DataSufficiencyError carrying the disqualifications."""
    idx = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")
    df = pd.DataFrame(
        {"observed": np.linspace(10, 20, 30), "temperature": np.linspace(30, 90, 30)},
        index=idx,
    )

    model = DailyModel()
    with pytest.raises(DataSufficiencyError) as exc_info:
        model.fit(df, is_electricity_data=True)

    assert len(exc_info.value.entries) > 0
    assert exc_info.value.entries == model.compliance.disqualification


def test_one_row_baseline_raises_infeasible_data_error():
    """A single day carries no adjacent day to pair with, so no model is feasible."""
    idx = pd.date_range("2020-01-01", periods=1, freq="D", tz="America/Los_Angeles")
    df = pd.DataFrame({"observed": [10.0], "temperature": [50.0]}, index=idx)

    with pytest.raises(InfeasibleDataError) as exc_info:
        DailyModel().fit(df, is_electricity_data=True)

    assert isinstance(exc_info.value, DataSufficiencyError)
    assert exc_info.value.entries
    assert all(entry.severity == "infeasible" for entry in exc_info.value.entries)


def test_zero_overlap_after_alignment_raises_infeasible_data_error():
    """A meter frame and a temperature frame with no shared days align to nothing."""
    rng = np.random.default_rng(0)
    meter_idx = pd.date_range("2020-01-01", periods=200, freq="D", tz="America/Los_Angeles")
    temp_idx = pd.date_range("2021-01-01", periods=200, freq="D", tz="America/Los_Angeles")
    meter = pd.DataFrame({"observed": rng.normal(30.0, 3.0, 200)}, index=meter_idx)
    temperature = pd.DataFrame(
        {"temperature": rng.uniform(20.0, 90.0, 200)}, index=temp_idx
    )
    df = pd.concat([meter, temperature], axis=1, sort=True)

    with pytest.raises(InfeasibleDataError) as exc_info:
        DailyModel().fit(df, is_electricity_data=True)

    assert isinstance(exc_info.value, DataSufficiencyError)
    assert exc_info.value.entries
    assert all(entry.severity == "infeasible" for entry in exc_info.value.entries)


def _degenerate_frame(observed, temperature):
    """A year of daily data built from an observed and a temperature array."""
    idx = pd.date_range("2020-01-01", periods=365, freq="D", tz="UTC")

    return pd.DataFrame({"observed": observed, "temperature": temperature}, index=idx)


def _named_entry(model, qualified_name):
    matches = [
        entry
        for entry in model.compliance.disqualification
        if entry.qualified_name == qualified_name
    ]

    assert len(matches) == 1, (
        f"expected exactly one {qualified_name} disqualification, found "
        f"{[entry.qualified_name for entry in model.compliance.disqualification]}"
    )

    return matches[0]


def test_constant_temperature_fits_intercept_model():
    """Zero temperature variation leaves the balance points unidentifiable, so every
    submodel collapses to the intercept-only fit rather than crashing the optimizer."""
    rng = np.random.default_rng(0)
    df = _degenerate_frame(
        observed=rng.normal(30.0, 3.0, 365),
        temperature=np.full(365, 65.0),
    )

    model = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    assert [submodel.model_key for submodel in model.model.values()] == ["tidd"] * len(
        model.model
    )


def test_constant_temperature_records_insufficient_unique_temperature_values():
    """A constant temperature column disqualifies the baseline on the data member."""
    rng = np.random.default_rng(0)
    df = _degenerate_frame(
        observed=rng.normal(30.0, 3.0, 365),
        temperature=np.full(365, 65.0),
    )

    model = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    entry = _named_entry(
        model, "eemeter.sufficiency_criteria.insufficient_unique_temperature_values"
    )

    assert entry.severity == "disqualified"
    assert entry.member == "data"
    assert entry.data["n_unique_values"] == 1


def test_constant_observed_fits_with_zero_residual():
    """A constant observed column is a zero-residual fit: the selection criteria stay
    finite and the intercept reproduces the observed level."""
    rng = np.random.default_rng(1)
    df = _degenerate_frame(
        observed=np.full(365, 12.0),
        temperature=rng.uniform(20.0, 90.0, 365),
    )

    model = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=False)
    predicted = model.predict(df, enforce_compliance=False)["predicted"].values

    np.testing.assert_allclose(predicted, 12.0, atol=1e-4)


def test_constant_observed_records_insufficient_unique_observed_values():
    """A constant observed column disqualifies the baseline on the data member."""
    rng = np.random.default_rng(1)
    df = _degenerate_frame(
        observed=np.full(365, 12.0),
        temperature=rng.uniform(20.0, 90.0, 365),
    )

    model = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    entry = _named_entry(
        model, "eemeter.sufficiency_criteria.insufficient_unique_observed_values"
    )

    assert entry.severity == "disqualified"
    assert entry.member == "data"
    assert entry.data["n_unique_values"] == 1


def test_winter_segment_constant_temperature_fit_does_not_raise():
    """A winter-only constant temperature (other seasons vary) leaves the global
    unique-value threshold satisfied, so the per-segment fit must still complete."""
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-01-01", periods=365, freq="D", tz="America/Los_Angeles")
    temperature = rng.uniform(20.0, 90.0, 365)
    winter_mask = idx.month.isin([1, 2, 11, 12])
    temperature[winter_mask] = 65.0
    observed = rng.normal(30.0, 3.0, 365)
    df = pd.DataFrame({"observed": observed, "temperature": temperature}, index=idx)

    model = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    assert model.is_fitted


def test_json_daily(comstock_daily):
    df_b, df_r = comstock_daily
    baseline_model = DailyModel().fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    metered_savings_dataframe = baseline_model.predict(df_r.reset_index())
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    json_str = baseline_model.to_json()
    loaded_model = DailyModel.from_json(json_str)
    prediction_json = loaded_model.predict(df_r.reset_index())
    total_metered_savings_loaded = (
        prediction_json["observed"] - prediction_json["predicted"]
    ).sum()

    assert total_metered_savings == total_metered_savings_loaded


def test_daily_model_rejects_legacy_model_kwarg():
    with pytest.raises(TypeError):
        DailyModel(model="legacy")


def test_daily_model_rejects_settings_of_another_type():
    with pytest.raises(TypeError, match="must be None, a dict, or a DailySettings"):
        DailyModel(settings="legacy")


@pytest.fixture(scope="module")
def default_fitted_daily_model(comstock_daily):
    df_b, _ = comstock_daily
    model = DailyModel().fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    return model


def _has_nonstandard_settings_warning(warnings):
    return any(w.qualified_name == "eemeter.settings.nonstandard" for w in warnings)


def test_default_settings_fit_has_no_nonstandard_settings_warning(default_fitted_daily_model):
    assert not _has_nonstandard_settings_warning(default_fitted_daily_model.compliance.disqualification)


def test_nonstandard_setting_fit_carries_deviation_warning(comstock_daily):
    df_b, _ = comstock_daily
    baseline_model = DailyModel(settings={"segment_minimum_count": 8}).fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    matching = [
        w
        for w in baseline_model.compliance.disqualification
        if w.qualified_name == "eemeter.settings.nonstandard"
    ]
    assert len(matching) == 1

    deviation = matching[0].data["deviations"]["segment_minimum_count"]
    assert deviation["value"] == 8
    assert deviation["default"] == 6


def test_legacy_preset_fit_has_no_nonstandard_settings_warning(comstock_daily):
    df_b, _ = comstock_daily
    baseline_model = DailyModel(settings={"preset": "legacy"}).fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    assert not _has_nonstandard_settings_warning(baseline_model.compliance.disqualification)


def test_nonstandard_settings_warning_survives_to_dict_round_trip(comstock_daily):
    df_b, _ = comstock_daily
    baseline_model = DailyModel(settings={"segment_minimum_count": 8}).fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    rebuilt = DailyModel.from_dict(baseline_model.to_dict())

    assert _has_nonstandard_settings_warning(rebuilt.compliance.disqualification)

    matching = [
        w
        for w in rebuilt.compliance.disqualification
        if w.qualified_name == "eemeter.settings.nonstandard"
    ]
    deviation = matching[0].data["deviations"]["segment_minimum_count"]
    assert deviation == {"value": 8, "default": 6}


def test_from_dict_accepts_developer_mode_settings_without_preset(default_fitted_daily_model):
    model_dict = default_fitted_daily_model.to_dict()
    model_dict["settings"]["developer_mode"] = True
    del model_dict["settings"]["preset"]

    DailyModel.from_dict(model_dict)


def test_legacy_deserialization_daily(comstock_daily, snapshot):
    legacy_model_dict = {
        "model_type": "hdd_only",
        "formula": "meter_value ~ hdd_46",
        "status": "QUALIFIED",
        "model_params": {"intercept": 12, "beta_hdd": 2, "heating_balance_point": 50},
        "r_squared_adj": 0.3,
        "warnings": [],
    }
    serialized_str = json.dumps(legacy_model_dict)
    baseline_model = DailyModel.from_2_0_json(serialized_str, is_electricity_data=True)

    _, df_r = comstock_daily
    df_r = df_r.copy()
    df_r.index = df_r.index.tz_convert("UTC")
    metered_savings_dataframe = baseline_model.predict(
        df_r.reset_index(), enforce_compliance=["model", "settings"]
    )
    total_metered_savings = (
        metered_savings_dataframe["observed"] - metered_savings_dataframe["predicted"]
    ).sum()

    assert round(float(total_metered_savings), 2) == snapshot(name="total_metered_savings")


def test_meter_fact_round_trips_through_to_dict_from_dict(default_fitted_daily_model):
    model_dict = default_fitted_daily_model.to_dict()
    rebuilt = DailyModel.from_dict(model_dict)

    assert rebuilt._is_electricity_data == default_fitted_daily_model._is_electricity_data


def test_from_dict_requires_keyword_when_payload_lacks_meter_fact(
    default_fitted_daily_model, comstock_daily
):
    model_dict = default_fitted_daily_model.to_dict()
    del model_dict["info"]["is_electricity_data"]

    with pytest.raises(ValueError):
        DailyModel.from_dict(model_dict)

    rebuilt = DailyModel.from_dict(model_dict, is_electricity_data=True)
    prior_format = [
        w
        for w in rebuilt.compliance.warnings
        if w.qualified_name == "eemeter.serialization.prior_format"
    ]
    assert len(prior_format) == 1

    _, df_r = comstock_daily
    result = rebuilt.predict(df_r.reset_index())
    assert "predicted" in result.columns


def test_from_dict_rejects_keyword_when_payload_carries_meter_fact(default_fitted_daily_model):
    model_dict = default_fitted_daily_model.to_dict()

    with pytest.raises(ValueError):
        DailyModel.from_dict(model_dict, is_electricity_data=True)


@pytest.fixture(scope="module")
def poor_fit_daily_model(comstock_daily):
    """A fitted model whose fit metrics disqualify it, so its record is written after the fit."""
    df_b, _ = comstock_daily
    df = df_b.reset_index().copy()
    df.loc[:50, "observed"] += df["observed"].median() * 50
    model = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    return model


def test_fit_metrics_disqualification_survives_a_round_trip(poor_fit_daily_model):
    """The record is written after the fit, so a payload read at call time carries it."""
    rebuilt = DailyModel.from_dict(poor_fit_daily_model.to_dict())

    entry = _named_entry(rebuilt, "eemeter.model_fit_metrics")

    assert entry.severity == "disqualified"
    assert entry.member == "model"
    assert entry.data["cvrmse"] == pytest.approx(
        _named_entry(poor_fit_daily_model, "eemeter.model_fit_metrics").data["cvrmse"]
    )


def test_round_tripped_disqualified_model_still_refuses_to_predict(
    poor_fit_daily_model, comstock_daily
):
    rebuilt = DailyModel.from_dict(poor_fit_daily_model.to_dict())

    _, df_r = comstock_daily
    with pytest.raises(DisqualifiedModelError):
        rebuilt.predict(df_r.reset_index())


def test_from_dict_rejects_a_newer_schema_version(default_fitted_daily_model):
    model_dict = default_fitted_daily_model.to_dict()
    model_dict["schema_version"] += 1

    with pytest.raises(ModelLoadError):
        DailyModel.from_dict(model_dict)


def test_from_dict_reports_the_validation_failure_that_rejected_a_payload(
    default_fitted_daily_model,
):
    model_dict = default_fitted_daily_model.to_dict()
    model_dict["submodels"] = "not a mapping of submodels"

    with pytest.raises(ModelLoadError) as exc_info:
        DailyModel.from_dict(model_dict)

    assert exc_info.value.__cause__ is not None


def test_prior_format_payload_loads_and_carries_its_advisory(default_fitted_daily_model):
    """A payload without a schema version is read by the prior-format rules."""
    model_dict = default_fitted_daily_model.to_dict()
    del model_dict["schema_version"]
    del model_dict["compliance"]
    del model_dict["info"]["is_electricity_data"]

    rebuilt = DailyModel.from_dict(model_dict, is_electricity_data=True)

    assert [w.qualified_name for w in rebuilt.compliance.warnings] == [
        "eemeter.serialization.prior_format"
    ]


def test_prior_format_records_rebuild_from_the_info_lists(poor_fit_daily_model):
    """Entries without a severity take the severity of the list they came from."""
    model_dict = poor_fit_daily_model.to_dict()
    stored = model_dict.pop("compliance")
    del model_dict["schema_version"]
    model_dict["info"]["disqualification"] = [
        {key: entry[key] for key in ("qualified_name", "description", "data")}
        for entry in stored
        if entry["severity"] == "disqualified"
    ]

    rebuilt = DailyModel.from_dict(model_dict)

    entry = _named_entry(rebuilt, "eemeter.model_fit_metrics")

    assert entry.severity == "disqualified"
    assert entry.member == "model"


def test_prior_format_model_fit_metrics_disqualification_reports_model_status(
    poor_fit_daily_model,
):
    """A prior-format model-tier disqualification derives `member` from its qualified name."""
    model_dict = poor_fit_daily_model.to_dict()
    stored = model_dict.pop("compliance")
    del model_dict["schema_version"]
    model_dict["info"]["disqualification"] = [
        {key: entry[key] for key in ("qualified_name", "description", "data")}
        for entry in stored
        if entry["qualified_name"] == "eemeter.model_fit_metrics"
    ]

    rebuilt = DailyModel.from_dict(model_dict)

    assert rebuilt.compliance.status == ["model"]


def test_prior_format_settings_payload_drops_removed_sufficiency_keys(
    default_fitted_daily_model,
):
    """A payload without a schema version that carries an undeclared threshold key loads
    through the prior-format path."""
    model_dict = default_fitted_daily_model.to_dict()
    del model_dict["schema_version"]
    del model_dict["compliance"]
    del model_dict["info"]["is_electricity_data"]
    sufficiency = model_dict["settings"]["data"]["sufficiency"]
    sufficiency["temperature"]["min_pct_hourly_coverage"] = 0.5
    sufficiency["observed"]["min_pct_period_coverage"] = 0.5
    sufficiency["joint"]["min_pct_period_coverage"] = 0.5

    rebuilt = DailyModel.from_dict(model_dict, is_electricity_data=True)

    assert rebuilt.settings.data.sufficiency.temperature.min_pct_daily_coverage == 0.9


def test_versioned_payload_drops_removed_sufficiency_keys(default_fitted_daily_model):
    """A payload stamped with the current schema version also sheds undeclared threshold keys."""
    model_dict = default_fitted_daily_model.to_dict()
    sufficiency = model_dict["settings"]["data"]["sufficiency"]
    sufficiency["temperature"]["min_pct_hourly_coverage"] = 0.5
    sufficiency["observed"]["min_pct_period_coverage"] = 0.5
    sufficiency["joint"]["min_pct_period_coverage"] = 0.5

    rebuilt = DailyModel.from_dict(model_dict)

    assert rebuilt.settings.data.sufficiency.temperature.min_pct_daily_coverage == 0.9


def test_daily_settings_data_block_has_sufficiency_defaults():
    settings = DailySettings()

    assert settings.data.sufficiency.min_baseline_length == np.ceil(0.9 * 365)


def test_data_mapping_on_settings_instance_builds_daily_data_settings():
    settings = DailySettings(data={"sufficiency": {"min_baseline_length": 30}})

    assert isinstance(settings.data, DailyDataSettings)
    assert settings.data.sufficiency.min_baseline_length == 30


def test_sufficiency_settings_survive_to_dict_round_trip(comstock_daily):
    df_b, _ = comstock_daily
    model = DailyModel(settings={"data": {"sufficiency": {"min_baseline_length": 100}}}).fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    rebuilt = DailyModel.from_dict(model.to_dict())

    assert rebuilt.settings.data.sufficiency.min_baseline_length == 100


def test_changed_min_baseline_length_fires_nonstandard_settings_warning(comstock_daily):
    df_b, _ = comstock_daily
    model = DailyModel(settings={"data": {"sufficiency": {"min_baseline_length": 100}}}).fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    assert _has_nonstandard_settings_warning(model.compliance.disqualification)


def test_changed_requested_start_does_not_fire_nonstandard_settings_warning(comstock_daily):
    df_b, _ = comstock_daily
    model = DailyModel(
        settings={"data": {"sufficiency": {"requested_start": df_b.index.min()}}}
    ).fit(df_b.reset_index(), is_electricity_data=True, enforce_compliance=False)

    assert not _has_nonstandard_settings_warning(model.compliance.disqualification)


def test_baseline_df_is_none_before_fit(daily_series):
    assert DailyModel().baseline_df is None


def test_baseline_df_is_dataframe_after_fit(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    model = DailyModel().fit(df, is_electricity_data=True)

    assert isinstance(model.baseline_df, pd.DataFrame)


def test_settings_deviations_property_matches_the_warning_payload(comstock_daily):
    """The property and the warning report the same deviation, from one formatter."""
    df_b, _ = comstock_daily
    model = DailyModel(settings={"segment_minimum_count": 8}).fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    warning = next(
        w
        for w in model.compliance.disqualification
        if w.qualified_name == "eemeter.settings.nonstandard"
    )

    assert model.settings_deviations == warning.data["deviations"]


def test_settings_deviations_property_is_empty_for_default_settings(default_fitted_daily_model):
    assert default_fitted_daily_model.settings_deviations == {}


def test_settings_deviations_property_recomputes_after_a_round_trip(comstock_daily):
    """The property is derived from the model's own settings, so deserialization
    restores it without storing it."""
    df_b, _ = comstock_daily
    model = DailyModel(settings={"segment_minimum_count": 8}).fit(
        df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
    )

    rebuilt = DailyModel.from_dict(model.to_dict())

    assert rebuilt.settings_deviations == model.settings_deviations
    assert rebuilt.settings_deviations["segment_minimum_count"]["value"] == 8


def test_settings_deviations_property_is_available_before_fit():
    model = DailyModel(settings={"segment_minimum_count": 8})

    assert model.settings_deviations["segment_minimum_count"]["default"] == 6


# ---------------------------------------------------------------------------
# compliance: fit refusal order
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fit_prioritizes_nonstandard_settings_error_over_data_sufficiency_error(
    missing_daily_data,
):
    """Settings compliance is checked before the baseline object is built, so nonstandard
    settings win the race against a data sufficiency failure on the same fit."""
    with pytest.raises(NonstandardSettingsError):
        DailyModel(settings={"segment_minimum_count": 8}).fit(
            missing_daily_data, is_electricity_data=True
        )


@pytest.mark.slow
def test_fit_raises_data_sufficiency_error_when_settings_excluded_from_enforced_list(
    missing_daily_data,
):
    """With settings excluded from the enforced list, the same fit falls through to the
    data sufficiency check instead."""
    with pytest.raises(DataSufficiencyError):
        DailyModel(settings={"segment_minimum_count": 8}).fit(
            missing_daily_data, is_electricity_data=True, enforce_compliance=["data"]
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
        DailyModel().fit(df, is_electricity_data=True, enforce_compliance=enforce_compliance)


# ---------------------------------------------------------------------------
# compliance: predict refusal order
# ---------------------------------------------------------------------------

def test_predict_raises_disqualified_model_error_when_model_member_enforced(
    poor_fit_daily_model, comstock_daily
):
    _, df_r = comstock_daily
    with pytest.raises(DisqualifiedModelError):
        poor_fit_daily_model.predict(df_r.reset_index(), enforce_compliance=["model"])


def test_predict_succeeds_when_disqualified_member_excluded_from_enforced_list(
    poor_fit_daily_model, comstock_daily
):
    _, df_r = comstock_daily
    result = poor_fit_daily_model.predict(
        df_r.reset_index(), enforce_compliance=["data", "settings"]
    )

    assert "predicted" in result.columns


def test_predict_raises_data_sufficiency_error_for_disqualified_reporting_data(
    daily_series,
):
    """A reporting frame with no temperature is disqualified on the data member, and with
    data enforced, predict refuses on it."""
    meter, temp = daily_series
    baseline_df = _daily_frame(meter, temp)
    model = DailyModel().fit(baseline_df, is_electricity_data=True)

    no_temperature = baseline_df[["temperature"]].assign(temperature=np.nan)
    with pytest.raises(DataSufficiencyError):
        model.predict(no_temperature)


def _utc_year_frame():
    """A year of daily heating-response data on a UTC index, clean enough to fit."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=365, freq="D", tz="UTC")
    temperature = 60.0 - 25.0 * np.cos(2 * np.pi * np.arange(365) / 365)
    observed = 10.0 + 1.5 * np.maximum(0.0, 60.0 - temperature) + rng.normal(0.0, 0.25, 365)

    return pd.DataFrame({"observed": observed, "temperature": temperature}, index=idx)


def test_utc_baseline_records_an_advisory_warning():
    """A UTC index is an advisory warning, not a disqualification."""
    baseline = DailyBaselineData(_utc_year_frame(), is_electricity_data=True)

    utc_entries = [
        entry
        for entry in baseline.compliance.warnings
        if entry.qualified_name == "eemeter.data_quality.utc_index"
    ]

    assert len(utc_entries) == 1, "UTC index did not record an advisory warning"
    assert utc_entries[0].severity == "advisory"
    assert "data" not in baseline.compliance.status


def test_utc_baseline_fits_under_default_enforcement():
    """The advisory warning does not stop a fit, and the model carries it."""
    model = DailyModel().fit(_utc_year_frame(), is_electricity_data=True)

    names = {entry.qualified_name for entry in model.compliance.warnings}
    assert model.is_fitted
    assert "eemeter.data_quality.utc_index" in names, f"model warnings: {sorted(names)}"


# ---------------------------------------------------------------------------
# compliance: enforce_compliance list normalization
# ---------------------------------------------------------------------------

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
    missing_daily_data, enforce_compliance, should_raise
):
    if should_raise:
        with pytest.raises(DataSufficiencyError):
            DailyModel().fit(
                missing_daily_data,
                is_electricity_data=True,
                enforce_compliance=enforce_compliance,
            )
    else:
        model = DailyModel().fit(
            missing_daily_data,
            is_electricity_data=True,
            enforce_compliance=enforce_compliance,
        )
        assert model.is_fitted


def test_fit_rejects_unknown_enforce_compliance_member_name(daily_series):
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    with pytest.raises(ValueError, match="bogus"):
        DailyModel().fit(df, is_electricity_data=True, enforce_compliance=["bogus"])


# ---------------------------------------------------------------------------
# compliance: null control and invariance
# ---------------------------------------------------------------------------

def test_clean_fit_is_qualified_with_no_disqualification(daily_series):
    """A well-behaved meter carries no disqualification, so it is qualified under the
    default enforcement."""
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    model = DailyModel().fit(df, is_electricity_data=True)

    assert model.compliance.disqualification == []
    assert model.compliance.qualified is True


def test_enforce_compliance_true_and_false_yield_identical_parameters(daily_series):
    """Relaxing enforce_compliance changes only whether a refusal is raised, never the
    numbers a successful fit produces."""
    meter, temp = daily_series
    df = _daily_frame(meter, temp)

    model_true = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=True)
    model_false = DailyModel().fit(df, is_electricity_data=True, enforce_compliance=False)

    assert model_true.params.submodels.keys() == model_false.params.submodels.keys()
    for key, submodel in model_true.params.submodels.items():
        expected = submodel.coefficients.to_np_array()
        actual = model_false.params.submodels[key].coefficients.to_np_array()
        np.testing.assert_allclose(actual, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# compliance: settings deviation
# ---------------------------------------------------------------------------

def test_nonstandard_settings_deviation_is_recorded_without_emitting(comstock_daily):
    """The settings-deviation entry lands in compliance.status, but is never emitted on
    the warnings channel."""
    df_b, _ = comstock_daily
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = DailyModel(settings={"segment_minimum_count": 8}).fit(
            df_b.reset_index(), is_electricity_data=True, enforce_compliance=False
        )

    assert not any(
        getattr(w.message, "qualified_name", None) == "eemeter.settings.nonstandard"
        for w in caught
    )
    assert "settings" in model.compliance.status


def test_nonstandard_settings_enforced_raises(comstock_daily):
    df_b, _ = comstock_daily
    with pytest.raises(NonstandardSettingsError):
        DailyModel(settings={"segment_minimum_count": 8}).fit(
            df_b.reset_index(), is_electricity_data=True
        )
