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

from opendsm.eemeter import (
    HourlyModel,
    HourlySolarSettings,
    HourlyNonSolarSettings,
)
from opendsm.eemeter.models.hourly.data import HourlyBaselineData
from opendsm.eemeter.models.hourly.model import (
    _fit_exp_growth_decay,
    _get_dst_indices,
    _hourly_profiles,
)
from opendsm.eemeter.models.hourly.settings import BaseHourlySettings
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)
from opendsm.eemeter.common.warnings import EEMeterWarning
import numpy as np
import pandas as pd
import pytest
from datetime import date
from math import ceil
from threadpoolctl import threadpool_info



def test_good_data(baseline, reporting):
    hm = HourlyModel().fit(baseline, is_electricity_data=True)
    p1 = hm.predict(reporting)
    assert np.isclose(
        p1["predicted"].sum(), 1135000, rtol=1e-2
    )  # quick check that model fit isn't changing drastically
    serialized = hm.to_json()
    hm2 = HourlyModel.from_json(serialized)
    p2 = hm2.predict(reporting)
    assert p1.equals(p2)


def test_fit_is_unchanged_by_last_bit_noise_on_a_temperature_bin_edge(
    baseline, reporting
):
    """A reading exactly on a temperature bin edge lands in the same bin whatever its
    last bit, so moving it by one unit in the last place leaves the fit's predictions
    in place."""
    settings = HourlyNonSolarSettings(seed=42)
    on_edge = baseline["temperature"].isin(settings.temperature_bin.fixed_bins)
    assert on_edge.any(), "the fixture has no reading on a bin edge"

    nudged = baseline.copy()
    nudged.loc[on_edge, "temperature"] = np.nextafter(
        nudged.loc[on_edge, "temperature"].to_numpy(), np.inf
    )
    fit = HourlyModel(settings=settings).fit(baseline, is_electricity_data=True)
    fit_nudged = HourlyModel(settings=HourlyNonSolarSettings(seed=42)).fit(
        nudged, is_electricity_data=True
    )

    np.testing.assert_allclose(
        fit_nudged.predict(reporting)["predicted"].to_numpy(),
        fit.predict(reporting)["predicted"].to_numpy(),
        rtol=1e-9,
    )


def test_misaligned_data(baseline, reporting):
    reporting.index = reporting.index.shift(8, freq="h")
    hm = HourlyModel().fit(baseline, is_electricity_data=True)

    prediction = hm.predict(reporting)

    # the shifted reporting period is padded out to whole local days
    expected_index = pd.date_range(
        reporting.index.min().replace(hour=0, minute=0, second=0, microsecond=0),
        reporting.index.max().replace(hour=23, minute=0, second=0, microsecond=0),
        freq="h",
    )

    assert prediction.index.equals(expected_index)


def test_tz_naive(baseline):
    baseline.index = baseline.index.tz_localize(None)
    with pytest.raises(ValueError):
        HourlyModel().fit(baseline, is_electricity_data=True)


def test_tz_mismatch(baseline):
    # might allow automatic adjustment from the model in the future, but hard requirement for now
    baseline.index = baseline.index.tz_convert("America/Los_Angeles")
    reporting = baseline.copy()
    reporting.index = reporting.index.tz_convert("America/New_York")
    hm = HourlyModel().fit(baseline, is_electricity_data=True)
    with pytest.raises(ValueError):
        hm.predict(reporting)


def test_predict_missing_fit_features(baseline, reporting):
    hm = HourlyModel(settings=HourlySolarSettings()).fit(baseline, is_electricity_data=True)
    reporting.drop("ghi", axis=1, inplace=True)
    with pytest.raises(ValueError):
        hm.predict(reporting)


def test_nonsolar_predict_with_ghi(baseline, reporting, caplog):
    baseline.drop("ghi", axis=1, inplace=True)
    hm = HourlyModel().fit(baseline, is_electricity_data=True)
    with caplog.at_level("WARNING"):
        hm.predict(reporting)
        assert "GHI" in caplog.text


def test_forced_solar_model_fit_no_ghi(baseline):
    baseline = baseline.drop("ghi", axis=1)
    with pytest.raises(ValueError):
        HourlyModel(settings=HourlySolarSettings()).fit(baseline, is_electricity_data=True)


def test_forced_nonsolar_model_fit_with_ghi(baseline):
    hm = HourlyModel(settings=HourlyNonSolarSettings()).fit(baseline, is_electricity_data=True)
    assert [
        w for w in hm.warnings if w.qualified_name == "eemeter.potential_model_mismatch"
    ]


def test_no_data(baseline):
    baseline["observed"] = 0
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline, is_electricity_data=True)


def test_negative_meter_values(baseline):
    baseline.loc["2018-01-08", "observed"] = -1

    # gas data can't be negative
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline, is_electricity_data=False)

    # elec can
    HourlyModel().fit(baseline, is_electricity_data=True)


def test_invalid_baseline_lengths(baseline):
    # TODO import min/max length from constants
    MAX_BASELINE_HOURS = 8760
    MIN_BASELINE_HOURS = ceil(MAX_BASELINE_HOURS * 0.9) - 24
    short_df = baseline.iloc[:MIN_BASELINE_HOURS]

    extra_days = baseline.iloc[-24*2:]
    extra_days.index += pd.Timedelta(days=2)
    long_df = pd.concat([baseline, extra_days])

    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(short_df, is_electricity_data=True)
    HourlyModel().fit(short_df, is_electricity_data=True, enforce_compliance=False)
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(long_df, is_electricity_data=True)
    HourlyModel().fit(long_df, is_electricity_data=True, enforce_compliance=False)


def test_low_freq_temp(baseline):
    baseline["temperature"] = baseline["temperature"].resample("D").mean()
    with pytest.raises(DataSufficiencyError) as exc_info:
        HourlyModel().fit(baseline, is_electricity_data=True)
    assert_dq(
        exc_info.value,
        ["eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data"],
    )


def test_low_freq_meter(baseline):
    baseline["observed"] = baseline["observed"].resample("D").mean()
    with pytest.raises(DataSufficiencyError) as exc_info:
        HourlyModel().fit(baseline, is_electricity_data=True)
    assert_dq(
        exc_info.value,
        ["eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data"],
    )


def test_monthly_percentage(baseline):
    missing_idx = pd.date_range(
        start=baseline.index.min(), end=baseline.index.max(), freq="h"
    )
    # create datetimeindex where a little over 10% of days are missing in feb, but still 90% overall
    missing_idx = missing_idx[missing_idx.day < 4]
    invalid_baseline = baseline[~baseline.index.isin(missing_idx)]
    # create datetimeindex where a little under 10% of days are missing in feb
    missing_idx = missing_idx[missing_idx.day < 3]
    valid_baseline = baseline[~baseline.index.isin(missing_idx)]

    invalid_temp = baseline.copy()
    invalid_temp.loc[invalid_temp.index.day < 5, "temperature"] = np.nan

    invalid_meter = baseline.copy()
    invalid_meter.loc[invalid_meter.index.day < 5, "observed"] = np.nan

    with pytest.raises(DataSufficiencyError) as exc_info:
        HourlyModel().fit(invalid_baseline, is_electricity_data=True)
    assert_dq(
        exc_info.value, ["eemeter.sufficiency_criteria.missing_monthly_temperature_data"]
    )
    HourlyModel().fit(valid_baseline, is_electricity_data=True)

    with pytest.raises(DataSufficiencyError) as exc_info:
        HourlyModel().fit(invalid_temp, is_electricity_data=True)
    assert_dq(
        exc_info.value,
        [
            "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
            "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
            "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
        ],
    )

    with pytest.raises(DataSufficiencyError) as exc_info:
        HourlyModel().fit(invalid_meter, is_electricity_data=True)
    assert_dq(
        exc_info.value,
        [
            "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
            "eemeter.sufficiency_criteria.missing_monthly_observed_data",
            "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
        ],
    )


def test_monthly_ghi_percentage(baseline):
    invalid_ghi = baseline.copy()
    invalid_ghi.loc[invalid_ghi.index.day < 5, "ghi"] = np.nan

    with pytest.raises(DataSufficiencyError) as exc_info:
        HourlyModel().fit(invalid_ghi, is_electricity_data=True)
    assert_dq(
        exc_info.value,
        [
            "eemeter.sufficiency_criteria.missing_monthly_ghi_data",
        ],
    )


def test_hourly_fit_daily_threshold(baseline):
    """confirm that days with >50% interpolated data are excluded from fit step"""

    # bit fragile testing private methods this way, but fine for now
    m = HourlyModel()
    b1 = baseline.copy()
    b1.loc["2018-01-08":"2018-01-08 11", "temperature"] = np.nan
    b1 = m._daily_fitting_sufficiency(b1)
    assert b1.loc["2018-01-08", "include_date"].sum() == 24

    b2 = baseline.copy()
    b2.loc["2018-01-08":"2018-01-08 12", "temperature"] = np.nan
    b2 = m._daily_fitting_sufficiency(b2)
    assert b2.loc["2018-01-08", "include_date"].sum() == 0
    assert b2.loc["2018-01-09", "include_date"].sum() == 24


@pytest.mark.filterwarnings("ignore:Objective did not converge.")
def test_hourly_error_metric_dq(baseline):
    baseline["observed"] = np.random.normal(-1, 10, len(baseline)) ** 3
    model = HourlyModel().fit(baseline, is_electricity_data=True)
    assert_dq(model, ["eemeter.model_fit_metrics"])
    with pytest.raises(DisqualifiedModelError):
        model.predict(baseline)


def assert_dq(data, expected_disqualifications):
    remaining_dq = set(expected_disqualifications)
    for dq in data.disqualification:
        if dq.qualified_name in remaining_dq:
            remaining_dq.remove(dq.qualified_name)
    assert not remaining_dq


def test_hourly_dict_settings():
    m = HourlyModel(settings={"train_features": ["feature_col"]})
    assert isinstance(m.settings, HourlyNonSolarSettings)
    assert set(m.settings.train_features) == {"temperature", "feature_col"}
    m = HourlyModel(settings={"train_features": ["feature_col", "ghi"]})
    assert isinstance(m.settings, HourlySolarSettings)
    assert set(m.settings.train_features) == {"temperature", "ghi", "feature_col"}
    m = HourlyModel(settings={"cvrmse_threshold": 1.0})
    assert isinstance(m.settings, BaseHourlySettings)
    assert m.settings.train_features == None


def _nonstandard_settings_warnings(hm):
    return [w for w in hm.warnings if w.qualified_name == "eemeter.settings.nonstandard"]


def test_default_settings_fit_carries_no_nonstandard_settings_warning(baseline):
    hm = HourlyModel().fit(baseline, is_electricity_data=True)

    assert _nonstandard_settings_warnings(hm) == []


def test_nonstandard_developer_setting_fit_carries_one_warning(baseline):
    hm = HourlyModel(settings={"cvrmse_threshold": 1.2}).fit(baseline, is_electricity_data=True)

    warnings = _nonstandard_settings_warnings(hm)
    assert len(warnings) == 1
    deviations = warnings[0].data["deviations"]
    assert deviations["cvrmse_threshold"]["value"] == 1.2
    assert deviations["cvrmse_threshold"]["default"] == 1.4


def test_nonstandard_developer_setting_warning_survives_dict_round_trip(baseline):
    hm = HourlyModel(settings={"cvrmse_threshold": 1.2}).fit(baseline, is_electricity_data=True)

    rebuilt = HourlyModel.from_dict(hm.to_dict())

    warnings = _nonstandard_settings_warnings(rebuilt)
    assert len(warnings) == 1
    deviations = warnings[0].data["deviations"]
    assert deviations["cvrmse_threshold"]["value"] == 1.2
    assert deviations["cvrmse_threshold"]["default"] == 1.4


def test_solar_default_settings_fit_carries_no_nonstandard_settings_warning(baseline):
    hm = HourlyModel(settings=HourlySolarSettings()).fit(baseline, is_electricity_data=True)

    assert _nonstandard_settings_warnings(hm) == []


class TestFitExpGrowthDecay:
    """The edge-bin rate estimator: k for data with exponential curvature, a
    deterministic no-evidence result (k = inf) otherwise, and never an exception."""

    def test_recovers_known_growth_rate(self):
        x = np.linspace(-1, 1, 2001)
        y = 2.0 + 0.5 * np.exp(4.0 * x)
        k = _fit_exp_growth_decay(x, y, is_x_sorted=True)
        assert np.isclose(k, 0.25, rtol=1e-3), f"expected k=0.25, got {k}"

    def test_recovers_known_decay_rate(self):
        x = np.linspace(-1, 1, 2001)
        y = 2.0 + 0.5 * np.exp(-4.0 * x)
        k = _fit_exp_growth_decay(x, y, is_x_sorted=True)
        assert np.isclose(k, -0.25, rtol=1e-3), f"expected k=-0.25, got {k}"

    def test_sorts_unsorted_input(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, 2001)
        y = 2.0 + 0.5 * np.exp(4.0 * x)
        k = _fit_exp_growth_decay(x, y, is_x_sorted=False)
        assert np.isclose(k, 0.25, rtol=1e-2), f"expected k=0.25, got {k}"

    def test_weak_but_real_curvature_is_kept(self):
        """A 0.1% exponential ripple on a constant load is measurable curvature,
        far above the no-evidence floor, and its rate must still be recovered."""
        x = np.linspace(-1, 1, 2001)
        y = 10.0 + 0.01 * np.exp(4.0 * x)
        k = _fit_exp_growth_decay(x, y, is_x_sorted=True)
        assert np.isfinite(k), "weak curvature was wrongly excluded as no-evidence"
        assert np.isclose(k, 0.25, rtol=1e-2), f"expected k=0.25, got {k}"

    def test_constant_load_returns_no_evidence(self):
        x = np.linspace(-1, 1, 200)
        y = np.full(200, 3.7)
        k = _fit_exp_growth_decay(x, y, is_x_sorted=True)
        assert np.isinf(k), f"constant load must return k=inf, got {k}"

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_too_few_points_returns_no_evidence(self, n):
        x = np.linspace(-1, 1, n)
        y = np.full(n, 3.7)
        k = _fit_exp_growth_decay(x, y, is_x_sorted=True)
        assert np.isinf(k), f"n={n} points must return k=inf, got {k}"

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_load_returns_no_evidence(self, bad):
        x = np.linspace(-1, 1, 200)
        y = 2.0 + 0.5 * np.exp(4.0 * x)
        y[100] = bad
        k = _fit_exp_growth_decay(x, y, is_x_sorted=True)
        assert np.isinf(k) and k > 0, f"non-finite load must return k=inf, got {k}"

    def test_non_finite_temperature_returns_no_evidence(self):
        x = np.linspace(-1, 1, 200)
        y = 2.0 + 0.5 * np.exp(4.0 * x)
        x[100] = np.nan
        k = _fit_exp_growth_decay(x, y, is_x_sorted=True)
        assert np.isinf(k) and k > 0, f"non-finite x must return k=inf, got {k}"

    def test_zero_fitted_rate_returns_no_evidence(self, monkeypatch):
        """A well-conditioned system whose solution has c = 0 is a flat exponential;
        it must report no evidence rather than divide by zero."""
        monkeypatch.setattr(np.linalg, "solve", lambda A, b: np.array([1.0, 0.0]))
        x = np.linspace(-1, 1, 200)
        y = 2.0 + 0.5 * np.exp(4.0 * x)
        k = _fit_exp_growth_decay(x, y, is_x_sorted=True)
        assert np.isinf(k) and k > 0, f"c=0 must return k=inf, got {k}"

    def test_float_noise_cannot_flip_the_no_evidence_decision(self):
        """Noise at the 1e-13 level on curvature-free load must not produce a
        finite rate on any draw: near-singular hours are excluded by rule, not
        by whichever side of a singular solve the noise lands on."""
        x = np.linspace(-1, 1, 200)
        y_flat = np.full(200, 3.7)
        ks = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            y = y_flat * (1 + 1e-13 * rng.standard_normal(200))
            ks.append(_fit_exp_growth_decay(x, y, is_x_sorted=True))

        assert all(np.isinf(k) for k in ks), (
            f"perturbed curvature-free load produced finite rates: {ks}"
        )

    def test_float_noise_leaves_well_conditioned_rate_unchanged(self):
        x = np.linspace(-1, 1, 200)
        y_exp = 2.0 + 0.5 * np.exp(2.0 * x)
        k_base = _fit_exp_growth_decay(x, y_exp, is_x_sorted=True)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            y = y_exp * (1 + 1e-13 * rng.standard_normal(200))
            k = _fit_exp_growth_decay(x, y, is_x_sorted=True)
            assert np.isclose(k, k_base, rtol=1e-9, atol=0), (
                f"1e-13 input noise (seed {seed}) moved k from {k_base} to {k}"
            )


class TestBlasThreadLimit:
    """Fitting and predicting run with the native thread pools pinned to one.

    Setting the ``*_NUM_THREADS`` environment variables cannot do this: the
    native pools read them when they load, on the first numpy import, long
    before any opendsm module runs.
    """

    def test_pools_are_single_threaded_during_fit(self, baseline, monkeypatch):
        """Every native pool reports one thread while the model is fitting."""
        seen = []
        original = HourlyModel._prepare_features

        def spy(self, meter_data):
            seen.append([info["num_threads"] for info in threadpool_info()])

            return original(self, meter_data)

        monkeypatch.setattr(HourlyModel, "_prepare_features", spy)
        HourlyModel().fit(baseline, is_electricity_data=True)

        assert seen, "_prepare_features was never called; the spy missed the fit path"
        for counts in seen:
            assert set(counts) == {1}, (
                f"expected every native pool pinned to 1 thread during fit, got {counts}"
            )

    def test_limit_does_not_leak_past_fit(self, baseline):
        """The cap is scoped, so the caller's threading is restored afterwards."""
        before = [info["num_threads"] for info in threadpool_info()]
        HourlyModel().fit(baseline, is_electricity_data=True)
        after = [info["num_threads"] for info in threadpool_info()]
        assert before == after, f"thread limits leaked out of fit: {before} -> {after}"


_UNMEASURED_FLAGS = ["interpolated_temperature", "interpolated_observed"]


@pytest.fixture
def flagged_hourly():
    """A year of complete hourly data whose measured and imputed hours are given explicitly.

    Supplying the ``interpolated_*`` flags lets a test mark hours unmeasured while choosing the
    values they carry; without the flags the data class imputes missing hours from their
    neighbors.
    """
    rng = np.random.default_rng(0)
    index = pd.date_range(
        "2019-01-01", "2020-01-01", freq="h", tz="America/New_York", inclusive="left"
    )
    day_of_year = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy()
    temperature = (
        55
        + 25 * np.sin(2 * np.pi * (day_of_year - 110) / 365)
        + 8 * np.sin(2 * np.pi * (hour - 9) / 24)
        + rng.normal(0, 3, len(index))
    )
    occupied = (index.dayofweek < 5) & (hour >= 8) & (hour < 18)
    observed = (
        3
        + 2 * occupied
        + 0.15 * np.maximum(temperature - 72, 0)
        + 0.05 * np.maximum(45 - temperature, 0)
        + rng.normal(0, 0.3, len(index))
    )
    df = pd.DataFrame({"temperature": temperature, "observed": observed}, index=index)
    for flag in _UNMEASURED_FLAGS:
        df[flag] = False

    return df


def _fit_flagged(df, **settings):
    model = HourlyModel(settings=HourlyNonSolarSettings(seed=42, **settings))

    return model.fit(df, is_electricity_data=True, enforce_compliance=False)


def _changed_fields(model_a, model_b):
    dict_a = model_a.to_dict()
    dict_b = model_b.to_dict()

    return sorted(key for key in dict_a if dict_a[key] != dict_b[key])


class TestFitStatisticsFromIncludedDays:
    """A day with fewer than ``min_daily_training_hours`` measured hours is left out of the fit,
    including every statistic the fit learns from the data: the temporal clusters, the
    temperature bin edges, the edge-bin exponential fits, and the feature and load scalers."""

    @pytest.mark.parametrize(
        "temperature_bin",
        [{}, {"method": "set_bin_width", "bin_width": 10, "edge_bin_percent": 0.02}],
        ids=["default-bins", "set-width-bins"],
    )
    @pytest.mark.parametrize(
        "excluded",
        [
            lambda index: index.date == date(2019, 7, 17),
            lambda index: index.date == date(2019, 3, 10),
            lambda index: index.date == date(2019, 11, 3),
            lambda index: (index.month == 3) & (index.dayofweek == 0),
        ],
        ids=["ordinary-day", "23-hour-dst-day", "25-hour-dst-day", "every-march-monday"],
    )
    def test_unmeasured_days_values_leave_the_fitted_model_unchanged(
        self, flagged_hourly, excluded, temperature_bin
    ):
        rows = excluded(flagged_hourly.index)
        flagged_hourly.loc[rows, _UNMEASURED_FLAGS] = True
        temperature = flagged_hourly["temperature"]
        beyond_the_data = np.where(
            np.arange(rows.sum()) % 2 == 0, temperature.max() + 30, temperature.min() - 30
        )
        perturbed = flagged_hourly.copy()
        perturbed.loc[rows, "temperature"] = beyond_the_data
        perturbed.loc[rows, "observed"] *= 5

        original = _fit_flagged(flagged_hourly, temperature_bin=temperature_bin)
        refit = _fit_flagged(perturbed, temperature_bin=temperature_bin)

        changed = _changed_fields(original, refit)
        assert not changed, f"unmeasured days' values changed these fitted fields: {changed}"

    @pytest.mark.parametrize("measured_hours", [12, 24])
    def test_day_with_enough_measured_hours_changes_the_model(
        self, flagged_hourly, measured_hours
    ):
        day = flagged_hourly.index.date == date(2019, 7, 17)
        unmeasured = day & (flagged_hourly.index.hour >= measured_hours)
        flagged_hourly.loc[unmeasured, _UNMEASURED_FLAGS] = True
        perturbed = flagged_hourly.copy()
        perturbed.loc[day & ~unmeasured, "observed"] *= 5

        original = _fit_flagged(flagged_hourly, min_daily_training_hours=12)
        refit = _fit_flagged(perturbed, min_daily_training_hours=12)

        changed = _changed_fields(original, refit)
        assert "coefficients" in changed, (
            f"a day with {measured_hours} measured hours, at a 12-hour minimum, left the "
            f"coefficients unchanged; changed fields: {changed}"
        )

    def test_unmeasured_day_changes_the_model_when_no_minimum_is_set(self, flagged_hourly):
        day = flagged_hourly.index.date == date(2019, 7, 17)
        flagged_hourly.loc[day, _UNMEASURED_FLAGS] = True
        perturbed = flagged_hourly.copy()
        perturbed.loc[day, "observed"] *= 5

        original = _fit_flagged(flagged_hourly, min_daily_training_hours=0)
        refit = _fit_flagged(perturbed, min_daily_training_hours=0)

        changed = _changed_fields(original, refit)
        assert "coefficients" in changed, (
            "with min_daily_training_hours=0 every day is fit, but an unmeasured day's load left "
            f"the coefficients unchanged; changed fields: {changed}"
        )

    @pytest.mark.parametrize(
        "day", [date(2019, 3, 10), date(2019, 11, 3)], ids=["23-hour", "25-hour"]
    )
    def test_complete_dst_day_counts_as_fully_measured(self, flagged_hourly, day):
        model = HourlyModel(settings=HourlyNonSolarSettings(min_daily_training_hours=24))

        df = model._daily_fitting_sufficiency(flagged_hourly)

        on_day = df.index.date == day
        assert df.loc[on_day, "include_date"].all(), (
            f"a complete {on_day.sum()}-hour day was excluded at min_daily_training_hours=24"
        )

    def test_fitted_clusters_omit_a_month_and_weekday_with_no_included_day(
        self, flagged_hourly
    ):
        index = flagged_hourly.index
        flagged_hourly.loc[(index.month == 3) & (index.dayofweek == 0), _UNMEASURED_FLAGS] = True

        model = _fit_flagged(flagged_hourly)

        fitted = [tuple(row[:2]) for row in model.to_dict()["temporal_clusters"]]
        assert (3, 0) not in fitted, "March Mondays have no included day but got a fitted cluster"

    def test_rows_of_a_month_and_weekday_with_no_included_day_get_a_fitted_cluster(
        self, flagged_hourly
    ):
        index = flagged_hourly.index
        flagged_hourly.loc[(index.month == 3) & (index.dayofweek == 0), _UNMEASURED_FLAGS] = True

        model = _fit_flagged(flagged_hourly)

        fitted_labels = {row[2] for row in model.to_dict()["temporal_clusters"]}
        features = model._processed_meter_data_full
        march_mondays = features[(features["month"] == 3) & (features["day_of_week"] == 0)]
        assigned = set(march_mondays["temporal_cluster"])
        assert assigned <= fitted_labels, (
            f"March Monday rows got cluster labels {assigned}, expected a subset of the fitted "
            f"labels {fitted_labels}"
        )

    def test_fit_raises_when_no_day_has_enough_measured_hours(self, flagged_hourly):
        for flag in _UNMEASURED_FLAGS:
            flagged_hourly[flag] = True

        with pytest.raises(DataSufficiencyError, match="no day has at least 12 measured hours"):
            _fit_flagged(flagged_hourly, min_daily_training_hours=12)

    def test_fit_clusters_a_month_and_weekday_seen_only_on_the_23_hour_day(
        self, flagged_hourly
    ):
        """March 10, 2019 is the spring-forward day: its load profile has no hour 2, and it is
        the only measured March Sunday, so the combination's profile is completed from the
        neighbouring hours before clustering."""
        index = flagged_hourly.index
        other_march_sundays = (
            (index.month == 3)
            & (index.dayofweek == 6)
            & (index.date != date(2019, 3, 10))
        )
        flagged_hourly.loc[other_march_sundays, _UNMEASURED_FLAGS] = True

        model = _fit_flagged(flagged_hourly)

        fitted = {tuple(row[:2]) for row in model.to_dict()["temporal_clusters"]}
        message = "March Sundays, measured only on the 23-hour day, got no cluster"
        assert (3, 6) in fitted, message

    def test_predict_labels_a_month_and_weekday_seen_only_on_the_23_hour_day(
        self, flagged_hourly
    ):
        """With every March Sunday unmeasured in the fit, a prediction window whose only Sunday
        is the 23-hour day labels that Sunday from its completed profile: the same label the
        following week's ordinary Sunday gets."""
        index = flagged_hourly.index
        flagged_hourly.loc[
            (index.month == 3) & (index.dayofweek == 6), _UNMEASURED_FLAGS
        ] = True
        model = _fit_flagged(flagged_hourly)
        measured = flagged_hourly.drop(columns=_UNMEASURED_FLAGS)

        def sunday_label(window):
            # each window predicts from its own copy of the fitted model
            fresh = HourlyModel.from_dict(model.to_dict())
            fresh.predict(measured.loc[window], enforce_compliance=False)
            features = fresh._processed_meter_data_full
            labels = set(features.loc[features["day_of_week"] == 6, "temporal_cluster"])
            assert len(labels) == 1, f"the window's one Sunday got labels {labels}"

            return labels.pop()

        dst_week = sunday_label(slice("2019-03-04", "2019-03-10"))
        ordinary_week = sunday_label(slice("2019-03-11", "2019-03-17"))

        assert dst_week == ordinary_week, (
            f"the week whose Sunday is the 23-hour day labeled it {dst_week}, "
            f"the following week labeled its Sunday {ordinary_week}"
        )


_SUPPLEMENTAL = {
    "supplemental_time_series_columns": ["occupancy"],
    "supplemental_categorical_columns": ["shift"],
}

# to_dict fields computed from every baseline day rather than learned from the fit days
_NOT_LEARNED_FIELDS = {"baseline_metrics", "baseline_hour_metrics", "info"}


@pytest.fixture
def supplemental_hourly(flagged_hourly):
    """``flagged_hourly`` with an occupancy series behind part of its load and a day-level shift
    code whose most common value is 0."""
    rng = np.random.default_rng(1)
    index = flagged_hourly.index
    occupied = (index.dayofweek < 5) & (index.hour >= 8) & (index.hour < 18)
    flagged_hourly["occupancy"] = occupied + rng.normal(0, 0.1, len(index))
    dates = pd.Index(index.date)
    days = dates.unique()
    codes = pd.Series(rng.choice([0, 1, 2], size=len(days), p=[0.6, 0.3, 0.1]), index=days)
    flagged_hourly["shift"] = codes.reindex(dates).to_numpy()

    return flagged_hourly


def _supplemental_week(df):
    week = df.drop(columns=_UNMEASURED_FLAGS).loc["2019-07-15":"2019-07-21"].copy()

    return week


def _predicted(model, df, column="predicted"):
    predicted = model.predict(df)[column]

    return predicted


def _warning_data(model, qualified_name):
    data = [warning.data for warning in model.warnings if warning.qualified_name == qualified_name]

    return data


def _restored_without_fill_values(df):
    """A model fit on ``df`` and restored from model JSON stripped of its supplemental fill
    values, with a week of ``df`` missing one occupancy value on 2019-07-17."""
    model_json = _fit_flagged(df, **_SUPPLEMENTAL).to_dict()
    del model_json["supplemental_fill_values"]
    restored = HourlyModel.from_dict(model_json)
    week = _supplemental_week(df)
    week.loc["2019-07-17 15:00", "occupancy"] = np.nan

    return restored, week


class TestMissingSupplementalValues:
    """The model does not impute supplemental inputs: a day missing one is left out of the fit,
    and a missing value at prediction takes its typical baseline value, the median of its hour
    for a series and the most common value for a categorical."""

    @pytest.mark.parametrize(
        ("column", "missing"),
        [
            ("occupancy", lambda index: (index.date == date(2019, 7, 17)) & (index.hour == 15)),
            ("occupancy", lambda index: (index.date == date(2019, 3, 10)) & (index.hour == 15)),
            ("occupancy", lambda index: (index.date == date(2019, 11, 3)) & (index.hour == 1)),
            ("shift", lambda index: index.date == date(2019, 7, 17)),
        ],
        ids=["series-hour", "series-hour-23-hour-day", "series-repeated-hour", "categorical-day"],
    )
    def test_day_missing_a_supplemental_value_leaves_the_fitted_model_unchanged(
        self, supplemental_hourly, column, missing
    ):
        index = supplemental_hourly.index
        rows = missing(index)
        supplemental_hourly.loc[rows, column] = np.nan
        day = index.date == index[rows][0].date()
        perturbed = supplemental_hourly.copy()
        perturbed.loc[day, "temperature"] = supplemental_hourly["temperature"].max() + 30
        perturbed.loc[day, "observed"] *= 5
        perturbed.loc[day, "occupancy"] *= 10

        # with no measured-hour minimum, only a missing supplemental value can leave a day out
        original = _fit_flagged(supplemental_hourly, min_daily_training_hours=0, **_SUPPLEMENTAL)
        refit = _fit_flagged(perturbed, min_daily_training_hours=0, **_SUPPLEMENTAL)

        changed = set(_changed_fields(original, refit)) - _NOT_LEARNED_FIELDS
        assert not changed, (
            f"a day missing a supplemental value changed these fitted fields: {sorted(changed)}"
        )

    @pytest.mark.parametrize(
        "shift_missing",
        [
            lambda index: np.zeros(len(index), dtype=bool),
            lambda index: (index.date == date(2019, 7, 17)) & (index.hour >= 12),
        ],
        ids=["complete-day", "shift-missing-after-noon"],
    )
    def test_day_with_every_supplemental_input_changes_the_model(
        self, supplemental_hourly, shift_missing
    ):
        """A categorical present at some hours gives the day its first value, so the day is
        fit."""
        index = supplemental_hourly.index
        supplemental_hourly.loc[shift_missing(index), "shift"] = np.nan
        perturbed = supplemental_hourly.copy()
        perturbed.loc[index.date == date(2019, 7, 17), "observed"] *= 5

        original = _fit_flagged(supplemental_hourly, **_SUPPLEMENTAL)
        refit = _fit_flagged(perturbed, **_SUPPLEMENTAL)

        changed = _changed_fields(original, refit)
        assert "coefficients" in changed, (
            f"a day with every supplemental input left the coefficients unchanged; changed "
            f"fields: {changed}"
        )

    def test_baseline_metrics_leave_out_only_the_hours_missing_a_supplemental_value(
        self, supplemental_hourly
    ):
        """Ten hours missing occupancy take their day out of the fit, but only those ten hours
        out of the baseline metrics: the day's other fourteen are scored."""
        index = supplemental_hourly.index
        missing = (index.date == date(2019, 7, 17)) & (index.hour < 10)
        supplemental_hourly.loc[missing, "occupancy"] = np.nan

        model = _fit_flagged(supplemental_hourly, **_SUPPLEMENTAL)

        scored = int((~missing).sum())
        assert model.baseline_metrics.n == scored, (
            f"expected {scored} scored hours, got {model.baseline_metrics.n}"
        )
        expected_by_hour = pd.Series(index.hour[~missing]).value_counts().to_dict()
        scored_by_hour = {hour: metrics.n for hour, metrics in model.baseline_hour_metrics.items()}
        assert scored_by_hour == expected_by_hour, (
            f"scored hours by hour of day {scored_by_hour}, expected {expected_by_hour}"
        )

    def test_prediction_uncertainty_leaves_out_hours_given_a_typical_value(
        self, supplemental_hourly
    ):
        """A reporting hour whose supplemental value is filled counts toward its hour's
        uncertainty the way an interpolated hour does: not at all."""
        model = _fit_flagged(supplemental_hourly, **_SUPPLEMENTAL)
        filled = _supplemental_week(supplemental_hourly)
        filled.loc["2019-07-17 15:00", "occupancy"] = np.nan
        interpolated = _supplemental_week(supplemental_hourly)
        interpolated.loc["2019-07-17 15:00", "observed"] = np.nan

        filled_unc = _predicted(model, filled, column="predicted_unc")
        interpolated_unc = _predicted(model, interpolated, column="predicted_unc")

        hour_15 = filled_unc.index.hour == 15
        assert filled_unc[hour_15].notna().all(), "hour 15 got no uncertainty"
        np.testing.assert_allclose(filled_unc[hour_15], interpolated_unc[hour_15], rtol=1e-12)

    def test_fit_raises_when_every_day_is_missing_a_supplemental_value(self, supplemental_hourly):
        supplemental_hourly.loc[supplemental_hourly.index.hour == 0, "occupancy"] = np.nan

        with pytest.raises(DataSufficiencyError, match="missing a supplemental value"):
            _fit_flagged(supplemental_hourly, **_SUPPLEMENTAL)

    def test_fit_reports_the_days_left_out(self, supplemental_hourly):
        supplemental_hourly.loc["2019-07-17 15:00", "occupancy"] = np.nan
        supplemental_hourly.loc["2019-08-01", "shift"] = np.nan

        model = _fit_flagged(supplemental_hourly, **_SUPPLEMENTAL)

        reported = _warning_data(model, "eemeter.data_quality.supplemental_days_excluded")
        assert reported == [{"days": 2}], f"expected one warning for 2 days, got {reported}"

    def test_missing_series_value_predicts_as_its_hours_baseline_median(
        self, supplemental_hourly
    ):
        model = _fit_flagged(supplemental_hourly, **_SUPPLEMENTAL)
        index = supplemental_hourly.index
        median = supplemental_hourly.loc[index.hour == 15, "occupancy"].median()
        with_gap = _supplemental_week(supplemental_hourly)
        with_gap.loc["2019-07-17 15:00", "occupancy"] = np.nan
        with_median = _supplemental_week(supplemental_hourly)
        with_median.loc["2019-07-17 15:00", "occupancy"] = median

        predicted = _predicted(model, with_gap)

        assert predicted.notna().all(), f"{predicted.isna().sum()} hours were not predicted"
        np.testing.assert_allclose(predicted, _predicted(model, with_median), rtol=1e-12)

    def test_day_without_a_categorical_value_predicts_as_its_most_common_value(
        self, supplemental_hourly
    ):
        model = _fit_flagged(supplemental_hourly, **_SUPPLEMENTAL)
        index = supplemental_hourly.index
        most_common = supplemental_hourly["shift"].groupby(index.date).first().mode().iloc[0]
        with_gap = _supplemental_week(supplemental_hourly)
        with_gap.loc["2019-07-17", "shift"] = np.nan
        with_most_common = _supplemental_week(supplemental_hourly)
        with_most_common.loc["2019-07-17", "shift"] = most_common

        predicted = _predicted(model, with_gap)

        assert predicted.notna().all(), f"{predicted.isna().sum()} hours were not predicted"
        np.testing.assert_allclose(predicted, _predicted(model, with_most_common), rtol=1e-12)

    def test_prediction_reports_the_values_filled(self, supplemental_hourly):
        model = _fit_flagged(supplemental_hourly, **_SUPPLEMENTAL)
        week = _supplemental_week(supplemental_hourly)
        week.loc["2019-07-17 15:00":"2019-07-17 16:00", "occupancy"] = np.nan
        week.loc["2019-07-18", "shift"] = np.nan

        _predicted(model, week)

        reported = _warning_data(model, "eemeter.data_quality.supplemental_values_filled")
        expected = [{"days": 2, "values": 3}]
        assert reported == expected, f"expected {expected}, got {reported}"

    def test_serialized_model_fills_missing_values_as_fitted(self, supplemental_hourly):
        model = _fit_flagged(supplemental_hourly, **_SUPPLEMENTAL)
        restored = HourlyModel.from_json(model.to_json())
        week = _supplemental_week(supplemental_hourly)
        week.loc["2019-07-17 15:00", "occupancy"] = np.nan
        week.loc["2019-07-18", "shift"] = np.nan

        restored_predicted = _predicted(restored, week)

        assert restored_predicted.notna().all(), "the restored model left hours unpredicted"
        np.testing.assert_array_equal(restored_predicted, _predicted(model, week))

    def test_model_json_without_fill_values_leaves_a_gap_day_unpredicted(
        self, supplemental_hourly
    ):
        restored, week = _restored_without_fill_values(supplemental_hourly)

        predicted = _predicted(restored, week)

        gap_day = predicted.index.date == date(2019, 7, 17)
        assert predicted[gap_day].isna().all(), "the day with a gap was predicted"
        assert predicted[~gap_day].notna().all(), "days without a gap were left unpredicted"

    def test_model_json_without_fill_values_reports_nothing_filled(self, supplemental_hourly):
        restored, week = _restored_without_fill_values(supplemental_hourly)

        _predicted(restored, week)

        reported = _warning_data(restored, "eemeter.data_quality.supplemental_values_filled")
        assert reported == [], f"nothing was filled, but the model reported {reported}"

    def test_baseline_ending_before_its_last_hour_predicts_every_hour(
        self, baseline_with_supplemental_features
    ):
        """The bundled baseline ends at 22:00, so its last day gets a 23:00 row without
        supplemental values; that day is left out of the fit and still predicted."""
        df = baseline_with_supplemental_features
        model = HourlyModel(
            settings={
                "supplemental_time_series_columns": ["custom_ts_feature"],
                "supplemental_categorical_columns": ["custom_cat_feature"],
            }
        ).fit(df, is_electricity_data=True)

        predicted = model.predict(df)["predicted"]

        assert predicted.notna().all(), f"{predicted.isna().sum()} hours were not predicted"


def test_dst_days_are_found_by_row_count_not_by_missing_values():
    """A missing value in the frame's first column does not make its day look 23 hours long."""
    index = pd.date_range(
        "2019-01-01", "2020-01-01", freq="h", tz="America/New_York", inclusive="left"
    )
    df = pd.DataFrame({"occupancy": 1.0, "observed": 1.0}, index=index)
    df.loc["2019-07-17 15:00", "occupancy"] = np.nan
    days = sorted(set(index.date))

    interp, mean = _get_dst_indices(df)

    assert [(days[i], hour) for i, hour in interp] == [(date(2019, 3, 10), 2)]
    assert [(days[i], hour) for i, hour in mean] == [(date(2019, 11, 3), 1)]


def test_hourly_profiles_fill_a_skipped_hour_from_its_neighbours():
    """A combination missing one hour of the day gets the mean of the neighbouring hours there,
    a combination missing an edge hour gets its one neighbour, and a complete combination is
    unchanged."""
    rows = []
    for hour in range(24):
        if hour != 2:
            rows.append((3, 6, hour, 10.0 * hour))

        if hour != 0:
            rows.append((3, 0, hour, 100.0 + hour))

        rows.append((4, 0, hour, float(hour)))

    columns = ["month", "day_of_week", "hour_of_day", "observed"]
    df = pd.DataFrame(rows, columns=columns)

    profiles = _hourly_profiles(df, ["month", "day_of_week"], "mean")

    assert profiles.shape == (3, 24), f"expected 3 by 24 hours, got {profiles.shape}"
    assert not profiles.isna().any().any(), "a profile kept a missing hour"
    hour_2 = profiles.loc[(3, 6), 2]
    assert hour_2 == pytest.approx(20.0), "hour 2 is not the mean of hours 1 and 3"
    hour_0 = profiles.loc[(3, 0), 0]
    assert hour_0 == pytest.approx(101.0), "hour 0 did not take hour 1's value"
    np.testing.assert_allclose(profiles.loc[(4, 0)].to_numpy(), np.arange(24.0))


def _first_month(reporting):
    first_month = reporting.index < reporting.index.min() + pd.DateOffset(months=1)

    return reporting.loc[first_month]


def test_predict_leaves_the_fitted_temporal_clusters_unchanged(baseline, reporting):
    """A prediction over a window missing most month and weekday combinations labels
    them for that window only: the fitted cluster table and its serialized form stay."""
    hm = HourlyModel().fit(baseline, is_electricity_data=True)
    fitted = hm._df_temporal_clusters.copy()
    serialized = hm.to_dict()["temporal_clusters"]

    hm.predict(_first_month(reporting))

    pd.testing.assert_frame_equal(hm._df_temporal_clusters, fitted)
    assert hm.to_dict()["temporal_clusters"] == serialized


def test_predict_labels_unseen_combinations_for_that_prediction_only(baseline, reporting):
    """A fit through November has no December clusters; a prediction reaching into
    December labels those combinations for itself and leaves the fitted table as fitted."""
    through_november = baseline[baseline.index.month != 12]
    hm = HourlyModel().fit(through_november, is_electricity_data=True, enforce_compliance=False)
    fitted = hm._df_temporal_clusters.copy()
    assert 12 not in fitted.index.get_level_values("month")

    days = reporting.index
    window = reporting.loc[((days.month == 11) & (days.day >= 16)) | ((days.month == 12) & (days.day <= 15))]
    predicted = hm.predict(window)

    assert predicted.loc[predicted.index.month == 12, "predicted"].notna().all()
    pd.testing.assert_frame_equal(hm._df_temporal_clusters, fitted)


class TestFrameAPI:
    """fit/predict on plain dataframes: the argument contract, trimming, refusal,
    and serialization behavior that the frame API adds on top of the data classes."""

    def test_fit_rejects_data_object(self, baseline):
        baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
        with pytest.raises(TypeError):
            HourlyModel().fit(baseline_data, is_electricity_data=True)

    def test_fit_rejects_positional_is_electricity_data(self, baseline):
        with pytest.raises(TypeError):
            HourlyModel().fit(baseline, True)

    def test_fit_with_pv_start(self, baseline):
        pv_start = baseline.index[len(baseline) // 2].date()
        hm = HourlyModel(settings=HourlySolarSettings()).fit(
            baseline, is_electricity_data=True, pv_start=pv_start
        )
        assert hm._pv_start == pv_start

        # pv_start must reach the baseline data object and split has_pv accordingly,
        # not just be stored on the model
        expected_has_pv = (hm.baseline_df.index.date >= pv_start).sum()
        assert hm.baseline_df["has_pv"].sum() == expected_has_pv

    def test_predict_without_observed_column_works(self, baseline, reporting):
        hm = HourlyModel().fit(baseline, is_electricity_data=True)
        reporting_no_observed = reporting.drop(columns=["observed"])

        prediction = hm.predict(reporting_no_observed)

        assert "predicted" in prediction.columns

    def test_baseline_df_is_none_before_fit_and_a_dataframe_after(self, baseline):
        hm = HourlyModel()
        assert hm.baseline_df is None

        hm.fit(baseline, is_electricity_data=True)

        assert isinstance(hm.baseline_df, pd.DataFrame)

    def test_padded_baseline_trims_to_the_same_fit_and_warns_once(self, baseline):
        pad = pd.DataFrame(
            np.nan,
            index=pd.date_range(
                baseline.index.min() - pd.Timedelta(hours=3),
                baseline.index.min() - pd.Timedelta(hours=1),
                freq="h",
            ),
            columns=baseline.columns,
        )
        trailing_pad = pd.DataFrame(
            np.nan,
            index=pd.date_range(
                baseline.index.max() + pd.Timedelta(hours=1),
                baseline.index.max() + pd.Timedelta(hours=5),
                freq="h",
            ),
            columns=baseline.columns,
        )
        padded = pd.concat([pad, baseline, trailing_pad])

        hm_padded = HourlyModel().fit(padded, is_electricity_data=True)
        hm_unpadded = HourlyModel().fit(baseline, is_electricity_data=True)

        np.testing.assert_allclose(
            hm_padded._model.coef_, hm_unpadded._model.coef_, rtol=1e-12
        )

        trim_warnings = [
            w for w in hm_padded.warnings
            if w.qualified_name == "eemeter.data_quality.edge_rows_trimmed"
        ]
        assert len(trim_warnings) == 1
        assert trim_warnings[0].data == {"leading": 3, "trailing": 5}

    def test_predict_does_not_trim_padded_frame(self, baseline, reporting):
        hm = HourlyModel().fit(baseline, is_electricity_data=True)
        pad_start = reporting.index.min() - pd.Timedelta(hours=3)
        pad = pd.DataFrame(
            np.nan,
            index=pd.date_range(pad_start, reporting.index.min() - pd.Timedelta(hours=1), freq="h"),
            columns=reporting.columns,
        )
        padded_reporting = pd.concat([pad, reporting])

        prediction = hm.predict(padded_reporting)

        assert prediction.index.min() <= pad_start

    def test_disqualified_frame_raises_with_matching_disqualification(self, baseline):
        baseline["observed"] = 0
        hm = HourlyModel()

        with pytest.raises(DataSufficiencyError) as exc_info:
            hm.fit(baseline, is_electricity_data=True)

        assert exc_info.value.disqualification
        assert exc_info.value.disqualification == hm.disqualification

    def test_is_electricity_data_and_pv_start_round_trip_dict(self, baseline):
        pv_start = baseline.index[len(baseline) // 2].date()
        hm = HourlyModel(settings=HourlySolarSettings()).fit(
            baseline, is_electricity_data=False, pv_start=pv_start
        )

        rebuilt = HourlyModel.from_dict(hm.to_dict())

        assert rebuilt._is_electricity_data is False
        assert rebuilt._pv_start == pv_start

    def test_deserialized_model_predicts_on_a_frame(self, baseline, reporting):
        hm = HourlyModel().fit(baseline, is_electricity_data=True)
        rebuilt = HourlyModel.from_dict(hm.to_dict())

        prediction = rebuilt.predict(reporting)

        assert "predicted" in prediction.columns

    def test_prior_format_payload_requires_is_electricity_data(self, baseline):
        hm = HourlyModel().fit(baseline, is_electricity_data=True)
        payload = hm.to_dict()
        del payload["info"]["is_electricity_data"]
        del payload["info"]["pv_start"]

        with pytest.raises(ValueError):
            HourlyModel.from_dict(payload)

    def test_prior_format_payload_loads_with_is_electricity_data_and_warns(self, baseline, reporting):
        hm = HourlyModel().fit(baseline, is_electricity_data=True)
        payload = hm.to_dict()
        del payload["info"]["is_electricity_data"]
        del payload["info"]["pv_start"]

        rebuilt = HourlyModel.from_dict(payload, is_electricity_data=True)

        prior_format_warnings = [
            w for w in rebuilt.warnings
            if isinstance(w, EEMeterWarning)
            and w.qualified_name == "eemeter.serialization.prior_format"
        ]
        assert len(prior_format_warnings) == 1

        prediction = rebuilt.predict(reporting)
        assert "predicted" in prediction.columns

    def test_is_electricity_data_rejected_when_payload_already_carries_it(self, baseline):
        hm = HourlyModel().fit(baseline, is_electricity_data=True)
        payload = hm.to_dict()

        with pytest.raises(ValueError):
            HourlyModel.from_dict(payload, is_electricity_data=True)


def test_settings_deviations_property_reports_hourly_developer_tier_changes(baseline):
    model = HourlyModel(settings=HourlyNonSolarSettings(cvrmse_threshold=1.2)).fit(
        baseline, is_electricity_data=True
    )

    assert model.settings_deviations == {
        "cvrmse_threshold": {"value": 1.2, "default": 1.4}
    }


def test_settings_deviations_property_is_empty_for_default_hourly_settings(baseline):
    model = HourlyModel().fit(baseline, is_electricity_data=True)

    assert model.settings_deviations == {}
