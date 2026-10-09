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
    HourlyBaselineData,
    HourlyReportingData,
    HourlyModel,
    HourlySolarSettings,
    HourlyNonSolarSettings,
)
from opendsm.eemeter.models.hourly.model import _fit_exp_growth_decay, _hourly_profiles
from opendsm.eemeter.models.hourly.settings import BaseHourlySettings
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)
import numpy as np
import pandas as pd
import pytest
from datetime import date
from math import ceil
from threadpoolctl import threadpool_info



def test_good_data(baseline, reporting):
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    p1 = hm.predict(reporting_data)
    assert np.isclose(
        p1["predicted"].sum(), 1135000, rtol=1e-2
    )  # quick check that model fit isn't changing drastically
    serialized = hm.to_json()
    hm2 = HourlyModel.from_json(serialized)
    p2 = hm2.predict(reporting_data)
    assert p1.equals(p2)


def test_misaligned_data(baseline, reporting):
    reporting.index = reporting.index.shift(8, freq="h")
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    hm.predict(reporting_data)


def test_tz_naive(baseline):
    baseline.index = baseline.index.tz_localize(None)
    with pytest.raises(ValueError):
        HourlyBaselineData(baseline, is_electricity_data=True)


def test_tz_mismatch(baseline):
    # might allow automatic adjustment from the model in the future, but hard requirement for now
    baseline.index = baseline.index.tz_convert("America/Los_Angeles")
    reporting = baseline.copy()
    reporting.index = reporting.index.tz_convert("America/New_York")
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    with pytest.raises(ValueError):
        hm.predict(reporting_data)


def test_predict_missing_fit_features(baseline, reporting):
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    hm = HourlyModel(settings=HourlySolarSettings()).fit(baseline_data)
    reporting.drop("ghi", axis=1, inplace=True)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    with pytest.raises(ValueError):
        hm.predict(reporting_data)


def test_nonsolar_predict_with_ghi(baseline, reporting, caplog):
    baseline.drop("ghi", axis=1, inplace=True)
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    hm = HourlyModel().fit(baseline_data)
    reporting_data = HourlyReportingData(reporting, is_electricity_data=True)
    with caplog.at_level("WARNING"):
        hm.predict(reporting_data)
        assert "GHI" in caplog.text


def test_forced_solar_model_fit_no_ghi(baseline):
    baseline = baseline.drop("ghi", axis=1)
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    with pytest.raises(ValueError):
        HourlyModel(settings=HourlySolarSettings()).fit(baseline_data)


def test_forced_nonsolar_model_fit_with_ghi(baseline):
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    hm = HourlyModel(settings=HourlyNonSolarSettings()).fit(baseline_data)
    assert [
        w for w in hm.warnings if w.qualified_name == "eemeter.potential_model_mismatch"
    ]


def test_no_data(baseline):
    baseline["observed"] = 0
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


def test_negative_meter_values(baseline):
    baseline.loc["2018-01-08", "observed"] = -1

    # gas data can't be negative
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=False)
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)

    # elec can
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    HourlyModel().fit(baseline_data)


def test_invalid_baseline_lengths(baseline):
    # TODO import min/max length from constants
    MAX_BASELINE_HOURS = 8760
    MIN_BASELINE_HOURS = ceil(MAX_BASELINE_HOURS * 0.9) - 24
    short_df = baseline.iloc[:MIN_BASELINE_HOURS]

    extra_days = baseline.iloc[-24*2:]
    extra_days.index += pd.Timedelta(days=2)
    long_df = pd.concat([baseline, extra_days])

    short_baseline = HourlyBaselineData(short_df, is_electricity_data=True)
    long_baseline = HourlyBaselineData(long_df, is_electricity_data=True)
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(short_baseline)
    hm_short = HourlyModel().fit(short_baseline, ignore_disqualification=True)
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(long_baseline)
    hm_long = HourlyModel().fit(long_baseline, ignore_disqualification=True)


def test_low_freq_temp(baseline):
    baseline["temperature"] = baseline["temperature"].resample("D").mean()
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    assert_dq(
        baseline_data,
        ["eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data"],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


def test_low_freq_meter(baseline):
    baseline["observed"] = baseline["observed"].resample("D").mean()
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    assert_dq(
        baseline_data,
        ["eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data"],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


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

    baseline_data = HourlyBaselineData(invalid_baseline, is_electricity_data=True)
    assert_dq(
        baseline_data, ["eemeter.sufficiency_criteria.missing_monthly_temperature_data"]
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)
    baseline_data = HourlyBaselineData(valid_baseline, is_electricity_data=True)
    HourlyModel().fit(baseline_data)

    baseline_data = HourlyBaselineData(invalid_temp, is_electricity_data=True)
    assert_dq(
        baseline_data,
        [
            "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
            "eemeter.sufficiency_criteria.missing_monthly_temperature_data",
            "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data",
        ],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)

    baseline_data = HourlyBaselineData(invalid_meter, is_electricity_data=True)
    assert_dq(
        baseline_data,
        [
            "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
            "eemeter.sufficiency_criteria.missing_monthly_observed_data",
            "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
        ],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


def test_monthly_ghi_percentage(baseline):
    # create datetimeindex where a little over 10% of days are missing in feb, but still 90% overall
    missing_idx = pd.date_range(
        start=baseline.index.min(), end=baseline.index.max(), freq="h"
    )
    missing_idx = missing_idx[missing_idx.day < 4]

    invalid_ghi = baseline.copy()
    invalid_ghi.loc[invalid_ghi.index.day < 5, "ghi"] = np.nan

    baseline_data = HourlyBaselineData(invalid_ghi, is_electricity_data=True)
    assert_dq(
        baseline_data,
        [
            "eemeter.sufficiency_criteria.missing_monthly_ghi_data",
        ],
    )
    with pytest.raises(DataSufficiencyError):
        HourlyModel().fit(baseline_data)


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
    baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
    model = HourlyModel().fit(baseline_data)
    assert_dq(baseline_data, ["eemeter.model_fit_metrics"])
    with pytest.raises(DisqualifiedModelError):
        model.predict(baseline_data)


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
        baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
        seen = []
        original = HourlyModel._prepare_features

        def spy(self, meter_data):
            seen.append([info["num_threads"] for info in threadpool_info()])

            return original(self, meter_data)

        monkeypatch.setattr(HourlyModel, "_prepare_features", spy)
        HourlyModel().fit(baseline_data)

        assert seen, "_prepare_features was never called; the spy missed the fit path"
        for counts in seen:
            assert set(counts) == {1}, (
                f"expected every native pool pinned to 1 thread during fit, got {counts}"
            )

    def test_limit_does_not_leak_past_fit(self, baseline):
        """The cap is scoped, so the caller's threading is restored afterwards."""
        baseline_data = HourlyBaselineData(baseline, is_electricity_data=True)
        before = [info["num_threads"] for info in threadpool_info()]
        HourlyModel().fit(baseline_data)
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
    data = HourlyBaselineData(df, is_electricity_data=True)
    model = HourlyModel(settings=HourlyNonSolarSettings(seed=42, **settings))

    return model.fit(data, ignore_disqualification=True)


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
            reporting = HourlyReportingData(
                measured.loc[window], is_electricity_data=True
            )
            fresh.predict(reporting, ignore_disqualification=True)
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
