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

import numpy as np
import pandas as pd
import pydantic
import pytest
import statsmodels.api as sm

from opendsm.common.metrics import (
    BaselineMetrics,
    BaselineMetricsFromDict,
    BaselineProfileMetricsFromDict,
    ColumnMetrics,
    ProfileSettings,
    ReportingMetrics,
    acf,
    compute_baseline_profile,
)
from opendsm.common.stats.basic import t_stat



# ---------------------------------------------------------------------------
# acf
# ---------------------------------------------------------------------------

def test_acf():
    """ACF per method on a linear ramp; pins each method's defined output."""
    x = np.array([1, 2, 3, 4, 5])

    # Default is MOVING_STATS: every shifted slice of a linear ramp is perfectly
    # correlated, so each lag is 1.0.
    assert np.allclose(acf(x), [1.0, 1.0, 1.0, 1.0])
    assert np.allclose(acf(x, ac_type="moving_stats"), [1.0, 1.0, 1.0, 1.0])

    # STATIONARY_CORRELATE is the standard biased ACF of [1, 2, 3, 4, 5].
    assert np.allclose(acf(x, ac_type="stationary_correlate"), [1.0, 0.4, -0.1, -0.4])

    # lag_n caps the number of lags returned (lags 0..lag_n).
    assert np.allclose(acf(x, lag_n=1), [1.0, 1.0])
    assert np.allclose(acf(x, ac_type="stationary_correlate", lag_n=1), [1.0, 0.4])


def test_acf_stationary_fft_matches_correlate():
    """The zero-padded FFT method equals the correlate method (both linear ACF)."""
    x = np.array([1, 2, 3, 4, 5])

    assert np.allclose(
        acf(x, ac_type="stationary_stats_fft"),
        acf(x, ac_type="stationary_correlate"),
    )


@pytest.mark.parametrize("ac_type", ["stationary_correlate", "stationary_stats_fft"])
def test_acf_matches_statsmodels_on_ar1(ac_type):
    """Both stationary methods reproduce the statsmodels biased ACF of an AR(1)."""
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(2000)
    x = np.zeros(2000)
    for i in range(1, 2000):
        x[i] = 0.7 * x[i - 1] + noise[i]

    reference = sm.tsa.acf(x, nlags=5, fft=False)

    assert np.allclose(acf(x, lag_n=5, ac_type=ac_type), reference, atol=1e-12)


def test_acf_constant_series_is_nan():
    """A zero-variance series has undefined ACF (divide by zero variance)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        result = acf(np.array([5.0, 5.0, 5.0, 5.0]), ac_type="stationary_correlate")

    assert np.all(np.isnan(result))


def test_acf_single_point_lag_zero():
    """Lag 0 of any series is perfect self-correlation."""
    assert np.allclose(acf(np.array([5.0]), lag_n=0), [1.0])


# ---------------------------------------------------------------------------
# BaselineMetrics — analytic values on a hand-computed case
# ---------------------------------------------------------------------------

@pytest.fixture
def known_case():
    """observed-predicted residuals are ±10, so RMSE=MAE=10, MBE=0, mean obs=250."""
    df = pd.DataFrame(
        {"observed": [100.0, 200.0, 300.0, 400.0], "predicted": [110.0, 190.0, 310.0, 390.0]}
    )

    return BaselineMetrics(df=df, num_model_params=1)


def test_baseline_metrics_error_terms(known_case):
    """Error aggregates equal their closed forms on the ±10-residual case."""
    assert known_case.n == 4
    assert known_case.mae == pytest.approx(10.0)
    assert known_case.rmse == pytest.approx(10.0)
    assert known_case.mse == pytest.approx(100.0)
    assert known_case.sse == pytest.approx(400.0)
    assert known_case.max_error == pytest.approx(10.0)
    assert known_case.medae == pytest.approx(10.0)


def test_baseline_metrics_normalized_terms(known_case):
    """CVRMSE/NMBE normalize by the observed mean (250); MBE is zero here."""
    assert known_case.observed.mean == pytest.approx(250.0)
    assert known_case.mbe == pytest.approx(0.0)
    assert known_case.nmbe == pytest.approx(0.0)
    assert known_case.cvrmse == pytest.approx(0.04)
    assert known_case.nmae == pytest.approx(0.04)


def test_baseline_metrics_perfect_fit_is_zero_error():
    """Identical observed/predicted gives zero error and unit correlation."""
    df = pd.DataFrame({"observed": [1.0, 2.0, 3.0, 4.0], "predicted": [1.0, 2.0, 3.0, 4.0]})
    m = BaselineMetrics(df=df, num_model_params=1)

    assert m.rmse == pytest.approx(0.0)
    assert m.cvrmse == pytest.approx(0.0)
    assert m.r_squared == pytest.approx(1.0)
    assert m.pearson_r == pytest.approx(1.0)


def test_baseline_metrics_zero_observed_mean_clamps_to_nan():
    """safe_divide returns NaN (not inf) when the observed mean is ~0."""
    df = pd.DataFrame({"observed": [-1.0, 1.0, -1.0, 1.0], "predicted": [0.0, 0.0, 0.0, 0.0]})
    m = BaselineMetrics(df=df, num_model_params=1)

    assert np.isnan(m.cvrmse)
    assert np.isnan(m.nmbe)


def test_baseline_metrics_negative_mean_clamps_to_nan():
    """A negative observed mean is below the min denominator, so CVRMSE is NaN."""
    df = pd.DataFrame({"observed": [-100.0, -200.0], "predicted": [-110.0, -190.0]})
    m = BaselineMetrics(df=df, num_model_params=1)

    assert m.mae == pytest.approx(10.0)
    assert np.isnan(m.cvrmse)


def test_baseline_metrics_single_point_no_crash():
    """A single observation yields RMSE from its residual and a floored ddof of 1."""
    df = pd.DataFrame({"observed": [5.0], "predicted": [4.0]})
    m = BaselineMetrics(df=df, num_model_params=1)

    assert m.n == 1
    assert m.rmse == pytest.approx(1.0)
    assert m.ddof == 1
    assert m.observed.std == pytest.approx(0.0)


def test_baseline_metrics_all_nan_filtered_to_empty():
    """Non-finite rows are dropped; an all-NaN frame leaves n=0 and NaN metrics."""
    df = pd.DataFrame({"observed": [np.nan, np.nan], "predicted": [np.nan, np.nan]})
    m = BaselineMetrics(df=df, num_model_params=1)

    assert m.n == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        assert np.isnan(m.rmse)


def test_baseline_metrics_empty_dataframe_raises():
    """An empty input frame raises before any metric is computed."""
    df = pd.DataFrame({"observed": [], "predicted": []})
    m = BaselineMetrics(df=df, num_model_params=1)

    with pytest.raises(ValueError, match="at least one row"):
        _ = m.n


# ---------------------------------------------------------------------------
# ColumnMetrics
# ---------------------------------------------------------------------------

def test_column_metrics_basic_statistics():
    """ColumnMetrics computes sum/mean/variance/std/median on a known series."""
    cm = ColumnMetrics(series=pd.Series([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]))

    assert cm.sum == pytest.approx(40.0)
    assert cm.mean == pytest.approx(5.0)
    assert cm.variance == pytest.approx(4.0)
    assert cm.std == pytest.approx(2.0)
    assert cm.median == pytest.approx(4.5)
    assert cm.sum_squared == pytest.approx(232.0)


def test_column_metrics_empty_series_mean_is_zero():
    """An empty series reports mean 0.0 rather than dividing by zero."""
    cm = ColumnMetrics(series=pd.Series([], dtype=float))

    assert cm.mean == 0.0


def test_column_metrics_dispersion_descriptors_finite():
    """IQR, scaled MAD, skew, kurtosis and cvstd are finite on a normal sample."""
    cm = ColumnMetrics(series=pd.Series(np.random.default_rng(1).normal(10, 2, 500)))

    assert cm.iqr > 0
    assert cm.MAD_scaled == pytest.approx(2.0, abs=0.4)
    assert cm.cvstd == pytest.approx(0.2, abs=0.05)
    assert np.isfinite(cm.skew)
    assert np.isfinite(cm.kurtosis)


# ---------------------------------------------------------------------------
# BaselineMetrics — full property surface on a realistic case
# ---------------------------------------------------------------------------

@pytest.fixture
def realistic_baseline():
    """48 hourly points, predicted = observed + small noise (a good fit)."""
    rng = np.random.default_rng(0)
    observed = rng.normal(100.0, 10.0, 48)
    predicted = observed + rng.normal(0.0, 3.0, 48)
    df = pd.DataFrame(
        {"observed": observed, "predicted": predicted},
        index=pd.date_range("2020-01-01", periods=48, freq="h"),
    )

    return BaselineMetrics(df=df, num_model_params=2)


def test_baseline_metrics_goodness_of_fit_relationships(realistic_baseline):
    """The fit-quality metrics obey their defining bounds and identities."""
    m = realistic_baseline

    assert 0.0 <= m.r_squared <= 1.0
    assert m.r_squared == pytest.approx(m.pearson_r**2, abs=1e-9)
    assert m.nse <= 1.0
    assert 0.0 < m.nnse <= 1.0
    assert 0.0 <= m.explained_variance_score <= 1.0
    assert 0.0 <= m.wi <= 1.0
    assert m.pi == pytest.approx(m.pearson_r * m.wi)
    assert m.kge <= 1.0


def test_baseline_metrics_accuracy_fractions_are_monotone(realistic_baseline):
    """A10 <= A20 <= A30 and each is a fraction in [0, 1]."""
    m = realistic_baseline

    assert 0.0 <= m.a10 <= m.a20 <= m.a30 <= 1.0


def test_baseline_metrics_percentage_errors_nonnegative(realistic_baseline):
    """All percentage-error families are non-negative."""
    m = realistic_baseline

    for value in [m.mape, m.smape, m.wape, m.swape, m.maape]:
        assert value >= 0.0


def test_baseline_metrics_adjusted_rmse_identities(realistic_baseline):
    """Each adjusted/autocorr RMSE equals √(SSE / its dof), and exceeds plain RMSE.

    ddof = n - p and n_prime (autocorr-effective n) are both < n, so dividing
    SSE by them inflates the RMSE relative to the unadjusted √(SSE/n).
    """
    m = realistic_baseline

    assert 1 <= m.ddof < m.n
    assert m.n_prime >= 1
    assert m.ddof_autocorr >= 1
    assert m.rmse_adj == pytest.approx((m.sse / m.ddof) ** 0.5)
    assert m.rmse_autocorr == pytest.approx((m.sse / m.n_prime) ** 0.5)
    assert m.rmse_autocorr_adj == pytest.approx((m.sse / m.ddof_autocorr) ** 0.5)
    assert m.rmse_adj > m.rmse


def test_baseline_metrics_normalization_identities(realistic_baseline):
    """CVRMSE/PN variants divide their RMSE by the observed mean / IQR respectively."""
    m = realistic_baseline
    mean = m.observed.mean
    iqr = m.observed.iqr

    assert m.cvrmse_adj == pytest.approx(m.rmse_adj / mean)
    assert m.cvrmse_autocorr == pytest.approx(m.rmse_autocorr / mean)
    assert m.cvrmse_autocorr_adj == pytest.approx(m.rmse_autocorr_adj / mean)
    assert m.pnrmse == pytest.approx(m.rmse / iqr)
    assert m.pnrmse_adj == pytest.approx(m.rmse_adj / iqr)
    assert m.pnmae == pytest.approx(m.mae / iqr)
    assert m.pnmbe == pytest.approx(m.mbe / iqr)
    assert 0.0 <= m.index_of_agreement <= 1.0


def test_baseline_metrics_pi_rating_buckets_by_quality():
    """pi_rating maps a near-perfect fit to 'excellent' and a poor fit to 'very bad'.

    Performance index pi = pearson_r · willmott_index, so a tight fit scores
    high (pi >= 0.85) and a fit uncorrelated with truth scores low (pi < 0.40).
    """
    rng = np.random.default_rng(0)
    truth = rng.normal(100.0, 10.0, 200)

    good = BaselineMetrics(
        df=pd.DataFrame({"observed": truth, "predicted": truth + rng.normal(0, 1, 200)}),
        num_model_params=2,
    )
    bad = BaselineMetrics(
        df=pd.DataFrame({"observed": truth, "predicted": rng.normal(100.0, 10.0, 200)}),
        num_model_params=2,
    )

    assert good.pi_rating == "excellent"
    assert bad.pi >= 0.0  # uncorrelated → pi collapses toward 0
    assert bad.pi_rating == "very bad"


def test_baseline_metrics_from_dict_roundtrip(realistic_baseline):
    """BaselineMetricsFromDict rebuilds an equivalent metrics object."""
    restored = BaselineMetricsFromDict(realistic_baseline.model_dump())

    assert restored.cvrmse == pytest.approx(realistic_baseline.cvrmse)
    assert restored.r_squared == pytest.approx(realistic_baseline.r_squared)


# ---------------------------------------------------------------------------
# ReportingMetrics — savings and ASHRAE uncertainty
# ---------------------------------------------------------------------------

def test_reporting_metrics_savings_is_predicted_minus_observed(realistic_baseline):
    """Savings equals the predicted-minus-observed sum over the reporting period."""
    reporting = realistic_baseline.df
    rm = ReportingMetrics(
        baseline_metrics=realistic_baseline, reporting_df=reporting, data_frequency="hourly"
    )

    expected = reporting["predicted"].sum() - reporting["observed"].sum()
    assert rm.savings == pytest.approx(expected)
    assert rm.n == len(reporting)


def test_reporting_metrics_no_load_change_zero_savings(realistic_baseline):
    """Observed equal to predicted in the reporting period gives ~0 savings."""
    reporting = realistic_baseline.df.copy()
    reporting["observed"] = reporting["predicted"]
    rm = ReportingMetrics(
        baseline_metrics=realistic_baseline, reporting_df=reporting, data_frequency="hourly"
    )

    assert rm.savings == pytest.approx(0.0, abs=1e-9)


def test_reporting_metrics_t_stat_matches_basic(realistic_baseline):
    """The reporting t-stat equals basic.t_stat at the configured confidence/dof."""
    rm = ReportingMetrics(
        baseline_metrics=realistic_baseline,
        reporting_df=realistic_baseline.df,
        data_frequency="hourly",
        confidence_level=0.90,
        t_tail=2,
    )

    assert rm.t_stat == pytest.approx(t_stat(1 - 0.90, realistic_baseline.ddof, tail=2))


def test_reporting_metrics_ashrae_frequency_factors(realistic_baseline):
    """The per-frequency uncertainty scales the shared base by the ASHRAE factors.

    Hourly uses the flat 1.26 correction; daily and billing apply the
    Sun & Baltazar polynomials evaluated at the number of reporting months
    (here a single January, so M=1).
    """
    df = realistic_baseline.df
    rm_h = ReportingMetrics(baseline_metrics=realistic_baseline, reporting_df=df, data_frequency="hourly")
    rm_d = ReportingMetrics(baseline_metrics=realistic_baseline, reporting_df=df, data_frequency="daily")
    rm_b = ReportingMetrics(baseline_metrics=realistic_baseline, reporting_df=df, data_frequency="billing")

    base = rm_h.total_savings_uncertainty / 1.26
    months = 1

    assert rm_d.total_savings_uncertainty == pytest.approx(
        base * np.polyval([-0.00024, 0.03535, 1.00286], months)
    )
    assert rm_b.total_savings_uncertainty == pytest.approx(
        base * np.polyval([-0.00022, 0.03306, 0.94054], months)
    )


def test_reporting_metrics_fsu_and_per_point_definitions(realistic_baseline):
    """FSU = uncertainty / savings and per-point unc = uncertainty / √n."""
    df = realistic_baseline.df
    rm = ReportingMetrics(baseline_metrics=realistic_baseline, reporting_df=df, data_frequency="hourly")

    assert rm.fsu == pytest.approx(rm.total_savings_uncertainty / rm.savings)
    assert rm.predicted_data_point_unc == pytest.approx(
        rm.total_savings_uncertainty / np.sqrt(rm.n)
    )


# ---------------------------------------------------------------------------
# compute_baseline_profile
# ---------------------------------------------------------------------------

_PROFILE_TZ = "America/Los_Angeles"
_PROFILE_ARRAYS = ["n", "typical_load", "residual_rms", "state_spread", "residual_acf"]
_PROFILE_SCALARS = ["residual_cv", "annual_scale", "yj_lambda"]


def _profile_frame(cadence, periods=None, seed=0):
    """Synthetic baseline with a daily shape, a weekend drop and noisy observations."""
    if cadence == "hourly":
        index = pd.date_range("2023-01-01", periods=periods or 8760, freq="h", tz=_PROFILE_TZ)
    elif cadence == "daily":
        index = pd.date_range("2023-01-01", periods=periods or 365, freq="D", tz=_PROFILE_TZ)
    elif cadence == "billing":
        starts = pd.date_range("2023-01-01", periods=13, freq="MS", tz=_PROFILE_TZ)
        index = pd.DatetimeIndex(starts[:-1] + (starts[1:] - starts[:-1]) / 2)
    else:
        raise ValueError(cadence)

    rng = np.random.default_rng(seed)
    predicted = (
        2.0
        + np.sin(2 * np.pi * np.asarray(index.hour) / 24)
        - 0.5 * (np.asarray(index.dayofweek) >= 5)
        + 0.5 * np.cos(2 * np.pi * np.asarray(index.month) / 12)
    )
    observed = predicted + rng.normal(0.0, 0.3, len(index))
    df = pd.DataFrame({"observed": observed, "predicted": predicted}, index=index)

    return df


@pytest.fixture(scope="module")
def profiles():
    cadences = ["hourly", "daily", "billing"]
    profiles = {c: compute_baseline_profile(_profile_frame(c), c) for c in cadences}

    return profiles


def _assert_profiles_equal(a, b, atol=1e-12):
    assert (a.cadence, a.scheme, a.settings) == (b.cadence, b.scheme, b.settings)

    for name in _PROFILE_ARRAYS + _PROFILE_SCALARS:
        close = np.allclose(getattr(a, name), getattr(b, name), rtol=0, atol=atol, equal_nan=True)
        assert close, name


def _assert_no_load_block(profile, n_cells, k_acf):
    assert profile.typical_load.tolist() == [0.0] * n_cells
    assert profile.residual_rms.tolist() == [0.0] * n_cells
    assert profile.state_spread.tolist() == [0.0] * n_cells
    assert profile.residual_acf.tolist() == [0.0] * k_acf
    assert np.isnan(profile.residual_cv)
    assert profile.annual_scale == 0.0
    assert profile.yj_lambda == 1.0


@pytest.mark.parametrize("cadence", ["hourly", "daily", "billing"])
def test_baseline_profile_dict_roundtrip(profiles, cadence):
    profile = profiles[cadence]

    from_python = BaselineProfileMetricsFromDict(profile.model_dump())
    from_json = BaselineProfileMetricsFromDict(json.loads(json.dumps(profile.model_dump())))

    _assert_profiles_equal(from_python, profile)
    _assert_profiles_equal(from_json, profile)


@pytest.mark.parametrize("cadence", ["hourly", "daily", "billing"])
def test_baseline_profile_is_rounded_to_eight_significant_digits(profiles, cadence):
    """Every array is rounded relative to its largest magnitude and every scalar to
    itself, so the block is its own JSON round trip byte for byte."""
    profile = profiles[cadence]

    for name in _PROFILE_ARRAYS:
        values = getattr(profile, name)
        if values.size == 0 or np.max(np.abs(values)) == 0:
            continue
        decimals = 7 - int(np.floor(np.log10(np.max(np.abs(values)))))
        assert np.array_equal(np.round(values, decimals), values), name

    for name in _PROFILE_SCALARS:
        value = getattr(profile, name)
        decimals = 7 - int(np.floor(np.log10(abs(value))))
        assert round(value, decimals) == value, name

    payload = json.dumps(profile.model_dump())
    restored = BaselineProfileMetricsFromDict(json.loads(payload))

    assert json.dumps(restored.model_dump()) == payload


def test_baseline_profile_no_load_roundtrip_serializes_nan_cv_as_null():
    profile = compute_baseline_profile(_profile_frame("daily").assign(predicted=0.0), "daily")

    payload = json.dumps(profile.model_dump())
    restored = BaselineProfileMetricsFromDict(json.loads(payload))

    assert json.loads(payload)["residual_cv"] is None
    assert profile.model_dump(mode="json")["residual_cv"] is None
    assert np.isnan(restored.residual_cv)
    _assert_profiles_equal(restored, profile)


@pytest.mark.parametrize(
    "cadence, scheme, n_cells, k_acf",
    [
        ("hourly", "hour_of_week_x_month", 2016, 168),
        ("daily", "day_of_week_x_month", 84, 7),
        ("billing", "month", 12, 0),
    ],
)
def test_baseline_profile_shapes_per_cadence(profiles, cadence, scheme, n_cells, k_acf):
    profile = profiles[cadence]

    assert profile.scheme == scheme
    assert profile.residual_acf.shape == (k_acf,)

    for name in ["n", "typical_load", "residual_rms", "state_spread"]:
        assert getattr(profile, name).shape == (n_cells,), name


def test_baseline_profile_counts_rows_per_cell(profiles):
    assert profiles["billing"].n.tolist() == [1] * 12
    assert profiles["daily"].n.sum() == 365
    assert profiles["hourly"].n.sum() == 8760


def test_baseline_profile_per_cell_arrays_positive_and_finite(profiles):
    for cadence, profile in profiles.items():
        for name in ["typical_load", "residual_rms", "state_spread"]:
            values = getattr(profile, name)
            assert np.all(np.isfinite(values) & (values > 0)), f"{cadence} {name}"


def test_baseline_profile_wide_bandwidth_reduces_to_global_statistics():
    """With bandwidths at 1e6 every cell carries the global mean squares."""
    df = _profile_frame("billing")
    settings = ProfileSettings(
        hour_bandwidth_h=1e6, day_of_week_bandwidth_d=1e6, calendar_bandwidth_d=1e6
    )
    residual = df["predicted"] - df["observed"]
    rms_predicted = np.sqrt(np.mean(df["predicted"] ** 2))
    rms_residual = np.sqrt(np.mean(residual**2))

    profile = compute_baseline_profile(df, "billing", settings)

    assert np.allclose(profile.typical_load, rms_predicted, rtol=1e-6, atol=0)
    assert np.allclose(profile.residual_rms, rms_residual, rtol=1e-6, atol=0)
    # scalars carry eight significant digits
    assert profile.annual_scale == pytest.approx(rms_predicted, rel=1e-7)
    assert profile.residual_cv == pytest.approx(
        rms_residual / np.mean(np.abs(df["predicted"])), rel=1e-7
    )


def test_baseline_profile_state_spread_positive_when_predictions_vary(profiles):
    assert np.all(profiles["hourly"].state_spread > 0)


def test_baseline_profile_state_spread_zero_for_constant_predictions():
    df = _profile_frame("hourly", periods=24 * 60).assign(predicted=3.0)

    profile = compute_baseline_profile(df, "hourly")

    assert np.allclose(profile.state_spread, 0.0, rtol=0, atol=1e-7)


def test_baseline_profile_drops_non_finite_rows():
    df = _profile_frame("daily")
    df_gaps = df.copy()
    df_gaps.iloc[[3, 50, 51], 0] = np.nan
    df_gaps.iloc[[100], 1] = np.inf

    with_gaps = compute_baseline_profile(df_gaps, "daily")
    without = compute_baseline_profile(df.drop(df.index[[3, 50, 51, 100]]), "daily")

    _assert_profiles_equal(with_gaps, without, atol=0)


def test_baseline_profile_short_series_gives_zero_acf():
    profile = compute_baseline_profile(_profile_frame("hourly", periods=169), "hourly")

    assert profile.residual_acf.tolist() == [0.0] * 168


def test_baseline_profile_yj_lambda_below_one_for_right_skewed_residual():
    x = np.random.default_rng(0).lognormal(0, 1, 10000)
    index = pd.date_range("2023-01-01", periods=len(x), freq="h", tz=_PROFILE_TZ)
    df = pd.DataFrame({"observed": 1.0 - x, "predicted": 1.0}, index=index)

    profile = compute_baseline_profile(df, "hourly")

    assert np.isfinite(profile.yj_lambda)
    assert profile.yj_lambda < 1, f"yj_lambda {profile.yj_lambda}"


def test_baseline_profile_constant_residual_gives_identity_lambda():
    df = _profile_frame("hourly", periods=24 * 30).assign(predicted=1.0, observed=0.5)

    profile = compute_baseline_profile(df, "hourly")

    assert profile.yj_lambda == 1.0


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda df: df.assign(predicted=0.0), id="zero_predictions"),
        pytest.param(lambda df: df.assign(observed=np.nan), id="no_finite_rows"),
        pytest.param(lambda df: df.iloc[:0], id="empty_frame"),
    ],
)
@pytest.mark.parametrize(
    "cadence, n_cells, k_acf", [("hourly", 2016, 168), ("daily", 84, 7), ("billing", 12, 0)]
)
def test_baseline_profile_no_load_block(transform, cadence, n_cells, k_acf):
    df = transform(_profile_frame(cadence, periods=24 * 14))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = compute_baseline_profile(df, cadence)

    _assert_no_load_block(profile, n_cells, k_acf)


def test_baseline_profile_stores_settings():
    settings = ProfileSettings(hour_bandwidth_h=3.0)

    profile = compute_baseline_profile(_profile_frame("daily"), "daily", settings)

    assert profile.settings == settings


@pytest.mark.parametrize("value", [0.001, 2e6])
def test_profile_settings_bandwidth_bounds(value):
    with pytest.raises(pydantic.ValidationError):
        ProfileSettings(calendar_bandwidth_d=value)


def test_baseline_profile_unknown_cadence_raises():
    with pytest.raises(ValueError, match="Unknown cadence"):
        compute_baseline_profile(_profile_frame("daily"), "monthly")
