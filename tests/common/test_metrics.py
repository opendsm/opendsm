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
import statsmodels.api as sm

from opendsm.common.metrics import (
    BaselineMetrics,
    BaselineMetricsFromDict,
    ColumnMetrics,
    ReportingMetrics,
    acf,
)



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


def test_baseline_metrics_autocorrelation_adjusted_terms_finite(realistic_baseline):
    """n_prime, the adjusted dofs, and every autocorr/adjusted RMSE are finite."""
    m = realistic_baseline

    assert m.n_prime >= 1
    assert m.ddof >= 1
    assert m.ddof_autocorr >= 1
    for value in [
        m.rmse_autocorr, m.rmse_adj, m.rmse_autocorr_adj,
        m.cvrmse_autocorr, m.cvrmse_adj, m.cvrmse_autocorr_adj,
        m.pnrmse, m.pnrmse_autocorr, m.pnrmse_adj, m.pnrmse_autocorr_adj,
        m.pnmae, m.pnmbe, m.r_squared_adj, m.index_of_agreement,
    ]:
        assert np.isfinite(value)


def test_baseline_metrics_pi_rating_is_valid_label(realistic_baseline):
    """pi_rating returns one of the defined qualitative buckets."""
    assert realistic_baseline.pi_rating in {
        "excellent", "very good", "good", "satisfactory", "poor", "bad", "very bad"
    }


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


@pytest.mark.parametrize("frequency", ["hourly", "daily", "billing"])
def test_reporting_metrics_uncertainty_finite_per_frequency(realistic_baseline, frequency):
    """Total savings uncertainty, FSU and per-point uncertainty are finite."""
    rm = ReportingMetrics(
        baseline_metrics=realistic_baseline,
        reporting_df=realistic_baseline.df,
        data_frequency=frequency,
    )

    assert rm.t_stat > 0
    assert np.isfinite(rm.total_savings_uncertainty)
    assert rm.total_savings_uncertainty > 0
    assert np.isfinite(rm.fsu)
    assert np.isfinite(rm.predicted_data_point_unc)
