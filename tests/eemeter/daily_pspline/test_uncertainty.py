"""Tests for prediction uncertainty infrastructure."""

import numpy as np
import pytest

from opendsm.eemeter.models.daily_pspline.fitting import fit_segment
from opendsm.eemeter.models.daily_pspline.spline import PSpline
from opendsm.eemeter.models.daily_pspline.uncertainty import (
    _bartlett_vif,
    _compute_acf_K,
    _interpolate_sigma_sq,
)


class TestPredictionUncertainty:
    def test_returns_positive_array(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        spl = fit_segment(T, y, dev_settings)
        unc = spl.prediction_uncertainty(T)
        assert unc.shape == T.shape
        assert np.all(unc > 0), "Uncertainty must be strictly positive"

    def test_shape_matches_input(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        spl = fit_segment(T, y, dev_settings)
        T_new = np.array([30.0, 50.0, 70.0])
        unc = spl.prediction_uncertainty(T_new)
        assert unc.shape == (3,)

    def test_wider_at_extrapolation_model_term(self, rng, dev_settings):
        """Model uncertainty (leverage) should grow outside training range."""
        T = np.sort(rng.uniform(30, 70, 200))
        y = 30 + 0.5 * np.maximum(50 - T, 0) + rng.normal(0, 0.1, 200)
        spl = fit_segment(T, y, dev_settings)

        # Compare model variance only (remove noise term by checking
        # that far-extrapolation > near-extrapolation)
        T_near = np.array([28.0])
        T_far = np.array([10.0])
        unc_near = spl.prediction_uncertainty(T_near)
        unc_far = spl.prediction_uncertainty(T_far)
        assert unc_far[0] > unc_near[0], (
            f"Far extrapolation ({unc_far[0]:.3f}) should have wider uncertainty "
            f"than near extrapolation ({unc_near[0]:.3f})"
        )

    def test_lower_alpha_gives_wider_interval(self, v_shaped_data):
        """Lower significance → wider PI (more confidence)."""
        from opendsm.eemeter.models.daily_pspline.settings import DailyPSplineSettings
        T, y = v_shaped_data

        s90 = DailyPSplineSettings(
            developer_mode=True, silent_developer_mode=True,
            uncertainty_alpha=0.1,
        )
        s99 = DailyPSplineSettings(
            developer_mode=True, silent_developer_mode=True,
            uncertainty_alpha=0.01,
        )
        spl90 = fit_segment(T, y, s90)
        spl99 = fit_segment(T, y, s99)

        unc90 = spl90.prediction_uncertainty(T)
        unc99 = spl99.prediction_uncertainty(T)
        assert np.all(unc99 > unc90), (
            "99% PI should be wider than 90% PI at every point"
        )

    def test_vif_disabled_gives_narrower_interval(self, v_shaped_data):
        """Disabling autocorrelation should reduce uncertainty."""
        from opendsm.eemeter.models.daily_pspline.settings import DailyPSplineSettings
        T, y = v_shaped_data

        s_with = DailyPSplineSettings(
            developer_mode=True, silent_developer_mode=True,
            include_autocorrelation_in_uncertainty=True,
        )
        s_without = DailyPSplineSettings(
            developer_mode=True, silent_developer_mode=True,
            include_autocorrelation_in_uncertainty=False,
        )
        spl_with = fit_segment(T, y, s_with)
        spl_without = fit_segment(T, y, s_without)

        unc_with = spl_with.prediction_uncertainty(T)
        unc_without = spl_without.prediction_uncertainty(T)
        # When VIF=1 (no autocorrelation), both should be equal
        # When VIF>1, _with should be >= _without
        assert np.all(unc_with >= unc_without - 1e-10), (
            "VIF-enabled uncertainty should be >= VIF-disabled"
        )

    def test_raises_without_fit(self):
        """prediction_uncertainty requires uncertainty state from fit_segment."""
        spl = PSpline(
            knots_std=np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float),
            coefs_std=np.array([1.0, 0.8, 0.5, 0.3, 0.6], dtype=float),
            degree=3,
            x_mean=50.0, x_std=15.0, y_mean=30.0, y_std=10.0,
            bp=np.array([45.0, 60.0]),
            fit_bnds=np.array([20.0, 90.0]),
            bc_type="natural",
            config={"n_min": 5, "lambda_smoothing": 0.0, "kappa_penalty": 1e6, "maxiter": 30},
        )
        with pytest.raises(RuntimeError, match="UncertaintyEstimator"):
            spl.prediction_uncertainty(np.array([50.0]))


class TestInterpolateSigmaSq:
    def test_interior_matches_training(self):
        x_train = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
        sigma = np.array([2.0, 1.5, 1.0, 1.5, 2.0])
        x_new = np.array([30.0, 50.0, 70.0])
        result = _interpolate_sigma_sq(x_new, x_train, sigma)
        np.testing.assert_allclose(
            result, sigma[1:4] ** 2, rtol=1e-10,
            err_msg="Interior points should match training σ² exactly",
        )

    def test_extrapolation_grows(self):
        x_train = np.linspace(20, 80, 50)
        sigma = 1.0 + 0.02 * np.abs(x_train - 50)
        x_extrap = np.array([10.0, 90.0])
        result = _interpolate_sigma_sq(x_extrap, x_train, sigma)
        boundary_sq = np.array([sigma[0] ** 2, sigma[-1] ** 2])
        assert np.all(result >= boundary_sq - 1e-10), (
            "Extrapolated σ² should be >= boundary σ²"
        )


class TestBartlettVif:
    def test_iid_residuals_vif_near_one(self, rng):
        residuals = rng.standard_normal(200)
        vif = _bartlett_vif(residuals, K=5)
        assert 0.8 < vif < 1.3, f"IID residuals should give VIF ≈ 1, got {vif:.3f}"

    def test_correlated_residuals_vif_above_one(self):
        rng = np.random.default_rng(42)
        n = 200
        residuals = np.zeros(n)
        residuals[0] = rng.standard_normal()
        for i in range(1, n):
            residuals[i] = 0.7 * residuals[i - 1] + rng.standard_normal()
        vif = _bartlett_vif(residuals, K=10)
        assert vif > 2.0, f"AR(0.7) residuals should give VIF > 2, got {vif:.3f}"

    def test_vif_floored_at_one(self, rng):
        """VIF should never be less than 1 (negative autocorrelation clamped)."""
        # Alternating residuals: strong negative autocorrelation
        residuals = np.array([1, -1] * 50, dtype=float)
        vif = _bartlett_vif(residuals, K=5)
        assert vif >= 1.0, f"VIF should be >= 1, got {vif:.3f}"


class TestComputeAcfK:
    def test_iid_returns_low_K(self, rng):
        residuals = rng.standard_normal(200)
        K = _compute_acf_K(residuals)
        assert K <= 3, f"IID residuals should give K <= 3, got {K}"

    def test_correlated_returns_higher_K(self):
        rng = np.random.default_rng(42)
        n = 300
        residuals = np.zeros(n)
        residuals[0] = rng.standard_normal()
        for i in range(1, n):
            residuals[i] = 0.8 * residuals[i - 1] + rng.standard_normal()
        K = _compute_acf_K(residuals)
        assert K >= 3, f"AR(0.8) residuals should give K >= 3, got {K}"

    def test_short_data_returns_zero(self):
        K = _compute_acf_K(np.array([1.0, 2.0, 3.0]))
        assert K == 0
