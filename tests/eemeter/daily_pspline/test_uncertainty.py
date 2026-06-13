"""Tests for prediction uncertainty infrastructure."""

import numpy as np
import pytest

from opendsm.eemeter.models.daily_pspline.fitting import fit_segment
from opendsm.eemeter.models.daily_pspline.spline import PSpline
from opendsm.eemeter.models.daily_pspline.settings import DailyPSplineSettings
from opendsm.eemeter.models.daily_pspline.uncertainty import (
    _bartlett_vif,
    _compute_acf_K,
    _fit_sigma_scale,
)



class TestPredictionUncertainty:
    def test_bounds_bracket_prediction(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        spl = fit_segment(T, y, dev_settings)
        lower, upper = spl.prediction_uncertainty(T)
        pred = spl.predict(T)

        assert lower.shape == upper.shape == T.shape
        assert np.all(upper > lower), "Interval must have positive width"
        assert np.all(lower <= pred) and np.all(pred <= upper)

    def test_shape_matches_input(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        spl = fit_segment(T, y, dev_settings)
        T_new = np.array([30.0, 50.0, 70.0])
        lower, upper = spl.prediction_uncertainty(T_new)

        assert lower.shape == (3,) and upper.shape == (3,)

    def test_wider_at_extrapolation(self, rng, dev_settings):
        """Interval width should grow outside the training range."""
        T = np.sort(rng.uniform(30, 70, 200))
        y = 30 + 0.5 * np.maximum(50 - T, 0) + rng.normal(0, 0.1, 200)
        spl = fit_segment(T, y, dev_settings)

        lo_near, hi_near = spl.prediction_uncertainty(np.array([28.0]))
        lo_far, hi_far = spl.prediction_uncertainty(np.array([10.0]))
        width_near = (hi_near - lo_near)[0]
        width_far = (hi_far - lo_far)[0]

        assert width_far > width_near, (
            f"Far extrapolation width ({width_far:.3f}) should exceed "
            f"near ({width_near:.3f})"
        )

    def test_lower_alpha_gives_wider_interval(self, v_shaped_data):
        """Lower significance → wider PI (more confidence)."""
        T, y = v_shaped_data
        s90 = DailyPSplineSettings(
            developer_mode=True, silent_developer_mode=True, uncertainty_alpha=0.1,
        )
        s99 = DailyPSplineSettings(
            developer_mode=True, silent_developer_mode=True, uncertainty_alpha=0.01,
        )
        lo90, hi90 = fit_segment(T, y, s90).prediction_uncertainty(T)
        lo99, hi99 = fit_segment(T, y, s99).prediction_uncertainty(T)

        assert np.all((hi99 - lo99) > (hi90 - lo90)), (
            "99% PI should be wider than 90% PI at every point"
        )

    def test_vif_disabled_does_not_widen_interval(self, v_shaped_data):
        """Disabling autocorrelation should not widen the interval."""
        T, y = v_shaped_data
        s_with = DailyPSplineSettings(
            developer_mode=True, silent_developer_mode=True,
            include_autocorrelation_in_uncertainty=True,
        )
        s_without = DailyPSplineSettings(
            developer_mode=True, silent_developer_mode=True,
            include_autocorrelation_in_uncertainty=False,
        )
        lo_w, hi_w = fit_segment(T, y, s_with).prediction_uncertainty(T)
        lo_wo, hi_wo = fit_segment(T, y, s_without).prediction_uncertainty(T)

        assert np.all((hi_w - lo_w) >= (hi_wo - lo_wo) - 1e-9), (
            "VIF-enabled interval should be >= VIF-disabled"
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


class TestFitSigmaScale:
    def test_shape_matches_input(self, rng):
        x = np.linspace(0, 100, 200)
        scale = _fit_sigma_scale(x, rng.standard_normal(200))

        assert scale.shape == x.shape

    def test_clipped_to_bounds(self, rng):
        x = np.linspace(0, 100, 200)
        # extreme heteroscedasticity would push the raw ratio past [0.5, 2.0]
        residuals = rng.standard_normal(200) * (0.01 + 0.5 * x)
        scale = _fit_sigma_scale(x, residuals)

        assert np.all(scale >= 0.5) and np.all(scale <= 2.0)

    def test_homoscedastic_near_one(self, rng):
        x = np.linspace(0, 100, 300)
        scale = _fit_sigma_scale(x, rng.standard_normal(300))

        assert scale.mean() == pytest.approx(1.0, abs=0.15)
        assert scale.std() < 0.2

    def test_recovers_heteroscedastic_trend(self, rng):
        """Scale must be larger where residuals are noisier."""
        x = np.linspace(0, 100, 300)
        residuals = rng.standard_normal(300) * (0.3 + 0.03 * x)
        scale = _fit_sigma_scale(x, residuals)

        assert scale[x > 75].mean() > scale[x < 25].mean()

    def test_near_zero_residuals_returns_ones(self):
        x = np.linspace(0, 100, 50)
        scale = _fit_sigma_scale(x, np.full(50, 1e-15))

        np.testing.assert_array_equal(scale, np.ones(50))


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
