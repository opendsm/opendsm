"""Tests for PSpline — the prediction and serialization object."""

import json

import numpy as np
import pytest

from opendsm.eemeter.models.daily_pspline.spline import PSpline


@pytest.fixture
def fitted_spline():
    """Build a simple PSpline from known parameters."""
    knots = np.array([0, 0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1], dtype=float)
    coefs = np.array([1.0, 0.8, 0.5, 0.3, 0.3, 0.5, 0.9], dtype=float)
    return PSpline(
        knots_std=knots, coefs_std=coefs, degree=3,
        x_mean=50.0, x_std=15.0, y_mean=30.0, y_std=10.0,
        bp=np.array([45.0, 60.0]),
        fit_bnds=np.array([20.0, 90.0]),
        bc_type="natural",
        config={"n_min": 5, "lambda_smoothing": 0.0, "kappa_penalty": 1e6, "maxiter": 30},
    )


class TestPSplinePredict:
    def test_predict_returns_array(self, fitted_spline):
        x = np.array([30.0, 50.0, 70.0])
        result = fitted_spline.predict(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_call_and_predict_are_identical(self, fitted_spline):
        x = np.array([30.0, 50.0, 70.0])
        np.testing.assert_array_equal(
            fitted_spline(x), fitted_spline.predict(x),
            err_msg="__call__ and predict should be identical",
        )

    def test_scalar_input(self, fitted_spline):
        result = fitted_spline.predict(np.array([50.0]))
        assert np.isfinite(result[0])


class TestExtrapolation:
    def test_natural_bc_extrapolates_linearly(self, fitted_spline):
        lo = fitted_spline.fit_bnds[0]
        hi = fitted_spline.fit_bnds[1]
        x_below = np.array([lo - 20, lo - 10, lo])
        y_below = fitted_spline(x_below)
        # Linear extrapolation → constant spacing should give constant differences
        diffs = np.diff(y_below)
        np.testing.assert_allclose(
            diffs[0], diffs[1], rtol=1e-6,
            err_msg="Natural BC should extrapolate linearly below fit bounds",
        )

    def test_clamped_bc_extrapolates_flat(self):
        knots = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=float)
        coefs = np.array([1.0, 0.8, 0.5, 0.3, 0.6], dtype=float)
        spl = PSpline(
            knots_std=knots, coefs_std=coefs, degree=3,
            x_mean=50.0, x_std=15.0, y_mean=30.0, y_std=10.0,
            bp=np.array([45.0, 60.0]),
            fit_bnds=np.array([20.0, 90.0]),
            bc_type="clamped",
            config={"n_min": 5, "lambda_smoothing": 0.0, "kappa_penalty": 1e6, "maxiter": 30},
        )
        y_far_below = spl(np.array([0.0]))
        y_at_lo = spl(np.array([20.0]))
        np.testing.assert_allclose(
            y_far_below, y_at_lo, rtol=1e-10,
            err_msg="Clamped BC should give constant extrapolation",
        )


class TestSerialization:
    def test_round_trip_preserves_predictions(self, fitted_spline):
        x = np.linspace(20, 90, 50)
        pred_before = fitted_spline.predict(x)
        d = fitted_spline.to_dict()
        restored = PSpline.from_dict(d)
        pred_after = restored.predict(x)
        np.testing.assert_allclose(
            pred_before, pred_after, atol=1e-12,
            err_msg="Serialization round-trip should preserve predictions exactly",
        )

    def test_json_round_trip(self, fitted_spline):
        j = fitted_spline.to_json()
        restored = PSpline.from_json(j)
        x = np.array([40.0, 60.0])
        np.testing.assert_allclose(
            fitted_spline.predict(x), restored.predict(x), atol=1e-12,
        )

    def test_dict_contains_expected_keys(self, fitted_spline):
        d = fitted_spline.to_dict()
        for key in ("knots", "coefficients", "degree", "breakpoints",
                     "fit_bounds", "x_mean", "x_std", "y_mean", "y_std",
                     "n_min", "lambda_smoothing", "kappa_penalty", "maxiter"):
            assert key in d, f"Missing key '{key}' in serialized dict"


class TestDerivative:
    def test_derivative_returns_fitted_pspline(self, fitted_spline):
        d = fitted_spline.derivative(nu=1)
        assert isinstance(d, PSpline)

    def test_derivative_degree_decreases(self, fitted_spline):
        d = fitted_spline.derivative(nu=1)
        assert d.k == fitted_spline.k - 1


class TestIntegrate:
    def test_integrate_positive_interval(self, fitted_spline):
        result = fitted_spline.integrate(30.0, 70.0)
        assert np.isfinite(result)

    def test_integrate_zero_width(self, fitted_spline):
        result = fitted_spline.integrate(50.0, 50.0)
        assert abs(result) < 1e-10, f"Zero-width integral should be ~0, got {result}"
