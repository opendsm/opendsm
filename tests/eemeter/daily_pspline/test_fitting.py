"""Tests for fit_segment — the main fitting entry point."""

import numpy as np
import pytest

from opendsm.eemeter.models.daily_pspline.fitting import fit_segment, _clipped_std, _rescale_to_range
from opendsm.eemeter.models.daily_pspline.spline import PSpline


class TestFitSegment:
    def test_returns_fitted_pspline(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        result = fit_segment(T, y, dev_settings)
        assert isinstance(result, PSpline)

    def test_prediction_shape_matches_input(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        result = fit_segment(T, y, dev_settings)
        pred = result.predict(T)
        assert pred.shape == T.shape, (
            f"Expected prediction shape {T.shape}, got {pred.shape}"
        )

    def test_rmse_below_threshold_v_shape(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        result = fit_segment(T, y, dev_settings)
        pred = result.predict(T)
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        assert rmse < 3.0, f"RMSE={rmse:.3f} too high for V-shaped data with σ=1 noise"

    def test_heating_only_data(self, heating_only_data, dev_settings):
        T, y = heating_only_data
        result = fit_segment(T, y, dev_settings)
        pred = result.predict(T)
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        assert rmse < 3.0, f"RMSE={rmse:.3f} too high for heating-only data"

    def test_flat_data(self, flat_data, dev_settings):
        T, y = flat_data
        result = fit_segment(T, y, dev_settings)
        pred = result.predict(T)
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        assert rmse < 1.0, f"RMSE={rmse:.3f} too high for flat data with σ=0.3 noise"

    def test_breakpoints_within_data_range(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        result = fit_segment(T, y, dev_settings)
        assert result.bp[0] >= T[0] - 1e-6, (
            f"bp[0]={result.bp[0]:.2f} below data min {T[0]:.2f}"
        )
        assert result.bp[1] <= T[-1] + 1e-6, (
            f"bp[1]={result.bp[1]:.2f} above data max {T[-1]:.2f}"
        )

    def test_fit_bounds_match_data_range(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        result = fit_segment(T, y, dev_settings)
        np.testing.assert_allclose(
            result.fit_bnds, [T[0], T[-1]], atol=1e-10,
            err_msg="fit_bnds should match data range",
        )

    def test_training_metrics_populated(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        result = fit_segment(T, y, dev_settings)
        assert result.training_metrics is not None
        assert hasattr(result.training_metrics, "rmse")

    def test_serialization_after_fit(self, v_shaped_data, dev_settings):
        T, y = v_shaped_data
        result = fit_segment(T, y, dev_settings)
        d = result.to_dict()
        restored = PSpline.from_dict(d)
        np.testing.assert_allclose(
            result.predict(T), restored.predict(T), atol=1e-10,
            err_msg="Serialization after fit should preserve predictions",
        )


class TestClippedStd:
    def test_normal_values(self):
        assert _clipped_std(np.array([1, 2, 3, 4, 5])) == pytest.approx(np.std([1, 2, 3, 4, 5]))

    def test_constant_returns_one(self):
        assert _clipped_std(np.array([5.0, 5.0, 5.0])) == 1.0

    def test_near_zero_returns_one(self):
        assert _clipped_std(np.array([1.0, 1.0 + 1e-8])) == 1.0


class TestRescaleToRange:
    def test_basic_rescale(self):
        values = np.array([0.0, 5.0, 10.0])
        result = _rescale_to_range(values, 1.0, 10.0)
        np.testing.assert_allclose(result, [1.0, 5.5, 10.0])

    def test_constant_input(self):
        values = np.array([3.0, 3.0, 3.0])
        result = _rescale_to_range(values, 1.0, 10.0)
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0])
