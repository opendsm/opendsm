"""Tests for shared helpers: mu_sigma, _bc_yj_shared, _base, outliers_transformed."""

import numpy as np
import pytest

from opendsm.common.stats.distribution_transform.mu_sigma import robust_mu_sigma
from opendsm.common.stats.distribution_transform._bc_yj_shared import (
    _huber_std,
    _brent_min,
    _normal_scores,
    _bisquare_weights,
    _secant,
    _fit_lambda,
)
from opendsm.common.stats.distribution_transform._base import TransformBase
from opendsm.common.stats.distribution_transform import (
    Standardize, Bisymlog, YeoJohnson, BoxCox,
)
from opendsm.common.stats.outliers_transformed import remove_outliers

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaExperimentalFeatureWarning"
)


# ── robust_mu_sigma ─────────────────────────────────────────────────────

class TestRobustMuSigma:
    def test_iqr_symmetric(self):
        x = np.arange(1.0, 11.0)
        mu, sigma = robust_mu_sigma(x, "iqr")
        assert np.isclose(float(np.asarray(mu).flat[0]), 5.5, rtol=1e-8)
        assert np.isclose(float(np.asarray(sigma).flat[0]), 3.7065055, rtol=1e-5)

    def test_iqr_robust_to_outlier(self):
        mu_c, _ = robust_mu_sigma(np.array([1., 2., 3., 4., 5.]), "iqr")
        mu_o, _ = robust_mu_sigma(np.array([1., 2., 3., 4., 5., 100.]), "iqr")
        assert abs(float(np.asarray(mu_o).flat[0]) - float(np.asarray(mu_c).flat[0])) < 1.0

    @pytest.mark.parametrize("rtype", ["huber_m_estimate", "iqr"])
    def test_small_n_falls_back_to_iqr(self, rtype):
        x = np.array([1., 3., 7.])
        mu_req, sig_req = robust_mu_sigma(x, rtype)
        mu_iqr, sig_iqr = robust_mu_sigma(x, "iqr")
        assert np.isclose(float(np.asarray(mu_req).flat[0]), float(np.asarray(mu_iqr).flat[0]))

    def test_huber_pinned(self):
        x = np.random.default_rng(42).standard_normal(200)
        mu, sigma = robust_mu_sigma(x, "huber_m_estimate")
        assert np.isclose(float(np.asarray(mu).flat[0]), -0.050897856, rtol=1e-5)
        assert np.isclose(float(np.asarray(sigma).flat[0]), 0.886829223, rtol=1e-5)

    def test_returns_two_values(self):
        assert len(robust_mu_sigma(np.arange(1., 21.), "iqr")) == 2


# ── _huber_std ───────────────────────────────────────────────────────────

class TestHuberStd:
    def test_output_centered(self):
        x = np.random.default_rng(0).standard_normal(200)
        out = _huber_std(x)
        assert abs(np.median(out)) < 0.1

    def test_constant_input(self):
        out = _huber_std(np.ones(50))
        assert np.allclose(out, 0.0)  # sigma forced to 1.0


# ── _brent_min ───────────────────────────────────────────────────────────

class TestBrentMin:
    def test_finds_minimum_of_parabola(self):
        lam = _brent_min(lambda x: (x - 1.5) ** 2)
        assert abs(lam - 1.5) < 1e-3

    def test_respects_bounds(self):
        lam = _brent_min(lambda x: (x - 10) ** 2, bounds=(-4, 4))
        assert -4 <= lam <= 4


# ── _normal_scores ───────────────────────────────────────────────────────

class TestNormalScores:
    def test_shape(self):
        assert _normal_scores(5).shape == (5,)

    def test_symmetric(self):
        s = _normal_scores(6)
        assert np.isclose(s[0], -s[-1], atol=1e-10)

    def test_cached(self):
        a = _normal_scores(10)
        b = _normal_scores(10)
        assert a is b  # same object from cache


# ── _bisquare_weights ────────────────────────────────────────────────────

class TestBisquareWeights:
    def test_all_inliers(self):
        vals = np.array([0.0, 0.1, -0.1])
        w, W = _bisquare_weights(vals, 0.0, 1.0, 3.0)
        assert np.all(w > 0)
        assert W > 0

    def test_all_outliers(self):
        vals = np.array([100.0, -100.0])
        w, W = _bisquare_weights(vals, 0.0, 1.0, 1.0)
        assert np.all(w == 0)
        assert W == 0.0

    def test_weight_sum(self):
        vals = np.array([0.0, 0.5, 1.0, 5.0])
        w, W = _bisquare_weights(vals, 0.0, 1.0, 2.0)
        assert np.isclose(W, w.sum())


# ── _secant ──────────────────────────────────────────────────────────────

class TestSecant:
    def test_finds_root(self):
        result = _secant(lambda x: x - 1.5, lam0=0.0)
        assert result is not None
        assert abs(result - 1.5) < 1e-3

    def test_returns_none_on_flat_gradient(self):
        result = _secant(lambda x: 0.0, lam0=0.0)
        # flat gradient → dg ≈ 0 → degenerate
        assert result is None or abs(result) < 1e-3


# ── TransformBase ────────────────────────────────────────────────────────

class TestTransformBase:
    def test_to_2d_1d(self):
        X, is_1d = TransformBase._to_2d(np.array([1., 2., 3.]))
        assert X.shape == (3, 1)
        assert is_1d is True

    def test_to_2d_2d(self):
        X, is_1d = TransformBase._to_2d(np.ones((3, 4)))
        assert X.shape == (3, 4)
        assert is_1d is False

    def test_to_2d_3d_raises(self):
        with pytest.raises(ValueError, match="Expected 1-D or 2-D"):
            TransformBase._to_2d(np.zeros((2, 3, 4)))


# ── __init__ public exports ──────────────────────────────────────────────

@pytest.mark.parametrize("cls", [Standardize, Bisymlog, YeoJohnson, BoxCox])
def test_public_exports_are_callable(cls):
    assert callable(cls)


# ── outliers_transformed ─────────────────────────────────────────────────

class TestRemoveOutliers:
    @pytest.fixture
    def clean_data(self):
        return np.random.default_rng(0).standard_normal(200)

    @pytest.fixture
    def data_with_outlier(self):
        x = np.random.default_rng(0).standard_normal(200)
        x = np.append(x, 100.0)
        return x

    def test_none_transform_passthrough(self, clean_data):
        x_out, idx = remove_outliers(clean_data, transform=None)
        assert len(idx) > 0

    @pytest.mark.parametrize("transform", [
        "standardize", "bisymlog",
        "yeo_johnson", "robust_yeo_johnson",
    ])
    def test_valid_transforms_run(self, clean_data, transform):
        x_out, idx = remove_outliers(clean_data, transform=transform)
        assert len(x_out) <= len(clean_data)

    @pytest.mark.parametrize("transform", ["box_cox", "robust_box_cox"])
    def test_box_cox_transforms_positive_data(self, transform):
        x = np.random.default_rng(0).lognormal(0, 1, 200)
        x_out, idx = remove_outliers(x, transform=transform)
        assert len(x_out) <= len(x)

    def test_invalid_transform_raises(self, clean_data):
        with pytest.raises(ValueError, match="Unknown transform"):
            remove_outliers(clean_data, transform="nonexistent")

    def test_detects_extreme_outlier(self, data_with_outlier):
        x_out, idx = remove_outliers(data_with_outlier, transform="standardize")
        assert len(x_out) < len(data_with_outlier)
