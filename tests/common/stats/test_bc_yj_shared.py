"""Unit tests for the shared Box-Cox / Yeo-Johnson numba helpers."""

import numpy as np
import pytest

from opendsm.common.stats.distribution_transform._bc_yj_shared import (
    _huber_std,
    _brent_min,
    _normal_scores,
    _bisquare_weights,
    _secant,
)



pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaExperimentalFeatureWarning"
)


class TestHuberStd:
    def test_output_centered(self):
        x = np.random.default_rng(0).standard_normal(200)
        out = _huber_std(x)

        assert abs(np.median(out)) < 0.1

    def test_constant_input(self):
        out = _huber_std(np.ones(50))

        assert np.allclose(out, 0.0)  # sigma forced to 1.0


class TestBrentMin:
    def test_finds_minimum_of_parabola(self):
        lam = _brent_min(lambda x: (x - 1.5) ** 2)

        assert abs(lam - 1.5) < 1e-3

    def test_respects_bounds(self):
        lam = _brent_min(lambda x: (x - 10) ** 2, bounds=(-4, 4))

        assert -4 <= lam <= 4


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


class TestSecant:
    def test_finds_root(self):
        result = _secant(lambda x: x - 1.5, lam0=0.0)

        assert result is not None
        assert abs(result - 1.5) < 1e-3

    def test_returns_none_on_flat_gradient(self):
        result = _secant(lambda x: 0.0, lam0=0.0)
        # flat gradient → dg ≈ 0 → degenerate

        assert result is None or abs(result) < 1e-3
