"""Tests for Bisymlog class and numba kernels."""

import numpy as np
import pytest

from opendsm.common.stats.distribution_transform.bisymlog import (
    Bisymlog,
    _bisymlog_forward,
    _bisymlog_inverse,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaExperimentalFeatureWarning"
)

LOG10_INV = 1.0 / np.log10(10)  # = 1.0 for base 10


# ── Numba kernels ────────────────────────────────────────────────────────

class TestKernels:
    @pytest.mark.parametrize("x_val, expected", [
        (0.0, 0.0),
        (9.0, 1.0),               # log10(9/1 + 1) = log10(10) = 1
        (-9.0, -1.0),
        (1.0, np.log10(2)),
        (-1.0, -np.log10(2)),
    ])
    def test_forward_known_c1(self, x_val, expected):
        out = _bisymlog_forward(np.array([x_val]), 1.0, LOG10_INV)
        assert np.isclose(out[0], expected)

    def test_forward_zero_always_zero(self):
        for C in [0.1, 1.0, 100.0]:
            assert _bisymlog_forward(np.array([0.0]), C, LOG10_INV)[0] == 0.0

    @pytest.mark.parametrize("C", [0.1, 1.0, 10.0, 100.0])
    def test_roundtrip(self, C):
        x = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        y = _bisymlog_forward(x, C, LOG10_INV)
        x_back = _bisymlog_inverse(y, C, 10.0)
        assert np.allclose(x_back, x, rtol=1e-10)

    def test_inverse_known(self):
        # y=1, C=1, base=10 → 1 * (10^1 - 1) = 9
        out = _bisymlog_inverse(np.array([1.0]), 1.0, 10.0)
        assert np.isclose(out[0], 9.0)

    def test_base_2(self):
        log2_inv = 1.0 / np.log10(2)
        out = _bisymlog_forward(np.array([1.0]), 1.0, log2_inv)
        # sign(1) * log10(2) / log10(2) = 1.0
        assert np.isclose(out[0], 1.0)


# ── Heuristic C ──────────────────────────────────────────────────────────

class TestHeuristicC:
    def test_constant_returns_none(self):
        assert Bisymlog._heuristic_C(np.ones(50)) is None

    def test_positive_data(self):
        C = Bisymlog._heuristic_C(np.array([1.0, 10.0, 100.0]))
        assert C is not None and C > 0

    def test_mixed_sign(self):
        C = Bisymlog._heuristic_C(np.array([-10.0, 0.0, 10.0]))
        assert C is not None and C > 0


# ── Class API ────────────────────────────────────────────────────────────

@pytest.fixture
def lognormal_1d():
    return np.random.default_rng(7).lognormal(0, 1, 100)


@pytest.fixture
def mixed_2d():
    rng = np.random.default_rng(0)
    return np.column_stack([rng.lognormal(0, 1, 100), rng.standard_normal(100)])


class TestFitTransform:
    def test_1d_shape(self, lognormal_1d):
        assert Bisymlog(robust=False).fit_transform(lognormal_1d).shape == lognormal_1d.shape

    def test_2d_shape(self, mixed_2d):
        assert Bisymlog(robust=False).fit_transform(mixed_2d).shape == mixed_2d.shape

    def test_output_finite(self, lognormal_1d):
        assert np.isfinite(Bisymlog().fit_transform(lognormal_1d)).all()


class TestRoundtrip:
    @pytest.mark.parametrize("robust", [True, False])
    def test_1d(self, lognormal_1d, robust):
        b = Bisymlog(robust=robust)
        out = b.fit_transform(lognormal_1d)
        assert np.allclose(b.inverse_transform(out), lognormal_1d, rtol=1e-6)

    def test_2d(self, mixed_2d):
        b = Bisymlog(robust=False)
        out = b.fit_transform(mixed_2d)
        assert np.allclose(b.inverse_transform(out), mixed_2d, rtol=1e-6)


class TestNonFinite:
    def test_nan_passthrough(self):
        x = np.array([1.0, np.nan, 9.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        b = Bisymlog(robust=False)
        out = b.fit_transform(x)
        assert np.isnan(out[1])
        assert np.isfinite(out[~np.isnan(x)]).all()


class TestRescale:
    def test_rescale_quantile_out_of_range_raises(self):
        with pytest.raises(ValueError):
            Bisymlog(rescale_quantile=0.6)
        with pytest.raises(ValueError):
            Bisymlog(rescale_quantile=-0.1)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_rescale_roundtrip(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.lognormal(0, 1, 200), rng.standard_normal(200)])
        b = Bisymlog(robust=False, rescale_quantile=0.25)
        out = b.fit_transform(x)
        active = np.where(~b.skip_dims_)[0]
        if len(active) > 0:
            assert np.allclose(b.inverse_transform(out), x, rtol=1e-5)


class TestSerialisation:
    def test_dict_roundtrip(self, lognormal_1d):
        b = Bisymlog(robust=False)
        b.fit(lognormal_1d)
        b2 = Bisymlog.from_dict(b.to_dict())
        assert np.allclose(b.transform(lognormal_1d), b2.transform(lognormal_1d))

    def test_json_roundtrip(self, lognormal_1d):
        b = Bisymlog(robust=False)
        b.fit(lognormal_1d)
        b2 = Bisymlog.from_json(b.to_json())
        assert np.allclose(b.C_, b2.C_)


class TestSkip:
    def test_constant_skipped(self):
        data = np.column_stack([np.ones(50), np.random.default_rng(0).standard_normal(50)])
        b = Bisymlog(robust=False)
        b.fit(data)
        assert b.skip_dims_[0]
        assert not b.skip_dims_[1]

    def test_few_samples_skipped(self):
        data = np.random.default_rng(0).standard_normal((2, 3))
        b = Bisymlog(robust=False, min_samples=3)
        out = b.fit_transform(data)
        assert b.skip_dims_.all()
        assert np.allclose(out, data)


class TestErrors:
    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            Bisymlog().transform(np.array([1, 2, 3]))

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            Bisymlog().fit(np.zeros((2, 3, 4)))
