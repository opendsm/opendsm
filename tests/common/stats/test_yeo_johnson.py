"""Tests for Yeo-Johnson scalar kernels, fitting algorithm, and YeoJohnson class."""

import numpy as np
import pytest

from opendsm.common.stats.distribution_transform.yeo_johnson import (
    _yj,
    _yj_inverse,
    yj_transform,
    yj_inverse_transform,
    _yj_rectified_transform,
    _yj_hg,
    _fit_yj_lambda,
    YeoJohnson,
)
from opendsm.common.stats.distribution_transform._bc_yj_shared import (
    _WANT_VALUE, _WANT_DERIV, _LAM_EPS,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaExperimentalFeatureWarning"
)

RNG = np.random.default_rng


# ── Scalar _yj forward ──────────────────────────────────────────────────

class TestYJScalar:
    @pytest.mark.parametrize("x, lam, expected", [
        (0.0, 1.0, 0.0),
        (3.0, 1.0, 3.0),                          # identity
        (1.0, 0.0, np.log(2)),                     # log(1+x)
        (0.0, 0.0, 0.0),
        (-1.0, 2.0, -np.log(2)),                   # -log(1-x)
        (-0.5, 2.0, -np.log(1.5)),
        (2.0, 2.0, (9 - 1) / 2),                   # ((1+2)^2-1)/2 = 4
    ])
    def test_forward_known(self, x, lam, expected):
        assert np.isclose(_yj(x, lam, _WANT_VALUE), expected, atol=1e-12)

    @pytest.mark.parametrize("x, lam, expected", [
        (3.0, 2.0, 4.0),       # (3+1)^(2-1) = 4
        (0.0, 0.5, 1.0),       # (0+1)^(-0.5) = 1
        (-0.5, 1.0, 1.0),      # (1-(-0.5))^(1-1) = 1.5^0 = 1
    ])
    def test_derivative_known(self, x, lam, expected):
        assert np.isclose(_yj(x, lam, _WANT_DERIV), expected, atol=1e-12)

    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 1.5, 2.0])
    def test_zero_maps_to_zero(self, lam):
        assert abs(_yj(0.0, lam, _WANT_VALUE)) < 1e-12


# ── Scalar _yj_inverse ──────────────────────────────────────────────────

class TestYJInverse:
    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("x", [2.5, -1.5, 0.0])
    def test_roundtrip(self, x, lam):
        y = _yj(x, lam, _WANT_VALUE)
        assert abs(_yj_inverse(y, lam) - x) < 1e-10

    @pytest.mark.parametrize("lam", [0.0, 1.0, 2.0])
    def test_zero_maps_to_zero(self, lam):
        assert abs(_yj_inverse(0.0, lam)) < 1e-12

    def test_clamp_positive_branch(self):
        # y=10, lam=-0.5 → inner = 10*(-0.5)+1 = -4 → clamp to -1
        assert _yj_inverse(10.0, -0.5) == -1.0

    def test_clamp_negative_branch(self):
        # y=-10, lam=3 → p=-1, inner = 1-(-10)*(-1) = -9 → clamp to 1
        assert _yj_inverse(-10.0, 3.0) == 1.0


# ── Vectorised transforms ───────────────────────────────────────────────

class TestVectorised:
    @pytest.mark.parametrize("lam", [0.0, 0.7, 1.0, 1.5, 2.0])
    def test_array_roundtrip(self, lam):
        x = RNG(0).standard_normal(50)
        y = yj_transform(x, lam)
        assert np.allclose(yj_inverse_transform(y, lam), x, atol=1e-10)

    def test_shape_preserved(self):
        x = np.linspace(-2, 2, 50)
        assert yj_transform(x, 0.5).shape == x.shape


# ── Rectified transform ─────────────────────────────────────────────────

class TestRectified:
    @pytest.fixture
    def sorted_data(self):
        return np.sort(np.linspace(-3, 3, 100))

    def test_matches_unrectified_inside_bounds(self, sorted_data):
        Q = np.array([-1.0, 1.0])
        r = _yj_rectified_transform(sorted_data, 0.5, Q)
        u = yj_transform(sorted_data, 0.5)
        inside = (sorted_data >= Q[0]) & (sorted_data < Q[1])
        assert np.allclose(r[inside], u[inside])

    def test_differs_in_tails(self, sorted_data):
        Q = np.array([-1.0, 1.0])
        r = _yj_rectified_transform(sorted_data, 0.5, Q)
        u = yj_transform(sorted_data, 0.5)
        outside = (sorted_data < Q[0]) | (sorted_data >= Q[1])
        assert not np.allclose(r[outside], u[outside])

    def test_shape(self):
        xs = np.sort(np.linspace(-2, 2, 60))
        assert _yj_rectified_transform(xs, 1.0, np.array([-0.5, 0.5])).shape == xs.shape


# ── _yj_hg ───────────────────────────────────────────────────────────────

class TestYJHG:
    def test_at_lambda_zero(self):
        h, g = _yj_hg(3.0, 0.0)
        assert np.isclose(h, np.log(4.0))
        assert np.isclose(g, 0.5 * np.log(4.0) ** 2)

    def test_positive_branch(self):
        h, g = _yj_hg(2.0, 1.0)
        assert np.isclose(h, 2.0)  # identity at lam=1


# ── _fit_yj_lambda ──────────────────────────────────────────────────────

class TestFitYJLambda:
    def test_normal_lambda_near_one(self):
        _, lam = _fit_yj_lambda(RNG(42).standard_normal(200))
        assert abs(lam - 1.0) < 0.25

    def test_lognormal_lambda_less_than_one(self):
        _, lam = _fit_yj_lambda(RNG(43).lognormal(0, 1, 300))
        assert lam < 1.0

    def test_left_skewed_lambda_greater_than_one(self):
        _, lam = _fit_yj_lambda(-RNG(44).chisquare(3, 300))
        assert lam > 1.0

    def test_robust_to_outliers(self):
        rng = RNG(44)
        x = np.concatenate([rng.standard_normal(180), rng.standard_normal(20) * 20])
        _, lam = _fit_yj_lambda(x)
        assert abs(lam - 1.0) < 0.5

    def test_Q_perc_changes_lambda(self):
        x = RNG(45).lognormal(0, 1, 200)
        _, l1 = _fit_yj_lambda(x)
        _, l2 = _fit_yj_lambda(x, Q_perc=0.20)
        assert l1 != l2

    def test_pinned_normal(self):
        x = RNG(42).standard_normal(200)
        out, lam = _fit_yj_lambda(x)
        assert np.isclose(lam, 1.0395588659, rtol=1e-6)
        assert np.isclose(out[0], 0.396206124748, rtol=1e-10)
        assert np.isclose(out[1], -1.107365497916, rtol=1e-10)

    def test_pinned_lognormal(self):
        x = RNG(43).lognormal(0, 1, 200)
        out, lam = _fit_yj_lambda(x)
        assert np.isclose(lam, 0.2476838695, rtol=1e-6)
        assert np.isclose(out[0], 0.256640773525, rtol=1e-10)


# ── YeoJohnson class ────────────────────────────────────────────────────

@pytest.fixture
def sample_2d():
    rng = RNG(42)
    return np.column_stack([
        rng.standard_normal(200),
        rng.lognormal(2, 1.5, 200),
        rng.standard_normal(200) * 0.3,
    ])


class TestYeoJohnsonClass:
    def test_fit_transform_2d(self, sample_2d):
        assert YeoJohnson().fit_transform(sample_2d).shape == sample_2d.shape

    def test_fit_transform_1d(self):
        x = RNG(0).standard_normal(100)
        out = YeoJohnson().fit_transform(x)
        assert out.shape == x.shape and out.ndim == 1

    def test_roundtrip_2d(self, sample_2d):
        yj = YeoJohnson()
        out = yj.fit_transform(sample_2d)
        assert np.abs(sample_2d - yj.inverse_transform(out)).max() < 1e-6

    def test_roundtrip_1d(self):
        x = RNG(0).lognormal(0, 1, 200)
        yj = YeoJohnson()
        assert np.abs(x - yj.inverse_transform(yj.fit_transform(x))).max() < 1e-6

    def test_single_sample_roundtrip(self, sample_2d):
        yj = YeoJohnson()
        yj.fit(sample_2d)
        s = sample_2d[:1]
        assert np.allclose(yj.inverse_transform(yj.transform(s)), s, atol=1e-8)

    def test_batch_roundtrip(self, sample_2d):
        yj = YeoJohnson()
        yj.fit(sample_2d)
        b = sample_2d[:10]
        assert np.allclose(yj.inverse_transform(yj.transform(b)), b, atol=1e-6)

    def test_n_features(self, sample_2d):
        yj = YeoJohnson()
        yj.fit(sample_2d)
        assert yj.n_features_ == 3

    def test_dtype_float64(self, sample_2d):
        assert YeoJohnson().fit_transform(sample_2d.astype(np.float32)).dtype == np.float64


class TestYeoJohnsonSkip:
    def test_constant_dim(self):
        data = np.column_stack([np.ones(50), RNG(0).standard_normal(50)])
        yj = YeoJohnson()
        yj.fit(data)
        assert yj.skip_dims_[0] and not yj.skip_dims_[1]

    def test_small_dataset_all_skipped(self):
        data = RNG(0).standard_normal((3, 4))
        yj = YeoJohnson()
        out = yj.fit_transform(data)
        assert yj.skip_dims_.all()
        assert np.allclose(out, data)


class TestYeoJohnsonNonFinite:
    def test_nan_passthrough_transform(self):
        data = RNG(0).standard_normal((50, 2))
        yj = YeoJohnson()
        yj.fit(data)
        data[5, 0] = np.nan
        out = yj.transform(data)
        assert np.isnan(out[5, 0]) and np.isfinite(out[5, 1])

    def test_nan_passthrough_inverse(self):
        data = RNG(0).standard_normal((50, 2))
        yj = YeoJohnson()
        t = yj.fit_transform(data)
        t[3, 1] = np.nan
        r = yj.inverse_transform(t)
        assert np.isnan(r[3, 1]) and np.isfinite(r[3, 0])

    def test_inf_passthrough(self):
        data = RNG(0).standard_normal((50, 2))
        yj = YeoJohnson()
        yj.fit(data)
        data[0, 0] = np.inf
        assert np.isinf(yj.transform(data)[0, 0])


class TestYeoJohnsonRobust:
    def test_default_is_true(self):
        assert YeoJohnson().robust is True

    def test_non_robust_roundtrip(self):
        x = RNG(0).lognormal(0, 1, 200)
        yj = YeoJohnson(robust=False)
        assert np.allclose(yj.inverse_transform(yj.fit_transform(x)), x, atol=1e-5)

    def test_robust_vs_non_robust_differ(self):
        rng = RNG(42)
        x = np.concatenate([rng.standard_normal(180), rng.standard_normal(20) * 20])
        yj_r = YeoJohnson(robust=True)
        yj_r.fit(x)
        yj_nr = YeoJohnson(robust=False)
        yj_nr.fit(x)
        assert yj_r.lambdas_[0] != yj_nr.lambdas_[0]


class TestYeoJohnsonErrors:
    def test_unfitted_transform(self):
        with pytest.raises(RuntimeError):
            YeoJohnson().transform(np.array([1, 2, 3]))

    def test_unfitted_inverse(self):
        with pytest.raises(RuntimeError):
            YeoJohnson().inverse_transform(np.array([1, 2, 3]))

    def test_wrong_features(self):
        yj = YeoJohnson()
        yj.fit(RNG(0).standard_normal((50, 3)))
        with pytest.raises(ValueError, match="Expected 3 features"):
            yj.transform(RNG(0).standard_normal((10, 5)))

    def test_3d_input(self):
        with pytest.raises(ValueError, match="Expected 1-D or 2-D"):
            YeoJohnson().fit(np.zeros((2, 3, 4)))


class TestYeoJohnsonSerialisation:
    def test_dict_roundtrip(self, sample_2d):
        yj = YeoJohnson()
        yj.fit(sample_2d)
        yj2 = YeoJohnson.from_dict(yj.to_dict())
        s = sample_2d[:1]
        assert np.allclose(yj.transform(s), yj2.transform(s))

    def test_json_string(self, sample_2d):
        yj = YeoJohnson()
        yj.fit(sample_2d)
        yj2 = YeoJohnson.from_json(yj.to_json())
        assert np.allclose(yj.lambdas_, yj2.lambdas_)

    def test_json_file(self, sample_2d, tmp_path):
        yj = YeoJohnson()
        yj.fit(sample_2d)
        p = tmp_path / "yj.json"
        assert yj.to_json(p) is None
        yj2 = YeoJohnson.from_json(p)
        assert np.allclose(yj.lambdas_, yj2.lambdas_)

    def test_robust_flag_preserved(self):
        yj = YeoJohnson(robust=False)
        yj.fit(RNG(0).standard_normal(50))
        assert YeoJohnson.from_dict(yj.to_dict()).robust is False
