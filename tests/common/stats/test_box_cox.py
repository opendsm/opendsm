"""Tests for Box-Cox scalar kernels, fitting algorithm, and BoxCox class."""

import warnings

import numpy as np
import pytest

from opendsm.common.stats.distribution_transform.box_cox import (
    _bc,
    _bc_inverse,
    _bc_hg,
    bc_transform,
    bc_inverse_transform,
    _bc_rectified_transform,
    _fit_bc_lambda,
    BoxCox,
)
from opendsm.common.stats.distribution_transform._bc_yj_shared import (
    _WANT_VALUE, _WANT_DERIV, _LAM_EPS,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaExperimentalFeatureWarning"
)

RNG = np.random.default_rng


# ── Scalar _bc forward ──────────────────────────────────────────────────

class TestBCScalar:
    @pytest.mark.parametrize("x, lam, expected", [
        (1.0, 1.0, 0.0),                              # (1^1 - 1)/1 = 0
        (2.0, 1.0, 1.0),                              # (2-1)/1 = 1
        (4.0, 1.0, 3.0),                              # (4-1)/1 = 3
        (np.e, 0.0, 1.0),                             # log(e) = 1
        (np.e ** 2, 0.0, 2.0),                        # log(e^2) = 2
        (4.0, 0.5, 2.0),                              # (sqrt(4)-1)/0.5 = 2
        (9.0, 0.5, 4.0),                              # (3-1)/0.5 = 4
    ])
    def test_forward_known(self, x, lam, expected):
        assert np.isclose(_bc(x, lam, _WANT_VALUE), expected, atol=1e-12)

    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.0])
    def test_one_maps_to_zero(self, lam):
        """BC(1, λ) = 0 for all λ (since 1^λ = 1)."""
        assert abs(_bc(1.0, lam, _WANT_VALUE)) < 1e-12

    @pytest.mark.parametrize("x, lam, expected", [
        (2.0, 2.0, 2.0),          # x^(lam-1) = 2^1 = 2
        (4.0, 0.5, 0.5),          # 4^(0.5-1) = 4^(-0.5) = 1/2
        (np.e, 0.0, 1.0 / np.e),  # 1/x
    ])
    def test_derivative_known(self, x, lam, expected):
        assert np.isclose(_bc(x, lam, _WANT_DERIV), expected, atol=1e-12)


# ── Scalar _bc_inverse ──────────────────────────────────────────────────

class TestBCInverse:
    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.0])
    def test_roundtrip(self, lam):
        x = 3.0
        assert abs(_bc_inverse(_bc(x, lam, _WANT_VALUE), lam) - x) < 1e-10

    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.0])
    def test_zero_maps_to_one(self, lam):
        """BC_inverse(0, λ) = 1 for all λ."""
        assert np.isclose(_bc_inverse(0.0, lam), 1.0, atol=1e-12)

    def test_clamp(self):
        # y=-10, lam=0.5 → inner = -10*0.5+1 = -4 ≤ 0 → clamp to 0
        assert _bc_inverse(-10.0, 0.5) == 0.0


# ── Vectorised transforms ───────────────────────────────────────────────

class TestVectorised:
    @pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.0])
    def test_array_roundtrip(self, lam):
        x = np.abs(RNG(0).standard_normal(50)) + 0.1
        y = bc_transform(x, lam)
        assert np.allclose(bc_inverse_transform(y, lam), x, atol=1e-10)

    def test_shape_preserved(self):
        x = np.linspace(0.1, 5, 50)
        assert bc_transform(x, 0.5).shape == x.shape


# ── _bc_hg ───────────────────────────────────────────────────────────────

class TestBCHG:
    def test_at_lambda_zero(self):
        h, g = _bc_hg(4.0, 0.0)
        assert np.isclose(h, np.log(4.0))
        assert np.isclose(g, 0.5 * np.log(4.0) ** 2)

    def test_at_lambda_one(self):
        h, g = _bc_hg(3.0, 1.0)
        # h = (3^1-1)/1 = 2
        assert np.isclose(h, 2.0)


# ── Rectified transform ─────────────────────────────────────────────────

class TestRectified:
    def test_inside_matches_unrectified(self):
        xs = np.sort(np.linspace(0.1, 5, 100))
        Q = np.array([1.0, 3.0])
        r = _bc_rectified_transform(xs, 0.5, Q)
        u = bc_transform(xs, 0.5)
        inside = (xs >= Q[0]) & (xs < Q[1])
        assert np.allclose(r[inside], u[inside])

    def test_shape(self):
        xs = np.sort(np.linspace(0.1, 5, 60))
        assert _bc_rectified_transform(xs, 1.0, np.array([1.0, 3.0])).shape == xs.shape


# ── _fit_bc_lambda ──────────────────────────────────────────────────────

class TestFitBCLambda:
    def test_lognormal(self):
        x = RNG(43).lognormal(0, 1, 300)
        out, lam = _fit_bc_lambda(x)
        assert np.isfinite(lam)
        assert np.isfinite(out).all()

    def test_output_shape(self):
        x = RNG(0).lognormal(0, 1, 100)
        out, lam = _fit_bc_lambda(x)
        assert out.shape == x.shape


# ── BoxCox class ─────────────────────────────────────────────────────────

@pytest.fixture
def positive_2d():
    rng = RNG(42)
    return np.column_stack([
        rng.lognormal(0, 1, 200),
        rng.lognormal(2, 0.5, 200),
        rng.uniform(0.1, 10, 200),
    ])


@pytest.fixture
def positive_1d():
    return RNG(0).lognormal(0, 1, 200)


class TestBoxCoxClass:
    def test_fit_transform_2d(self, positive_2d):
        assert BoxCox().fit_transform(positive_2d).shape == positive_2d.shape

    def test_fit_transform_1d(self, positive_1d):
        out = BoxCox().fit_transform(positive_1d)
        assert out.shape == positive_1d.shape and out.ndim == 1

    def test_roundtrip_2d(self, positive_2d):
        bc = BoxCox()
        out = bc.fit_transform(positive_2d)
        assert np.abs(positive_2d - bc.inverse_transform(out)).max() < 1e-5

    def test_roundtrip_1d(self, positive_1d):
        bc = BoxCox()
        assert np.abs(positive_1d - bc.inverse_transform(bc.fit_transform(positive_1d))).max() < 1e-5

    def test_n_features(self, positive_2d):
        bc = BoxCox()
        bc.fit(positive_2d)
        assert bc.n_features_ == 3


class TestBoxCoxNonPositive:
    def test_non_positive_dim_skipped_with_warning(self):
        data = np.column_stack([
            np.random.default_rng(0).standard_normal(50),  # has negatives
            np.random.default_rng(0).lognormal(0, 1, 50),  # all positive
        ])
        bc = BoxCox()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bc.fit(data)
        assert bc.skip_dims_[0], "Dim with negatives should be skipped"
        assert not bc.skip_dims_[1]

    def test_all_positive_no_skip(self, positive_2d):
        bc = BoxCox()
        bc.fit(positive_2d)
        assert not bc.skip_dims_.any()


class TestBoxCoxSkip:
    def test_constant_dim(self):
        data = np.column_stack([np.ones(50), RNG(0).lognormal(0, 1, 50)])
        bc = BoxCox()
        bc.fit(data)
        assert bc.skip_dims_[0] and not bc.skip_dims_[1]

    def test_small_dataset_all_skipped(self):
        data = RNG(0).lognormal(0, 1, (3, 4))
        bc = BoxCox()
        out = bc.fit_transform(data)
        assert bc.skip_dims_.all()
        assert np.allclose(out, data)


class TestBoxCoxNonFinite:
    def test_nan_passthrough(self, positive_2d):
        bc = BoxCox()
        bc.fit(positive_2d)
        data = positive_2d.copy()
        data[5, 0] = np.nan
        out = bc.transform(data)
        assert np.isnan(out[5, 0]) and np.isfinite(out[5, 1])


class TestBoxCoxRobust:
    def test_default_true(self):
        assert BoxCox().robust is True

    def test_non_robust_roundtrip(self, positive_1d):
        bc = BoxCox(robust=False)
        assert np.abs(positive_1d - bc.inverse_transform(bc.fit_transform(positive_1d))).max() < 1e-5

    @pytest.mark.parametrize("robust", [True, False])
    def test_finite_output(self, positive_1d, robust):
        assert np.isfinite(BoxCox(robust=robust).fit_transform(positive_1d)).all()


class TestBoxCoxSerialisation:
    def test_dict_roundtrip(self, positive_2d):
        bc = BoxCox()
        bc.fit(positive_2d)
        bc2 = BoxCox.from_dict(bc.to_dict())
        s = positive_2d[:1]
        assert np.allclose(bc.transform(s), bc2.transform(s))

    def test_json_string(self, positive_2d):
        bc = BoxCox()
        bc.fit(positive_2d)
        bc2 = BoxCox.from_json(bc.to_json())
        assert np.allclose(bc.lambdas_, bc2.lambdas_)

    def test_json_file(self, positive_2d, tmp_path):
        bc = BoxCox()
        bc.fit(positive_2d)
        p = tmp_path / "bc.json"
        bc.to_json(p)
        bc2 = BoxCox.from_json(p)
        assert np.allclose(bc.lambdas_, bc2.lambdas_)

    def test_robust_flag_preserved(self):
        bc = BoxCox(robust=False)
        bc.fit(RNG(0).lognormal(0, 1, 50))
        assert BoxCox.from_dict(bc.to_dict()).robust is False


class TestBoxCoxErrors:
    def test_unfitted(self):
        with pytest.raises(RuntimeError):
            BoxCox().transform(np.array([1, 2, 3]))

    def test_wrong_features(self):
        bc = BoxCox()
        bc.fit(RNG(0).lognormal(0, 1, (50, 3)))
        with pytest.raises(ValueError, match="Expected 3 features"):
            bc.transform(np.ones((10, 5)))

    def test_3d_input(self):
        with pytest.raises(ValueError):
            BoxCox().fit(np.zeros((2, 3, 4)))
