"""Tests for Standardize class."""

import numpy as np
import pytest

from opendsm.common.stats.distribution_transform.standardize import Standardize

pytestmark = pytest.mark.filterwarnings(
    "ignore::numba.core.errors.NumbaExperimentalFeatureWarning"
)


@pytest.fixture
def iqr_data():
    return np.arange(1.0, 11.0)


@pytest.fixture
def normal_2d():
    rng = np.random.default_rng(0)
    return np.column_stack([rng.standard_normal(100), rng.lognormal(0, 1, 100)])


class TestFitTransform:
    def test_1d_shape(self, iqr_data):
        s = Standardize()
        assert s.fit_transform(iqr_data).shape == iqr_data.shape

    def test_2d_shape(self, normal_2d):
        s = Standardize()
        assert s.fit_transform(normal_2d).shape == normal_2d.shape

    def test_iqr_median_zero(self, iqr_data):
        out = Standardize(robust_type="iqr").fit_transform(iqr_data)
        assert np.isclose(np.median(out), 0.0, atol=1e-10)

    def test_iqr_pinned(self, iqr_data):
        out = Standardize(robust_type="iqr").fit_transform(iqr_data)
        assert np.isclose(out[0], -1.21408155, rtol=1e-5)
        assert np.isclose(out[4], -0.13489795, rtol=1e-5)
        assert np.isclose(out[9], 1.21408155, rtol=1e-5)

    @pytest.mark.parametrize("rtype", ["iqr", "huber_m_estimate"])
    def test_output_finite(self, rtype):
        x = np.random.default_rng(1).standard_normal(100)
        assert np.isfinite(Standardize(robust_type=rtype).fit_transform(x)).all()


class TestRoundtrip:
    def test_1d(self, iqr_data):
        s = Standardize(robust_type="iqr")
        out = s.fit_transform(iqr_data)
        assert np.allclose(s.inverse_transform(out), iqr_data, atol=1e-10)

    def test_2d(self, normal_2d):
        s = Standardize()
        out = s.fit_transform(normal_2d)
        assert np.allclose(s.inverse_transform(out), normal_2d, atol=1e-10)


class TestSkip:
    def test_constant_dim_skipped(self):
        data = np.column_stack([np.ones(50), np.random.default_rng(0).standard_normal(50)])
        s = Standardize()
        s.fit(data)
        assert s.skip_dims_[0]
        assert not s.skip_dims_[1]

    def test_few_samples_skipped(self):
        data = np.random.default_rng(0).standard_normal((2, 3))
        s = Standardize(min_samples=3)
        s.fit(data)
        assert s.skip_dims_.all()


class TestSerialisation:
    def test_dict_roundtrip(self, normal_2d):
        s = Standardize(robust_type="huber_m_estimate")
        s.fit(normal_2d)
        s2 = Standardize.from_dict(s.to_dict())
        assert np.allclose(s.transform(normal_2d), s2.transform(normal_2d))

    def test_preserves_robust_type(self):
        s = Standardize(robust_type="huber_m_estimate")
        s.fit(np.random.default_rng(0).standard_normal(50))
        assert Standardize.from_dict(s.to_dict()).robust_type == "huber_m_estimate"

    def test_json_string_roundtrip(self, normal_2d):
        s = Standardize()
        s.fit(normal_2d)
        s2 = Standardize.from_json(s.to_json())
        assert np.allclose(s.mu_, s2.mu_)

    def test_json_file_roundtrip(self, normal_2d, tmp_path):
        s = Standardize()
        s.fit(normal_2d)
        fpath = tmp_path / "std.json"
        s.to_json(fpath)
        s2 = Standardize.from_json(fpath)
        assert np.allclose(s.sigma_, s2.sigma_)


class TestErrors:
    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            Standardize().transform(np.array([1, 2, 3]))

    def test_wrong_features_raises(self):
        s = Standardize()
        s.fit(np.random.default_rng(0).standard_normal((50, 3)))
        with pytest.raises(ValueError, match="Expected 3 features"):
            s.transform(np.ones((10, 5)))

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="Expected 1-D or 2-D"):
            Standardize().fit(np.zeros((2, 3, 4)))
