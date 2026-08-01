#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for Parallel Analysis component selection and its helpers.

PA picks the retained component count by comparing real eigenvalues against a
permutation null.  These tests pin the deterministic helpers (permutation
count, percentile, eigenvalue normalisation, block permutation) and the
selection contract (always >= 1, deterministic under seed).
"""

import numpy as np
import pytest

from opendsm.common.clustering.transform import parallel_analysis as pa
from opendsm.common.test_data import load_test_data



def _blobs(k, n=50, d=20, sep=30.0, seed=0):
    """k well-separated isotropic blobs stacked into one array."""
    rng = np.random.default_rng(seed)
    data = np.vstack([rng.normal(c * sep, 1.0, (n, d)) for c in range(k)])

    return data


class TestPermutationCountHeuristic:
    """``_pa_n_permutations`` decreases smoothly toward an asymptote of 15."""

    @pytest.mark.parametrize("n,expected", [(7, 279), (25, 157), (365, 16), (2000, 15)])
    def test_representative_values(self, n, expected):
        """Pinned values from the documented inverse-square decay."""
        assert pa._pa_n_permutations(n) == expected

    def test_monotone_decreasing(self):
        """More samples never increases the permutation count."""
        counts = [pa._pa_n_permutations(n) for n in (5, 25, 100, 500, 5000)]
        assert all(a >= b for a, b in zip(counts, counts[1:]))

    def test_floor_at_fifteen(self):
        """The count never drops below the asymptotic floor of 15."""
        assert pa._pa_n_permutations(10**6) >= 15


class TestPercentileHeuristic:
    """``_pa_percentile`` rises smoothly from 75 to 95."""

    @pytest.mark.parametrize("n,expected", [(5, 76.0), (40, 85.0), (200, 95.0)])
    def test_representative_values(self, n, expected):
        """Pinned values from the documented sigmoid transition (centre n=40)."""
        assert pa._pa_percentile(n) == pytest.approx(expected, abs=0.5)

    def test_bounded_between_75_and_95(self):
        """The threshold stays within the documented band for all n."""
        for n in (1, 10, 40, 100, 10**5):
            assert 75.0 <= pa._pa_percentile(n) <= 95.0

    def test_monotone_increasing(self):
        """More samples never lowers the percentile threshold."""
        pcts = [pa._pa_percentile(n) for n in (1, 20, 40, 80, 500)]
        assert all(a <= b for a, b in zip(pcts, pcts[1:]))


class TestSigmoidScalar:
    """The numerically-stable scalar sigmoid used by the percentile heuristic."""

    def test_centre_is_half(self):
        """At x == x0 the sigmoid is exactly 0.5."""
        assert pa._sigmoid_scalar(5.0, x0=5.0, k=2.0) == pytest.approx(0.5)

    def test_saturates_both_tails(self):
        """Large |x - x0| saturates toward 1 and 0 without overflow."""
        assert pa._sigmoid_scalar(1e6, x0=0.0, k=1.0) == pytest.approx(1.0)
        assert pa._sigmoid_scalar(-1e6, x0=0.0, k=1.0) == pytest.approx(0.0)


class TestEigenvalues:
    """``_compute_pa_eigenvalues`` returns a normalised, padded spectrum."""

    def test_sums_to_one(self):
        """Non-degenerate eigenvalues are normalised to sum to 1."""
        eigs = pa._compute_pa_eigenvalues(_blobs(3), "pca", None, n_max=10)
        assert float(eigs.sum()) == pytest.approx(1.0)

    def test_length_is_n_max_zero_padded(self):
        """Output length equals n_max regardless of available components."""
        eigs = pa._compute_pa_eigenvalues(_blobs(2), "pca", None, n_max=15)
        assert eigs.shape == (15,)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
    def test_constant_data_sums_to_zero(self):
        """Zero-variance data has no spectrum; the result is all zeros.

        sklearn PCA emits a benign 0/0 divide warning on constant input; the
        near-zero-total guard then returns the un-normalised (all-zero) array.
        """
        eigs = pa._compute_pa_eigenvalues(np.ones((50, 20)), "pca", None, n_max=10)
        assert float(eigs.sum()) == 0.0

    def test_unknown_method_raises(self):
        """An unrecognised decomposition method is rejected."""
        with pytest.raises(ValueError, match="Unknown PA method"):
            pa._compute_pa_eigenvalues(_blobs(2), "lda", None, n_max=5)


class TestBlockPermute:
    """``_block_permute_dwt`` shuffles rows per band, preserving column values."""

    def test_output_shape_is_concatenated_bands(self):
        """The result stacks all subbands side by side."""
        rng = np.random.RandomState(0)
        bands = [np.random.default_rng(1).normal(0, 1, (30, 4)),
                 np.random.default_rng(2).normal(0, 1, (30, 6))]
        out = pa._block_permute_dwt(bands, rng)
        assert out.shape == (30, 10)

    def test_preserves_each_column_value_multiset(self):
        """Row shuffling reorders but never alters a column's set of values."""
        rng = np.random.RandomState(0)
        band = np.arange(30 * 4, dtype=float).reshape(30, 4)
        out = pa._block_permute_dwt([band], rng)
        for col in range(4):
            assert np.array_equal(np.sort(out[:, col]), np.sort(band[:, col]))


class TestComponentSelection:
    """Selection contract: deterministic, always >= 1."""

    def test_deterministic_under_seed(self):
        """Same seed and data -> identical component count."""
        a = pa._parallel_analysis_n_components(_blobs(3), method="pca", seed=7)
        b = pa._parallel_analysis_n_components(_blobs(3), method="pca", seed=7)
        assert a == b

    def test_always_at_least_one(self):
        """The retained count is floored at 1 for structured and noise inputs."""
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1, (100, 20))
        assert pa._parallel_analysis_n_components(_blobs(3), method="pca", seed=0) >= 1
        assert pa._parallel_analysis_n_components(noise, method="pca", seed=0) >= 1

    def test_single_feature_returns_one(self):
        """When no component can be tested (n_max < 1) the floor of 1 applies."""
        single = np.random.default_rng(0).normal(0, 1, (50, 1))
        assert pa._parallel_analysis_n_components(single, method="pca", seed=0) == 1


class TestFourierBasis:
    """The basis must stay orthonormal: FPCA-as-PCA depends on it."""

    @pytest.mark.parametrize("n_basis", [5, 9, 13, 21])
    def test_basis_is_orthonormal_in_l2(self, n_basis):
        """The L2 Gram matrix is the identity, so no Gram weighting is needed.

        Quadrature on a fine grid stands in for the analytic inner product.
        """
        x = np.linspace(0.0, 1.0, 4001)
        design = pa._fourier_design(x, n_basis)
        gram = np.trapezoid(design[:, :, None] * design[:, None, :], x, axis=0)
        np.testing.assert_allclose(
            gram, np.eye(design.shape[1]), atol=1e-12,
            err_msg="Fourier basis lost L2 orthonormality; FPCA can no longer "
                    "reduce to PCA on the basis coefficients",
        )

    def test_even_n_basis_is_rounded_up(self):
        """Sine/cosine terms only pair off at odd sizes, so 8 yields 9 columns."""
        assert pa._fourier_design(np.arange(48), 8).shape[1] == 9

    def test_integer_grid_is_promoted_to_float(self):
        """``fpca_transform`` passes ``np.arange``; integer maths would zero the basis."""
        design = pa._fourier_design(np.arange(24), 7)
        assert design.dtype == np.float64
        assert np.all(design[:, 0] != 0.0)


class TestFpcaRegression:
    """Values pinned from the bundled seasonal-hourly loadshape.

    The recorded numbers come from the functional-PCA implementation these
    helpers replaced, so a drift here means the transform changed behaviour.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def loadshape(cls):
        """First 200 meters of the real seasonal-hourly-day-of-week loadshape."""
        df = load_test_data("seasonal_hourly_day_of_week_loadshape")
        data = df.to_numpy(dtype=float)[:200]

        return data

    def test_explained_variance_matches_recorded_spectrum(self, loadshape):
        """The retained variance spectrum is unchanged."""
        expected = [
            0.97572792048, 0.021075190686, 0.002041678573,
            0.000661840778, 0.000308584399, 9.3366674e-05,
        ]
        actual = pa._fpca_explained_variance(loadshape, np.arange(loadshape.shape[1]), 6)
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_transform_matches_recorded_scores(self, loadshape):
        """Scores keep both magnitude and sign convention."""
        expected = [
            [-507.654806453, -177.194137477, -64.644063494, -28.377360186],
            [-1875.158207141, -58.601913192, 31.757905136, 73.847300205],
            [-911.7350883, -56.447863958, -80.121285112, 173.219428201],
        ]
        scores = pa._fpca_transform_with_n(np.arange(loadshape.shape[1]), loadshape, 4)
        np.testing.assert_allclose(scores[:3, :4], expected, rtol=1e-7)

    def test_parallel_analysis_selects_recorded_component_count(self, loadshape):
        """The retained component count is the quantity clustering consumes."""
        n = pa._parallel_analysis_n_components(
            loadshape, method="fpca", grid_points=np.arange(loadshape.shape[1]), seed=0,
        )
        assert n == 2, f"expected 2 retained components, got {n}"

    def test_transform_is_reproducible(self, loadshape):
        """Guards against an SVD solver that samples a random subspace."""
        x = np.arange(loadshape.shape[1])
        first = pa._fpca_transform_with_n(x, loadshape, 4)
        second = pa._fpca_transform_with_n(x, loadshape, 4)
        np.testing.assert_array_equal(
            first, second,
            err_msg="FPCA scores differ between identical calls; the PCA solver "
                    "is no longer deterministic",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
