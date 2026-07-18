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

"""Unit tests for the spectral affinity helpers.

Covers the self-tuning (Zelnik-Manor & Perona) dense kernel, the sigma floor
that guards degenerate neighbourhoods, and the auto diffusion-time heuristic.
"""

import numpy as np
import pytest

from opendsm.common.clustering.algorithms.spectral import _affinity as af



def _two_blobs(n=50, d=5, sep=50.0, seed=0):
    """Two well-separated isotropic blobs."""
    rng = np.random.default_rng(seed)
    data = np.vstack([rng.normal(0, 1, (n, d)), rng.normal(sep, 1, (n, d))])

    return data


class TestSelfTuningAffinityDense:
    """Properties of the locally-scaled dense affinity matrix."""

    def test_symmetric_nonnegative_unit_diagonal(self):
        """Affinity is symmetric, non-negative, with self-affinity 1."""
        A = af._self_tuning_affinity_dense(_two_blobs(), 7)
        assert np.allclose(A, A.T)
        assert (A >= 0).all()
        assert np.allclose(np.diag(A), 1.0)

    def test_identical_data_is_all_ones(self):
        """Coincident points produce a rank-1 all-ones affinity (no structure)."""
        A = af._self_tuning_affinity_dense(np.ones((10, 4)), 7)
        assert np.allclose(A, 1.0)

    def test_scale_invariant(self):
        """Self-tuning normalises by local scale, so rescaling data is a no-op."""
        data = _two_blobs()
        A = af._self_tuning_affinity_dense(data, 7)
        A_scaled = af._self_tuning_affinity_dense(100.0 * data, 7)
        assert np.allclose(A, A_scaled, atol=1e-8)


class TestSigmaFloor:
    """The sigma floor raises degenerate local scales to median * 0.1."""

    def test_floors_small_values_only(self):
        """Values below the floor are raised; larger values are untouched."""
        sigma = np.array([1.0, 2.0, 3.0, 0.001])
        floored = af._sigma_floor(sigma)
        expected_floor = np.median(sigma) * 0.1
        assert floored[3] == pytest.approx(expected_floor)
        assert np.array_equal(floored[:3], sigma[:3])

    def test_no_change_when_all_above_floor(self):
        """A well-conditioned sigma vector passes through unchanged."""
        sigma = np.array([10.0, 11.0, 12.0])
        assert np.array_equal(af._sigma_floor(sigma), sigma)


class TestAutoDiffusionTime:
    """Diffusion time is chosen from the spectral gap, clamped to [2, 10]."""

    def test_no_gap_uses_max_smoothing(self):
        """A flat spectrum (no clear gap) selects the maximum time of 10."""
        eigenvalues = np.array([1.0, 0.99, 0.98, 0.97])
        assert af._auto_diffusion_time(eigenvalues) == 10

    def test_large_early_gap_uses_minimal_smoothing(self):
        """A spectrum that decays immediately needs only minimal diffusion."""
        eigenvalues = np.array([1.0, 0.001, 0.0005])
        assert af._auto_diffusion_time(eigenvalues) == 2

    @pytest.mark.parametrize("tail_len", range(1, 8))
    def test_always_clamped_to_range(self, tail_len):
        """The returned time always lies within [2, 10] regardless of spectrum."""
        eigenvalues = np.array([1.0] + list(np.linspace(0.9, 0.1, tail_len)))
        assert 2 <= af._auto_diffusion_time(eigenvalues) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
