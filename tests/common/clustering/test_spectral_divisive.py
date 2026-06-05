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

"""Tests for recursive Fiedler-bisection spectral clustering.

Degenerate-data fallbacks (identical points, k=1 allowance) are covered by
the shared harness in ``test_algorithms.py``; this file targets recovery,
balanced bisection, refinement, and the Nystrom large-data path.
"""

import numpy as np
import pytest

from sklearn.metrics import adjusted_rand_score

from opendsm.common.clustering.settings import ClusteringSettings
from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive



def _blobs(k, n=50, d=5, sep=50.0, seed=0):
    """k well-separated blobs with ground-truth labels."""
    rng = np.random.default_rng(seed)
    data = np.vstack([rng.normal(c * sep, 1.0, (n, d)) for c in range(k)])
    truth = np.repeat(np.arange(k), n)

    return data, truth


def _sd_cs(lower, upper, **algo):
    """ClusteringSettings selecting spectral_divisive over a cluster range."""
    cs = ClusteringSettings(
        algorithm_selection="spectral_divisive",
        seed=42,
        spectral_divisive={"n_cluster": {"lower": lower, "upper": upper}, **algo},
    )

    return cs


class TestRecovery:
    """Fiedler bisection recovers well-separated structure."""

    def test_forced_k_recovers_blobs(self):
        """Forcing k=3 recovers three well-separated blobs (ARI ~ 1)."""
        data, truth = _blobs(3)
        result = spectral_divisive(data, _sd_cs(3, 3))
        assert result.k == 3
        assert adjusted_rand_score(truth, result.labels) > 0.99

    def test_first_split_is_balanced(self):
        """The first Fiedler cut bisects two equal blobs evenly, not 1-vs-rest."""
        data, _ = _blobs(2, n=50)
        result = spectral_divisive(data, _sd_cs(2, 2))
        _, counts = np.unique(result.labels, return_counts=True)
        assert min(counts) > len(data) // 4


class TestRefinement:
    """k-medians refinement after bisection does not harm recovery."""

    @pytest.mark.parametrize("refine", [True, False])
    def test_recovers_with_refinement_toggle(self, refine):
        """Recovery holds whether or not post-bisection refinement runs."""
        data, truth = _blobs(3)
        result = spectral_divisive(data, _sd_cs(3, 3, refinement_enabled=refine))
        assert adjusted_rand_score(truth, result.labels) > 0.99


class TestNystromPath:
    """The Nystrom approximation path activates above the sample threshold."""

    def test_large_data_path_recovers_blobs(self):
        """With n (150) above nystrom_samples (100), recovery still holds."""
        data, truth = _blobs(3, n=50)
        result = spectral_divisive(data, _sd_cs(3, 3, nystrom_samples=100))
        assert adjusted_rand_score(truth, result.labels) > 0.99


class TestDeterminism:
    """Seeded runs are reproducible."""

    def test_same_seed_identical_labels(self):
        """Identical seed and data -> identical labels."""
        data, _ = _blobs(3)
        a = spectral_divisive(data, _sd_cs(3, 3)).labels
        b = spectral_divisive(data, _sd_cs(3, 3)).labels
        assert np.array_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
