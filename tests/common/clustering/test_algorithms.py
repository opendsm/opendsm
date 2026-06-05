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

"""Parametrized tests for behaviour shared across clustering algorithms.

Each test is parametrized over *algorithm*, so a signature change in one
algorithm's entry point only breaks the tests for that algorithm -- not
all of them.
"""

import numpy as np
import pytest

from sklearn.metrics import adjusted_rand_score

from opendsm.common.clustering.algorithms.spectral import spectral
from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive
from opendsm.common.clustering.algorithms.bisect_k_means import bisect_k_means
from opendsm.common.clustering.algorithms.bisect_k_medians import bisect_k_medians

from .conftest import make_clustering_settings


# ---------------------------------------------------------------------------
# Dispatch helper
# ---------------------------------------------------------------------------

_ALGO_FN = {
    "spectral": spectral,
    "bisecting_kmeans": bisect_k_means,
}

# Algorithm names that the parametrized tests iterate over.
_ALGORITHMS = list(_ALGO_FN.keys())


def _algo_settings_key(algorithm):
    """Return the keyword for algorithm-specific sub-settings."""
    return algorithm  # "spectral" or "bisecting_kmeans"


def _run(data, algorithm, seed=42, n_lower=2, n_upper=5, **extra_algo):
    """Run an algorithm directly and return its ClusteringResult."""
    algo_kw = {_algo_settings_key(algorithm): {"n_cluster": {"lower": n_lower, "upper": n_upper}, **extra_algo}}
    cs = make_clustering_settings(algorithm, seed=seed, **algo_kw)
    return _ALGO_FN[algorithm](data, cs)


# ---------------------------------------------------------------------------
# Shared-behaviour tests
# ---------------------------------------------------------------------------

class TestAlgorithmSharedBehavior:
    """Tests that apply identically to every supported algorithm."""

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    def test_valid_labels(self, simple_2d_data, algorithm):
        """Labels have correct length, are non-negative, and contain >0 unique values."""
        labels = _run(simple_2d_data, algorithm).labels
        assert len(labels) == len(simple_2d_data)
        assert len(np.unique(labels)) > 0
        assert np.all(labels >= 0)

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    def test_determinism(self, simple_2d_data, algorithm):
        """Same seed + same data -> identical labels."""
        labels1 = _run(simple_2d_data, algorithm, seed=42).labels
        labels2 = _run(simple_2d_data, algorithm, seed=42).labels
        assert np.array_equal(labels1, labels2)

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    def test_different_seeds(self, simple_2d_data, algorithm):
        """Different seeds both produce valid labels."""
        labels1 = _run(simple_2d_data, algorithm, seed=42).labels
        labels2 = _run(simple_2d_data, algorithm, seed=123).labels
        assert len(np.unique(labels1)) > 0
        assert len(np.unique(labels2)) > 0

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    @pytest.mark.parametrize("n_clusters", [2, 3, 10])
    def test_exact_cluster_count(self, simple_2d_data, algorithm, n_clusters):
        """Requesting exactly k clusters produces k clusters."""
        labels = _run(simple_2d_data, algorithm, n_lower=n_clusters, n_upper=n_clusters).labels
        assert len(np.unique(labels)) == n_clusters

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    def test_cluster_range(self, simple_2d_data, algorithm):
        """Requesting [2,5] clusters produces k in that range."""
        labels = _run(simple_2d_data, algorithm, n_lower=2, n_upper=5).labels
        n_clusters = len(np.unique(labels))
        assert 2 <= n_clusters <= 5

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    def test_uniform_data(self, algorithm):
        """Uniform data produces valid (if not meaningful) clusters."""
        np.random.seed(42)
        data = np.random.uniform(-1, 1, (100, 10))
        labels = _run(data, algorithm, n_lower=3, n_upper=3).labels
        assert len(labels) == 100
        assert len(np.unique(labels)) > 0

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    def test_identical_samples(self, algorithm):
        """All-identical data is handled without a hang or IndexError: the
        algorithm either returns a degenerate labeling or raises a clear
        ValueError when the fixed k cannot be formed.
        """
        data = np.ones((50, 10))
        try:
            labels = _run(data, algorithm, n_lower=3, n_upper=3).labels
        except ValueError:
            return

        assert len(labels) == 50
        assert len(np.unique(labels)) > 0

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    def test_negative_values(self, algorithm):
        """Negative-valued data is handled correctly."""
        np.random.seed(42)
        data = np.random.randn(100, 10) - 5
        labels = _run(data, algorithm, n_lower=3, n_upper=3).labels
        assert len(labels) == 100
        assert len(np.unique(labels)) == 3

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    def test_mixed_scale_features(self, algorithm):
        """Data with features at very different scales is handled."""
        np.random.seed(42)
        data = np.column_stack([
            np.random.randn(100) * 0.01,
            np.random.randn(100) * 1.0,
            np.random.randn(100) * 100.0,
        ])
        labels = _run(data, algorithm, n_lower=2, n_upper=2).labels
        assert len(labels) == 100
        assert len(np.unique(labels)) == 2

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    @pytest.mark.parametrize("n_samples,n_features,n_clusters", [
        (10, 5, 2),
        (1000, 20, 5),
        (100, 50, 3),
        (100, 2, 3),
    ], ids=["small", "large", "high_dim", "low_dim"])
    @pytest.mark.slow
    def test_varied_data_shapes(self, algorithm, n_samples, n_features, n_clusters):
        """Handles datasets of varying size and dimensionality."""
        np.random.seed(42)
        data = np.random.randn(n_samples, n_features)
        labels = _run(data, algorithm, n_lower=n_clusters, n_upper=n_clusters).labels
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == n_clusters

    @pytest.mark.parametrize("algorithm", _ALGORITHMS)
    def test_well_separated_clusters(self, algorithm):
        """Well-separated clusters are correctly identified."""
        np.random.seed(42)
        cluster1 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(30, 5) + np.array([10, 10, 10, 10, 10])
        cluster3 = np.random.randn(30, 5) + np.array([20, 20, 20, 20, 20])
        data = np.vstack([cluster1, cluster2, cluster3])

        true_labels = np.array([0] * 30 + [1] * 30 + [2] * 30)

        labels = _run(data, algorithm, n_lower=3, n_upper=3).labels

        assert len(np.unique(labels)) == 3
        # ground-truth recovery; labels may be permuted, so compare with ARI
        assert adjusted_rand_score(true_labels, labels) > 0.95


# Bisection-family algorithms cannot manufacture clusters from identical
# points (each split collapses to one side), so they report degeneracy
# honestly — unlike spectral, whose k-means label step fabricates clusters.
_BISECTION_ALGOS = {
    "bisecting_kmedians": bisect_k_medians,
    "spectral_divisive": spectral_divisive,
}


class TestDegenerateDataFallback:
    """Unclusterable data: terminate (no hang) and honour the k=1 allowance."""

    @pytest.mark.parametrize("algorithm", list(_BISECTION_ALGOS))
    def test_k1_allowed_returns_single_cluster(self, algorithm):
        """Identical data with k=1 allowed terminates and yields one cluster."""
        data = np.ones((30, 4))
        algo_kw = {algorithm: {"n_cluster": {"lower": 1, "upper": 6}}}
        cs = make_clustering_settings(algorithm, **algo_kw)

        result = _BISECTION_ALGOS[algorithm](data, cs)

        assert result.k == 1

    @pytest.mark.parametrize("algorithm", list(_BISECTION_ALGOS))
    def test_k1_disallowed_raises(self, algorithm):
        """Identical data with k=1 disallowed terminates and raises rather
        than returning a below-lower-bound clustering.
        """
        data = np.ones((30, 4))
        algo_kw = {algorithm: {"n_cluster": {"lower": 2, "upper": 6}}}
        cs = make_clustering_settings(algorithm, **algo_kw)

        result = _BISECTION_ALGOS[algorithm](data, cs)

        with pytest.raises(ValueError):
            _ = result.k


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
