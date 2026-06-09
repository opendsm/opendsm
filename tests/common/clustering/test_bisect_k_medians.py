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

"""Settings, recovery and baseline tests for bisecting KMedians.

Shared behaviour (valid labels, determinism, cluster range, edge/degenerate
cases) is exercised by ``test_algorithms.py`` via the parametrized harness;
this file covers bisecting-KMedians-specific settings and a drift snapshot.
"""

import numpy as np
import pytest

from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from opendsm.common.clustering.algorithms.bisect_k_medians import bisect_k_medians

from .conftest import make_clustering_settings



def _bkmed_cs(seed=42, lower=3, upper=3, small_cluster_mode=None, min_cluster_size=None, **algo):
    """ClusteringSettings for bisecting_kmedians with optional outlier coupling."""
    overrides = {}
    if small_cluster_mode is not None:
        overrides["small_cluster_mode"] = small_cluster_mode
    if min_cluster_size is not None:
        overrides["min_cluster_size"] = min_cluster_size
    cs = make_clustering_settings(
        "bisecting_kmedians",
        seed=seed,
        bisecting_kmedians={"n_cluster": {"lower": lower, "upper": upper}, **algo},
        **overrides,
    )

    return cs


def _well_separated_blobs(n_per=40, d=5, sep=50.0, seed=0):
    """Three well-separated blobs with ground-truth labels."""
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(np.full(d, i * sep), 1.0, size=(n_per, d)) for i in range(3)]
    data = np.vstack(blocks)
    truth = np.repeat(np.arange(3), n_per)

    return data, truth


class TestBisectKMediansSettings:
    """Per-setting behaviour on well-separated data."""

    @pytest.mark.parametrize("strategy", ["largest_cluster", "biggest_inertia"])
    def test_bisecting_strategy_recovers_blobs(self, strategy):
        """Both split strategies recover three well-separated blobs."""
        data, truth = _well_separated_blobs()
        cs = _bkmed_cs(bisecting_strategy=strategy)
        result = bisect_k_medians(data, cs)
        assert result.k == 3
        assert adjusted_rand_score(truth, result.labels) > 0.99

    @pytest.mark.parametrize("recluster_count", [0, 2, 3])
    def test_recluster_count_paths(self, recluster_count):
        """Single-run (0) and multi-restart (>0) paths both recover blobs."""
        data, truth = _well_separated_blobs()
        cs = _bkmed_cs(recluster_count=recluster_count)
        result = bisect_k_medians(data, cs)
        assert adjusted_rand_score(truth, result.labels) > 0.99

    @pytest.mark.parametrize("refine", [True, False])
    def test_refinement_toggle_recovers_blobs(self, refine):
        """Refinement on/off both recover well-separated blobs."""
        data, truth = _well_separated_blobs()
        cs = _bkmed_cs(refinement_enabled=refine)
        result = bisect_k_medians(data, cs)
        assert adjusted_rand_score(truth, result.labels) > 0.99

    def test_min_cluster_size_limits_reachable_k(self):
        """A large min_cluster_size caps how finely the data can be split.

        40 points with min_cluster_size=15 admit at most 2 clusters of the
        required size, so k is capped at 2 even when up to 6 is requested.
        """
        rng = np.random.default_rng(1)
        data = np.vstack([rng.normal(0, 1, (20, 5)), rng.normal(50, 1, (20, 5))])
        cs = _bkmed_cs(
            lower=2, upper=6,
            min_cluster_size=15, small_cluster_mode="outlier",
        )
        result = bisect_k_medians(data, cs)
        assert result.k == 2


class TestBisectKMediansFailures:
    """Invalid input is rejected with a specific ValueError."""

    def test_too_few_samples_raises(self):
        """n < 2 raises with a clear message."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            bisect_k_medians(np.ones((1, 4)), _bkmed_cs())

    def test_non_finite_raises(self):
        """Non-finite data raises."""
        data, _ = _well_separated_blobs()
        data[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            bisect_k_medians(data, _bkmed_cs())

    def test_more_clusters_than_samples_yields_no_valid_labeling(self):
        """Forcing k far above n leaves the store empty -> ValueError on access."""
        np.random.seed(42)
        data = np.random.randn(5, 10)
        result = bisect_k_medians(data, _bkmed_cs(lower=10, upper=10))
        with pytest.raises(ValueError):
            _ = result.labels


class TestBisectKMediansBaseline:
    """Drift snapshot on deterministic synthetic data."""

    def test_expected_baseline_output(self):
        """Output is pinned against a known-good labeling to catch drift.

        The council prefers k=2 on these moderately-overlapping blobs; the
        partition is correct (matches bisecting k-means on the same data) and
        pinned here so an algorithmic change surfaces as a failure.
        """
        data, _ = make_blobs(
            n_samples=40, n_features=10, centers=3, cluster_std=2.0, random_state=42,
        )
        cs = _bkmed_cs(
            lower=2, upper=4,
            recluster_count=2, bisecting_strategy="largest_cluster",
            refinement_enabled=False,
        )

        labels = bisect_k_medians(data, cs).labels

        expected = np.array([
            0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
            1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 1, 1, 1, 0, 0,
        ])
        np.testing.assert_array_equal(labels, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
