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

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from opendsm.common.clustering.algorithms.bisect_k_means import bisect_k_means
from opendsm.common.clustering.settings import ClusteringSettings
from opendsm.common.clustering.algorithms.settings import BisectingKMeansSettings

from .conftest import make_clustering_settings


def _bkm_cs(algo_settings=None, seed=42):
    """Build ClusteringSettings for bisecting_kmeans tests."""
    if algo_settings is None:
        algo_settings = {}
    elif hasattr(algo_settings, 'model_dump'):
        algo_settings = algo_settings.model_dump(exclude_defaults=True)
    return make_clustering_settings("bisecting_kmeans", seed=seed, bisecting_kmeans=algo_settings)



def get_default_settings_dict():
    """Return a default settings dictionary that can be modified."""
    return {
        "algorithm_selection": "bisecting_kmeans",
        "seed": 42,
    }


def _make_bisect_settings(settings_dict: dict) -> BisectingKMeansSettings:
    """Extract BisectingKMeansSettings from a ClusteringSettings-style dict."""
    bk = settings_dict.get("bisecting_kmeans", {})
    return BisectingKMeansSettings(**bk)


# simple_2d_data fixture is provided by conftest.py


@pytest.fixture
def default_settings():
    """Create default bisecting k-means ClusteringSettings."""
    return _bkm_cs()


@pytest.fixture
def custom_bisect_settings():
    """Create custom bisecting k-means settings."""
    return _bkm_cs(BisectingKMeansSettings(
        recluster_count=2,
        internal_recluster_count=3,
        n_cluster={
            "lower": 2,
            "upper": 5
        }
    ))


# ---------------------------------------------------------------------------
# Shared behaviour (valid labels, determinism, cluster range, data shapes,
# edge cases, cluster quality) is tested in test_algorithms.py via
# TestAlgorithmSharedBehavior parametrized over all algorithms.
# ---------------------------------------------------------------------------


class TestAlgorithmSettings:
    """Tests for different algorithm configuration settings."""

    @pytest.mark.parametrize("inner_algorithm", ["lloyd", "elkan"])
    def test_inner_algorithm(self, simple_2d_data, inner_algorithm):
        """Test clustering with different inner algorithms."""
        settings_dict = get_default_settings_dict()
        settings_dict["bisecting_kmeans"] = {
                "inner_algorithm": inner_algorithm,
                "n_cluster": {"lower": 3, "upper": 3}
        }
        settings = _make_bisect_settings(settings_dict)

        labels = bisect_k_means(simple_2d_data, _bkm_cs(settings, seed=42)).labels
        assert len(np.unique(labels)) == 3

    @pytest.mark.parametrize("bisecting_strategy", ["largest_cluster", "biggest_inertia"])
    def test_bisecting_strategy(self, simple_2d_data, bisecting_strategy):
        """Test clustering with different bisecting strategies."""
        settings_dict = get_default_settings_dict()
        settings_dict["bisecting_kmeans"] = {
                "bisecting_strategy": bisecting_strategy,
                "n_cluster": {"lower": 3, "upper": 3}
        }
        settings = _make_bisect_settings(settings_dict)

        labels = bisect_k_means(simple_2d_data, _bkm_cs(settings, seed=42)).labels
        assert len(np.unique(labels)) == 3

    @pytest.mark.parametrize("recluster_count", [1, 3, 5])
    def test_recluster_count(self, simple_2d_data, recluster_count):
        """Test that different recluster counts work correctly."""
        settings_dict = get_default_settings_dict()
        settings_dict["bisecting_kmeans"] = {
            "recluster_count": recluster_count,
            "n_cluster": {"lower": 3, "upper": 3}
        }
        settings = _make_bisect_settings(settings_dict)

        labels = bisect_k_means(simple_2d_data, _bkm_cs(settings, seed=42)).labels
        assert len(np.unique(labels)) == 3


# TestDataShapes, TestEdgeCases, TestClusterQuality are now in
# test_algorithms.py::TestAlgorithmSharedBehavior.

class TestBisectKMeansEdgeCases:
    """Bisecting k-means specific edge cases not covered by the shared suite."""

    def test_more_clusters_than_samples(self):
        """Test clustering with more clusters requested than samples available."""
        np.random.seed(42)
        data = np.random.randn(5, 10)

        settings_dict = get_default_settings_dict()
        settings_dict["bisecting_kmeans"] = {
                "n_cluster": {"lower": 10, "upper": 10}
        }
        settings = _make_bisect_settings(settings_dict)

        # Should raise ValueError due to insufficient samples
        with pytest.raises(ValueError, match="Insufficient samples for clustering"):
            bisect_k_means(data, _bkm_cs(settings, seed=42))


class TestBaselineConsistency:
    """Tests to ensure algorithm output doesn't change across versions."""

    def test_expected_baseline_output(self):
        """Test that bisecting k-means produces expected baseline output.

        This test ensures the algorithm produces consistent results across
        different versions of the code. If this test fails, it indicates
        a breaking change in the clustering algorithm.
        """
        # Create deterministic test data with well-separated clusters
        data, _ = make_blobs(
            n_samples=40,
            n_features=10,
            centers=3,
            cluster_std=2.0,
            random_state=42
        )

        # Configure settings for reproducible clustering
        settings_dict = get_default_settings_dict()
        algo_settings = BisectingKMeansSettings(
            n_cluster={"lower": 2, "upper": 4},
            recluster_count=2,
            internal_recluster_count=3,
            inner_algorithm="lloyd",
            bisecting_strategy="largest_cluster",
        )

        # Run clustering
        labels = bisect_k_means(data, _bkm_cs(algo_settings, seed=42)).labels

        # Expected baseline output - saved for version consistency
        expected_labels = np.array([
            0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
            1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
            1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 1, 1, 1, 0, 0
        ])

        # Verify exact match against baseline
        np.testing.assert_array_equal(
            labels,
            expected_labels,
            err_msg="Bisecting k-means output does not match saved baseline. "
                    "This indicates a breaking change in the algorithm."
        )

        # Verify cluster properties
        unique_labels, counts = np.unique(labels, return_counts=True)
        expected_counts = {0: 27, 1: 13}

        assert len(unique_labels) == 2
        for label, count in zip(unique_labels, counts):
            assert count == expected_counts[label], \
                f"Cluster {label} has {count} samples, expected {expected_counts[label]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
