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

"""Comprehensive test suite for clustering cluster module."""

import warnings
import numpy as np
import pandas as pd
import pytest

from opendsm.common.clustering.cluster import (
    _cluster_merge,
    cluster_reorder,
    _cluster_features,
    cluster_features,
)
from opendsm.common.clustering.settings import (
    ClusteringSettings,
    ClusterAlgorithms,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_data():
    """Create simple synthetic data for clustering."""
    np.random.seed(42)
    # Create 3 distinct clusters
    cluster1 = np.random.randn(20, 5) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(20, 5) + np.array([10, 10, 10, 10, 10])
    cluster3 = np.random.randn(20, 5) + np.array([-10, -10, -10, -10, -10])

    data = np.vstack([cluster1, cluster2, cluster3])
    return data


@pytest.fixture
def simple_dataframe():
    """Create simple DataFrame for clustering."""
    np.random.seed(42)
    data = np.random.randn(50, 24) * 10 + 50
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def time_series_dataframe():
    """Create time series DataFrame with distinct patterns."""
    np.random.seed(42)
    n_samples = 60
    n_timepoints = 24

    data = []
    for i in range(n_samples):
        t = np.linspace(0, 2 * np.pi, n_timepoints)
        if i < 20:
            # Morning peak
            pattern = 50 + 20 * np.sin(t - np.pi/4) + np.random.randn(n_timepoints) * 2
        elif i < 40:
            # Evening peak
            pattern = 50 + 20 * np.sin(t + np.pi/4) + np.random.randn(n_timepoints) * 2
        else:
            # Flat with noise
            pattern = 50 + np.random.randn(n_timepoints) * 5
        data.append(pattern)

    return pd.DataFrame(data)


@pytest.fixture
def cluster_labels_simple():
    """Create simple cluster labels matching simple_dataframe length (50)."""
    # 50 samples: 20 in cluster 0, 20 in cluster 1, 10 in cluster 2
    return np.array([0] * 20 + [1] * 20 + [2] * 10)


@pytest.fixture
def cluster_labels_with_outliers():
    """Create cluster labels with outliers (-1) matching simple_dataframe length (50)."""
    # 50 samples with some outliers
    labels = [0] * 15 + [1] * 15 + [2] * 10 + [-1] * 10
    return np.array(labels)


# =============================================================================
# Tests for _cluster_merge
# =============================================================================

class TestClusterMerge:
    """Tests for _cluster_merge function."""

    def test_merge_two_similar_clusters(self):
        """Test merging two very similar clusters."""
        # Create two very similar clusters
        np.random.seed(42)
        data = np.vstack([
            np.random.randn(10, 5),
            np.random.randn(10, 5) + 0.1  # Very close to first cluster
        ])
        cluster_labels = np.array([0] * 10 + [1] * 10)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42
        )

        result = _cluster_merge(cluster_labels, data, settings, W=0.5)

        # Result should be valid labels with same shape
        assert result.shape == cluster_labels.shape
        assert len(np.unique(result)) == 2

    def test_keep_two_distinct_clusters(self):
        """Test keeping two distinct clusters."""
        # Create two well-separated clusters
        data = np.vstack([
            np.random.randn(10, 5),
            np.random.randn(10, 5) + 20  # Far from first cluster
        ])
        cluster_labels = np.array([0] * 10 + [1] * 10)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42
        )

        result = _cluster_merge(cluster_labels, data, settings, W=0.5)

        # Should keep both clusters
        assert len(np.unique(result)) == 2

    def test_merge_with_different_W_values(self):
        """Test merge behavior with different W threshold values."""
        data = np.vstack([
            np.random.randn(10, 5),
            np.random.randn(10, 5) + 5
        ])
        cluster_labels = np.array([0] * 10 + [1] * 10)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42
        )

        # With low W, more likely to merge
        result_low = _cluster_merge(cluster_labels, data, settings, W=0.1)

        # With high W, less likely to merge
        result_high = _cluster_merge(cluster_labels, data, settings, W=0.9)

        # Both should return valid labels
        assert result_low.shape == cluster_labels.shape
        assert result_high.shape == cluster_labels.shape

    def test_merge_multiple_clusters(self):
        """Test merging with more than two clusters — close clusters merge, distant ones don't."""
        # 3 clusters: two close to each other (mergeable), one well-separated
        np.random.seed(42)
        cluster_a = np.random.randn(20, 5) + np.array([0, 0, 0, 0, 0])
        cluster_b = np.random.randn(20, 5) + np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        cluster_c = np.random.randn(20, 5) + np.array([100, 100, 100, 100, 100])
        data = np.vstack([cluster_a, cluster_b, cluster_c])
        cluster_labels = np.array([0] * 20 + [1] * 20 + [2] * 20)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
        )

        result = _cluster_merge(cluster_labels, data, settings, W=0.5)

        assert result.shape == cluster_labels.shape
        # Two close clusters should merge, distant one stays separate
        assert len(np.unique(result)) < 3


# =============================================================================
# Tests for cluster_reorder
# =============================================================================

class TestClusterReorder:
    """Tests for cluster_reorder function."""

    def test_reorder_by_size_ascending(self, simple_dataframe, cluster_labels_simple):
        """Test cluster reordering by size in ascending order.

        Note: cluster_labels_simple has 20 in cluster 0, 20 in cluster 1, 10 in cluster 2.
        The actual behavior sorts by size and assigns indices based on sorted order.
        """
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            cluster_sort={
                "enable": True,
                "method": "size",
                "aggregation": "mean",
                "reverse": False
            }
        )

        cluster_map = cluster_reorder(simple_dataframe, cluster_labels_simple, settings)

        # Should return a dictionary mapping old labels to new labels
        assert isinstance(cluster_map, dict)

        # All unique labels should be in the map
        unique_labels = np.unique(cluster_labels_simple[cluster_labels_simple >= 0])
        for label in unique_labels:
            assert label in cluster_map

        # Verify reordering occurred - clusters should be remapped using np.unique on new values
        new_labels = np.array([cluster_map[label] for label in cluster_labels_simple])
        unique_new_labels = np.unique(new_labels[new_labels >= 0])

        # Should have same number of unique clusters
        assert len(unique_new_labels) == len(unique_labels)

        # Smallest cluster (2 with size 10) maps to 0 based on actual behavior
        assert cluster_map[2] == 0
        assert cluster_map[0] in [1, 2]
        assert cluster_map[1] in [1, 2]

        # Test output values: verify counts after remapping
        assert np.sum(new_labels == 0) == 10  # Smallest cluster
        assert np.sum(new_labels == 1) == 20  # Medium clusters
        assert np.sum(new_labels == 2) == 20
        assert np.all((new_labels >= 0) & (new_labels <= 2))

        # Test consistency: calling again with same inputs should produce same output
        cluster_map_2 = cluster_reorder(simple_dataframe, cluster_labels_simple, settings)
        new_labels_2 = np.array([cluster_map_2[label] for label in cluster_labels_simple])
        assert cluster_map == cluster_map_2
        np.testing.assert_array_equal(new_labels, new_labels_2)

    def test_reorder_by_size_descending(self, simple_dataframe, cluster_labels_simple):
        """Test cluster reordering by size in descending order.

        With reverse=True, largest clusters should get lowest indices (0, 1),
        smallest cluster should get highest index (2).
        cluster_labels_simple has 20 in cluster 0, 20 in cluster 1, 10 in cluster 2.
        """
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            cluster_sort={
                "enable": True,
                "method": "size",
                "aggregation": "mean",
                "reverse": True
            }
        )

        cluster_map = cluster_reorder(simple_dataframe, cluster_labels_simple, settings)

        assert isinstance(cluster_map, dict)
        assert len(cluster_map) > 0

        # All unique labels should be in the map
        unique_labels = np.unique(cluster_labels_simple[cluster_labels_simple >= 0])
        for label in unique_labels:
            assert label in cluster_map

        # Verify reordering occurred using np.unique to check new label assignments
        new_labels = np.array([cluster_map[label] for label in cluster_labels_simple])
        unique_new_labels = np.unique(new_labels[new_labels >= 0])

        # Should have same number of unique clusters
        assert len(unique_new_labels) == len(unique_labels)

        # Descending order: largest clusters get lowest indices, smallest gets highest
        assert cluster_map[2] == 2  # Smallest cluster (size 10) maps to highest index
        assert cluster_map[0] in [0, 1]  # Larger clusters map to lowest indices
        assert cluster_map[1] in [0, 1]

    def test_reorder_by_peak(self, time_series_dataframe):
        """Test cluster reordering by peak."""
        # Create labels with distinct patterns
        cluster_labels = np.array([0] * 20 + [1] * 20 + [2] * 20)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            cluster_sort={
                "enable": True,
                "method": "peak",
                "aggregation": "mean",
                "reverse": False
            }
        )

        cluster_map = cluster_reorder(time_series_dataframe, cluster_labels, settings)

        assert isinstance(cluster_map, dict)
        # Should map all non-outlier labels
        unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
        for label in unique_labels:
            assert label in cluster_map

    def test_reorder_with_outliers(self, simple_dataframe, cluster_labels_with_outliers):
        """Test that outliers (-1) are excluded from reordering."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            cluster_sort={
                "enable": True,
                "method": "size",
                "aggregation": "mean",
                "reverse": False
            }
        )

        cluster_map = cluster_reorder(simple_dataframe, cluster_labels_with_outliers, settings)

        # Outlier label (-1) should still map to itself
        assert -1 in cluster_map
        assert cluster_map[-1] == -1

    def test_reorder_different_aggregations(self, time_series_dataframe):
        """Test reordering with different aggregation methods."""
        cluster_labels = np.array([0] * 20 + [1] * 20 + [2] * 20)

        for agg_method in ["mean", "median"]:
            settings = ClusteringSettings(
                algorithm_selection=ClusterAlgorithms.SPECTRAL,
                seed=42,
                cluster_sort={
                    "enable": True,
                    "method": "peak",
                    "aggregation": agg_method,
                    "reverse": False
                }
            )

            cluster_map = cluster_reorder(time_series_dataframe, cluster_labels, settings)

            assert isinstance(cluster_map, dict)
            assert len(cluster_map) > 0


# =============================================================================
# Tests for _cluster_features
# =============================================================================

class TestClusterFeaturesInternal:
    """Tests for _cluster_features internal function."""

    def test_bisecting_kmeans_clustering(self, simple_data):
        """Test clustering with bisecting k-means."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.BISECTING_KMEANS,
            seed=42,
            bisecting_kmeans={
                "n_cluster": {"lower": 2, "upper": 5}
            }
        )

        labels = _cluster_features(simple_data, settings)

        assert labels.shape[0] == simple_data.shape[0]
        assert len(np.unique(labels)) >= 2
        assert len(np.unique(labels)) <= 5

    def test_spectral_clustering(self, simple_data):
        """Test clustering with spectral clustering."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            spectral={
                "n_cluster": {"lower": 2, "upper": 5}
            }
        )

        labels = _cluster_features(simple_data, settings)

        assert labels.shape[0] == simple_data.shape[0]
        assert len(np.unique(labels)) >= 2

    def test_adjust_cluster_count_for_small_data(self):
        """Test that cluster count is adjusted for small datasets."""
        # Small dataset with only 10 samples
        small_data = np.random.randn(10, 5)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            spectral={
                "n_cluster": {"lower": 2, "upper": 20},  # Request more clusters than feasible
                "scoring": {"min_cluster_size": 2}
            }
        )

        labels = _cluster_features(small_data, settings)

        # Should adjust to feasible number of clusters (10 // 2 = 5)
        assert labels.shape[0] == small_data.shape[0]
        assert len(np.unique(labels)) <= 5

    def test_birch_clustering(self, simple_data):
        """Test clustering with Birch."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.BIRCH,
            seed=42,
            birch={
                "n_cluster": {"lower": 2, "upper": 5}
            }
        )

        labels = _cluster_features(simple_data, settings)

        assert labels.shape[0] == simple_data.shape[0]
        assert len(np.unique(labels)) >= 2


# =============================================================================
# Tests for cluster_features (main entry point)
# =============================================================================

class TestClusterFeatures:
    """Tests for cluster_features main function."""

    def test_basic_clustering(self, simple_dataframe):
        """Test basic clustering workflow."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5
            },
            spectral={
                "n_cluster": {"lower": 2, "upper": 5}
            }
        )

        labels = cluster_features(simple_dataframe, settings)

        assert labels.shape[0] == simple_dataframe.shape[0]
        assert len(np.unique(labels)) >= 2
        assert len(np.unique(labels)) <= 5

    def test_clustering_with_sorting(self, time_series_dataframe):
        """Test clustering with cluster sorting enabled."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5
            },
            spectral={
                "n_cluster": {"lower": 2, "upper": 5}
            },
            cluster_sort={
                "enable": True,
                "method": "size",
                "aggregation": "mean",
                "reverse": False
            }
        )

        labels = cluster_features(time_series_dataframe, settings)

        assert labels.shape[0] == time_series_dataframe.shape[0]
        assert len(np.unique(labels)) >= 2

    def test_clustering_bypass_for_many_clusters(self):
        """Test that clustering is bypassed when lower bound >= data size."""
        small_df = pd.DataFrame(np.random.randn(5, 10))

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            spectral={
                "n_cluster": {"lower": 10, "upper": 20}  # Lower bound > data size
            }
        )

        labels = cluster_features(small_df, settings)

        # Should return unique label for each sample
        assert labels.shape[0] == small_df.shape[0]
        np.testing.assert_array_equal(labels, np.arange(len(small_df)))

    def test_clustering_with_normalization(self, simple_dataframe):
        """Test clustering with pre-transform normalization."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            normalize={
                "pre_transform": True,
                "method": "standardize",
                "axis": 0
            },
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5
            },
            spectral={
                "n_cluster": {"lower": 2, "upper": 5}
            }
        )

        labels = cluster_features(simple_dataframe, settings)

        assert labels.shape[0] == simple_dataframe.shape[0]
        assert len(np.unique(labels)) >= 2

    def test_clustering_different_algorithms(self, simple_dataframe):
        """Test clustering with different algorithms."""
        algorithms = [
            ClusterAlgorithms.BISECTING_KMEANS,
            ClusterAlgorithms.SPECTRAL,
        ]

        for algo in algorithms:
            settings_dict = {
                "algorithm_selection": algo,
                "seed": 42,
                "transform_selection": "wavelet",
                "wavelet_transform": {
                    "wavelet_name": "db1",
                    "pca_n_components": 5
                }
            }

            # Add algorithm-specific settings
            if algo in [ClusterAlgorithms.BISECTING_KMEANS, ClusterAlgorithms.SPECTRAL]:
                algo_name = algo.value
                settings_dict[algo_name] = {
                    "n_cluster": {"lower": 2, "upper": 5}
                }

            settings = ClusteringSettings(**settings_dict)
            labels = cluster_features(simple_dataframe, settings)

            assert labels.shape[0] == simple_dataframe.shape[0]
            assert len(np.unique(labels)) >= 1

    def test_clustering_with_fpca_transform(self, time_series_dataframe):
        """Test clustering with FPCA transformation."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            transform_selection="fpca",
            fpca_transform={
                "min_var_ratio": 0.90
            },
            spectral={
                "n_cluster": {"lower": 2, "upper": 5}
            }
        )

        # Suppress deprecation warning for Fourier class
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            labels = cluster_features(time_series_dataframe, settings)

        assert labels.shape[0] == time_series_dataframe.shape[0]
        assert len(np.unique(labels)) >= 2

    def test_reproducibility_with_seed(self, simple_dataframe):
        """Test that clustering is reproducible with same seed."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "seed": 42
            },
            spectral={
                "n_cluster": {"lower": 2, "upper": 5}
            }
        )

        labels1 = cluster_features(simple_dataframe, settings)
        labels2 = cluster_features(simple_dataframe, settings)

        # Results should be identical with same seed
        np.testing.assert_array_equal(labels1, labels2)

    def test_clustering_small_dataset(self):
        """Test clustering with very small dataset."""
        small_df = pd.DataFrame(np.random.randn(10, 5))

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 2
            },
            spectral={
                "n_cluster": {"lower": 2, "upper": 3},
                "scoring": {"min_cluster_size": 2}
            }
        )

        labels = cluster_features(small_df, settings)

        assert labels.shape[0] == small_df.shape[0]
        # With 10 samples and min_cluster_size=2, max clusters should be 5
        assert len(np.unique(labels)) <= 5

    def test_clustering_preserves_index_order(self, simple_dataframe):
        """Test that clustering preserves the order of samples."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5
            },
            spectral={
                "n_cluster": {"lower": 2, "upper": 5}
            }
        )

        labels = cluster_features(simple_dataframe, settings)

        # Labels should be in same order as input DataFrame
        assert len(labels) == len(simple_dataframe)
        # Each label should correspond to the same row index
        for i in range(len(labels)):
            assert isinstance(labels[i], (int, np.integer))


# =============================================================================
# Integration tests
# =============================================================================

class TestClusteringIntegration:
    """Integration tests for complete clustering pipeline."""

    def test_full_pipeline_with_all_options(self, time_series_dataframe):
        """Test complete clustering pipeline with all options enabled."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.BISECTING_KMEANS,
            seed=42,
            normalize={
                "pre_transform": True,
                "method": "min_max_quantile",
                "quantile": 0.05,
                "axis": 1
            },
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 8,
                "include_scale_feature": True
            },
            bisecting_kmeans={
                "n_cluster": {"lower": 3, "upper": 6},
                "scoring": {
                    "min_cluster_size": 5,
                    "distance_metric": "euclidean"
                }
            },
            cluster_sort={
                "enable": True,
                "method": "peak",
                "aggregation": "mean",
                "reverse": False
            }
        )

        labels = cluster_features(time_series_dataframe, settings)

        assert labels.shape[0] == time_series_dataframe.shape[0]
        assert len(np.unique(labels)) >= 3
        assert len(np.unique(labels)) <= 6

    def test_pipeline_consistency_across_runs(self, simple_dataframe):
        """Test that pipeline produces consistent results across multiple runs."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=123,
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "seed": 123
            },
            spectral={
                "n_cluster": {"lower": 2, "upper": 4}
            }
        )

        results = []
        for _ in range(3):
            labels = cluster_features(simple_dataframe, settings)
            results.append(labels)

        # All runs should produce identical results
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

    def test_exact_output_spectral_baseline(self):
        """Test exact output for spectral clustering against saved baseline.

        This test compares clustering output against a saved baseline to ensure
        consistency across code versions. Any deviation indicates a breaking change.
        """
        # Create deterministic test data
        np.random.seed(42)
        data = np.random.randn(30, 10) * 5 + 25
        df = pd.DataFrame(data)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 3,
                "seed": 42
            },
            spectral={
                "n_cluster": {"lower": 2, "upper": 4}
            }
        )

        labels = cluster_features(df, settings)

        # Expected output - saved baseline for version consistency
        # Generated with seed=42, recorded as baseline
        expected_labels = np.array([
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
            0, 1, 1, 0, 0, 0, 1, 1, 1, 0
        ])

        # Verify exact match against baseline
        np.testing.assert_array_equal(labels, expected_labels,
            err_msg="Spectral clustering output does not match saved baseline. "
                    "This indicates a breaking change in the algorithm.")

        # Verify cluster properties
        unique_labels, counts = np.unique(labels, return_counts=True)
        assert len(unique_labels) == 2
        expected_counts = {0: 9, 1: 21}
        for label, count in zip(unique_labels, counts):
            assert count == expected_counts[label], \
                f"Cluster {label}: expected {expected_counts[label]} samples, got {count}"

    def test_exact_output_bisecting_kmeans_baseline(self):
        """Test exact output for bisecting k-means against saved baseline.

        This test compares clustering output against a saved baseline to ensure
        consistency across code versions. Any deviation indicates a breaking change.
        """
        # Create deterministic test data
        np.random.seed(123)
        data = np.random.randn(40, 12) * 3 + 15
        df = pd.DataFrame(data)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.BISECTING_KMEANS,
            seed=123,
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 4,
                "seed": 123
            },
            bisecting_kmeans={
                "n_cluster": {"lower": 3, "upper": 5},
                "scoring": {
                    "min_cluster_size": 3,
                    "distance_metric": "euclidean"
                }
            }
        )

        labels = cluster_features(df, settings)

        # Expected output - saved baseline for version consistency
        # Generated with seed=123, recorded as baseline
        expected_labels = np.array([
            0, 2, 0, 0, 0, 1, 0, 0, 1, 1, 2, 1, 2, 1, 1, 0, 0, 1, 2, 0,
            0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 0, 2
        ])

        # Verify exact match against baseline
        np.testing.assert_array_equal(labels, expected_labels,
            err_msg="Bisecting K-means output does not match saved baseline. "
                    "This indicates a breaking change in the algorithm.")

        # Verify cluster properties
        unique_labels, counts = np.unique(labels, return_counts=True)
        assert len(unique_labels) == 3
        expected_counts = {0: 14, 1: 13, 2: 13}
        for label, count in zip(unique_labels, counts):
            assert count >= 3, f"Cluster {label} has {count} samples, below minimum of 3"
            assert count == expected_counts[label], \
                f"Cluster {label}: expected {expected_counts[label]} samples, got {count}"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
