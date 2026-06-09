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
    _build_label_remap,
    _cluster_features,
    cluster_features,
)
from opendsm.common.clustering.metrics.label_ops import (
    assign_small_clusters_outlier,
    assign_small_clusters_nearest,
)
from opendsm.common.clustering.settings import (
    ClusteringSettings,
    ClusterAlgorithms,
)

from .conftest import make_clustering_settings


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
# Tests for _build_label_remap (formerly cluster_reorder)
# =============================================================================

class TestClusterReorder:
    """Tests for cluster_reorder function."""

    @pytest.mark.parametrize("reverse, expected_smallest_map, expected_smallest_count_at", [
        (False, 0, 0),   # ascending: smallest cluster (size 10) maps to 0
        (True, 2, 2),    # descending: smallest cluster (size 10) maps to 2
    ])
    def test_reorder_by_size(self, cluster_labels_simple, reverse,
                             expected_smallest_map, expected_smallest_count_at):
        """Test cluster reordering by size in ascending and descending order.

        cluster_labels_simple has 20 in cluster 0, 20 in cluster 1, 10 in cluster 2.
        """
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            cluster_sort={
                "enable": True,
                "method": "size",
                "aggregation": "mean",
                "reverse": reverse
            }
        )

        cluster_map = _build_label_remap(cluster_labels_simple, settings)

        # All unique labels should be in the map
        unique_labels = np.unique(cluster_labels_simple[cluster_labels_simple >= 0])
        for label in unique_labels:
            assert label in cluster_map

        # Verify reordering: same number of unique clusters
        new_labels = np.array([cluster_map[label] for label in cluster_labels_simple])
        unique_new_labels = np.unique(new_labels[new_labels >= 0])
        assert len(unique_new_labels) == len(unique_labels)

        # Smallest cluster (2, size 10) maps to expected position
        assert cluster_map[2] == expected_smallest_map
        # Larger clusters map to the remaining positions
        other_positions = {0, 1, 2} - {expected_smallest_map}
        assert cluster_map[0] in other_positions
        assert cluster_map[1] in other_positions

        if not reverse:
            # Ascending: verify counts after remapping
            assert np.sum(new_labels == 0) == 10  # Smallest cluster
            assert np.sum(new_labels == 1) == 20
            assert np.sum(new_labels == 2) == 20
            assert np.all((new_labels >= 0) & (new_labels <= 2))

            # Consistency: calling again with same inputs should produce same output
            cluster_map_2 = _build_label_remap(cluster_labels_simple, settings)
            new_labels_2 = np.array([cluster_map_2[label] for label in cluster_labels_simple])
            assert cluster_map == cluster_map_2
            np.testing.assert_array_equal(new_labels, new_labels_2)

    @pytest.mark.parametrize("aggregation", ["mean", "median"])
    def test_reorder_by_peak_raises(self, aggregation):
        """Test cluster reordering by peak raises NotImplementedError."""
        cluster_labels = np.array([0] * 20 + [1] * 20 + [2] * 20)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            cluster_sort={
                "enable": True,
                "method": "peak",
                "aggregation": aggregation,
                "reverse": False
            }
        )

        with pytest.raises(NotImplementedError):
            _build_label_remap(cluster_labels, settings)

    def test_reorder_with_outliers(self, cluster_labels_with_outliers):
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

        cluster_map = _build_label_remap(cluster_labels_with_outliers, settings)

        # Outlier label (-1) should still map to itself
        assert -1 in cluster_map
        assert cluster_map[-1] == -1


# =============================================================================
# Tests for _cluster_features
# =============================================================================

class TestClusterFeaturesInternal:
    """Tests for _cluster_features internal function."""

    @pytest.mark.parametrize("algorithm, algo_settings_key", [
        (ClusterAlgorithms.BISECTING_KMEANS, "bisecting_kmeans"),
        (ClusterAlgorithms.SPECTRAL, "spectral"),
        (ClusterAlgorithms.BIRCH, "birch"),
    ])
    def test_algorithm_clustering(self, simple_data, algorithm, algo_settings_key):
        """Test clustering with each supported algorithm."""
        settings = ClusteringSettings(
            algorithm_selection=algorithm,
            seed=42,
            **{algo_settings_key: {"n_cluster": {"lower": 2, "upper": 5}}}
        )

        lbl = _cluster_features(simple_data, settings)
        labels = lbl.labels

        assert labels.shape[0] == simple_data.shape[0]
        assert len(np.unique(labels)) >= 2
        assert len(np.unique(labels)) <= 5

    def test_adjust_cluster_count_for_small_data(self):
        """Test that cluster count is adjusted for small datasets."""
        # Small dataset with only 10 samples
        small_data = np.random.randn(10, 5)

        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            min_cluster_size=3,
            small_cluster_mode="outlier",
            spectral={
                "n_cluster": {"lower": 2, "upper": 20},
            }
        )

        lbl = _cluster_features(small_data, settings)
        labels = lbl.labels

        # Should adjust to feasible number of clusters (10 // 2 = 5)
        assert labels.shape[0] == small_data.shape[0]
        assert len(np.unique(labels)) <= 5


# =============================================================================
# Tests for cluster_features (main entry point)
# =============================================================================

def _wavelet_spectral_settings(**overrides):
    """Helper to build common wavelet+spectral ClusteringSettings."""
    defaults = dict(
        transform_selection="wavelet",
        feature_transform={"wavelet": {
            "wavelet_name": "db1",
            "pca_n_components": 5
        }},
        spectral={
            "n_cluster": {"lower": 2, "upper": 5}
        },
        cluster_sort={
            "enable": True,
            "method": "size",
        }
    )
    defaults.update(overrides)
    return make_clustering_settings("spectral", seed=42, **defaults)


class TestClusterFeatures:
    """Tests for cluster_features main function."""

    def test_basic_clustering(self, simple_dataframe):
        """Test basic clustering workflow."""
        settings = _wavelet_spectral_settings()
        labels = cluster_features(simple_dataframe, settings)

        assert labels.shape[0] == simple_dataframe.shape[0]
        assert len(np.unique(labels)) >= 2
        assert len(np.unique(labels)) <= 5

    def test_clustering_with_sorting(self, time_series_dataframe):
        """Test clustering with cluster sorting enabled."""
        settings = _wavelet_spectral_settings(
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
        settings = _wavelet_spectral_settings(
            normalize={
                "pre_transform": True,
                "method": "standardize",
                "axis": 0
            }
        )
        labels = cluster_features(simple_dataframe, settings)

        assert labels.shape[0] == simple_dataframe.shape[0]
        assert len(np.unique(labels)) >= 2

    @pytest.mark.parametrize("algorithm", [
        ClusterAlgorithms.BISECTING_KMEANS,
        ClusterAlgorithms.SPECTRAL,
    ])
    def test_clustering_different_algorithms(self, simple_dataframe, algorithm):
        """Test clustering with different algorithms."""
        algo_name = algorithm.value
        settings = ClusteringSettings(
            algorithm_selection=algorithm,
            seed=42,
            transform_selection="wavelet",
            wavelet_transform={
                "wavelet_name": "db1",
                "pca_n_components": 5
            },
            cluster_sort={
                "enable": True,
                "method": "size",
            },
            **{algo_name: {"n_cluster": {"lower": 2, "upper": 5}}}
        )

        labels = cluster_features(simple_dataframe, settings)

        assert labels.shape[0] == simple_dataframe.shape[0]
        assert len(np.unique(labels)) >= 1

    def test_clustering_with_fpca_transform(self, time_series_dataframe):
        """Test clustering with FPCA transformation."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=42,
            transform_selection="fpca",
            feature_transform={"fpca": {
                "min_var_ratio": 0.90
            }},
            spectral={
                "n_cluster": {"lower": 2, "upper": 5}
            },
            cluster_sort={
                "enable": True,
                "method": "size",
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
        settings = _wavelet_spectral_settings(
            feature_transform={"wavelet": {
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "seed": 42
            }}
        )

        labels1 = cluster_features(simple_dataframe, settings)
        labels2 = cluster_features(simple_dataframe, settings)

        # Results should be identical with same seed
        np.testing.assert_array_equal(labels1, labels2)

    def test_clustering_small_dataset(self):
        """Test clustering with very small dataset."""
        small_df = pd.DataFrame(np.random.randn(10, 5))

        settings = _wavelet_spectral_settings(
            min_cluster_size=3,
            small_cluster_mode="outlier",
            feature_transform={"wavelet": {
                "wavelet_name": "db1",
                "pca_n_components": 2
            }},
            spectral={
                "n_cluster": {"lower": 2, "upper": 3},
            }
        )

        labels = cluster_features(small_df, settings)

        assert labels.shape[0] == small_df.shape[0]
        # With 10 samples and min_cluster_size=3, max clusters should be 3
        assert len(np.unique(labels)) <= 3

    def test_clustering_preserves_sample_count(self, simple_dataframe):
        """Test that clustering returns one integer label per sample."""
        settings = _wavelet_spectral_settings()
        labels = cluster_features(simple_dataframe, settings)

        assert len(labels) == len(simple_dataframe)
        assert labels.dtype.kind == "i"  # integer dtype


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
            feature_transform={"wavelet": {
                "wavelet_name": "db1",
                "pca_n_components": 8,
                "include_scale_feature": True
            }},
            min_cluster_size=5,
            small_cluster_mode="outlier",
            bisecting_kmeans={
                "n_cluster": {"lower": 3, "upper": 6},
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
        assert len(np.unique(labels)) >= 3
        assert len(np.unique(labels)) <= 6

    def test_pipeline_consistency_across_runs(self, simple_dataframe):
        """Test that pipeline produces consistent results across multiple runs."""
        settings = ClusteringSettings(
            algorithm_selection=ClusterAlgorithms.SPECTRAL,
            seed=123,
            transform_selection="wavelet",
            feature_transform={"wavelet": {
                "wavelet_name": "db1",
                "pca_n_components": 5,
                "seed": 123
            }},
            spectral={
                "n_cluster": {"lower": 2, "upper": 4}
            },
            cluster_sort={
                "enable": True,
                "method": "size",
            }
        )

        labels_first = cluster_features(simple_dataframe, settings)
        for _ in range(2):
            labels_next = cluster_features(simple_dataframe, settings)
            np.testing.assert_array_equal(labels_first, labels_next)

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
            outlier_removal_sigma=None,
            transform_selection="wavelet",
            feature_transform={"wavelet": {
                "wavelet_name": "db1",
                "pca_n_components": 3,
                "seed": 42
            }},
            spectral={
                "n_cluster": {"lower": 2, "upper": 4}
            },
            cluster_sort={
                "enable": True,
                "method": "size",
                "reverse": False,
            }
        )

        labels = cluster_features(df, settings)

        # Expected output - saved baseline for version consistency
        expected_labels = np.array([
            1, 2, 2, 2, 2, 1, 2, 0, 1, 1, 1, 0, 0, 2, 2, 1, 1, 0, 2, 2,
            0, 1, 0, 1, 2, 0, 2, 2, 0, 1
        ])

        # Verify exact match against baseline
        np.testing.assert_array_equal(labels, expected_labels,
            err_msg="Spectral clustering output does not match saved baseline. "
                    "This indicates a breaking change in the algorithm.")

        # Verify cluster properties
        unique_labels, counts = np.unique(labels, return_counts=True)
        assert len(unique_labels) == 3
        expected_counts = {0: 8, 1: 10, 2: 12}
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
            outlier_removal_sigma=None,
            min_cluster_size=3,
            small_cluster_mode="outlier",
            transform_selection="wavelet",
            feature_transform={"wavelet": {
                "wavelet_name": "db1",
                "pca_n_components": 4,
                "seed": 123
            }},
            bisecting_kmeans={
                "n_cluster": {"lower": 3, "upper": 5},
            },
            cluster_sort={
                "enable": True,
                "method": "size",
                "reverse": False,
            }
        )

        labels = cluster_features(df, settings)

        # Expected output - saved baseline for version consistency
        expected_labels = np.array([
            1, 0, 1, 2, 0, 3, 1, 3, 3, 3, 2, 0, 2, 3, 2, 3, 2, 2, 1, 2,
            3, 2, 3, 1, 3, 2, 2, 2, 3, 1, 2, 2, 2, 3, 3, 1, 3, 3, 0, 3
        ])

        # Verify exact match against baseline
        np.testing.assert_array_equal(labels, expected_labels,
            err_msg="Bisecting K-means output does not match saved baseline. "
                    "This indicates a breaking change in the algorithm.")

        # Verify cluster properties
        unique_labels, counts = np.unique(labels, return_counts=True)
        assert len(unique_labels) == 4
        expected_counts = {0: 4, 1: 7, 2: 14, 3: 15}
        for label, count in zip(unique_labels, counts):
            assert count == expected_counts[label], \
                f"Cluster {label}: expected {expected_counts[label]} samples, got {count}"


class TestMergeSmallClustersNoMutation:
    """assign_small_clusters_outlier must not mutate the input array."""

    @pytest.mark.parametrize("labels, min_size, check_result", [
        # Basic: input unchanged after call
        (np.array([0, 0, 0, 1, 2, 2, 2, 2]), 2, None),
        # All small: input unchanged after call
        (np.array([0, 1, 2, 3]), 2, None),
    ])
    def test_input_array_unchanged(self, labels, min_size, check_result):
        original = labels.copy()
        assign_small_clusters_outlier(labels, min_cluster_size=min_size)
        np.testing.assert_array_equal(labels, original)

    def test_singleton_not_mutated_in_caller(self):
        labels = np.array([0, 0, 0, 1, 1, 1, 2])
        original = labels.copy()
        result = assign_small_clusters_outlier(labels, min_cluster_size=2)
        np.testing.assert_array_equal(labels, original)
        assert result[-1] == -1

    def test_no_small_clusters(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        original = labels.copy()
        result = assign_small_clusters_outlier(labels, min_cluster_size=2)
        np.testing.assert_array_equal(labels, original)
        assert (result != -1).all()


# =============================================================================
# Tests for prepare_labels modes
# =============================================================================

from opendsm.common.clustering.metrics.label_ops import prepare_labels
from opendsm.common.clustering.metrics.settings import ScoreSettings, SmallClusterMode


class TestPrepareLabelsModes:
    """Tests for prepare_labels with different SmallClusterMode values and edge cases."""

    def _make_settings(self, mode: SmallClusterMode, min_cluster_size: int | None = None,
                       max_non_outlier_cluster_count: int = 200) -> tuple[ScoreSettings, int, SmallClusterMode]:
        if min_cluster_size is None:
            min_cluster_size = 1 if mode == SmallClusterMode.KEEP else 2
        return (
            ScoreSettings(max_non_outlier_cluster_count=max_non_outlier_cluster_count),
            min_cluster_size,
            mode,
        )

    # ----- ABSORB mode -----

    def test_absorb_reassigns_small_cluster_to_nearest_centroid(self):
        """Small cluster points are absorbed into the nearest large cluster centroid."""
        # Cluster 0: 5 points near origin, cluster 1: 5 points near [10,...],
        # cluster 2: 1 point (small) near [10,...] -- should absorb into cluster 1's centroid.
        data = np.vstack([
            np.zeros((5, 2)),                      # cluster 0 at origin
            np.full((5, 2), 10.0),                  # cluster 1 at (10,10)
            np.full((1, 2), 9.5),                   # cluster 2 (small) near cluster 1
        ])
        labels = np.array([0]*5 + [1]*5 + [2]*1)
        score_settings, min_cs, scm = self._make_settings(SmallClusterMode.ABSORB, min_cluster_size=3)

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        # The small-cluster point should have been absorbed, not turned into -1
        assert -1 not in merged
        # Should have exactly 2 clusters after absorbing
        assert len(np.unique(merged)) == 2
        # The single point (index 10) should now share the label of cluster 1 points
        assert merged[10] == merged[5]
        # data_clean / labels_clean should be valid (not None)
        assert data_clean is not None
        assert labels_clean is not None
        # coverage should be 1.0 since there are no outliers
        assert coverage == 1.0

    def test_absorb_no_small_clusters(self):
        """ABSORB with no small clusters returns all clusters unchanged."""
        data = np.vstack([np.zeros((5, 2)), np.full((5, 2), 10.0)])
        labels = np.array([0]*5 + [1]*5)
        score_settings, min_cs, scm = self._make_settings(SmallClusterMode.ABSORB, min_cluster_size=3)

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        assert len(np.unique(merged)) == 2
        assert data_clean is not None
        assert coverage == 1.0

    def test_absorb_preserves_existing_outliers(self):
        """ABSORB leaves pre-existing -1 outliers unchanged."""
        data = np.vstack([
            np.zeros((5, 2)),
            np.full((5, 2), 10.0),
            np.full((1, 2), 100.0),   # outlier point
        ])
        labels = np.array([0]*5 + [1]*5 + [-1])
        score_settings, min_cs, scm = self._make_settings(SmallClusterMode.ABSORB, min_cluster_size=3)

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        # The -1 outlier should remain -1
        assert merged[10] == -1
        assert coverage == pytest.approx(10.0 / 11.0)

    # ----- KEEP mode -----

    def test_keep_preserves_small_clusters(self):
        """KEEP mode keeps all clusters regardless of size."""
        data = np.vstack([
            np.zeros((5, 2)),
            np.full((5, 2), 10.0),
            np.full((1, 2), 20.0),   # cluster of size 1 -- below min_cluster_size
        ])
        labels = np.array([0]*5 + [1]*5 + [2]*1)
        score_settings, min_cs, scm = self._make_settings(SmallClusterMode.KEEP, min_cluster_size=1)

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        # All 3 clusters should be preserved (no outlier creation)
        assert -1 not in merged
        assert len(np.unique(merged)) == 3
        assert data_clean is not None
        assert labels_clean is not None
        assert coverage == 1.0

    def test_keep_preserves_existing_outliers(self):
        """KEEP mode preserves pre-existing -1 noise labels."""
        data = np.vstack([
            np.zeros((5, 2)),
            np.full((5, 2), 10.0),
            np.full((2, 2), 50.0),   # noise points
        ])
        labels = np.array([0]*5 + [1]*5 + [-1]*2)
        score_settings, min_cs, scm = self._make_settings(SmallClusterMode.KEEP, min_cluster_size=1)

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        # Pre-existing -1 values remain
        assert np.sum(merged == -1) == 2
        assert len(np.unique(labels_clean)) == 2
        assert coverage == pytest.approx(10.0 / 12.0)

    def test_keep_reindexes_contiguously(self):
        """KEEP mode reindexes labels to be contiguous from 0."""
        data = np.ones((9, 2))  # data values don't matter for KEEP
        # Gap in label IDs: 0, 3, 7
        labels = np.array([0]*3 + [3]*3 + [7]*3)
        score_settings, min_cs, scm = self._make_settings(SmallClusterMode.KEEP, min_cluster_size=1)

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        unique = np.unique(merged)
        np.testing.assert_array_equal(unique, np.arange(len(unique)))

    # ----- n_clusters < 1 → returns None for data_clean -----

    def test_all_outliers_returns_none(self):
        """When all points are outliers, n_clusters < 1 so data_clean is None."""
        data = np.ones((5, 2))
        labels = np.array([-1]*5)
        score_settings, min_cs, scm = self._make_settings(SmallClusterMode.KEEP, min_cluster_size=1)

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        assert data_clean is None
        assert labels_clean is None
        assert coverage == 0.0

    def test_outlier_mode_all_small_returns_none(self):
        """OUTLIER mode: if all clusters are below min_cluster_size, everything
        becomes -1 and data_clean should be None."""
        data = np.vstack([np.zeros((2, 2)), np.ones((2, 2))])
        labels = np.array([0, 0, 1, 1])
        score_settings, min_cs, scm = self._make_settings(SmallClusterMode.OUTLIER, min_cluster_size=3)

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        assert np.all(merged == -1)
        assert data_clean is None
        assert labels_clean is None
        assert coverage == 0.0

    # ----- n_clusters exceeds max_non_outlier_cluster_count -----

    def test_too_many_clusters_returns_none(self):
        """When n_clusters > max_non_outlier_cluster_count, data_clean is None."""
        data = np.arange(20).reshape(10, 2).astype(float)
        labels = np.arange(10)  # 10 unique clusters, one point each
        score_settings, min_cs, scm = self._make_settings(
            SmallClusterMode.KEEP,
            min_cluster_size=1,
            max_non_outlier_cluster_count=5,
        )

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        assert data_clean is None
        assert labels_clean is None
        # merged should still be valid (reindexed labels)
        assert len(np.unique(merged)) == 10

    def test_exactly_at_max_clusters_is_valid(self):
        """When n_clusters == max_non_outlier_cluster_count, result is valid."""
        data = np.arange(20).reshape(10, 2).astype(float)
        labels = np.arange(10)
        score_settings, min_cs, scm = self._make_settings(
            SmallClusterMode.KEEP,
            min_cluster_size=1,
            max_non_outlier_cluster_count=10,
        )

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=None,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        assert data_clean is not None
        assert labels_clean is not None

    # ----- n_cluster_lower guard -----

    def test_below_n_cluster_lower_returns_none(self):
        """When n_clusters < n_cluster_lower, data_clean is None."""
        data = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
        labels = np.array([0]*5 + [1]*5)
        score_settings, min_cs, scm = self._make_settings(SmallClusterMode.KEEP, min_cluster_size=1)

        merged, data_clean, labels_clean, coverage = prepare_labels(
            labels, data, score_settings, n_cluster_lower=5,
            min_cluster_size=min_cs, small_cluster_mode=scm,
        )

        assert data_clean is None
        assert labels_clean is None


# =============================================================================
# Tests for remove_outliers_mad
# =============================================================================

from opendsm.common.clustering.metrics.label_ops import remove_outliers_mad
from opendsm.common.stats.basic import MAD_k


def _sigma_to_mad_threshold(sigma: float) -> float:
    """Convert a sigma-based threshold to the MAD-unit threshold expected by
    ``remove_outliers_mad``.  Mirrors ``ClusteringSettings._init_outlier_threshold``.
    """
    return sigma / MAD_k


class TestOutlierRemovalMAD:
    """Tests for MAD-based outlier removal post-processing."""

    @staticmethod
    def _make_three_clusters_with_outliers(rng):
        """3 tight clusters (30 pts each, std=0.3) + 2 extreme outlier points.

        Outlier at index 90 is assigned to cluster 0 (near mean 0) but at [20,...].
        Outlier at index 91 is assigned to cluster 2 (near mean -5) but at [-20,...].
        """
        n_per = 30
        d = 5
        c0 = rng.normal(loc=0.0, scale=0.3, size=(n_per, d))
        c1 = rng.normal(loc=5.0, scale=0.3, size=(n_per, d))
        c2 = rng.normal(loc=-5.0, scale=0.3, size=(n_per, d))
        outliers = np.array([[20.0] * d, [-20.0] * d])
        data = np.vstack([c0, c1, c2, outliers])
        labels = np.array([0] * n_per + [1] * n_per + [2] * n_per + [0, 2])
        return data, labels

    def test_clear_outliers_flagged(self):
        """KEEP mode: extreme outliers are separated into their own cluster."""
        rng = np.random.default_rng(42)
        data, labels = self._make_three_clusters_with_outliers(rng)

        # Use a high enough MAD threshold (10) that only the extreme outliers
        # are flagged, not the tails of the tight clusters.
        result = remove_outliers_mad(
            data, labels, mad_threshold=10.0,
            small_cluster_mode=SmallClusterMode.KEEP,
        )

        # The two outlier points (indices 90, 91) should share a label
        # different from every non-outlier cluster.
        outlier_label = result[90]
        assert result[91] == outlier_label
        main_labels = set(np.unique(result[:90]))
        assert outlier_label not in main_labels

    def test_no_outliers_unchanged(self):
        """Tight clusters with no outliers should be unchanged at a moderate threshold."""
        rng = np.random.default_rng(42)
        n_per = 50
        d = 5
        c0 = rng.normal(loc=0.0, scale=1.0, size=(n_per, d))
        c1 = rng.normal(loc=10.0, scale=1.0, size=(n_per, d))
        c2 = rng.normal(loc=-10.0, scale=1.0, size=(n_per, d))
        data = np.vstack([c0, c1, c2])
        labels = np.array([0] * n_per + [1] * n_per + [2] * n_per)

        result = remove_outliers_mad(
            data, labels, mad_threshold=_sigma_to_mad_threshold(10.0),
            small_cluster_mode=SmallClusterMode.KEEP,
        )

        np.testing.assert_array_equal(result, labels)

    def test_outlier_mode_relabels_minus_one(self):
        """OUTLIER mode: flagged points should have label -1."""
        rng = np.random.default_rng(42)
        data, labels = self._make_three_clusters_with_outliers(rng)

        result = remove_outliers_mad(
            data, labels, mad_threshold=10.0,
            small_cluster_mode=SmallClusterMode.OUTLIER,
        )

        assert result[90] == -1
        assert result[91] == -1
        # Non-outlier points should not be -1
        assert np.all(result[:90] >= 0)

    def test_absorb_mode_reassigns(self):
        """ABSORB mode: flagged points should be assigned to nearest cluster."""
        rng = np.random.default_rng(42)
        data, labels = self._make_three_clusters_with_outliers(rng)

        result = remove_outliers_mad(
            data, labels, mad_threshold=10.0,
            small_cluster_mode=SmallClusterMode.ABSORB,
        )

        # Should not have -1 labels (outliers are absorbed)
        assert np.all(result >= 0)
        # Should have exactly 3 clusters (no new outlier cluster)
        assert len(np.unique(result)) == 3
        # Outlier at [20,...] (idx 90) nearest to cluster at 5 (label 1)
        # Outlier at [-20,...] (idx 91) nearest to cluster at -5 (label 2)
        assert result[90] != result[91]

    def test_small_cluster_skipped(self):
        """Clusters with < _MIN_CLUSTER_FOR_MAD=5 members should not be flagged."""
        rng = np.random.default_rng(42)
        d = 5
        # Large cluster: 30 pts at origin
        c0 = rng.normal(loc=0.0, scale=0.3, size=(30, d))
        # Tiny cluster: 3 pts at 100 (extreme but too small for MAD check)
        c1 = rng.normal(loc=100.0, scale=0.3, size=(3, d))
        data = np.vstack([c0, c1])
        labels = np.array([0] * 30 + [1] * 3)

        result = remove_outliers_mad(
            data, labels, mad_threshold=_sigma_to_mad_threshold(3.0),
            small_cluster_mode=SmallClusterMode.OUTLIER,
        )

        # The 3-member cluster should not have its points flagged as outliers
        assert np.all(result[30:] >= 0)

    def test_high_threshold_flags_nothing(self):
        """A very high mad_threshold (100.0) should flag nothing on moderate data."""
        rng = np.random.default_rng(42)
        n_per = 50
        d = 5
        c0 = rng.normal(loc=0.0, scale=1.0, size=(n_per, d))
        c1 = rng.normal(loc=10.0, scale=1.0, size=(n_per, d))
        data = np.vstack([c0, c1])
        labels = np.array([0] * n_per + [1] * n_per)

        result = remove_outliers_mad(
            data, labels, mad_threshold=_sigma_to_mad_threshold(100.0),
            small_cluster_mode=SmallClusterMode.KEEP,
        )

        np.testing.assert_array_equal(result, labels)

    def test_high_threshold_conservative(self):
        """sigma_threshold=10.0 should flag nothing on moderately spread data."""
        rng = np.random.default_rng(42)
        n_per = 50
        d = 5
        c0 = rng.normal(loc=0.0, scale=1.0, size=(n_per, d))
        c1 = rng.normal(loc=10.0, scale=1.0, size=(n_per, d))
        data = np.vstack([c0, c1])
        labels = np.array([0] * n_per + [1] * n_per)

        result = remove_outliers_mad(
            data, labels, mad_threshold=_sigma_to_mad_threshold(10.0),
            small_cluster_mode=SmallClusterMode.KEEP,
        )

        np.testing.assert_array_equal(result, labels)

    def test_low_threshold_aggressive(self):
        """A low sigma threshold (1.5) should flag many tail points."""
        rng = np.random.default_rng(42)
        n_per = 200
        d = 5
        c0 = rng.normal(loc=0.0, scale=1.0, size=(n_per, d))
        c1 = rng.normal(loc=10.0, scale=1.0, size=(n_per, d))
        data = np.vstack([c0, c1])
        labels = np.array([0] * n_per + [1] * n_per)

        result = remove_outliers_mad(
            data, labels, mad_threshold=_sigma_to_mad_threshold(1.5),
            small_cluster_mode=SmallClusterMode.OUTLIER,
        )

        n_flagged = np.sum(result == -1)
        # sigma=1.5 → mad_threshold≈1.01.  With the any-PC rule across
        # multiple PCs, a substantial fraction of tail points are flagged.
        assert n_flagged > 10

    def test_labels_renumbered_contiguously(self):
        """After KEEP mode removal, labels should be 0,1,...,k with no gaps."""
        rng = np.random.default_rng(42)
        data, labels = self._make_three_clusters_with_outliers(rng)

        result = remove_outliers_mad(
            data, labels, mad_threshold=10.0,
            small_cluster_mode=SmallClusterMode.KEEP,
        )

        unique = np.unique(result)
        # No -1 in KEEP mode (outliers become a new cluster)
        assert np.all(unique >= 0)
        # Labels should be contiguous: 0, 1, ..., max
        np.testing.assert_array_equal(unique, np.arange(len(unique)))

    def test_sigma_to_mad_conversion(self):
        """Verify the sigma → MAD threshold conversion uses MAD_k correctly.

        With a single 1-D cluster of 10k Gaussian points, using
        ``mad_threshold = 3.0`` (i.e. 3 MAD units, which equals
        3 / MAD_k ≈ 2.02 σ), approximately 4.3% of points should be
        flagged (two-tailed P(|Z| > 2.02σ)).
        """
        rng = np.random.default_rng(42)
        n = 10_000
        data = rng.normal(loc=0.0, scale=1.0, size=(n, 1))
        labels = np.zeros(n, dtype=int)

        # Use mad_threshold=3.0 directly (3 MAD units)
        result = remove_outliers_mad(
            data, labels, mad_threshold=3.0,
            small_cluster_mode=SmallClusterMode.OUTLIER,
            n_pcs=1,
        )

        n_flagged = np.sum(result == -1)
        # 3 MAD = 3 × (σ / MAD_k) → cutoff at 3 / MAD_k ≈ 2.02 σ
        # P(|Z| > 2.02) ≈ 0.043 → expect ~430 flagged
        assert 300 <= n_flagged <= 600, (
            f"Expected ~430 flagged at 3 MAD on 10k Gaussian points, got {n_flagged}"
        )

        # Verify the constant itself: MAD_k ≈ 1.4826
        assert abs(MAD_k - 1.4826) < 0.001


# =============================================================================
# Tests for DBSCAN / HDBSCAN bug fixes
# =============================================================================


class TestDBSCANHDBSCAN:
    """Regression tests for bugs found and fixed in the DBSCAN/HDBSCAN wrappers."""

    @staticmethod
    def _make_two_clusters_df(seed=42):
        """Two well-separated 5-d clusters, 30 points each, as a DataFrame."""
        rng = np.random.default_rng(seed)
        c0 = rng.normal(loc=0.0, scale=0.3, size=(30, 5))
        c1 = rng.normal(loc=10.0, scale=0.3, size=(30, 5))
        return pd.DataFrame(np.vstack([c0, c1]))

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_hdbscan_distance_metric_string(self):
        """HDBSCAN must receive the distance metric as a string, not an enum.

        Previously the enum value was passed directly, causing sklearn to reject
        the metric parameter.  The fix calls ``.value`` on the enum.
        """
        df = self._make_two_clusters_df()
        settings = make_clustering_settings("hdbscan", seed=42)
        labels = cluster_features(df, settings)

        assert labels is not None
        assert np.all(labels >= -1)
        # At least one non-outlier cluster should be found
        assert np.max(labels) >= 0

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_hdbscan_min_samples_1_relabeling(self):
        """When min_samples=1, outlier points are relabeled with unique cluster IDs.

        The special branch (line 48 of hdbscan.py) shifts non-outlier labels up
        and assigns outlier indices 0..N-1.  Verify this executes without error
        and that a clear outlier gets its own label (not -1).
        """
        rng = np.random.default_rng(42)
        # Two tight clusters + one far-away point that HDBSCAN would mark as noise
        c0 = rng.normal(loc=0.0, scale=0.3, size=(30, 5))
        c1 = rng.normal(loc=10.0, scale=0.3, size=(30, 5))
        outlier = np.full((1, 5), 100.0)
        df = pd.DataFrame(np.vstack([c0, c1, outlier]))

        # min_samples=1 triggers the relabeling branch
        settings = make_clustering_settings(
            "hdbscan", seed=42,
            hdbscan={"min_samples": 1},
        )
        labels = cluster_features(df, settings)

        assert labels is not None
        # With min_samples=1 relabeling, there should be no -1 labels at all
        # (outliers get their own sequential cluster IDs)
        assert np.all(labels >= 0), (
            f"Expected no -1 labels after min_samples=1 relabeling, got {labels}"
        )

    def test_dbscan_produces_valid_labels(self):
        """Basic smoke test: DBSCAN should produce valid cluster labels."""
        df = self._make_two_clusters_df()
        settings = make_clustering_settings(
            "dbscan", seed=42,
            dbscan={"epsilon": 2.0, "min_samples": 3},
        )
        labels = cluster_features(df, settings)

        assert labels is not None
        assert np.all(labels >= -1)
        # Should find at least one cluster
        assert np.max(labels) >= 0

    def test_dbscan_no_unused_variables(self):
        """Code quality regression: DBSCAN runs cleanly (unused seed was removed)."""
        df = self._make_two_clusters_df()
        settings = make_clustering_settings(
            "dbscan", seed=42,
            dbscan={"epsilon": 2.0, "min_samples": 3},
        )
        # This should complete without any errors — the previously unused
        # ``seed`` variable has been removed from the dbscan() function.
        labels = cluster_features(df, settings)
        assert labels is not None


# =============================================================================
# Tests for label_ops edge cases
# =============================================================================


class TestLabelOpsEdgeCases:
    """Edge-case tests for assign_small_clusters_nearest."""

    def test_assign_small_clusters_no_large_clusters(self):
        """When ALL clusters are below min_cluster_size, return labels unchanged.

        This exercises the early-return guard at line 102 of label_ops.py
        (``if len(large_ids) == 0: return clusters.copy()``).
        """
        data = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
            [11.0, 11.0],
        ])
        labels = np.array([0, 0, 1, 1])  # two clusters of size 2

        result = assign_small_clusters_nearest(
            labels, data, min_cluster_size=5,
        )

        # No large clusters exist, so the function should return a copy of the
        # original labels unchanged.
        np.testing.assert_array_equal(result, labels)

    def test_assign_small_clusters_all_outliers(self):
        """All-outlier input (-1 labels) should be handled gracefully."""
        data = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        labels = np.array([-1, -1, -1])

        result = assign_small_clusters_nearest(
            labels, data, min_cluster_size=2,
        )

        # No valid clusters at all — early return with original labels
        np.testing.assert_array_equal(result, labels)


# =============================================================================
# Run tests
# =============================================================================

class TestClusterFeaturesContract:
    """Public-API contract of cluster_features (DataFrame -> label array)."""

    @pytest.fixture
    def two_shape_df(self):
        """20 rising ramps then 20 falling ramps — shapes survive per-sample norm."""
        rng = np.random.default_rng(0)
        rising = [np.linspace(0, 1, 24) + rng.normal(0, 0.02, 24) for _ in range(20)]
        falling = [np.linspace(1, 0, 24) + rng.normal(0, 0.02, 24) for _ in range(20)]
        df = pd.DataFrame(np.vstack([rising, falling]))

        return df

    def _k2_settings(self):
        """Forced k=2 with outlier removal off so blocks map cleanly to labels."""
        cs = make_clustering_settings(
            "kmedians",
            outlier_removal_sigma=None,
            kmedians={"n_cluster": {"lower": 2, "upper": 2}},
        )

        return cs

    def test_returns_integer_array_length_n(self, two_shape_df):
        """Result is an integer ndarray with one label per input row."""
        labels = cluster_features(two_shape_df, make_clustering_settings("kmedians"))
        assert isinstance(labels, np.ndarray)
        assert np.issubdtype(labels.dtype, np.integer)
        assert len(labels) == len(two_shape_df)

    def test_row_order_preserved(self, two_shape_df):
        """Label i corresponds to input row i: each shape block maps to one label."""
        labels = cluster_features(two_shape_df, self._k2_settings())
        rising_labels = set(labels[:20])
        falling_labels = set(labels[20:])
        assert len(rising_labels) == 1
        assert len(falling_labels) == 1
        assert rising_labels.isdisjoint(falling_labels)

    def test_lower_bound_above_n_returns_identity(self, two_shape_df):
        """When n_cluster_lower >= n_samples the pipeline returns identity labels."""
        settings = make_clustering_settings(
            "kmedians", kmedians={"n_cluster": {"lower": 100, "upper": 100}},
        )
        labels = cluster_features(two_shape_df, settings)
        assert np.array_equal(labels, np.arange(len(two_shape_df)))


class TestClusterFeaturesDegenerateColumns:
    """Pipeline behaviour on NaN and zero-variance feature columns."""

    def _k2_settings(self):
        cs = make_clustering_settings(
            "kmedians",
            outlier_removal_sigma=None,
            kmedians={"n_cluster": {"lower": 2, "upper": 2}},
        )

        return cs

    @pytest.fixture
    def two_block_df(self):
        rng = np.random.default_rng(0)
        data = np.vstack([rng.normal(0, 1, (15, 6)), rng.normal(20, 1, (15, 6))])

        return pd.DataFrame(data)

    def test_whole_column_nan_raises(self, two_block_df):
        """A wholly-NaN feature column is rejected as non-finite, not clustered."""
        two_block_df[2] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            cluster_features(two_block_df, self._k2_settings())

    def test_interior_nan_raises(self, two_block_df):
        """A single NaN cell is likewise rejected."""
        two_block_df.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            cluster_features(two_block_df, self._k2_settings())

    def test_zero_variance_column_handled(self, two_block_df):
        """A constant (zero-variance) column is absorbed by the safe scalers,
        leaving finite labels rather than dividing by a zero scale."""
        two_block_df[3] = 5.0
        labels = cluster_features(two_block_df, self._k2_settings())
        assert len(labels) == len(two_block_df)
        assert np.all(np.isfinite(labels))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
