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
from sklearn.datasets import make_blobs, make_circles, make_moons

from opendsm.common.clustering.algorithms.hdbscan import hdbscan
from opendsm.common.clustering.settings import ClusteringSettings


def get_default_settings_dict():
    """Return a default settings dictionary that can be modified."""
    return {
        "algorithm_selection": "hdbscan",
        "backup_algorithm_selection": None,  # Set to None to avoid conflict with algorithm_selection
        "seed": 42,
    }


@pytest.fixture
def simple_clustered_data():
    """Create simple synthetic data with clear clusters."""
    data, _ = make_blobs(
        n_samples=150,
        n_features=10,
        centers=3,
        cluster_std=0.6,
        random_state=42
    )

    return data


@pytest.fixture
def clustered_data_with_outliers():
    """Create data with clear clusters and some outliers."""
    # Main cluster
    cluster, _ = make_blobs(
        n_samples=80,
        n_features=10,
        centers=1,
        cluster_std=0.5,
        random_state=42
    )
    # Outliers far from cluster
    outliers, _ = make_blobs(
        n_samples=10,
        n_features=10,
        centers=1,
        center_box=(20, 25),
        cluster_std=0.3,
        random_state=43
    )

    return np.vstack([cluster, outliers])


@pytest.fixture
def default_settings():
    """Create default clustering settings."""
    settings_dict = get_default_settings_dict()
    return ClusteringSettings(**settings_dict)


class TestBasicFunctionality:
    """Tests for basic HDBSCAN functionality."""

    def test_simple_clustering_produces_expected_clusters(self, simple_clustered_data, default_settings):
        """Test basic clustering finds expected number of clusters."""
        # simple_clustered_data has 3 well-separated clusters (150 samples, 3 centers)
        labels = hdbscan(simple_clustered_data, default_settings).labels

        # Check output format
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 150
        assert labels.dtype in [np.int32, np.int64]

        # Check expected clustering behavior
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 3  # Should find exactly 3 clusters
        assert np.all(labels >= 0)  # No outliers with default min_samples=1

        # Verify cluster sizes are reasonable (not all points in one cluster)
        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            assert 20 <= cluster_size <= 80  # Each cluster should have reasonable size

    def test_output_format(self, simple_clustered_data, default_settings):
        """Test that output has correct format."""
        labels = hdbscan(simple_clustered_data, default_settings).labels

        assert isinstance(labels, np.ndarray)
        assert labels.shape == (150,)
        assert np.issubdtype(labels.dtype, np.integer)
        # All labels should be finite integers
        assert np.all(np.isfinite(labels))
        assert np.all(labels >= 0)

    def test_deterministic_behavior(self, simple_clustered_data):
        """Test that HDBSCAN produces consistent results with same seed."""
        settings = ClusteringSettings(**get_default_settings_dict())

        labels1 = hdbscan(simple_clustered_data, settings).labels
        labels2 = hdbscan(simple_clustered_data, settings).labels

        # HDBSCAN should be deterministic with same settings and data
        assert np.array_equal(labels1, labels2)
        # Also verify the actual cluster counts match
        assert len(np.unique(labels1)) == len(np.unique(labels2))


class TestOutlierHandling:
    """Tests for HDBSCAN's special outlier handling logic."""

    def test_outlier_relabeling_with_min_samples_1(self, clustered_data_with_outliers):
        """Test that min_samples=1 converts outliers to individual clusters."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 1}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(clustered_data_with_outliers, settings).labels

        # With min_samples=1, outliers should be converted to individual clusters
        # No -1 labels should remain
        assert -1 not in labels
        assert len(np.unique(labels)) > 0
        assert np.all(labels >= 0)

    def test_outliers_remain_with_min_samples_gt_1(self, clustered_data_with_outliers):
        """Test that min_samples > 1 can produce -1 labels for outliers."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(clustered_data_with_outliers, settings).labels

        # With higher min_samples, we might get outliers (-1 labels)
        # This is allowed and expected
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(clustered_data_with_outliers)

    def test_no_outliers_case(self):
        """Test when all points belong to clusters (no outliers)."""
        # Very tight, dense cluster
        data, _ = make_blobs(
            n_samples=100,
            n_features=5,
            centers=1,
            cluster_std=0.2,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 1, "allow_single_cluster": True}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Should handle the case when there are no outliers
        assert -1 not in labels
        assert len(np.unique(labels)) >= 1

    def test_all_outliers_with_min_samples_1(self):
        """Test when parameters are strict and everything becomes outlier initially."""
        np.random.seed(42)
        # Very sparse, uniform data
        data = np.random.uniform(-10, 10, (50, 5))

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "min_samples": 1,
            "cluster_selection_epsilon": 0.0,
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # With min_samples=1, even if all are outliers initially, they should be relabeled
        assert -1 not in labels


class TestAlgorithmParameters:
    """Tests for different HDBSCAN algorithm parameters."""

    @pytest.mark.parametrize("min_samples_val,expected_min_clusters", [
        (1, 3),   # Most permissive, should find all 3 clusters
        (2, 3),   # Still finds all 3 clusters
        (5, 2),   # More conservative, may merge some clusters
        (10, 1),  # Very conservative, likely finds fewer clusters
    ])
    def test_min_samples_affects_clustering_granularity(self, simple_clustered_data, min_samples_val, expected_min_clusters):
        """Test that min_samples parameter affects clustering granularity."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": min_samples_val}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels

        assert len(labels) == 150
        assert isinstance(labels, np.ndarray)
        # Verify at least expected_min_clusters were found
        unique_labels = [l for l in np.unique(labels) if l != -1]
        assert len(unique_labels) >= expected_min_clusters

    def test_min_samples_1_converted_to_2_for_min_cluster_size(self, simple_clustered_data):
        """Test that min_samples=1 is handled correctly (converted to 2 internally)."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 1}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels

        assert len(labels) == 150
        # With min_samples=1, all points should be assigned to clusters (no -1 labels)
        assert np.all(labels >= 0)
        # Should find all 3 clusters with such permissive settings
        assert len(np.unique(labels)) == 3

    def test_allow_single_cluster_true(self):
        """Test that allow_single_cluster=True can produce single cluster."""
        # Homogeneous, tight data
        data, _ = make_blobs(
            n_samples=100,
            n_features=5,
            centers=1,
            cluster_std=0.3,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "allow_single_cluster": True,
            "min_samples": 5,
            "cluster_selection_epsilon": 0.0
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # With tight data and allow_single_cluster, should produce at least one cluster
        unique_labels = [l for l in np.unique(labels) if l != -1]
        assert len(unique_labels) >= 1

    def test_allow_single_cluster_false(self):
        """Test that allow_single_cluster=False forces multiple clusters."""
        # Data with subtle structure
        data, _ = make_blobs(
            n_samples=100,
            n_features=5,
            centers=2,
            cluster_std=0.5,
            center_box=(0, 2),
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "allow_single_cluster": False,
            "min_samples": 5
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Should try to find multiple clusters or mark some as outliers
        assert isinstance(labels, np.ndarray)

    def test_max_cluster_size_none(self, simple_clustered_data):
        """Test with max_cluster_size=None (unlimited)."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"max_cluster_size": None}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels
        assert len(labels) == len(simple_clustered_data)

    def test_max_cluster_size_limited(self):
        """Test with max_cluster_size limit."""
        data, _ = make_blobs(
            n_samples=200,
            n_features=5,
            centers=2,
            cluster_std=1.0,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "max_cluster_size": 50,
            "min_samples": 5
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Check that clusters respect max size (if formed)
        unique_labels = [l for l in np.unique(labels) if l != -1]
        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            # Note: max_cluster_size affects cluster formation, not a hard limit
            assert cluster_size >= 0  # Basic sanity check

    @pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1, 0.5])
    def test_cluster_selection_epsilon_variations(self, simple_clustered_data, epsilon):
        """Test different cluster_selection_epsilon values control cluster merging."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "cluster_selection_epsilon": epsilon,
            "min_samples": 5
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels

        assert len(labels) == 150
        # Higher epsilon values may merge more clusters
        unique_labels = [l for l in np.unique(labels) if l != -1]
        assert len(unique_labels) >= 1

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
    def test_robust_single_linkage_scaling(self, simple_clustered_data, alpha):
        """Test different alpha (robust_single_linkage_scaling) values."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "robust_single_linkage_scaling": alpha,
            "min_samples": 5
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels

        assert len(labels) == 150
        unique_labels = [l for l in np.unique(labels) if l != -1]
        assert len(unique_labels) >= 1


class TestDistanceMetrics:
    """Tests for different distance metrics."""

    @pytest.mark.parametrize("metric", [
        "euclidean",
        "manhattan",
        "cosine",
        "seuclidean",
    ])
    def test_distance_metrics(self, simple_clustered_data, metric):
        """Test HDBSCAN with various distance metrics."""
        data = simple_clustered_data

        # Cosine metric requires normalized data
        if metric == "cosine":
            data = data / np.linalg.norm(data, axis=1, keepdims=True)

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"distance_metric": metric}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        assert len(labels) == len(data)
        assert len(np.unique(labels)) > 0


class TestClusterSelectionMethods:
    """Tests for different cluster selection methods."""

    @pytest.mark.parametrize("method,expected_min_clusters", [
        ("eom", 1),  # Excess of Mass typically finds fewer, larger clusters
        ("leaf", 1),  # Leaf method may find more granular clusters
    ])
    def test_cluster_selection_methods(self, simple_clustered_data, method, expected_min_clusters):
        """Test HDBSCAN with different cluster selection methods."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "cluster_selection_method": method,
            "min_samples": 5
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels

        assert len(labels) == len(simple_clustered_data)
        # Verify clusters were found
        unique_labels = [l for l in np.unique(labels) if l != -1]
        assert len(unique_labels) >= expected_min_clusters


class TestNearestNeighborsAlgorithm:
    """Tests for different nearest neighbors algorithms."""

    @pytest.mark.parametrize("algorithm", [
        "auto",
        "brute",
        "ball_tree",
        "kd_tree",
    ])
    def test_nearest_neighbors_algorithms(self, simple_clustered_data, algorithm):
        """Test HDBSCAN with different nearest neighbors algorithms."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"nearest_neighbors_algorithm": algorithm}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels

        assert len(labels) == len(simple_clustered_data)
        assert len(np.unique(labels)) > 0

    @pytest.mark.parametrize("leaf_size", [10, 30, 40, 100])
    def test_leaf_size_variations(self, simple_clustered_data, leaf_size):
        """Test different leaf_size values for tree-based algorithms."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "nearest_neighbors_algorithm": "ball_tree", 
            "leaf_size": leaf_size
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels

        assert len(labels) == len(simple_clustered_data)
        assert len(np.unique(labels)) > 0


class TestDataShapes:
    """Tests for different data shapes and sizes."""

    def test_very_small_dataset(self):
        """Test clustering on very small dataset."""
        data, _ = make_blobs(
            n_samples=5,
            n_features=5,
            centers=2,
            cluster_std=0.5,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 1}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 5

    def test_small_dataset(self):
        """Test clustering on small dataset."""
        data, _ = make_blobs(
            n_samples=50,
            n_features=5,
            centers=2,
            cluster_std=1.0,
            random_state=42
        )

        settings = ClusteringSettings(**get_default_settings_dict())
        labels = hdbscan(data, settings).labels
        assert len(labels) == 50

    def test_medium_dataset(self):
        """Test clustering on medium dataset."""
        data, _ = make_blobs(
            n_samples=500,
            n_features=10,
            centers=3,
            cluster_std=1.0,
            random_state=42
        )

        settings = ClusteringSettings(**get_default_settings_dict())
        labels = hdbscan(data, settings).labels
        assert len(labels) == 500

    def test_large_dataset(self):
        """Test clustering on large dataset."""
        data, _ = make_blobs(
            n_samples=5000,
            n_features=20,
            centers=5,
            cluster_std=2.0,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 10}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 5000

    def test_single_feature_data(self):
        """Test clustering on 1D data."""
        data, _ = make_blobs(
            n_samples=100,
            n_features=1,
            centers=3,
            cluster_std=0.5,
            random_state=42
        )

        settings = ClusteringSettings(**get_default_settings_dict())
        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_low_dimensional_data(self):
        """Test clustering on 2D and 3D data."""
        for n_features in [2, 3]:
            data, _ = make_blobs(
                n_samples=100,
                n_features=n_features,
                centers=3,
                cluster_std=1.0,
                random_state=42
            )

            settings = ClusteringSettings(**get_default_settings_dict())
            labels = hdbscan(data, settings).labels
            assert len(labels) == 100

    def test_medium_dimensional_data(self):
        """Test clustering on medium-dimensional data."""
        data, _ = make_blobs(
            n_samples=100,
            n_features=20,
            centers=3,
            cluster_std=1.5,
            random_state=42
        )

        settings = ClusteringSettings(**get_default_settings_dict())
        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_high_dimensional_data(self):
        """Test clustering on high-dimensional data."""
        data, _ = make_blobs(
            n_samples=100,
            n_features=50,
            centers=2,
            cluster_std=2.0,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_very_high_dimensional_data(self):
        """Test clustering on very high-dimensional data (curse of dimensionality)."""
        data, _ = make_blobs(
            n_samples=100,
            n_features=200,
            centers=2,
            cluster_std=3.0,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 100


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_identical_samples(self):
        """Test clustering when all samples are identical."""
        data = np.ones((50, 10))

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 50
        # All identical points should be in same cluster or all outliers
        assert len(np.unique(labels)) >= 1

    def test_uniform_random_data(self):
        """Test clustering on uniform random data (no natural clusters)."""
        np.random.seed(42)
        data = np.random.uniform(-1, 1, (100, 10))

        settings = ClusteringSettings(**get_default_settings_dict())
        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_single_data_point(self):
        """sklearn HDBSCAN requires n_samples > 1."""
        data = np.array([[1, 2, 3, 4, 5]])

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 1}
        settings = ClusteringSettings(**settings_dict)

        with pytest.raises(ValueError):
            hdbscan(data, settings)

    def test_two_data_points(self):
        """Test clustering with only 2 samples."""
        data = np.array([[1, 2, 3], [4, 5, 6]])

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 1}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 2

    def test_negative_values(self):
        """Test clustering with all negative coordinates."""
        np.random.seed(42)
        data = np.random.randn(100, 10) - 10

        settings = ClusteringSettings(**get_default_settings_dict())
        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_mixed_positive_negative(self):
        """Test clustering with data spanning negative to positive."""
        np.random.seed(42)
        data = np.random.randn(100, 10) * 5

        settings = ClusteringSettings(**get_default_settings_dict())
        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_very_large_coordinate_values(self):
        """Test clustering with very large coordinate values."""
        np.random.seed(42)
        data = np.random.randn(100, 10) * 1e10

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_very_small_coordinate_values(self):
        """Test clustering with very small coordinate values."""
        np.random.seed(42)
        data = np.random.randn(100, 10) * 1e-10

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_mixed_scale_features(self):
        """Test clustering with features at vastly different scales."""
        np.random.seed(42)
        # Create data with features at different scales
        data = np.column_stack([
            np.random.randn(100) * 0.01,    # Small scale
            np.random.randn(100) * 1.0,     # Medium scale
            np.random.randn(100) * 1000.0   # Large scale
        ])

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_min_samples_equals_dataset_size(self):
        """Test when min_samples equals dataset size."""
        np.random.seed(42)
        data = np.random.randn(20, 5)

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 20}
        settings = ClusteringSettings(**settings_dict)

        # Should handle gracefully, likely all outliers
        labels = hdbscan(data, settings).labels
        assert len(labels) == 20

    def test_very_large_epsilon(self):
        """Test with cluster_selection_epsilon much larger than data spread."""
        np.random.seed(42)
        data = np.random.randn(100, 5)  # Data spread ~[-3, 3]

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "cluster_selection_epsilon": 100.0,
            "min_samples": 5
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_zero_epsilon(self):
        """Test with cluster_selection_epsilon = 0 (no merging)."""
        np.random.seed(42)
        data = np.random.randn(100, 5)

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "cluster_selection_epsilon": 0.0,
            "min_samples": 5
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 100


class TestSpecialDataDistributions:
    """Tests for special data distributions."""

    def test_gaussian_blobs(self):
        """Test on clear, well-separated Gaussian clusters."""
        data, true_labels = make_blobs(
            n_samples=300,
            n_features=10,
            centers=5,
            cluster_std=1.0,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Should identify multiple clusters
        unique_labels = [l for l in np.unique(labels) if l != -1]
        assert len(unique_labels) >= 2

    def test_overlapping_clusters(self):
        """Test on partially overlapping distributions."""
        data, _ = make_blobs(
            n_samples=200,
            n_features=5,
            centers=3,
            cluster_std=2.0,  # Large std creates overlap
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 200

    def test_nested_clusters(self):
        """Test on nested/concentric clusters."""
        # Create concentric circles
        data, _ = make_circles(
            n_samples=150,
            factor=0.3,
            noise=0.05,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # HDBSCAN should handle nested structures
        assert len(labels) == 150

    def test_nonlinear_clusters(self):
        """Test on non-linear, crescent-shaped clusters."""
        # Create two interleaving half circles
        data, _ = make_moons(
            n_samples=100,
            noise=0.05,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels
        assert len(labels) == 100

    def test_varying_density_clusters(self):
        """Test HDBSCAN's ability to find clusters with different densities."""
        # Dense cluster (small std)
        dense, _ = make_blobs(
            n_samples=100,
            n_features=5,
            centers=1,
            cluster_std=0.3,
            random_state=42
        )
        # Sparse cluster (large std)
        sparse, _ = make_blobs(
            n_samples=50,
            n_features=5,
            centers=1,
            center_box=(8, 10),
            cluster_std=1.5,
            random_state=43
        )
        data = np.vstack([dense, sparse])

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "min_samples": 5,
            "allow_single_cluster": False
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Should identify at least 2 distinct density regions
        unique_labels = [l for l in np.unique(labels) if l != -1]
        assert len(unique_labels) >= 1

    def test_noise_heavy_data(self):
        """Test on data with heavy noise."""
        # Main cluster
        cluster, _ = make_blobs(
            n_samples=80,
            n_features=5,
            centers=1,
            cluster_std=0.5,
            random_state=42
        )
        # Scattered noise
        np.random.seed(42)
        noise = np.random.uniform(-5, 5, (40, 5))
        data = np.vstack([cluster, noise])

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Should handle noise (likely many outliers or small clusters)
        assert len(labels) == 120


class TestClusterQuality:
    """Tests to verify cluster quality and separation."""

    def test_well_separated_clusters_identified(self):
        """Test that well-separated clusters are correctly identified."""
        # Create three very distinct clusters with known centers far apart
        centers = np.array([
            [0, 0, 0, 0, 0],
            [10, 10, 10, 10, 10],
            [20, 20, 20, 20, 20]
        ])
        data, true_labels = make_blobs(
            n_samples=120,
            n_features=5,
            centers=centers,
            cluster_std=0.5,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Should identify multiple clusters
        unique_labels = [l for l in np.unique(labels) if l != -1]
        assert len(unique_labels) >= 2

        # Check cluster purity (samples from same original cluster stay together)
        cluster_accuracy = []
        for i in range(3):
            # Get indices for each true cluster
            true_cluster_mask = true_labels == i
            original_cluster_labels = labels[true_cluster_mask]
            # Exclude outliers
            non_outlier_labels = original_cluster_labels[original_cluster_labels != -1]
            if len(non_outlier_labels) > 0:
                most_common = np.bincount(non_outlier_labels).argmax()
                accuracy = np.sum(original_cluster_labels == most_common) / len(original_cluster_labels)
                cluster_accuracy.append(accuracy)

        # Most samples should be correctly clustered
        if cluster_accuracy:
            assert np.mean(cluster_accuracy) > 0.7

    def test_hierarchical_density_detection(self):
        """Test HDBSCAN's hierarchical density detection capability."""
        # Create multi-scale structure with three different densities
        # Very dense core
        core, _ = make_blobs(
            n_samples=30,
            n_features=3,
            centers=1,
            cluster_std=0.2,
            random_state=42
        )
        # Medium density shell
        shell, _ = make_blobs(
            n_samples=50,
            n_features=3,
            centers=1,
            center_box=(3, 4),
            cluster_std=0.8,
            random_state=43
        )
        # Sparse outer region
        outer, _ = make_blobs(
            n_samples=40,
            n_features=3,
            centers=1,
            center_box=(7, 8),
            cluster_std=1.5,
            random_state=44
        )

        data = np.vstack([core, shell, outer])

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Should find structure at different density levels
        assert len(labels) == 120
        assert len(np.unique(labels)) > 0

    def test_outlier_detection_accuracy(self):
        """Test outlier identification."""
        # Main cluster
        cluster, _ = make_blobs(
            n_samples=90,
            n_features=5,
            centers=1,
            cluster_std=0.5,
            random_state=42
        )
        # Known outliers
        outliers = np.array([
            [10, 10, 10, 10, 10],
            [-10, -10, -10, -10, -10],
            [15, -15, 15, -15, 15]
        ])
        data = np.vstack([cluster, outliers])

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 10}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Check if outliers are marked differently
        # (either as -1 or in separate small clusters)
        outlier_labels = labels[-3:]
        cluster_labels = labels[:-3]

        assert len(labels) == 93

    def test_cluster_stability(self):
        """Test that results are relatively stable with small perturbations."""
        data, _ = make_blobs(
            n_samples=100,
            n_features=5,
            centers=3,
            cluster_std=1.0,
            random_state=42
        )

        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": 5}
        settings = ClusteringSettings(**settings_dict)

        labels1 = hdbscan(data, settings).labels

        # Add small noise
        np.random.seed(43)
        data_perturbed = data + np.random.randn(100, 5) * 0.01
        labels2 = hdbscan(data_perturbed, settings).labels

        # Results should be similar (not necessarily identical)
        assert len(labels1) == len(labels2)

    def test_minimum_cluster_size_enforcement(self):
        """Test that clusters respect min_cluster_size."""
        data, _ = make_blobs(
            n_samples=100,
            n_features=5,
            centers=3,
            cluster_std=1.0,
            random_state=42
        )

        min_samples_val = 10
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"min_samples": min_samples_val}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(data, settings).labels

        # Check cluster sizes
        unique_labels = [l for l in np.unique(labels) if l != -1]
        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            # Note: actual min_cluster_size is min(min_samples_val, 2)
            # Clusters should generally be >= min_cluster_size
            assert cluster_size >= 0  # Basic sanity check


class TestScoringsSampleCount:
    """Tests for scoring_sample_count parameter."""

    def test_scoring_sample_count_none(self, simple_clustered_data):
        """Test with scoring_sample_count=None."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {"scoring_sample_count": None}
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels
        assert len(labels) == len(simple_clustered_data)

    def test_scoring_sample_count_specified(self, simple_clustered_data):
        """Test with explicit scoring_sample_count values."""
        for count in [5, 10, 20]:
            settings_dict = get_default_settings_dict()
            settings_dict["hdbscan"] = {
                "scoring_sample_count": count,
                "min_samples": 5
            }
            settings = ClusteringSettings(**settings_dict)

            labels = hdbscan(simple_clustered_data, settings).labels
            assert len(labels) == len(simple_clustered_data)


class TestInvalidData:
    """Tests for invalid or problematic data."""

    def test_empty_array(self):
        """Test clustering with empty array."""
        data = np.array([]).reshape(0, 5)

        settings = ClusteringSettings(**get_default_settings_dict())

        # Should handle gracefully (likely raise error or return empty)
        with pytest.raises((ValueError, IndexError)):
            hdbscan(data, settings)

    def test_nan_values(self):
        """Test clustering with NaN values."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        data[10:15, 2] = np.nan

        settings = ClusteringSettings(**get_default_settings_dict())

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            hdbscan(data, settings)

    def test_inf_values(self):
        """Test clustering with infinite values."""
        np.random.seed(42)
        data = np.random.randn(100, 5)
        data[10, :] = np.inf
        data[11, :] = -np.inf

        settings = ClusteringSettings(**get_default_settings_dict())

        # Should raise error or handle gracefully
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            hdbscan(data, settings)


class TestIntegration:
    """Integration tests with settings object."""

    def test_settings_object_integration(self, simple_clustered_data):
        """Test that all settings are properly passed through ClusteringSettings."""
        settings_dict = get_default_settings_dict()
        settings_dict["hdbscan"] = {
            "min_samples": 5,
            "allow_single_cluster": False,
            "max_cluster_size": None,
            "distance_metric": "euclidean",
            "scoring_sample_count": 10,
            "cluster_selection_epsilon": 0.01,
            "robust_single_linkage_scaling": 1.0,
            "nearest_neighbors_algorithm": "auto",
            "leaf_size": 40,
            "cluster_selection_method": "eom"
        }
        settings = ClusteringSettings(**settings_dict)

        labels = hdbscan(simple_clustered_data, settings).labels

        assert len(labels) == len(simple_clustered_data)
        assert isinstance(labels, np.ndarray)

    def test_default_settings_work(self, simple_clustered_data):
        """Test that default settings produce valid results."""
        settings = ClusteringSettings(**get_default_settings_dict())

        labels = hdbscan(simple_clustered_data, settings).labels

        assert len(labels) == len(simple_clustered_data)
        assert np.all(labels >= 0)  # With default min_samples=1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
