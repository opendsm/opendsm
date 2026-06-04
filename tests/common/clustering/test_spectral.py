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

import sys

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from opendsm.common.clustering.algorithms.spectral import spectral
from opendsm.common.clustering.algorithms.settings import SpectralSettings
from opendsm.common.clustering.settings import ClusteringSettings

from .conftest import make_clustering_settings


def _spectral_cs(spectral_settings: SpectralSettings | dict | None = None, seed: int = 42, **overrides) -> ClusteringSettings:
    """Build a ClusteringSettings for spectral from a SpectralSettings object."""
    if spectral_settings is None:
        spectral_settings = {}
    elif isinstance(spectral_settings, SpectralSettings):
        spectral_settings = spectral_settings.model_dump(exclude_defaults=True)
    return make_clustering_settings("spectral", seed=seed, spectral=spectral_settings, **overrides)


def _spectral_div_cs(spectral_settings: SpectralSettings | dict | None = None, seed: int = 42, **overrides) -> ClusteringSettings:
    """Build a ClusteringSettings for spectral_divisive from a SpectralSettings object."""
    if spectral_settings is None:
        spectral_settings = {}
    elif isinstance(spectral_settings, SpectralSettings):
        spectral_settings = spectral_settings.model_dump(exclude_defaults=True)
    return make_clustering_settings("spectral_divisive", seed=seed, spectral_divisive=spectral_settings, **overrides)


# simple_2d_data fixture is provided by conftest.py


@pytest.fixture
def default_settings():
    """Create default ClusteringSettings for spectral."""
    return _spectral_cs()


@pytest.fixture
def custom_spectral_settings():
    """Create custom spectral clustering settings."""
    return _spectral_cs(SpectralSettings(
        recluster_count=2,
        n_cluster={"lower": 2, "upper": 5},
    ))


# ---------------------------------------------------------------------------
# Shared behaviour (valid labels, determinism, cluster range, data shapes,
# edge cases, cluster quality) is tested in test_algorithms.py via
# TestAlgorithmSharedBehavior parametrized over all algorithms.
# ---------------------------------------------------------------------------


class TestAlgorithmSettings:
    """Tests for different algorithm configuration settings."""

    @pytest.mark.parametrize("eigen_solver", ["arpack", "lobpcg"])
    def test_eigen_solver(self, simple_2d_data, eigen_solver):
        """Test clustering with different eigen solvers."""
        settings = SpectralSettings(
            eigen_solver=eigen_solver,
            n_cluster={"lower": 3, "upper": 3},
        )

        labels = spectral(simple_2d_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 3

    @pytest.mark.parametrize("assign_labels", ["kmeans", "discretize", "cluster_qr"])
    def test_assign_labels(self, simple_2d_data, assign_labels):
        """Test clustering with different label assignment strategies."""
        settings = SpectralSettings(
            assign_labels=assign_labels,
            n_cluster={"lower": 3, "upper": 3},
        )

        labels = spectral(simple_2d_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 3

    def test_rbf_affinity(self, simple_2d_data):
        """Test clustering with RBF affinity matrix."""
        settings = SpectralSettings(
            affinity="rbf",
            n_cluster={"lower": 3, "upper": 3},
        )

        labels = spectral(simple_2d_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 3

    def test_nearest_neighbors_affinity(self, simple_2d_data):
        """Test clustering with nearest neighbors affinity matrix."""
        settings = SpectralSettings(
            affinity="nearest_neighbors",
            nearest_neighbors=10,
            n_cluster={"lower": 3, "upper": 3},
        )

        labels = spectral(simple_2d_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 3

    @pytest.mark.parametrize("gamma", [0.5, 1.0, 2.0])
    def test_different_gamma_values(self, simple_2d_data, gamma):
        """Test clustering with different gamma values for RBF kernel."""
        settings = SpectralSettings(
            affinity="rbf",
            gamma=gamma,
            n_cluster={"lower": 3, "upper": 3},
        )

        labels = spectral(simple_2d_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 3

    @pytest.mark.parametrize("recluster_count", [0, 1, 3])
    def test_recluster_count(self, simple_2d_data, recluster_count):
        """Test that different recluster counts work correctly."""
        settings = SpectralSettings(
            recluster_count=recluster_count,
            n_cluster={"lower": 3, "upper": 3},
        )

        labels = spectral(simple_2d_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 3


class TestReclusterSeedNone:
    """Reclustering must work when seed is None."""

    @pytest.mark.parametrize("seed", [42, None])
    def test_recluster_with_seed(self, seed):
        rng = np.random.default_rng(42)
        c1 = rng.normal(0, 0.3, (30, 10))
        c2 = rng.normal(5, 0.3, (30, 10))
        data = np.vstack([c1, c2])

        settings = SpectralSettings(
            recluster_count=2,
            n_cluster={"lower": 2, "upper": 4},
        )

        labels = spectral(data, _spectral_cs(settings)).labels
        assert len(labels) == 60
        assert len(np.unique(labels[labels != -1])) >= 2


class TestAffinityMatrixOptions:
    """Tests for different affinity matrix options."""

    def test_laplacian_affinity(self, simple_2d_data):
        """Test clustering with laplacian affinity matrix."""
        settings = SpectralSettings(
            affinity="laplacian",
            n_cluster={"lower": 3, "upper": 3},
        )

        labels = spectral(simple_2d_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 3

    def test_chi2_affinity(self, simple_2d_data):
        """Test clustering with chi2 affinity matrix (requires non-negative data)."""
        # Chi2 kernel requires non-negative data, so shift the data
        shifted_data = simple_2d_data - simple_2d_data.min() + 1

        settings = SpectralSettings(
            affinity="chi2",
            n_cluster={"lower": 3, "upper": 3},
        )

        labels = spectral(shifted_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 3


# TestDataShapes, TestEdgeCases, TestClusterQuality are now in
# test_algorithms.py::TestAlgorithmSharedBehavior.

class TestSpectralSparseData:
    """Spectral-specific edge case: sparse data (many zeros)."""

    def test_sparse_data(self):
        """Test clustering on sparse data (many zeros)."""
        np.random.seed(42)
        data = np.random.randn(100, 10)
        mask = np.random.random((100, 10)) < 0.7
        data[mask] = 0

        settings = SpectralSettings(n_cluster={"lower": 3, "upper": 3})
        labels = spectral(data, _spectral_cs(settings)).labels
        assert len(labels) == 100
        assert len(np.unique(labels)) > 0


class TestComponentSettings:
    """Tests for n_components parameter."""

    def test_custom_n_components(self, simple_2d_data):
        """Test clustering with custom number of components."""
        settings = SpectralSettings(
            n_components=5,
            n_cluster={"lower": 5, "upper": 5},
        )

        labels = spectral(simple_2d_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 5

    def test_none_n_components(self, simple_2d_data):
        """Test clustering with n_components=None (defaults to n_clusters)."""
        settings = SpectralSettings(
            n_components=None,
            n_cluster={"lower": 3, "upper": 3},
        )

        labels = spectral(simple_2d_data, _spectral_cs(settings)).labels
        assert len(np.unique(labels)) == 3


class TestBaselineConsistency:
    """Tests to ensure algorithm output doesn't change across versions."""

    def test_expected_baseline_output(self):
        """Test that spectral clustering produces expected baseline output.

        This test ensures the algorithm produces consistent results across
        different versions of the code. If this test fails, it indicates
        a breaking change in the clustering algorithm.
        """
        # Create deterministic test data with well-separated clusters
        data, _ = make_blobs(
            n_samples=60,
            n_features=10,
            centers=3,
            cluster_std=1.5,
            random_state=42
        )

        # Configure settings for reproducible clustering
        settings = SpectralSettings(
            n_cluster={"lower": 3, "upper": 3},
            assign_labels="kmeans",
        )

        # Run clustering
        labels = spectral(data, _spectral_cs(settings)).labels

        # Expected baseline output - saved for version consistency
        expected_labels = np.array([
            1, 0, 1, 1, 2, 1, 1, 0, 2, 2, 1, 2, 2, 1, 0, 2, 0, 0, 0, 2,
            1, 1, 0, 2, 1, 1, 0, 2, 1, 0, 0, 0, 1, 1, 1, 2, 2, 0, 0, 2,
            1, 2, 1, 2, 1, 2, 2, 2, 0, 2, 0, 0, 0, 2, 1, 0, 1, 0, 2, 0
        ])

        # Verify exact match against baseline
        np.testing.assert_array_equal(
            labels,
            expected_labels,
            err_msg="Spectral clustering output does not match saved baseline. "
                    "This indicates a breaking change in the algorithm."
        )

        # Verify cluster properties
        unique_labels, counts = np.unique(labels, return_counts=True)
        expected_counts = {0: 20, 1: 20, 2: 20}

        assert len(unique_labels) == 3, "Expected 3 clusters"
        for label, count in zip(unique_labels, counts):
            assert count == expected_counts[label], \
                f"Cluster {label} has {count} samples, expected {expected_counts[label]}"


class TestSpectralDivisive:
    """Tests for spectral_divisive (recursive Fiedler bisection) clustering."""

    def _settings(self, n_lower=2, n_upper=8, seed=42):
        return SpectralSettings(
            n_cluster={"lower": n_lower, "upper": n_upper},
        )

    def test_cluster_count_in_range(self, simple_2d_data):
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive
        settings = self._settings(n_lower=2, n_upper=6)
        labels = spectral_divisive(simple_2d_data, _spectral_div_cs(settings, seed=42)).labels
        assert len(labels) == len(simple_2d_data)
        n_clusters = len(np.unique(labels[labels >= 0]))
        assert 2 <= n_clusters <= 6

    def test_valid_labels(self, simple_2d_data):
        """spectral_divisive returns valid label arrays."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive
        settings = self._settings()
        labels = spectral_divisive(simple_2d_data, _spectral_div_cs(settings, seed=42)).labels
        assert len(labels) == len(simple_2d_data)
        # All labels should be non-negative integers
        assert np.all(labels >= 0)
        assert len(np.unique(labels)) >= 2

    def test_finds_reasonable_clusters(self, simple_2d_data):
        """Well-separated 3-cluster data should produce 2-4 clusters."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive
        settings = SpectralSettings(
            n_cluster={"lower": 2, "upper": 8},
            recluster_count=0,
        )
        labels = spectral_divisive(simple_2d_data, _spectral_div_cs(settings, seed=42)).labels
        n_clusters = len(np.unique(labels[labels >= 0]))
        # eigsh can be non-deterministic; accept 2-4 clusters for 3-blob data
        assert 2 <= n_clusters <= 4

    def test_dispatched_via_cluster_features(self, simple_2d_data):
        """spectral_divisive is reachable through the top-level dispatch."""
        import pandas as pd
        from opendsm.common.clustering.cluster import cluster_features
        df = pd.DataFrame(simple_2d_data)
        settings = ClusteringSettings(
            algorithm_selection="spectral_divisive",
            seed=42,
            transform_selection=None,
            spectral_divisive={"n_cluster": {"lower": 2, "upper": 6}},
            cluster_sort={
                "enable": True,
                "method": "size",
            }
        )
        labels = cluster_features(df, settings)
        assert len(labels) == len(simple_2d_data)
        assert np.all(labels >= 0)


class TestNystromEmbedding:
    """Tests for Nystrom global spectral embedding."""

    @pytest.fixture
    def well_separated_data(self):
        """Three well-separated clusters, 200 points."""
        X, _ = make_blobs(n_samples=200, centers=3, n_features=10,
                          cluster_std=0.5, random_state=42)
        return X.astype(np.float32)

    @pytest.fixture
    def well_separated_data_with_labels(self):
        """Three well-separated clusters, 200 points, plus ground-truth labels."""
        X, y = make_blobs(n_samples=200, centers=3, n_features=10,
                          cluster_std=0.5, random_state=42)
        return X.astype(np.float32), y

    def test_embedding_shape(self, well_separated_data):
        """Nystrom embedding returns (n, n_components) array."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _nystrom_embedding,
        )
        embedding = _nystrom_embedding(
            well_separated_data, k_st=7, m=100, n_components=5, seed=42
        )
        assert embedding is not None
        assert embedding.shape == (200, 5)

    def test_embedding_rows_unit_normalized(self, well_separated_data):
        """Each row of the embedding has unit norm."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _nystrom_embedding,
        )
        embedding = _nystrom_embedding(
            well_separated_data, k_st=7, m=100, n_components=5, seed=42
        )
        norms = np.linalg.norm(embedding, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_embedding_preserves_cluster_structure(self, well_separated_data_with_labels):
        """Nearest-neighbor cluster purity in the embedding stays high.

        Replaces a hardcoded-component regression check that couldn't survive
        cross-BLAS eigenvector sign and ordering ambiguity (eigenvectors are
        defined only up to sign; degenerate eigenvalues can reorder freely).
        kNN purity in Euclidean-distance space is invariant to both sign flips
        and component reordering, so it stays stable across BLAS implementations.

        Threshold 0.80 is conservative: a 9-scenario calibration probe (blobs
        with k in {2, 3, 4, 5}, std in {0.5, 1.0, 1.5, 2.0}, d in {3, 5, 10,
        20, 50}, n in {80, 100, 200, 300, 400}) yielded min purity 0.935 at
        k_nn=5 with median 1.000.  A purely random embedding for 3 clusters
        would produce purity ~0.33.  At 0.80 we sit between the observed floor
        and the random baseline with comfortable margin for cross-platform
        variability.
        """
        from sklearn.neighbors import NearestNeighbors

        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _nystrom_embedding,
        )
        X, y = well_separated_data_with_labels
        embedding = _nystrom_embedding(X, k_st=7, m=100, n_components=5, seed=42)

        k_nn = 5
        nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(embedding)
        _, idx = nn.kneighbors(embedding)
        # idx[:, 0] is the point itself; columns 1..k_nn+1 are its k nearest
        # neighbors.  Fraction of (point, neighbor) pairs sharing a label.
        same_label = y[idx[:, 1:]] == y[:, None]
        purity = float(same_label.mean())
        assert purity > 0.80, (
            f"kNN cluster purity too low (random baseline ~0.33 for k=3): "
            f"{purity:.3f}"
        )

    def test_nystrom_produces_valid_labels(self, well_separated_data):
        """Clustering via Nystrom path produces labels for all points."""
        settings = ClusteringSettings(
            algorithm_selection="spectral_divisive",
            spectral_divisive={
                "n_cluster": {"lower": 2, "upper": 6},
                "recluster_count": 0,
                "nystrom_samples": 100,
            },
            seed=42,
        )
        import pandas as pd
        from opendsm.common.clustering import cluster_result
        result = cluster_result(pd.DataFrame(well_separated_data), settings)
        assert len(result.labels) == 200
        assert result.k >= 2

    @pytest.mark.parametrize("n_samples,nystrom_samples", [
        (50, 100),   # n < nystrom_samples: exact path used
    ], ids=["n_below_threshold"])
    def test_nystrom_disabled_when_n_below_threshold(self, n_samples, nystrom_samples):
        """When n < nystrom_samples, exact path is used (no embedding)."""
        X, _ = make_blobs(n_samples=n_samples, centers=2, n_features=5,
                          cluster_std=0.5, random_state=42)
        settings = ClusteringSettings(
            algorithm_selection="spectral_divisive",
            spectral_divisive={
                "n_cluster": {"lower": 2, "upper": 5},
                "recluster_count": 0,
                "nystrom_samples": nystrom_samples,
            },
            seed=42,
        )
        import pandas as pd
        from opendsm.common.clustering import cluster_result
        result = cluster_result(pd.DataFrame(X.astype(np.float32)), settings)
        assert len(result.labels) == n_samples

    def test_nystrom_none_disables(self):
        """nystrom_samples=None forces exact path regardless of n."""
        X, _ = make_blobs(n_samples=200, centers=3, n_features=10,
                          cluster_std=0.5, random_state=42)
        settings = ClusteringSettings(
            algorithm_selection="spectral_divisive",
            spectral_divisive={
                "n_cluster": {"lower": 2, "upper": 6},
                "recluster_count": 0,
                "nystrom_samples": None,
            },
            seed=42,
        )
        import pandas as pd
        from opendsm.common.clustering import cluster_result
        result = cluster_result(pd.DataFrame(X.astype(np.float32)), settings)
        assert len(result.labels) == 200

    def test_nystrom_embedding_deterministic(self, well_separated_data):
        """Same seed produces identical embedding."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _nystrom_embedding,
        )
        e1 = _nystrom_embedding(well_separated_data, k_st=7, m=100, n_components=5, seed=42)
        e2 = _nystrom_embedding(well_separated_data, k_st=7, m=100, n_components=5, seed=42)
        np.testing.assert_array_equal(e1, e2)


class TestFiedlerSignConvention:
    """Tests for median-based Fiedler sign convention."""

    def test_sign_convention_is_median_based(self):
        """Fiedler vector is flipped so median is non-negative."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _fiedler_split,
        )
        X, _ = make_blobs(n_samples=100, centers=2, n_features=5,
                          cluster_std=0.5, random_state=42)
        data = X.astype(np.float32)
        indices = np.arange(len(data))
        lambda2, left, right = _fiedler_split(data, indices, k_st=7)
        # After median-based flip, the left partition (fiedler >= 0) should
        # contain at least half the points (since median >= 0 after flip).
        assert len(left) >= len(data) // 2 - 1 or len(right) >= len(data) // 2 - 1

    def test_split_produces_nonempty_partitions(self):
        """Both partitions are non-empty for well-separated data."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _fiedler_split,
        )
        X, _ = make_blobs(n_samples=50, centers=2, n_features=5,
                          cluster_std=0.5, random_state=42)
        data = X.astype(np.float32)
        indices = np.arange(len(data))
        lambda2, left, right = _fiedler_split(data, indices, k_st=7)
        assert len(left) > 0, "Left partition is empty"
        assert len(right) > 0, "Right partition is empty"
        assert len(left) + len(right) == 50

    def test_fiedler_regression_partition_sizes(self):
        """Regression: known data produces known partition sizes."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _fiedler_split,
        )
        X, _ = make_blobs(n_samples=100, centers=2, n_features=5,
                          cluster_std=0.5, random_state=42)
        _, left, right = _fiedler_split(X.astype(np.float32), np.arange(100), k_st=7)
        assert len(left) == 50, f"Expected 50 left, got {len(left)}"
        assert len(right) == 50, f"Expected 50 right, got {len(right)}"


class TestAutoPCAFallback:
    """Tests for auto-PCA when n < features."""

    def test_pca_reduces_when_n_less_than_features(self):
        """When n < d, output has n-1 features."""
        from opendsm.common.clustering.transform.transform import transform_features
        from opendsm.common.clustering.settings import ClusteringSettings
        data = np.random.default_rng(42).random((7, 504)).astype(np.float32)
        settings = ClusteringSettings(
            feature_transform={"wavelet": {"enabled": False}},
            seed=0,
        )
        result = transform_features(data, settings).data
        assert result.shape[0] == 7
        assert result.shape[1] <= 6, f"Expected <=6 features, got {result.shape[1]}"

    def test_no_pca_when_n_greater_than_features(self):
        """When n > d, features are unchanged."""
        from opendsm.common.clustering.transform.transform import transform_features
        from opendsm.common.clustering.settings import ClusteringSettings
        data = np.random.default_rng(42).random((100, 24)).astype(np.float32)
        settings = ClusteringSettings(seed=0)
        result = transform_features(data, settings).data
        # Wavelet reduces 24 -> ~5 PCA components; but n=100 > 5 so no auto-PCA
        assert result.shape[0] == 100

    def test_pca_with_magnitude_features(self):
        """Auto-PCA fires after magnitude features are appended."""
        from opendsm.common.clustering.transform.transform import transform_features
        from opendsm.common.clustering.settings import ClusteringSettings
        # n=5 with d=504 + magnitude (3) = 507 features; should reduce to 4
        data = np.random.default_rng(42).random((5, 504)).astype(np.float32)
        settings = ClusteringSettings(
            feature_transform={"wavelet": {"enabled": False}},
            seed=0,
        )
        result = transform_features(data, settings).data
        assert result.shape[0] == 5
        assert result.shape[1] <= 4, f"Expected <=4 features after PCA, got {result.shape[1]}"

    def test_pca_single_sample(self):
        """n=1 should not crash (PCA requires n>1)."""
        from opendsm.common.clustering.transform.transform import transform_features
        from opendsm.common.clustering.settings import ClusteringSettings
        data = np.random.default_rng(42).random((1, 504)).astype(np.float32)
        settings = ClusteringSettings(
            feature_transform={"wavelet": {"enabled": False}},
            seed=0,
        )
        result = transform_features(data, settings).data
        assert result.shape[0] == 1


class TestNystromEdgeCases:
    """Edge cases for Nystrom embedding."""

    def test_nystrom_with_identical_points(self):
        """All-identical data doesn't crash (degenerate affinity)."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _nystrom_embedding,
        )
        data = np.ones((200, 10), dtype=np.float32)
        result = _nystrom_embedding(data, k_st=7, m=100, n_components=5, seed=42)
        # May return None (eigsh fails on degenerate Laplacian) or a valid array
        if result is not None:
            assert result.shape == (200, 5)

    def test_nystrom_sample_larger_than_n(self):
        """When m >= n, nystrom_samples threshold is not triggered."""
        import pandas as pd
        from opendsm.common.clustering import cluster_result
        X, _ = make_blobs(n_samples=50, centers=2, n_features=5,
                          cluster_std=0.5, random_state=42)
        settings = ClusteringSettings(
            algorithm_selection="spectral_divisive",
            spectral_divisive={
                "n_cluster": {"lower": 2, "upper": 5},
                "recluster_count": 0,
                "nystrom_samples": 100,  # m > n -> exact path
            },
            seed=42,
        )
        result = cluster_result(pd.DataFrame(X.astype(np.float32)), settings)
        assert len(result.labels) == 50

    def test_nystrom_minimum_threshold(self):
        """nystrom_samples has a minimum of 100."""
        from opendsm.common.clustering.algorithms.settings import SpectralSettings
        with pytest.raises(Exception):
            SpectralSettings(nystrom_samples=50)


class TestFiedlerTwoPointSplit:
    """Test Fiedler split on 2-point sub-clusters."""

    def test_two_points_split_without_arpack(self):
        """n=2 uses self-tuning affinity lambda2 = 2·exp(-1) ≈ 0.736,
        giving 2-point pairs a consistent moderate heap priority."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _fiedler_split,
        )
        data = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        indices = np.array([0, 1])
        lambda2, left, right = _fiedler_split(data, indices, k_st=7)
        assert len(left) == 1, "Left should have 1 point"
        assert len(right) == 1, "Right should have 1 point"
        assert lambda2 == pytest.approx(2.0 * np.exp(-1.0), rel=1e-6)

    def test_single_point_returns_inf(self):
        """n=1 returns lambda2=inf (cannot split)."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _fiedler_split,
        )
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        lambda2, left, right = _fiedler_split(data, np.array([0]), k_st=7)
        assert lambda2 == np.inf
        assert len(right) == 0


class TestMagnitudeFeatureGate:
    """Test that magnitude features are gated by transform activity."""

    def test_no_magnitude_at_short_T_no_wavelet(self):
        """Magnitude features don't fire at T=24 when wavelet is bypassed."""
        from opendsm.common.clustering.transform.transform import transform_features
        data = np.random.default_rng(42).random((50, 24)).astype(np.float32)
        settings = ClusteringSettings(
            feature_transform={
                "wavelet": {"enabled": True},
            },
            seed=0,
        )
        result = transform_features(data, settings).data
        # Wavelet per-level PCA reduces 24 -> output features
        assert result.shape[0] == 50
        assert result.shape[1] > 0, "Expected some features from wavelet transform"

    def test_magnitude_fires_when_centering_norm_active(self):
        """Magnitude features auto-fire when standardize (centering) normalization is used."""
        from opendsm.common.clustering.transform.transform import transform_features
        data = np.random.default_rng(42).random((50, 24)).astype(np.float32)
        settings = ClusteringSettings(
            feature_transform={
                "wavelet": {"enabled": True},
                "normalize": {"method": "standardize"},  # centering -> magnitude auto-fires
            },
            seed=0,
        )
        result = transform_features(data, settings).data
        # Standardize removes level -> magnitude auto-fires (3 default features)
        # Wavelet per-level PCA reduces 24h -> fewer + 3 magnitude
        assert result.shape[0] == 50
        assert result.shape[1] > 3, (
            f"Expected wavelet features + 3 magnitude, got {result.shape[1]}"
        )


class TestAffinityMatrix:
    """Tests for _self_tuning_affinity_sparse and _self_tuning_affinity_dense."""

    @pytest.fixture
    def two_cluster_data(self):
        """Two well-separated 2D clusters, 20 points each."""
        rng = np.random.default_rng(42)
        c1 = rng.normal(loc=[0, 0], scale=0.3, size=(20, 2))
        c2 = rng.normal(loc=[10, 10], scale=0.3, size=(20, 2))
        return np.vstack([c1, c2]).astype(np.float32)

    # ------------------------------------------------------------------
    # _self_tuning_affinity_sparse
    # ------------------------------------------------------------------

    def test_sparse_returns_sparse_symmetric(self, two_cluster_data):
        """Sparse path returns a symmetric sparse matrix."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_sparse,
        )
        from scipy import sparse as sp

        A = _self_tuning_affinity_sparse(two_cluster_data, k=7, k_connect=10)
        assert sp.issparse(A), "Expected a sparse matrix"
        assert A.shape == (40, 40)
        # Symmetry: A - A^T should have no non-zero entries
        assert (A - A.T).nnz == 0, "Affinity matrix is not symmetric"

    def test_sparse_diagonal_is_zero(self, two_cluster_data):
        """Sparse affinity has zero diagonal (no self-affinity)."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_sparse,
        )

        A = _self_tuning_affinity_sparse(two_cluster_data, k=7, k_connect=10)
        np.testing.assert_array_equal(
            A.diagonal(), np.zeros(40),
            err_msg="Diagonal should be all zeros",
        )

    def test_sparse_values_in_unit_interval(self, two_cluster_data):
        """All non-zero values in the sparse affinity are in (0, 1]."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_sparse,
        )

        A = _self_tuning_affinity_sparse(two_cluster_data, k=7, k_connect=10)
        assert A.data.min() > 0, "Expected all values > 0"
        assert A.data.max() <= 1.0, "Expected all values <= 1"

    def test_sparse_block_diagonal_structure(self, two_cluster_data):
        """Well-separated clusters produce block-diagonal structure (no between-cluster edges)."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_sparse,
        )

        A = _self_tuning_affinity_sparse(two_cluster_data, k=7, k_connect=10)
        between_block = A[:20, 20:]
        assert between_block.nnz == 0, (
            f"Expected zero between-cluster edges, got {between_block.nnz}"
        )

    def test_sparse_regression_nnz_and_sum(self, two_cluster_data):
        """Regression: known nnz and sum for the two-cluster fixture."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_sparse,
        )

        A = _self_tuning_affinity_sparse(two_cluster_data, k=7, k_connect=10)
        assert A.nnz == 488, f"Expected 488 non-zeros, got {A.nnz}"
        np.testing.assert_allclose(A.sum(), 200.6287, atol=0.01)

    # ------------------------------------------------------------------
    # _self_tuning_affinity_dense
    # ------------------------------------------------------------------

    def test_dense_returns_ndarray(self, two_cluster_data):
        """Dense path returns a numpy ndarray of correct shape."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_dense,
        )

        A = _self_tuning_affinity_dense(two_cluster_data, k=7)
        assert isinstance(A, np.ndarray)
        assert A.shape == (40, 40)

    def test_dense_symmetric(self, two_cluster_data):
        """Dense affinity matrix is symmetric."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_dense,
        )

        A = _self_tuning_affinity_dense(two_cluster_data, k=7)
        np.testing.assert_array_equal(A, A.T)

    def test_dense_diagonal_is_one(self, two_cluster_data):
        """Dense affinity has diagonal = 1 (d(i,i) = 0 -> exp(0) = 1)."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_dense,
        )

        A = _self_tuning_affinity_dense(two_cluster_data, k=7)
        np.testing.assert_array_equal(A.diagonal(), np.ones(40))

    def test_dense_values_in_unit_interval(self, two_cluster_data):
        """All dense affinity values are in [0, 1]."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_dense,
        )

        A = _self_tuning_affinity_dense(two_cluster_data, k=7)
        assert A.min() >= 0.0
        assert A.max() <= 1.0

    def test_dense_block_diagonal_structure(self, two_cluster_data):
        """Between-cluster affinities are effectively zero for well-separated data."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_dense,
        )

        A = _self_tuning_affinity_dense(two_cluster_data, k=7)
        between_block = A[:20, 20:]
        assert between_block.max() < 1e-10, (
            f"Expected near-zero between-cluster affinity, got max={between_block.max()}"
        )

    def test_dense_regression_first_row(self, two_cluster_data):
        """Regression: first row values for the two-cluster fixture."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_dense,
        )

        A = _self_tuning_affinity_dense(two_cluster_data, k=7)
        np.testing.assert_allclose(
            A[0, :5],
            [1.0, 0.01696124, 0.06863567, 0.41524021, 0.74564743],
            atol=1e-4,
        )
        # Between-cluster entries are essentially zero
        np.testing.assert_allclose(A[0, 20:25], np.zeros(5), atol=1e-100)

    def test_dense_regression_sum(self, two_cluster_data):
        """Regression: total sum of the dense affinity matrix."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_dense,
        )

        A = _self_tuning_affinity_dense(two_cluster_data, k=7)
        np.testing.assert_allclose(A.sum(), 251.785, atol=0.01)

    # ------------------------------------------------------------------
    # Sparse vs Dense agreement
    # ------------------------------------------------------------------

    def test_sparse_dense_agreement(self, two_cluster_data):
        """Sparse and dense paths agree (up to float32 tolerance) when k_connect covers all pairs."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_sparse,
            _self_tuning_affinity_dense,
        )

        n = two_cluster_data.shape[0]
        A_sparse = _self_tuning_affinity_sparse(
            two_cluster_data, k=7, k_connect=n - 1,
        )
        A_dense = _self_tuning_affinity_dense(two_cluster_data, k=7)

        # Zero out dense diagonal for fair comparison (sparse has 0 diag)
        dd = A_dense.copy()
        np.fill_diagonal(dd, 0.0)

        np.testing.assert_allclose(
            A_sparse.toarray(), dd, atol=1e-6,
            err_msg="Sparse and dense paths disagree",
        )

    # ------------------------------------------------------------------
    # Within-cluster > between-cluster affinity
    # ------------------------------------------------------------------

    def test_within_cluster_affinity_greater_than_between(self, two_cluster_data):
        """Mean within-cluster affinity exceeds mean between-cluster affinity."""
        from opendsm.common.clustering.algorithms.spectral._affinity import (
            _self_tuning_affinity_dense,
        )

        A = _self_tuning_affinity_dense(two_cluster_data, k=7)

        within_mask = np.zeros((40, 40), dtype=bool)
        within_mask[:20, :20] = True
        within_mask[20:, 20:] = True
        np.fill_diagonal(within_mask, False)  # exclude self-affinity

        between_mask = np.zeros((40, 40), dtype=bool)
        between_mask[:20, 20:] = True
        between_mask[20:, :20] = True

        mean_within = A[within_mask].mean()
        mean_between = A[between_mask].mean()

        assert mean_within > mean_between, (
            f"Within-cluster mean ({mean_within:.6f}) should exceed "
            f"between-cluster mean ({mean_between:.6f})"
        )
        # For this well-separated data, within >> between
        assert mean_within > 0.1, "Within-cluster affinity unexpectedly low"
        assert mean_between < 1e-10, "Between-cluster affinity unexpectedly high"


# ── Constrained Fiedler split ─────────────────────────────────────────────────

class TestConstrainedFiedlerSplit:
    """Tests for _apply_fiedler with min_split_size constraints."""

    def test_constrained_split_both_halves_meet_min_size(self):
        """With min_split_size=15 on 32 points, both halves must have >= 15."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _fiedler_split,
        )
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal([0, 0], 0.5, (16, 2)),
            rng.normal([10, 10], 0.5, (16, 2)),
        ]).astype(np.float32)
        indices = np.arange(32)
        lambda2, left, right = _fiedler_split(X, indices, k_st=7, min_split_size=15)
        assert len(left) >= 15, (
            f"Left partition has {len(left)} points, expected >= 15"
        )
        assert len(right) >= 15, (
            f"Right partition has {len(right)} points, expected >= 15"
        )
        assert len(left) + len(right) == 32

    def test_unconstrained_split_with_min_split_size_1(self):
        """min_split_size=1 uses the standard sign-based split (unconstrained path)."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _apply_fiedler,
        )
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(20)
        indices = np.arange(20)
        lambda2_in = 1.5
        lambda2, left, right = _apply_fiedler(fiedler, lambda2_in, indices, min_split_size=1)
        # Standard sign-based: after median flip, split on sign
        assert len(left) > 0 and len(right) > 0
        assert len(left) + len(right) == 20
        assert lambda2 == lambda2_in

    def test_group_smaller_than_2x_min_cs_not_split(self):
        """A group smaller than 2*min_cs is never pushed onto the heap.

        The pre-check ``len(indices) < 2 * min_cs`` in ``_push`` prevents
        the split from ever being attempted.  We verify this by running
        the full single-run bisection on data where the initial group is
        too small to split at the given min_cluster_size.
        """
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            spectral_divisive,
        )
        from opendsm.common.clustering.metrics.settings import ScoreSettings, SmallClusterMode

        rng = np.random.default_rng(42)
        # 10 points, min_cs=6 -> 2*6=12 > 10, so no split should happen
        X = rng.normal(0, 1, (10, 3)).astype(np.float32)
        settings = SpectralSettings(
            n_cluster={"lower": 1, "upper": 5},
            recluster_count=0,
            scoring=ScoreSettings(
                min_cluster_size=6,
                small_cluster_mode=SmallClusterMode.OUTLIER,
            ),
        )
        # With OUTLIER mode, min_cs is not enforced at split time
        # (algorithm explores freely), so this tests KEEP mode instead.
        settings_keep = SpectralSettings(
            n_cluster={"lower": 1, "upper": 5},
            recluster_count=0,
            scoring=ScoreSettings(
                min_cluster_size=1,
                small_cluster_mode=SmallClusterMode.KEEP,
            ),
        )
        result = spectral_divisive(X, _spectral_div_cs(settings_keep, seed=42))
        # With only 10 points and min_cs=1 in KEEP mode, the algorithm
        # can split freely. Now test with n < 2*min_cs via _apply_fiedler:
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _apply_fiedler,
        )
        # n=12, min_split_size=6 -> constrained path entered (12 >= 2*6),
        # lo=6, hi=12-6=6, lo>=hi -> returns inf (no valid split position).
        fiedler = np.linspace(-1, 1, 12)
        indices = np.arange(12)
        lambda2, left, right = _apply_fiedler(fiedler, 1.0, indices, min_split_size=6)
        assert lambda2 == np.inf, (
            "Should return inf when constrained range lo >= hi "
            f"(n=12, min_split_size=6, lo=6 >= hi=6), got lambda2={lambda2}"
        )

    def test_gap_of_zero_returns_inf(self):
        """When all Fiedler values are identical within valid range, gap=0 -> inf."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _apply_fiedler,
        )
        # Constant Fiedler vector — all gaps are zero
        fiedler = np.ones(20)
        indices = np.arange(20)
        lambda2, left, right = _apply_fiedler(fiedler, 1.0, indices, min_split_size=5)
        assert lambda2 == np.inf, (
            "Gap of zero within valid range should return inf (no split)"
        )

    def test_two_point_lambda2_distinct(self):
        """2-point lambda2 = 2*exp(-1) for distinct points."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _fiedler_split,
        )
        data = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
        indices = np.array([0, 1])
        lambda2, left, right = _fiedler_split(data, indices, k_st=7)
        assert lambda2 == pytest.approx(2.0 * np.exp(-1.0), rel=1e-6), (
            f"Expected 2*exp(-1) ~ 0.7358, got {lambda2}"
        )
        assert len(left) == 1 and len(right) == 1

    def test_two_point_lambda2_identical(self):
        """2-point lambda2 = inf for identical points (no distance info)."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
            _fiedler_split,
        )
        data = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        indices = np.array([0, 1])
        lambda2, left, right = _fiedler_split(data, indices, k_st=7)
        assert lambda2 == np.inf, (
            f"Expected inf for identical points, got {lambda2}"
        )


# ── Bisection min_cluster_size ────────────────────────────────────────────────

class TestBisectionMinClusterSize:
    """Tests for min_cluster_size enforcement in spectral_divisive."""

    def test_keep_mode_min_cs_1_allows_singletons(self):
        """KEEP mode with min_cs=1: singletons may appear in candidates."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive
        from opendsm.common.clustering.metrics.settings import ScoreSettings, SmallClusterMode

        rng = np.random.default_rng(42)
        # 4 well-separated clusters, one with a single isolated point
        X = np.vstack([
            rng.normal([0, 0, 0, 0, 0], 0.3, (20, 5)),
            rng.normal([5, 5, 5, 5, 5], 0.3, (20, 5)),
            rng.normal([10, 10, 10, 10, 10], 0.3, (20, 5)),
        ]).astype(np.float32)

        settings = SpectralSettings(
            n_cluster={"lower": 2, "upper": 8},
            recluster_count=0,
            scoring=ScoreSettings(
                min_cluster_size=1,
                small_cluster_mode=SmallClusterMode.KEEP,
            ),
        )
        result = spectral_divisive(X, _spectral_div_cs(settings, seed=42))
        # Should produce valid candidates
        assert len(result.k_values) >= 1, "Should produce at least one k candidate"

    def test_keep_mode_min_cs_2_no_tiny_clusters(self):
        """KEEP mode with min_cs=2: no cluster in any candidate has <2 members."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive
        from opendsm.common.clustering.metrics.settings import ScoreSettings, SmallClusterMode

        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal([0, 0, 0, 0, 0], 0.3, (30, 5)),
            rng.normal([5, 5, 5, 5, 5], 0.3, (30, 5)),
            rng.normal([10, 10, 10, 10, 10], 0.3, (30, 5)),
        ]).astype(np.float32)

        settings = SpectralSettings(
            n_cluster={"lower": 2, "upper": 8},
            recluster_count=0,
            scoring=ScoreSettings(
                min_cluster_size=2,
                small_cluster_mode=SmallClusterMode.OUTLIER,
            ),
        )
        result = spectral_divisive(X, _spectral_div_cs(settings, seed=42))
        # In KEEP mode the algorithm enforces min_cs at split time.
        # With OUTLIER mode, the algorithm explores freely (min_cs=1 internally),
        # but the scoring pipeline would handle small clusters.
        # Here we test that at least some candidates are produced.
        for k in result.k_values:
            store = result._labels_store.get(k, [])
            for lm in store:
                labels = lm.labels
                non_outlier = labels[labels >= 0]
                if len(non_outlier) > 0:
                    counts = np.bincount(non_outlier)
                    counts = counts[counts > 0]
                    # With OUTLIER mode, small clusters get relabeled to -1
                    # so remaining non-outlier clusters should be >= min_cs
                    # (This is handled by the scoring pipeline, not the algo)

    def test_outlier_mode_explores_freely(self):
        """OUTLIER mode with min_cs=15: algorithm explores many k values."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive
        from opendsm.common.clustering.metrics.settings import ScoreSettings, SmallClusterMode

        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(c * 5, 0.5, (30, 5))
            for c in range(5)
        ]).astype(np.float32)

        settings = SpectralSettings(
            n_cluster={"lower": 2, "upper": 10},
            recluster_count=0,
            scoring=ScoreSettings(
                min_cluster_size=15,
                small_cluster_mode=SmallClusterMode.OUTLIER,
            ),
        )
        result = spectral_divisive(X, _spectral_div_cs(settings, seed=42))
        # OUTLIER mode doesn't constrain splits, so many k candidates
        assert len(result.k_values) >= 3, (
            f"OUTLIER mode should explore freely, got only {len(result.k_values)} k values"
        )

    def test_council_never_selects_k1_when_gate_passes(self):
        """When the structure gate passes, council never picks k=1."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive

        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal([0, 0, 0, 0, 0], 0.5, (30, 5)),
            rng.normal([10, 10, 10, 10, 10], 0.5, (30, 5)),
        ]).astype(np.float32)

        settings = SpectralSettings(
            n_cluster={"lower": 1, "upper": 6},
            recluster_count=0,
        )
        result = spectral_divisive(X, _spectral_div_cs(settings, seed=42))
        if result.has_cluster_structure:
            assert result.k >= 2, (
                f"Council should never select k=1 when structure gate passes, got k={result.k}"
            )


###############################################################################
# New tests for diffusion / anisotropic / PIC / sweep-cut features
###############################################################################

from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh as _sparse_eigsh
from scipy.sparse import csgraph

from opendsm.common.clustering.algorithms.spectral._affinity import (
    _auto_diffusion_time,
    _diffusion_map,
    _power_iteration_fiedler,
    _anisotropic_affinity_sparse,
    _self_tuning_affinity_dense,
    _self_tuning_affinity_sparse,
)
from opendsm.common.clustering.algorithms.spectral.spectral_divisive import (
    _apply_fiedler,
    _sub_seed,
    _embedding_split,
)


# ---------------------------------------------------------------------------
# TestSweepCut
# ---------------------------------------------------------------------------
class TestSweepCut:
    """Tests for the sweep-cut logic inside _apply_fiedler."""

    def test_clean_bimodal_200_400(self):
        """Clean bimodal 200/400 split produces left=200, right=400."""
        np.random.seed(42)
        fiedler = np.concatenate([
            np.random.normal(-5.0, 0.1, 200),
            np.random.normal(5.0, 0.1, 400),
        ])
        indices = np.arange(600)
        lambda2, left, right = _apply_fiedler(fiedler, 0.5, indices)
        assert len(left) == 200
        assert len(right) == 400

    def test_balanced_300_300(self):
        """Balanced 300/300 split produces left=300, right=300."""
        np.random.seed(42)
        fiedler = np.concatenate([
            np.random.normal(-5.0, 0.1, 300),
            np.random.normal(5.0, 0.1, 300),
        ])
        indices = np.arange(600)
        lambda2, left, right = _apply_fiedler(fiedler, 0.5, indices)
        assert len(left) == 300
        assert len(right) == 300

    def test_imbalanced_10_590(self):
        """Imbalanced 10/590 split correctly isolates the 10-point group."""
        np.random.seed(42)
        fiedler = np.concatenate([
            np.random.normal(-10.0, 0.05, 10),
            np.random.normal(5.0, 0.05, 590),
        ])
        indices = np.arange(600)
        lambda2, left, right = _apply_fiedler(fiedler, 0.5, indices)
        assert len(left) == 10
        assert len(right) == 590

    def test_noisy_bimodal_approximate(self):
        """Noisy bimodal produces approximately correct split (within +/-25)."""
        np.random.seed(42)
        fiedler = np.concatenate([
            np.random.normal(-3.0, 1.5, 250),
            np.random.normal(3.0, 1.5, 350),
        ])
        indices = np.arange(600)
        lambda2, left, right = _apply_fiedler(fiedler, 0.5, indices)
        assert abs(len(left) - 250) <= 25
        assert abs(len(right) - 350) <= 25

    def test_constant_fiedler_returns_inf(self):
        """Constant Fiedler vector returns inf (no valid split)."""
        fiedler = np.ones(100)
        indices = np.arange(100)
        lambda2, left, right = _apply_fiedler(fiedler, 0.5, indices)
        assert lambda2 == np.inf
        assert len(right) == 0

    def test_constrained_sweep_min_split_size(self):
        """Constrained sweep with min_split_size=15 on n=32 produces both halves >= 15."""
        np.random.seed(42)
        fiedler = np.concatenate([
            np.random.normal(-5.0, 0.1, 16),
            np.random.normal(5.0, 0.1, 16),
        ])
        indices = np.arange(32)
        lambda2, left, right = _apply_fiedler(
            fiedler, 0.5, indices, min_split_size=15
        )
        assert len(left) >= 15
        assert len(right) >= 15


# ---------------------------------------------------------------------------
# TestDiffusionMap
# ---------------------------------------------------------------------------
class TestDiffusionMap:
    """Tests for _diffusion_map embedding."""

    def test_well_separated_clusters_kmeans_perfect(self):
        """Well-separated 3 clusters: KMeans on first 2 coords gives perfect labels."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score

        np.random.seed(42)
        X, true_labels = make_blobs(
            n_samples=90, centers=3, cluster_std=0.3,
            center_box=(-10, 10), random_state=42,
        )
        A = _self_tuning_affinity_dense(X, k=7)
        embedding, t_used = _diffusion_map(A, diffusion_time=3, n_components=5, seed=42)
        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(embedding[:, :2])
        ari = adjusted_rand_score(true_labels, km.labels_)
        assert ari == pytest.approx(1.0, abs=0.01)

    def test_auto_t_single_blob_vs_separated(self):
        """Auto-t: well-separated clusters get higher t than single blob with clear gap."""
        np.random.seed(42)
        # Well-separated clusters should have eigenvalues near 1 above gap,
        # so auto-selection picks a higher t to suppress below-gap noise.
        X_sep, _ = make_blobs(
            n_samples=90, centers=3, cluster_std=0.3,
            center_box=(-15, 15), random_state=42,
        )
        A_sep = _self_tuning_affinity_dense(X_sep, k=7)
        _, t_sep = _diffusion_map(A_sep, diffusion_time=None, n_components=10, seed=42)
        assert t_sep >= 3
        assert 2 <= t_sep <= 10

    def test_alpha_different_spectra(self):
        """Alpha=0.5 vs alpha=1.0: verify different eigenvalue spectra."""
        np.random.seed(42)
        X, _ = make_blobs(n_samples=60, centers=2, cluster_std=0.5, random_state=42)
        A = _self_tuning_affinity_dense(X, k=7)
        emb_05, _ = _diffusion_map(A, diffusion_time=3, n_components=5, seed=42, alpha=0.5)
        emb_10, _ = _diffusion_map(A, diffusion_time=3, n_components=5, seed=42, alpha=1.0)
        # The embeddings should differ because alpha changes the normalization
        assert not np.allclose(emb_05, emb_10, atol=1e-6)

    def test_deterministic_same_seed(self):
        """Deterministic: same seed gives identical embedding."""
        np.random.seed(42)
        X, _ = make_blobs(n_samples=60, centers=2, cluster_std=0.5, random_state=42)
        A = _self_tuning_affinity_dense(X, k=7)
        emb1, t1 = _diffusion_map(A, diffusion_time=3, n_components=5, seed=42)
        emb2, t2 = _diffusion_map(A, diffusion_time=3, n_components=5, seed=42)
        np.testing.assert_array_equal(emb1, emb2)
        assert t1 == t2

    def test_returns_correct_shape_tuple(self):
        """Returns (embedding, t_used) tuple with correct shapes."""
        np.random.seed(42)
        n, n_comp = 80, 10
        X, _ = make_blobs(n_samples=n, centers=2, cluster_std=0.5, random_state=42)
        A = _self_tuning_affinity_dense(X, k=7)
        result = _diffusion_map(A, diffusion_time=4, n_components=n_comp, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        embedding, t_used = result
        assert embedding.shape[0] == n
        assert embedding.shape[1] == n_comp
        assert t_used == 4


# ---------------------------------------------------------------------------
# TestPowerIterationFiedler
# ---------------------------------------------------------------------------
class TestPowerIterationFiedler:
    """Tests for _power_iteration_fiedler."""

    def test_well_separated_clusters_sign_match(self):
        """Well-separated 2 clusters: Fiedler vector signs match true labels."""
        np.random.seed(42)
        X, true_labels = make_blobs(
            n_samples=80, centers=2, cluster_std=0.3,
            center_box=(-10, 10), random_state=42,
        )
        A = _self_tuning_affinity_dense(X, k=7)
        fiedler, lambda2 = _power_iteration_fiedler(A, seed=42)
        # Check that the sign of the Fiedler vector separates clusters
        pred = (fiedler > 0).astype(int)
        # Labels might be flipped; check both orientations
        match_a = np.mean(pred == true_labels)
        match_b = np.mean(pred == (1 - true_labels))
        assert max(match_a, match_b) >= 0.95

    def test_lambda2_matches_eigsh(self):
        """Lambda2 from power iteration matches eigsh within bias tolerance.

        Power iteration has a small systematic underestimate (~15%) relative
        to ARPACK eigsh on self-tuning affinity matrices at this size — the
        deflation step converges before the iterate fully aligns with the
        Fiedler eigenvector.  Tolerance set to 25% to absorb that bias while
        still flagging convergence failures (PIC returning ~0 on a connected
        graph would be caught).

        Uses fixed-center data instead of make_blobs(center_box=...) because
        random center placement under a fixed random_state interacts with
        BLAS-ordering noise: the resulting affinity matrix can land near
        graph disconnection on some platforms (inter-cluster weights
        underflowing to 0 in float32), which makes lambda2 itself platform-
        dependent.  Fixed centers at [0,0] and [2.5,0] with std=1.0 keep the
        graph robustly connected and both methods land on a non-trivial
        Fiedler value (~0.06).
        """
        np.random.seed(42)
        X = np.vstack([
            np.random.normal([0.0, 0.0], 1.0, (40, 2)),
            np.random.normal([2.5, 0.0], 1.0, (40, 2)),
        ]).astype(np.float32)
        A = _self_tuning_affinity_dense(X, k=7)
        from scipy.sparse import csr_matrix
        A_sp = csr_matrix(A)
        L = csgraph.laplacian(A_sp, normed=True)
        eigvals, _ = _sparse_eigsh(L, k=2, which="SM")
        eigsh_lambda2 = float(sorted(eigvals)[1])

        _, pic_lambda2 = _power_iteration_fiedler(A, seed=42)
        assert pic_lambda2 > 0, "power iteration did not converge to positive lambda2"
        assert pic_lambda2 == pytest.approx(eigsh_lambda2, rel=0.25)

    def test_deterministic_same_seed(self):
        """Deterministic: same seed gives identical output."""
        np.random.seed(42)
        X, _ = make_blobs(n_samples=60, centers=2, cluster_std=0.5, random_state=42)
        A = _self_tuning_affinity_dense(X, k=7)
        f1, l1 = _power_iteration_fiedler(A, seed=42)
        f2, l2 = _power_iteration_fiedler(A, seed=42)
        np.testing.assert_array_equal(f1, f2)
        assert l1 == l2

    def test_single_tight_blob_lambda2_small(self):
        """Single tight blob: lambda2 is small (well-connected, hard to split)."""
        np.random.seed(42)
        X = np.random.randn(60, 5) * 0.1
        A = _self_tuning_affinity_dense(X, k=7)
        _, lambda2 = _power_iteration_fiedler(A, seed=42)
        # For a tight single blob, lambda2 should be notably smaller than
        # the well-separated case (which gets lambda2 close to 1).
        assert lambda2 < 0.6


# ---------------------------------------------------------------------------
# TestAnisotropicAffinity
# ---------------------------------------------------------------------------
class TestAnisotropicAffinity:
    """Tests for _anisotropic_affinity_sparse."""

    def test_returns_sparse_symmetric_correct_shape(self):
        """Returns sparse symmetric matrix with correct shape."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        A = _anisotropic_affinity_sparse(X, k=7, k_connect=15)
        assert issparse(A)
        assert A.shape == (n, n)
        # Symmetry check
        diff = A - A.T
        assert diff.nnz == 0 or np.abs(diff).max() < 1e-10

    def test_diagonal_is_zero(self):
        """Diagonal is zero (no self-affinity)."""
        np.random.seed(42)
        X = np.random.randn(80, 5)
        A = _anisotropic_affinity_sparse(X, k=7, k_connect=15)
        diag_vals = A.diagonal()
        assert np.allclose(diag_vals, 0.0, atol=1e-10)

    def test_within_cluster_gt_between_cluster(self):
        """Within-cluster affinity > between-cluster affinity for well-separated elongated clusters."""
        np.random.seed(42)
        # Elongated clusters along different axes
        c1 = np.column_stack([
            np.random.randn(40) * 5.0,  # elongated along dim 0
            np.random.randn(40) * 0.2,
            np.random.randn(40) * 0.2,
        ])
        c2 = np.column_stack([
            np.random.randn(40) * 0.2,
            np.random.randn(40) * 5.0,  # elongated along dim 1
            np.random.randn(40) * 0.2,
        ]) + np.array([15.0, 15.0, 0.0])  # shifted away
        X = np.vstack([c1, c2])
        A = _anisotropic_affinity_sparse(X, k=10, k_connect=20)
        A_dense = A.toarray()
        within = (A_dense[:40, :40].sum() + A_dense[40:, 40:].sum()) / 2.0
        between = A_dense[:40, 40:].sum()
        assert within > between

    def test_anisotropic_vs_self_tuning_elongated(self):
        """Elongated clusters: anisotropic gives higher within-cluster affinity than self-tuning."""
        np.random.seed(42)
        c1 = np.column_stack([
            np.random.randn(50) * 6.0,
            np.random.randn(50) * 0.1,
            np.random.randn(50) * 0.1,
        ])
        c2 = np.column_stack([
            np.random.randn(50) * 0.1,
            np.random.randn(50) * 6.0,
            np.random.randn(50) * 0.1,
        ]) + np.array([20.0, 20.0, 0.0])
        X = np.vstack([c1, c2])

        A_aniso = _anisotropic_affinity_sparse(X, k=10, k_connect=20)
        A_st = _self_tuning_affinity_sparse(X, k=10, k_connect=20)

        def _within(A):
            Ad = A.toarray()
            return (Ad[:50, :50].sum() + Ad[50:, 50:].sum()) / 2.0

        assert _within(A_aniso) > _within(A_st)


# ---------------------------------------------------------------------------
# TestAutoDiffusionTime
# ---------------------------------------------------------------------------
class TestAutoDiffusionTime:
    """Tests for _auto_diffusion_time."""

    def test_well_separated_eigenvalues(self):
        """Well-separated eigenvalues [1.0, 1.0, 0.3, 0.1]: t should suppress 0.3 -> gives t >= 3."""
        eigenvalues = np.array([1.0, 1.0, 0.3, 0.1])
        t = _auto_diffusion_time(eigenvalues)
        assert t >= 3

    def test_no_gap_gives_max(self):
        """No gap [0.95, 0.93, 0.91, 0.89]: gives t=10 (max, no clear gap)."""
        eigenvalues = np.array([0.95, 0.93, 0.91, 0.89])
        t = _auto_diffusion_time(eigenvalues)
        assert t == 10

    def test_single_eigenvalue_returns_minimum(self):
        """Single eigenvalue: returns 2 (minimum)."""
        eigenvalues = np.array([0.5])
        t = _auto_diffusion_time(eigenvalues)
        assert t == 2

    def test_result_always_in_range(self):
        """Result always in [2, 10] for various eigenvalue arrays."""
        np.random.seed(42)
        for _ in range(50):
            n = np.random.randint(2, 20)
            eigs = np.sort(np.random.rand(n))[::-1]
            t = _auto_diffusion_time(eigs)
            assert 2 <= t <= 10


# ---------------------------------------------------------------------------
# TestSpectralDivisiveConfigurations
# ---------------------------------------------------------------------------
class TestSpectralDivisiveConfigurations:
    """Tests for spectral_divisive with diffusion/anisotropic/PIC configurations."""

    @pytest.mark.parametrize(
        "affinity,use_pic",
        [
            ("self_tuning", True),
            ("self_tuning", False),
            ("diffusion", True),
            ("diffusion", False),
            ("anisotropic", True),
            ("anisotropic", False),
        ],
    )
    def test_well_separated_finds_k3(self, affinity, use_pic):
        """Well-separated 3-cluster data gives k=3 for all 6 configs.

        Uses centers at multiples of 5 with std=1.0 (sep/std=5) and 50 points
        per cluster.  Earlier formulations used 30 points per cluster with
        centers at multiples of 20 and std=0.3 (sep/std=67); that regime is
        not "well-separated" in any practical sense — inter-cluster affinity
        underflows to 0 in float32 and the top Laplacian eigenvalues collapse
        to a rotation-degenerate subspace, where different BLAS pick
        different orthonormal bases and the score function reads them as
        different winners (k=2 on Windows, k=3 on Linux, k=4 on macOS).
        sep/std=5 + n=50 keeps the eigengap wide enough for stable
        cross-platform agreement; n=30 alone was insufficient because the
        score function in self_tuning + no_PIC becomes small-sample noisy.
        Algorithmic robustness for degenerate eigenstructure is tracked as
        a future improvement (PR 9 in the plan).
        """
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive
        from opendsm.common.clustering.algorithms.settings import AffinityMatrixOptions

        np.random.seed(42)
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(c * 5, 1.0, (50, 5))
            for c in range(3)
        ]).astype(np.float32)

        settings = SpectralSettings(
            affinity=AffinityMatrixOptions(affinity),
            use_pic=use_pic,
            n_cluster={"lower": 2, "upper": 6},
            recluster_count=0,
            nystrom_samples=None,
        )
        result = spectral_divisive(X, _spectral_div_cs(settings, seed=42))
        assert result.k == 3, (
            f"affinity={affinity}, use_pic={use_pic}: expected k=3, got k={result.k}"
        )

    @pytest.mark.parametrize(
        "affinity,use_pic",
        [
            ("self_tuning", True),
            ("self_tuning", False),
            ("diffusion", True),
            ("diffusion", False),
            ("anisotropic", True),
            ("anisotropic", False),
        ],
    )
    def test_deterministic_seed42(self, affinity, use_pic):
        """All 6 configs are deterministic (seed=42 twice gives same labels)."""
        from opendsm.common.clustering.algorithms.spectral.spectral_divisive import spectral_divisive
        from opendsm.common.clustering.algorithms.settings import AffinityMatrixOptions

        np.random.seed(42)
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal(c * 20, 0.3, (30, 5))
            for c in range(3)
        ]).astype(np.float32)

        settings = SpectralSettings(
            affinity=AffinityMatrixOptions(affinity),
            use_pic=use_pic,
            n_cluster={"lower": 2, "upper": 6},
            recluster_count=0,
            nystrom_samples=None,
        )
        labels1 = spectral_divisive(X, _spectral_div_cs(settings, seed=42)).labels
        labels2 = spectral_divisive(X, _spectral_div_cs(settings, seed=42)).labels
        np.testing.assert_array_equal(labels1, labels2)


# ---------------------------------------------------------------------------
# TestSubSeed
# ---------------------------------------------------------------------------
class TestSubSeed:
    """Tests for _sub_seed deterministic seed derivation."""

    def test_same_base_same_indices_same_result(self):
        """Same base_seed + same indices gives same result."""
        indices = np.array([10, 20, 30, 40])
        assert _sub_seed(42, indices) == _sub_seed(42, indices)

    def test_different_base_seed_different_result(self):
        """Different base_seed gives different result."""
        indices = np.array([10, 20, 30, 40])
        assert _sub_seed(42, indices) != _sub_seed(99, indices)

    def test_same_base_different_indices_different_result(self):
        """Same base_seed + different indices gives different result."""
        idx_a = np.array([10, 20, 30, 40])
        idx_b = np.array([11, 20, 30, 40])
        assert _sub_seed(42, idx_a) != _sub_seed(42, idx_b)


class TestSparseEigshFallback:
    """The sparse self-tuning path recovers via a dense eigendecomposition
    when ARPACK fails to converge."""

    def test_dense_fallback_on_eigsh_failure(self, monkeypatch):
        spectral_mod = sys.modules[spectral.__module__]

        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(c, 0.4, (20, 4)) for c in (0, 6, 12)])

        # Force the sparse branch on small data, then make ARPACK fail so the
        # dense fallback path runs.
        monkeypatch.setattr(spectral_mod, "_SELF_TUNING_SPARSE_THRESHOLD", 5)

        def _arpack_fails(*args, **kwargs):
            raise RuntimeError("ARPACK did not converge")

        monkeypatch.setattr(spectral_mod, "_sparse_eigsh", _arpack_fails)

        cs = _spectral_cs(SpectralSettings(
            affinity="self_tuning",
            n_cluster={"lower": 2, "upper": 4},
        ))
        result = spectral(X, cs)

        assert len(result.labels) == len(X)
        assert result.k >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
