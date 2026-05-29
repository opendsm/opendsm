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

"""Shared helpers and fixtures for clustering tests.

Centralises algorithm-settings construction so that if a function
signature changes, only the helpers here need updating -- not every
individual test file.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from opendsm.common.clustering.settings import ClusteringSettings


# ---------------------------------------------------------------------------
# Settings builders
# ---------------------------------------------------------------------------

def make_clustering_settings(algorithm, seed=42, **overrides):
    """Create a ClusteringSettings for *any* supported algorithm.

    Parameters
    ----------
    algorithm : str
        One of the ClusterAlgorithms values, e.g. ``"spectral"``,
        ``"bisecting_kmeans"``, ``"spectral_divisive"``.
    seed : int
        Random seed.
    **overrides
        Extra keyword arguments forwarded to ``ClusteringSettings``.
        Algorithm-specific sub-settings can be passed as a dict under the
        algorithm name key (e.g. ``spectral={"n_cluster": {...}}``).
    """
    return ClusteringSettings(
        algorithm_selection=algorithm,
        seed=seed,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_simple_data(n_clusters=3, n_per=30, d=10, seed=42):
    """Create well-separated test data (NumPy array).

    Returns an (n_clusters * n_per, d) array where clusters are separated by
    10 units per dimension.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for i in range(n_clusters):
        center = np.full(d, i * 10.0)
        blocks.append(rng.normal(center, 0.3, size=(n_per, d)))
    return np.vstack(blocks)


def run_cluster(data, algorithm, seed=42, **overrides):
    """Run a clustering algorithm via the internal ``_cluster_features`` dispatcher.

    Returns the ``ClusteringResult`` object.
    """
    from opendsm.common.clustering.cluster import _cluster_features

    cs = make_clustering_settings(algorithm, seed=seed, **overrides)
    return _cluster_features(data, cs)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_2d_data():
    """Three well-separated 10-d clusters, 50 points each (150 total).

    Uses the legacy ``np.random.seed`` approach so that existing tests that
    relied on the fixture from test_spectral.py / test_bisect_k_means.py
    keep producing the exact same data.
    """
    np.random.seed(42)
    cluster1 = np.random.randn(50, 10) + np.array([0] * 10)
    cluster2 = np.random.randn(50, 10) + np.array([5] * 10)
    cluster3 = np.random.randn(50, 10) + np.array([10] * 10)
    return np.vstack([cluster1, cluster2, cluster3])


@pytest.fixture
def simple_3_cluster_data():
    """Three well-separated 5-d clusters, 30 points each (90 total).

    Uses ``np.random.default_rng`` for a cleaner seed.
    """
    return make_simple_data(n_clusters=3, n_per=30, d=5, seed=42)
