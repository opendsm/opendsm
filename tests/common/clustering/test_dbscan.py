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

"""Unit tests for the DBSCAN wrapper.

DBSCAN is density-based: it pairs with ``small_cluster_mode='keep'`` so that
-1 noise carries genuine semantics.  When the run produces only noise the
ClusteringResult honestly refuses to report a clustering (raises).
"""

import numpy as np
import pytest

from sklearn.metrics import adjusted_rand_score

from opendsm.common.clustering.settings import ClusteringSettings
from opendsm.common.clustering.algorithms.dbscan import dbscan



def _dcs(**dbscan_settings):
    """ClusteringSettings selecting DBSCAN with the given sub-settings."""
    cs = ClusteringSettings(algorithm_selection="dbscan", seed=42, dbscan=dbscan_settings)

    return cs


def _three_blobs(n_per=30, d=5, sep=50.0, seed=0):
    """Three well-separated blobs with ground-truth labels."""
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(np.full(d, i * sep), 1.0, size=(n_per, d)) for i in range(3)]
    data = np.vstack(blocks)
    truth = np.repeat(np.arange(3), n_per)

    return data, truth


class TestDBSCANHappyPath:
    """Valid density structure is recovered."""

    def test_recovers_well_separated_blobs(self):
        """eps inside the blob, below the gap -> exact recovery (ARI ~ 1)."""
        data, truth = _three_blobs()
        result = dbscan(data, _dcs(epsilon=5.0, min_samples=3))
        assert result.k == 3
        assert adjusted_rand_score(truth, result.labels) > 0.99

    def test_huge_epsilon_merges_into_one_cluster(self):
        """eps spanning the whole space yields a single cluster."""
        data, _ = _three_blobs()
        result = dbscan(data, _dcs(epsilon=1e9, min_samples=1))
        assert result.k == 1
        assert len(np.unique(result.labels)) == 1

    def test_identical_points_form_one_cluster(self):
        """All-identical points within eps collapse to one cluster, not noise."""
        data = np.ones((50, 4))
        result = dbscan(data, _dcs(epsilon=0.5, min_samples=1))
        assert result.k == 1

    def test_deterministic(self):
        """DBSCAN is deterministic: same input -> identical labels."""
        data, _ = _three_blobs()
        a = dbscan(data, _dcs(epsilon=5.0, min_samples=3)).labels
        b = dbscan(data, _dcs(epsilon=5.0, min_samples=3)).labels
        assert np.array_equal(a, b)


class TestDBSCANNoiseOnly:
    """A run that finds only noise produces no valid clustering."""

    def test_tiny_epsilon_all_noise_raises(self):
        """eps below every pairwise distance with min_samples>1 -> all noise."""
        data, _ = _three_blobs()
        result = dbscan(data, _dcs(epsilon=1e-9, min_samples=5))
        with pytest.raises(ValueError, match="No clustering met"):
            _ = result.labels

    def test_min_samples_above_n_all_noise_raises(self):
        """min_samples exceeding n leaves every point as noise."""
        data, _ = _three_blobs()
        result = dbscan(data, _dcs(epsilon=5.0, min_samples=200))
        with pytest.raises(ValueError, match="No clustering met"):
            _ = result.labels


class TestDBSCANInputHandling:
    """Empty input fails; NaN is tolerated by sklearn (no finiteness guard)."""

    def test_empty_raises(self):
        """An empty dataset is rejected by the underlying estimator."""
        with pytest.raises(ValueError, match="0 sample"):
            dbscan(np.empty((0, 5)), _dcs(epsilon=5.0, min_samples=3))

    def test_nan_point_becomes_noise(self):
        """A NaN row is silently labelled noise; the wrapper does not validate.

        sklearn's DBSCAN finds no finite-distance neighbours for the NaN row,
        so it falls out as -1 while the remaining points cluster normally.
        """
        data, _ = _three_blobs()
        data[0, 0] = np.nan
        labels = dbscan(data, _dcs(epsilon=5.0, min_samples=3)).labels
        assert labels[0] == -1
        assert len(np.unique(labels[labels >= 0])) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
