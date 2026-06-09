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

"""Unit tests for the BIRCH wrapper.

BIRCH has no random component, so it is deterministic without a seed.  The
wrapper sweeps n_clusters across the requested range and scores each.
"""

import numpy as np
import pytest

from sklearn.metrics import adjusted_rand_score

from opendsm.common.clustering.settings import ClusteringSettings
from opendsm.common.clustering.algorithms.birch import birch



def _bcs(lower, upper, **birch_settings):
    """ClusteringSettings selecting BIRCH over a cluster range."""
    cs = ClusteringSettings(
        algorithm_selection="birch",
        seed=42,
        birch={"n_cluster": {"lower": lower, "upper": upper}, **birch_settings},
    )

    return cs


def _three_blobs(n_per=30, d=5, sep=50.0, seed=0):
    """Three well-separated blobs with ground-truth labels."""
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(np.full(d, i * sep), 1.0, size=(n_per, d)) for i in range(3)]
    data = np.vstack(blocks)
    truth = np.repeat(np.arange(3), n_per)

    return data, truth


class TestBirchHappyPath:
    """Valid structure is recovered across the requested range."""

    def test_recovers_well_separated_blobs(self):
        """A [2,5] sweep selects k=3 and recovers the ground truth (ARI ~ 1)."""
        data, truth = _three_blobs()
        result = birch(data, _bcs(2, 5))
        assert result.k == 3
        assert adjusted_rand_score(truth, result.labels) > 0.99

    def test_forced_k_returns_exactly_k(self):
        """Forcing k=3 on three blobs yields exactly three clusters."""
        data, _ = _three_blobs()
        result = birch(data, _bcs(3, 3))
        assert len(np.unique(result.labels)) == 3

    def test_deterministic(self):
        """BIRCH has no seed; repeated runs are bit-identical."""
        data, _ = _three_blobs()
        a = birch(data, _bcs(3, 3)).labels
        b = birch(data, _bcs(3, 3)).labels
        assert np.array_equal(a, b)


class TestBirchDegenerate:
    """Degenerate input is handled honestly, not fabricated."""

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_identical_data_forced_k_raises(self):
        """All-identical points cannot form k=3, so no valid labeling exists.

        BIRCH emits a ConvergenceWarning ("subclusters found < n_clusters")
        rather than fabricating clusters; the collapsed labeling is then
        rejected at the result level.
        """
        data = np.ones((50, 4))
        result = birch(data, _bcs(3, 3))
        with pytest.raises(ValueError, match="No clustering met"):
            _ = result.labels


class TestBirchFailures:
    """Invalid input raises a specific error."""

    def test_nan_raises(self):
        """Non-finite data is rejected by the underlying distance computation."""
        data, _ = _three_blobs()
        data[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            birch(data, _bcs(3, 3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
