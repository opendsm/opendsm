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

"""Tests for the vendored sklearn BisectingKMeans fork.

High line-coverage on the inherited estimator is meaningless; this file
asserts the two reasons the fork exists: the ``labels_full`` dict of every
intermediate bisection, and the forced single-threaded determinism.
"""

import numpy as np
import pytest

from sklearn.cluster import BisectingKMeans as VanillaBisectingKMeans
from sklearn.metrics import adjusted_rand_score

from opendsm.common.clustering.algorithms.sklearn_bisect_k_means import BisectingKMeans



@pytest.fixture
def blobs():
    """Five well-separated 4-d blobs, 30 points each (150 total)."""
    rng = np.random.default_rng(0)
    data = np.vstack([rng.normal(np.full(4, c * 50.0), 1.0, size=(30, 4)) for c in range(5)])

    return data


class TestLabelsFullDelta:
    """``labels_full`` records every intermediate bisection step."""

    def test_keys_span_two_through_n_clusters(self, blobs):
        """Fitting n_clusters=5 records labels for k = 2, 3, 4, 5."""
        model = BisectingKMeans(n_clusters=5, random_state=42).fit(blobs)
        assert sorted(model.labels_full.keys()) == [2, 3, 4, 5]

    def test_each_entry_has_exactly_k_clusters(self, blobs):
        """labels_full[k] is a full-length labeling with exactly k clusters."""
        model = BisectingKMeans(n_clusters=5, random_state=42).fit(blobs)
        for k, labels in model.labels_full.items():
            assert len(labels) == len(blobs)
            assert len(np.unique(labels)) == k

    def test_final_entry_matches_labels_(self, blobs):
        """The top-k entry equals the estimator's final ``labels_``."""
        model = BisectingKMeans(n_clusters=5, random_state=42).fit(blobs)
        assert np.array_equal(model.labels_full[5], model.labels_)


class TestSingleThreadDeterminism:
    """The fork forces ``_n_threads = 1`` for reproducible results."""

    def test_n_threads_forced_to_one(self, blobs):
        """Fitting pins single-threaded execution regardless of the environment."""
        model = BisectingKMeans(n_clusters=5, random_state=42).fit(blobs)
        assert model._n_threads == 1

    def test_repeated_fits_identical(self, blobs):
        """Same seed -> bit-identical labels across fits."""
        a = BisectingKMeans(n_clusters=5, random_state=42).fit(blobs).labels_
        b = BisectingKMeans(n_clusters=5, random_state=42).fit(blobs).labels_
        assert np.array_equal(a, b)


class TestVanillaEquivalence:
    """The fork only adds bookkeeping; the partition matches stock sklearn."""

    def test_final_partition_matches_vanilla(self, blobs):
        """Final labels equal stock BisectingKMeans up to a relabeling (ARI = 1)."""
        fork = BisectingKMeans(n_clusters=5, random_state=42).fit(blobs).labels_
        vanilla = VanillaBisectingKMeans(n_clusters=5, random_state=42).fit(blobs).labels_
        assert adjusted_rand_score(fork, vanilla) == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
