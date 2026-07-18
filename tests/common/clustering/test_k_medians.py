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

"""Unit tests for the KMedians algorithm and its building blocks.

Covers the public surface directly (init strategies, single-k fit,
refinement, schedule helpers) and the ClusterAlgorithm entry point.
"""

import numpy as np
import pytest

from sklearn.metrics import adjusted_rand_score

from opendsm.common.clustering.algorithms.k_medians import (
    kmeanspp_init,
    farthest_first_init,
    bisecting_init,
    random_init,
    kmedians_fit,
    kmedians_refine,
    _relabel_and_inertia,
    _adaptive_n_init,
    _scale_init_schedule,
    _sub_seed,
    kmedians,
    DEFAULT_INIT_SCHEDULE,
)

from .conftest import make_clustering_settings



def _three_blobs(n_per=40, d=5, sep=50.0, seed=0):
    """Three well-separated isotropic blobs with known ground-truth labels."""
    rng = np.random.default_rng(seed)
    centers = np.array([np.full(d, i * sep) for i in range(3)])
    blocks = [rng.normal(c, 1.0, size=(n_per, d)) for c in centers]
    data = np.vstack(blocks)
    truth = np.repeat(np.arange(3), n_per)

    return data, truth, centers


def _nearest_center_index(point, centers):
    """Index of the ground-truth center closest to *point*."""
    dists = np.sum((centers - point) ** 2, axis=1)

    return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# Initialization strategies
# ---------------------------------------------------------------------------

class TestInitStrategies:
    """Each init returns k centroids of the right shape, deterministically."""

    INIT_FNS = [kmeanspp_init, farthest_first_init, bisecting_init, random_init]

    @pytest.mark.parametrize("init_fn", INIT_FNS)
    def test_shape(self, init_fn):
        """Init returns exactly (k, d) centroids."""
        data, _, _ = _three_blobs()
        centers = init_fn(data, 3, np.random.default_rng(1))
        assert centers.shape == (3, data.shape[1])

    @pytest.mark.parametrize("init_fn", INIT_FNS)
    def test_deterministic_given_seed(self, init_fn):
        """Same seeded generator -> identical centroids."""
        data, _, _ = _three_blobs()
        a = init_fn(data, 3, np.random.default_rng(7))
        b = init_fn(data, 3, np.random.default_rng(7))
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("init_fn", [kmeanspp_init, farthest_first_init, random_init])
    def test_centers_are_data_points(self, init_fn):
        """Point-selecting inits return actual rows of the data."""
        data, _, _ = _three_blobs()
        centers = init_fn(data, 3, np.random.default_rng(2))
        for c in centers:
            assert np.any(np.all(np.isclose(data, c), axis=1))

    def test_farthest_first_covers_every_blob(self):
        """Farthest-first places one seed in each well-separated blob."""
        data, _, centers = _three_blobs(sep=100.0)
        seeds = farthest_first_init(data, 3, np.random.default_rng(3))
        covered = {_nearest_center_index(s, centers) for s in seeds}
        assert covered == {0, 1, 2}

    def test_random_init_picks_distinct_points(self):
        """Random init returns k distinct data points (replace=False)."""
        data, _, _ = _three_blobs()
        centers = random_init(data, 5, np.random.default_rng(4))
        unique_rows = np.unique(centers, axis=0)
        assert unique_rows.shape[0] == 5

    def test_kmeanspp_handles_identical_data(self):
        """KMeans++ on all-identical points falls back to random picks, no error."""
        data = np.ones((20, 4))
        centers = kmeanspp_init(data, 3, np.random.default_rng(5))
        assert centers.shape == (3, 4)
        assert np.allclose(centers, 1.0)


# ---------------------------------------------------------------------------
# kmedians_fit
# ---------------------------------------------------------------------------

class TestKMediansFit:
    """Single-k fit: ground-truth recovery, contiguity, inertia behaviour."""

    def test_recovers_well_separated_blobs(self):
        """k=3 fit on 3 separated blobs recovers the ground truth (ARI ~ 1)."""
        data, truth, _ = _three_blobs()
        labels, _ = kmedians_fit(data, 3, rng=np.random.default_rng(0))
        assert adjusted_rand_score(truth, labels) > 0.99

    def test_labels_are_contiguous(self):
        """Returned labels are a contiguous 0..k-1 range with no gaps."""
        data, _, _ = _three_blobs()
        labels, _ = kmedians_fit(data, 3, rng=np.random.default_rng(0))
        assert set(np.unique(labels)) == {0, 1, 2}

    def test_inertia_is_finite_nonnegative(self):
        """Inertia is a finite, non-negative float."""
        data, _, _ = _three_blobs()
        _, inertia = kmedians_fit(data, 3, rng=np.random.default_rng(0))
        assert np.isfinite(inertia)
        assert inertia >= 0.0

    def test_more_clusters_lower_inertia(self):
        """Inertia decreases monotonically as k grows (k=3 < k=1)."""
        data, _, _ = _three_blobs()
        _, inertia_k1 = kmedians_fit(data, 1, rng=np.random.default_rng(0))
        _, inertia_k3 = kmedians_fit(data, 3, rng=np.random.default_rng(0))
        assert inertia_k3 < inertia_k1

    def test_min_cluster_size_absorbs_small_clusters(self):
        """With min_cluster_size set, no surviving cluster is below threshold."""
        rng = np.random.default_rng(0)
        big = np.vstack([rng.normal(0, 1, (40, 4)), rng.normal(50, 1, (40, 4))])
        outliers = rng.normal(200, 0.1, (2, 4))
        data = np.vstack([big, outliers])
        labels, _ = kmedians_fit(data, 3, rng=np.random.default_rng(0), min_cluster_size=10)
        counts = np.bincount(labels[labels >= 0])
        assert counts[counts > 0].min() >= 10


# ---------------------------------------------------------------------------
# kmedians_refine
# ---------------------------------------------------------------------------

class TestKMediansRefine:
    """Refinement: optimal is a no-op, bad assignments are corrected."""

    def test_optimal_labels_unchanged(self):
        """Refining an already-correct partition preserves it (ARI ~ 1)."""
        data, truth, _ = _three_blobs()
        refined = kmedians_refine(data, truth.copy())
        assert adjusted_rand_score(truth, refined) > 0.99

    def test_corrects_bad_assignment_and_lowers_inertia(self):
        """A deliberately scrambled labeling is repaired toward ground truth."""
        data, truth, _ = _three_blobs()
        scrambled = np.arange(len(data)) % 3  # ignores spatial structure
        _, bad_inertia = _relabel_and_inertia(data, scrambled)

        refined = kmedians_refine(data, scrambled.copy())

        _, refined_inertia = _relabel_and_inertia(data, refined)
        assert refined_inertia < bad_inertia
        assert adjusted_rand_score(truth, refined) > 0.99

    def test_single_label_returns_unchanged(self):
        """<=1 distinct active label triggers the early return unchanged."""
        data, _, _ = _three_blobs()
        all_one = np.zeros(len(data), dtype=np.intp)
        refined = kmedians_refine(data, all_one.copy())
        assert np.array_equal(refined, all_one)

    def test_outlier_labels_are_frozen(self):
        """-1 entries are preserved (frozen) through refinement."""
        data, truth, _ = _three_blobs()
        labels = truth.copy()
        labels[:5] = -1
        refined = kmedians_refine(data, labels)
        assert np.all(refined[:5] == -1)
        assert np.all(refined[5:] >= 0)


# ---------------------------------------------------------------------------
# Schedule + seed helpers
# ---------------------------------------------------------------------------

class TestScheduleHelpers:
    """Deterministic restart-budget arithmetic."""

    @pytest.mark.parametrize("k,expected", [
        (2, 5), (4, 5),         # k<=4: full budget
        (5, 3), (12, 3),        # 5<=k<=12: ceil(5*0.6)=3
        (13, 2), (50, 2),       # k>=13: ceil(5*0.4)=2
    ])
    def test_adaptive_n_init(self, k, expected):
        """Restart budget steps down at the k thresholds."""
        assert _adaptive_n_init(5, k) == expected

    def test_adaptive_n_init_floor(self):
        """Budget never drops below 2 even for a tiny base at high k."""
        assert _adaptive_n_init(1, 100) == 2

    def test_scale_schedule_no_op_when_target_high(self):
        """Target >= current total returns the schedule unchanged."""
        result = _scale_init_schedule(DEFAULT_INIT_SCHEDULE, 100)
        assert result == DEFAULT_INIT_SCHEDULE

    def test_scale_schedule_shrinks_keeping_one_each(self):
        """Scaling down preserves length, keeps >=1 per strategy, never grows.

        The per-strategy floor of 1 means the total can exceed an aggressive
        target (4 -> 5 on the default schedule); it is bounded by the original.
        """
        result = _scale_init_schedule(DEFAULT_INIT_SCHEDULE, 4)
        counts = [c for _, c in result]
        original_total = sum(c for _, c in DEFAULT_INIT_SCHEDULE)
        assert len(result) == len(DEFAULT_INIT_SCHEDULE)
        assert all(c >= 1 for c in counts)
        assert sum(counts) <= original_total

    def test_scale_schedule_exact_on_clean_division(self):
        """Proportional scaling hits the target when counts divide evenly."""
        schedule = [("a", 4), ("b", 4)]
        result = _scale_init_schedule(schedule, 4)
        assert sum(c for _, c in result) == 4

    def test_sub_seed_deterministic(self):
        """Same base + parts -> same child seed."""
        assert _sub_seed(42, 3) == _sub_seed(42, 3)

    def test_sub_seed_varies_with_parts(self):
        """Different parts -> different child seeds."""
        assert _sub_seed(42, 3) != _sub_seed(42, 4)


# ---------------------------------------------------------------------------
# kmedians entry point
# ---------------------------------------------------------------------------

class TestKMediansEntryPoint:
    """ClusterAlgorithm protocol behaviour: recovery, determinism, failures."""

    def _settings(self, seed=42, lower=3, upper=3, **kw):
        cs = make_clustering_settings(
            "kmedians",
            seed=seed,
            kmedians={"n_cluster": {"lower": lower, "upper": upper}, **kw},
        )

        return cs

    def test_recovers_blobs(self):
        """Forced k=3 recovers the ground-truth partition (ARI ~ 1)."""
        data, truth, _ = _three_blobs()
        result = kmedians(data, self._settings())
        assert adjusted_rand_score(truth, result.labels) > 0.99

    def test_determinism(self):
        """Same seed -> identical labels."""
        data, _, _ = _three_blobs()
        a = kmedians(data, self._settings(seed=1))
        b = kmedians(data, self._settings(seed=1))
        assert np.array_equal(a.labels, b.labels)

    def test_recluster_count_zero_path(self):
        """recluster_count=0 takes the single-run path and still recovers blobs."""
        data, truth, _ = _three_blobs()
        result = kmedians(data, self._settings(recluster_count=0))
        assert adjusted_rand_score(truth, result.labels) > 0.99

    def test_recluster_count_positive_deterministic(self):
        """The multi-restart path is deterministic under a fixed seed."""
        data, _, _ = _three_blobs()
        a = kmedians(data, self._settings(seed=5, recluster_count=2))
        b = kmedians(data, self._settings(seed=5, recluster_count=2))
        assert np.array_equal(a.labels, b.labels)

    def test_too_few_samples_raises(self):
        """n < 2 is rejected with a clear ValueError."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            kmedians(np.ones((1, 4)), self._settings())

    def test_non_finite_raises(self):
        """NaN/inf in the data is rejected with a clear ValueError."""
        data, _, _ = _three_blobs()
        data[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            kmedians(data, self._settings())

    def test_identical_data_k1_allowed_collapses_honestly(self):
        """Identical points with k=1 allowed yield a single cluster, not fabricated k."""
        data = np.ones((30, 4))
        result = kmedians(data, self._settings(lower=1, upper=6))
        assert result.k == 1
