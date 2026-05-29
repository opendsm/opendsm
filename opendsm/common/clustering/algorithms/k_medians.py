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

"""KMedians clustering with mixed initialization strategies.

Provides:
- ``kmedians``: ClusterAlgorithm entry point — runs KMedians at each k
  with mixed initialization and multi-restart.
- ``kmedians_fit``: fit KMedians at a single k with mixed init strategies.
- ``kmedians_refine``: refine existing labels using KMedians iterations.
- ``kmeanspp_init``: KMeans++ initialization for arbitrary k.
- ``farthest_first_init``: maxmin initialization for maximum coverage.
- ``bisecting_init``: greedy bisection to seed initial centroids.

Uses median centroids (robust to outliers) with L2 assignment distance
(consistent with the scoring pipeline). The default mixed initialization
combines 5 restarts with different biases:
  1x farthest-first (coverage), 1x bisecting (hierarchical),
  2x KMeans++ (density-aware), 1x random (unbiased).
"""

from __future__ import annotations

import numpy as np

from scipy.spatial.distance import cdist

from opendsm.common.clustering.metrics.label_ops import assign_small_clusters_nearest
from opendsm.common.clustering.metrics.labels import ClusteringResult


def _sub_seed(base_seed: int, *parts: int) -> int:
    """Derive a deterministic child seed from a base seed and integer parts."""
    h = base_seed
    for p in parts:
        h ^= (p * 2654435761) & 0xFFFFFFFF
        h = ((h << 13) ^ h) & 0xFFFFFFFF
    return int(h)


# ---------------------------------------------------------------------------
# KMeans++ initialization
# ---------------------------------------------------------------------------

def kmeanspp_init(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """KMeans++ initialization for arbitrary k.

    Sequentially selects k centers with probability proportional to
    squared distance from the nearest existing center.

    Returns (k, d) centroid array.
    """
    n, d = data.shape
    centers = np.empty((k, d), dtype=data.dtype)

    idx0 = rng.integers(n)
    centers[0] = data[idx0]

    # KMeans++ samples proportional to distance. For Euclidean variants,
    # use manual squared L2 (fast). For other metrics, use cdist.
    use_fast_l2 = metric in ("sqeuclidean", "euclidean")

    if use_fast_l2:
        min_d = np.sum((data - centers[0]) ** 2, axis=1)
    else:
        min_d = cdist(data, centers[:1], metric=metric).ravel()

    for c in range(1, k):
        total = min_d.sum()
        if total < 1e-20:
            centers[c] = data[rng.integers(n)]
        else:
            probs = min_d / total
            idx = rng.choice(n, p=probs)
            centers[c] = data[idx]

        if use_fast_l2:
            d_new = np.sum((data - centers[c]) ** 2, axis=1)
        else:
            d_new = cdist(data, centers[c:c+1], metric=metric).ravel()
        np.minimum(min_d, d_new, out=min_d)

    return centers


# ---------------------------------------------------------------------------
# Farthest-first (maxmin) initialization
# ---------------------------------------------------------------------------

def farthest_first_init(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """Farthest-first (maxmin) initialization for maximum spatial coverage.

    Each successive center is the point farthest from all existing centers.
    Guarantees every distinct region of the feature space gets a center.
    Deterministic given the first center (chosen randomly).

    Returns (k, d) centroid array.
    """
    n, d = data.shape
    centers = np.empty((k, d), dtype=data.dtype)

    idx0 = rng.integers(n)
    centers[0] = data[idx0]

    use_fast_l2 = metric in ("sqeuclidean", "euclidean")

    if use_fast_l2:
        min_d = np.sum((data - centers[0]) ** 2, axis=1)
    else:
        min_d = cdist(data, centers[:1], metric=metric).ravel()

    for c in range(1, k):
        idx = int(np.argmax(min_d))
        centers[c] = data[idx]

        if use_fast_l2:
            d_new = np.sum((data - centers[c]) ** 2, axis=1)
        else:
            d_new = cdist(data, centers[c:c+1], metric=metric).ravel()
        np.minimum(min_d, d_new, out=min_d)

    return centers


# ---------------------------------------------------------------------------
# Bisecting initialization
# ---------------------------------------------------------------------------

def bisecting_init(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """Greedy bisection to get initial centroids.

    Repeatedly splits the highest-inertia cluster using KMedians(k=2).
    Returns the median centroids of the resulting k clusters.
    Captures hierarchical structure (dominant splits first).

    Returns (k, d) centroid array.
    """
    n, d = data.shape
    # Start with all points in one cluster
    clusters = [np.arange(n)]

    while len(clusters) < k:
        # Find highest-inertia cluster
        best_idx = 0
        best_inertia = -1.0
        for i, cl_indices in enumerate(clusters):
            if len(cl_indices) < 4:
                continue
            cl_data = data[cl_indices]
            centroid = np.median(cl_data, axis=0)
            inertia = float(np.sum((cl_data - centroid) ** 2))
            if inertia > best_inertia:
                best_inertia = inertia
                best_idx = i

        to_split = clusters[best_idx]
        if len(to_split) < 4:
            break

        # Quick KMedians(k=2) on the sub-cluster
        sub_data = data[to_split]
        sub_centers = kmeanspp_init(sub_data, 2, rng)
        for _ in range(10):
            dists = cdist(sub_data, sub_centers, metric=metric)
            labels_01 = np.argmin(dists, axis=1)
            for c in range(2):
                mask = labels_01 == c
                if mask.sum() > 0:
                    sub_centers[c] = np.median(sub_data[mask], axis=0)

        left = to_split[labels_01 == 0]
        right = to_split[labels_01 == 1]

        if len(left) == 0 or len(right) == 0:
            break

        clusters[best_idx] = left
        clusters.append(right)

    # Compute median centroids from each cluster
    centroids = np.array([
        np.median(data[cl], axis=0) for cl in clusters
    ])

    # Pad with random points if bisecting couldn't reach k
    while len(centroids) < k:
        centroids = np.vstack([centroids, data[rng.integers(n)][None]])

    return centroids[:k]


# ---------------------------------------------------------------------------
# Random initialization
# ---------------------------------------------------------------------------

def random_init(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """Random initialization: k randomly chosen data points as centroids."""
    n = data.shape[0]
    indices = rng.choice(n, size=min(k, n), replace=False)
    return data[indices].copy()


# ---------------------------------------------------------------------------
# Mixed initialization schedule
# ---------------------------------------------------------------------------

#: Default init schedule: (strategy_name, count) pairs.
#: Total restarts = sum of counts = 5.
DEFAULT_INIT_SCHEDULE = [
    ("farthest_first", 1),
    ("bisecting", 1),
    ("kmeanspp", 2),
    ("random", 1),
]

_INIT_DISPATCH = {
    "kmeanspp": kmeanspp_init,
    "farthest_first": farthest_first_init,
    "bisecting": bisecting_init,
    "random": random_init,
}


# ---------------------------------------------------------------------------
# Core KMedians
# ---------------------------------------------------------------------------

def kmedians_fit(
    data: np.ndarray,
    k: int,
    max_iter: int = 30,
    n_init: int = 5,
    rng: np.random.Generator | None = None,
    min_cluster_size: int = 1,
    init_schedule: list[tuple[str, int]] | None = None,
    metric: str = "sqeuclidean",
    early_stop_inits: bool = False,
) -> tuple[np.ndarray, float]:
    """Run KMedians at a specific k with mixed initialization strategies.

    Parameters
    ----------
    data : (n, d) array
    k : number of clusters
    max_iter : max iterations per restart
    n_init : number of total restarts (used only when init_schedule is None
        and the caller wants homogeneous KMeans++ restarts, e.g. bisecting).
    rng : random generator
    min_cluster_size : clusters below this are absorbed
    init_schedule : list of (strategy_name, count) pairs.
        Defaults to DEFAULT_INIT_SCHEDULE (5 mixed restarts).
        Strategies: "kmeanspp", "farthest_first", "bisecting", "random".
    early_stop_inits : if True, stop after 3 consecutive inits converge
        to the same inertia (within 1%).

    Returns
    -------
    labels : (n,) int array, contiguous 0..k_actual-1
    inertia : sum of squared L2 distances to assigned median centroid
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if init_schedule is None:
        # Backward compat: if n_init was explicitly set, use all KMeans++
        if n_init != 5:
            init_schedule = [("kmeanspp", n_init)]
        else:
            init_schedule = DEFAULT_INIT_SCHEDULE

    n = data.shape[0]
    best_labels = np.zeros(n, dtype=np.intp)
    best_inertia = np.inf
    consecutive_same = 0
    _EARLY_STOP_THRESHOLD = 3  # stop after this many consecutive same-inertia inits

    for strategy_name, count in init_schedule:
        init_fn = _INIT_DISPATCH[strategy_name]
        for _ in range(count):
            centroids = init_fn(data, k, rng, metric=metric)
            labels = _kmedians_iterate(data, centroids, k, max_iter, metric=metric)

            # Absorb sub-threshold clusters
            if min_cluster_size >= 2:
                labels = assign_small_clusters_nearest(
                    labels, data, min_cluster_size, centroid="median",
                )

            labels, inertia = _relabel_and_inertia(data, labels, metric=metric)

            if inertia < best_inertia:
                # Check if improvement is marginal (within 1%)
                if best_inertia < np.inf and abs(inertia - best_inertia) / max(best_inertia, 1e-20) < 0.01:
                    consecutive_same += 1
                else:
                    consecutive_same = 0
                best_inertia = inertia
                best_labels = labels.copy()
            else:
                # No improvement — check if same as best
                if best_inertia > 0 and abs(inertia - best_inertia) / best_inertia < 0.01:
                    consecutive_same += 1
                else:
                    consecutive_same = 0

            if early_stop_inits and consecutive_same >= _EARLY_STOP_THRESHOLD:
                return best_labels, best_inertia

    return best_labels, best_inertia


def _kmedians_iterate(
    data: np.ndarray,
    centroids: np.ndarray,
    k: int,
    max_iter: int,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """Run KMedians iterations from given centroids until convergence."""
    n = data.shape[0]
    labels = -np.ones(n, dtype=np.intp)
    centroids = centroids.copy()

    for _ in range(max_iter):
        dists = cdist(data, centroids, metric=metric)
        new_labels = np.argmin(dists, axis=1)

        if labels[0] >= 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = np.median(data[mask], axis=0)

    return labels


def kmedians_refine(
    data: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 10,
    min_cluster_size: int = 1,
    metric: str = "sqeuclidean",
) -> np.ndarray:
    """Refine existing cluster assignments using KMedians iterations.

    Takes labels from any algorithm (bisecting, spectral, etc.) and
    runs median-centroid reassignment until convergence. Points can
    migrate between clusters.

    Parameters
    ----------
    data : (n, d) array
    labels : (n,) int array. -1 = outlier (frozen).
    max_iter : max refinement iterations
    min_cluster_size : clusters below this are absorbed

    Returns
    -------
    (n,) int array, relabeled to contiguous 0..k_actual-1.
    -1 entries are preserved.
    """
    result = labels.copy()
    active_mask = result >= 0

    unique_labels = np.unique(result[active_mask])
    if len(unique_labels) <= 1:
        return result

    centroids = np.array([
        np.median(data[result == c], axis=0) for c in unique_labels
    ])

    for _ in range(max_iter):
        dists = cdist(data[active_mask], centroids, metric=metric)
        new_idx = np.argmin(dists, axis=1)
        new_labels_active = unique_labels[new_idx]

        if np.array_equal(new_labels_active, result[active_mask]):
            break

        result[active_mask] = new_labels_active

        if min_cluster_size >= 2:
            result = assign_small_clusters_nearest(
                result, data, min_cluster_size, centroid="median",
            )
            active_mask = result >= 0  # recompute after reassignment
            unique_labels = np.unique(result[result >= 0])
        else:
            unique_labels = np.unique(result[active_mask])

        if len(unique_labels) <= 1:
            break

        centroids = np.array([
            np.median(data[result == c], axis=0) for c in unique_labels
        ])

    # Relabel to contiguous 0..k_actual-1
    final_unique = np.unique(result[active_mask])
    if len(final_unique) > 0:
        remap = {int(old): new for new, old in enumerate(final_unique)}
        active_indices = np.where(active_mask)[0]
        for i in active_indices:
            result[i] = remap[int(result[i])]

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _relabel_and_inertia(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str = "sqeuclidean",
) -> tuple[np.ndarray, float]:
    """Relabel to contiguous 0..k-1 and compute inertia."""
    unique = np.unique(labels[labels >= 0])
    remap = {int(old): new for new, old in enumerate(unique)}
    remap[-1] = -1
    relabeled = np.array([remap[int(l)] for l in labels], dtype=np.intp)

    centroids = np.array([
        np.median(data[relabeled == c], axis=0) for c in range(len(unique))
    ])
    active = relabeled >= 0
    dists = cdist(data[active], centroids, metric=metric)
    inertia = float(dists[np.arange(active.sum()), relabeled[active]].sum())

    return relabeled, inertia


# ---------------------------------------------------------------------------
# ClusterAlgorithm entry point
# ---------------------------------------------------------------------------

def _adaptive_n_init(base_n_init: int, k: int) -> int:
    """Reduce restarts for high k where the solution space is constrained.

    k=2-4: full n_init (diverse coverage matters)
    k=5-12: 60% of n_init
    k=13+: 40% of n_init
    """
    import math
    if k <= 4:
        return base_n_init
    elif k <= 12:
        return max(2, math.ceil(base_n_init * 0.6))
    else:
        return max(2, math.ceil(base_n_init * 0.4))


def _scale_init_schedule(
    schedule: list[tuple[str, int]],
    target_total: int,
) -> list[tuple[str, int]]:
    """Scale an init schedule to a target total number of restarts.

    Preserves the relative proportions. Ensures at least 1 per strategy
    (unless target_total < len(schedule)).
    """
    current_total = sum(c for _, c in schedule)
    if target_total >= current_total:
        return schedule

    # Scale down proportionally, ensure at least 1 per strategy
    result = []
    remaining = target_total
    for i, (name, count) in enumerate(schedule):
        if i == len(schedule) - 1:
            scaled = max(1, remaining)
        else:
            scaled = max(1, round(count * target_total / current_total))
            remaining -= scaled
        result.append((name, scaled))

    return result


def _kmedians_single(data, settings, seed):
    """Single KMedians run across all k values."""
    algo_settings = getattr(settings, settings.algorithm_selection.value)
    n_lower = algo_settings.n_cluster.lower
    n_upper = algo_settings.n_cluster.upper
    n_init = algo_settings.n_init
    max_iter = algo_settings.max_iter
    min_cs = settings.min_cluster_size
    metric = settings._argmin_metric
    early_stop = algo_settings.early_stop_inits
    adaptive = algo_settings.adaptive_n_init

    n = data.shape[0]
    if n < 2:
        raise ValueError(f"Need at least 2 samples for clustering, got {n}")
    if not np.all(np.isfinite(data)):
        raise ValueError("Data contains non-finite values")

    lbl = ClusteringResult(
        data=data,
        score_settings=algo_settings.scoring,
        seed=seed,
        n_cluster_lower=n_lower,
        min_cluster_size=settings.min_cluster_size,
        small_cluster_mode=settings.small_cluster_mode,
    )

    if n_lower <= 1:
        lbl.add(1, np.zeros(n, dtype=np.intp))

    for k in range(max(2, n_lower), n_upper + 1):
        if k * min_cs > n:
            break

        # Adaptive n_init: fewer restarts for high k
        k_n_init = _adaptive_n_init(n_init, k) if adaptive else n_init
        k_schedule = _scale_init_schedule(DEFAULT_INIT_SCHEDULE, k_n_init)

        rng = np.random.default_rng(_sub_seed(seed, k))
        labels, _ = kmedians_fit(
            data, k, max_iter, k_n_init, rng, min_cs,
            init_schedule=k_schedule, metric=metric,
            early_stop_inits=early_stop,
        )
        actual_k = len(np.unique(labels[labels >= 0]))
        if actual_k >= n_lower:
            lbl.add(actual_k, labels)

    return lbl


def kmedians(data, settings):
    """KMedians clustering with optional multi-restart.

    Runs KMedians at each k independently with KMeans++ initialization,
    producing balanced partitions for the scoring council.

    Conforms to the ClusterAlgorithm protocol.
    """
    algo_settings = getattr(settings, settings.algorithm_selection.value)
    seed = settings._seed
    recluster_count = algo_settings.recluster_count
    n_lower = algo_settings.n_cluster.lower

    if recluster_count == 0:
        return _kmedians_single(data, settings, seed)

    base_rng = np.random.default_rng(seed)
    restart_seeds = [int(base_rng.integers(0, 2**31)) for _ in range(recluster_count + 1)]

    all_by_k: dict[int, list] = {}
    for restart_seed in restart_seeds:
        lbl_i = _kmedians_single(data, settings, restart_seed)
        for k, lms in lbl_i._labels_store.items():
            all_by_k.setdefault(k, []).extend(lms)

    lbl = ClusteringResult(
        data=data,
        score_settings=algo_settings.scoring,
        seed=seed,
        n_cluster_lower=n_lower,
        min_cluster_size=settings.min_cluster_size,
        small_cluster_mode=settings.small_cluster_mode,
    )
    for k in sorted(all_by_k):
        for lm in all_by_k[k]:
            lbl._add_scored(k, lm)

    return lbl
