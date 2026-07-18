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

"""Bisecting KMedians clustering.

Top-down bisection where each split uses KMedians(k=2) with KMeans++
initialization. Produces partitions at every k from n_lower to n_upper
for the scoring council to evaluate.

Uses the shared KMedians implementation from k_medians.py.
"""

from __future__ import annotations

import heapq

import numpy as np

from opendsm.common.clustering.algorithms.k_medians import (
    _sub_seed,
    kmedians_fit,
    kmedians_refine,
)
from opendsm.common.clustering.metrics.labels import ClusteringResult


_MAX_SPLIT_ATTEMPTS = 5  # reseed budget before a node is accepted as unsplittable


def _cluster_inertia(data: np.ndarray, indices: np.ndarray) -> float:
    """Sum of squared L2 distances to the median centroid."""
    sub = data[indices]
    centroid = np.median(sub, axis=0)
    return float(np.sum(np.sum((sub - centroid) ** 2, axis=1)))


def _bisect_k_medians_single(data, settings, seed):
    """Single bisecting KMedians run.

    Maintains a priority queue of clusters ordered by inertia (or size),
    splitting the highest-priority cluster at each step.
    """
    algo_settings = getattr(settings, settings.algorithm_selection.value)
    n_lower = algo_settings.n_cluster.lower
    n_upper = algo_settings.n_cluster.upper
    n_init = algo_settings.n_init
    max_iter = algo_settings.max_iter
    refine = algo_settings.refinement_enabled
    refine_max_iter = algo_settings.refinement_max_iter
    use_inertia = algo_settings.bisecting_strategy == "biggest_inertia"

    min_cs = settings.min_cluster_size
    metric = settings._argmin_metric

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

    rng = np.random.default_rng(seed)
    flat_labels = np.zeros(n, dtype=np.intp)

    if n_lower <= 1:
        lbl.add(1, flat_labels.copy())

    tiebreak = 0
    heap: list = []
    all_indices = np.arange(n)

    init_priority = (
        -_cluster_inertia(data, all_indices) if use_inertia else -float(n)
    )
    heapq.heappush(heap, (init_priority, tiebreak, all_indices, 0))
    tiebreak += 1

    current_k = 1
    next_label = 1
    split_count = 0

    while current_k < n_upper:
        if not heap:
            break

        _, _, indices, attempts = heapq.heappop(heap)

        if len(indices) < 2 * min_cs:
            continue

        split_rng = np.random.default_rng(_sub_seed(seed, split_count))
        split_count += 1

        sub_data = data[indices]
        labels_01, _ = kmedians_fit(sub_data, 2, max_iter, n_init, split_rng, metric=metric)

        left_mask = labels_01 == 0
        right_mask = labels_01 == 1

        n_left = int(left_mask.sum())
        n_right = int(right_mask.sum())
        if n_left < min_cs or n_right < min_cs:
            # Reseed and retry; after the budget is spent, leave the node
            # unsplit (a terminal cluster) so the loop always terminates.
            if attempts + 1 < _MAX_SPLIT_ATTEMPTS:
                heapq.heappush(heap, (0.0, tiebreak, indices, attempts + 1))
                tiebreak += 1

            continue

        left_idx = indices[left_mask]
        right_idx = indices[right_mask]

        flat_labels[right_idx] = next_label
        next_label += 1
        current_k += 1

        for child_idx in (left_idx, right_idx):
            if len(child_idx) >= 2 * min_cs:
                priority = (
                    -_cluster_inertia(data, child_idx) if use_inertia
                    else -float(len(child_idx))
                )
                heapq.heappush(heap, (priority, tiebreak, child_idx, 0))
                tiebreak += 1

        if current_k >= n_lower:
            if refine and current_k >= 2:
                refined = kmedians_refine(
                    data, flat_labels,
                    max_iter=refine_max_iter,
                    min_cluster_size=min_cs,
                )
                actual_k = len(np.unique(refined[refined >= 0]))
                if actual_k >= n_lower:
                    lbl.add(actual_k, refined)
            else:
                lbl.add(current_k, flat_labels.copy())

    return lbl


def bisect_k_medians(data, settings):
    """Bisecting KMedians clustering with optional multi-restart.

    Conforms to the ClusterAlgorithm protocol.
    """
    algo_settings = getattr(settings, settings.algorithm_selection.value)
    seed = settings._seed
    recluster_count = algo_settings.recluster_count
    n_lower = algo_settings.n_cluster.lower

    if recluster_count == 0:
        return _bisect_k_medians_single(data, settings, seed)

    base_rng = np.random.default_rng(seed)
    restart_seeds = [int(base_rng.integers(0, 2**31)) for _ in range(recluster_count + 1)]

    all_by_k: dict[int, list] = {}
    for restart_seed in restart_seeds:
        lbl_i = _bisect_k_medians_single(data, settings, restart_seed)
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
