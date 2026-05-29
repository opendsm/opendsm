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

from __future__ import annotations

import numpy as np

from opendsm.common.clustering.metrics.settings import ScoreSettings, SmallClusterMode


def _valid_cluster_counts(clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, counts) for non-outlier clusters (filtering out -1)."""
    uniq_ids, uniq_counts = np.unique(clusters, return_counts=True)
    mask = uniq_ids != -1
    return uniq_ids[mask], uniq_counts[mask]


def reindex_labels(clusters: np.ndarray, reorder: bool) -> np.ndarray:
    """Reindex cluster IDs to be contiguous from 0, optionally sorting by cluster size.

    After merging or reclustering there may be gaps in cluster IDs
    (e.g. [0, 1, 2, 5, 7]).  This function closes those gaps.

    When *reorder* is True, clusters are additionally sorted so that
    the largest cluster gets label 0.

    The outlier label ``-1`` is always preserved.
    """
    if reorder:
        # if outlier cluster exists, don't include it in the ordering
        uniq_id, counts = np.unique(clusters[clusters != -1], return_counts=True)
        count_order = np.argsort(counts)[::-1]

        uniq_id = uniq_id[count_order]

    else:
        # Exclude -1 from renumbering; the outlier mapping is fixed below.
        uniq_id = np.unique(clusters[clusters != -1])

    # if outlier cluster exists, don't change it
    conv = {-1: -1}
    conv.update({uniq_id[i]: i for i in range(len(uniq_id))})

    clusters = np.array([conv[idx] for idx in clusters])

    return clusters


def assign_small_clusters_outlier(clusters: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """Relabel clusters smaller than *min_cluster_size* as outliers (-1).

    Returns a **copy** — the input array is never mutated.
    """
    uniq_ids, uniq_counts = _valid_cluster_counts(clusters)

    outlier_ids = uniq_ids[uniq_counts < min_cluster_size]
    clusters = clusters.copy()
    clusters[np.isin(clusters, outlier_ids)] = -1

    return reindex_labels(clusters, reorder=True)


def assign_small_clusters_nearest(
    clusters: np.ndarray,
    data: np.ndarray,
    min_cluster_size: int,
    centroid: str = "mean",
) -> np.ndarray:
    """Assign points in sub-threshold clusters to the nearest large-cluster centroid.

    Pre-existing outlier (-1) points are left unchanged — only algorithm-produced
    small clusters (size < min_cluster_size) are reassigned.  If no large clusters
    exist, returns the input unchanged (caller will detect n_clusters < 2 and
    treat the labeling as invalid).

    Parameters
    ----------
    centroid : {"mean", "median"}
        How to compute the representative point for each large cluster.
        Use "median" for KMedians-consistent absorption.

    Returns a **copy** — the input array is never mutated.
    """
    uniq_ids, uniq_counts = _valid_cluster_counts(clusters)

    large_ids = uniq_ids[uniq_counts >= min_cluster_size]
    small_ids = set(uniq_ids[uniq_counts < min_cluster_size])

    if not small_ids:
        return clusters.copy()
    if len(large_ids) == 0:
        return clusters.copy()

    _agg = np.median if centroid == "median" else np.mean
    centroids = np.array([_agg(data[clusters == cid], axis=0) for cid in large_ids])

    result = clusters.copy()
    for sid in small_ids:
        pts = data[clusters == sid]
        dists = np.linalg.norm(pts[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
        nearest = large_ids[np.argmin(dists, axis=1)]
        result[clusters == sid] = nearest

    return reindex_labels(result, reorder=True)


def prepare_labels(
    labels: np.ndarray,
    data: np.ndarray,
    score_settings: ScoreSettings,
    n_cluster_lower: int | None,
    *,
    min_cluster_size: int = 1,
    small_cluster_mode: SmallClusterMode = SmallClusterMode.KEEP,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    """Apply the small-cluster strategy, then filter and validate the result.

    Branches on ``small_cluster_mode``:
    - OUTLIER: relabel small clusters as -1 (default)
    - ABSORB:  reassign small-cluster points to nearest large-cluster centroid
    - KEEP:    compact IDs only, all clusters kept regardless of size

    Returns ``(merged, data_clean, labels_clean, coverage)``.
    ``data_clean`` and ``labels_clean`` are ``None`` if the labeling is
    invalid after processing (too few clusters, too many, or below the
    algorithm's lower bound).  ``coverage`` = n_non_outlier / n_total.
    """
    mode = small_cluster_mode
    min_size = min_cluster_size

    if mode == SmallClusterMode.ABSORB:
        merged = assign_small_clusters_nearest(labels, data, min_size)
    elif mode == SmallClusterMode.KEEP:
        merged = reindex_labels(labels.copy(), reorder=True)
    else:  # OUTLIER — default
        merged = assign_small_clusters_outlier(labels, min_size)

    non_outlier = merged != -1
    data_clean = data[non_outlier]
    labels_clean = merged[non_outlier]

    n_total = len(merged)
    coverage = float(non_outlier.sum()) / n_total if n_total > 0 else 1.0

    n_clusters = len(np.unique(labels_clean)) if labels_clean.size else 0
    if n_clusters < 1:
        return merged, None, None, coverage
    if n_clusters > score_settings.max_non_outlier_cluster_count:
        return merged, None, None, coverage
    if n_cluster_lower is not None and n_clusters < n_cluster_lower:
        return merged, None, None, coverage

    return merged, data_clean, labels_clean, coverage


_MIN_CLUSTER_FOR_MAD = 5  # need enough points for reliable MAD estimation


def remove_outliers_mad(
    data: np.ndarray,
    labels: np.ndarray,
    mad_threshold: float,
    small_cluster_mode: SmallClusterMode = SmallClusterMode.KEEP,
    n_pcs: int = 3,
) -> np.ndarray:
    """Post-council MAD-based outlier removal.

    For each cluster with >= _MIN_CLUSTER_FOR_MAD members, projects onto the
    cluster's top *n_pcs* principal components and flags points whose
    deviation from the cluster median exceeds *mad_threshold* × MAD along
    any PC.

    The caller provides *mad_threshold* in MAD units (pre-converted from
    sigma via ClusteringSettings._outlier_mad_threshold).

    Flagged points are handled according to *small_cluster_mode*:
      - KEEP: flagged points form a new cluster (appended, renumbered)
      - OUTLIER: flagged points relabeled as -1
      - ABSORB: flagged points reassigned to nearest non-flagged cluster centroid

    Returns a new labels array.
    """
    threshold = mad_threshold
    labels = labels.copy()
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]  # skip existing outliers

    outlier_mask = np.zeros(len(labels), dtype=bool)

    for cl in unique_labels:
        cl_mask = labels == cl
        n_cl = cl_mask.sum()
        if n_cl < _MIN_CLUSTER_FOR_MAD:
            continue

        cluster_data = data[cl_mask]
        median = np.median(cluster_data, axis=0)
        centered = cluster_data - median

        # Top PCs of the cluster
        d = centered.shape[1]
        k = min(n_pcs, d, n_cl - 1)
        if k < 1:
            continue

        # Check for zero/constant data (all centered values near zero)
        if np.max(np.abs(centered)) < 1e-10:
            continue

        if k >= min(n_cl, d):
            # Need all components — use full SVD
            _, s, Vt = np.linalg.svd(centered, full_matrices=False)
            Vt = Vt[:k]
        else:
            # Truncated SVD with seeded v0 to avoid zero-start ARPACK error
            from scipy.sparse.linalg import svds
            v0 = np.random.default_rng(n_cl).standard_normal(min(centered.shape))
            try:
                _, s, Vt = svds(centered.astype(np.float64), k=k, v0=v0)
                s = s[::-1]
                Vt = Vt[::-1]
            except Exception:
                # Fallback to full SVD if svds still fails
                _, s, Vt = np.linalg.svd(centered, full_matrices=False)
                Vt = Vt[:k]
                s = s[:k]

        # Keep only PCs with nonzero singular values
        nonzero = s[:len(Vt)] > 1e-10
        if not nonzero.any():
            continue
        Vt = Vt[nonzero]

        # Project onto PCs
        projections = centered @ Vt.T  # (n_cl, n_kept_pcs)

        # MAD per PC — skip PCs with zero MAD (constant projection)
        pc_median = np.median(projections, axis=0)
        deviations = np.abs(projections - pc_median)
        mad = np.median(deviations, axis=0)
        valid_pcs = mad > 1e-10
        if not valid_pcs.any():
            continue

        # Flag points exceeding threshold in ANY valid PC
        is_outlier = np.any(
            deviations[:, valid_pcs] > threshold * mad[valid_pcs], axis=1
        )

        # Map back to global indices
        global_idx = np.where(cl_mask)[0]
        outlier_mask[global_idx[is_outlier]] = True

    if not outlier_mask.any():
        return labels

    if small_cluster_mode == SmallClusterMode.OUTLIER:
        labels[outlier_mask] = -1

    elif small_cluster_mode == SmallClusterMode.ABSORB:
        # Reassign to nearest non-outlier cluster centroid
        non_outlier_mask = ~outlier_mask & (labels >= 0)
        if non_outlier_mask.any():
            centroids = {}
            for cl in np.unique(labels[non_outlier_mask]):
                centroids[cl] = data[(labels == cl) & ~outlier_mask].mean(axis=0)
            if centroids:
                centroid_labels = np.array(list(centroids.keys()))
                centroid_data = np.array(list(centroids.values()))
                outlier_points = data[outlier_mask]
                # Nearest centroid assignment
                dists = np.linalg.norm(
                    outlier_points[:, None, :] - centroid_data[None, :, :], axis=2
                )
                labels[outlier_mask] = centroid_labels[dists.argmin(axis=1)]
        else:
            labels[outlier_mask] = -1  # no non-outlier clusters left

    elif small_cluster_mode == SmallClusterMode.KEEP:
        # Flagged points form a new cluster
        valid_max = labels[labels >= 0].max() if (labels >= 0).any() else -1
        next_label = int(valid_max) + 1
        labels[outlier_mask] = next_label

    # Renumber contiguously (preserve -1)
    labels = reindex_labels(labels, reorder=False)
    return labels
