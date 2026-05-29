#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2025 OpenDSM contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
from __future__ import annotations

import warnings

import pydantic

import numpy as np

from scipy.spatial.distance import cdist, pdist, squareform

from opendsm.common.stats.basic import median_absolute_deviation
from opendsm.common.pydantic_utils import (
    ArbitraryPydanticModel,
    computed_field_cached_property,
)
import numba
from opendsm.common.clustering.metrics.dbcv import dbcv_prevalidated
from opendsm.common.clustering.metrics.settings import DistanceMetric


@numba.jit(nopython=True, cache=True)
def _wc_bc_split_numba(D, labels, n, n_wc, n_bc):
    """Split upper-triangle pairwise distances into within- and between-cluster."""
    wc = np.empty(n_wc, dtype=D.dtype)
    bc = np.empty(n_bc, dtype=D.dtype)
    wi = 0
    bi = 0
    for i in range(n):
        li = labels[i]
        for j in range(i + 1, n):
            if li == labels[j]:
                wc[wi] = D[i, j]
                wi += 1
            else:
                bc[bi] = D[i, j]
                bi += 1
    return wc, bc


class _LabeledDistanceProvider:
    """Extracts pairwise distance submatrices between cluster pairs on demand.

    Within-cluster submatrices (i == j) are cached since multiple indices
    reuse them. Between-cluster submatrices are extracted fresh each time
    to avoid duplicating the full n×n matrix in memory.
    """

    __slots__ = ("_dist", "_label_indices", "_diag_cache")

    def __init__(self, dist: np.ndarray, label_indices: dict[int, np.ndarray]) -> None:
        self._dist = dist
        self._label_indices = label_indices
        self._diag_cache: dict[int, np.ndarray] = {}

    def __getitem__(self, key: tuple[int, int]) -> np.ndarray:
        li, lj = key
        if li == lj:
            if li not in self._diag_cache:
                idx = self._label_indices[li]
                sub = self._dist[np.ix_(idx, idx)]
                sub.flags.writeable = False
                self._diag_cache[li] = sub
            return self._diag_cache[li]
        
        return self._dist[np.ix_(self._label_indices[li], self._label_indices[lj])]


class _ClusterPairDistanceMetrics(ArbitraryPydanticModel):
    """
    Metrics between clusters
    """

    cluster_ids: tuple[int, int] | None = pydantic.Field(
        default=None,
        description="The two clusters to compare"
    )

    distance: np.ndarray = pydantic.Field(
        exclude=True,
        repr=False,
    )

    @computed_field_cached_property()
    def n(self) -> int:
        return self.distance.size

    @computed_field_cached_property()
    def sum_of_squares(self) -> float:
        return np.sum(self.distance**2)

    @computed_field_cached_property()
    def mean(self) -> float:
        return np.mean(self.distance)

    @computed_field_cached_property()
    def median(self) -> float:
        return np.median(self.distance)

    @computed_field_cached_property()
    def var(self) -> float:
        return np.var(self.distance)

    @computed_field_cached_property()
    def std(self) -> float:
        return np.std(self.distance)

    @computed_field_cached_property()
    def mad(self) -> float:
        return median_absolute_deviation(self.distance)
    
    @computed_field_cached_property()
    def lower_quantile(self) -> float:
        return np.quantile(self.distance, 0.05)

    @computed_field_cached_property()
    def upper_quantile(self) -> float:
        return np.quantile(self.distance, 0.95)

    @computed_field_cached_property()
    def min(self) -> float:
        positive = self.distance[self.distance > 0]
        return float(np.min(positive)) if positive.size > 0 else 0.0
    
    @computed_field_cached_property()
    def max(self) -> float:
        return np.max(self.distance)


class _SingleClusterMetrics(ArbitraryPydanticModel):
    """
    Metrics within a single cluster
    """
    cluster_id: int | None = pydantic.Field(
        default=None,
    )

    n: int = pydantic.Field()

    mean: np.ndarray = pydantic.Field()

    median: np.ndarray = pydantic.Field()

    var: np.ndarray | None = pydantic.Field(
        default=None,
    )

    distance: dict[int, _ClusterPairDistanceMetrics] | _ClusterPairDistanceMetrics = pydantic.Field()

    distance_to_mean: dict[int | str, _ClusterPairDistanceMetrics] | _ClusterPairDistanceMetrics = pydantic.Field()

    distance_to_median: dict[int | str, _ClusterPairDistanceMetrics] | _ClusterPairDistanceMetrics = pydantic.Field()

    mean_distance_intra_cluster: np.ndarray | None = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
    )

    median_distance_intra_cluster: np.ndarray | None = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
    )

    mean_distance_to_nearest_cluster: np.ndarray | None = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
    )

    median_distance_to_nearest_cluster: np.ndarray | None = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
    )

    @computed_field_cached_property()
    def var_norm(self) -> float | None:
        """Norm of the per-dimension variance vector: ||σ||"""
        if self.var is None:
            return None
        
        return np.linalg.norm(self.var)

    @computed_field_cached_property()
    def within_pairwise_distances(self) -> np.ndarray | None:
        """Upper triangle of intra-cluster pairwise distances (no diagonal, no duplicates)"""
        if self.cluster_id is None or not isinstance(self.distance, dict):
            return None

        d = self.distance[self.cluster_id].distance
        return d[np.triu_indices_from(d, k=1)]

    @computed_field_cached_property()
    def between_pairwise_distances(self) -> np.ndarray | None:
        """All pairwise distances from this cluster to other clusters"""
        if self.cluster_id is None or not isinstance(self.distance, dict):
            return None

        parts = []
        for label_j, pair_metrics in self.distance.items():
            if label_j != self.cluster_id:
                parts.append(pair_metrics.distance.ravel())

        return np.concatenate(parts) if parts else np.array([])

    @computed_field_cached_property()
    def mean_silhouette_coefficient(self) -> np.ndarray:
        if self.mean_distance_intra_cluster is None:
            return None

        a = self.mean_distance_intra_cluster
        b = self.mean_distance_to_nearest_cluster
        
        return (b - a) / np.maximum(a, b)

    @computed_field_cached_property()
    def median_silhouette_coefficient(self) -> np.ndarray:
        if self.median_distance_intra_cluster is None:
            return None

        a = self.median_distance_intra_cluster
        b = self.median_distance_to_nearest_cluster

        return (b - a) / np.maximum(a, b)


class SingleKMetrics(ArbitraryPydanticModel):
    # TODO: Update the doc string
    """Input dataframe to be used for metrics calculations"""
    data: np.ndarray = pydantic.Field(
        exclude=True,
        repr=False,
    )

    labels: np.ndarray = pydantic.Field(
        exclude=True,
        repr=False,
    )

    distance_metric: DistanceMetric = pydantic.Field(
        default=DistanceMetric.EUCLIDEAN,
    )

    seed: int | None = pydantic.Field(
        default=None,
    )


    # Distance matrix — computed lazily on first access.  LabelStore injects
    # a shared DistanceProvider so the O(n²p) pdist is computed at most once
    # per dataset regardless of how many k-values are evaluated.  Indices that
    # never read the distance matrix (e.g. davies_bouldin, calinski_harabasz)
    # will not trigger computation at all.
    distance: np.ndarray | None = pydantic.Field(
        default=None, exclude=True, repr=False,
    )

    _eps: float = 1e-10
    _all: int = -999
    _merged_full: np.ndarray = pydantic.PrivateAttr(default_factory=lambda: np.array([]))
    _coverage: float = pydantic.PrivateAttr(default=1.0)
    _dist_provider: object | None = pydantic.PrivateAttr(default=None)


    @classmethod
    def available_indices(cls) -> list[str]:
        """Return names of all *_index computed properties.

        This is the single source of truth for valid metric names.
        Adding a new *_index property to this class automatically
        makes it available in the score council.
        """
        return sorted(name for name in dir(cls) if name.endswith('_index'))

    @pydantic.model_validator(mode='after')
    def _validate_data(self) -> 'SingleKMetrics':
        if self.data.shape[0] == 0:
            raise ValueError("Data must have at least one row")

        if self.labels.shape[0] == 0:
            raise ValueError("Labels must have at least one row")

        if self.labels.shape[0] != self.data.shape[0]:
            raise ValueError("Labels and data must have the same length")

        label_min = self.labels.min()

        # Ensure _all sentinel doesn't collide with actual labels
        if label_min < self._all:
            len_label_min = len(str(abs(int(label_min))))
            self._all = -int('9' * len_label_min)

            # and just in case in case
            if self._all == label_min:
                self._all = -int('9' * (len_label_min + 1))

        return self

    @property
    def _distance(self) -> np.ndarray:
        """Return the pairwise distance matrix, computing it lazily on first access."""
        if self.distance is None:
            if self._dist_provider is not None:
                self.distance = self._dist_provider.get()
            else:
                self.distance = squareform(pdist(self.data.astype(np.float32, copy=False))).astype(np.float32)
        return self.distance

    @computed_field_cached_property()
    def n_total(self) -> int:
        return self.data.shape[0]

    @computed_field_cached_property()
    def unique_labels(self) -> np.ndarray:
        return np.unique(self.labels)

    @computed_field_cached_property()
    def label_count(self) -> int:
        return len(self.unique_labels)

    @computed_field_cached_property()
    def _is_single_cluster(self) -> bool:
        """True when there is only one cluster (k=1). Most indices are
        undefined for k=1 and should return NaN to abstain from voting."""
        return self.label_count < 2

    @computed_field_cached_property()
    def _has_rank_deficient_cluster(self) -> bool:
        """True when any cluster has fewer points than features, making
        its scatter matrix singular.  Determinant-based indices must
        abstain (return NaN) because det(W_k) = 0 trivially."""
        d = self.data.shape[1]
        return any(len(self._label_indices[lbl]) < d
                   for lbl in self.unique_labels)

    def _safe_index(self, fn) -> float:
        """Wrap an index computation: return NaN for k=1 and suppress
        numpy divide/invalid warnings from degenerate edge cases."""
        if self._is_single_cluster:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                return float(fn())
            except (ZeroDivisionError, FloatingPointError):
                return np.nan

    @computed_field_cached_property()
    def _label_indices(self) -> dict[int, np.ndarray]:
        return {label: np.where(self.labels == label)[0]
                for label in self.unique_labels}

    @computed_field_cached_property()
    def _n(self) -> np.ndarray:
        cluster_sizes = [len(self._label_indices[label]) for label in self.unique_labels]
        return np.array([self.n_total, *cluster_sizes])

    @computed_field_cached_property()
    def _mean(self) -> np.ndarray:
        means = [np.mean(self.data, axis=0)]
        for label in self.unique_labels:
            means.append(np.mean(self.data[self._label_indices[label]], axis=0))

        return np.array(means)

    @computed_field_cached_property()
    def _median(self) -> np.ndarray:
        medians = [np.median(self.data, axis=0)]
        for label in self.unique_labels:
            medians.append(np.median(self.data[self._label_indices[label]], axis=0))

        return np.array(medians)

    @computed_field_cached_property()
    def _var(self) -> np.ndarray:
        variances = [np.var(self.data, axis=0)]
        for label in self.unique_labels:
            variances.append(np.var(self.data[self._label_indices[label]], axis=0))

        return np.array(variances)

    @computed_field_cached_property()
    def _cluster_mean_distances(self) -> np.ndarray:
        """(n, k) matrix: mean distance from every point to every cluster.

        Built via one matrix multiply: D @ M / sizes, where M is the
        (n, k) cluster-membership matrix (column c is the indicator for
        cluster c).  The self-distance (0) is included in the sum but
        since it equals zero it does not affect the result; divide by
        (n_c - 1) at the call site when excluding self for intra-cluster
        mean.
        """
        n, k = self.n_total, self.label_count
        sizes = np.array(
            [len(self._label_indices[lbl]) for lbl in self.unique_labels],
            dtype=self._distance.dtype,
        )
        M = np.zeros((n, k), dtype=self._distance.dtype)
        for c_idx, lbl in enumerate(self.unique_labels):
            M[self._label_indices[lbl], c_idx] = 1.0
        return (self._distance @ M) / sizes

    @computed_field_cached_property()
    def _distance_to_mean(self) -> np.ndarray:
        return cdist(self.data, self._mean)

    @computed_field_cached_property()
    def _distance_to_median(self) -> np.ndarray:
        return cdist(self.data, self._median)
    
    @computed_field_cached_property()
    def _labeled_distance(self) -> _LabeledDistanceProvider:
        return _LabeledDistanceProvider(self._distance, self._label_indices)

    def _labeled_distance_to_centroid(self, distance_matrix: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
        unique_labels = [self._all, *self.unique_labels]
        all_idx = np.arange(self.n_total)

        data = {}
        for label_i in unique_labels:
            idx_i = all_idx if label_i == self._all else self._label_indices[label_i]

            for col_idx, label_j in enumerate(unique_labels):
                data[label_i, label_j] = distance_matrix[idx_i, col_idx]

        return data

    @computed_field_cached_property()
    def _labeled_distance_to_mean(self) -> dict[tuple[int, int], np.ndarray]:
        return self._labeled_distance_to_centroid(self._distance_to_mean)

    @computed_field_cached_property()
    def _labeled_distance_to_median(self) -> dict[tuple[int, int], np.ndarray]:
        return self._labeled_distance_to_centroid(self._distance_to_median)

    def _labeled_distance_to_nearest_cluster(self, agg: str = "mean") -> dict[int, np.ndarray]:
        if agg == "mean":
            # One matrix multiply gives mean distances from every point to
            # every cluster; then per-cluster_i: mask out self and take min.
            mean_dists = self._cluster_mean_distances   # (n, k)
            data = {}
            for c_i, label_i in enumerate(self.unique_labels):
                idx_i = self._label_indices[label_i]
                rows  = mean_dists[idx_i, :].copy()    # (n_i, k)
                rows[:, c_i] = np.inf
                data[label_i] = rows.min(axis=1)
            return data

        # Vectorised median via pre-sorted distance matrix.
        #
        # For each cluster_i, precompute all-cluster boolean masks in one
        # vectorised broadcast, then use them for both intra- and inter-cluster
        # median extraction without re-sorting.
        return self._median_silhouette_arrays[1]
    
    @computed_field_cached_property()
    def _cluster_median_distances(self) -> np.ndarray:
        """(n, k) matrix: median distance from every point to every cluster.

        cluster_median_distances[p, c] = median of the n_c distances from point p
        to all members of cluster c (self-distance 0 is included for p ∈ cluster c).

        Own-cluster column values are masked out in _median_silhouette_arrays so the
        self-distance (0) inclusion does not affect silhouette scores.
        """
        n, k = self.n_total, self.label_count
        dist = self._distance
        result = np.empty((n, k), dtype=dist.dtype)

        for c, label_c in enumerate(self.unique_labels):
            idx_c = self._label_indices[label_c]
            n_c = len(idx_c)
            dists_c = dist[idx_c, :].T   # (n, n_c) via (n_c, n).T
            mid = n_c // 2
            if n_c % 2 == 1:
                part = np.partition(dists_c, mid, axis=1)
                result[:, c] = part[:, mid]
            else:
                sr = np.sort(dists_c, axis=1)
                result[:, c] = (sr[:, mid - 1] + sr[:, mid]) * 0.5

        return result

    @computed_field_cached_property()
    def _median_silhouette_arrays(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Intra- and nearest-cluster median distances, computed together.

        Returns
        -------
        intra   : dict label → (n_i,) array of median intra-cluster distances
        nearest : dict label → (n_i,) array of min-median nearest-cluster distances
        """
        dist            = self._distance                   # (n, n) float32
        cluster_medians = self._cluster_median_distances   # (n, k)

        intra   = {}
        nearest = {}

        for c_i, label_i in enumerate(self.unique_labels):
            idx_i = self._label_indices[label_i]
            n_i   = len(idx_i)

            if n_i == 1:
                intra[label_i]   = np.array([0.0], dtype=dist.dtype)
                nearest[label_i] = np.array([0.0], dtype=dist.dtype)
                continue

            # ── intra-cluster median (exclude self) ──────────────────────
            # Extract (n_i, n_i) submatrix; set diagonal to inf to exclude
            # self-distance (0), then partition to find the median of n_i-1
            # distances.  Self is placed last by inf → offset position by 1.
            dists_ii = dist[np.ix_(idx_i, idx_i)].copy()
            np.fill_diagonal(dists_ii, np.inf)
            n_rem = n_i - 1   # number of non-self distances per row
            # inf sorts last → non-self values occupy positions 0..n_rem-1
            if n_rem % 2 == 1:
                pos = n_rem // 2
                part = np.partition(dists_ii, pos, axis=1)
                intra[label_i] = part[:, pos]
            else:
                lo, hi = n_rem // 2 - 1, n_rem // 2
                sr = np.sort(dists_ii, axis=1)
                intra[label_i] = (sr[:, lo] + sr[:, hi]) * 0.5

            # ── nearest-cluster median ────────────────────────────────────
            med_i = cluster_medians[idx_i, :].copy()   # (n_i, k)
            med_i[:, c_i] = np.inf
            nearest[label_i] = med_i.min(axis=1)

        return intra, nearest

    @computed_field_cached_property()
    def _labeled_mean_distance_to_nearest_cluster(self) -> dict[int, np.ndarray]:
        return self._labeled_distance_to_nearest_cluster(agg="mean")
    
    @computed_field_cached_property()
    def _labeled_median_distance_to_nearest_cluster(self) -> dict[int, np.ndarray]:
        return self._labeled_distance_to_nearest_cluster(agg="median")

    def _labeled_distance_intra_cluster(self, agg: str = "mean") -> dict[int, np.ndarray]:
        if agg == "mean":
            # Reuse _cluster_mean_distances (already one matmul).
            # _cluster_mean_distances[i, c] = (sum of distances from i to
            # all cluster-c points, including self=0) / n_c.
            # For intra-cluster mean excluding self: multiply by n_c/(n_c-1).
            mean_dists = self._cluster_mean_distances   # (n, k)
            data = {}
            for c_idx, label_i in enumerate(self.unique_labels):
                idx_i = self._label_indices[label_i]
                n_i   = len(idx_i)
                if n_i == 1:
                    data[label_i] = np.array([0.0], dtype=mean_dists.dtype)
                else:
                    data[label_i] = mean_dists[idx_i, c_idx] * (n_i / (n_i - 1))
            return data

        return self._median_silhouette_arrays[0]
    
    @computed_field_cached_property()
    def _labeled_mean_distance_intra_cluster(self) -> dict[int, np.ndarray]:
        return self._labeled_distance_intra_cluster(agg="mean")
    
    @computed_field_cached_property()
    def _labeled_median_distance_intra_cluster(self) -> dict[int, np.ndarray]:
        return self._labeled_distance_intra_cluster(agg="median")
        
    @computed_field_cached_property()
    def all(self) -> _SingleClusterMetrics:
        key = (self._all, self._all)

        distance = _ClusterPairDistanceMetrics(
            distance=self._distance,
        )

        distance_to_mean = _ClusterPairDistanceMetrics(
            distance=self._labeled_distance_to_mean[key],
        )
        
        distance_to_median = _ClusterPairDistanceMetrics(
            distance=self._labeled_distance_to_median[key],
        )

        return _SingleClusterMetrics(
            cluster_id=None,
            n=self._n[0],
            mean=self._mean[0],
            median=self._median[0],
            var=self._var[0],
            distance=distance,
            distance_to_mean=distance_to_mean,
            distance_to_median=distance_to_median,
        )

    @computed_field_cached_property()
    def cluster(self) -> dict[int, _SingleClusterMetrics]:
        data = {}
        for i, label in enumerate(self.unique_labels):
            # single cluster metrics
            n = self._n[i + 1]
            mean = self._mean[i + 1]
            median = self._median[i + 1]
            var = self._var[i + 1]

            # pair distance metrics
            distance = {}
            distance_to_mean = {"all": _ClusterPairDistanceMetrics(
                cluster_ids=(label, self._all),
                distance=self._labeled_distance_to_mean[(label, self._all)],
            )}
            distance_to_median = {"all": _ClusterPairDistanceMetrics(
                cluster_ids=(label, self._all),
                distance=self._labeled_distance_to_median[(label, self._all)],
            )}

            for label_j in self.unique_labels:
                key = (label, label_j)

                distance[label_j] = _ClusterPairDistanceMetrics(
                    cluster_ids=key,
                    distance=self._labeled_distance[key],
                )

                distance_to_mean[label_j] = _ClusterPairDistanceMetrics(
                    cluster_ids=key,
                    distance=self._labeled_distance_to_mean[key],
                )

                distance_to_median[label_j] = _ClusterPairDistanceMetrics(
                    cluster_ids=key,
                    distance=self._labeled_distance_to_median[key],
                )

            mean_distance_intra_cluster = self._labeled_mean_distance_intra_cluster[label]
            median_distance_intra_cluster = self._labeled_median_distance_intra_cluster[label]
            mean_distance_to_nearest_cluster = self._labeled_mean_distance_to_nearest_cluster[label]
            median_distance_to_nearest_cluster = self._labeled_median_distance_to_nearest_cluster[label]
            
            data[label] = _SingleClusterMetrics(
                cluster_id=label,
                n=n,
                mean=mean,
                median=median,
                var=var,
                distance=distance,
                distance_to_mean=distance_to_mean,
                distance_to_median=distance_to_median,
                mean_distance_intra_cluster=mean_distance_intra_cluster,
                median_distance_intra_cluster=median_distance_intra_cluster,
                mean_distance_to_nearest_cluster=mean_distance_to_nearest_cluster,
                median_distance_to_nearest_cluster=median_distance_to_nearest_cluster,
            )

        return data

    # -------------------------------------------------------------------------
    # Private infrastructure: scatter matrices, sum-of-squares, pairwise vectors
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def _WCSM(self) -> dict[int, np.ndarray]:
        """
        Within-Cluster Scatter Matrices
        Returns a dictionary mapping cluster labels to their scatter matrices
        """
        # Compute scatter matrix for each cluster
        scatter_matrices = {}
        for i, label in enumerate(self.unique_labels):
            cluster_data = self.data[self._label_indices[label]].astype(np.float64, copy=False)
            cluster_mean = self._mean[i + 1].astype(np.float64, copy=False)

            # Compute scatter matrix for this cluster: Σ(x - mean)(x - mean)^T
            centered_data = cluster_data - cluster_mean
            scatter_matrices[label] = centered_data.T @ centered_data

        return scatter_matrices

    @computed_field_cached_property()
    def _sum_WCSM(self) -> np.ndarray:
        """
        Pooled Within-Cluster Scatter Matrix
        Returns the sum of all within-cluster scatter matrices
        """
        return sum(self._WCSM.values())

    @computed_field_cached_property()
    def _TSM(self) -> np.ndarray:
        """
        Total Scatter Matrix
        """
        centered_data = self.data.astype(np.float64, copy=False) - self._mean[0].astype(np.float64, copy=False)
        return centered_data.T @ centered_data

    @computed_field_cached_property()
    def _WCSS(self) -> float:
        """
        Within-Cluster Sum of Squares
        Computed as the trace of the summed within-cluster scatter matrix
        """
        return np.trace(self._sum_WCSM)

    @computed_field_cached_property()
    def _BCSS(self) -> float:
        """
        Between-Cluster Sum of Squares
        """
        diffs = self._mean[1:] - self._mean[0]
        sq_dists = np.sum(diffs ** 2, axis=1)
        BCSS = np.dot(self._n[1:], sq_dists)

        return BCSS

    @computed_field_cached_property()
    def _wc_bc_pairwise(self) -> tuple[np.ndarray, np.ndarray]:
        """Within- and between-cluster pairwise distances (upper triangle)."""
        sizes = np.array([len(self._label_indices[lbl]) for lbl in self.unique_labels])
        n_wc = int(np.sum(sizes * (sizes - 1)) // 2)
        n_bc = self.n_total * (self.n_total - 1) // 2 - n_wc
        return _wc_bc_split_numba(self._distance, self.labels, self.n_total, n_wc, n_bc)

    @computed_field_cached_property()
    def _WC_pairwise_distances(self) -> np.ndarray:
        """Aggregated within-cluster pairwise distances across all clusters"""
        return self._wc_bc_pairwise[0]

    @computed_field_cached_property()
    def _BC_pairwise_distances(self) -> np.ndarray:
        """Aggregated between-cluster pairwise distances (deduplicated across cluster pairs)"""
        return self._wc_bc_pairwise[1]

    @computed_field_cached_property()
    def _mean_scatter(self) -> float:
        """Average scattering: (1/K) × Σ_k ||σ(C_k)|| / ||σ(D)||

        Computed directly from ``_var`` to avoid pydantic object construction in
        ``self.all`` / ``self.cluster``.
        """
        total_var_norm = float(np.linalg.norm(self._var[0]))
        if total_var_norm < self._eps:
            return 0.0
        cluster_var_norms = np.linalg.norm(self._var[1:], axis=1)
        return float(np.mean(cluster_var_norms) / total_var_norm)

    # -------------------------------------------------------------------------
    # Compactness indices (within-cluster quality only)
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def sum_of_squared_errors_index(self) -> float:
        """Sum of squared errors — equivalent to WCSS.

        ``SSE = Σ_k Σ_{x_i ∈ C_k} ||x_i − c_k||²``

        Range [0, ∞). Natural direction: **minimize**.
        """
        res = self._WCSS

        return res

    @computed_field_cached_property()
    def mean_squared_error_index(self) -> float:
        """WCSS normalised by sample count.

        ``MSE = WCSS / n``

        Range [0, ∞). Natural direction: **minimize**.
        """

        n = self.n_total  # number of data points
        WCSS = self._WCSS

        res = WCSS / n

        return res

    @computed_field_cached_property()
    def ball_hall_index(self) -> float:
        """Ball-Hall index (Ball & Hall, 1965) — WCSS per cluster.

        ``BH = WCSS / k``

        Range [0, ∞). Natural direction: **minimize**.
        """

        k = self.label_count  # number of clusters
        WCSS = self._WCSS  # Within Cluster Sum of Squares (WCSS)
        res = WCSS / k

        return res

    @computed_field_cached_property()
    def banfeld_raftery_index(self) -> float:
        """Banfeld-Raftery index (1992) — log-trace scatter per cluster.

        ``BR = Σ_k n_k · log(tr(W_k) / n_k)``

        Penalises large within-cluster scatter relative to cluster size.
        Range (−∞, ∞). Natural direction: **minimize**.
        """
        if self._is_single_cluster or self._has_rank_deficient_cluster:
            return np.nan

        n_k = self._n[1:]  # cluster sizes
        traces = np.array([np.trace(self._WCSM[label]) for label in self.unique_labels])

        # Replace zero traces with _eps to avoid log(0)
        traces_safe = np.where(traces > 0, traces, self._eps)
        res = np.sum(n_k * np.log(traces_safe / n_k))

        return res

    @computed_field_cached_property()
    def scott_symons_index(self) -> float:
        """Scott-Symons index (1987) — log-determinant scatter per cluster.

        ``SS = Σ_k n_k · log(det(W_k / n_k))``

        Requires n_k > p for each cluster; penalises ellipsoidal volume.
        Range (−∞, ∞). Natural direction: **minimize**.
        """
        if self._is_single_cluster or self._has_rank_deficient_cluster:
            return np.nan

        res = 0.0
        for i, label in enumerate(self.unique_labels):
            n_k = self._n[i + 1]
            W_k = self._WCSM[label]

            try:
                sign, logdet = np.linalg.slogdet(W_k / n_k)
                if sign <= 0:
                    logdet = np.log(self._eps)
                res += n_k * logdet
            except np.linalg.LinAlgError:
                res += n_k * np.log(self._eps)

        return res

    @computed_field_cached_property()
    def trace_w_index(self) -> float:
        """Trace-W index — trace of the pooled within-cluster scatter matrix.

        ``tr(W) = Σ_k tr(W_k)``

        Equivalent to WCSS expressed as a matrix trace.
        Range [0, ∞). Natural direction: **minimize**.
        """

        res = np.trace(self._sum_WCSM)

        return res

    # -------------------------------------------------------------------------
    # Compactness + Separation indices (combined)
    # -------------------------------------------------------------------------

    def _silhouette_coefficients(self, variant: str = "mean") -> np.ndarray:
        if self._is_single_cluster:
            return np.zeros(self.n_total)
        if variant == "mean":
            intra = self._labeled_mean_distance_intra_cluster
            nearest = self._labeled_mean_distance_to_nearest_cluster
        else:
            intra = self._labeled_median_distance_intra_cluster
            nearest = self._labeled_median_distance_to_nearest_cluster

        idx = 0
        coefficients = np.empty(self.n_total)
        for label in self.unique_labels:
            a = intra[label]
            b = nearest[label]
            n_points = len(a)
            if n_points < 2:
                # Singleton cluster: s_i = 0 (neutral, not "perfect").
                # A singleton has no intra-cluster information, so its
                # silhouette is undefined.  Convention matches sklearn.
                coefficients[idx:idx+n_points] = 0.0
            else:
                denom = np.maximum(a, b)
                coefs = np.divide(b - a, denom, out=np.zeros_like(denom), where=denom > 0)
                coefficients[idx:idx+n_points] = coefs
            idx += n_points

        return coefficients

    @computed_field_cached_property()
    def silhouette_index(self) -> float:
        """Mean silhouette coefficient (Rousseeuw, 1987).

        ``s_i = (b_i − a_i) / max(a_i, b_i)`` averaged over all points,
        where a_i = mean intra-cluster distance, b_i = mean distance to
        nearest other cluster.

        Range [−1, 1]. Natural direction: **maximize** — negated to minimize.
        Prefer ``silhouette_median_index`` for imbalanced clusters.
        """
        if self._is_single_cluster:
            return np.nan
        res = np.mean(self._silhouette_coefficients("mean"))

        res *= -1

        return res

    @computed_field_cached_property()
    def silhouette_median_index(self) -> float:
        """Median silhouette coefficient.

        Median-aggregated variant of the silhouette score using median intra-
        and nearest-cluster distances. More robust than the mean to imbalanced
        or high-variance clusters.

        Range [−1, 1]. Natural direction: **maximize** — negated to minimize.
        """
        if self._is_single_cluster:
            return np.nan
        res = np.median(self._silhouette_coefficients("median"))

        res *= -1

        return res

    @computed_field_cached_property()
    def davies_bouldin_index(self) -> float:
        """Davies-Bouldin index (Davies & Bouldin, 1979).

        Mean over clusters of the worst-neighbour similarity ratio:
        ``DB = (1/k) Σ_i max_{j≠i} (s_i + s_j) / d(c_i, c_j)``
        where s_i = mean intra-cluster distance, d = centroid separation.

        Range [0, ∞). Natural direction: **minimize**.
        """
        if self._is_single_cluster:
            return np.nan

        # Exclude singleton clusters: their scatter is trivially 0,
        # which makes them look "perfect" and biases DB downward.
        # Compute DB only over clusters with >= 2 members.
        multi_mask = np.array([
            len(self._label_indices[lbl]) >= 2
            for lbl in self.unique_labels
        ])
        if multi_mask.sum() < 2:
            return np.nan

        multi_indices = np.where(multi_mask)[0]
        k_eff = len(multi_indices)

        intracluster_distance = np.array([
            self._distance_to_mean[self._label_indices[self.unique_labels[i]], i + 1].mean()
            for i in multi_indices
        ], dtype=np.float32)

        centroids = self._mean[1:].astype(np.float32, copy=False)[multi_indices]
        intercluster_distance = squareform(pdist(centroids))

        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = (intracluster_distance[:, None] + intracluster_distance[None, :]) / intercluster_distance

        np.fill_diagonal(similarity, 0)
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0)

        res = np.sum(np.max(similarity, axis=1)) / k_eff

        return res

    @computed_field_cached_property()
    def calinski_harabasz_index(self) -> float:
        """Calinski-Harabasz index / Variance Ratio Criterion (1974).

        Ratio of between-cluster to within-cluster variance, corrected for
        degrees of freedom.
        ``CH = (BCSS / (k−1)) / (WCSS / (n−k))``

        Range [0, ∞). Natural direction: **maximize** — negated to minimize.
        Carries a low-k bias; counterbalanced by other council members.
        """
        if self._is_single_cluster:
            return np.nan
        k = self.label_count
        n = self.n_total

        BCSS = self._BCSS  # Between Cluster Sum of Squares (BCSS)
        WCSS = self._WCSS  # Within Cluster Sum of Squares (WCSS)

        if WCSS < self._eps:
            # Perfect compactness — CH → ∞. Negated so that minimize = better.
            return -np.inf

        res = (BCSS / WCSS) * ((n - k) / (k - 1.0))

        res *= -1

        return res

    @computed_field_cached_property()
    def variance_ratio_criterion(self) -> float:
        return self.calinski_harabasz_index

    @computed_field_cached_property()
    def dunn_index(self) -> float:
        """Dunn index (Dunn, 1974) — separation over compactness.

        ``DI = min_{i≠j}(inter_dist(i,j)) / max_k(diameter(k))``

        Uses a modified inter-cluster distance (point-to-centroid) to reduce
        sensitivity to outliers vs. the standard all-pairs definition.
        Range [0, ∞). Natural direction: **maximize** — negated to minimize.
        Fragile at high variance; prefer ``generalized_dunn_index``.
        """
        if self._is_single_cluster:
            return np.nan

        n_clusters = len(self.unique_labels)

        use_modified = True

        # Find minimum inter-cluster distance
        min_inter_distance = np.inf
        if use_modified:
            # Modified: min distance from any point in cluster k1 to centroid of cluster k0
            # Reuse precomputed distances to means: column i+1 is distance to cluster i's mean
            for k0 in range(n_clusters - 1):
                for k1 in range(k0 + 1, n_clusters):
                    label_k1 = self.unique_labels[k1]
                    # _distance_to_mean column k0+1 has distances to cluster k0's centroid
                    dists = self._distance_to_mean[self._label_indices[label_k1], k0 + 1]
                    min_inter_distance = min(min_inter_distance, np.min(dists))
        else:
            # Standard: distance between all point pairs in different clusters
            for i, label_i in enumerate(self.unique_labels):
                for label_j in self.unique_labels[i+1:]:
                    min_dist = np.min(self._labeled_distance[label_i, label_j])
                    min_inter_distance = min(min_inter_distance, min_dist)

        # Find maximum intra-cluster diameter
        max_intra_diameter = max(
            np.max(self._labeled_distance[label, label])
            for label in self.unique_labels
        )

        # Avoid division by zero
        if max_intra_diameter < self._eps:
            res = np.inf
        else:
            res = min_inter_distance / max_intra_diameter

        res *= -1

        return res

    @computed_field_cached_property()
    def xie_beni_index(self) -> float:
        """Xie-Beni index (Xie & Beni, 1991).

        ``XB = WCSS / (n · d²_min)``
        where d_min is the minimum centroid-to-centroid distance.

        Sensitive to closely-spaced centroids. Range [0, ∞).
        Natural direction: **minimize**.
        """
        if self._is_single_cluster:
            return np.nan

        # Exclude singleton clusters: their WCSS contribution is trivially
        # 0, which biases XB downward.  Compute over non-singleton clusters
        # and their centroids only.
        multi_labels = [lbl for lbl in self.unique_labels
                        if len(self._label_indices[lbl]) >= 2]
        if len(multi_labels) < 1:
            return np.nan

        # WCSS and n over non-singleton points only
        n_eff = 0
        wcss_eff = 0.0
        for lbl in multi_labels:
            idx = self._label_indices[lbl]
            c_idx = list(self.unique_labels).index(lbl)
            dists = self._distance_to_mean[idx, c_idx + 1]
            wcss_eff += float(np.sum(dists ** 2))
            n_eff += len(idx)

        # Centroid separation over non-singleton centroids
        multi_c_indices = [list(self.unique_labels).index(lbl)
                           for lbl in multi_labels]
        centroids = self._mean[1:][multi_c_indices]

        if len(centroids) > 1:
            d_sq = pdist(centroids, metric='sqeuclidean')
            d_min_squared = float(np.min(d_sq))
        else:
            return np.inf

        if d_min_squared < self._eps:
            return np.inf

        return wcss_eff / (n_eff * d_min_squared)

    @computed_field_cached_property()
    def duda_hart_index(self) -> float:
        """Duda-Hart index (Duda & Hart, 1973).

        Ratio of mean intra-cluster distance to mean inter-cluster distance,
        aggregated across all clusters.

        Range [0, ∞). Natural direction: **minimize**.
        Returns NaN (abstains) when total inter-cluster distance is zero
        (coincident centroids — ratio undefined).
        """

        intracluster_distance = 0.0
        intercluster_distance = 0.0
        for label_i in self.unique_labels:
            intracluster_distance += np.mean(self._labeled_distance_to_mean[(label_i, label_i)])

            inter_sum = sum(
                self._labeled_distance[label_i, lj].sum()
                for lj in self.unique_labels if lj != label_i
            )
            inter_count = sum(
                self._labeled_distance[label_i, lj].size
                for lj in self.unique_labels if lj != label_i
            )
            intercluster_distance += inter_sum / inter_count

        if intercluster_distance < self._eps:
            return np.nan  # coincident centroids → ratio undefined

        return intracluster_distance / intercluster_distance

    @computed_field_cached_property()
    def c_index(self) -> float:
        """C-index (Hubert & Schultz, 1976).

        Ranks within-cluster distances against all pairwise distances:
        ``C = (S_w − S_min) / (S_max − S_min)``
        where S_w = sum of within-cluster distances, S_min/S_max = sum of
        the N_w smallest/largest overall pairwise distances.

        Range [0, 1]. Natural direction: **minimize**. Nonparametric;
        no distributional assumptions. Returns NaN (abstains) when all
        clusters are singletons (no within-cluster distances).
        """

        within_dists = self._WC_pairwise_distances
        n_w = len(within_dists)

        if n_w == 0:
            return np.nan  # no within-cluster distances → undefined

        S_w = np.sum(within_dists)

        all_dists = np.concatenate([within_dists, self._BC_pairwise_distances])
        n_all = len(all_dists)

        if n_w >= n_all:
            return 0.0  # all distances are within-cluster

        partitioned = np.partition(all_dists, (n_w - 1, n_all - n_w))
        S_min = partitioned[:n_w].sum()
        S_max = partitioned[-n_w:].sum()

        denom = S_max - S_min
        if denom < self._eps:
            res = 0.0
        else:
            res = (S_w - S_min) / denom

        return res

    @computed_field_cached_property()
    def mcclain_rao_index(self) -> float:
        """McClain-Rao index (McClain & Rao, 1975).

        ``MR = mean(within-cluster distances) / mean(between-cluster distances)``

        Range [0, ∞). Natural direction: **minimize**.
        """

        within_dists = self._WC_pairwise_distances
        between_dists = self._BC_pairwise_distances

        if len(within_dists) == 0 or len(between_dists) == 0:
            return np.nan  # no within/between distances → undefined

        mean_within = np.mean(within_dists)
        mean_between = np.mean(between_dists)

        if mean_between < self._eps:
            # 0/0 — all distances are zero (degenerate data) → undefined
            # finite/0 — clusters coincide but have internal spread → worst
            res = np.nan if mean_within < self._eps else np.inf
        else:
            res = mean_within / mean_between

        return res

    @computed_field_cached_property()
    def i_index(self) -> float:
        """I-index / Maulik-Bandyopadhyay index (2002).

        ``I(k) = (E_1 / (k · E_k) · D_k)^2``
        where E_1 = total distance to grand centroid, E_k = total distance
        to assigned centroids, D_k = maximum inter-centroid distance.

        Range [0, ∞). Natural direction: **maximize** — negated to minimize.
        Contains a 1/k penalty similar to CH; avoid using both simultaneously.
        """

        k = self.label_count
        p = 2

        # E_1: sum of distances from all points to grand centroid
        E_1 = np.sum(self._labeled_distance_to_mean[(self._all, self._all)])

        # E_K: sum of distances from each point to its assigned cluster centroid
        E_K = 0.0
        for label in self.unique_labels:
            E_K += np.sum(self._labeled_distance_to_mean[(label, label)])

        # D_K: max distance between any pair of cluster centroids
        cluster_means = self._mean[1:]
        if len(cluster_means) > 1:
            D_K = np.max(pdist(cluster_means))
        else:
            return np.nan  # fewer than 2 cluster means → undefined

        if E_K < self._eps:
            res = np.inf
        else:
            res = ((1.0 / k) * (E_1 / E_K) * D_K) ** p

        res *= -1

        return res

    @computed_field_cached_property()
    def log_ss_ratio_index(self) -> float:
        """Log sum-of-squares ratio.

        ``log(BCSS / WCSS)``

        Monotonically increases with k — not Schulze-compatible as a
        stand-alone voter; use cross-k KL or Hartigan for elbow detection.
        Range (−∞, ∞). Natural direction: **maximize** — negated to minimize.
        """
        if self._is_single_cluster:
            return np.nan

        BCSS = self._BCSS
        WCSS = self._WCSS

        if WCSS < self._eps or BCSS < self._eps:
            return np.nan
        else:
            # log(BCSS / WCSS) = log(BCSS) - log(WCSS)
            res = np.log(BCSS) - np.log(WCSS)

        res *= -1

        return res

    # -------------------------------------------------------------------------
    # Statistical / Correlation indices
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def gamma_index(self) -> float:
        """Hubert's Gamma index (Hubert & Schultz, 1976).

        Rank-correlation between pairwise distances and cluster membership:
        ``Γ = (s⁺ − s⁻) / (s⁺ + s⁻)``
        where s⁺ = concordant pairs (within-cluster dist < between-cluster),
        s⁻ = discordant pairs.

        Range [−1, 1]. Natural direction: **maximize** — negated to minimize.
        Returns NaN (abstains) when no within/between distances exist.
        Returns 0 (neutral) when all pairs tie.
        """

        within_dists = self._WC_pairwise_distances
        between_dists = self._BC_pairwise_distances

        n_b = len(between_dists)
        if n_b == 0 or len(within_dists) == 0:
            return np.nan  # no within/between distances → undefined

        between_sorted = np.sort(between_dists)

        # For each within_dist, count concordant (between > within) and discordant (between < within)
        left_indices = np.searchsorted(between_sorted, within_dists, side='left')
        right_indices = np.searchsorted(between_sorted, within_dists, side='right')

        s_plus = np.sum(n_b - right_indices)   # between > within (concordant)
        s_minus = np.sum(left_indices)          # between < within (discordant)

        denom = s_plus + s_minus
        if denom == 0:
            return np.nan  # all pairs tied → 0/0 undefined
        res = float(s_plus - s_minus) / float(denom)

        res *= -1

        return res

    @computed_field_cached_property()
    def point_biserial_index(self) -> float:
        """Point-biserial correlation (Milligan, 1981).

        Correlation between pairwise distances and binary within/between
        membership indicator:
        ``r_pb = (M_b − M_w) / σ_d · √(n_w · n_b / n_t²)``

        Range [−1, 1]. Natural direction: **maximize** — negated to minimize.
        Returns NaN (abstains) when no within/between distances exist or all
        distances are equal (correlation undefined).
        """

        within_dists = self._WC_pairwise_distances
        between_dists = self._BC_pairwise_distances

        n_w = len(within_dists)
        n_b = len(between_dists)
        n_t = n_w + n_b

        if n_w == 0 or n_b == 0:
            return np.nan  # no within/between distances → undefined

        mean_within = np.mean(within_dists)
        mean_between = np.mean(between_dists)

        # Combined std via parallel variance — avoids concatenating 500K+ elements
        mean_all = (n_w * mean_within + n_b * mean_between) / n_t
        var_all = (
            n_w * (np.var(within_dists) + (mean_within - mean_all) ** 2)
            + n_b * (np.var(between_dists) + (mean_between - mean_all) ** 2)
        ) / n_t
        std_all = np.sqrt(var_all)

        if std_all < self._eps:
            return np.nan  # all distances equal → correlation undefined

        res = ((mean_between - mean_within) / std_all) * np.sqrt(n_w * n_b / (n_t ** 2))

        res *= -1

        return res

    # -------------------------------------------------------------------------
    # Matrix / Determinant indices
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def _det_ratio(self) -> float:
        """
        Raw det(T) / det(W)
        """
        try:
            det_T = np.linalg.det(self._TSM)
            det_W = np.linalg.det(self._sum_WCSM)

            if np.abs(det_W) < self._eps:
                return np.inf
            else:
                return det_T / det_W

        except np.linalg.LinAlgError:
            return np.nan  # decomposition failed → undefined

    @computed_field_cached_property()
    def ksq_detw_index(self) -> float:
        """KSq-DetW index — k² × det(W).

        Measures the volume of the within-cluster scatter, penalised by k².
        The scatter matrix is normalised to [0,1] before computing the
        determinant to improve numerical stability.

        Range (−∞, ∞). Natural direction: **maximize** — negated to minimize.
        Requires n_k > p; unstable with imbalanced or degenerate clusters.
        """
        if self._is_single_cluster or self._has_rank_deficient_cluster:
            return np.nan

        k = self.label_count  # number of clusters
        normalize_scatter_matrix = True

        # Get summed within-cluster scatter matrix
        W = self._sum_WCSM

        # Apply normalization if enabled
        if normalize_scatter_matrix:
            W_min = np.min(W)
            W_max = np.max(W)
            W = (W - W_min) / (W_max - W_min)

        # Compute determinant
        try:
            det_W = np.linalg.det(W)
        except np.linalg.LinAlgError:
            return np.nan  # determinant uncomputable → undefined

        # KSq-DetW = K² × det(W)
        res = (k ** 2) * det_W

        res *= -1

        return res

    @computed_field_cached_property()
    def det_ratio_index(self) -> float:
        """Determinant ratio index (Friedman & Rubin, 1967).

        ``DR = det(T) / det(W)``
        where T = total scatter matrix, W = pooled within-cluster scatter.

        Range [0, ∞). Natural direction: **maximize** — negated to minimize.
        Requires n > p and non-degenerate scatter matrices.
        """
        if self._is_single_cluster:
            return np.nan
        # Singular W when any cluster has fewer points than features
        if self._det_ratio == np.inf:
            return np.nan

        res = self._det_ratio

        res *= -1

        return res

    @computed_field_cached_property()
    def log_det_ratio_index(self) -> float:
        """Log determinant ratio index.

        ``n · log(det(T) / det(W))``

        Log-scale version of ``det_ratio_index``, more numerically stable
        for large n or p.
        Range (−∞, ∞). Natural direction: **maximize** — negated to minimize.
        """
        if self._is_single_cluster:
            return np.nan
        if self._det_ratio == np.inf:
            return np.nan

        n = self.n_total
        res = n * np.log(np.abs(self._det_ratio))

        res *= -1

        return res

    @computed_field_cached_property()
    def trace_wb_index(self) -> float:
        """Trace-WB index — multivariate generalisation of Calinski-Harabasz.

        ``tr(W⁻¹B)`` where B = T − W is the between-cluster scatter matrix.

        Range [0, ∞). Natural direction: **maximize** — negated to minimize.
        Requires W to be invertible; falls back to pseudo-inverse.
        """

        W = self._sum_WCSM
        B = self._TSM - W

        try:
            W_inv = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            W_inv = np.linalg.pinv(W)

        res = np.trace(W_inv @ B)

        res *= -1

        return res

    # -------------------------------------------------------------------------
    # Density-based indices
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def s_dbw_index(self) -> float:
        """S_Dbw index (Halkidi & Vazirgiannis, 2001).

        ``S_Dbw = Scat + Dens_bw``
        Scat = average ratio of cluster variance norm to total variance norm.
        Dens_bw = average density at midpoints between centroid pairs relative
        to the denser cluster centroid.

        Range [0, ∞). Natural direction: **minimize**.
        Requires n ≥ 50; degrades at small n or very high variance.
        """

        k = self.label_count
        if k < 2:
            return self._mean_scatter

        scat = self._mean_scatter

        # stdev: average norm of cluster variance vectors (neighborhood radius)
        # _var[1:] contains per-cluster per-dimension variance; norm gives scalar per cluster.
        cluster_var_norms = np.linalg.norm(self._var[1:], axis=1)
        stdev = float(np.mean(cluster_var_norms))

        if stdev < self._eps:
            # All clusters are single points; no inter-cluster density
            res = scat
            return res

        cluster_means = self._mean[1:]
        n = self.n_total
        p = self.data.shape[1]

        # Vectorised Dens_bw over all k*(k-1) directed pairs.
        #
        # Precompute once:
        #   dtm[i, c] = dist from point i to centroid c  (n, k)
        #   in_radius_dtm[i, c] = dtm[i, c] <= stdev     (n, k) bool
        #   M[i, c]             = point i ∈ cluster c    (n, k) bool
        #   union_mask[i, ci, cj] = M[i,ci] | M[i,cj]   (n, k, k) bool
        dtm = self._distance_to_mean[:, 1:]                    # (n, k)
        in_radius_dtm = dtm <= stdev                           # (n, k) bool

        M = (self.labels[:, None] == self.unique_labels[None, :])  # (n, k) bool
        union_mask = M[:, :, None] | M[:, None, :]             # (n, k, k) bool

        # Distances from all n points to all k² midpoints in one cdist call.
        midpoints = (cluster_means[:, None, :] + cluster_means[None, :, :]) / 2.0
        dist_mid = cdist(self.data, midpoints.reshape(k * k, p)).reshape(n, k, k)
        in_radius_mid = dist_mid <= stdev                      # (n, k, k) bool

        # density_ci[ci, cj] = count of union(ci,cj) points within stdev of c_ci
        # in_radius_dtm[:, ci] broadcast over cj axis:
        density_ci = (in_radius_dtm[:, :, None] & union_mask).sum(axis=0)   # (k, k)
        density_cj = (in_radius_dtm[:, None, :] & union_mask).sum(axis=0)   # (k, k)
        density_midpoint = (in_radius_mid & union_mask).sum(axis=0)          # (k, k)

        max_density = np.maximum(density_ci, density_cj)                     # (k, k)
        offdiag = np.eye(k, dtype=bool) ^ True
        valid = offdiag & (max_density > 0)
        dens_bw = (
            np.sum(density_midpoint[valid] / max_density[valid]) / (k * (k - 1))
        )

        res = scat + dens_bw

        return res

    @computed_field_cached_property()
    def sd_validity_index(self) -> float:
        """SD validity index (Halkidi, Vazirgiannis & Batistakis, 2000).

        ``SD = α · Scat(k) + Dis(k)``
        Scat = mean ratio of cluster to total variance norm.
        Dis = (D_max/D_min) · Σ_k 1/(Σ_j d(c_k, c_j)), penalising poor
        centroid separation.  α = 1.0 here (in multi-k sweeps α = Dis(k_max)).

        Range [0, ∞). Natural direction: **minimize**.
        """
        if self._is_single_cluster:
            return np.nan

        k = self.label_count
        scat = self._mean_scatter

        if k < 2:
            res = scat
            return res

        # Dis component: separation based on centroid distances
        cluster_means = self._mean[1:]
        centroid_dists = squareform(pdist(cluster_means))

        D_max = np.max(centroid_dists)

        # D_min: minimum non-zero inter-centroid distance
        centroid_dists_no_diag = centroid_dists.copy()
        np.fill_diagonal(centroid_dists_no_diag, np.inf)
        D_min = np.min(centroid_dists_no_diag)

        if D_min < self._eps:
            dis = np.inf
        else:
            # Σ_k (Σ_j ||c_k - c_j||)^{-1}
            row_sums = np.sum(centroid_dists, axis=1)
            row_sums_safe = np.where(row_sums > self._eps, row_sums, self._eps)
            dis = (D_max / D_min) * np.sum(1.0 / row_sums_safe)

        alpha = 1.0
        res = alpha * scat + dis

        return res

    # -------------------------------------------------------------------------
    # Centroid-based indices (O(nK) — no pairwise distances required)
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def simplified_silhouette_index(self) -> float:
        """Simplified Silhouette (Hruschka et al. / Vendramin et al.).

        O(nK) approximation of the silhouette using centroid distances in
        place of all-pairs distances:
        ``a_i = ||x_i − c_own||``, ``b_i = min_{j≠own} ||x_i − c_j||``,
        ``s_i = (b_i − a_i) / max(a_i, b_i)``, mean-aggregated.

        Range [−1, 1]. Natural direction: **maximize** — negated to minimize.
        """
        # _distance_to_mean columns: [0]=grand mean, [1..K]=cluster centroids
        # Column ordering follows self.unique_labels.
        dtm = self._distance_to_mean[:, 1:]  # (n, K)
        k = self.label_count

        if k < 2:
            return np.nan  # single cluster → undefined

        # Build per-point own-centroid column index
        label_to_col = {label: i for i, label in enumerate(self.unique_labels)}
        own_col = np.empty(self.n_total, dtype=np.intp)
        for label, idx in self._label_indices.items():
            own_col[idx] = label_to_col[label]

        a = dtm[np.arange(self.n_total), own_col]  # distance to own centroid

        # Mask own centroid with inf to find nearest *other* centroid
        dtm_masked = dtm.copy()
        dtm_masked[np.arange(self.n_total), own_col] = np.inf
        b = np.min(dtm_masked, axis=1)

        denom = np.maximum(a, b)
        # Use np.divide with where to avoid RuntimeWarning on division
        s = np.divide(b - a, denom, where=denom > self._eps, out=np.zeros_like(denom, dtype=float))
        res = float(np.mean(s))

        res *= -1

        return res

    @computed_field_cached_property()
    def cop_index(self) -> float:
        """COP index (Gurrutxaga et al., 2010).

        Mean ratio of distance to own centroid over distance to nearest other
        centroid: ``COP = (1/n) Σ d(x_i, c_own) / d(x_i, c_nearest_other)``.

        Range [0, ∞). Natural direction: **minimize**.
        """
        dtm = self._distance_to_mean[:, 1:]  # (n, K)
        k = self.label_count

        if k < 2:
            return np.nan  # single cluster → undefined

        label_to_col = {label: i for i, label in enumerate(self.unique_labels)}
        own_col = np.empty(self.n_total, dtype=np.intp)
        for label, idx in self._label_indices.items():
            own_col[idx] = label_to_col[label]

        a = dtm[np.arange(self.n_total), own_col]

        dtm_masked = dtm.copy()
        dtm_masked[np.arange(self.n_total), own_col] = np.inf
        b = np.min(dtm_masked, axis=1)

        ratio = np.where(a < self._eps, 0.0, np.inf)          # fallback when b ≈ 0
        np.divide(a, b, out=ratio, where=b > self._eps)       # only divide where safe
        res = float(np.mean(ratio))

        return res

    @computed_field_cached_property()
    def negentropy_index(self) -> float:
        """Negentropy index (Hyvärinen approximation, 1998).

        Measures within-cluster departure from Gaussianity:
        ``J_k ≈ [E{G(z)} − E{G(v)}]²``, ``G(u) = −exp(−u²/2)``,
        aggregated as ``Σ_k (n_k/n) J_k``.
        Projects each cluster onto its leading PC via power iteration (single
        pass); falls back to full SVD for small clusters (n_k ≤ p).
        Well-partitioned Gaussian clusters score near zero.

        Range [0, ∞). Natural direction: **minimize**.
        """
        if self._is_single_cluster:
            return np.nan
        n = self.n_total
        k = self.label_count
        p = self.data.shape[1]

        # E{G(v)} for a standard normal: E{-exp(-v²/2)} = -1/√2
        eg_normal = -1.0 / np.sqrt(2.0)

        total_negentropy = 0.0
        for i, label in enumerate(self.unique_labels):
            idx = self._label_indices[label]
            n_k = len(idx)
            if n_k < 3:
                # Too few points to estimate negentropy meaningfully
                continue

            cluster_data = self.data[idx]
            mean_k = self._mean[i + 1]
            centered = cluster_data - mean_k

            # Project onto leading PC for a scalar negentropy estimate
            if p == 1:
                proj = centered.ravel()
            else:
                # Fast rank-1 SVD via power method (one iteration suffices
                # for a rough direction — we only need the dominant axis).
                # Falls back to full SVD only if the data is tiny.
                if n_k <= p:
                    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                    proj = centered @ Vt[0]
                else:
                    # Covariance-free: C v = (X^T X) v / (n-1); one matmul pass
                    rng = np.random.RandomState(self.seed)
                    v = rng.randn(p).astype(centered.dtype)
                    v /= np.linalg.norm(v) + self._eps
                    v = centered.T @ (centered @ v)
                    v /= np.linalg.norm(v) + self._eps
                    proj = centered @ v

            # Standardise to unit variance
            std = np.std(proj)
            if std < self._eps:
                continue
            z = proj / std

            # Negentropy approximation
            eg_z = np.mean(-np.exp(-0.5 * z * z))
            j_k = (eg_z - eg_normal) ** 2

            total_negentropy += (n_k / n) * j_k

        res = float(total_negentropy)

        return res

    @computed_field_cached_property()
    def wb_index(self) -> float:
        """WB index (Zhao et al., 2009).

        ``WB(k) = k · WCSS / BCSS``

        Penalises cluster proliferation that does not proportionally improve
        between-cluster separation. Reuses cached ``_WCSS`` and ``_BCSS``.

        Range [0, ∞). Natural direction: **minimize**.
        """
        if self._is_single_cluster:
            return np.nan
        k = self.label_count
        bcss = self._BCSS
        if bcss < self._eps:
            return np.inf
        val = k * self._WCSS / bcss
        return val

    @computed_field_cached_property()
    def generalized_dunn_index(self) -> float:
        """Generalized Dunn index (Bezdek & Pal, 1998).

        Centroid-based reformulation of the Dunn index, immune to the
        single-outlier fragility of the standard min/max formulation:
        ``GD = min_{i≠j}(||c_i − c_j||) / max_k(mean ||x − c_k||)``

        Range [0, ∞). Natural direction: **maximize** — negated to minimize.
        """
        k = self.label_count
        if k < 2:
            return np.nan  # single cluster → undefined

        # Inter-cluster: min centroid-to-centroid distance
        # _mean[0] is global mean; _mean[1:] are cluster centroids
        centroids = self._mean[1:]  # (K, p)
        min_inter = np.inf
        for i in range(k - 1):
            diffs = centroids[i + 1:] - centroids[i]
            dists = np.sqrt(np.sum(diffs ** 2, axis=1))
            min_inter = min(min_inter, np.min(dists))

        # Intra-cluster: max over clusters of mean distance to own centroid
        # _distance_to_mean[:, i+1] = distance of each point to centroid i
        dtm = self._distance_to_mean[:, 1:]  # (n, K)
        max_intra = 0.0
        for i, label in enumerate(self.unique_labels):
            idx = self._label_indices[label]
            mean_dist = np.mean(dtm[idx, i])
            max_intra = max(max_intra, mean_dist)

        if max_intra < self._eps:
            res = np.inf
        else:
            res = min_inter / max_intra

        res *= -1

        return res

    # -------------------------------------------------------------------------
    # Density-based indices (continued)
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def density_based_clustering_validation_index(self) -> float:
        """Density-Based Clustering Validation (DBCV) index (Moulavi et al., 2014).

        Measures cluster quality using mutual reachability distances and the
        minimum spanning tree of each cluster. Designed for arbitrary-shape
        and density-varying clusters; handles noise (-1) labels natively.

        Range [−1, 1]. Natural direction: **maximize** — negated to minimize.
        Requires n ≥ 50; expensive for large n (MST construction).
        """
        if self._is_single_cluster:
            return np.nan
        # DBCV requires at least 2 points per cluster for core distance
        # computation. If any cluster is a singleton, abstain.
        if any(len(self._label_indices[lbl]) < 2 for lbl in self.unique_labels):
            return np.nan

        cluster_members = [self._label_indices[lbl] for lbl in self.unique_labels]
        cluster_sizes = np.array([m.size for m in cluster_members], dtype=np.intp)

        return -dbcv_prevalidated(
            n_total=self.n_total,
            n_features=self.data.shape[1],
            cluster_sizes=cluster_sizes,
            cluster_members=cluster_members,
            precomputed_distances=self._distance,
        )