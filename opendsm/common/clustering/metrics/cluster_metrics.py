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

import pydantic
from typing import Optional, Literal
from enum import Enum

import numpy as np

from scipy.spatial.distance import cdist, pdist, squareform

from opendsm.common.stats.basic import median_absolute_deviation
from opendsm.common.pydantic_utils import (
    ArbitraryPydanticModel,
    computed_field_cached_property,
)
from opendsm.common.clustering.metrics.density_based_clustering_validation import dbcv



class DistanceMetric(str, Enum):
    """
    what distance method to use
    """
    EUCLIDEAN = "euclidean"
    STANDARDIZED_EUCLIDEAN = "seuclidean"
    SQUARED_EUCLIDEAN = "sqeuclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class ClusterPairDistanceMetrics(ArbitraryPydanticModel):
    """
    Metrics between clusters
    """

    cluster_ids: Optional[tuple[int, int]] = pydantic.Field(
        default=None,
        description="The two clusters to compare"
    )

    distance: np.array = pydantic.Field(
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
        return np.min(self.distance[self.distance > 0])
    
    @computed_field_cached_property()
    def max(self) -> float:
        return np.max(self.distance)


class SingleClusterMetrics(ArbitraryPydanticModel):
    """
    Metrics within a single cluster
    """
    cluster_id: int | None = pydantic.Field(
        default=None,
    )

    n: int = pydantic.Field()

    mean: np.array = pydantic.Field()

    median: np.array = pydantic.Field()

    var: Optional[np.array] = pydantic.Field(
        default=None,
    )

    distance: dict[int, ClusterPairDistanceMetrics] | ClusterPairDistanceMetrics = pydantic.Field()

    distance_to_mean: dict[int | str, ClusterPairDistanceMetrics] | ClusterPairDistanceMetrics = pydantic.Field()

    distance_to_median: dict[int | str, ClusterPairDistanceMetrics] | ClusterPairDistanceMetrics = pydantic.Field()

    mean_distance_intra_cluster: Optional[np.array] = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
    )

    median_distance_intra_cluster: Optional[np.array] = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
    )

    mean_distance_to_nearest_cluster: Optional[np.array] = pydantic.Field(
        default=None,
        exclude=True,
        repr=False,
    )

    median_distance_to_nearest_cluster: Optional[np.array] = pydantic.Field(
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
    def mean_silhouette_coefficient(self) -> np.array:
        if self.mean_distance_intra_cluster is None:
            return None

        a = self.mean_distance_intra_cluster
        b = self.mean_distance_to_nearest_cluster
        
        return (b - a) / np.maximum(a, b)

    @computed_field_cached_property()
    def median_silhouette_coefficient(self) -> np.array:
        if self.median_distance_intra_cluster is None:
            return None

        a = self.median_distance_intra_cluster
        b = self.median_distance_to_nearest_cluster

        return (b - a) / np.maximum(a, b)


class ClusterMetrics(ArbitraryPydanticModel):
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

    index_direction: Literal["minimize", "maximize"] = pydantic.Field(
        default="minimize",
        description="Force the indice direction to `minimize` or `maximize` as best",
    )

    _eps: float = 1e-10
    _all: int = -999

    @pydantic.model_validator(mode='after')
    def _validate_data(self) -> 'ClusterMetrics':
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

    @computed_field_cached_property()
    def n_total(self) -> int:
        return self.data.shape[0]

    @computed_field_cached_property()
    def unique_labels(self) -> np.array:
        return np.unique(self.labels)

    @computed_field_cached_property()
    def label_count(self) -> int:
        return len(self.unique_labels)

    @computed_field_cached_property()
    def _label_indices(self) -> dict[int, np.ndarray]:
        return {label: np.where(self.labels == label)[0]
                for label in self.unique_labels}

    @computed_field_cached_property()
    def _n(self) -> np.array:
        cluster_sizes = [len(self._label_indices[label]) for label in self.unique_labels]
        return np.array([self.n_total, *cluster_sizes])

    @computed_field_cached_property()
    def _mean(self) -> np.array:
        means = [np.mean(self.data, axis=0)]
        for label in self.unique_labels:
            means.append(np.mean(self.data[self._label_indices[label]], axis=0))

        return np.array(means)

    @computed_field_cached_property()
    def _median(self) -> np.array:
        medians = [np.median(self.data, axis=0)]
        for label in self.unique_labels:
            medians.append(np.median(self.data[self._label_indices[label]], axis=0))

        return np.array(medians)

    @computed_field_cached_property()
    def _var(self) -> np.array:
        variances = [np.var(self.data, axis=0)]
        for label in self.unique_labels:
            variances.append(np.var(self.data[self._label_indices[label]], axis=0))

        return np.array(variances)

    @computed_field_cached_property()
    def _distance(self) -> np.array:
        return squareform(pdist(self.data))

    @computed_field_cached_property()
    def _distance_to_mean(self) -> np.array:
        return cdist(self.data, self._mean)

    @computed_field_cached_property()
    def _distance_to_median(self) -> np.array:
        return cdist(self.data, self._median)
    
    @computed_field_cached_property()
    def _labeled_distance(self) -> dict[tuple[int, int], np.array]:
        data = {}
        for label_i in self.unique_labels:
            idx_i = self._label_indices[label_i]

            for label_j in self.unique_labels:
                idx_j = self._label_indices[label_j]
                data[label_i, label_j] = self._distance[np.ix_(idx_i, idx_j)]

        return data

    def _labeled_distance_to_centroid(self, distance_matrix: np.array) -> dict[tuple[int, int], np.array]:
        unique_labels = [self._all, *self.unique_labels]
        all_idx = np.arange(self.n_total)

        data = {}
        for label_i in unique_labels:
            idx_i = all_idx if label_i == self._all else self._label_indices[label_i]

            for col_idx, label_j in enumerate(unique_labels):
                data[label_i, label_j] = distance_matrix[idx_i, col_idx]

        return data

    @computed_field_cached_property()
    def _labeled_distance_to_mean(self) -> dict[tuple[int, int], np.array]:
        return self._labeled_distance_to_centroid(self._distance_to_mean)

    @computed_field_cached_property()
    def _labeled_distance_to_median(self) -> dict[tuple[int, int], np.array]:
        return self._labeled_distance_to_centroid(self._distance_to_median)

    def _labeled_distance_to_nearest_cluster(self, agg: str = "mean") -> dict[int, np.array]:
        agg_fcn = np.mean if agg == "mean" else np.median

        data = {}
        for label_i in self.unique_labels:
            n = self._labeled_distance[label_i, label_i].shape[0]
            dist_to_nearest = np.full(n, np.inf)

            for label_j in self.unique_labels:
                if label_i == label_j:
                    continue

                dist_matrix = self._labeled_distance[label_i, label_j]
                avg_dists = agg_fcn(dist_matrix, axis=1)
                dist_to_nearest = np.minimum(dist_to_nearest, avg_dists)

            data[label_i] = dist_to_nearest

        return data
    
    @computed_field_cached_property()
    def _labeled_mean_distance_to_nearest_cluster(self) -> dict[int, np.array]:
        return self._labeled_distance_to_nearest_cluster(agg="mean")
    
    @computed_field_cached_property()
    def _labeled_median_distance_to_nearest_cluster(self) -> dict[int, np.array]:
        return self._labeled_distance_to_nearest_cluster(agg="median")

    def _labeled_distance_intra_cluster(self, agg: str = "mean") -> dict[int, np.array]:
        data = {}
        for label_i in self.unique_labels:
            distance_array = self._labeled_distance[label_i, label_i]
            n = distance_array.shape[0]

            if agg == "mean":
                # Mean excluding self: row_sum / (n-1), diagonal is 0
                data[label_i] = np.sum(distance_array, axis=1) / (n - 1)
            else:
                # Median excluding self: mask diagonal with nan, use nanmedian
                masked = distance_array.copy()
                np.fill_diagonal(masked, np.nan)
                data[label_i] = np.nanmedian(masked, axis=1)

        return data
    
    @computed_field_cached_property()
    def _labeled_mean_distance_intra_cluster(self) -> dict[int, np.array]:
        return self._labeled_distance_intra_cluster(agg="mean")
    
    @computed_field_cached_property()
    def _labeled_median_distance_intra_cluster(self) -> dict[int, np.array]:
        return self._labeled_distance_intra_cluster(agg="median")
        
    @computed_field_cached_property()
    def all(self) -> SingleClusterMetrics:
        key = (self._all, self._all)

        distance = ClusterPairDistanceMetrics(
            distance=self._distance,
        )

        distance_to_mean = ClusterPairDistanceMetrics(
            distance=self._labeled_distance_to_mean[key],
        )
        
        distance_to_median = ClusterPairDistanceMetrics(
            distance=self._labeled_distance_to_median[key],
        )

        return SingleClusterMetrics(
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
    def cluster(self) -> dict[int, SingleClusterMetrics]:
        data = {}
        for i, label in enumerate(self.unique_labels):
            # single cluster metrics
            n = self._n[i + 1]
            mean = self._mean[i + 1]
            median = self._median[i + 1]
            var = self._var[i + 1]

            # pair distance metrics
            distance = {}
            distance_to_mean = {"all": ClusterPairDistanceMetrics(
                cluster_ids=(label, self._all),
                distance=self._labeled_distance_to_mean[(label, self._all)],
            )}
            distance_to_median = {"all": ClusterPairDistanceMetrics(
                cluster_ids=(label, self._all),
                distance=self._labeled_distance_to_median[(label, self._all)],
            )}

            for label_j in self.unique_labels:
                key = (label, label_j)

                distance[label_j] = ClusterPairDistanceMetrics(
                    cluster_ids=key,
                    distance=self._labeled_distance[key],
                )

                distance_to_mean[label_j] = ClusterPairDistanceMetrics(
                    cluster_ids=key,
                    distance=self._labeled_distance_to_mean[key],
                )

                distance_to_median[label_j] = ClusterPairDistanceMetrics(
                    cluster_ids=key,
                    distance=self._labeled_distance_to_median[key],
                )

            mean_distance_intra_cluster = self._labeled_mean_distance_intra_cluster[label]
            median_distance_intra_cluster = self._labeled_median_distance_intra_cluster[label]
            mean_distance_to_nearest_cluster = self._labeled_mean_distance_to_nearest_cluster[label]
            median_distance_to_nearest_cluster = self._labeled_median_distance_to_nearest_cluster[label]
            
            data[label] = SingleClusterMetrics(
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
            cluster_data = self.data[self._label_indices[label]]
            cluster_mean = self._mean[i + 1]

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
        centered_data = self.data - self._mean[0]
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
    def _WC_pairwise_distances(self) -> np.ndarray:
        """Aggregated within-cluster pairwise distances across all clusters"""
        parts = [
            c.within_pairwise_distances
            for c in self.cluster.values()
            if c.within_pairwise_distances is not None and len(c.within_pairwise_distances) > 0
        ]
        return np.concatenate(parts) if parts else np.array([])

    @computed_field_cached_property()
    def _BC_pairwise_distances(self) -> np.ndarray:
        """Aggregated between-cluster pairwise distances (deduplicated across cluster pairs)"""
        parts = []
        labels = list(self.unique_labels)
        for i, label_i in enumerate(labels):
            for label_j in labels[i+1:]:
                parts.append(self.cluster[label_i].distance[label_j].distance.ravel())
        return np.concatenate(parts) if parts else np.array([])

    @computed_field_cached_property()
    def _mean_scatter(self) -> float:
        """Average scattering: (1/K) × Σ_k ||σ(C_k)|| / ||σ(D)||"""
        total_var_norm = self.all.var_norm
        if total_var_norm is None or total_var_norm < self._eps:
            return 0.0

        cluster_var_norms = np.array([
            self.cluster[label].var_norm
            for label in self.unique_labels
        ])
        return np.mean(cluster_var_norms) / total_var_norm

    # -------------------------------------------------------------------------
    # Compactness indices (within-cluster quality only)
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def sum_of_squared_errors_index(self) -> float:
        # Sum of Squared Errors (SSE) Index
        # Range is 0 to inf, 0 is the best
        # Formula: SSE = Σ_k Σ_{x_i ∈ C_k} ||x_i - c_k||²
        # This is equivalent to the Within-Cluster Sum of Squares (WCSS)

        # Within Cluster Sum of Squares (WCSS) = SSE
        res = self._WCSS

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def mean_squared_error_index(self) -> float:
        # Mean Squared Error (MSE) Index
        # Range is 0 to inf, 0 is the best
        # Formula: MSE = SSE / n = WCSS / n
        # where SSE = sum of squared errors,
        #       n = total number of data points

        n = self.n_total  # number of data points
        WCSS = self._WCSS

        res = WCSS / n

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def ball_hall_index(self) -> float:
        # Ball and Hall Index
        # Range is 0 to inf, 0 is the best
        # Formula: (1/K) * Σ(sum of squared distances from points to cluster centroids)

        k = self.label_count  # number of clusters
        WCSS = self._WCSS  # Within Cluster Sum of Squares (WCSS)
        res = WCSS / k

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def banfeld_raftery_index(self) -> float:
        # Banfeld-Raftery Index
        # Range is -inf to inf, -inf is the best
        # Formula: Σ [n_k × log(trace(W_k) / n_k)]
        # where n_k = number of points in cluster k,
        #       trace(W_k) = sum of squared distances to centroid for cluster k

        n_k = self._n[1:]  # cluster sizes
        traces = np.array([np.trace(self._WCSM[label]) for label in self.unique_labels])

        # Replace zero traces with _eps to avoid log(0)
        traces_safe = np.where(traces > 0, traces, self._eps)
        res = np.sum(n_k * np.log(traces_safe / n_k))

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def scott_symons_index(self) -> float:
        # Scott-Symons Index
        # Range is -inf to inf, -inf is the best (minimize)
        # Formula: Σ n_k × log(det(W_k / n_k))
        # where W_k = scatter matrix for cluster k,
        #       n_k = number of points in cluster k

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

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def trace_w_index(self) -> float:
        # Trace W Index
        # Range is 0 to inf, 0 is the best (minimize)
        # Formula: trace(W)
        # where W = pooled within-cluster scatter matrix
        # Equivalent to WCSS but formalized as a matrix trace measure

        res = np.trace(self._sum_WCSM)

        if self.index_direction == "maximize":
            res *= -1

        return res

    # -------------------------------------------------------------------------
    # Compactness + Separation indices (combined)
    # -------------------------------------------------------------------------

    def _silhouette_coefficients(self, variant: str = "mean") -> np.ndarray:
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
            coefs = (b - a) / np.maximum(a, b)
            n_points = len(coefs)
            coefficients[idx:idx+n_points] = coefs
            idx += n_points

        return coefficients

    @computed_field_cached_property()
    def silhouette_index(self) -> float:
        # range is -1 to 1, 1 is the best
        res = np.mean(self._silhouette_coefficients("mean"))

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def silhouette_median_index(self) -> float:
        # range is -1 to 1, 1 is the best
        res = np.median(self._silhouette_coefficients("median"))

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def davies_bouldin_index(self) -> float:
        # range is 0 to inf, 0 is the best
        k = self.label_count
        # Could abstract with scipy.stats.moment

        intracluster_distance = np.empty(k)
        for i, label in enumerate(self.unique_labels):
            intracluster_distance[i] = np.mean(self._labeled_distance_to_mean[(label, label)])

        intercluster_distance = squareform(pdist(self._mean[1:]))

        # Compute similarity matrix using broadcasting
        # similarity[i,j] = (dist_i + dist_j) / dist_ij for i != j
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = (intracluster_distance[:, None] + intracluster_distance[None, :]) / intercluster_distance

        # Set diagonal to 0 (i == j case)
        np.fill_diagonal(similarity, 0)

        res = np.sum(np.max(similarity, axis=1)) / k

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def calinski_harabasz_index(self) -> float:
        # range is 0 to inf, inf is the best
        k = self.label_count # number of clusters
        n = self.n_total # number of data points

        BCSS = self._BCSS  # Between Cluster Sum of Squares (BCSS)
        WCSS = self._WCSS  # Within Cluster Sum of Squares (WCSS)

        if WCSS < self._eps:
            return 1.0

        res = (BCSS / WCSS) * ((n - k) / (k - 1.0))

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def variance_ratio_criterion(self) -> float:
        return self.calinski_harabasz_index

    @computed_field_cached_property()
    def dunn_index(self) -> float:
        # Dunn Index
        # Range is 0 to inf, inf is the best
        # Formula: min(inter-cluster distance) / max(intra-cluster diameter)
        # where inter-cluster distance = min distance between points in different clusters
        #       intra-cluster diameter = max distance between points in same cluster

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

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def xie_beni_index(self) -> float:
        # Xie-Beni Index
        # Range is 0 to inf, 0 is the best
        # Formula: WCSS / (n × d_min²)
        # where WCSS = within-cluster sum of squares,
        #       n = number of data points,
        #       d_min = minimum distance between cluster centroids

        n = self.n_total  # number of data points
        cluster_means = self._mean[1:] # Skip the overall mean at index 0

        # define numerator
        use_assigned_cluster_centroids = True

        if use_assigned_cluster_centroids:
            num = self._WCSS
        else: # uses nearest centroid instead
            # Compute WGSS: sum of squared distances from each point to its nearest centroid
            # This matches the standard Xie-Beni definition
            d_sq_to_centroids = cdist(
                self.data,
                cluster_means,
                metric='sqeuclidean'
            )
            min_d_sq_to_centroids = np.min(d_sq_to_centroids, axis=1)
            num = np.sum(min_d_sq_to_centroids)

        # Calculate squared pairwise distances between centroids
        if len(cluster_means) > 1:
            d_sq = pdist(
                cluster_means,
                metric='sqeuclidean'
            )
            d_min_squared = np.min(d_sq)
        else:
            # If only one cluster, return infinity (worst score)
            return np.inf if self.index_direction == "minimize" else -np.inf

        # Avoid division by zero
        if d_min_squared < self._eps:
            res = np.inf
        else:
            res = num / (n * d_min_squared)

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def duda_hart_index(self) -> float:
        # Duda and Hart Index
        # Range is 0 to inf, 0 is the best

        intracluster_distance = 0
        intercluster_distance = 0
        for label_i in self.unique_labels:
            intracluster_distance += np.mean(self._labeled_distance_to_mean[(label_i, label_i)])

            # Mean distance from points in label_i to all points NOT in label_i
            inter_dists = [self._labeled_distance[label_i, label_j]
                           for label_j in self.unique_labels if label_j != label_i]
            intercluster_distance += np.mean(np.hstack(inter_dists))

        res = intracluster_distance / intercluster_distance

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def c_index(self) -> float:
        # C-Index
        # Range is 0 to 1, 0 is the best
        # Formula: (S_w - S_min) / (S_max - S_min)
        # where S_w = sum of within-cluster pairwise distances,
        #       S_min = sum of the N_w smallest pairwise distances overall,
        #       S_max = sum of the N_w largest pairwise distances overall,
        #       N_w = number of within-cluster pairs

        within_dists = self._WC_pairwise_distances
        n_w = len(within_dists)

        if n_w == 0:
            return 0.0

        S_w = np.sum(within_dists)

        all_dists = self._distance[np.triu_indices(self.n_total, k=1)]
        all_dists_sorted = np.sort(all_dists)

        S_min = np.sum(all_dists_sorted[:n_w])
        S_max = np.sum(all_dists_sorted[-n_w:])

        denom = S_max - S_min
        if denom < self._eps:
            res = 0.0
        else:
            res = (S_w - S_min) / denom

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def mcclain_rao_index(self) -> float:
        # McClain-Rao Index
        # Range is 0 to inf, 0 is the best
        # Formula: mean(within-cluster distances) / mean(between-cluster distances)

        within_dists = self._WC_pairwise_distances
        between_dists = self._BC_pairwise_distances

        if len(within_dists) == 0 or len(between_dists) == 0:
            return np.inf

        mean_within = np.mean(within_dists)
        mean_between = np.mean(between_dists)

        if mean_between < self._eps:
            res = np.inf
        else:
            res = mean_within / mean_between

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def i_index(self) -> float:
        # I-Index (Maulik-Bandyopadhyay)
        # Range is 0 to inf, inf is the best
        # Formula: I(K) = (1/K × E_1/E_K × D_K)^p
        # where E_1 = Σ ||x_i - grand_centroid|| (total distance to grand mean),
        #       E_K = Σ_k Σ_{x ∈ C_k} ||x - c_k|| (total distance to cluster centroids),
        #       D_K = max inter-centroid distance,
        #       p = 2

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
            return 0.0

        if E_K < self._eps:
            res = np.inf
        else:
            res = ((1.0 / k) * (E_1 / E_K) * D_K) ** p

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def log_ss_ratio_index(self) -> float:
        # Log SS Ratio Index (Log Sum of Squares Ratio)
        # Range is -inf to inf, inf is the best
        # Formula: log(BCSS / WCSS) = log(BCSS) - log(WCSS)
        # where BCSS = between-cluster sum of squares,
        #       WCSS = within-cluster sum of squares

        BCSS = self._BCSS  # Between Cluster Sum of Squares (BCSS)
        WCSS = self._WCSS  # Within Cluster Sum of Squares (WCSS)

        # Avoid log of zero or division by zero
        if WCSS < self._eps:
            res = np.inf
        elif BCSS < self._eps:
            res = -np.inf
        else:
            # log(BCSS / WCSS) = log(BCSS) - log(WCSS)
            res = np.log(BCSS) - np.log(WCSS)

        if self.index_direction == "minimize":
            res *= -1

        return res

    # -------------------------------------------------------------------------
    # Statistical / Correlation indices
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def gamma_index(self) -> float:
        # Hubert's Gamma Index
        # Range is -1 to 1, 1 is the best
        # Concordance measure: Γ = (s+ - s-) / (s+ + s-)
        # where s+ = concordant pairs (within < between),
        #       s- = discordant pairs (within > between)

        within_dists = self._WC_pairwise_distances
        between_dists = self._BC_pairwise_distances

        n_b = len(between_dists)
        if n_b == 0 or len(within_dists) == 0:
            return 0.0

        between_sorted = np.sort(between_dists)

        # For each within_dist, count concordant (between > within) and discordant (between < within)
        left_indices = np.searchsorted(between_sorted, within_dists, side='left')
        right_indices = np.searchsorted(between_sorted, within_dists, side='right')

        s_plus = np.sum(n_b - right_indices)   # between > within (concordant)
        s_minus = np.sum(left_indices)          # between < within (discordant)

        denom = s_plus + s_minus
        if denom == 0:
            res = 0.0
        else:
            res = float(s_plus - s_minus) / float(denom)

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def point_biserial_index(self) -> float:
        # Point-Biserial Correlation
        # Range is -1 to 1, 1 is the best
        # Formula: r_pb = (M_b - M_w) / s_d × sqrt(n_w × n_b / n_t²)
        # where M_b = mean between-cluster distance,
        #       M_w = mean within-cluster distance,
        #       s_d = std of all pairwise distances,
        #       n_w, n_b = number of within/between pairs

        within_dists = self._WC_pairwise_distances
        between_dists = self._BC_pairwise_distances

        n_w = len(within_dists)
        n_b = len(between_dists)
        n_t = n_w + n_b

        if n_w == 0 or n_b == 0:
            return 0.0

        mean_within = np.mean(within_dists)
        mean_between = np.mean(between_dists)

        all_dists = np.concatenate([within_dists, between_dists])
        std_all = np.std(all_dists)

        if std_all < self._eps:
            return 0.0

        res = ((mean_between - mean_within) / std_all) * np.sqrt(n_w * n_b / (n_t ** 2))

        if self.index_direction == "minimize":
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
            return np.inf

    @computed_field_cached_property()
    def ksq_detw_index(self) -> float:
        # KSq-DetW Index (K² × det(W))
        # Range is -inf to inf, inf is the best
        # Formula: K² × det(W)
        # where K = number of clusters,
        #       W = summed within-cluster scatter matrix (normalized by default),
        #       det(W) = determinant of W

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
            # Handle singular matrix
            det_W = 0

        # KSq-DetW = K² × det(W)
        res = (k ** 2) * det_W

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def det_ratio_index(self) -> float:
        # Det Ratio Index
        # Range is 0 to inf, inf is the best
        # Formula: det(T) / det(W)
        # where T = total scatter matrix (covariance of all data),
        #       W = summed within-cluster scatter matrix

        res = self._det_ratio

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def log_det_ratio_index(self) -> float:
        # Log Det Ratio Index
        # Range is -inf to inf, inf is the best
        # Formula: n * log(det(T) / det(W)) = n * (log(det(T)) - log(det(W)))
        # where T = total scatter matrix,
        #       W = summed within-cluster scatter matrix,
        #       n = number of data points

        n = self.n_total
        res = n * np.log(np.abs(self._det_ratio))

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def trace_wb_index(self) -> float:
        # Trace WB Index (Trace of W^-1 × B)
        # Range is 0 to inf, inf is the best (maximize)
        # Formula: trace(W^-1 × B) where B = T - W
        # Multivariate generalization of Calinski-Harabasz

        W = self._sum_WCSM
        B = self._TSM - W

        try:
            W_inv = np.linalg.inv(W)
        except np.linalg.LinAlgError:
            W_inv = np.linalg.pinv(W)

        res = np.trace(W_inv @ B)

        if self.index_direction == "minimize":
            res *= -1

        return res

    # -------------------------------------------------------------------------
    # Density-based indices
    # -------------------------------------------------------------------------

    @computed_field_cached_property()
    def s_dbw_index(self) -> float:
        # S_Dbw Index (Halkidi and Vazirgiannis, 2001)
        # Range is 0 to inf, 0 is the best (minimize)
        # Formula: S_Dbw = Scat + Dens_bw
        # Scat = average scattering (cluster variance / total variance)
        # Dens_bw = average inter-cluster density at midpoints between centroids

        k = self.label_count
        if k < 2:
            return self._mean_scatter

        scat = self._mean_scatter

        # stdev: average norm of cluster variance vectors (neighborhood radius)
        cluster_var_norms = np.array([
            self.cluster[label].var_norm
            for label in self.unique_labels
        ])
        stdev = np.mean(cluster_var_norms)

        if stdev < self._eps:
            # All clusters are single points; no inter-cluster density
            res = scat
            if self.index_direction == "maximize":
                res *= -1
            return res

        cluster_means = self._mean[1:]

        # Compute Dens_bw: for each pair (i, j), evaluate density at midpoint
        # relative to density at the denser centroid
        dens_bw_sum = 0.0
        for i in range(k):
            label_i = self.unique_labels[i]
            idx_i = self._label_indices[label_i]

            for j in range(k):
                if i == j:
                    continue

                label_j = self.unique_labels[j]
                idx_j = self._label_indices[label_j]

                # Union of points in clusters i and j
                union_idx = np.concatenate([idx_i, idx_j])
                union_data = self.data[union_idx]

                # Midpoint between centroids
                u_ij = (cluster_means[i] + cluster_means[j]) / 2.0

                # Count points within stdev of each reference point
                density_midpoint = np.sum(
                    np.linalg.norm(union_data - u_ij, axis=1) <= stdev
                )
                density_ci = np.sum(
                    np.linalg.norm(union_data - cluster_means[i], axis=1) <= stdev
                )
                density_cj = np.sum(
                    np.linalg.norm(union_data - cluster_means[j], axis=1) <= stdev
                )

                max_density = max(density_ci, density_cj)
                if max_density > 0:
                    dens_bw_sum += density_midpoint / max_density

        dens_bw = dens_bw_sum / (k * (k - 1))

        res = scat + dens_bw

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def sd_validity_index(self) -> float:
        # SD Validity Index (Halkidi, Vazirgiannis, Batistakis, 2000)
        # Range is 0 to inf, 0 is the best (minimize)
        # Formula: SD = α × Scat(K) + Dis(K)
        # Scat = (1/K) × Σ_k ||σ(C_k)|| / ||σ(D)||
        # Dis = (D_max/D_min) × Σ_k (Σ_j ||c_k - c_j||)^{-1}
        # α = 1.0 (default; in multi-K sweeps this is set to Dis(K_max))

        k = self.label_count
        scat = self._mean_scatter

        if k < 2:
            res = scat
            if self.index_direction == "maximize":
                res *= -1
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

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def density_based_clustering_validation_index(self) -> float:
        # Density-Based Clustering Validation Index
        # https://epubs.siam.org/doi/pdf/10.1137/1.9781611973440.96
        # Metric is between -1 and 1, 1 is the best

        precomputed_distances = self._distance
        # if self.distance_metric == DistanceMetric.EUCLIDEAN:
        #     precomputed_distances = np.power(precomputed_distances, 2)

        dbcvi = dbcv(
            X = self.data,
            y = self.labels,
            precomputed_distances = precomputed_distances,
            metric = self.distance_metric,
            noise_id = -1, # what label is the noise index
            check_duplicates = False,
            n_processes = 1,
            enable_dynamic_precision = False,
            bits_of_precision = 512,
            use_original_mst_implementation = False
        )

        if self.index_direction == "minimize":
            dbcvi *= -1

        return dbcvi