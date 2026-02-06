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

import sys
import pydantic
from typing import Union, Optional, Literal
from enum import Enum

import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist, pdist, squareform

from functools import cached_property

from opendsm.common.stats.basic import median_absolute_deviation
from opendsm.common.pydantic_utils import (
    ArbitraryPydanticModel,
    PydanticDf,
    PydanticFromDict,
    computed_field_cached_property,
)
from opendsm.common.clustering.metrics.density_based_clustering_validation import dbcv

# TODO Delete this import
from permetrics import ClusteringMetric



def get_max_score_from_system_size() -> float:
    """
    Get the max score from the system size.
    Running as function so unforeseen issues are less likely when running on 
    distributed env.
    """

    return sys.float_info.max**0.5


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
    def mean_silhouette_coefficient(self) -> np.array:
        if self.mean_distance_intra_cluster is None: 
            return None

        silhouette = np.empty(self.n)
        for idx in range(self.n):
            a = self.mean_distance_intra_cluster[idx]
            b = self.mean_distance_to_nearest_cluster[idx]

            silhouette[idx] = (b - a) / np.max([b, a])

        return silhouette

    @computed_field_cached_property()
    def median_silhouette_coefficient(self) -> np.array:
        if self.median_distance_intra_cluster is None:
            return None

        silhouette = np.empty(self.n)
        for idx in range(self.n):
            a = self.median_distance_intra_cluster[idx]
            b = self.median_distance_to_nearest_cluster[idx]
            
            silhouette[idx] = (b - a) / np.max([b, a])

        return silhouette


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

    _min_denominator: float = 1e-3

    _all: int = -999

    @cached_property
    def _df(self) -> pd.DataFrame:
        if self.data.shape[0] == 0:
            raise ValueError("Data must have at least one row")
        
        if self.labels.shape[0] == 0:
            raise ValueError("Labels must have at least one row")
        
        if self.labels.shape[0] != self.data.shape[0]:
            raise ValueError("Labels and data must have the same length")
        
        _df = pd.DataFrame(self.data)
        _df["labels"] = self.labels

        _df = _df.reset_index().set_index(["index", "labels"])

        label_min = self.labels.min()

        # this should never trigger, but just in case
        if label_min < self._all:
            len_label_min = len(str(abs(int(label_min))))
            self._all = -int('9' * len_label_min)

            # and just in case in case
            if self._all == label_min:
                self._all = -int('9' * (len_label_min + 1))

        return _df

    @computed_field_cached_property()
    def n_total(self) -> float:
        return len(self._df)

    @computed_field_cached_property()
    def unique_labels(self) -> np.array:
        return np.unique(self.labels)

    @computed_field_cached_property()
    def label_count(self) -> int:
        return len(self.unique_labels)

    @computed_field_cached_property()
    def _n(self) -> np.array:
        n = [self.n_total]
        for label in self.unique_labels:
            n.append(len(self._df.xs(label, level="labels")))

        return np.array(n)

    @computed_field_cached_property()
    def _mean(self) -> np.array:
        means = [np.mean(self._df.values, axis=0)]
        for label in self.unique_labels:
            cluster_data = self._df.xs(label, level="labels").values
            means.append(np.mean(cluster_data, axis=0))

        return np.array(means)

    @computed_field_cached_property()
    def _median(self) -> np.array:
        medians = [np.median(self._df.values, axis=0)]
        for label in self.unique_labels:
            cluster_data = self._df.xs(label, level="labels").values
            medians.append(np.median(cluster_data, axis=0))

        return np.array(medians)
    
    @computed_field_cached_property()
    def _distance(self) -> np.array:
        # return cdist(self._df.values, self._df.values)
        return squareform(pdist(self._df.values))
    
    @computed_field_cached_property()
    def _distance_to_mean(self) -> np.array:
        return cdist(self._df.values, self._mean)
    
    @computed_field_cached_property()
    def _distance_to_median(self) -> np.array:
        return cdist(self._df.values, self._median)
    
    @computed_field_cached_property()
    def _labeled_distance(self) -> dict[tuple[int, int], np.array]:
        data = {}
        for label_i in self.unique_labels:
            idx_i = np.argwhere(self.labels == label_i).flatten()

            for label_j in self.unique_labels:
                idx_j = np.argwhere(self.labels == label_j).flatten()
                data[label_i, label_j] = self._distance[np.ix_(idx_i, idx_j)]

        return data
    
    @computed_field_cached_property()
    def _labeled_distance_to_mean(self) -> dict[tuple[int, int], np.array]:
        unique_labels = [self._all, *self.unique_labels]

        data = {}
        for label_i in unique_labels:
            if label_i == self._all:
                idx_i = np.arange(self.n_total)
            else:
                idx_i = np.argwhere(self.labels == label_i).flatten()

            for idx_j, label_j in enumerate(unique_labels):
                # if (label_i == self._all) and (label_j != self._all):
                #     continue

                idx_j = np.array([idx_j])
                data[label_i, label_j] = self._distance_to_mean[np.ix_(idx_i, idx_j)].flatten()

        return data
    
    @computed_field_cached_property()
    def _labeled_distance_to_median(self) -> dict[tuple[int, int], np.array]:
        unique_labels = [self._all, *self.unique_labels]

        data = {}
        for label_i in unique_labels:
            if label_i == self._all:
                idx_i = np.arange(self.n_total)
            else:
                idx_i = np.argwhere(self.labels == label_i).flatten()

            for idx_j, label_j in enumerate(unique_labels):
                # if (label_i == self._all) and (label_j != self._all):
                #     continue

                idx_j = np.array([idx_j])
                data[label_i, label_j] = self._distance_to_median[np.ix_(idx_i, idx_j)].flatten()

        return data

    def _labeled_distance_to_nearest_cluster(self, agg: str = "mean") -> dict[int, np.array]:
        data = {}
        for label_i in self.unique_labels:
            n = self._labeled_distance[label_i, label_i].shape[0]
            dist_to_nearest = np.ones(n) * np.inf
            for idx_i in range(n):
                for label_j in self.unique_labels:
                    if label_i == label_j:
                        continue

                    dist_matrix = self._labeled_distance[label_i, label_j]
                    if agg == "mean":
                        avg_dist = np.mean(dist_matrix[idx_i, :])
                    else:
                        avg_dist = np.median(dist_matrix[idx_i, :])

                    if avg_dist < dist_to_nearest[idx_i]:
                        dist_to_nearest[idx_i] = avg_dist

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
            n = self._labeled_distance[label_i, label_i].shape[0]

            distance_array = self._labeled_distance[label_i, label_i]
            dist_to_nearest = np.empty(n)
            for idx in range(n):
                mask = np.ones(n, dtype=bool)
                mask[idx] = False
                distance_masked = distance_array[idx][mask]

                if agg == "mean":
                    dist_to_nearest[idx] = np.mean(distance_masked)
                else:
                    dist_to_nearest[idx] = np.median(distance_masked)

            data[label_i] = dist_to_nearest

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

            # pair distance metrics
            distance = {}
            distance_to_mean = {}
            distance_to_median = {}
            for key in self._labeled_distance_to_mean.keys():
                if key[0] != label:
                    continue
                
                if key[1] == self._all:
                    key_alias = "all"

                else:
                    key_alias = key[1]

                    distance[key[1]] = ClusterPairDistanceMetrics(
                        cluster_ids=key,
                        distance=self._labeled_distance[key],
                    )


                distance_to_mean[key_alias] = ClusterPairDistanceMetrics(
                    cluster_ids=key,
                    distance=self._labeled_distance_to_mean[key],
                )
                
                distance_to_median[key_alias] = ClusterPairDistanceMetrics(
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
                distance=distance,
                distance_to_mean=distance_to_mean,
                distance_to_median=distance_to_median,
                mean_distance_intra_cluster=mean_distance_intra_cluster,
                median_distance_intra_cluster=median_distance_intra_cluster,
                mean_distance_to_nearest_cluster=mean_distance_to_nearest_cluster,
                median_distance_to_nearest_cluster=median_distance_to_nearest_cluster,
            )

        return data

    @computed_field_cached_property()
    def _WCSS(self) -> float:
        """
        Within-Cluster Sum of Squares
        """

        wcss = 0.0
        for label in self.unique_labels:
            wcss += self.cluster[label].distance_to_mean[label].sum_of_squares

        return wcss

    @computed_field_cached_property()
    def _BCSS(self) -> float:
        """
        Between-Cluster Sum of Squares
        """

        overall_mean = self._mean[0]

        bcss = 0.0
        for i, label in enumerate(self.unique_labels):
            n = self._n[i + 1]
            cluster_mean = self._mean[i + 1]

            dist = np.linalg.norm(cluster_mean - overall_mean)

            bcss += n * (dist**2)

        return bcss
    
    @computed_field_cached_property()
    def duda_hart_index(self) -> float:
        # Duda and Hart Index
        # Range is 0 to inf, 0 is the best

        intracluster_distance = 0
        intercluster_distance = 0
        for label in self.unique_labels:
            intracluster_distance +=self.cluster[label].distance_to_mean[label].mean

            intra_idx = np.argwhere(self.labels == label).flatten()
            inter_idx = np.argwhere(self.labels != label).flatten()

            intercluster_distance += np.mean(self._distance[np.ix_(intra_idx, inter_idx)])

        res = intracluster_distance / intercluster_distance

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

    @computed_field_cached_property()
    def ball_hall_index(self) -> float:
        # Ball and Hall Index
        # Range is 0 to inf, 0 is the best
        # Formula: (1/K) * Σ(sum of squared distances from points to cluster centroids)

        k = self.label_count  # number of clusters

        # Sum of squared distances to cluster means across all clusters
        inner_cluster_ss = 0
        for label in self.unique_labels:
            inner_cluster_ss += self.cluster[label].distance_to_mean[label].sum_of_squares

        res = inner_cluster_ss / k

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def pm_hartigan_index(self) -> float:
        # Hartigan Index
        # Range is 0 to inf, inf is the best
        cm = ClusteringMetric(
            X=self.data,
            y_pred=self.labels,
        )

        res = cm.hartigan_index()

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def hartigan_index(self) -> float:
        # Hartigan Index
        # Range is 0 to inf, inf is the best
        # Formula: log(BCSS / WCSS) 
        # where BCSS = between-cluster sum of squares,
        #       WCSS = within-cluster sum of squares

        k = self.label_count  # number of clusters
        n = self.n_total  # number of data points
        mean_all = self._mean[0]

        BCSS = self._BCSS # Between Cluster Sum of Squares (BCSS)
        WCSS = self._WCSS # Within Cluster Sum of Squares (WCSS)

        # Avoid division by zero
        # TODO: change to near zero
        if WCSS == 0:
            res = np.inf
        elif BCSS == 0:
            res = 0
        else:
            res = np.log(BCSS / WCSS)

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def silhouette_index(self) -> float:
        # range is -1 to 1, 1 is the best
        silhouette_coefficients = []
        for cluster_id in self.unique_labels:
            silhouette_coefficients.append(self.cluster[cluster_id].mean_silhouette_coefficient)

        silhouette_coefficients = np.hstack(silhouette_coefficients)

        res = np.mean(silhouette_coefficients)

        if self.index_direction == "minimize":
            res *= -1

        return res
    
    @computed_field_cached_property()
    def silhouette_median_index(self) -> float:
        # range is -1 to 1, 1 is the best
        silhouette_coefficients = []
        for cluster_id in self.unique_labels:
            silhouette_coefficients.append(self.cluster[cluster_id].median_silhouette_coefficient)

        silhouette_coefficients = np.hstack(silhouette_coefficients)

        res = np.median(silhouette_coefficients)

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def calinski_harabasz_index(self) -> float:
        # range is 0 to inf, inf is the best
        k = self.label_count # number of clusters
        n = self.n_total # number of data points
        mean_all = self._mean[0]

        BCSS = self._BCSS  # Between Cluster Sum of Squares (BCSS)
        WCSS = self._WCSS  # Within Cluster Sum of Squares (WCSS)

        if WCSS == 0:
            return 1.0

        res = (BCSS / WCSS) * ((n - k) / (k - 1.0))

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def calinski_harabasz_index(self) -> float:
        # range is 0 to inf, inf is the best
        k = self.label_count # number of clusters
        n = self.n_total # number of data points
        mean_all = self._mean[0]

        # Between Cluster Sum of Squares
        BCSS = 0
        for label in self.unique_labels:
            cluster_mean = self.cluster[label].mean
            dist = cdist(cluster_mean[None, :], mean_all[None, :]).flatten()[0]
            dist = dist**2
            BCSS += self.cluster[label].n * dist

        # Within Cluster Sum of Squares
        WCSS = 0
        for label in self.unique_labels:
            WCSS += np.sum(self.cluster[label].distance_to_mean[label].sum_of_squares)

        if WCSS == 0:
            return 1.0

        res = (BCSS / WCSS) * ((n - k) / (k - 1.0))

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def variance_ratio_criterion(self) -> float:
        return self.calinski_harabasz_index

    @computed_field_cached_property()
    def davies_bouldin_index(self) -> float:
        # range is 0 to inf, 0 is the best
        k = self.label_count
        # Could abstract with scipy.stats.moment

        intracluster_distance = []
        for label in self.unique_labels:
            intracluster_distance.append(self.cluster[label].distance_to_mean[label].mean)

        intracluster_distance = np.array(intracluster_distance)

        intercluster_distance = squareform(pdist(self._mean[1:]))

        similarity = np.zeros((self.label_count, self.label_count))
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                
                elif j < i:
                    similarity[i, j] = similarity[j, i]
                    continue

                dist_i = intracluster_distance[i]
                dist_j = intracluster_distance[j]
                dist_ij = intercluster_distance[i, j]

                similarity[i, j] = (dist_i + dist_j) / dist_ij

        res = np.sum(np.max(similarity, axis=1)) / k

        if self.index_direction == "maximize":
            res *= -1

        return res


    @computed_field_cached_property()
    def pm_xie_beni_index(self) -> float:
        # Range is 0 to inf, 0 is the best
        cm = ClusteringMetric(
            X=self.data,
            y_pred=self.labels,
        )

        res = cm.xie_beni_index()

        if self.index_direction == "maximize":
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

        WCSS = self._WCSS

        # Find minimum distance between cluster centroids
        # Get all cluster means
        cluster_means = []
        for label in self.unique_labels:
            cluster_means.append(self.cluster[label].mean)
        cluster_means = np.array(cluster_means)

        # Calculate pairwise distances between centroids
        if len(cluster_means) > 1:
            centroid_distances = pdist(cluster_means)
            d_min_squared = np.min(centroid_distances) ** 2
        else:
            # If only one cluster, return infinity (worst score)
            return np.inf if self.index_direction == "minimize" else -np.inf

        # Avoid division by zero
        if d_min_squared == 0:
            res = np.inf
        else:
            res = WCSS / (n * d_min_squared)

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

        res = 0.0
        for label in self.unique_labels:
            n_k = self.cluster[label].n
            # trace of within-cluster scatter matrix = sum of squared distances
            trace_W_k = self.cluster[label].distance_to_mean[label].sum_of_squares

            # Avoid log(0) or division by zero
            if n_k > 0 and trace_W_k > 0:
                res += n_k * np.log(trace_W_k / n_k)
            elif trace_W_k == 0:
                # Perfect clustering (no variance) -> very negative value
                res += n_k * np.log(1e-10)

        if self.index_direction == "maximize":
            res *= -1

        return res

    @computed_field_cached_property()
    def pm_ksq_detw_index(self) -> float:
        # Range is -inf to inf, inf is the best
        cm = ClusteringMetric(
            X=self.data,
            y_pred=self.labels,
        )

        res = cm.ksq_detw_index()

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def ksq_detw_index(self) -> float:
        # KSq-DetW Index (K² × det(W))
        # Range is -inf to inf, inf is the best
        # Formula: K² × det(W)
        # where K = number of clusters,
        #       W = pooled within-cluster scatter matrix,
        #       det(W) = determinant of W

        k = self.label_count  # number of clusters
        n_features = self.data.shape[1]

        # Initialize pooled within-cluster scatter matrix
        W = np.zeros((n_features, n_features))

        # Compute scatter matrix for each cluster and sum them
        for label in self.unique_labels:
            cluster_mask = self.labels == label
            cluster_data = self.data[cluster_mask]
            cluster_mean = self.cluster[label].mean

            # Compute scatter matrix for this cluster: Σ(x - mean)(x - mean)^T
            centered_data = cluster_data - cluster_mean
            W_k = centered_data.T @ centered_data
            W += W_k

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
        #       W = pooled within-cluster scatter matrix

        n_features = self.data.shape[1]
        mean_all = self._mean[0]

        # Compute total scatter matrix T
        centered_data = self.data - mean_all
        T = centered_data.T @ centered_data

        # Compute pooled within-cluster scatter matrix W
        W = np.zeros((n_features, n_features))
        for label in self.unique_labels:
            cluster_mask = self.labels == label
            cluster_data = self.data[cluster_mask]
            cluster_mean = self.cluster[label].mean

            centered_cluster = cluster_data - cluster_mean
            W_k = centered_cluster.T @ centered_cluster
            W += W_k

        # Compute determinants
        try:
            det_T = np.linalg.det(T)
            det_W = np.linalg.det(W)

            # Avoid division by zero
            if det_W == 0 or np.abs(det_W) < 1e-10:
                res = np.inf
            else:
                res = det_T / det_W
        except np.linalg.LinAlgError:
            # Handle singular matrices
            res = np.inf

        if self.index_direction == "minimize":
            res *= -1

        return res
    
    @computed_field_cached_property()
    def pm_dunn_index(self) -> float:
        # Dunn Index
        # Range is 0 to inf, inf is the best
        cm = ClusteringMetric(
            X=self.data,
            y_pred=self.labels,
        )

        res = cm.dunn_index()

        if self.index_direction == "minimize":
            res *= -1

        return res
    
    @computed_field_cached_property()
    def dunn_index(self) -> float:
        # Dunn Index
        # Range is 0 to inf, inf is the best
        # Formula: min(inter-cluster distance) / max(intra-cluster diameter)
        # where inter-cluster distance = min distance between points in different clusters
        #       intra-cluster diameter = max distance between points in same cluster

        # Find minimum inter-cluster distance
        min_inter_distance = np.inf
        for i, label_i in enumerate(self.unique_labels):
            for label_j in self.unique_labels[i+1:]:
                # Get distances between all points in cluster i and j
                distances = self._labeled_distance[label_i, label_j]
                min_dist = np.min(distances)
                if min_dist < min_inter_distance:
                    min_inter_distance = min_dist

        # Find maximum intra-cluster diameter
        max_intra_diameter = 0
        for label in self.unique_labels:
            # Get distances within cluster (diameter is max distance)
            intra_distances = self._labeled_distance[label, label]
            max_dist = np.max(intra_distances)
            if max_dist > max_intra_diameter:
                max_intra_diameter = max_dist

        # Avoid division by zero
        if max_intra_diameter == 0:
            res = np.inf
        else:
            res = min_inter_distance / max_intra_diameter

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def pm_log_det_ratio_index(self) -> float:
        # Range is -inf to inf, inf is the best
        cm = ClusteringMetric(
            X=self.data,
            y_pred=self.labels,
        )

        res = cm.log_det_ratio_index()

        if self.index_direction == "minimize":
            res *= -1

        return res

    @computed_field_cached_property()
    def log_det_ratio_index(self) -> float:
        # Log Det Ratio Index
        # Range is -inf to inf, inf is the best
        # Formula: log(det(T) / det(W)) = log(det(T)) - log(det(W))
        # where T = total scatter matrix,
        #       W = pooled within-cluster scatter matrix

        n_features = self.data.shape[1]
        mean_all = self._mean[0]

        # Compute total scatter matrix T
        centered_data = self.data - mean_all
        T = centered_data.T @ centered_data

        # Compute pooled within-cluster scatter matrix W
        W = np.zeros((n_features, n_features))
        for label in self.unique_labels:
            cluster_mask = self.labels == label
            cluster_data = self.data[cluster_mask]
            cluster_mean = self.cluster[label].mean

            centered_cluster = cluster_data - cluster_mean
            W_k = centered_cluster.T @ centered_cluster
            W += W_k

        # Compute log determinants (more numerically stable)
        try:
            det_T = np.linalg.det(T)
            det_W = np.linalg.det(W)

            # Avoid log of zero or negative values
            if det_W <= 0 or np.abs(det_W) < 1e-10:
                res = np.inf
            elif det_T <= 0:
                res = -np.inf
            else:
                # log(det(T) / det(W)) = log(det(T)) - log(det(W))
                res = np.log(det_T) - np.log(det_W)
        except np.linalg.LinAlgError:
            # Handle singular matrices
            res = np.inf

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

        k = self.label_count  # number of clusters
        n = self.n_total  # number of data points
        mean_all = self._mean[0]

        BCSS = self._BCSS  # Between Cluster Sum of Squares (BCSS)
        WCSS = self._WCSS  # Within Cluster Sum of Squares (WCSS)

        # Avoid log of zero or division by zero
        if WCSS == 0 or WCSS < 1e-10:
            res = np.inf
        elif BCSS == 0:
            res = -np.inf
        else:
            # log(BCSS / WCSS) = log(BCSS) - log(WCSS)
            res = np.log(BCSS) - np.log(WCSS)

        if self.index_direction == "minimize":
            res *= -1

        return res

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

        # MSE = SSE / n
        res = WCSS / n

        if self.index_direction == "maximize":
            res *= -1

        return res
    
    @computed_field_cached_property()
    def pm_r_squared_index(self) -> float:
        # R-Squared Index
        # Range is -inf to 1, 1 is the best
        cm = ClusteringMetric(
            X=self.data,
            y_pred=self.labels,
        )

        res = cm.r_squared_index()

        if self.index_direction == "minimize":
            res = 1 - res

        return res

    @computed_field_cached_property()
    def r_squared_index(self) -> float:
        # R-Squared Index (Coefficient of Determination)
        # Range is -inf to 1, 1 is the best
        # Formula: R² = 1 - (WCSS / TSS)
        # where WCSS = within-cluster sum of squares,
        #       TSS = total sum of squares (variance around global mean)

        mean_all = self._mean[0]

        WCSS = self._WCSS

        # Total Sum of Squares (TSS)
        # TSS = sum of squared distances from all points to global mean
        TSS = np.sum((self.data - mean_all) ** 2)

        # Avoid division by zero
        if TSS == 0:
            res = 1.0  # Perfect clustering if no variance
        else:
            res = 1 - (WCSS / TSS)

        if self.index_direction == "minimize":
            res = 1 - res

        return res