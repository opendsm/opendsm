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
from typing import Union, Optional
from enum import Enum

import numpy as np
import pandas as pd

from scipy.stats import moment
from scipy.spatial.distance import cdist, pdist, squareform

from functools import cached_property  # TODO: This requires Python 3.8

from opendsm.common.utils import median_absolute_deviation, t_stat, safe_divide
from opendsm.common.pydantic_utils import (
    ArbitraryPydanticModel,
    PydanticDf,
    PydanticFromDict,
    computed_field_cached_property,
)



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
    SEUCLIDEAN = "seuclidean"
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
    def n(self) -> float:
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

    n: float = pydantic.Field()

    mean: np.array = pydantic.Field()

    median: np.array = pydantic.Field()

    distance: dict[int, ClusterPairDistanceMetrics] | ClusterPairDistanceMetrics = pydantic.Field()

    distance_to_mean: dict[int | str, ClusterPairDistanceMetrics] | ClusterPairDistanceMetrics = pydantic.Field()

    distance_to_median: dict[int | str, ClusterPairDistanceMetrics] | ClusterPairDistanceMetrics = pydantic.Field()
    

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
            
            data[label] = SingleClusterMetrics(
                cluster_id=label,
                n=n,
                mean=mean,
                median=median,
                distance=distance,
                distance_to_mean=distance_to_mean,
                distance_to_median=distance_to_median,
            )

        return data
    
    @computed_field_cached_property()
    def duda_hart_index(self) -> float:
        # Duda and Hart Index
        pass

    @computed_field_cached_property()
    def ssei_index(self) -> float:
        # Sum of Squared Errors Index
        pass

    @computed_field_cached_property()
    def beale_index(self) -> float:
        # aka “variance ratio criterion” or the “F-ratio”
        pass

    @computed_field_cached_property()
    def silhouette_index(self) -> float:
        pass

    @computed_field_cached_property()
    def inverse_silhouette_index(self) -> float:
        pass

    @computed_field_cached_property()
    def calinski_harabasz_index(self) -> float:
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

        return (BCSS / WCSS) * ((n - k) / (k - 1.0))

    @computed_field_cached_property()
    def davies_bouldin_index(self) -> float:
        k = self.label_count

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

        return np.sum(np.max(similarity, axis=1)) / k
