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

import sys

import numpy as np

from pydantic import BaseModel, ConfigDict

import sklearn.metrics as _metrics

from opendsm.common.clustering.metrics.cluster_metrics import ClusterMetrics
from opendsm.common.clustering import settings as _settings


def get_max_score_from_system_size() -> float:
    """
    recreates the call to sys.float_info.max in order to
    follow what was used in ads repo.

    Making into function which executes each time
    so unforseen issues are less likely when running on
    distributed env
    """

    return sys.float_info.max**0.5


def renumber_clusters(clusters: np.ndarray, reorder: bool):
    """Takes in cluster identifiers and renumbers them.
        After merging or reclustering there are many cluster numbers left blank and need to be renumbered
            Example: [0, 1, 2, 5, 7]
        Additionally the clusters are reordered from largest cluster to smallest

    Args:
        clusters (list|np.array): an array in which a cluster number is defined for each load shape

    Returns:
        clusters (np.array): an array in which a cluster number is defined for each load shape
    """

    if reorder:
        # if outlier cluster exists, don't include it in the ordering
        uniq_id, counts = np.unique(clusters[clusters != -1], return_counts=True)
        count_order = np.argsort(counts)[::-1]

        uniq_id = uniq_id[count_order]

    else:
        uniq_id = np.unique(clusters)

    # if outlier cluster exists, don't change it
    conv = {-1: -1}
    conv.update({uniq_id[i]: i for i in range(len(uniq_id))})

    clusters = np.array([conv[idx] for idx in clusters])

    return clusters


def merge_small_clusters(clusters: np.ndarray, min_cluster_size: int):
    """
    OG DOCSTRING:
    Merges clusters which consist of less than the minumum number into the outlier cluster

    Args:
        clusters (list|np.array): A list defining what cluster each load shape belongs to
        min_cluster_size (int): Minumum number of meters for a cluster
            Options: 2 < val

    Returns:
        _type_: _description_
    """

    uniq_ids, uniq_counts = np.unique(clusters, return_counts=True)

    uniq_counts = uniq_counts[uniq_ids != -1]
    uniq_ids = uniq_ids[uniq_ids != -1]

    outlier_ids = uniq_ids[uniq_counts < min_cluster_size]
    clusters[np.isin(clusters, outlier_ids)] = -1

    return renumber_clusters(clusters, reorder=True)


class LabelResult(BaseModel):
    """
    contains metrics about a cluster label
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    labels: np.ndarray
    score: dict[str, float]
    score_unable_to_be_calculated: dict[str, bool]
    n_clusters: int


_score_council_init = {
    'calinski_harabasz_index': 1.0,
    'davies_bouldin_index': 1.0,
    'density_based_clustering_validation_index': 1.0,
    'dunn_index': 1.0,
    'silhouette_index': 1.0,
    'silhouette_median_index': 1.0,
    'xie_beni_index': 1.0,
}

def _score_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    n_cluster_lower: int,
    score_council: dict[str, float] = _score_council_init, 
    dist_metric="euclidean",
    min_cluster_size=2,
    max_non_outlier_cluster_count=200,
) -> LabelResult:
    """
    ---
    Original docstring:

    Score clusters of the given data with the selected choices.
    Small clusters are first merged to only score clusters above the minimum size
    and not in the outlier cluster.

    Args:
        data (np.array): Load shapes being clustered
        labels (list|np.array): A list defining what cluster each load shape belongs to

    Returns:
        score (float): Lower is better
        unable_to_calc_score (bool): Boolean that if true, means max score was used
    """

    n_clusters = len(np.unique(labels))

    # merge clusters to outlier cluster
    labels = merge_small_clusters(labels, min_cluster_size)

    non_outlier_cluster_count = labels.max() + 1
    invalid = False
    if non_outlier_cluster_count < n_cluster_lower:
        invalid = True
    elif non_outlier_cluster_count > max_non_outlier_cluster_count:
        invalid = True
        
    if invalid:
        return LabelResult(
            labels=labels,
            score={voter: np.inf for voter in score_council.keys()},
            score_unable_to_be_calculated={voter: True for voter in score_council.keys()},
            n_clusters=n_clusters,
        )

    # don't include outlier cluster in scoring
    idx = np.argwhere(labels != -1).flatten()
    data_non_outlier = data[idx, :]
    labels_non_outlier = labels[idx]

    metrics = ClusterMetrics(
        data=data_non_outlier,
        labels=labels_non_outlier,
        distance_metric=dist_metric,
    )
    score = {}
    score_unable_to_be_calculated = {}
    for score_choice, score_weight in score_council.items():
        score[score_choice] = np.inf
        score_unable_to_be_calculated[score_choice] = True

        if score_weight > 0:
            try:
                score[score_choice] = getattr(metrics, score_choice)

                if np.isfinite(score[score_choice]):
                    score_unable_to_be_calculated[score_choice] = False
            except:
                continue
    
    label_res = LabelResult(
        labels=labels,
        score=score,
        score_unable_to_be_calculated=score_unable_to_be_calculated,
        n_clusters=n_clusters,
    )

    return label_res


def score_council(settings: _settings.ClusteringSettings):
    """
    Set the score council for the given settings.
    """

    algo_settings = getattr(settings, settings.algorithm_selection)
    algo_scoring = algo_settings.scoring

    score_council = {
        'calinski_harabasz_index': algo_scoring.calinski_harabasz_weight,
        'davies_bouldin_index': algo_scoring.davies_bouldin_weight,
        'density_based_clustering_validation_index': algo_scoring.density_based_clustering_validation_weight,
        'dunn_index': algo_scoring.dunn_weight,
        'silhouette_index': algo_scoring.silhouette_weight,
        'silhouette_median_index': algo_scoring.silhouette_median_weight,
        'xie_beni_index': algo_scoring.xie_beni_weight,
    }

    return score_council


def score_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    Score clusters of the given data with the selected choices.
    """
    algo_settings = getattr(settings, settings.algorithm_selection)
    algo_scoring = algo_settings.scoring

    n_cluster_lower = algo_settings.n_cluster.lower
    dist_metric = algo_scoring.distance_metric
    min_cluster_size = algo_scoring.min_cluster_size
    max_non_outlier_cluster_count = algo_scoring.max_non_outlier_cluster_count
    
    label_res = _score_clusters(
        data,
        labels,
        n_cluster_lower,
        score_council(settings),
        dist_metric,
        min_cluster_size,
        max_non_outlier_cluster_count,
    )

    return label_res