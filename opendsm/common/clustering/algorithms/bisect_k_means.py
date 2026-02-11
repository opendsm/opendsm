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

from opendsm.common.clustering.algorithms import sklearn_bisect_k_means as _bisect_k_means
from opendsm.common.clustering import (
    scoring as _scoring,
    settings as _settings,
    voting as _voting,
)



def bisect_k_means(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using Bisecting K-Means algorithm
    """

    algo_settings = settings.bisecting_kmeans
    recluster_count = algo_settings.recluster_count
    n_cluster_lower = algo_settings.n_cluster.lower
    n_cluster_upper = algo_settings.n_cluster.upper
    n_init = algo_settings.internal_recluster_count
    inner_algorithm = algo_settings.inner_algorithm
    bisecting_strategy = algo_settings.bisecting_strategy

    window_size = algo_settings.scoring.window_size
    min_cluster_size = algo_settings.scoring.min_cluster_size

    seed = settings._seed

    # Validate that we have enough samples to create the minimum number of clusters
    n_samples = data.shape[0]
    min_required_samples = n_cluster_lower * min_cluster_size
    if n_samples <= min_required_samples:
        raise ValueError(
            f"Insufficient samples for clustering: need more than {min_required_samples} samples "
            f"(n_cluster_lower={n_cluster_lower} * min_cluster_size={min_cluster_size}), "
            f"but only have {n_samples} samples"
        )

    results = []
    for i in range(recluster_count + 1):
        algo = _bisect_k_means.BisectingKMeans(
            n_clusters=n_cluster_upper,
            init="k-means++",  # does not benefit from k-means++ like other k-means
            n_init=n_init,
            random_state=seed + i,
            algorithm=inner_algorithm,
            bisecting_strategy=bisecting_strategy,
        )
        algo.fit(data)
        labels_dict = algo.labels_full

        # if specifying clusters, only score the specified clusters
        if n_cluster_lower == n_cluster_upper:
            labels_dict = {n_cluster_lower: labels_dict[n_cluster_lower]}

        for n_cluster, labels in labels_dict.items():
            label_res = _scoring.score_clusters(data, labels, settings)
            results.append(label_res)

    # Check if all results have score_unable_to_be_calculated == True
    if all(all(result.score_unable_to_be_calculated.values()) for result in results):
        return results[0].labels

    # Construct voting df and perform voting to select best cluster count
    df_votes = _voting.construct_voting_df(results)
    winner_idx = _voting.shulze_voting(
        df_votes, 
        _scoring.score_council(settings), 
        window_size
    )
    # get labels of winner from results
    winner_labels = results[winner_idx].labels

    return winner_labels