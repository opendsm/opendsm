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

from sklearn.cluster import Birch

from opendsm.common.clustering import (
    scoring as _scoring,
    settings as _settings,
    voting as _voting,
)



def birch(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    Clusters features using Birch algorithm
    """

    n_cluster_lower = settings.birch.n_cluster.lower
    n_cluster_upper = settings.birch.n_cluster.upper
    threshold = settings.birch.threshold
    branching_factor = settings.birch.branching_factor

    window_size = settings.birch.scoring.window_size

    results = []
    for n_clusters in range(n_cluster_lower, n_cluster_upper + 1):
        algo = Birch(
            n_clusters=n_clusters,
            threshold=threshold,
            branching_factor=branching_factor,
        )
        labels = algo.fit_predict(data)
        
        # Calculate score for the clusters
        label_res = _scoring.score_clusters(data, labels, settings)
        
        results.append(label_res)

    df_votes = _voting.construct_voting_df(results)
    winner_idx = _voting.shulze_voting(
        df_votes, 
        _scoring.score_council(settings), 
        window_size
    )
    # get labels of winner from results
    winner_labels = results[winner_idx].labels

    return winner_labels