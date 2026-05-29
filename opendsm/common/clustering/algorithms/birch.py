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

from opendsm.common.clustering.metrics.labels import ClusteringResult


def birch(data, settings):
    """Clusters features using Birch algorithm. Returns ClusteringResult."""
    algo_settings = getattr(settings, settings.algorithm_selection.value)
    seed = settings._seed

    n_cluster_lower = algo_settings.n_cluster.lower
    n_cluster_upper = algo_settings.n_cluster.upper

    lbl = ClusteringResult(
        data=data,
        score_settings=algo_settings.scoring,
        seed=seed,
        n_cluster_lower=n_cluster_lower,
        min_cluster_size=settings.min_cluster_size,
        small_cluster_mode=settings.small_cluster_mode,
    )

    for n_clusters in range(n_cluster_lower, n_cluster_upper + 1):
        algo = Birch(
            n_clusters=n_clusters,
            threshold=algo_settings.threshold,
            branching_factor=algo_settings.branching_factor,
        )
        labels = algo.fit_predict(data)
        lbl.add(n_clusters, labels)

    return lbl
