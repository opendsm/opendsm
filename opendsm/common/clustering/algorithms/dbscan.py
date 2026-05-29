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

from sklearn.cluster import DBSCAN

from opendsm.common.clustering.metrics.labels import ClusteringResult


def dbscan(data, settings):
    """Clusters features using DBSCAN algorithm. Returns ClusteringResult."""
    algo_settings = getattr(settings, settings.algorithm_selection.value)

    algo = DBSCAN(
        eps=algo_settings.epsilon,
        min_samples=algo_settings.min_samples,
        metric=settings._metric_value,
        algorithm=algo_settings.nearest_neighbors_algorithm,
        leaf_size=algo_settings.leaf_size,
        p=algo_settings.minkowski_p,
    )
    labels = algo.fit_predict(data)

    lbl = ClusteringResult(
        data=data,
        score_settings=algo_settings.scoring,
        seed=settings._seed,
        min_cluster_size=settings.min_cluster_size,
        small_cluster_mode=settings.small_cluster_mode,
    )
    lbl.add(0, labels)
    return lbl
