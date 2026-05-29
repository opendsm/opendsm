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

from sklearn.cluster import HDBSCAN

from opendsm.common.clustering.metrics.labels import ClusteringResult


def hdbscan(data, settings):
    """Clusters features using HDBSCAN algorithm. Returns ClusteringResult."""
    algo_settings = getattr(settings, settings.algorithm_selection.value)
    seed = settings._seed

    min_samples = algo_settings.min_samples
    if algo_settings.min_samples == 1:
        min_samples = 2

    algo = HDBSCAN(
        min_samples=algo_settings.neighborhood_min_samples,
        min_cluster_size=min_samples,
        allow_single_cluster=algo_settings.allow_single_cluster,
        max_cluster_size=algo_settings.max_cluster_size,
        metric=settings._metric_value,
        cluster_selection_epsilon=algo_settings.cluster_selection_epsilon,
        alpha=algo_settings.robust_single_linkage_scaling,
        algorithm=algo_settings.nearest_neighbors_algorithm,
        leaf_size=algo_settings.leaf_size,
        cluster_selection_method=algo_settings.cluster_selection_method,
        copy=True,
    )
    labels = algo.fit_predict(data)
    del algo

    if algo_settings.min_samples == 1:
        outlier_count = np.sum(labels == -1)
        if outlier_count > 0:
            labels[labels != -1] += outlier_count
            labels[labels == -1] = np.arange(0, outlier_count)

    lbl = ClusteringResult(
        data=data,
        score_settings=algo_settings.scoring,
        seed=seed,
        min_cluster_size=settings.min_cluster_size,
        small_cluster_mode=settings.small_cluster_mode,
    )
    lbl.add(0, labels)
    return lbl
