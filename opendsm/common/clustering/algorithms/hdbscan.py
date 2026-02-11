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

from opendsm.common.clustering import settings as _settings



def hdbscan(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using HDBSCAN algorithm
    """
    min_samples = settings.hdbscan.min_samples
    if settings.hdbscan.min_samples == 1:
        min_samples = 2

    algo = HDBSCAN(
        min_samples=settings.hdbscan.scoring_sample_count, 
        min_cluster_size=min_samples,
        allow_single_cluster=settings.hdbscan.allow_single_cluster,
        max_cluster_size=settings.hdbscan.max_cluster_size,
        metric=settings.hdbscan.distance_metric,
        cluster_selection_epsilon=settings.hdbscan.cluster_selection_epsilon,
        alpha=settings.hdbscan.robust_single_linkage_scaling,
        algorithm=settings.hdbscan.nearest_neighbors_algorithm,
        leaf_size=settings.hdbscan.leaf_size,
        cluster_selection_method=settings.hdbscan.cluster_selection_method,
    )
    labels = algo.fit_predict(data)

    if settings.hdbscan.min_samples == 1:
        # get count of -1 labels
        outlier_count = np.sum(labels == -1)

        if outlier_count == 0:
            return labels

        # add to all labels to make room for outliers
        labels[labels != -1] += outlier_count

        # make labels with -1 defined as arange(max_label+1, n_samples)
        labels[labels == -1] = np.arange(0, outlier_count)

    return labels