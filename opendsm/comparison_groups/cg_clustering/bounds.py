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


def _get_num_cluster_min(
    data_size: int, 
    min_cluster_size: int, 
    num_cluster_bound_lower: int
):
    """
    returns lower bounds using data_size which is number models in cluster
    """

    linear = False

    # assume we want 8 clusters of min size 15 meters with a pool of 1000 meters
    base_pool = 1000
    base_cluster_size = 15
    base_min_clusters = 8

    if linear:
        k = (base_cluster_size*base_min_clusters)/base_pool
        num_cluster_min = k*data_size/min_cluster_size
    
    else:
        k = 30 + 4.58*np.exp(data_size/335)
        num_cluster_min = k/min_cluster_size

    n = max(int(np.floor(num_cluster_min)), 2)
    n = min(num_cluster_bound_lower, n)

    return n


def _get_num_cluster_max(
    data_size: int, 
    min_cluster_size: int, 
    num_cluster_bound_upper: int
):
    """
    returns upper bounds using data_size which is number models in cluster
    """
    n_min = min_cluster_size

    min_clusters = 1
    max_clusters = num_cluster_bound_upper

    # assume we want 250 with a size of 1000
    n_set = 1000
    n_max_set = 250

    k = (n_set - n_min) * (
        np.log(
            (
                ((n_max_set - min_clusters) / (2 * max_clusters - min_clusters) + 0.5)
                ** -1
                - 1
            )
            ** -1
        )
    ) ** -1

    if not np.isfinite(k):
        """
        TODO: Figure out better way to handle this.
        Currently occurs when num_cluster_bound_upper is less than n_max_set
        """
        return min(data_size, num_cluster_bound_upper)

    num_cluster_max = (2 * max_clusters - min_clusters) * (
        1 / (1 + np.exp(-(data_size - n_min) / k)) - 0.5
    ) + min_clusters

    n = max(int(np.floor(num_cluster_max)), 2)

    return n


def get_cluster_bounds(
    data_size: int,
    min_cluster_size: int,
    num_cluster_bound_lower: int,
    num_cluster_bound_upper: int,
):
    """
    function which returns lower and upper bound based off config values and number of data points
    """

    num_cluster_min = _get_num_cluster_min(
        data_size=data_size,
        min_cluster_size=min_cluster_size,
        num_cluster_bound_lower=num_cluster_bound_lower,
    )

    num_cluster_max = _get_num_cluster_max(
        data_size=data_size,
        min_cluster_size=min_cluster_size,
        num_cluster_bound_upper=num_cluster_bound_upper,
    )

    num_cluster_bounds = sorted([num_cluster_min, num_cluster_max])

    if num_cluster_bounds[0] == num_cluster_bounds[1]:
        num_cluster_bounds[1] += 1

    return num_cluster_bounds[0], num_cluster_bounds[1]