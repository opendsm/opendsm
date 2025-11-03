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
from typing import Optional

import pandas as pd

from opendsm.comparison_groups.common.base_comparison_group import Comparison_Group_Algorithm
from opendsm.comparison_groups.cg_clustering import settings as _settings
import opendsm.comparison_groups.cg_clustering.bounds as _bounds
from opendsm.comparison_groups.cg_clustering import treatment_fit as _treatment_fit 
from opendsm.common.clustering.cluster import cluster_features as _cluster


class CG_Clustering(Comparison_Group_Algorithm):
    clusters = None
    comparison_pool_loadshape = None
    treatment_loadshape = None

    def __init__(self, settings: Optional[_settings.CG_Clustering_Settings] = None):
        if settings is None:
            settings = _settings.CG_Clustering_Settings()

        self.settings = settings

    def get_labels(self, comparison_pool_data):
        self.comparison_pool_data = comparison_pool_data
        self.comparison_pool_loadshape = comparison_pool_data.loadshape

        # update cluster count
        algo = f"{self.settings.algorithm_selection.value}"
        algo_settings = getattr(self.settings, algo)

        n_cluster_min, n_cluster_max = _bounds.get_cluster_bounds(
            data_size=len(self.comparison_pool_data.ids),
            min_cluster_size=algo_settings.scoring.min_cluster_size,
            num_cluster_bound_lower=algo_settings.n_cluster.lower,
            num_cluster_bound_upper=algo_settings.n_cluster.upper
        )

        settings_dict = self.settings.model_dump()
        settings_dict[algo]["n_cluster"]["lower"] = n_cluster_min
        settings_dict[algo]["n_cluster"]["upper"] = n_cluster_max
        self.settings = _settings.CG_Clustering_Settings(**settings_dict)

        # perform clustering
        labels = _cluster(
            self.comparison_pool_loadshape.copy(), # copy is only necessary for plotting later
            self.settings
        )

        self.clusters = pd.DataFrame(
            {"cluster": labels}, 
            index=self.comparison_pool_data.ids
        )
        self.clusters.index.name = "id"

        return self.clusters

    def match_treatment_to_clusters(self, treatment_data):
        if self.clusters is None:
            raise ValueError(
                "Comparison group has been not been clustered. Run 'get_labels' first."
            )
        
        self.treatment_data = treatment_data
        self.treatment_ids = treatment_data.ids
        self.treatment_loadshape = treatment_data.loadshape

        self.treatment_weights = _treatment_fit.match_treatment_to_clusters(
            self.treatment_loadshape,
            self.comparison_pool_loadshape,
            self.clusters,
            settings=self.settings
        )

        return self.treatment_weights

    def get_comparison_group(self, treatment_data, comparison_pool_data):
        df_cg = self.get_labels(comparison_pool_data)
        df_t_coeffs = self.match_treatment_to_clusters(treatment_data)

        return df_cg, df_t_coeffs
