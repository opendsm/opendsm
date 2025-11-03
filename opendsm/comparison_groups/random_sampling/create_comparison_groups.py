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

import numpy as np
import pandas as pd

from opendsm.comparison_groups.common.base_comparison_group import Comparison_Group_Algorithm

from opendsm.comparison_groups.random_sampling.settings import Settings


class Random_Sampling(Comparison_Group_Algorithm):
    def __init__(self, settings: Optional[Settings] = None):
        if settings is None:
            settings = Settings()
        
        self.settings = settings

    def _create_clusters_df(self, df_raw):
        clusters = df_raw
        clusters["cluster"] = 0
        clusters["weight"] = 1.0
        clusters = clusters.reset_index().set_index("id")

        # reorder columns
        clusters = clusters[["cluster", "weight"]]

        return clusters
    

    def _create_treatment_weights_df(self, ids):
        coeffs = np.ones(len(ids))

        treatment_weights = pd.DataFrame(coeffs, index=ids, columns=["pct_cluster_0"])
        treatment_weights.index.name = "id"

        return treatment_weights
    

    def get_comparison_group(self, treatment_data, comparison_pool_data, weights=None):
        settings = self.settings

        if settings.n_meters_total is not None:
            n_meters = self.settings.n_meters_total

        elif settings.n_meters_per_treatment is not None:
            n_treatment_meters = len(treatment_data.ids)
            n_meters = n_treatment_meters * settings.n_meters_per_treatment

        else:
            raise ValueError("`n_meters_total` or `n_meters_per_treatment` must be defined")

        self.treatment_data = treatment_data
        self.comparison_pool_data = comparison_pool_data

        self.treatment_ids = treatment_data.ids
        self.treatment_loadshape = treatment_data.loadshape
        self.comparison_pool_loadshape = comparison_pool_data.loadshape

        # randomly sample n_meters from comparison pool
        df_cg = comparison_pool_data.loadshape.sample(n_meters, random_state=settings.seed)

        clusters = self._create_clusters_df(df_cg)

        # Create treatment_weights
        treatment_weights = self._create_treatment_weights_df(self.treatment_ids)

        # Assign dfs to self
        self.clusters = clusters
        self.treatment_weights = treatment_weights

        return clusters, treatment_weights
