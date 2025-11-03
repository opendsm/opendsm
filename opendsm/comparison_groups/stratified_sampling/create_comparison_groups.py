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

from opendsm.comparison_groups.stratified_sampling.model import StratifiedSampling
from opendsm.comparison_groups.stratified_sampling.bins import ModelSamplingException
from opendsm.comparison_groups.stratified_sampling.diagnostics import StratifiedSamplingDiagnostics
from opendsm.comparison_groups.stratified_sampling.bin_selection import StratifiedSamplingBinSelector

from opendsm.comparison_groups.stratified_sampling.settings import Settings


class Stratified_Sampling(Comparison_Group_Algorithm):
    def __init__(self, settings: Optional[Settings] = None):
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.df_raw = None

        self.model = StratifiedSampling()
        self.model_bin_selector = None

        for settings in self.settings.stratification_column:
            self.model.add_column(
                settings.column_name,
                n_bins=settings.n_bins,
                min_value_allowed=settings.min_value_allowed,
                max_value_allowed=settings.max_value_allowed,
                fixed_width=settings.is_fixed_width,
                auto_bin_require_equivalence=settings.auto_bin_equivalence,
            )

        self._diagnostics = None


    def _create_clusters_df(self, ids):
        clusters = pd.DataFrame(ids, columns=["id"])
        clusters["cluster"] = 0
        clusters["weight"] = 1.0

        clusters = clusters.reset_index().set_index("id")
        clusters = clusters[["cluster", "weight"]]

        return clusters


    def _create_treatment_weights_df(self, ids):
        coeffs = np.ones(len(ids))

        treatment_weights = pd.DataFrame(coeffs, index=ids, columns=["pct_cluster_0"])
        treatment_weights.index.name = "id"

        return treatment_weights
    
    def _create_output_dfs(self, t_ids):
        self.df_raw = self.model.data_sample.df

        # Create comparison group
        df_cg = self.df_raw[self.df_raw["_outlier_bin"] == False]
        clusters = self._create_clusters_df(df_cg["meter_id"].unique())

        # Create treatment_weights
        
        treatment_weights = self._create_treatment_weights_df(t_ids)

        # Assign dfs to self
        self.clusters = clusters
        self.treatment_weights = treatment_weights

        return clusters, treatment_weights


    def get_comparison_group(self, treatment_data, comparison_pool_data):
        settings = self.settings

        self.treatment_data = treatment_data
        self.comparison_pool_data = comparison_pool_data

        t_ids = treatment_data.ids
        t_features = treatment_data.features
        t_features = t_features.reset_index().rename(columns={"id": "meter_id"})

        cp_features = comparison_pool_data.features
        cp_features = cp_features.reset_index().rename(columns={"id": "meter_id"})

        if settings.equivalence_method is None:
            self.model.fit_and_sample(
                t_features, 
                cp_features,
                n_samples_approx=settings.n_samples_approx,
                relax_n_samples_approx_constraint=settings.relax_n_samples_approx_constraint,
                min_n_treatment_per_bin=settings.min_n_treatment_per_bin,
                min_n_sampled_to_n_treatment_ratio=settings.min_n_sampled_to_n_treatment_ratio,
                random_seed=settings.seed,
            )
        else:
            self.treatment_ids = t_ids
            self.treatment_loadshape = treatment_data.loadshape
            self.comparison_pool_loadshape = comparison_pool_data.loadshape
            t_loadshape = self.treatment_loadshape
            cp_loadshape = self.comparison_pool_loadshape

            df_equiv = pd.concat([t_loadshape, cp_loadshape])
            df_equiv.index.name = "meter_id"

            self.model_bin_selector = StratifiedSamplingBinSelector(
                self.model,
                t_features, 
                cp_features,
                equivalence_feature_ids=df_equiv.index,
                equivalence_feature_matrix=df_equiv,
                df_id_col="meter_id",
                equivalence_method=settings.equivalence_method,
                equivalence_quantile_size=settings.equivalence_quantile,
                
                n_samples_approx=settings.n_samples_approx,
                relax_n_samples_approx_constraint=settings.relax_n_samples_approx_constraint,

                min_n_bins=settings.min_n_bins,
                max_n_bins=settings.max_n_bins,
                min_n_treatment_per_bin=settings.min_n_treatment_per_bin,
                min_n_sampled_to_n_treatment_ratio=settings.min_n_sampled_to_n_treatment_ratio,
                random_seed=settings.seed,
            )

        clusters, treatment_weights = self._create_output_dfs(t_ids)

        return clusters, treatment_weights


    def diagnostics(self):
        if self.df_raw is None:
            raise RuntimeError("Must run get_comparison_group() before calling diagnostics()")
        
        if self._diagnostics is None:
            self._diagnostics = StratifiedSamplingDiagnostics(model=self.model)
        
        return self._diagnostics