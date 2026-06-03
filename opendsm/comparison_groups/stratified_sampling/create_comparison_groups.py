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

from opendsm.comparison_groups.stratified_sampling.sampling import StratifiedSampler
from opendsm.comparison_groups.stratified_sampling.diagnostics import StratifiedSamplingDiagnostics
from opendsm.comparison_groups.stratified_sampling.bin_selection import StratifiedSamplingBinSelector

from opendsm.comparison_groups.stratified_sampling.settings import Settings



class Stratified_Sampling(Comparison_Group_Algorithm):
    def __init__(self, settings: Optional[Settings] = None):
        if settings is None:
            settings = Settings()

        self.settings = settings
        self.sampler = None
        self.bin_selector = None
        self.df_raw = None
        self._diagnostics = None

    def get_comparison_group(self, treatment_data, comparison_pool_data):
        self.treatment_data = treatment_data
        self.comparison_pool_data = comparison_pool_data
        self.treatment_ids = treatment_data.ids
        self.treatment_loadshape = treatment_data.loadshape
        self.comparison_pool_loadshape = comparison_pool_data.loadshape

        treatment_features = self._stratification_features(treatment_data)
        pool_features = self._stratification_features(comparison_pool_data)

        self.sampler = self._build_sampler()
        if self.settings.equivalence_method is None:
            self._sample_fixed_bins(treatment_features, pool_features)
        else:
            self._select_bins_by_equivalence(treatment_features, pool_features)

        clusters, treatment_weights = self._assemble_comparison_group()

        return clusters, treatment_weights

    def _stratification_features(self, data):
        """Stratification feature frame with the id column the sampler bins on."""
        features = data.features.reset_index()

        return features

    def _build_sampler(self):
        """A sampler with one stratification column per settings entry."""
        sampler = StratifiedSampler()
        for column in self.settings.stratification_column:
            sampler.add_column(
                column.column_name,
                n_bins=column.n_bins,
                min_value_allowed=column.min_value_allowed,
                max_value_allowed=column.max_value_allowed,
                fixed_width=column.is_fixed_width,
                auto_bin_require_equivalence=column.auto_bin_equivalence,
            )

        return sampler

    def _sample_fixed_bins(self, treatment_features, pool_features):
        """Fixed-bin path: bin counts come from settings, no equivalence search."""
        self.sampler.fit_and_sample(
            treatment_features,
            pool_features,
            n_samples_approx=self.settings.n_samples_approx,
            relax_n_samples_approx_constraint=self.settings.relax_n_samples_approx_constraint,
            min_n_treatment_per_bin=self.settings.min_n_treatment_per_bin,
            min_n_sampled_to_n_treatment_ratio=self.settings.min_n_sampled_to_n_treatment_ratio,
            random_seed=self.settings.seed,
        )

    def _select_bins_by_equivalence(self, treatment_features, pool_features):
        """Distance-stratified path: search bin counts that maximize load-shape equivalence."""
        equivalence_features = pd.concat([self.treatment_loadshape, self.comparison_pool_loadshape])
        equivalence_features.index.name = "id"

        self.bin_selector = StratifiedSamplingBinSelector(
            self.sampler,
            treatment_features,
            pool_features,
            equivalence_feature_ids=equivalence_features.index,
            equivalence_feature_matrix=equivalence_features,
            df_id_col="id",
            equivalence_method=self.settings.equivalence_method,
            equivalence_quantile_size=self.settings.equivalence_quantile,
            n_samples_approx=self.settings.n_samples_approx,
            relax_n_samples_approx_constraint=self.settings.relax_n_samples_approx_constraint,
            min_n_bins=self.settings.min_n_bins,
            max_n_bins=self.settings.max_n_bins,
            min_n_treatment_per_bin=self.settings.min_n_treatment_per_bin,
            min_n_sampled_to_n_treatment_ratio=self.settings.min_n_sampled_to_n_treatment_ratio,
            random_seed=self.settings.seed,
        )

    def _assemble_comparison_group(self):
        """Map the sampled non-outlier meters into the (clusters, treatment_weights) contract."""
        self.df_raw = self.sampler.data_sample.df
        sampled = self.df_raw[self.df_raw["_outlier_bin"] == False]

        clusters = self._clusters_df(sampled["id"].unique())
        treatment_weights = self._treatment_weights_df(self.treatment_ids)

        self.clusters = clusters
        self.treatment_weights = treatment_weights

        return clusters, treatment_weights

    def _clusters_df(self, ids):
        clusters = pd.DataFrame(ids, columns=["id"])
        clusters["cluster"] = 0
        clusters["weight"] = 1.0

        clusters = clusters.reset_index().set_index("id")
        clusters = clusters[["cluster", "weight"]]

        return clusters

    def _treatment_weights_df(self, ids):
        coeffs = np.ones(len(ids))

        treatment_weights = pd.DataFrame(coeffs, index=ids, columns=["pct_cluster_0"])
        treatment_weights.index.name = "id"

        return treatment_weights

    def diagnostics(self):
        if self.df_raw is None:
            raise RuntimeError("Must run get_comparison_group() before calling diagnostics()")

        if self._diagnostics is None:
            self._diagnostics = StratifiedSamplingDiagnostics(model=self.sampler)

        return self._diagnostics
