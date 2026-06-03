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

import numpy as np
import pytest

from opendsm.comparison_groups.stratified_sampling.create_comparison_groups import Stratified_Sampling
from opendsm.comparison_groups.stratified_sampling.settings import (
    StratifiedSamplingSettings,
    StratificationColumnSettings,
)


def _settings(seed=42):
    columns = [
        StratificationColumnSettings(
            column_name="summer_usage", n_bins=3, min_value_allowed=0, max_value_allowed=10000
        ),
        StratificationColumnSettings(
            column_name="winter_usage", n_bins=3, min_value_allowed=0, max_value_allowed=10000
        ),
    ]
    settings = StratifiedSamplingSettings(
        seed=seed,
        n_samples_approx=100,
        relax_n_samples_approx_constraint=True,
        min_n_sampled_to_n_treatment_ratio=0,
        stratification_column=columns,
    )

    return settings


def test_get_comparison_group_returns_clusters_and_weights(stratified_feature_loadshape_data):
    treatment_data, comparison_pool_data = stratified_feature_loadshape_data
    sampler = Stratified_Sampling(_settings())

    clusters, treatment_weights = sampler.get_comparison_group(treatment_data, comparison_pool_data)

    assert not clusters.empty
    assert (clusters["cluster"] == 0).all()
    assert len(treatment_weights) == len(treatment_data.ids)


def test_default_path_sets_loadshape_attributes(stratified_feature_loadshape_data):
    """Regression: with equivalence_method None (the non-DSS path) the loadshape
    attributes must be populated so get_loadshapes() works. They were previously
    left unset on this branch, crashing downstream loadshape access."""
    treatment_data, comparison_pool_data = stratified_feature_loadshape_data
    sampler = Stratified_Sampling(_settings())

    sampler.get_comparison_group(treatment_data, comparison_pool_data)

    assert sampler.treatment_loadshape is not None
    assert sampler.comparison_pool_loadshape is not None
    assert sampler.treatment_ids is not None

    loadshapes = sampler.get_loadshapes()
    assert len(loadshapes) == 3


def test_diagnostics_ratio_is_float(stratified_feature_loadshape_data):
    treatment_data, comparison_pool_data = stratified_feature_loadshape_data
    sampler = Stratified_Sampling(_settings())
    sampler.get_comparison_group(treatment_data, comparison_pool_data)

    ratio = sampler.diagnostics().n_sampled_to_n_treatment_ratio()

    assert isinstance(ratio, (float, np.floating))


def test_seed_makes_sampling_reproducible(stratified_feature_loadshape_data):
    treatment_data, comparison_pool_data = stratified_feature_loadshape_data

    clusters_1, _ = Stratified_Sampling(_settings(seed=42)).get_comparison_group(
        treatment_data, comparison_pool_data
    )
    clusters_2, _ = Stratified_Sampling(_settings(seed=42)).get_comparison_group(
        treatment_data, comparison_pool_data
    )

    assert sorted(clusters_1.index) == sorted(clusters_2.index)


def test_sampled_meters_snapshot(stratified_feature_loadshape_data, snapshot):
    """Pin the seeded sample: count and the sorted sampled pool ids."""
    treatment_data, comparison_pool_data = stratified_feature_loadshape_data
    clusters, _ = Stratified_Sampling(_settings(seed=42)).get_comparison_group(
        treatment_data, comparison_pool_data
    )

    summary = {
        "n_sampled": int(len(clusters)),
        "sampled_ids": sorted(clusters.index.tolist()),
    }

    assert summary == snapshot
