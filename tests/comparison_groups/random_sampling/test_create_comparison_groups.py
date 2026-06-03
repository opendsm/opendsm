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

import pytest

from opendsm.comparison_groups.random_sampling.create_comparison_groups import Random_Sampling
from opendsm.comparison_groups.random_sampling.settings import Settings



def test_n_meters_total_samples_exact_count(cg_loadshape_data):
    treatment_data, comparison_pool_data = cg_loadshape_data
    rs = Random_Sampling(Settings(n_meters_total=10, n_meters_per_treatment=None, seed=1))

    clusters, treatment_weights = rs.get_comparison_group(treatment_data, comparison_pool_data)

    assert len(clusters) == 10
    assert (clusters["cluster"] == 0).all()
    assert (clusters["weight"] == 1.0).all()
    assert len(treatment_weights) == len(treatment_data.ids)
    assert rs.comparison_pool_loadshape is not None
    assert rs.treatment_loadshape is not None


def test_n_meters_per_treatment_scales_with_treatment_count(cg_loadshape_data):
    treatment_data, comparison_pool_data = cg_loadshape_data
    n_per = 2
    rs = Random_Sampling(Settings(n_meters_per_treatment=n_per, seed=1))

    clusters, _ = rs.get_comparison_group(treatment_data, comparison_pool_data)

    assert len(clusters) == n_per * len(treatment_data.ids)


def test_seed_makes_sample_reproducible(cg_loadshape_data):
    treatment_data, comparison_pool_data = cg_loadshape_data
    settings = Settings(n_meters_total=10, n_meters_per_treatment=None, seed=7)

    clusters_1, _ = Random_Sampling(settings).get_comparison_group(treatment_data, comparison_pool_data)
    clusters_2, _ = Random_Sampling(settings).get_comparison_group(treatment_data, comparison_pool_data)

    assert list(clusters_1.index) == list(clusters_2.index)


def test_requires_one_of_n_meters_total_or_per_treatment():
    with pytest.raises(ValueError):
        Settings(n_meters_total=None, n_meters_per_treatment=None)


def test_rejects_both_n_meters_total_and_per_treatment():
    with pytest.raises(ValueError):
        Settings(n_meters_total=10, n_meters_per_treatment=4)


def test_oversampling_pool_raises(cg_loadshape_data):
    treatment_data, comparison_pool_data = cg_loadshape_data
    rs = Random_Sampling(Settings(n_meters_total=10_000, n_meters_per_treatment=None, seed=1))

    with pytest.raises(ValueError):
        rs.get_comparison_group(treatment_data, comparison_pool_data)
