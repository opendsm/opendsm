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

from opendsm.comparison_groups.individual_meter_matching.create_comparison_groups import (
    Individual_Meter_Matching,
)
from opendsm.comparison_groups.individual_meter_matching.settings import Settings
from opendsm.comparison_groups.individual_meter_matching.highs_settings import HiGHS_Settings


def test_highs_settings_defaults_construct():
    settings = HiGHS_Settings()

    assert settings.presolve in ("off", "choose", "on")


def test_highs_settings_rejects_invalid_literal():
    with pytest.raises(ValueError):
        HiGHS_Settings(presolve="maybe")


def test_get_comparison_group_returns_clusters_and_weights(cg_loadshape_data):
    treatment_data, comparison_pool_data = cg_loadshape_data
    imm = Individual_Meter_Matching(Settings(n_matches_per_treatment=4))

    clusters, treatment_weights = imm.get_comparison_group(treatment_data, comparison_pool_data)

    assert {"treatment", "distance", "duplicated", "cluster", "weight"}.issubset(clusters.columns)
    assert len(treatment_weights) == len(treatment_data.ids)
    assert imm.treatment_loadshape is not None
    assert imm.comparison_pool_loadshape is not None


def test_no_duplicates_assigns_unique_pool_meters(cg_loadshape_data):
    treatment_data, comparison_pool_data = cg_loadshape_data
    imm = Individual_Meter_Matching(
        Settings(n_matches_per_treatment=4, allow_duplicate_matches=False)
    )

    clusters, _ = imm.get_comparison_group(treatment_data, comparison_pool_data)

    assert not clusters["duplicated"].any()


def test_duplicates_allowed_matches_full_count(cg_loadshape_data):
    treatment_data, comparison_pool_data = cg_loadshape_data
    n_match = 4
    imm = Individual_Meter_Matching(
        Settings(n_matches_per_treatment=n_match, allow_duplicate_matches=True)
    )

    clusters, _ = imm.get_comparison_group(treatment_data, comparison_pool_data)

    assert len(clusters) == n_match * len(treatment_data.ids)


def test_matching_is_deterministic(cg_loadshape_data):
    treatment_data, comparison_pool_data = cg_loadshape_data
    settings = Settings(n_matches_per_treatment=4)

    clusters_1, _ = Individual_Meter_Matching(settings).get_comparison_group(
        treatment_data, comparison_pool_data
    )
    clusters_2, _ = Individual_Meter_Matching(settings).get_comparison_group(
        treatment_data, comparison_pool_data
    )

    assert list(clusters_1.index) == list(clusters_2.index)


def test_minimize_loadshape_distance_path(cg_loadshape_data):
    treatment_data, comparison_pool_data = cg_loadshape_data
    imm = Individual_Meter_Matching(
        Settings(
            selection_method="minimize_loadshape_distance",
            n_matches_per_treatment=2,
            allow_duplicate_matches=False,
        )
    )

    clusters, _ = imm.get_comparison_group(treatment_data, comparison_pool_data)

    assert not clusters.empty


def test_allow_duplicates_requires_meter_distance():
    with pytest.raises(ValueError):
        Settings(allow_duplicate_matches=True, selection_method="minimize_loadshape_distance")
