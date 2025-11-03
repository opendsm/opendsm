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

import random

import pandas as pd

from opendsm.comparison_groups.individual_meter_matching.settings import Settings
from opendsm.comparison_groups.individual_meter_matching.distance_calc_selection import DistanceMatching


def generate_group(n_entries, make_random=True, non_random_value=5, id_prefix="t"):
    return pd.DataFrame(
        [
            {
                "id": f"{id_prefix}_{i}",
                "month_1": random.random() if make_random else non_random_value,
                "month_2": random.random() if make_random else non_random_value,
                "month_3": random.random() if make_random else non_random_value,
            }
            for i in range(1, n_entries + 1)
        ]
    ).set_index("id")


def test_distance_match():
    random.seed(1)
    n_treatment = 10
    n_pool = 100
    n_matches_per_treatment = 4
    allow_duplicate_matches = False

    comparison_pool = pd.DataFrame(
        [
            {
                "id": f"c_{i}",
                "month_1": random.random(),
                "month_2": random.random(),
                "month_3": random.random(),
            }
            for i in range(1, n_pool + 1)
        ]
    ).set_index("id")
    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=True, id_prefix="c")
    for selection_method in ["minimize_meter_distance", "minimize_loadshape_distance"]:
        settings = Settings(
            selection_method=selection_method,
            n_matches_per_treatment=n_matches_per_treatment,
            allow_duplicate_matches=allow_duplicate_matches,
        )
        IMM = DistanceMatching(
            settings=settings
        )
        
        comparison_group = IMM.get_comparison_group(
            treatment_group=treatment_group,
            comparison_pool=comparison_pool
        )
        assert not comparison_group.empty


def test_distance_match_duplicates_allowed():
    random.seed(1)
    n_treatment = 10
    n_pool = 5
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = True
    n_matches_per_treatment = 1

    # this will run out of comparison pool meters and therefore still have duplicates
    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=False, id_prefix="c")
    settings = Settings(
        selection_method=selection_method,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    assert comparison_group["duplicated"].any()


def test_distance_match_duplicates_forbidden():
    random.seed(1)
    n_treatment = 8
    n_pool = 10
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = False
    n_matches_per_treatment = 1

    # this will run through the 'duplicates' loop several times before finding unique values
    # however since here are more 'max runs allowed' than treatment meters, it will be
    # able to iterate enough times to find unique matches

    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=False)
    settings = Settings(
        selection_method=selection_method,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    assert not comparison_group["duplicated"].any()


def test_distance_match_large_treatments():
    random.seed(1)

    n_treatment = 10000
    n_pool = 20000
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = False
    n_matches_per_treatment = 1
    n_treatments_per_chunk = 5000

    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=True, id_prefix="c")
    settings = Settings(
        selection_method=selection_method,
        n_treatments_per_chunk=n_treatments_per_chunk,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    assert not comparison_group.empty


def test_distance_duplicate_best_match():
    n_treatment = 2
    n_pool = 2
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = False
    n_matches_per_treatment = 1

    t_ids = ["far", "close"]
    t_vals = [10, 1]
    c_ids = ["match_1", "match_2"]
    c_vals = [2, 100]
    treatment_group = pd.DataFrame({"id": t_ids, "month_1": t_vals}).set_index("id")
    comparison_pool = pd.DataFrame({"id": c_ids, "month_1": c_vals}).set_index("id")

    settings = Settings(
        selection_method=selection_method,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    comparison_group.set_index("id", inplace=True)

    assert comparison_group.loc["match_1", "treatment"] == "close"
    assert comparison_group.loc["match_2", "treatment"] == "far"


def test_multiple_meter_matches():
    random.seed(1)

    n_treatment = 8
    n_pool = 2000
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = False
    n_matches_per_treatment = 5

    # this will run through the 'duplicates' loop several times before finding unique values
    # however since here are more 'max runs allowed' than treatment meters, it will be
    # able to iterate enough times to find unique matches

    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=True)
    settings = Settings(
        selection_method=selection_method,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    assert not comparison_group["duplicated"].any()
    assert len(comparison_group) == 40
    assert comparison_group.index.nunique() == 40
    assert comparison_group.treatment.value_counts().nunique() == 1
