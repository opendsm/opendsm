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

import pandas as pd

from opendsm.comparison_groups.stratified_sampling.model import StratifiedSampling


@pytest.fixture
def diagnostics_obj(df_treatment, df_pool, col_name):
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name, n_bins=4)
    stratified_sampling_obj.fit_and_sample(
        df_treatment, df_pool, n_samples_approx=len(df_treatment), random_seed=1
    )
    return stratified_sampling_obj.diagnostics()


def test_equivalence(diagnostics_obj):
    equivalence = diagnostics_obj.equivalence()
    assert equivalence["ks_ok"].all() == True and equivalence["t_ok"].all() == True
