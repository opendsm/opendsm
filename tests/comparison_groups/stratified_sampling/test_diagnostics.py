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

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from opendsm.comparison_groups.stratified_sampling.sampling import StratifiedSampler



@pytest.fixture
def diagnostics_obj_2d():
    """Two-column diagnostics so scatter (which needs column pairs) is exercised."""
    rng = np.random.default_rng(0)
    df_treatment = pd.DataFrame(
        {"id": [f"t{i}" for i in range(60)], "c1": rng.uniform(0, 100, 60), "c2": rng.uniform(0, 100, 60)}
    )
    df_pool = pd.DataFrame(
        {"id": [f"p{i}" for i in range(600)], "c1": rng.uniform(0, 100, 600), "c2": rng.uniform(0, 100, 600)}
    )
    model = StratifiedSampler()
    model.add_column("c1", n_bins=3)
    model.add_column("c2", n_bins=3)
    model.fit_and_sample(
        df_treatment, df_pool, n_samples_approx=100, random_seed=1,
        min_n_sampled_to_n_treatment_ratio=0, relax_n_samples_approx_constraint=True,
    )
    diagnostics = model.diagnostics()

    return diagnostics


@pytest.fixture
def diagnostics_obj(df_treatment, df_pool, col_name):
    stratified_sampling_obj = StratifiedSampler()
    stratified_sampling_obj.add_column(col_name, n_bins=4)
    stratified_sampling_obj.fit_and_sample(
        df_treatment, df_pool, n_samples_approx=len(df_treatment), random_seed=1
    )
    diagnostics = stratified_sampling_obj.diagnostics()

    return diagnostics


def test_equivalence(diagnostics_obj):
    equivalence = diagnostics_obj.equivalence()
    assert equivalence["ks_ok"].all() == True and equivalence["t_ok"].all() == True


def test_count_bins_returns_populated_frame(diagnostics_obj):
    counts = diagnostics_obj.count_bins()

    assert not counts.empty


def test_equivalence_passed_returns_bool(diagnostics_obj):
    passed = diagnostics_obj.equivalence_passed()

    assert isinstance(passed, (bool, np.bool_))


def test_histogram_builds_matplotlib_figure_per_column(diagnostics_obj):
    plots = diagnostics_obj.histogram()

    assert len(plots) >= 1
    assert all(isinstance(plot, Figure) for plot in plots)


def test_scatter_builds_matplotlib_figure_per_column_pair(diagnostics_obj_2d):
    plots = diagnostics_obj_2d.scatter()

    assert len(plots) == 1  # a single (c1, c2) pair
    assert isinstance(plots[0], Figure)


def test_quantile_equivalence_builds_matplotlib_figure(diagnostics_obj):
    figure = diagnostics_obj.quantile_equivalence()

    assert isinstance(figure, Figure)


def test_n_sampled_to_n_treatment_ratio_is_not_floored(diagnostics_obj):
    """Regression: the ratio must stay a float. Flooring it to int broke
    comparisons against fractional thresholds (e.g. the DSS 0.25 default)."""
    ratio = diagnostics_obj.n_sampled_to_n_treatment_ratio()

    assert isinstance(ratio, (float, np.floating))
