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

"""Snapshot regression tests pinning the exact stratified-sampling output.

These exist to guard a structural rewrite: the sampled comparison group and the
per-bin counts must stay identical across any refactor that preserves behavior.
"""

import numpy as np
import pandas as pd
import pytest

from opendsm.comparison_groups.stratified_sampling.model import StratifiedSamplingModel
from opendsm.comparison_groups.stratified_sampling.create_comparison_groups import Stratified_Sampling
from opendsm.comparison_groups.stratified_sampling.settings import (
    DistanceStratifiedSamplingSettings,
    DSS_StratificationColumnSettings,
)



pytestmark = pytest.mark.regression


def _engine_summary(model):
    counts = model.diagnostics().count_bins()
    counts = counts.assign(bin=counts["bin"].astype(str)).set_index("bin")
    sampled = model.data_sample.df
    sampled_ids = sorted(sampled[~sampled["_outlier_bin"]]["id"].unique().tolist())

    return {
        "sampled_ids": sampled_ids,
        "bin_counts": counts[["n_treatment", "n_pool", "n_sampled"]].round(6).to_dict(orient="index"),
    }


def test_engine_single_column_snapshot(df_treatment, df_pool, col_name, snapshot):
    model = StratifiedSamplingModel()
    model.add_column(col_name, n_bins=4)
    model.fit_and_sample(df_treatment, df_pool, n_samples_approx=len(df_treatment), random_seed=1)

    assert _engine_summary(model) == snapshot


def test_engine_multi_column_snapshot(snapshot):
    rng = np.random.default_rng(0)
    df_treatment = pd.DataFrame(
        {"id": [f"t{i}" for i in range(60)], "c1": rng.uniform(0, 100, 60), "c2": rng.uniform(0, 100, 60)}
    )
    df_pool = pd.DataFrame(
        {"id": [f"p{i}" for i in range(600)], "c1": rng.uniform(0, 100, 600), "c2": rng.uniform(0, 100, 600)}
    )
    model = StratifiedSamplingModel()
    model.add_column("c1", n_bins=3)
    model.add_column("c2", n_bins=3)
    model.fit_and_sample(
        df_treatment, df_pool, n_samples_approx=100, random_seed=1,
        min_n_sampled_to_n_treatment_ratio=0, relax_n_samples_approx_constraint=True,
    )

    assert _engine_summary(model) == snapshot


def test_public_dss_equivalence_flow_snapshot(stratified_feature_loadshape_data, snapshot):
    """Pins the distance-stratified (equivalence) path's sampled comparison group."""
    treatment_data, comparison_pool_data = stratified_feature_loadshape_data
    columns = [
        DSS_StratificationColumnSettings(column_name="summer_usage", min_value_allowed=0, max_value_allowed=10000),
        DSS_StratificationColumnSettings(column_name="winter_usage", min_value_allowed=0, max_value_allowed=10000),
    ]
    settings = DistanceStratifiedSamplingSettings(
        seed=42,
        n_samples_approx=100,
        relax_n_samples_approx_constraint=True,
        min_n_sampled_to_n_treatment_ratio=0,
        min_n_bins=1,
        max_n_bins=3,
        stratification_column=columns,
    )

    clusters, _ = Stratified_Sampling(settings).get_comparison_group(treatment_data, comparison_pool_data)

    assert sorted(clusters.index.tolist()) == snapshot
