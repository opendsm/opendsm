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

from opendsm.comparison_groups.stratified_sampling.model import StratifiedSampling, BinnedData
from opendsm.comparison_groups.stratified_sampling.bins import ModelSamplingException


def test_stratified_sampling_fit_and_sample():
    stratified_sampling_obj = StratifiedSampling()
    df_treatment = pd.DataFrame([{"id": f"id_{x}", "col1": x} for x in range(0, 10)])
    df_pool = pd.DataFrame([{"id": f"id_{x}", "col1": x / 2.0} for x in range(0, 1000)])
    stratified_sampling_obj.add_column("col1")

    stratified_sampling_obj.fit_and_sample(
        df_treatment,
        df_pool,
        n_samples_approx=10,
        random_seed=1,
        min_n_sampled_to_n_treatment_ratio=None,
    )
    sample1 = stratified_sampling_obj.data_sample.df.index.values

    stratified_sampling_obj.fit_and_sample(
        df_treatment,
        df_pool,
        n_samples_approx=10,
        random_seed=1,
        min_n_sampled_to_n_treatment_ratio=None,
    )
    sample2 = stratified_sampling_obj.data_sample.df.index.values
    assert set(sample1) == set(sample2)

    stratified_sampling_obj.fit_and_sample(
        df_treatment,
        df_pool,
        n_samples_approx=10,
        random_seed=1,
        min_n_sampled_to_n_treatment_ratio=None,
    )
    sample1 = stratified_sampling_obj.data_sample.df.index.values

    stratified_sampling_obj.fit_and_sample(
        df_treatment,
        df_pool,
        n_samples_approx=10,
        random_seed=5,
        min_n_sampled_to_n_treatment_ratio=None,
    )
    sample2 = stratified_sampling_obj.data_sample.df.index.values
    assert set(sample1) != set(sample2)


def test_stratified_sampling_fit_and_sample_random_seed_check():
    # perturb was returning different values since it was writing over the
    # df rather than using a copy
    df_comparison = pd.DataFrame(
        [
            {
                "id": f"id-{x}",
                "baseline_annual_kwh": np.random.random() * 10000,
                "baseline_bd_pct_heating_load": np.random.random(),
            }
            for x in range(0, 200000)
        ]
    )
    df_treatment = pd.DataFrame(
        [
            {
                "id": f"id-{x}",
                "baseline_annual_kwh": np.random.random() * 10000,
                "baseline_bd_pct_heating_load": np.random.random(),
            }
            for x in range(0, 500)
        ]
    )
    n_samples_approx = 500
    random_seed = 1
    stratification_params = ["baseline_annual_kwh", "baseline_bd_pct_heating_load"]

    model = StratifiedSampling(
        treatment_label="treatment", pool_label="comparison", output_name="control"
    )
    [model.add_column(col) for col in stratification_params]

    model.fit(df_treatment, min_n_treatment_per_bin=0)
    model.sample(
        df_comparison, n_samples_approx=n_samples_approx, random_seed=random_seed
    )

    for run_num in range(0, 10):
        model_temp = StratifiedSampling(
            treatment_label="treatment", pool_label="comparison", output_name="control"
        )
        [model_temp.add_column(col) for col in stratification_params]
        model_temp.fit(df_treatment, min_n_treatment_per_bin=0)
        model_temp.sample(
            df_comparison, n_samples_approx=n_samples_approx, random_seed=random_seed
        )
        pd.testing.assert_frame_equal(
            model_temp.data_sample.df[stratification_params + ["id"]],
            model.data_sample.df[stratification_params + ["id"]],
        )
        assert (
            len(
                set(model_temp.data_sample.df["id"].values)
                - set(model.data_sample.df["id"].values)
            )
            == 0
        )


@pytest.fixture
def stratified_sampling_obj():
    return StratifiedSampling()


def test_stratified_sampling_fit_and_sample_min_allowed_max_allowed(
    stratified_sampling_obj
):
    col_name = "col1"
    min_value_allowed = 5
    max_value_allowed = 8
    df_treatment = pd.DataFrame([{"id": f"id_{x}", col_name: x} for x in range(0, 10)])
    df_pool = pd.DataFrame(
        [{"id": f"id_{x}", col_name: x} for x in np.arange(0, 20, 0.1)]
    )
    stratified_sampling_obj.add_column(
        col_name,
        min_value_allowed=min_value_allowed,
        max_value_allowed=max_value_allowed,
    )

    stratified_sampling_obj.fit_and_sample(
        df_treatment,
        df_pool,
        n_samples_approx=4,
        random_seed=1,
        min_n_sampled_to_n_treatment_ratio=None,
    )
    output = stratified_sampling_obj.data_sample.df[col_name].values
    assert min(output) > min_value_allowed
    assert max(output) < max_value_allowed


def test_stratified_sampling_fit_and_sample_n_samples_approx_limit(
    df_treatment, df_pool, col_name
):
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)

    n_samples_approx = 40
    stratified_sampling_obj.fit_and_sample(
        df_treatment, df_pool, n_samples_approx=n_samples_approx, random_seed=1
    )
    output = stratified_sampling_obj.data_sample.df
    assert output["_bin_label"].nunique() == 2
    bins_df = stratified_sampling_obj.diagnostics().count_bins()
    assert (bins_df["n_sampled"] / bins_df["n_pct_sampled"]).round() == n_samples_approx


def test_stratified_sampling_fit_and_sample_n_samples_approx_limit(
    df_treatment, df_pool, col_name
):
    stratified_sampling_obj = StratifiedSampling()
    col_name = "col1"
    df_treatment = pd.DataFrame(
        [
            {"id": f"id_{x}", col_name: x}
            for x in (
                list(np.arange(0, 2, 0.1))
                + list(np.arange(2, 4, 0.5))
                + list(np.arange(4, 6, 1))
                + list(np.arange(6, 10, 0.2))
            )
        ]
    )
    df_pool = pd.DataFrame(
        [{"id": f"id_{x}", col_name: x} for x in np.arange(0, 20, 0.01)]
    )
    stratified_sampling_obj.add_column(col_name)

    n_samples_approx = 40
    stratified_sampling_obj.fit_and_sample(
        df_treatment,
        df_pool,
        n_samples_approx=n_samples_approx,
        random_seed=1,
        min_n_sampled_to_n_treatment_ratio=None,
    )
    output = stratified_sampling_obj.data_sample.df
    assert output["_bin_label"].nunique() == 2
    bins_df = stratified_sampling_obj.diagnostics().count_bins()
    assert abs(len(output) - n_samples_approx) <= 1


def test_stratified_sampling_fit_and_sample_n_samples_approx_variations(
    df_treatment, df_pool, col_name
):
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    ## attempting to estimate both n_bins and n_samples
    stratified_sampling_obj.fit_and_sample(df_treatment, df_pool, random_seed=1)
    output = stratified_sampling_obj.data_sample.df
    bins_df = stratified_sampling_obj.diagnostics().count_bins()
    assert len(bins_df) == 3

    ## enforcing 1 bin
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name, n_bins=1)
    stratified_sampling_obj.fit_and_sample(df_treatment, df_pool, random_seed=1)
    output = stratified_sampling_obj.data_sample.df
    bins_df = stratified_sampling_obj.diagnostics().count_bins()

    ## enforcing 4 bins
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name, n_bins=4)
    stratified_sampling_obj.fit_and_sample(df_treatment, df_pool, random_seed=1)
    output = stratified_sampling_obj.data_sample.df
    bins_df = stratified_sampling_obj.diagnostics().count_bins()
    assert len(bins_df) == 4

    ## enforcing n_samples_approx=40
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.fit_and_sample(
        df_treatment,
        df_pool,
        n_samples_approx=40,
        random_seed=1,
        min_n_sampled_to_n_treatment_ratio=None,
    )
    output = stratified_sampling_obj.data_sample.df
    bins_df = stratified_sampling_obj.diagnostics().count_bins()
    # should be within 1 of n_samples_approx
    assert abs(len(output) - 40) <= 1


def test_stratified_sampling_fit_and_sample_too_many_bins(df_treatment, df_pool, col_name):
    df_treatment["col2"] = df_treatment[col_name].astype(int)
    df_pool["col2"] = df_pool[col_name].astype(int)
    df_treatment["col3"] = df_treatment[col_name].astype(int) * 2
    df_pool["col3"] = df_pool[col_name].astype(int) / 2
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")
    ## attempting to estimate both n_bins and n_samples
    with pytest.raises(ValueError):
        stratified_sampling_obj.fit_and_sample(df_treatment, df_pool, random_seed=1)


def test_stratified_sampling_fit_and_sample_dont_require_equivalence(
    df_treatment, df_pool, col_name
):
    df_treatment["col2"] = df_treatment[col_name].astype(int)
    df_pool["col2"] = df_pool[col_name].astype(int)
    df_treatment["col3"] = df_treatment[col_name].astype(int) * 2
    df_pool["col3"] = df_pool[col_name].astype(int) / 2
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3", auto_bin_require_equivalence=False)
    ## attempting to estimate both n_bins and n_samples
    stratified_sampling_obj.fit_and_sample(df_treatment, df_pool, random_seed=1)
    output = stratified_sampling_obj.data_sample.df
    bins_df = stratified_sampling_obj.diagnostics().count_bins()
    assert not output.empty


def test_stratified_sampling_fit_and_sample_upper_limit_n_samples_approx(
    df_treatment, df_pool, col_name
):
    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    ## attempting to estimate both n_bins and n_samples
    with pytest.raises(ModelSamplingException):
        stratified_sampling_obj.fit_and_sample(
            df_treatment, df_pool, random_seed=1, n_samples_approx=1000
        )
    stratified_sampling_obj.fit_and_sample(
        df_treatment,
        df_pool,
        random_seed=1,
        n_samples_approx=1000,
        relax_n_samples_approx_constraint=True,
    )
    output = stratified_sampling_obj.data_sample.df
    bins_df = stratified_sampling_obj.diagnostics().count_bins()
    assert not output.empty
