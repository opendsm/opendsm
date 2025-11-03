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

from opendsm.comparison_groups.stratified_sampling.model import StratifiedSampling, BinnedData
from opendsm.comparison_groups.stratified_sampling.bin_selection import StratifiedSamplingBinSelector
from opendsm.comparison_groups.stratified_sampling.bins import ModelSamplingException



def test_stratified_sampling_fit_and_sample_records_equivalence(
    df_treatment, df_pool,  col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    stratified_sampling_obj = StratifiedSampling()
    df_pool["col2"] = df_pool[col_name]
    df_treatment["col2"] = df_treatment[col_name]
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    ## attempting to estimate both n_bins and n_samples
    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_treatment,
        df_pool,
        
        min_n_bins=4,
        max_n_bins=6,
        random_seed=1,
        equivalence_method='chisquare',
        equivalence_feature_ids = equivalence_feature_ids,
        equivalence_feature_matrix = equivalence_feature_matrix
    )
    output = stratified_sampling_obj.data_sample.df
    bins_df = stratified_sampling_obj.diagnostics().count_bins()


def test_stratified_sampling_fit_and_sample_records_equivalence_too_many_bins(
    df_treatment, df_pool,  col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    stratified_sampling_obj = StratifiedSampling()

    stratified_sampling_obj.add_column(col_name)
    ## attempting to estimate both n_bins and n_samples
    with pytest.raises(ModelSamplingException):
        model_w_selected_bins = StratifiedSamplingBinSelector(stratified_sampling_obj,
            df_treatment,
            df_pool,

            min_n_bins=1000,
            max_n_bins=1002,
            random_seed=1,
            equivalence_method='chisquare',
            relax_n_samples_approx_constraint=False,
            equivalence_feature_ids = equivalence_feature_ids,
            equivalence_feature_matrix = equivalence_feature_matrix
        )


def test_stratified_sampling_fit_and_sample_records_equivalence_idempotent_check(
    df_treatment, df_pool,  col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    df_treatment["col2"] = df_treatment[col_name] * 2
    df_treatment["col3"] = df_treatment[col_name] * 3

    df_pool["col2"] = df_pool[col_name] * 2
    df_pool["col3"] = df_pool[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_treatment,
        df_pool,
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        equivalence_method='chisquare',
        equivalence_feature_ids = equivalence_feature_ids,
        equivalence_feature_matrix = equivalence_feature_matrix
    )
    sample1 = stratified_sampling_obj.data_sample.df.index.values

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")
    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_treatment,
        df_pool,
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        equivalence_method='chisquare',
        equivalence_feature_ids = equivalence_feature_ids,
        equivalence_feature_matrix = equivalence_feature_matrix
    )
    sample2 = stratified_sampling_obj.data_sample.df.index.values
    assert set(sample1) == set(sample2)


def test_stratified_sampling_fit_and_sample_records_equivalence_euclidean_idempotent_check(
    df_treatment, df_pool,  col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    df_treatment["col2"] = df_treatment[col_name] * 2
    df_treatment["col3"] = df_treatment[col_name] * 3

    df_pool["col2"] = df_pool[col_name] * 2
    df_pool["col3"] = df_pool[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_treatment,
        df_pool,
        
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        equivalence_method='euclidean',
        equivalence_feature_ids = equivalence_feature_ids,
        equivalence_feature_matrix = equivalence_feature_matrix
    )
    sample1 = stratified_sampling_obj.data_sample.df.index.values

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")
    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_treatment,
        df_pool,
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        equivalence_method='euclidean',
        equivalence_feature_ids = equivalence_feature_ids,
        equivalence_feature_matrix = equivalence_feature_matrix
    )
    sample2 = stratified_sampling_obj.data_sample.df.index.values
    assert set(sample1) == set(sample2)


def test_stratified_sampling_fit_and_sample_records_equivalence_euclidean_idempotent_check(
    df_treatment, df_pool,  col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    df_treatment["col2"] = df_treatment[col_name] * 2
    df_treatment["col3"] = df_treatment[col_name] * 3

    df_pool["col2"] = df_pool[col_name] * 2
    df_pool["col3"] = df_pool[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_treatment,
        df_pool,
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        equivalence_method='euclidean',
        equivalence_feature_ids = equivalence_feature_ids,
        equivalence_feature_matrix = equivalence_feature_matrix
    )
    sample1 = stratified_sampling_obj.data_sample.df.index.values

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")
    StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_treatment,
        df_pool,
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        equivalence_method='euclidean',
        equivalence_feature_ids = equivalence_feature_ids,
        equivalence_feature_matrix = equivalence_feature_matrix        
    )
    sample2 = stratified_sampling_obj.data_sample.df.index.values
    assert set(sample1) == set(sample2)


def test_plot_records_based_equiv_average(
    df_treatment, df_pool,  col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    df_treatment["col2"] = df_treatment[col_name] * 2
    df_treatment["col3"] = df_treatment[col_name] * 3

    df_pool["col2"] = df_pool[col_name] * 2
    df_pool["col3"] = df_pool[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    bin_selection = StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_treatment,
        df_pool,
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        equivalence_method='euclidean',
        equivalence_feature_ids = equivalence_feature_ids,
        equivalence_feature_matrix = equivalence_feature_matrix
    )
    bin_selection.plot_records_based_equiv_average(plot=False)
    bin_selection.results_as_json()


def test_plot_records_based_equiv_average_chisquare(
    df_treatment, df_pool,  col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    df_treatment["col2"] = df_treatment[col_name] * 2
    df_treatment["col3"] = df_treatment[col_name] * 3

    df_pool["col2"] = df_pool[col_name] * 2
    df_pool["col3"] = df_pool[col_name] * 3

    stratified_sampling_obj = StratifiedSampling()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")
    stratified_sampling_obj.add_column("col3")

    bin_selection = StratifiedSamplingBinSelector(stratified_sampling_obj,
        df_treatment,
        df_pool,
        min_n_bins=2,
        max_n_bins=3,
        random_seed=1,
        equivalence_method='chisquare',
        equivalence_feature_ids = equivalence_feature_ids,
        equivalence_feature_matrix = equivalence_feature_matrix
    )
    bin_selection.plot_records_based_equiv_average(plot=False)
    results = bin_selection.results_as_json()
    assert 'bins_selected_str' in list(results['n_bin_results'][0].keys())
