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

from opendsm.comparison_groups.stratified_sampling.sampling import StratifiedSampler
from opendsm.comparison_groups.stratified_sampling.bin_selection import StratifiedSamplingBinSelector
from opendsm.comparison_groups.stratified_sampling.bins import ModelSamplingException



def _three_column_sampler(df_treatment, df_pool, col_name):
    """A sampler over ``col_name`` and two scaled copies of it, added to both frames."""
    df_treatment["col2"] = df_treatment[col_name] * 2
    df_treatment["col3"] = df_treatment[col_name] * 3
    df_pool["col2"] = df_pool[col_name] * 2
    df_pool["col3"] = df_pool[col_name] * 3
    sampler = StratifiedSampler()
    sampler.add_column(col_name)
    sampler.add_column("col2")
    sampler.add_column("col3")

    return sampler


def _select_bins(sampler, df_treatment, df_pool, method, ids, matrix, **kwargs):
    selector = StratifiedSamplingBinSelector(
        sampler,
        df_treatment,
        df_pool,
        random_seed=1,
        equivalence_method=method,
        equivalence_feature_ids=ids,
        equivalence_feature_matrix=matrix,
        **kwargs,
    )

    return selector


def test_stratified_sampling_fit_and_sample_records_equivalence(
    df_treatment, df_pool, col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    """Estimating both n_bins and n_samples lands within the requested bin range
    and samples every bin."""
    df_pool["col2"] = df_pool[col_name]
    df_treatment["col2"] = df_treatment[col_name]
    sampler = StratifiedSampler()
    sampler.add_column(col_name)
    sampler.add_column("col2")

    _select_bins(
        sampler, df_treatment, df_pool, "chisquare", equivalence_feature_ids,
        equivalence_feature_matrix, min_n_bins=4, max_n_bins=6,
    )
    output = sampler.data_sample.df
    bins_df = sampler.diagnostics().count_bins()

    assert not output.empty
    assert 4 <= sampler.data_sample.df["_bin_label"].nunique() <= 6
    assert (bins_df["n_sampled"] > 0).all()


def test_stratified_sampling_fit_and_sample_records_equivalence_too_many_bins(
    df_treatment, df_pool,  col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    stratified_sampling_obj = StratifiedSampler()

    stratified_sampling_obj.add_column(col_name)
    ## attempting to estimate both n_bins and n_samples
    with pytest.raises(ModelSamplingException):
        StratifiedSamplingBinSelector(stratified_sampling_obj,
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


@pytest.mark.slow
@pytest.mark.parametrize("method", ["chisquare", "euclidean"])
def test_bin_selection_is_idempotent_under_a_seed(
    method, df_treatment, df_pool, col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    samples = []

    for _ in range(2):
        sampler = _three_column_sampler(df_treatment, df_pool, col_name)
        _select_bins(
            sampler, df_treatment, df_pool, method, equivalence_feature_ids,
            equivalence_feature_matrix, min_n_bins=2, max_n_bins=3,
        )
        samples.append(set(sampler.data_sample.df.index.values))

    assert samples[0] == samples[1]


@pytest.mark.parametrize("method", ["chisquare", "euclidean"])
def test_records_based_equivalence_results_name_the_selected_bins(
    method, df_treatment, df_pool, col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    sampler = _three_column_sampler(df_treatment, df_pool, col_name)
    selector = _select_bins(
        sampler, df_treatment, df_pool, method, equivalence_feature_ids,
        equivalence_feature_matrix, min_n_bins=2, max_n_bins=3,
    )

    selector.plot_records_based_equiv_average(plot=False)
    results = selector.results_as_json()

    assert "bins_selected_str" in results["n_bin_results"][0]
    assert results["bins_selected"] in {r["bins_selected_str"] for r in results["n_bin_results"]}


def test_selected_bin_minimizes_distance(
    df_treatment, df_pool, col_name, equivalence_feature_ids, equivalence_feature_matrix
):
    """The selector returns the bin configuration with the smallest treatment-
    comparison distance among all non-disqualified options (optimality)."""
    df_treatment["col2"] = df_treatment[col_name] * 2
    df_pool["col2"] = df_pool[col_name] * 2

    stratified_sampling_obj = StratifiedSampler()
    stratified_sampling_obj.add_column(col_name)
    stratified_sampling_obj.add_column("col2")

    bin_selection = StratifiedSamplingBinSelector(
        stratified_sampling_obj, df_treatment, df_pool,
        min_n_bins=2, max_n_bins=4, random_seed=1,
        equivalence_method="euclidean",
        equivalence_feature_ids=equivalence_feature_ids,
        equivalence_feature_matrix=equivalence_feature_matrix,
    )

    results = bin_selection.results_as_json()
    selected = results["bins_selected"]
    scored = [r for r in results["n_bin_results"] if r["distance"] is not None]

    selected_distance = next(r["distance"] for r in scored if r["bins_selected_str"] == selected)
    assert selected_distance == min(r["distance"] for r in scored)
