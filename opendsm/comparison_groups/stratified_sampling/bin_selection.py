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

import copy
import itertools
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from . import equivalence

logger = logging.getLogger(__name__)

__all__ = ("StratifiedSamplingBinSelector",)


class StratifiedSamplingBinSelector:
    def __init__(
        self,
        model,
        df_treatment,
        df_pool,
        equivalence_feature_ids,
        equivalence_feature_matrix,
        equivalence_method="chisquare",
        df_id_col="id",
        n_samples_approx=5000,
        min_n_treatment_per_bin=0,
        random_seed=1,
        min_n_sampled_to_n_treatment_ratio=0.25,
        min_n_bins=1,
        max_n_bins=8,
        equivalence_quantile_size=25,
        relax_n_samples_approx_constraint=True,
    ):
        """
        Finds an optimal stratified sampling bin configuration which minimizes
        distance between treatmnt and comparison groups.  A bin configuration
        is a number of bins, `n_c`, for each stratification column `c`, where
        c is an integer between `min_n_bins` and `max_n_bin` inclusive.
        Using a grid search, all possible bin configurations will be constructed and
        tested, and the confiuration which minimizes treatment-comparison distance
        will be returned.  Distance is measured on a set of features
        provided in `df_for_equivalence` in 'long' format, i.e. multiple rows
        per meter, one column holds feature name, one column holds feature value.

        Distance is computed as follows. First cut treatment and comparison groups
        into quantiles, with the number of quantiles chosen such that the
        treatment group has quantiles of size `equivalence_quantile_size`.
        Then compute the distance between each treatment-comparison quantile pair
        according to the method in `equivalence_method`, either `euclidean` or
        `chisquare` distance; then sum the distances across all quantiles.  For
        example, if the treatment group is size 1000 and `equivalence_quantile_size`
        is 100, then treatment and comparison groups will each be cut into ten quantiles,
        and ten distances will be computed and summed.

        Example usage:

            m = StratifiedSamplingModel()
            m.add_column('annual_usage', min_value=0, max_value=20000)
            m.add_column('summer_usage', min_value=0, max_value=1000)
            s = StratifiedSamplingBinSelector(m, df_treatment, df_pool,
                equivalence_feature_ids, equivalence_feature_matrix, equivalence_method="chisquare")
            results = s.results_as_json()
            df_comparison = m.data_sample.df


        Attributes
        ==========
        model: eemeter.gridmeter.StratifiedSamplingModel
            Model with stratification columns added.
        df_treatment: pandas.DataFrame
            dataframe to use for constructing the stratified sampling bins.
        df_pool: pandas.DataFrame
            dataframe to sample from according to the constructed stratified sampling bins.
        df_for_equivalence: pandas.DataFrame
            dataframe with featues to use for computing equivalence, in 'long' form
        equivalence_feature_ids: str
            Array of meter IDs which maps to row indices in equivalence_feature_matrix.
        equivalence_feature_matrix: pandas.DataFrame or numpy.ndarray
            dataframe or array with featues to use for computing equivalence, in 'wide' form,
            i.e. one row per meter, one column per feature.  Must contain only
            numeric values, no ID column.
        equivalence_method: str
            Method for computing distance -- either 'euclidean' or 'chisquare'.
        df_id_col: str
            Name of column in df_treatment and df_pool which contains meter ID.
        n_samples_aprox: int
            approximate number of total samples from df_pool which are used to construct
            the comparison group. It is approximate because
            there may be some slight discrepencies around the total count to ensure
            that each bin has the correct percentage of the total.
            A None value means that it will take as many samples as it has available.
        min_n_treatment_per_bin: int
            Minimum number of treatment samples that must exist in a given bin for
            it to be considered a non-outlier bin (only applicable if there are
            cols with fixed_width=True)
        min_n_sampled_to_n_treatment_ratio: float
            Minimum number samples that must exist in each bin per treatment datapoint in that bin.
        min_n_bins: int
            Minimum number of bins to use in stratified sampling.
        max_n_bins: int
            Maximum number of bins to use in stratified sampling.
        equivalence_quantile_size: int
            Number of samples per quantile when computing distances quantile-by-quantile.
        relax_n_samples_approx_constraint: bool
            If True, treats n_samples_approx as an upper bound, but gets as many comparison group
            meters as available up to n_samples_approx. If False, it raises an exception
            if there are not enough comparison pool meters to reach n_samples_approx.
        """
        # Settings
        self.n_samples_approx = n_samples_approx
        self.min_n_treatment_per_bin = min_n_treatment_per_bin
        self.random_seed = random_seed
        self.min_n_sampled_to_n_treatment_ratio = min_n_sampled_to_n_treatment_ratio
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.df_id_col = df_id_col
        self.equivalence_feature_ids = equivalence_feature_ids
        self.equivalence_feature_matrix = equivalence_feature_matrix
        self.equivalence_method = equivalence_method
        self.equivalence_quantile_size = equivalence_quantile_size

        self.model = model
        self.df_treatment = df_treatment
        self.df_pool = df_pool
        self.n_bin_options_df = None
        self.equiv_treatment = None
        self.equiv_samples = []

        if len(self.model.columns) == 0:
            raise ValueError("You must add at least one column before fitting.")
        if any([not col["auto_bin"] for name, col in self.model.columns.items()]):
            raise ValueError("This form of fitting only works n_bins is not set")
        logger.debug(self.model.columns)
        min_distance = float("Inf")
        min_columns = None

        column_names = list(self.model.columns.keys())
        n_bin_results = []
        self.n_bin_options_df = pd.DataFrame(
            [
                {column_names[i - 1]: c for i, c in enumerate(comb)}
                for comb in itertools.product(
                    range(min_n_bins, max_n_bins + 1), repeat=len(column_names)
                )
            ]
        )
        disqualified_n_bin_options = []
        for n_bin_option in self.n_bin_options_df.to_dict("records"):
            [
                self.model.set_n_bins(name, n_bins)
                for name, n_bins in n_bin_option.items()
            ]
            bins_selected_str = self.model.get_all_n_bins_as_str()

            if n_bin_option in disqualified_n_bin_options:
                logger.debug(f"Skipping {bins_selected_str} (disqualified)")
                continue

            self.model.fit(
                self.df_treatment,
                min_n_treatment_per_bin=min_n_treatment_per_bin,
                random_seed=random_seed,
            )

            self.model.sample(
                self.df_pool,
                n_samples_approx=n_samples_approx,
                random_seed=random_seed,
                relax_n_samples_approx_constraint=relax_n_samples_approx_constraint,
            )
            n_sampled_to_n_treatment_ratio = (
                self.model.diagnostics().n_sampled_to_n_treatment_ratio()
            )
            if (
                not self.model.relax_ratio_constraint
                and n_sampled_to_n_treatment_ratio < min_n_sampled_to_n_treatment_ratio
            ):
                logger.info(
                    f"Insufficient pool data for {bins_selected_str}:"
                    f"found {n_sampled_to_n_treatment_ratio}:1 but need "
                    f"{min_n_sampled_to_n_treatment_ratio}:1."
                )
                disqualified_options = self.n_bin_options_df.loc[
                    (
                        self.n_bin_options_df[list(n_bin_option)]
                        >= pd.Series(n_bin_option)
                    ).all(axis=1)
                ].to_dict("records")
                disqualified_n_bin_options.extend(disqualified_options)
                n_bin_results.append(
                    dict(
                        **n_bin_option,
                        **{
                            "distance": None,
                            "status": "FAILED",
                            "bins_selected_str": bins_selected_str,
                        },
                    )
                )
                continue

            # todo set up equivalence_feature_matrix and equivalence_feature_ids

            treatment_ids = self.model.data_treatment.df[df_id_col].unique()
            comparison_ids = self.model.data_sample.df[df_id_col].unique()
            if len(treatment_ids) != len(pd.Series(treatment_ids).unique()):
                raise ValueError("Duplicate IDs found in treatment group.")
            if len(comparison_ids) != len(pd.Series(comparison_ids).unique()):
                raise ValueError("Duplicate IDs found in comparison group.")

            ix_x = equivalence.ids_to_index(treatment_ids, equivalence_feature_ids)
            ix_y = equivalence.ids_to_index(comparison_ids, equivalence_feature_ids)

            (
                equiv_treatment,
                equiv_sample,
                equivalence_distance,
            ) = equivalence.Equivalence(
                ix_x,
                ix_y,
                equivalence_feature_matrix,
                n_quantiles=equivalence_quantile_size,
                how=equivalence_method,
            ).compute()

            n_bin_results.append(
                dict(
                    **n_bin_option,
                    **{
                        "distance": equivalence_distance,
                        "status": "SUCCEEDED",
                        "bins_selected_str": bins_selected_str,
                    },
                )
            )

            # build a dataframe with the equivalence vectors so we can plot them
            equiv_sample["bin_str"] = bins_selected_str
            self.equiv_samples.append(equiv_sample.copy(deep=True))

            logging.info(
                f"Computing bins: {bins_selected_str} distance: "
                f"{equivalence_distance:.2f}, "
                #  f"pct: {100*equivalence_distance/sum(equiv_treatment[equivalence_value_col]):.2f}"
            )
            if equivalence_distance < min_distance:
                min_distance = equivalence_distance
                min_columns = copy.deepcopy(self.model.columns)

        self.n_bin_results = pd.DataFrame(n_bin_results)
        if not min_columns:
            raise ValueError("No valid bin configurations were discovered")

        # same for all of them anyway
        # TODO (ssuffian): Calculate this cleaner
        equiv_treatment.name = self.model.treatment_label
        self.equiv_treatment = equiv_treatment

        self.model.columns = min_columns
        bins_selected_str = self.model.get_all_n_bins_as_str()
        logging.info(
            f"Selected bin: {bins_selected_str} distance: "
            f"{min_distance:.2f}, "
            # f"pct: {100*min_distance/sum(equiv_treatment[equivalence_value_col]):.2f}, "
            f"random_seed: {random_seed}"
        )
        self.model.fit(
            self.df_treatment,
            min_n_treatment_per_bin=min_n_treatment_per_bin,
            random_seed=random_seed,
        )
        # if n_samples_approx is None, use the maximum available.
        self.model.sample(
            self.df_pool,
            n_samples_approx=n_samples_approx,
            random_seed=random_seed,
            relax_n_samples_approx_constraint=relax_n_samples_approx_constraint,
        )
        self.n_samples_approx = n_samples_approx

        # get averages that can be accessed later
        self.equiv_treatment_avg = self.equiv_treatment.groupby("feature_index")[
            "value"
        ].mean()
        self.equiv_treatment_avg = self.equiv_treatment_avg.rename("treatment")

        self.equiv_pool_avg = (
            pd.DataFrame(equivalence_feature_matrix)
            .mean()
            .to_frame()
            .rename(columns={0: "comparison pool"})
            .reset_index(drop=True)
        )

        self.equiv_samples_avg = (
            pd.concat(self.equiv_samples)
            .groupby(["bin_str", "feature_index"])["value"]
            .mean()
            .reset_index()
            .pivot(index="feature_index", columns="bin_str", values="value")
        )
        self.bins_selected_str = self.model.get_all_n_bins_as_str()

        # get distances for comparison pool
        treatment_ids = self.model.data_treatment.df[df_id_col].unique()
        comparison_pool_ids = self.model.data_pool.df[df_id_col].unique()
        ix_x = equivalence.ids_to_index(treatment_ids, equivalence_feature_ids)
        ix_y = equivalence.ids_to_index(comparison_pool_ids, equivalence_feature_ids)
        equiv_treatment, equiv_pool, equivalence_distance = equivalence.Equivalence(
            ix_x,
            ix_y,
            equivalence_feature_matrix,
            n_quantiles=equivalence_quantile_size,
            how=equivalence_method,
        ).compute()
        self.equiv_pool = equiv_pool

    def kwargs_as_json(self):
        return {
            "equivalence_method": self.equivalence_method,
            "n_samples_approx": self.n_samples_approx,
            "min_n_treatment_per_bin": self.min_n_treatment_per_bin,
            "random_seed": self.random_seed,
            "min_n_sampled_to_n_treatment_ratio": self.min_n_sampled_to_n_treatment_ratio,
            "min_n_bins": self.min_n_bins,
            "max_n_bins": self.max_n_bins,
            "equivalence_quantile_size": self.equivalence_quantile_size,
        }

    def results_as_json(self):
        equiv_samples_df = pd.concat(self.equiv_samples)
        selected_sample_df = equiv_samples_df[
            equiv_samples_df["bin_str"] == self.bins_selected_str
        ]

        return {
            "bins_selected": self.bins_selected_str,
            "random_seed": self.random_seed,
            "n_bin_results": self.n_bin_results.to_dict("records"),
            "chisquare_averages": {
                "selected_sample": selected_sample_df.to_dict("records"),
                self.model.treatment_label: self.equiv_treatment.to_dict("records"),
                "comparison_pool": self.equiv_pool.to_dict("records"),
            },
            "averages": {
                "samples": self.equiv_samples_avg.reset_index().to_dict("records"),
                "selected_sample": self.equiv_samples_avg[self.bins_selected_str]
                .reset_index()
                .to_dict("records"),
                self.model.treatment_label: self.equiv_treatment_avg.reset_index().to_dict(
                    "records"
                ),
                self.model.pool_label: self.equiv_pool_avg.reset_index().to_dict(
                    "records"
                ),
            },
        }

    def plot_records_based_equiv_average(self, plot=True):
        equiv_df = pd.concat(
            [self.equiv_treatment_avg, self.equiv_pool_avg, self.equiv_samples_avg],
            axis=1,
        )

        wrong_models = [
            m for m in self.equiv_samples_avg.columns if m != self.bins_selected_str
        ]

        if plot:
            fig, ax = plt.subplots()
            for wm in wrong_models:
                plt.plot(self.equiv_samples_avg[wm], alpha=0.1, color="b")
            equiv_df[[self.bins_selected_str, "treatment", "comparison pool"]].plot(
                color=["k", "r", "k"], style=["-", "-", "."], ax=ax
            )
            plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
