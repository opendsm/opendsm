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
import pandas as pd
import itertools
import logging
import numpy as np
from .diagnostics import StratifiedSamplingDiagnostics
from .bins import (
    Binning,
    BinnedData,
    ModelSamplingException,
    sample_bins,
    get_counts_and_update_n_samples_approx,
)

pd.options.mode.chained_assignment = None  # suppress warnings

logger = logging.getLogger(__name__)


class StratifiedSampling(object):
    """
    Perform stratified sampling on a treatment group and comparison pool.  
    
    Input data must be provided in the form of two data frames, df_treatment and df_pool, 
    which have identical columns.  These data frames should contain one row per meter, 
    one ID column, and one or more numerical feature columns.  The comparison pool
    will be stratified (i.e. binned) along one or more of these feature columns, and 
    a comparison group will be selected such that the distribution of features in the 
    comparison group is as close as possible to that of the treatment group.

    Stratification columns must be configured as follows:

        m = StratifiedSampling()
        m.add_column('annual_usage', min_value=0, max_value=20000)
        m.add_column('summer_usage', min_value=0, max_value=1000)

    In this case, `annual_usage` and `summer_usage` are feature columns that 
    are present in `df_treatment` and `df_pool`.
    See `StratifiedSampling.add_column()` for more information on configuring columns.
    Once columns are added, execute the model as follows:

        m.fit_and_sample(df_treatment, df_pool)

    See `StratifiedSampling.fit_and_sample()` for additional options, notably several
    parameters which determine the number of meters in the comparison group.

    After fitting the model, you can create a StratifiedSamplingDiagnostics object 
    which has methods for producing diagnostic plots and tables:

        d = m.diagnostics()
        d.scatter()
        d.bin_counts()


    """

    def __init__(
        self, treatment_label="treatment", pool_label="pool", output_name="output"
    ):
        self.columns = {}
        self.treatment_label = treatment_label
        self.pool_label = pool_label
        self.output_name = output_name
        self.trained = False
        self.sampled = False
        self.data_treatment = None
        self.data_pool = None
        self.data_sample = None

    def _chop_outliers(self, df):
        for name, c in self.columns.items():
            if c["min_value_allowed"] is not None:
                df = df[df[c["name"]] >= c["min_value_allowed"]]
            if c["max_value_allowed"] is not None:
                df = df[df[c["name"]] <= c["max_value_allowed"]]
        return df

    def _perturb(self, df_orig, col_names=None, random_seed=1):
        # qcut doesn't work if the same value recurs too many times, i.e. zero.  We can add a small amount of random noise to fix this
        np.random.seed(random_seed)
        df_pert = df_orig.copy()
        col_names = col_names if col_names else list(self.columns.keys())
        for col_name in col_names:
            df_pert[col_name] = df_pert[col_name].astype(float)
            range = df_pert[col_name].max() - df_pert[col_name].min()
            perturbation = (np.random.random(len(df_pert)) - 0.5) * range * 1e-6
            df_pert.loc[:, col_name] = df_pert[col_name] + perturbation
        return df_pert

    def add_column(
        self,
        name: str,
        n_bins: int = None,
        min_value_allowed: int = None,
        max_value_allowed: int = None,
        fixed_width: int = True,
        auto_bin_require_equivalence: bool = True,
    ):
        """
        Add a stratification column to the model.

        Attributes
        ----------
        name: str
            The name of the column to be added to the model.
        n_bins: int
            Fixed number of bins to stratify over for this column.
            If set to None, automatic binning occurs. 
        min_value_allowed: int
            Minimum treatment value used to construct bins (used to remove outliers).
        max_value_allowed: int
            Maximum treatment value used to construct bins (used to remove outliers).
        auto_bin_require_equivalence: bool
            Whether the column requires equivalence when auto-binning
        """
        auto_bin = n_bins is None
        n_bins = 1 if n_bins is None else n_bins

        self.columns[name] = {
            "name": name,
            "auto_bin": auto_bin,
            "n_bins": n_bins,
            "min_value_allowed": min_value_allowed,
            "max_value_allowed": max_value_allowed,
            "fixed_width": fixed_width,
            "auto_bin_require_equivalence": auto_bin_require_equivalence,
        }

        self.binning = None
        self.trained = False
        self.predicted = False
        self.col_names = list(self.columns.keys())
        return self

    def _check_columns_present(self, df):
        if not getattr(self, "col_names"):
            raise ValueError(
                "No columns found in model. Use add_columns(...) to add a column."
            )
        missing_cols = list(set(self.col_names) - set(df.columns))
        if len(missing_cols) > 0:
            raise ValueError(
                f"data is missing required columns: {','.join(missing_cols)}"
            )

    def fit_and_sample(
        self,
        df_treatment,
        df_pool,
        n_samples_approx=None,
        min_n_treatment_per_bin=0,
        random_seed=1,
        min_n_sampled_to_n_treatment_ratio=4,
        relax_n_samples_approx_constraint=False,
    ):
        """
        Attributes
        ----------
        df_treatment: pandas.DataFrame
            dataframe to use for constructing the stratified sampling bins.
        df_pool: pandas.DataFrame
            dataframe to sample from according to the constructed stratified sampling bins.
        n_samples_approx: int
            approximate number of total samples from df_pool. It is approximate because
            there may be some slight discrepencies around the total count to ensure
            that each bin has the correct percentage of the total.
        min_n_treatment_per_bin: int
            Minimum number of treatment samples that must exist in a given bin for 
            it to be considered a non-outlier bin (only applicable if there are 
            cols with fixed_width=True)
        min_n_sampled_to_n_treatment_ratio: int
        relax_n_samples_approx_constraint: bool
            If True, treats n_samples_approx as an upper bound, but gets as many comparison group
            meters as available up to n_samples_approx. If False, it raises an exception
            if there are not enough comparison pool meters to reach n_samples_approx.
            
        """
        if len(self.columns) == 0:
            raise ValueError("You must add at least one column before fitting.")
        logger.debug(self.columns)
        for name, col in self.columns.items():
            if col["auto_bin"]:
                completed = False
                while not completed:
                    logging.info(f"Computing bins: {self.get_all_n_bins_as_str()} ")
                    self.fit(
                        df_treatment,
                        min_n_treatment_per_bin=min_n_treatment_per_bin,
                        random_seed=random_seed,
                    )
                    self.sample(
                        df_pool,
                        n_samples_approx=n_samples_approx,
                        random_seed=random_seed,
                        relax_n_samples_approx_constraint=relax_n_samples_approx_constraint,
                    )

                    def _violates_ratio():
                        n_sampled_to_n_treatment_ratio = (
                            self.diagnostics().n_sampled_to_n_treatment_ratio()
                        )
                        if (
                            n_sampled_to_n_treatment_ratio
                            < min_n_sampled_to_n_treatment_ratio
                        ):
                            logger.info(
                                f"Insufficient pool data in one of the bins for {col['name']}:"
                                f"found {n_sampled_to_n_treatment_ratio}:1 but need "
                                f"{min_n_sampled_to_n_treatment_ratio}:1. Using last successful n_bins."
                            )
                            return True
                        return False

                    if col["auto_bin_require_equivalence"]:
                        if self.data_sample.df.empty:
                            raise ValueError(
                                "Too many bin divisions before finding equivalence"
                                f" for {col['name']} (usually occurs when several"
                                " stratification params are used)."
                            )
                        completed = self.diagnostics().equivalence_passed([col["name"]])
                        if min_n_sampled_to_n_treatment_ratio and _violates_ratio():
                            completed = True
                            self.set_n_bins(name, self.get_n_bins(name) - 1)
                        if not completed:
                            self.set_n_bins(name, self.get_n_bins(name) + 1)
                    else:
                        if min_n_sampled_to_n_treatment_ratio and _violates_ratio():
                            self.set_n_bins(name, self.get_n_bins(name) - 1)
                            completed = True
                        else:
                            self.set_n_bins(name, self.get_n_bins(name) + 1)

        self.fit(
            df_treatment,
            min_n_treatment_per_bin=min_n_treatment_per_bin,
            random_seed=random_seed,
        )
        n_treatment = len(df_treatment)
        # if n_samples_approx is None, use the maximum available.
        df_sample = self.sample(
            df_pool,
            n_samples_approx=n_samples_approx,
            random_seed=random_seed,
            relax_n_samples_approx_constraint=relax_n_samples_approx_constraint,
        )
        self.n_samples_approx = n_samples_approx
        return df_sample

    def print_n_bins(self):
        logger.info(self.get_all_n_bins_as_str())

    def get_all_n_bins_as_str(self):
        return ",".join(
            [f"{col}:{self.get_n_bins(col)} bins" for col in self.columns.keys()]
        )

    def get_n_bins(self, col_name):
        col = self.columns[col_name]
        return col["n_bins"]

    def set_n_bins(self, col_name, n_bins):
        col = self.columns[col_name]
        col["n_bins"] = n_bins
        self.columns[col_name] = col

    def fit(self, df_treatment, min_n_treatment_per_bin=0, random_seed=1):
        self._check_columns_present(df_treatment)
        df_treatment = self._perturb(
            self._chop_outliers(df_treatment), random_seed=random_seed
        )
        self.df_treatment = df_treatment.copy()
        self.binning = Binning()

        self.df_treatment["_outlier_value"] = False
        for name, col in self.columns.items():

            if col["min_value_allowed"] is not None:
                self.df_treatment.loc[
                    self.df_treatment[col["name"]] < col["min_value_allowed"],
                    "_outlier_value",
                ] = True
            if col["max_value_allowed"] is not None:
                self.df_treatment.loc[
                    self.df_treatment[col["name"]] > col["max_value_allowed"],
                    "_outlier_value",
                ] = True

        for name, col in self.columns.items():
            values = (
                self.df_treatment.loc[~self.df_treatment._outlier_value, col["name"]]
                .dropna()
                .astype(float)
            )
            self.binning.bin(
                values, col["name"], col["n_bins"], fixed_width=col["fixed_width"]
            )

        self.data_treatment = BinnedData(
            self.df_treatment,
            self.binning,
            min_n_treatment_per_bin=min_n_treatment_per_bin,
        )
        self.trained = True

    # what kinds of diagnostics?
    # - explore raw treatment data
    # - explore raw pool data
    # - compare treatment data vs pool data, pre-fit
    # - compare treatment data vs pool data, post-fit
    # - compare treatment data vs pool data, post-sampled

    def diagnostics(self):
        return StratifiedSamplingDiagnostics(model=self)

    def sample(
        self,
        df_pool,
        n_samples_approx=None,
        random_seed=1,
        relax_n_samples_approx_constraint=False,
    ):
        if not self.trained and self.data_treatment is not None:
            raise ValueError("No model found; please run fit()")
        
        self._check_columns_present(df_pool)
        df_pool = self._perturb(self._chop_outliers(df_pool), random_seed=random_seed)
        self.data_pool = BinnedData(df_pool, self.binning)
        (
            n_samples_approx,
            relax_ratio_constraint,
            counts,
        ) = get_counts_and_update_n_samples_approx(
            self.data_treatment,
            self.data_pool,
            n_samples_approx=n_samples_approx,
            relax_n_samples_approx_constraint=relax_n_samples_approx_constraint,
        )
        self.relax_ratio_constraint = relax_ratio_constraint
        df_sample = sample_bins(
            self.data_treatment,
            self.data_pool,
            n_samples_approx=n_samples_approx,
            relax_n_samples_approx_constraint=relax_n_samples_approx_constraint,
            random_seed=random_seed,
        )
        self.data_sample = BinnedData(df_sample, self.binning)
        self.sampled = True
        return self.data_sample
