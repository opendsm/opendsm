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

import numpy as np
import pandas as pd
import itertools
from operator import and_
from functools import reduce

__all__ = ("BinnedData", "Binning", "Bin", "MultiBin")


class ModelSamplingException(Exception):
    pass


class BinnedData:
    def __init__(self, df, binning, min_n_treatment_per_bin=0):
        self.binning = binning
        self.df = self._map_bins(df)
        self.min_n_treatment_per_bin = min_n_treatment_per_bin
        self.outlier_bins = self._outlier_bins()
        self._flag_outliers()

    def _map_bins(self, df):
        """Add '_bin' column to df indicating which bin each row maps to."""
        df.loc[:, "_bin"] = None

        for b in self.binning.multibins:
            df.loc[b.filter_expr()(df), "_bin"] = b
            df.loc[b.filter_expr()(df), "_bin_label"] = b.label
        return df

    def count_bins_1d(self, column):
        """Count number of elements within each 1-dimensional bin associated with
        column."""
        bins = self.binning.bins[column]
        df = pd.DataFrame(
            [
                {
                    "column": column,
                    "index": b.index,
                    "min": b.min,
                    "max": b.max,
                    "n": len(self.df[b.filter_expr(self.df)]),
                }
                for b in bins
            ]
        )
        df["n_pct"] = df["n"] / df["n"].sum()
        return df

    def count_bins(self, skip_outliers=False):
        """Count number of elements within each multi-dimensional bin."""
        df = self.df
        if skip_outliers:
            df = df[~df._outlier_bin & ~df._outlier_value]
        df = (
            df._bin.value_counts()
            .reset_index()
            .rename(columns={"_bin": "bin", "count": "n"})
        )
        df["n_pct"] = df["n"] / df["n"].sum()
        return df

    def _outlier_bins(self):
        df_bins = (
            self.df._bin.value_counts()
            .reset_index()
            .rename(columns={"_bin": "bin", "count": "n"})
        )
        df_bins["outlier"] = df_bins["n"] < self.min_n_treatment_per_bin
        return df_bins

    def _flag_outliers(self):
        """Flag elements that fall in bins that are too small."""
        df = self.outlier_bins
        self.df.loc[:, "_outlier_bin"] = self.df._bin.isin(
            df[df["outlier"]]["bin"].values
        )


class Binning:
    """Contains list of multidimensional bins"""

    def __init__(self):
        self.bins = {}  # 1-dimensional bins for each column
        self.edges_1d = {}  # array of bin edges
        self.multibins = []  # list of n-dimensional bin

    def edges(self):
        return pd.concat([b.edges() for b in self.multibins])

    def edges_xy(self, col_x, col_y):
        df = self.edges()
        df_x = df[df.column == col_x]
        df_x = df_x.rename(
            columns={"column": "column_x", "min": "x_min", "max": "x_max"}
        )
        df_y = df[df.column == col_y]
        df_y = df_y.rename(
            columns={"column": "column_y", "min": "y_min", "max": "y_max"}
        )
        return df_x.merge(df_y)

    def bin(self, values, column_name, n_bins, fixed_width):
        """Generate and store  1-dimensional binning for the specific column"""
        if fixed_width:
            bins, edges = pd.cut(values, n_bins, retbins=True, duplicates="drop")
        else:
            bins, edges = pd.qcut(values, q=n_bins, retbins=True, duplicates="drop")

        if len(edges) < n_bins + 1:
            raise ValueError(
                f"Duplicate bins were created for {column_name} -- this usually occurs if a large number of data points have the same value, i.e. zero. Try using fewer bins. Set n_bins to 1 and run model.diagnostics() to view data.  \nStats: \n{values.describe()}"
            )
        self._add_column(column=column_name, edges=edges)

    def _add_column(self, column, edges):
        """Add a new 1-demsnsional bin to internal data structure."""
        this_bins = []
        for i in range(len(edges) - 1):
            this_bins.append(Bin(column, edges[i], edges[i + 1], i))
        self.bins[column] = this_bins
        self.edges_1d[column] = edges
        self._update_multibins()
        return self

    def _update_multibins(self):
        """Update internal data structure."""
        bins = [b for column, b in self.bins.items()]
        self.multibins = [MultiBin(b) for b in itertools.product(*bins)]


class Bin:
    """Single-dimensional bin"""

    def __init__(self, column, min, max, index):
        self.column = column
        self.min = min
        self.max = max
        self.index = index

    def filter_expr(self):
        """Make  a function that filters a dataframe to keep only values witihn this
        bin."""
        return lambda df: (df[self.column] >= self.min) & (df[self.column] <= self.max)

    def __str__(self):
        return f"Bin: {self.column} {self.index} - [{self.min}, {self.max})"

    def __repr__(self):
        return str(self)


class MultiBin:
    """Multi-dimensional bin -- intersection of n Bins"""

    def __init__(self, bins):
        self.bins = bins
        self.label = "__".join(
            [f"{b.column}_{str(b.index).zfill(3)}" for b in self.bins]
        )

    def filter_expr(self):
        """Make  a function that filters a dataframe to keep only values witihn
        each dimension of this bin."""
        return lambda df: reduce(and_, [(b.filter_expr()(df)) for b in self.bins])

    def get_max_n_target(self, df):
        return len(df[self.filter_expr()(df)])

    def sample(self, df, n_target, min_n_treatment_per_bin, random_seed=1):
        """Sample n_target elements from dataframe df that fall
        within each dimension of this bin."""

        d1 = df[self.filter_expr()(df)]

        if n_target < min_n_treatment_per_bin:
            raise ModelSamplingException(
                f"Bin {self} has target of {n_target} control meters which is less than minimum of {min_n_treatment_per_bin}.  Try increasing n_outputs. \n\nBins: {chr(10).join([str(b) for b in self.bins])}"
            )

        if len(d1) < n_target:
            raise ModelSamplingException(
                f"Bin {self} has target of {n_target} control meters, but only {len(d1)} available.  Try reducing n_outputs or decreasing number of bins.  Run diagnostics.scatter_2d(), diagnostics.quantile_plot(), or diagnostics.histogram() to visualize data. \n\nBins: {chr(10).join([str(b) for b in self.bins])}"
            )
        return d1.sample(n_target, replace=False, random_state=random_seed)

    def edges(self):
        return pd.DataFrame(
            [
                {
                    "bin": self,
                    "label": self.label,
                    "column": b.column,
                    "min": b.min,
                    "max": b.max,
                }
                for b in self.bins
            ]
        )

    def __str__(self):
        return f"MultBin: {self.label}"

    def __repr__(self):
        return str(self)


def sample_bins(
    binned_data_treatment,
    binned_data_pool,
    random_seed,
    n_samples_approx,
    counts=None,
    skip_outliers=True,
    relax_n_samples_approx_constraint=False,
):
    if counts is None:
        counts = binned_data_treatment.count_bins(skip_outliers=True)
        counts["n_target"] = np.floor(counts["n_pct"] * n_samples_approx).astype(int)

    if len(counts) == 0:
        raise ValueError("No non-outlier treatment data remaining.")
    df = pd.concat(
        [
            row["bin"].sample(
                binned_data_pool.df,
                n_target=row["n_target"],
                min_n_treatment_per_bin=binned_data_treatment.min_n_treatment_per_bin,
                random_seed=random_seed,
            )
            for index, row in counts.iterrows()
        ]
    )
    return df


def get_counts_and_update_n_samples_approx(
    binned_data_treatment,
    binned_data_pool,
    n_samples_approx,
    relax_n_samples_approx_constraint,
):
    counts = binned_data_treatment.count_bins(skip_outliers=True)

    # Scenario 1: n_samples_approx = None
    # a way to ensure you get the max number of samples if n_samples_approx=None
    counts["n_samples_available"] = [
        row["bin"].get_max_n_target(binned_data_pool.df)
        for index, row in counts.iterrows()
    ]
    max_possible_n_samples_approx = int(
        min(counts["n_samples_available"] / counts["n_pct"])
    )
    n_samples_approx = (
        n_samples_approx if n_samples_approx else max_possible_n_samples_approx
    )
    # needs to be floor to ensure rounding errors don't leave one less than exists
    counts["n_target"] = np.floor(counts["n_pct"] * n_samples_approx).astype(int)

    # if you want to treat n_samples_approx as a max, but get as many as you can
    # if you can't reach that, then set relax_n_samples_approx_constraint=True
    has_enough_for_n_samples_approx = not any(
        counts["n_samples_available"] < counts["n_target"]
    )
    relax_ratio_constraint = False
    if has_enough_for_n_samples_approx:
        # Scenario 2: n_samples_approx=value so we want to ignore the ratio constraint
        relax_ratio_constraint = True
    elif relax_n_samples_approx_constraint:
        # Scenario 3: n_samples_approx=value and that value but that value can not
        # be met, so we want as many as possible and it is valid as long as it
        # meets the ratio constraint.
        n_samples_approx = max_possible_n_samples_approx
        counts["n_target"] = np.floor(counts["n_pct"] * n_samples_approx).astype(int)
    # else:
    # Scenario 4: It will fail during sampling because it can not meet
    # n_samples_approx and we did not relax that constraint.
    return n_samples_approx, relax_ratio_constraint, counts
