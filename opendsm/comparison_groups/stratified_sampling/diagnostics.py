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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_ind, ks_2samp
from scipy.spatial.distance import pdist
from scipy.stats import chisquare
import warnings

# Flag to check if plotnine is available
plotnine_available = True
try:
    import plotnine
    from plotnine import *
except ModuleNotFoundError:
    plotnine_available = False


def t_and_ks_test(x, y, thresh=0.05):
    t_p = "{:,.3f}".format(ttest_ind(x, y).pvalue)
    ks_p = "{:,.3f}".format(ks_2samp(x, y).pvalue)

    t_p = ttest_ind(x, y).pvalue
    ks_p = ks_2samp(x, y).pvalue
    t_ok = t_p > thresh
    ks_ok = ks_p > thresh
    t_p_str = "t pval: {:,.3f}".format(t_p)
    ks_p_str = "KS pval: {:,.3f}".format(ks_p)

    return pd.Series(
        {
            "ks_ok": ks_ok,
            "t_ok": t_ok,
            "ks_p": ks_p_str,
            "t_p": t_p_str,
            "t_value": t_p,
            "ks_value": ks_p,
        }
    )


class DiagnosticPlotter:
    def quantile(self, df, df_equiv, cols=None):
        if cols is None:
            cols = self.default_cols

        df_quantile = df[["population"] + cols].melt(id_vars=["population"])
        quantile_range = np.arange(0.005, 1.0, 0.01)
        df_quantile = (
            df_quantile.groupby(["population", "variable"])
            .apply(
                lambda x: pd.DataFrame(
                    {
                        "quantile": quantile_range,
                        "value": x["value"].quantile(quantile_range),
                    }
                )
            )
            .reset_index()
        )
        
        if plotnine_available:
            plotnine.options.figure_size = (6, 3 * df_quantile.variable.nunique())

            base_plot = (
                ggplot(df_quantile, aes(x="quantile", y="value", color="population"))
                + geom_point()
                + facet_wrap("~variable", scales="free_y", ncol=1)
                + theme_bw()
            )

            df_range = (
                df_quantile.groupby("variable")
                .apply(lambda df: pd.Series({"min": df.value.min(), "max": df.value.max()}))
                .reset_index()
            )

            df_equiv = df_equiv.merge(df_range)
            df_equiv["x"] = 0
            df_equiv["y"] = (df_equiv["max"] - df_equiv["min"]) * 0.95 + df_equiv["min"]
            df_equiv["y2"] = (df_equiv["max"] - df_equiv["min"]) * 0.8 + df_equiv["min"]

            p = (
                base_plot
                + geom_label(
                    aes(label="t_p", fill="t_ok", x="x", y="y"),
                    data=df_equiv,
                    ha="left",
                    va="top",
                    color="black",
                    size=10,
                )
                + geom_label(
                    aes(label="ks_p", fill="ks_ok", x="x", y="y2"),
                    data=df_equiv,
                    ha="left",
                    va="top",
                    color="black",
                    size=10,
                )
                + scale_fill_manual({True: "lightgreen", False: "orange"}, guide=None)
                + scale_color_discrete()
            )

            return p
        else:
            warnings.warn("Plotnine is not installed. Diagnostic functionality will not be available. Use 'pip install plotnine' to address this.")
            return None

    def scatter(self, df, cols=None):
        if cols is None:
            cols = self.default_cols
        col_pairs = combinations(cols, 2)
        plots = [self._scatter(df, p[0], p[1]) for p in col_pairs]
        
        return [p for p in plots]

    def _scatter(self, df, col_x, col_y):
        def sample_if_too_big(df):
            if len(df) > 2000:
                df = df.sample(2000)
            return df

        df = (
            df.groupby("population", group_keys=False)
            .apply(sample_if_too_big)
            .reset_index()
        )

        if plotnine_available:
            plotnine.options.figure_size = (12, 5)
            base_plot = (
                ggplot(df, aes(x=col_x, y=col_y, color="population"))
                + geom_point()
                + facet_wrap("~population", nrow=1)
                + theme_bw()
            )
            outlier_bins = self.data_treatment.outlier_bins
            outlier_bins = outlier_bins[outlier_bins["outlier"]]["bin"].values
            df_rects = self.binning.edges_xy(col_x, col_y)
            df_rects = df_rects[~df_rects["bin"].isin(outlier_bins)]

            # due to plotnine bug
            df_rects[col_x] = np.nan
            df_rects[col_y] = np.nan
            p = base_plot + geom_rect(
                aes(xmin="x_min", xmax="x_max", ymin="y_min", ymax="y_max"),
                data=df_rects,
                color="black",
                fill=None,
                size=0.2,
            )

            return p
        else:
            warnings.warn("Plotnine is not installed. Diagnostic functionality will not be available. Use 'pip install plotnine' to address this.")
            return None

    def histogram(self, df, cols=None):
        if cols is None:
            cols = self.default_cols

        return [self._histogram(df, c) for c in cols]

    def _histogram(self, df, col):
        if plotnine_available:
            plotnine.options.figure_size = (12, 5)
            p = (
                ggplot(df, aes(x=col, fill="population"))
                + geom_histogram(bins=30)
                + facet_wrap("~population", nrow=1, scales="free_y")
                + theme_bw()
            )

            outlier_bins = self.data_treatment.outlier_bins
            outlier_bins = outlier_bins[outlier_bins["outlier"]]["bin"].values
            df_rects = self.binning.edges_xy(col, col)
            df_rects = df_rects[~df_rects["bin"].isin(outlier_bins)]

            # due to plotnine bug
            df_rects[col] = np.nan
            p = p + geom_rect(
                aes(xmin="x_min", xmax="x_max", ymin=-np.inf, ymax=np.inf),
                data=df_rects,
                color="black",
                fill=None,
                size=0.2,
            )

            return p
        else:
            warnings.warn("Plotnine is not installed. Diagnostic functionality will not be available. Use 'pip install plotnine' to address this.")
            return None


class StratifiedSamplingDiagnostics(DiagnosticPlotter):
    """
    Construct plots and tables summarizing results of stratified sampling.
    Operates on a StratifiedSamplingModel.  Plots will show treatment,
    pool, and comparison group meters on the same axes to allow for easy comparisons.
    If fitting failed, plots will be available with treatment and pool meters only.

    Methods
    =======

    scatter():
        Construct 2-D scatter plots of all stratification columns with bins superimposed.

    histogram():
        Construct 1-D histogram plots of all stratification columns with bins superimposed.

    quantile_equivalence():
        Construct quantile plots to compare distributions; include t-test and ks-test
        p-values.

    count_bins():
        Construct a table of pins and relative densities for treatment, pool, and comparison.


    Attributes
    ==========

    model:
        A StratifiedSamplingModel, after fit() or fit_and_sample() have been run.

    """

    def __init__(self, model):
        self.model = model
        self.binning = self.model.binning
        self.data_treatment = self.model.data_treatment
        self.data_pool = self.model.data_pool
        self.sampled = self.model.sampled

        self.treatment_label = self.model.treatment_label
        self.pool_label = self.model.pool_label
        self.data_sample = self.model.data_sample

        # these two will always exist
        df_treatment = self.model.data_treatment.df
        df_pool = self.model.data_pool.df
        self.default_cols = self.model.col_names

        self.available_equiv_labels = [
            self.treatment_label,
            self.pool_label,
            "sample",
        ]
        df_sample = (
            self.data_sample.df if self.data_sample is not None else pd.DataFrame()
        )
        self.labeled_dfs = [df_treatment, df_pool, df_sample]

        def _concat_dfs(dfs_to_concat, concat_col, concat_values):
            if len(dfs_to_concat) != len(concat_values):
                raise ValueError(
                    "dfs_to_concat should be the same length as concat_values"
                )
            return pd.concat(
                [
                    df.assign(**{concat_col: value})
                    for df, value in zip(dfs_to_concat, concat_values)
                ],
                sort=False,
            )

        self.df_all = _concat_dfs(
            self.labeled_dfs, "population", self.available_equiv_labels
        )

    def histogram(self, cols=None):
        return super().histogram(self.df_all, cols)

    def scatter(self, cols=None):
        return super().scatter(self.df_all, cols)

    def quantile_equivalence(self, cols=None):
        df_equiv = self.equivalence(cols)
        return super().quantile(self.df_all, df_equiv, cols=cols)

    def _check_equiv_labels(self, equiv_label_x, equiv_label_y):
        if (
            equiv_label_x is not None
            and equiv_label_x not in self.available_equiv_labels
        ):
            raise ValueError(
                f"equiv_label_x must be one of: {self.available_equiv_labels}"
            )
        if (
            equiv_label_y is not None
            and equiv_label_y not in self.available_equiv_labels
        ):
            raise ValueError(
                f"equiv_label_y must be one of: {self.available_equiv_labels}"
            )
        equiv_label_x = equiv_label_x if equiv_label_x else self.treatment_label
        equiv_label_y = (
            equiv_label_y
            if equiv_label_y
            else ("sample" if self.data_sample else self.pool_label)
        )
        return equiv_label_x, equiv_label_y

    def equivalence(self, cols=None, equiv_label_x=None, equiv_label_y=None):
        """
        Attributes
        ----------
        cols: str
            Columns to plot and calculate equivalence for. Defaults to all available cols.
        equiv_label_x: str
            First label to measure equivalence against (defaults to treatment label)
        equiv_label_y: str
            Second label to measure equivalence against (defaults to sample if available,
             otherwise defaults to full pool set)
        """
        if (
            equiv_label_x is not None
            and equiv_label_x not in self.available_equiv_labels
        ):
            raise ValueError(
                f"equiv_label_x must be one of: {self.available_equiv_labels}"
            )
        if (
            equiv_label_y is not None
            and equiv_label_y not in self.available_equiv_labels
        ):
            raise ValueError(
                f"equiv_label_y must be one of: {self.available_equiv_labels}"
            )
        equiv_label_x = equiv_label_x if equiv_label_x else self.treatment_label
        equiv_label_y = (
            equiv_label_y
            if equiv_label_y
            else ("sample" if self.data_sample else self.pool_label)
        )
        cols = cols if cols else self.default_cols

        df = self.df_all[["population"] + cols].melt(id_vars=["population"])
        return (
            df.groupby("variable")
            .apply(
                lambda x: t_and_ks_test(
                    x[x["population"] == equiv_label_x].value.dropna(),
                    x[x["population"] == equiv_label_y].value.dropna(),
                ), 
                include_groups=False,
            )
            .reset_index()
        )

    def equivalence_passed(self, cols=None):
        df = self.equivalence(cols=cols)
        return all(df["ks_ok"]) & all(df["t_ok"])

    def count_bins(self):

        df_treatment = self.data_treatment.count_bins(skip_outliers=True).rename(
            columns={
                "n": f"n_{self.treatment_label}",
                "n_pct": f"n_pct_{self.treatment_label}",
            }
        )

        df_pool = self.data_pool.count_bins(skip_outliers=False).rename(
            columns={
                "n": f"n_{self.pool_label}",
                "n_pct": f"n_pct_{self.pool_label}",
            }
        )

        df = df_treatment.merge(df_pool)

        if self.sampled:
            df_sample = self.data_sample.count_bins(skip_outliers=False).rename(
                columns={"n": f"n_sampled", "n_pct": f"n_pct_sampled"}
            )
            df = df.merge(df_sample)
        return df

    def n_sampled_to_n_treatment_ratio(self):
        bin_df = self.count_bins()
        if bin_df.empty:
            return 0
        else:
            return (
                (bin_df["n_sampled"] / bin_df[f"n_{self.treatment_label}"])
                .min()
                .astype(int)
            )
