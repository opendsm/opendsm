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

import colorsys
from copy import deepcopy as copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from opendsm.common.stats.outliers import IQR_outlier
from opendsm.eemeter.models.daily.utilities.ellipsoid_test import (
    robust_confidence_ellipse,
)

fontsize = 14
mpl.rc("font", family="sans-serif")
c = ["tab:blue", "tab:green", "tab:purple"]


def adjust_lightness(color, amount=1.0):
    try:
        c = mpl.colors.cnames[color]
    except:
        c = color

    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))

    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plot(
    fit,
    meter_eval,
    include_resid=False,
    plot_gaussian_ellipses=False,
    plot_outliers=True,
    ax=None,
    include_scatter=True,
    model_color="tab:orange",
    include_uncertainty=False,
):
    # sort meter_eval by temperature
    meter_eval = meter_eval.sort_values(by="temperature")
    meter_eval = meter_eval.dropna(subset=["temperature", "predicted"])

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4), dpi=300)
    else:
        fig = ax.get_figure()

    # Plot scatter and Gaussian ellipses
    if include_scatter:
        for n, season in enumerate(["summer", "shoulder", "winter"]):
            for day_label in ["weekday", "weekend"]:
                if day_label == "weekday":
                    color = c[n]
                    marker = "o"
                    s = 7**2
                else:
                    color = adjust_lightness(copy(c[n]), amount=0.8)
                    marker = "D"
                    s = 5.5**2

                label = f"{season} {day_label}"
                meter_season = meter_eval[
                    (meter_eval["season"] == season)
                    & (meter_eval["weekday_weekend"] == day_label)
                    & (meter_eval["observed"].notna())
                ]

                T = meter_season["temperature"].values
                obs = meter_season["observed"].values
                model = meter_season["predicted"].values

                y = (obs - model) if include_resid else obs
                ax.scatter(T, y, color=color, marker=marker, s=s, label=label)

                if not plot_gaussian_ellipses:
                    continue

                std_sqr = std = np.array(fit.model_settings.reduce_splits_num_std)[:, None]
                std_sqr = std.T * std

                mu, cov, a, b, phi = robust_confidence_ellipse(T, obs, std_sqr)

                ell = mpl.patches.Ellipse(
                    mu, 2 * a, 2 * b, np.degrees(phi), color=color, zorder=10
                )
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.3)
                ax.add_artist(ell)

    # Plot models (only meaningful when not showing residuals)
    if not include_resid:
        splits = meter_eval["model_split"].unique()
        n_splits = len(splits)
        for i, split in enumerate(splits):
            # Cycle lightness across splits so each segment is visually distinct
            color = adjust_lightness(model_color, 1.0 - 0.25 * i) if n_splits > 1 else model_color
            meter_segment = meter_eval[meter_eval["model_split"] == split]
            name = f"{split}__{meter_segment['model_type'].iloc[0]}"
            T_seg = meter_segment["temperature"].values
            pred_seg = meter_segment["predicted"].values
            ax.plot(T_seg, pred_seg, color=color, label=f"{name}")

            if include_uncertainty and "predicted_unc" in meter_segment.columns:
                unc = meter_segment["predicted_unc"].values
                if not np.all(np.isnan(unc)):
                    ax.fill_between(
                        T_seg, pred_seg - unc, pred_seg + unc,
                        alpha=0.15, color=color,
                    )
    else:
        ax.axhline(y=0, linestyle=(0, (5, 1)), linewidth=1.5, color=(0.4, 0.4, 0.4))

    ax.set_xlabel("Temperature", labelpad=10, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=0.85 * fontsize)

    if not plot_outliers:
        # Ignores crazy points when plotting based on iqr
        ylim = IQR_outlier(
            meter_eval["observed"].values, sigma_threshold=1.0, quantile=0.025
        )
        ylim_idx = [
            np.argmin(np.abs(x - meter_eval["observed"].values), axis=0) for x in ylim
        ]
        ylim = meter_eval["observed"].values[ylim_idx]
    else:
        ylim = np.quantile(meter_eval["observed"], [0, 1])

    ylim_border = 0.1 * (ylim[1] - ylim[0])
    ax.set_ylim([ylim[0] - ylim_border, ylim[1] + ylim_border])
    ax.set_ylabel("Resid" if include_resid else "Usage", labelpad=10, fontsize=fontsize)

    legend = ax.legend(framealpha=0.0, fontsize=0.5 * fontsize)

    # plt.show()

    # if figsize is None:
    #     figsize = (10, 4)

    # if ax is None:
    #     fig, ax = plt.subplots(figsize=figsize)

    # color = "C1"
    # alpha = 1

    # temp_min, temp_max = (30, 90) if temp_range is None else temp_range

    # temps = np.arange(temp_min, temp_max)

    # prediction_index = pd.date_range(
    #     "2017-01-01T00:00:00Z", periods=len(temps), freq="D"
    # )

    # temps_daily = pd.Series(temps, index=prediction_index).resample("D").mean()
    # prediction = self._predict(temps_daily).model

    # plot_kwargs = {"color": color, "alpha": alpha or 0.3}
    # ax.plot(temps, prediction, **plot_kwargs)

    # if title is not None:
    #     ax.set_title(title)

    return fig, ax
