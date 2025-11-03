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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from opendsm.common.stats.outliers import IQR_outlier

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
):
    # sort meter_eval by temperature
    meter_eval = meter_eval.sort_values(by="temperature")

    fig = plt.figure(figsize=(14, 4), dpi=300)
    if include_resid:
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.5, 1])
        ax = gs.subplots()
    else:
        ax = [fig.subplots()]

    # Plot scatter and Gaussian ellipses
    for n, season in enumerate(["summer", "shoulder", "winter"]):
        color = c[n]
        marker = "o"
        s = 7**2
        label = f"{season}"

        meter_season = meter_eval[
            (meter_eval["season"] == season) & (meter_eval["observed"].notna())
        ]

        T = meter_season["temperature"].values
        obs = meter_season["observed"].values
        model = meter_season["predicted"].values
        resid = obs - model

        ax[0].scatter(T, obs, color=color, marker=marker, s=s, label=label)
        if include_resid:
            ax[1].scatter(T, resid, color=color, marker=marker, s=s)

    # Plot models
    for split in meter_eval["model_split"].unique():
        meter_segment = meter_eval[meter_eval["model_split"] == split]

        name = f"{split}__{meter_segment['model_type'].iloc[0]}"
        ax[0].plot(
            meter_segment["temperature"],
            meter_segment["predicted"],
            color="tab:orange",
            label=f"{name}",
        )

    # ax[0].plot(T, model["c_hdd_baseline"].model, color="tab:red", label=f"c_hdd_baseline")

    if include_resid:
        ax[1].axhline(y=0, linestyle=(0, (5, 1)), linewidth=1.5, color=(0.4, 0.4, 0.4))
        ax[0].get_shared_x_axes().join(ax[0], ax[1])
        ax[1].set_xlabel("Temperature", labelpad=10, fontsize=fontsize)
        ax[1].set_ylabel("Resid", labelpad=10, fontsize=fontsize)

    else:
        ax[0].set_xlabel("Temperature", labelpad=10, fontsize=fontsize)

    # ax.plot(hours, meter[:,2], linewidth=1.5, linestyle=(0, (6, 1)), color='firebrick')
    # ax.plot(hours, meter[:,2], linewidth=2.0, linestyle='-.')
    # ax.fill_between(hours, cg_lb, cg_ub, alpha=0.3, facecolor='peru')

    # ax.set_xlim([T[0], T[-1]])
    # ax.set_xticks(np.arange(0, 505, 168))
    ax[0].tick_params(axis="both", which="major", labelsize=0.85 * fontsize)

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
    ax[0].set_ylim([ylim[0] - ylim_border, ylim[1] + ylim_border])
    # ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(7))
    # ax.tick_params(axis='both', which='major', labelsize=0.85*fontsize)
    # ax.yaxis.set_tick_params(which='minor', left=False)
    ax[0].set_ylabel("Usage", labelpad=10, fontsize=fontsize)

    legend = ax[0].legend(framealpha=0.0, fontsize=0.5 * fontsize)
    # legend._legend_box.align = 'left'

    plt.show()

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

    # return ax
