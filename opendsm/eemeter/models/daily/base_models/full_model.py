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

import numba
import numpy as np

from opendsm.common.stats.adaptive_loss import adaptive_weights
from opendsm.common.utils import LN_MAX_POS_SYSTEM_VALUE, LN_MIN_POS_SYSTEM_VALUE


@numba.jit(nopython=True, error_model="numpy", cache=True)
def full_model(
    hdd_bp,
    hdd_beta,
    hdd_k,
    cdd_bp,
    cdd_beta,
    cdd_k,
    intercept,
    T_fit_bnds=np.array([]),
    T=np.array([]),
):
    """
    This function predicts the total energy consumption based on the given parameters.

    Parameters:
    hdd_bp (float): The base point for the heating model.
    hdd_beta (float): The beta value for the heating model.
    hdd_k (float): The k value for the heating model.
    cdd_bp (float): The base point for the cooling model.
    cdd_beta (float): The beta value for the cooling model.
    cdd_k (float): The k value for the cooling model.
    intercept (float): The intercept value for the model.
    T_fit_bnds (numpy array): The temperature bounds for the model fitting. Default is an empty numpy array.
    T (numpy array): The temperature values. Default is an empty numpy array.

    Returns:
    numpy array: The total energy consumption for each temperature value in T.
    """

    # if all variables are zero, return tidd model
    if (hdd_beta == 0) and (cdd_beta == 0):
        return np.ones_like(T) * intercept

    [T_min, T_max] = T_fit_bnds

    if cdd_bp < hdd_bp:
        hdd_bp, cdd_bp = cdd_bp, hdd_bp
        hdd_beta, cdd_beta = cdd_beta, hdd_beta
        hdd_k, cdd_k = cdd_k, hdd_k

    E_tot = np.empty_like(T)
    for n, Ti in enumerate(T):
        if (Ti < hdd_bp) or (
            (hdd_bp == cdd_bp) and (cdd_bp >= T_max)
        ):  # Temperature is within the heating model
            T_bp = hdd_bp
            beta = -hdd_beta
            k = hdd_k

        elif (Ti > cdd_bp) or (
            (hdd_bp == cdd_bp) and (hdd_bp <= T_min)
        ):  # Temperature is within the cooling model
            T_bp = cdd_bp
            beta = cdd_beta
            k = -cdd_k

        else:  # Temperature independent
            beta = 0.0

        # Evaluate
        if beta == 0:  # tidd
            E_tot[n] = intercept

        elif k == 0:  # c_hdd
            E_tot[n] = beta * (Ti - T_bp) + intercept

        else:  # smoothed c_hdd
            c_hdd = beta * (Ti - T_bp) + intercept

            exp_interior = 1 / k * (Ti - T_bp)
            exp_interior = np.clip(
                exp_interior, LN_MIN_POS_SYSTEM_VALUE, LN_MAX_POS_SYSTEM_VALUE
            )
            E_tot[n] = abs(beta * k) * (np.exp(exp_interior) - 1) + c_hdd

    return E_tot


@numba.jit(nopython=True, error_model="numpy", cache=True)
def get_full_model_x(model_key, x, T_min, T_max, T_min_seg, T_max_seg):
    """
    This function adjusts the parameters of a full model based on certain conditions.

    Parameters:
    x (list): A list containing the parameters of the model.
    T_min_seg (float): The minimum temperature segment.
    T_max_seg (float): The maximum temperature segment.

    Returns:
    list: A list of adjusted parameters.

    """

    if model_key == "hdd_tidd_cdd_smooth":
        [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept] = x

    elif model_key == "hdd_tidd_cdd":
        [hdd_bp, hdd_beta, cdd_bp, cdd_beta, intercept] = x
        hdd_k = cdd_k = 0.0

    elif model_key == "c_hdd_tidd_smooth":
        [c_hdd_bp, c_hdd_beta, c_hdd_k, intercept] = x
        hdd_bp = cdd_bp = c_hdd_bp

        if c_hdd_beta < 0:
            hdd_beta = -c_hdd_beta
            hdd_k = c_hdd_k
            cdd_beta = cdd_k = 0.0

        else:
            cdd_beta = c_hdd_beta
            cdd_k = c_hdd_k
            hdd_beta = hdd_k = 0.0

    elif model_key == "c_hdd_tidd":
        [c_hdd_bp, c_hdd_beta, intercept] = x

        if c_hdd_bp < T_min_seg:
            cdd_bp = hdd_bp = T_min_seg
        elif c_hdd_bp > T_max_seg:
            cdd_bp = hdd_bp = T_max_seg
        else:
            hdd_bp = cdd_bp = c_hdd_bp

        if c_hdd_beta < 0:
            hdd_beta = -c_hdd_beta
            cdd_beta = cdd_k = hdd_k = 0.0

        else:
            cdd_beta = c_hdd_beta
            hdd_beta = hdd_k = cdd_k = 0.0

    elif model_key == "tidd":
        [intercept] = x
        hdd_bp = hdd_beta = hdd_k = cdd_bp = cdd_beta = cdd_k = 0.0

    x = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

    return fix_full_model_x(x, T_min, T_max)


@numba.jit(nopython=True, error_model="numpy", cache=True)
def fix_full_model_x(x, T_min_seg, T_max_seg):
    """
    This function adjusts the parameters of a full model based on certain conditions.

    Parameters:
    x (list): A list containing the parameters of the model [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept].
    T_min_seg (float): The minimum temperature segment.
    T_max_seg (float): The maximum temperature segment.

    Returns:
    list: A list of adjusted parameters [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept].

    """

    hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept = x

    # swap breakpoint order if they are reversed [hdd, cdd]
    if cdd_bp < hdd_bp:
        hdd_bp, cdd_bp = cdd_bp, hdd_bp
        hdd_beta, cdd_beta = cdd_beta, hdd_beta
        hdd_k, cdd_k = cdd_k, hdd_k

    # if there is a slope, but the breakpoint is at the end, it's a c_hdd_tidd model
    if hdd_bp != cdd_bp:
        if cdd_bp >= T_max_seg:
            cdd_beta = 0.0
        elif hdd_bp <= T_min_seg:
            hdd_beta = 0.0

    # if slopes are zero then smoothing is zero
    if hdd_beta == 0:
        hdd_k = 0.0

    if cdd_beta == 0:
        cdd_k = 0.0

    return [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]


def full_model_weight(
    hdd_bp,
    hdd_beta,
    hdd_k,
    cdd_bp,
    cdd_beta,
    cdd_k,
    intercept,
    T,
    residual,
    sigma=3.0,
    quantile=0.25,
    alpha=2.0,
    min_weight=0.0,
):
    """
    This function calculates the weights, C and alpha for a full model using adaptive weights

    Parameters:
    hdd_bp (float): The base point for heating degree days.
    hdd_beta (float): The beta value for heating degree days.
    hdd_k (float): The k value for heating degree days.
    cdd_bp (float): The base point for cooling degree days.
    cdd_beta (float): The beta value for cooling degree days.
    cdd_k (float): The k value for cooling degree days.
    intercept (float): The intercept of the model.
    T (array-like): The temperature array.
    residual (array-like): The residual array.
    weights (array-like): The input weights array.
    sigma (float, optional): The standard deviation. Default is 3.0.
    quantile (float, optional): The quantile to be used. Default is 0.25.
    alpha (float, optional): The alpha value. Default is 2.0.
    min_weight (float, optional): The minimum weight. Default is 0.0.

    Returns:
    tuple: Returns a tuple containing the weights, C and alpha for the full model.
    """

    if hdd_bp > cdd_bp:
        hdd_bp, cdd_bp = cdd_bp, hdd_bp

    if (hdd_beta == 0) and (cdd_beta == 0):  # intercept only
        resid_all = [residual]

    elif (cdd_bp >= T[-1]) or (hdd_bp <= T[0]):  # hdd or cdd only
        resid_all = [residual]

    elif hdd_beta == 0:
        idx_cdd_bp = np.argmin(np.abs(T - cdd_bp))

        resid_all = [residual[:idx_cdd_bp], residual[idx_cdd_bp:]]

    elif cdd_beta == 0:
        idx_hdd_bp = np.argmin(np.abs(T - hdd_bp))

        resid_all = [residual[:idx_hdd_bp], residual[idx_hdd_bp:]]

    else:
        idx_hdd_bp = np.argmin(np.abs(T - hdd_bp))
        idx_cdd_bp = np.argmin(np.abs(T - cdd_bp))

        if hdd_bp == cdd_bp:
            resid_all = [residual[:idx_hdd_bp], residual[idx_cdd_bp:]]

        else:
            resid_all = [
                residual[:idx_hdd_bp],
                residual[idx_hdd_bp:idx_cdd_bp],
                residual[idx_cdd_bp:],
            ]

    weight = []
    C = []
    a = []
    for resid in resid_all:
        if len(resid) == 0:
            continue

        elif len(resid) < 3:
            weight.append(np.ones_like(resid))
            C.append(np.ones_like(resid))
            a.append(np.ones_like(resid) * 2.0)

            continue

        _weight, _C, _a = adaptive_weights(
            resid, alpha=alpha, sigma=sigma, quantile=quantile, min_weight=min_weight
        )

        weight.append(_weight)
        C.append(np.ones_like(resid) * _C)
        a.append(np.ones_like(resid) * _a)

    weight_out = np.hstack(weight)
    C_out = np.hstack(weight)
    a_out = np.hstack(a)

    return weight_out, C_out, a_out
