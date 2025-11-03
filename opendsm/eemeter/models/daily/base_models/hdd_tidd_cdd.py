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

from typing import Optional

import numba
import numpy as np

from opendsm.common.stats.adaptive_loss import adaptive_weights
from opendsm.eemeter.models.daily.base_models.full_model import (
    full_model,
    full_model_weight,
)
from opendsm.eemeter.models.daily.objective_function import obj_fcn_decorator
from opendsm.eemeter.models.daily.utilities.opt_settings import OptimizationSettings
from opendsm.eemeter.models.daily.optimize import InitialGuessOptimizer, Optimizer
from opendsm.eemeter.models.daily.parameters import ModelCoefficients, ModelType
from opendsm.eemeter.models.daily.utilities.base_model import (
    fix_identical_bnds,
    get_intercept,
    get_slope,
    get_smooth_coeffs,
)


def fit_hdd_tidd_cdd(
    T,
    obs,
    weights,
    settings,
    opt_options,
    smooth,
    x0: Optional[ModelCoefficients] = None,
    bnds=None,
    initial_fit=False,
):
    # assert x0 is None or x0.model_type is ModelType.HDD_TIDD_CDD_SMOOTH

    if initial_fit:
        alpha = settings.alpha_selection
    else:
        alpha = settings.alpha_final

    if x0 is None:
        x0 = _hdd_tidd_cdd_smooth_x0(T, obs, alpha, settings, smooth)

    max_slope = np.max([x0.hdd_beta, x0.cdd_beta])
    if max_slope != 0:
        max_slope += 10 ** (
            np.log10(np.abs(max_slope)) + np.log10(settings.maximum_slope_oom_scalar)
        )

    if initial_fit:
        T_min = np.min(T)
        T_max = np.max(T)
    else:
        N_min = settings.segment_minimum_count

        T_min = np.partition(T, N_min)[N_min]
        T_max = np.partition(T, -N_min)[-N_min]

    c_hdd_bnds = [T_min, T_max]
    c_hdd_beta_bnds = [0, np.abs(max_slope)]
    intercept_bnds = np.quantile(obs, [0.01, 0.99])
    if smooth:
        c_hdd_k_bnds = [0, 1]
        bnds_0 = [
            c_hdd_bnds,
            c_hdd_beta_bnds,
            c_hdd_k_bnds,
            c_hdd_bnds,
            c_hdd_beta_bnds,
            c_hdd_k_bnds,
            intercept_bnds,
        ]
    else:
        bnds_0 = [
            c_hdd_bnds,
            c_hdd_beta_bnds,
            c_hdd_bnds,
            c_hdd_beta_bnds,
            intercept_bnds,
        ]

    bnds = _hdd_tidd_cdd_smooth_update_bnds(bnds, bnds_0, smooth)

    if smooth:
        coef_id = [
            "hdd_bp",
            "hdd_beta",
            "hdd_k",
            "cdd_bp",
            "cdd_beta",
            "cdd_k",
            "intercept",
        ]
        model_fcn = evaluate_hdd_tidd_cdd_smooth
        weight_fcn = _hdd_tidd_cdd_smooth_weight
        TSS_fcn = None
    else:
        coef_id = ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"]
        model_fcn = _hdd_tidd_cdd
        weight_fcn = _hdd_tidd_cdd_weight
        TSS_fcn = _hdd_tidd_cdd_total_sum_of_squares

    obj_fcn = obj_fcn_decorator(
        model_fcn, weight_fcn, TSS_fcn, T, obs, weights, settings, alpha, coef_id, initial_fit
    )

    res = Optimizer(
        obj_fcn, x0.to_np_array(), bnds, coef_id, settings, opt_options
    ).run()

    return res


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _hdd_tidd_cdd(
    hdd_bp,
    hdd_beta,
    cdd_bp,
    cdd_beta,
    intercept,
    T_fit_bnds=np.array([]),
    T=np.array([]),
):
    hdd_k = cdd_k = 0

    return full_model(
        hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T
    )


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _hdd_tidd_cdd_smooth(*args):
    return full_model(*args)


def evaluate_hdd_tidd_cdd_smooth(
    hdd_bp,
    hdd_beta,
    hdd_k,
    cdd_bp,
    cdd_beta,
    cdd_k,
    intercept,
    T_fit_bnds,
    T,
    pct_k=True,
):
    if pct_k:
        [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_smooth_coeffs(hdd_bp, hdd_k, cdd_bp, cdd_k)

    return _hdd_tidd_cdd_smooth(
        hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T
    )


def _hdd_tidd_cdd_smooth_x0(T, obs, alpha, settings, smooth, min_weight=0.0):
    min_T_idx = settings.segment_minimum_count
    lasso_a = settings.regularization_alpha

    idx_sorted = np.argsort(T).flatten()
    T = T[idx_sorted]
    obs = obs[idx_sorted]

    N = len(obs)

    T_fit_bnds = np.array([T[0], T[-1]])

    def bp_obj_fcn_dec(T, obs, min_T_idx):
        def lasso_penalty(X, wRMSE):
            X_lasso = np.array(X).copy()

            T_range = T_fit_bnds[1] - T_fit_bnds[0]

            X_lasso = np.array(
                [np.min(np.abs(X[idx] - T_fit_bnds)) for idx in range(len(X))]
            )
            X_lasso += (X[1] - X[0]) / 2
            X_lasso *= wRMSE / T_range

            return lasso_a * np.linalg.norm(X_lasso, 1)

        def bp_obj_fcn(x, grad=[], optimize_flag=True):
            if len(x) == 1:
                hdd_bp = cdd_bp = x[0]
            else:
                if x[0] < x[1]:
                    [hdd_bp, cdd_bp] = x
                else:
                    [cdd_bp, hdd_bp] = x

            hdd_beta, cdd_beta, intercept = estimate_betas_and_intercept(
                T, obs, hdd_bp, cdd_bp, min_T_idx, alpha
            )
            hdd_k = cdd_k = 0

            model = _hdd_tidd_cdd_smooth(
                hdd_bp,
                hdd_beta,
                hdd_k,
                cdd_bp,
                cdd_beta,
                cdd_k,
                intercept,
                T_fit_bnds,
                T,
            )
            resid = model - obs

            if alpha == 2:
                resid_mean = np.mean(resid)
                resid -= resid_mean
                intercept += resid_mean
            else:
                resid_median = np.median(resid)
                resid -= resid_median
                intercept += resid_median

            weight, _, _ = adaptive_weights(
                resid, alpha=alpha, sigma=2.698, quantile=0.25, min_weight=min_weight
            )

            loss = np.sum(weight * (resid) ** 2)
            loss += lasso_penalty(x, np.sqrt(loss / N))

            if optimize_flag:
                return loss

            return np.array(
                [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]
            )

        return bp_obj_fcn

    obj_fcn = bp_obj_fcn_dec(T, obs, min_T_idx)

    T_bnds = [T[min_T_idx - 1], T[-min_T_idx]]
    if T_bnds[0] == T_bnds[1]:
        T_bnds = [
            np.min(T),
            np.max(T),
        ]  # should be able to do [0] and [-1] but getting error where min > max

    if T_bnds[1] < T_bnds[0]:
        T_bnds = [T_bnds[1], T_bnds[0]]

    T_min = T_bnds[0]
    T_max = T_bnds[1]
    T_range = T_max - T_min

    x0 = np.array([T_range * 0.10, T_range * 0.90]) + T_min
    bnds = np.array([T_bnds, T_bnds])

    opt_settings = OptimizationSettings(
        algorithm=settings.initial_guess_algorithm_choice,
        stop_criteria_type="iteration maximum",
        stop_criteria_value=200,
        initial_step=settings.initial_step_percentage,
        x_tol_rel=1e-3,
        f_tol_rel=0.5,
    )

    res = InitialGuessOptimizer(
        obj_fcn, x0, bnds, opt_settings
    ).run()

    x0 = obj_fcn(res.x, optimize_flag=False)

    if smooth:
        model_type = ModelType.HDD_TIDD_CDD_SMOOTH
        hdd_k = x0[2]
        cdd_k = x0[5]
    else:
        model_type = ModelType.HDD_TIDD_CDD
        hdd_k = cdd_k = None

    return ModelCoefficients(
        model_type=model_type,
        hdd_bp=x0[0],
        hdd_beta=x0[1],
        hdd_k=hdd_k,
        cdd_bp=x0[3],
        cdd_beta=x0[4],
        cdd_k=cdd_k,
        intercept=x0[6],
    )


def estimate_betas_and_intercept(T, obs, hdd_bp, cdd_bp, min_T_idx, alpha):
    idx_hdd = np.argwhere(T < hdd_bp).flatten()
    idx_tidd = np.argwhere((hdd_bp <= T) & (T <= cdd_bp)).flatten()
    idx_cdd = np.argwhere(cdd_bp < T).flatten()

    if len(idx_tidd) > 0:
        intercept = get_intercept(obs[idx_tidd], alpha)
    elif (
        (len(idx_cdd) >= min_T_idx)
        and (len(idx_hdd) >= min_T_idx)
        and (idx_cdd[min_T_idx - 1] - idx_hdd[-min_T_idx]) > 0
    ):
        intercept = get_intercept(
            obs[idx_hdd[-min_T_idx] : idx_cdd[min_T_idx - 1]], alpha
        )
    else:
        intercept = np.quantile(obs, 0.20)

    hdd_beta = get_slope(T[idx_hdd], obs[idx_hdd], hdd_bp, intercept, alpha)
    if hdd_beta > 0:
        hdd_beta = 0
    else:
        hdd_beta *= -1

    cdd_beta = get_slope(T[idx_cdd], obs[idx_cdd], cdd_bp, intercept, alpha)
    if cdd_beta < 0:
        cdd_beta = 0

    return hdd_beta, cdd_beta, intercept


def _hdd_tidd_cdd_smooth_update_bnds(new_bnds, bnds, smooth):
    if new_bnds is None:
        new_bnds = bnds

    # breakpoint bounds
    new_bnds[0] = bnds[0]
    if smooth:
        new_bnds[3] = bnds[3]
    else:
        new_bnds[2] = bnds[2]

    # intercept bounds at index 6 for smooth, 4 for unsmooth
    if smooth:
        new_bnds[6] = bnds[6]
    else:
        new_bnds[4] = bnds[4]

    new_bnds = np.sort(new_bnds, axis=1)
    new_bnds = fix_identical_bnds(new_bnds)

    # beta and k must be non-negative
    if smooth:
        beta_k_idx = [1, 2, 4, 5]
    else:
        beta_k_idx = [1, 3]
    for i in beta_k_idx:
        if new_bnds[i][0] < 0:
            new_bnds[i][0] = 0

    return new_bnds


def _hdd_tidd_cdd_weight(
    hdd_bp,
    hdd_beta,
    cdd_bp,
    cdd_beta,
    intercept,
    T,
    residual,
    sigma=3.0,
    quantile=0.25,
    alpha=2.0,
    min_weight=0.0,
):
    hdd_k = cdd_k = 0
    model_vars = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

    return full_model_weight(
        *model_vars, T, residual, sigma, quantile, alpha, min_weight
    )


def _hdd_tidd_cdd_smooth_weight(
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
    model_vars = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

    return full_model_weight(
        *model_vars, T, residual, sigma, quantile, alpha, min_weight
    )


def _hdd_tidd_cdd_total_sum_of_squares(
    hdd_bp, hdd_beta, cdd_bp, cdd_beta, intercept, T, obs
):
    if hdd_bp > cdd_bp:
        hdd_bp, cdd_bp = cdd_bp, hdd_bp

    idx_hdd_bp = np.argmin(np.abs(T - hdd_bp))
    idx_cdd_bp = np.argmin(np.abs(T - cdd_bp))

    TSS = []
    for observed in [obs[:idx_hdd_bp], obs[idx_hdd_bp:idx_cdd_bp], obs[idx_cdd_bp:]]:
        if len(observed) == 0:
            continue

        TSS.append(np.sum((observed - np.mean(observed)) ** 2))

    TSS = np.sum(TSS)

    return TSS
