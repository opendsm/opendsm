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

from math import isclose
from typing import Optional

import numba
import numpy as np

from opendsm.common.adaptive_loss import adaptive_weights
from opendsm.eemeter.models.daily.base_models.full_model import full_model
from opendsm.eemeter.models.daily.base_models.hdd_tidd_cdd import full_model_weight
from opendsm.eemeter.models.daily.objective_function import obj_fcn_decorator
from opendsm.eemeter.models.daily.utilities.opt_settings import OptimizationSettings
from opendsm.eemeter.models.daily.optimize import InitialGuessOptimizer, Optimizer
from opendsm.eemeter.models.daily.parameters import ModelCoefficients, ModelType
from opendsm.eemeter.models.daily.utilities.base_model import (
    fix_identical_bnds,
    get_intercept,
    get_slope,
    get_T_bnds,
    linear_fit,
)


def fit_c_hdd_tidd(
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
    """
    This function fits the HDD TIDD smooth model to the given data.
    Parameters:
    T (array-like): The independent variable data - temperature.
    obs (array-like): The dependent variable data - observed.
    settings (object): An object containing various settings for the model fitting.
    opt_options (dict): A dictionary containing options for the optimization process.
    x0 (ModelCoefficients, optional): Initial model coefficients. If None, they will be estimated.
    bnds (list of tuples, optional): Bounds for the optimization process. If None, they will be estimated.
    initial_fit (bool, optional): If True, the function performs an initial fit. Default is False.
    Returns:
    res (OptimizeResult): The result of the optimization process.
    """

    if initial_fit:
        alpha = settings.alpha_selection
    else:
        alpha = settings.alpha_final

    if x0 is None:
        x0 = _c_hdd_tidd_x0(T, obs, alpha, settings, smooth)
    else:
        x0 = _c_hdd_tidd_x0_final(T, obs, x0, alpha, settings)

    if x0.model_type in [ModelType.HDD_TIDD_SMOOTH, ModelType.HDD_TIDD]:
        tdd_beta = x0.hdd_beta
    elif x0.model_type in [ModelType.TIDD_CDD_SMOOTH, ModelType.TIDD_CDD]:
        tdd_beta = x0.cdd_beta
    else:
        raise ValueError

    # limit slope based on initial regression & configurable order of magnitude
    max_slope = np.abs(tdd_beta) + 10 ** (
        np.log10(np.abs(tdd_beta)) + np.log10(settings.maximum_slope_oom_scalar)
    )

    # initial fit bounded by Tmin:Tmax, final fit has minimum T segment buffer
    T_initial, T_segment = get_T_bnds(T, settings)
    c_hdd_bnds = T_initial if initial_fit else T_segment

    # set bounds and alter coefficient guess for single slope models w/o an intercept segment
    if not smooth and not initial_fit:
        T_min, T_max = T_initial
        T_min_seg, T_max_seg = T_segment
        rtol = 1e-5
        if x0.model_type is ModelType.HDD_TIDD and (
            x0.hdd_bp >= T_max_seg or isclose(x0.hdd_bp, T_max_seg, rel_tol=rtol)
        ):
            # model is heating only, and breakpoint is approximately within max temp buffer
            x0.intercept -= x0.hdd_bp * T_max
            x0.hdd_bp = T_max
            c_hdd_bnds = [T_max, T_max]
        if x0.model_type is ModelType.TIDD_CDD and (
            x0.cdd_bp <= T_min_seg or isclose(x0.cdd_bp, T_min_seg, rel_tol=rtol)
        ):
            # model is cooling only, and breakpoint is approximately within min temp buffer
            x0.intercept -= x0.cdd_bp * T_min
            x0.cdd_bp = T_min
            c_hdd_bnds = [T_min, T_min]

    # not known whether heating or cooling model on initial fit
    if initial_fit:
        c_hdd_beta_bnds = [-max_slope, max_slope]
    # stick with heating/cooling if using existing x0
    elif tdd_beta < 0:
        c_hdd_beta_bnds = [-max_slope, 0]
    else:
        c_hdd_beta_bnds = [0, max_slope]

    intercept_bnds = np.quantile(obs, [0.01, 0.99])
    if smooth:
        c_hdd_k_bnds = [0, 1e3]
        bnds_0 = [c_hdd_bnds, c_hdd_beta_bnds, c_hdd_k_bnds, intercept_bnds]
    else:
        bnds_0 = [c_hdd_bnds, c_hdd_beta_bnds, intercept_bnds]

    bnds = _c_hdd_tidd_update_bnds(bnds, bnds_0, smooth)
    if (
        c_hdd_bnds[0] == c_hdd_bnds[1]
    ):  # if breakpoint bounds are identical, don't expand
        bnds[0, :] = c_hdd_bnds

    if smooth:
        coef_id = ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"]
        model_fcn = _c_hdd_tidd_smooth
        weight_fcn = _c_hdd_tidd_smooth_weight
        TSS_fcn = None
    else:
        coef_id = ["c_hdd_bp", "c_hdd_beta", "intercept"]
        model_fcn = _c_hdd_tidd
        weight_fcn = _c_hdd_tidd_weight
        TSS_fcn = _c_hdd_tidd_total_sum_of_squares
    obj_fcn = obj_fcn_decorator(
        model_fcn, weight_fcn, TSS_fcn, T, obs, weights, settings, alpha, coef_id, initial_fit
    )
    res = Optimizer(
        obj_fcn, x0.to_np_array(), bnds, coef_id, settings, opt_options
    ).run()

    return res


@numba.jit(nopython=True, error_model="numpy", cache=True)
def set_full_model_coeffs_smooth(c_hdd_bp, c_hdd_beta, c_hdd_k, intercept):
    """
    This function sets the smoothed full model coefficients based on the given parameters.
    Parameters:
    c_hdd_bp (float): The base point coefficient for heating and cooling degree days.
    c_hdd_beta (float): The beta coefficient for heating and cooling degree days.
    c_hdd_k (float): The k coefficient for heating and cooling degree days.
    intercept (float): The intercept of the model.
    Returns:
    np.array: An array containing the coefficients for the full model.
    """
    hdd_bp = cdd_bp = c_hdd_bp

    if c_hdd_beta < 0:
        hdd_beta = -c_hdd_beta
        hdd_k = c_hdd_k
        cdd_beta = cdd_k = 0.0

    else:
        cdd_beta = c_hdd_beta
        cdd_k = c_hdd_k
        hdd_beta = hdd_k = 0.0

    return np.array([hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept])


@numba.jit(nopython=True, error_model="numpy", cache=True)
def set_full_model_coeffs(c_hdd_bp, c_hdd_beta, intercept):
    """
    This function sets the full model coefficients based on the given parameters.
    Parameters:
    c_hdd_bp (float): The base point coefficient for heating and cooling degree days.
    c_hdd_beta (float): The beta coefficient for heating and cooling degree days.
    intercept (float): The intercept of the model.
    Returns:
    np.array: An array containing the coefficients for the full model.
    """

    return set_full_model_coeffs_smooth(c_hdd_bp, c_hdd_beta, 0.0, intercept)


def _c_hdd_tidd_update_bnds(new_bnds, bnds, smooth):
    """
    This function updates the boundaries of the new_bnds array based on the given bnds array.
    It sorts the new_bnds array along the axis=1, fixes any identical boundaries, and ensures that the lower boundary is non-negative.
    Parameters:
    new_bnds (numpy.ndarray): The array of new boundaries to be updated.
    bnds (numpy.ndarray): The array of existing boundaries used for updating.
    Returns:
    new_bnds (numpy.ndarray): The updated array of new boundaries.
    """

    if new_bnds is None:
        new_bnds = bnds

    # breakpoint bounds
    new_bnds[0] = bnds[0]

    # intercept bnds at index 3 for smooth, 2 for unsmooth
    if smooth:
        new_bnds[3] = bnds[3]
    else:
        new_bnds[2] = bnds[2]

    new_bnds = np.sort(new_bnds, axis=1)
    new_bnds = fix_identical_bnds(new_bnds)

    # check for negative k bound if using smoothed model
    if smooth and new_bnds[2, 0] < 0:
        new_bnds[2, 0] = 0

    return new_bnds


def _tdd_coefficients(
    intercept, c_hdd_bp, c_hdd_beta, c_hdd_k=None
) -> ModelCoefficients:
    """
    infer cdd vs hdd given positive or negative slope.
    if slope is 0, model will be reduced later
    """
    if c_hdd_beta < 0:
        hdd_beta = c_hdd_beta
        hdd_bp = c_hdd_bp
        hdd_k = c_hdd_k
        cdd_beta = None
        cdd_bp = None
        cdd_k = None
        if c_hdd_k is not None:
            model_type = ModelType.HDD_TIDD_SMOOTH
        else:
            model_type = ModelType.HDD_TIDD
    else:
        cdd_beta = c_hdd_beta
        cdd_bp = c_hdd_bp
        cdd_k = c_hdd_k
        hdd_beta = None
        hdd_bp = None
        hdd_k = None
        if c_hdd_k is not None:
            model_type = ModelType.TIDD_CDD_SMOOTH
        else:
            model_type = ModelType.TIDD_CDD

    return ModelCoefficients(
        model_type=model_type,
        intercept=intercept,
        hdd_bp=hdd_bp,
        hdd_beta=hdd_beta,
        hdd_k=hdd_k,
        cdd_bp=cdd_bp,
        cdd_beta=cdd_beta,
        cdd_k=cdd_k,
    )


def _c_hdd_tidd_x0(T, obs, alpha, settings, smooth):
    min_T_idx = settings.segment_minimum_count

    # c_hdd_bp = initial_guess_bp_1(T, obs, s=2, int_method="trapezoid")
    c_hdd_bp = _c_hdd_tidd_bp0(T, obs, alpha, settings)
    c_hdd_bp = np.clip([c_hdd_bp], T[min_T_idx - 1], T[-min_T_idx])[0]

    idx_hdd = np.argwhere(T <= c_hdd_bp).flatten()
    idx_cdd = np.argwhere(T >= c_hdd_bp).flatten()

    hdd_beta, _ = linear_fit(T[idx_hdd], obs[idx_hdd], alpha)
    if hdd_beta > 0:
        hdd_beta = 0

    cdd_beta, _ = linear_fit(T[idx_cdd], obs[idx_cdd], alpha)
    if cdd_beta < 0:
        cdd_beta = 0

    # choose heating vs cooling based on larger slope
    # treat opposite degree days as flat tidd
    if -hdd_beta >= cdd_beta:
        c_hdd_beta = hdd_beta
        intercept = np.median(obs[idx_cdd])

    else:
        c_hdd_beta = cdd_beta
        intercept = np.median(obs[idx_hdd])

    c_hdd_k = None
    if smooth:
        c_hdd_k = 0.0

    return _tdd_coefficients(
        intercept=intercept,
        c_hdd_bp=c_hdd_bp,
        c_hdd_beta=c_hdd_beta,
        c_hdd_k=c_hdd_k,
    )


def _c_hdd_tidd_x0_final(T, obs, x0, alpha, settings):
    c_hdd_k = None
    if x0.is_smooth:
        c_hdd_bp, c_hdd_beta, c_hdd_k, intercept = x0.to_np_array()
    else:
        c_hdd_bp, c_hdd_beta, intercept = x0.to_np_array()

    min_T_idx = settings.segment_minimum_count
    idx_hdd = np.argwhere(T <= c_hdd_bp).flatten()
    idx_cdd = np.argwhere(T >= c_hdd_bp).flatten()

    # can use model type to do this
    # if x0.model_type in [ModelType.HDD_TIDD_SMOOTH, ModelType.HDD_TIDD]:  etc
    if (c_hdd_beta < 0) and (len(idx_hdd) >= min_T_idx):  # hdd
        c_hdd_beta = get_slope(T[idx_hdd], obs[idx_hdd], c_hdd_bp, intercept, alpha)

    elif (c_hdd_beta >= 0) and (len(idx_cdd) >= min_T_idx):  # cdd
        c_hdd_beta = get_slope(T[idx_cdd], obs[idx_cdd], c_hdd_bp, intercept, alpha)

    return _tdd_coefficients(
        c_hdd_bp=c_hdd_bp, c_hdd_beta=c_hdd_beta, c_hdd_k=c_hdd_k, intercept=intercept
    )


def _c_hdd_tidd_bp0(T, obs, alpha, settings, min_weight=0.0):
    min_T_idx = settings.segment_minimum_count

    idx_sorted = np.argsort(T).flatten()
    T = T[idx_sorted]
    obs = obs[idx_sorted]

    T_fit_bnds = np.array([T[0], T[-1]])

    def bp_obj_fcn_dec(T, obs):
        def bp_obj_fcn(x, grad=[]):
            [c_hdd_bp] = x

            idx_hdd = np.argwhere(T <= c_hdd_bp).flatten()
            idx_cdd = np.argwhere(T >= c_hdd_bp).flatten()

            hdd_beta, _ = linear_fit(T[idx_hdd], obs[idx_hdd], alpha)
            if hdd_beta > 0:
                hdd_beta = 0

            cdd_beta, _ = linear_fit(T[idx_cdd], obs[idx_cdd], alpha)
            if cdd_beta < 0:
                cdd_beta = 0

            if -hdd_beta >= cdd_beta:
                c_hdd_beta = hdd_beta
                intercept = get_intercept(obs[idx_cdd], alpha)

            else:
                c_hdd_beta = cdd_beta
                intercept = get_intercept(obs[idx_hdd], alpha)

            model = _c_hdd_tidd(
                c_hdd_bp, c_hdd_beta, intercept, T_fit_bnds=T_fit_bnds, T=T
            )

            resid = model - obs
            weight, _, _ = adaptive_weights(
                resid, alpha=alpha, sigma=2.698, quantile=0.25, min_weight=min_weight
            )

            loss = np.sum(weight * (resid) ** 2)

            return loss

        return bp_obj_fcn

    obj_fcn = bp_obj_fcn_dec(T, obs)

    T_min = T[min_T_idx - 1]
    T_max = T[-min_T_idx]
    T_range = T_max - T_min

    x0 = np.array([T_range * 0.5]) + T_min
    bnds = np.array([[T_min, T_max]])

    opt_settings = OptimizationSettings(
        algorithm=settings.initial_guess_algorithm_choice,
        stop_criteria_type="iteration maximum",
        stop_criteria_value=100,
        initial_step=settings.initial_step_percentage,
        x_tol_rel=1e-3,
        f_tol_rel=0.5,
    )

    res = InitialGuessOptimizer(
        obj_fcn, x0, bnds, opt_settings
    ).run()

    return res.x[0]


def _c_hdd_tidd(
    c_hdd_bp, c_hdd_beta, intercept, T_fit_bnds=np.array([]), T=np.array([])
):
    model_vars = set_full_model_coeffs(c_hdd_bp, c_hdd_beta, intercept)
    return full_model(*model_vars, T_fit_bnds, T)


def _c_hdd_tidd_smooth(
    c_hdd_bp, c_hdd_beta, c_hdd_k, intercept, T_fit_bnds=np.array([]), T=np.array([])
):
    x = set_full_model_coeffs_smooth(c_hdd_bp, c_hdd_beta, c_hdd_k, intercept)
    return full_model(*x, T_fit_bnds, T)


def _c_hdd_tidd_weight(
    c_hdd_bp,
    c_hdd_beta,
    intercept,
    T,
    residual,
    sigma=3.0,
    quantile=0.25,
    alpha=2.0,
    min_weight=0.0,
):
    model_vars = set_full_model_coeffs(c_hdd_bp, c_hdd_beta, intercept)
    return full_model_weight(
        *model_vars, T, residual, sigma, quantile, alpha, min_weight
    )


def _c_hdd_tidd_smooth_weight(
    c_hdd_bp,
    c_hdd_beta,
    c_hdd_k,
    intercept,
    T,
    residual,
    sigma=3.0,
    quantile=0.25,
    alpha=2.0,
    min_weight=0.0,
):
    """
    This function calculates the weight for the full model using the given parameters.
    Parameters:
    c_hdd_bp (float): The base point for the HDD.
    c_hdd_beta (float): The beta value for the HDD.
    c_hdd_k (float): The k value for the HDD.
    intercept (float): The intercept for the model.
    T (float): The temperature.
    residual (float): The residual value.
    sigma (float, optional): The sigma value. Default is 3.0.
    quantile (float, optional): The quantile value. Default is 0.25.
    alpha (float, optional): The alpha value. Default is 2.0.
    min_weight (float, optional): The minimum weight. Default is 0.0.
    Returns:
    float: The calculated weight for the full model.
    """

    model_vars = set_full_model_coeffs_smooth(c_hdd_bp, c_hdd_beta, c_hdd_k, intercept)
    return full_model_weight(
        *model_vars, T, residual, sigma, quantile, alpha, min_weight
    )


def _c_hdd_tidd_total_sum_of_squares(c_hdd_bp, c_hdd_beta, intercept, T, obs):
    idx_bp = np.argmin(np.abs(T - c_hdd_bp))

    TSS = []
    for observed in [obs[:idx_bp], obs[idx_bp:]]:
        if len(observed) == 0:
            continue

        TSS.append(np.sum((observed - np.mean(observed)) ** 2))

    TSS = np.sum(TSS)

    return TSS
