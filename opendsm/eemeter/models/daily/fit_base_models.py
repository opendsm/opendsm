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

from opendsm.common.utils import OoM
from opendsm.eemeter.models.daily.base_models.c_hdd_tidd import fit_c_hdd_tidd
from opendsm.eemeter.models.daily.base_models.hdd_tidd_cdd import fit_hdd_tidd_cdd
from opendsm.eemeter.models.daily.base_models.tidd import fit_tidd
from opendsm.eemeter.models.daily.optimize_results import OptimizedResult
from opendsm.eemeter.models.daily.parameters import ModelCoefficients
from opendsm.eemeter.models.daily.utilities.settings import FullModelSelection
from opendsm.eemeter.models.daily.utilities.opt_settings import OptimizationSettings


def _get_opt_settings(settings):
    """
    Returns a dictionary containing optimization options for the global and local optimization algorithms.

    Parameters:
        settings: A DailySettings object containing the settings for the optimization algorithm.

    Returns:
        A dictionary containing the optimization options for the optimization algorithm.
    """

    opt_dict = {
        "ALGORITHM": settings.algorithm_choice,
        "INITIAL_STEP": settings.initial_step_percentage,
    }
    opt_settings = OptimizationSettings(**opt_dict)

    return opt_settings


def fit_initial_models_from_full_model(df_meter, settings, print_res=False):
    """
    Fits initial models from the full model based on the given settings.

    Parameters:
        df_meter (pandas.DataFrame): The meter data to fit the models to. Columns : date, observed, temperature
        settings (Settings): The settings object containing the model selection and fitting options.
        print_res (bool, optional): Whether to print the results of the model fitting. Defaults to False.

    Returns:
        ModelResult: The result of the model fitting.
    """

    T = df_meter["temperature"].values
    obs = df_meter["observed"].values

    if "weights" in df_meter.columns:
        weights = df_meter["weights"].values
    else:
        weights = None

    opt_settings = _get_opt_settings(settings)
    fit_input = [T, obs, weights, settings, opt_settings]

    # initial fitting of the most complicated model allowed
    if settings.full_model == FullModelSelection.HDD_TIDD_CDD:
        model_res = fit_hdd_tidd_cdd(
            *fit_input, smooth=settings.allow_smooth_model, initial_fit=True
        )
    elif settings.full_model == FullModelSelection.C_HDD_TIDD:
        model_res = fit_c_hdd_tidd(
            *fit_input, smooth=settings.allow_smooth_model, initial_fit=True
        )
    elif settings.full_model == FullModelSelection.TIDD:
        model_res = fit_tidd(*fit_input, initial_fit=True)

    if print_res:
        criterion = model_res.selection_criterion
        # print(f"{model_key:<30s} {model_res.loss:<8.3f} {model_res.alpha:<8.2f} {model_res.C:<8.3f} {model_res.time_elapsed:>8.2f} ms")
        print(
            f"{model_res.model_name:<30s} {criterion:<8.3g} {model_res.alpha:<8.2f} {model_res.time_elapsed:>8.2f} ms"
        )

    return model_res


def fit_model(model_key, fit_input, x0: ModelCoefficients, bnds):
    """
    Fits a model based on the given model key and input data.

    Args:
        model_key (str): The key for the model to be fitted.
        fit_input (tuple): The input data for the model.
        x0 (ModelCoefficients): The initial coefficients for the model.
        bnds (tuple): The bounds for the model coefficients.

    Returns:
        The result of the model fitting.
    """

    if model_key == "hdd_tidd_cdd_smooth":
        res = fit_hdd_tidd_cdd(
            *fit_input, smooth=True, x0=x0, bnds=bnds, initial_fit=False
        )

    elif model_key == "hdd_tidd_cdd":
        res = fit_hdd_tidd_cdd(
            *fit_input, smooth=False, x0=x0, bnds=bnds, initial_fit=False
        )

    elif model_key == "c_hdd_tidd_smooth":
        res = fit_c_hdd_tidd(
            *fit_input, smooth=True, x0=x0, bnds=bnds, initial_fit=False
        )

    elif model_key == "c_hdd_tidd":
        res = fit_c_hdd_tidd(
            *fit_input, smooth=False, x0=x0, bnds=bnds, initial_fit=False
        )

    elif model_key == "tidd":
        res = fit_tidd(*fit_input, x0, bnds, initial_fit=False)

    return res


def fit_final_model(df_meter, HoF: OptimizedResult, settings, print_res=False):
    """
    Fits the final model using the optimized result and returns the optimized result with updated coefficients.
    HoF (Hall of Fame) denotes the optimized results.

    Args:
        df_meter (pandas.DataFrame): DataFrame containing temperature and observed values.
        HoF (OptimizedResult): OptimizedResult object containing the optimized model and coefficients.
        settings (Settings): DailySettings object containing the settings for the model fitting.
        print_res (bool, optional): Whether to print the results. Defaults to False.

    Returns:
        OptimizedResult: OptimizedResult object with updated coefficients.
    """

    def get_bnds(x0, bnds_scalar):
        x_oom = 10 ** (OoM(x0, method="exact") + np.log10(bnds_scalar))
        bnds = (x0 + (np.array([-1, 1]) * x_oom[:, None]).T).T

        return bnds

    T = df_meter["temperature"].values
    obs = df_meter["observed"].values

    if "weights" in df_meter.columns:
        weights = df_meter["weights"].values
    else:
        weights = None

    opt_settings = _get_opt_settings(settings)
    fit_input = [T, obs, weights, settings, opt_settings]

    x0 = HoF.x
    bnds = get_bnds(x0, settings.final_bounds_scalar)

    HoF = fit_model(HoF.model_key, fit_input, HoF.named_coeffs, bnds)

    if print_res:
        print(
            f"{HoF.model_name:<30s} {HoF.loss_alpha:<8.2f} {HoF.time_elapsed:>8.2f} ms"
        )

    return HoF
