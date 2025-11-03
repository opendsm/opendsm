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

from copy import deepcopy as copy

import numpy as np

from opendsm.common.stats.basic import unc_factor
from opendsm.eemeter.models.daily.base_models.full_model import (
    full_model,
    get_full_model_x,
)
from opendsm.eemeter.models.daily.parameters import ModelCoefficients
from opendsm.eemeter.models.daily.utilities.base_model import (
    get_smooth_coeffs,
    get_T_bnds,
)
from opendsm.common.metrics import BaselineMetrics
import pandas as pd

def get_k(X, T_min_seg, T_max_seg):
    """
    Calculates the heating and cooling degree day breakpoints and slopes based on the given input parameters.

    Parameters:
    X (tuple): A tuple containing the following parameters:
        - float: The maximum temperature for the segment.
        - float: The heating degree day value for the segment.
        - float: The minimum temperature for the segment.
        - float: The cooling degree day value for the segment.
    T_min_seg (float): The minimum temperature for the segment.
    T_max_seg (float): The maximum temperature for the segment.

    Returns:
    list: A list containing the following values:
        - float: The heating degree day breakpoint.
        - float: The heating degree day slope.
        - float: The cooling degree day breakpoint.
        - float: The cooling degree day slope.
    """

    [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_smooth_coeffs(*X)

    if X[0] >= T_max_seg:
        hdd_bp = X[0]
        hdd_k = 0.0

        if (cdd_k == 0) and (hdd_k == 0):
            cdd_bp = hdd_bp

    if X[2] <= T_min_seg:
        cdd_bp = X[2]
        cdd_k = 0.0

        if (cdd_k == 0) and (hdd_k == 0):
            hdd_bp = cdd_bp

    return [hdd_bp, hdd_k, cdd_bp, cdd_k]


def reduce_model(
    hdd_bp,
    hdd_beta,
    pct_hdd_k,
    cdd_bp,
    cdd_beta,
    pct_cdd_k,
    intercept,
    T_min,
    T_max,
    T_min_seg,
    T_max_seg,
    model_key,
):
    """
    This function takes in various parameters related to heating degree days (hdd) and cooling degree days (cdd) and
    returns a reduced model based on the values of these parameters. The reduced model is returned as a list of
    coefficients and a list of corresponding values.

    Parameters:
    hdd_bp (float): The heating degree day base point.
    hdd_beta (float): The heating degree day beta value.
    pct_hdd_k (float): The percentage of heating degree days.
    cdd_bp (float): The cooling degree day base point.
    cdd_beta (float): The cooling degree day beta value.
    pct_cdd_k (float): The percentage of cooling degree days.
    intercept (float): The intercept value.
    T_min (float): The minimum temperature value.
    T_max (float): The maximum temperature value.
    T_min_seg (float): The minimum temperature segment value.
    T_max_seg (float): The maximum temperature segment value.
    model_key (str): The key for the model.

    Returns:
    coef_id (list): A list of coefficients for the reduced model.
    x (list): A list of corresponding values for the reduced model.
    """

    if (cdd_beta != 0) and (hdd_beta != 0) and ((pct_cdd_k != 0) or (pct_hdd_k != 0)):
        coef_id = [
            "hdd_bp",
            "hdd_beta",
            "hdd_k",
            "cdd_bp",
            "cdd_beta",
            "cdd_k",
            "intercept",
        ]
        x = [hdd_bp, hdd_beta, pct_hdd_k, cdd_bp, cdd_beta, pct_cdd_k, intercept]

        return coef_id, x

    elif (cdd_beta != 0) and (hdd_beta != 0) and (pct_cdd_k == 0) and (pct_hdd_k == 0):
        coef_id = ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"]
        x = [hdd_bp, hdd_beta, cdd_bp, cdd_beta, intercept]

        return coef_id, x

    if (hdd_beta != 0) and (cdd_beta == 0) and (pct_hdd_k != 0):
        coef_id = ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"]
        if model_key == "hdd_tidd_cdd_smooth":
            [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_k(
                [hdd_bp, pct_hdd_k, cdd_bp, pct_cdd_k], T_min_seg, T_max_seg
            )
            if (hdd_k == 0) and (cdd_k == 0):
                x = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

                return reduce_model(
                    *x, T_min, T_max, T_min_seg, T_max_seg, "c_hdd_tidd_smooth"
                )
        else:
            hdd_k = pct_hdd_k

        hdd_beta = -hdd_beta
        x = [hdd_bp, hdd_beta, hdd_k, intercept]

    elif (hdd_beta == 0) and (cdd_beta != 0) and (pct_cdd_k != 0):
        coef_id = ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"]
        if model_key == "hdd_tidd_cdd_smooth":
            [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_k(
                [hdd_bp, pct_hdd_k, cdd_bp, pct_cdd_k], T_min_seg, T_max_seg
            )
            if (hdd_k == 0) and (cdd_k == 0):
                x = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

                return reduce_model(
                    *x, T_min, T_max, T_min_seg, T_max_seg, "c_hdd_tidd_smooth"
                )

        else:
            cdd_k = pct_cdd_k

        x = [cdd_bp, cdd_beta, cdd_k, intercept]

    elif (hdd_beta != 0) and (cdd_beta == 0) and (pct_hdd_k == 0):
        coef_id = ["c_hdd_bp", "c_hdd_beta", "intercept"]
        if hdd_bp >= T_max_seg:
            hdd_bp = T_max_seg

        hdd_beta = -hdd_beta
        x = [hdd_bp, hdd_beta, intercept]

    elif (hdd_beta == 0) and (cdd_beta != 0) and (pct_cdd_k == 0):
        coef_id = ["c_hdd_bp", "c_hdd_beta", "intercept"]
        if cdd_bp <= T_min_seg:
            cdd_bp = T_min_seg

        x = [cdd_bp, cdd_beta, intercept]

    elif (cdd_beta == 0) and (hdd_beta == 0):
        coef_id = ["intercept"]
        x = [intercept]

    return coef_id, x


# consider rename
class OptimizedResult:
    def __init__(
        self,
        x,
        bnds,
        coef_id,
        loss_alpha,
        C,
        T,
        model,
        weight,
        resid,
        jac,
        mean_loss,
        TSS,
        success,
        message,
        nfev,
        time_elapsed,
        settings,
    ):
        """
        Class representing the results of the optimization procedure, which can either be via Scipy or NLopt.

        Parameters:
            x (numpy.ndarray): Array of optimized coefficients.
            bnds (List[Tuple[float, float]]): List of bounds for each coefficient.
            coef_id (List[str]): List of coefficient names.
            loss_alpha (float): Alpha value for the loss function.
            C (numpy.ndarray): Array of C values.
            T (numpy.ndarray): Array of temperatures.
            model (numpy.ndarray): Array of model values.
            weight (numpy.ndarray): Array of weights.
            resid (numpy.ndarray): Array of residuals.
            jac (numpy.ndarray): Array of jacobian values.
            mean_loss (float): Mean loss value.
            TSS (float): Total sum of squares.
            success (bool): Whether the optimization was successful.
            message (str): Optimization message.
            nfev (int): Number of function evaluations.
            time_elapsed (float): Time elapsed during optimization.
            settings (OptimizationSettings): Optimization settings.
        """

        self.coef_id = coef_id
        self.x = x
        self.num_coeffs = len(x)
        self.bnds = bnds

        self.loss_alpha = loss_alpha
        self.C = C

        self.N = np.shape(T)[0]
        self.T = T
        [self.T_min, self.T_max], [self.T_min_seg, self.T_max_seg] = get_T_bnds(
            T, settings
        )

        self.obs = model - resid
        self.model = model
        self.weight = weight
        self.resid = resid
        self.wSSE = np.sum(weight * resid**2)

        self.baseline_metrics = BaselineMetrics(
            df=pd.DataFrame({
                "observed": self.obs,
                "predicted": self.model,
            }),
            num_model_params=self.num_coeffs
        )
        self.baseline_metrics.wsse = self.wSSE
        self.baseline_metrics.wrmse = np.sqrt(self.wSSE / self.N)

        self.mean_loss = mean_loss
        self.loss = mean_loss * self.N
        self.TSS = TSS

        self.settings = settings

        self.jac = []
        self.cov = []
        self.hess = []
        self.hess_inv = []
        self.x_unc = np.ones_like(x) * -1

        self._prediction_uncertainty()

        if jac is not None:  # for future uncertainty calculations
            self.jac = jac
            self.hess = jac.T * jac

            try:
                self.hess_inv = np.linalg.inv(self.hess)
            except:  # if unable to calculate inverse use Moore-Penrose pseudo-inverse
                self.hess_inv = np.linalg.pinv(self.hess)

            MSE = np.mean(resid**2)
            self.cov = MSE * self.hess_inv

            unc_alpha = self.settings.uncertainty_alpha
            self.x_unc = np.sqrt(np.diag(self.cov)) * unc_factor(
                self.DoF + 1, interval="PI", alpha=unc_alpha
            )

            print()
            print(self.jac)
            print(", ".join([f"{val:.3e}" for val in self.x]))
            print(", ".join([f"{val:.3e}" for val in self.x_unc]))
            print(f"full fcn: {self.f_unc:.2f}")
            print()

        self.success = success
        self.message = message
        self.nfev = nfev
        self.njev = -1
        self.nhev = -1
        self.nit = -1
        self.time_elapsed = time_elapsed * 1e3

        self._set_model_key()
        self._refine_model()

        self.named_coeffs = ModelCoefficients.from_np_arrays(self.x, self.coef_id)

        self.x = np.array(self.x)

    def _prediction_uncertainty(self):  # based on std
        """
        Calculate the prediction uncertainty based on the standard deviation of residuals.
        """

        self.DoF = self.baseline_metrics.ddof_autocorr

        alpha = self.settings.uncertainty_alpha
        f_unc = np.std(self.resid) * unc_factor(
            self.DoF + 1, interval="PI", alpha=alpha
        )
        self.f_unc = f_unc

    def _set_model_key(self):
        """
        Set the model key based on the coefficient names.
        """

        if self.coef_id == [
            "hdd_bp",
            "hdd_beta",
            "hdd_k",
            "cdd_bp",
            "cdd_beta",
            "cdd_k",
            "intercept",
        ]:
            self.model_key = "hdd_tidd_cdd_smooth"
        elif self.coef_id == ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"]:
            self.model_key = "hdd_tidd_cdd"
        elif self.coef_id == ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"]:
            self.model_key = "c_hdd_tidd_smooth"
        elif self.coef_id == ["c_hdd_bp", "c_hdd_beta", "intercept"]:
            self.model_key = "c_hdd_tidd"
        elif self.coef_id == ["intercept"]:
            self.model_key = "tidd"
        else:
            raise Exception(f"Unknown model type in 'OptimizeResult'")

        self.model_name = copy(self.model_key)
        if "c_hdd" in self.model_key:
            if self.x[self.coef_id.index("c_hdd_beta")] < 0:
                self.model_name = self.model_name.replace("c_hdd", "hdd")
            else:
                self.model_name = self.model_name.replace("c_hdd", "cdd")

    def _refine_model(self):
        """
        Refine the model based on the model key and coefficients.
        """
        # update coeffs based on model
        x = get_full_model_x(
            self.model_key,
            self.x,
            self.T_min,
            self.T_max,
            self.T_min_seg,
            self.T_max_seg,
        )

        # reduce model
        self.coef_id, self.x = reduce_model(
            *x, self.T_min, self.T_max, self.T_min_seg, self.T_max_seg, self.model_key
        )
        self.num_coeffs = len(self.x)

        self._set_model_key()

    def eval(self, T):
        """
        Evaluate the full model at given temperature inputs.

        Parameters:
            T (numpy.ndarray): Array of temperatures.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                Tuple containing the following arrays:
                - model: Array of model values.
                - f_unc: Array of uncertainties.
                - hdd_load: Array of heating degree day loads.
                - cdd_load: Array of cooling degree day loads.
        """

        # find if T is an array of floats, else convert to array
        if isinstance(T, (int, float)):
            T = np.array([T]).astype(float)
        elif T.dtype != float:
            T = T.astype(float)

        x = get_full_model_x(
            self.model_key,
            self.x,
            self.T_min,
            self.T_max,
            self.T_min_seg,
            self.T_max_seg,
        )

        if self.model_key == "hdd_tidd_cdd_smooth":
            [hdd_bp, hdd_beta, pct_hdd_k, cdd_bp, cdd_beta, pct_cdd_k, intercept] = x
            [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_smooth_coeffs(
                hdd_bp, pct_hdd_k, cdd_bp, pct_cdd_k
            )
            x = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

        hdd_bp, cdd_bp, intercept = x[0], x[3], x[6]
        T_fit_bnds = np.array([self.T_min, self.T_max])

        model = full_model(*x, T_fit_bnds, T)
        f_unc = np.ones_like(model) * self.f_unc

        load_only = model - intercept

        hdd_load = np.zeros_like(model)
        cdd_load = np.zeros_like(model)

        hdd_idx = np.argwhere(T <= hdd_bp).flatten()
        cdd_idx = np.argwhere(T >= cdd_bp).flatten()

        hdd_load[hdd_idx] = load_only[hdd_idx]
        cdd_load[cdd_idx] = load_only[cdd_idx]

        return model, f_unc, hdd_load, cdd_load
