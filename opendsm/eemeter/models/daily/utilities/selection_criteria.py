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


def neg_log_likelihood(loss, N):
    """
    This function calculates the negative log likelihood for least squares fitting.

    Parameters:
    loss (float): The sum of squared residuals.
    N (int): The number of data points.

    Returns:
    float: The negative log likelihood of the least squares fit.
    """

    # log likelihood for n independent identical normal distributions:
    # log_likelihood = -n/2*np.log(2*np.pi) - n/2*np.log(sigma**2) - 1/(2*sigma**2)*np.sum((x - mu)**2)

    if loss <= 0 or N <= 0:
        return np.inf

    # log likelihood for for least squares fitting:

    res = -N / 2 * (np.log(2 * np.pi) + np.log(loss / N) + 1)

    return res


def selection_criteria(
    loss,
    TSS,
    N,
    num_coeffs,
    model_selection_criteria="bic",
    penalty_multiplier=1.0,
    penalty_power=1.0,
):
    """
    This function calculates the selection criteria for a given model. There are different criteria that can be used,
    and the default is the Bayesian information criterion (BIC).

    Parameters:
    loss (float): The loss of the model.
    TSS (float): Total sum of squares.
    N (int): The number of observations.
    num_coeffs (int): The number of coefficients in the model.
    model_selection_criteria (str): The model selection criteria to use. Default is "bic".
    penalty_multiplier (float): The penalty multiplier. Default is 1.0.
    penalty_power (float): The penalty power. Default is 1.0.

    Returns:
    float: The calculated selection criteria.

    Raises:
    NotImplementedError: If the model selection criteria is "dic", "waic", or "wbic", as these are not implemented.
    """

    K = num_coeffs  # total number of coefficients
    c0 = penalty_multiplier
    d0 = penalty_power

    df_penalized = N - K - 1
    if df_penalized <= 0:
        df_penalized = 1e-6

    # Root-mean-square error adjusted
    if model_selection_criteria.lower() == "rmse":
        criteria = np.sqrt(loss / N)

    # Root-mean-square error adjusted
    elif model_selection_criteria.lower() == "rmse_adj":
        criteria = np.sqrt(loss / df_penalized)

    # 1 - R_squared (because we minimize)
    elif model_selection_criteria.lower() == "r_squared":
        r_squared = 1 - loss / TSS

        criteria = (1 - r_squared) * 100

    elif model_selection_criteria.lower() == "r_squared_adj":
        r_squared = 1 - loss / TSS
        r_squared_adj = 1 - (1 - r_squared) * ((N - 1) / df_penalized)
        criteria = (1 - r_squared_adj) * 100

    # Final prediction error
    elif model_selection_criteria.lower() == "fpe":
        criteria = loss * (N + K + 1) / df_penalized
        # penalized_loss = np.exp(-2/N*log_likelihood)*(N + K)/(N - K)

    # Akaike (ah-kah-ee-kay)
    # Akaike information criterion - Akaike (1973, 1974, 1981)
    elif model_selection_criteria.lower() == "aic":
        criteria = -2 * neg_log_likelihood(loss, N) + c0 * 2 * K**d0

    # Akaike information criterion corrected - Hurvich and Tsai (1989)
    elif model_selection_criteria.lower() == "aicc":
        criteria = (
            -2 * neg_log_likelihood(loss, N)
            + c0 * (2 * K + (2 * K * (K + 1) / df_penalized)) ** d0
        )

    # Consistent Akaike information criterion - Bozdogan (1987)
    elif model_selection_criteria.lower() == "caic":
        criteria = -2 * neg_log_likelihood(loss, N) + c0 * K * (np.log(N) + 1) ** d0

    # Bayesian information criterion
    elif model_selection_criteria.lower() == "bic":
        # if c0 = 0.299 and d0 = 2.1, this is the same as Liu, We, Zidek
        criteria = -2 * neg_log_likelihood(loss, N) + c0 * K * np.log(N) ** d0

    # Sample-size adjusted Bayesian information criteria
    elif model_selection_criteria.lower() == "sabic":
        criteria = (
            -2 * neg_log_likelihood(loss, N) + c0 * K * np.log((N + 2) / 24) ** d0
        )

    # Deviance information criterion
    elif model_selection_criteria.lower() == "dic":
        raise NotImplementedError(
            "DIC has not been implmented as a model selection criterion"
        )

    # Widely applicable (or Watanabe-Akaike) information criterion
    elif model_selection_criteria.lower() == "waic":
        raise NotImplementedError(
            "WAIC has not been implmented as a model selection criterion"
        )

    # Widely applicable (or Watanabe) Bayesian information criterion
    elif model_selection_criteria.lower() == "wbic":
        raise NotImplementedError(
            "WBIC has not been implmented as a model selection criterion"
        )

    if model_selection_criteria.lower() not in ["rmse", "rmse_adj"]:
        criteria /= N  # Normalize to number of datapoints

    return criteria
