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
from scipy.optimize import minimize_scalar

from opendsm.common.adaptive_loss_Z import ln_Z
from opendsm.common.utils import OoM_numba

LOSS_ALPHA_MIN = -100.0


@numba.jit(nopython=True, cache=True)
def weighted_quantile(
    values, quantiles, weights=None, values_presorted=False, old_style=False
):
    """https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_presorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.quantile.
    :return: numpy.array with computed quantiles.
    """
    finite_idx = np.where(np.isfinite(values))
    values = values[finite_idx]
    if weights is None or len(weights) == 0:
        weights = np.ones_like(values)
    else:
        weights = weights[finite_idx]

    assert np.all(quantiles >= 0) and np.all(
        quantiles <= 1
    ), "quantiles should be in [0, 1]"

    if not values_presorted:
        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

    res = np.cumsum(weights) - 0.5 * weights
    if old_style:  # To be convenient with numpy.quantile
        res -= res[0]
        res /= res[-1]
    else:
        res /= np.sum(weights)

    return np.interp(quantiles, res, values)


def remove_outliers(data, weights=None, sigma_threshold=3, quantile=0.25):
    outlier_bnds = IQR_outlier(data, weights, sigma_threshold, quantile)
    idx_no_outliers = np.argwhere(
        (data >= outlier_bnds[0]) & (data <= outlier_bnds[1])
    ).flatten()
    data_no_outliers = data[idx_no_outliers]

    return data_no_outliers, idx_no_outliers


@numba.jit(nopython=True, cache=True)
def IQR_outlier(data, weights=None, sigma_threshold=3, quantile=0.25):
    # only use finite data
    if weights is None:
        q13 = np.nanquantile(data[np.isfinite(data)], [quantile, 1 - quantile])
    else:  # weighted_quantile could be used always, don't know speed
        q13 = weighted_quantile(
            data[np.isfinite(data)], np.array([quantile, 1 - quantile]), weights=weights
        )

    q13_scalar = (
        0.7413 * sigma_threshold - 0.5
    )  # this is a pretty good fit to get the scalar for any sigma
    iqr = np.diff(q13)[0] * q13_scalar
    outlier_threshold = np.array([q13[0] - iqr, q13[1] + iqr])

    return outlier_threshold


@numba.jit(nopython=True, cache=True)
def sliding_window(
    arr, window_size, step=0
):  # https://giov.dev/2018/05/a-window-on-numpy-s-views.html
    """Assuming a time series with time advancing along dimension 0,
        window the time series with given size and step.

    :param arr : input array.
    :type arr: numpy.ndarray
    :param window_size: size of sliding window.
    :type window_size: int
    :param step: step size of sliding window. If 0, step size is set to obtain
        non-overlapping contiguous windows (that is, step=window_size).
        Defaults to 0.
    :type step: int

    :return: array
    :rtype: numpy.ndarray
    """
    n_obs = arr.shape[0]

    # validate arguments
    if window_size > n_obs:
        raise ValueError(
            "Window size must be less than or equal "
            "the size of array in first dimension."
        )
    if step < 0:
        raise ValueError("Step must be positive.")

    n_windows = 1 + int(np.floor((n_obs - window_size) / step))

    obs_stride = arr.strides[0]
    windowed_row_stride = obs_stride * step

    new_shape = (n_windows, window_size) + arr.shape[1:]
    new_strides = (windowed_row_stride,) + arr.strides

    strided = np.lib.stride_tricks.as_strided(
        arr,
        shape=new_shape,
        strides=new_strides,
    )
    return strided


def rolling_IQR_outlier(x, y, sigma_threshold=3, quantile=0.25, window=0.05, step=1):
    """
    This function calculates the outlier threshold using the rolling Interquartile Range (IQR) method.

    Parameters:
    x (numpy array): The x-values of the data.
    y (numpy array): The y-values of the data.
    sigma_threshold (float, optional): The sigma threshold for the IQR. Default is 3.
    quantile (float, optional): The quantile to use for the IQR calculation. Default is 0.25.
    window (float or int, optional): The window size for the rolling calculation. If less than 1, it is considered as a proportion of the data length. Default is 0.05.
    step (int, optional): The step size for the rolling calculation. Default is 1.

    Returns:
    numpy array: A 2D array where the first row is the lower outlier threshold and the second row is the upper outlier threshold.
    """

    if window <= 1.0:
        window = int(np.floor(len(y) * window))
    else:
        window = int(window)

    step = int(np.floor(window * step))
    if step < 1:
        step = 1

    y = np.abs(y)

    x_windows = sliding_window(x, window, step=step)
    y_windows = sliding_window(y, window, step=step)

    x_interp = np.zeros(np.shape(x_windows)[0])
    q1 = np.zeros(np.shape(y_windows)[0])
    q3 = np.zeros(np.shape(y_windows)[0])
    for i in range(len(q1)):
        x_interp[i] = np.mean(x_windows[i])

        q13 = np.quantile(y_windows[i], [quantile, 1 - quantile])
        q1[i] = q13[0]
        q3[i] = q13[1]

    q13_scalar = (
        0.7413 * sigma_threshold - 0.5
    )  # this is a pretty good fit to get the scalar for any sigma
    iqr = (q3 - q1) * q13_scalar
    outlier_bnds = [q1 - iqr, q3 + iqr]

    outlier_threshold = np.zeros((2, len(x)))
    outlier_threshold[0] = np.interp(x, x_interp, outlier_bnds[0])
    outlier_threshold[1] = np.interp(x, x_interp, outlier_bnds[1])

    # x_interp = np.arange(0, len(outlier_bnds[0]))
    # x_orig = np.linspace(0, len(outlier_bnds[0]), len(x))

    # outlier_threshold = np.zeros((2, len(x)))
    # outlier_threshold[0] = np.interp(x_orig, x_interp, outlier_bnds[0])
    # outlier_threshold[1] = np.interp(x_orig, x_interp, outlier_bnds[1])

    return outlier_threshold


@numba.jit(nopython=True, error_model="numpy", cache=True)
def get_C(resid, mu, sigma, quantile=0.25, algo="iqr_legacy"):
    """
    This function calculates the maximum absolute value of the Interquartile Range (IQR) of the residuals
    from a given mean (mu) and standard deviation (sigma). If the maximum absolute value is zero,
    it is replaced with the order of magnitude of the maximum value of the IQR.

    Parameters:
    resid (numpy.ndarray): The residuals of the model.
    mu (float): The mean of the residuals.
    sigma (float): The standard deviation of the residuals.
    quantile (float, optional): The quantile to use for the IQR calculation. Default is 0.25.

    Returns:
    float: The maximum absolute value of the IQR of the residuals, or the order of magnitude of the maximum value of the IQR.
    """
    # remove non-finite values
    resid = resid[np.isfinite(resid)]

    if algo == "iqr_legacy":
        # TODO: uncertain if these C functions should use np.min, np.mean, or np.max
        # suspect we can switch to IQR below, but need to test
        bounds = IQR_outlier(
            resid - mu, weights=None, sigma_threshold=sigma, quantile=quantile
        )
        C = np.max(np.abs(bounds))

    elif algo == "iqr":
        resid = np.abs(resid)

        bounds = IQR_outlier(
            resid - mu, weights=None, sigma_threshold=sigma, quantile=quantile
        )
        C = np.max(np.abs(bounds))

    elif algo == "mad":
        resid_median = np.median(resid)
        MAD = np.median(np.abs(resid - resid_median))

        C = sigma * 1.4826 * MAD

    elif algo == "stdev":
        C = sigma * np.std(resid)

    if C == 0:
        C = OoM_numba(np.array([C]), method="floor")[0]

    return C


def rolling_C(T, resid, mu, sigma=3, quantile=0.25, window=0.2, step=1.0):
    """
    Function to calculate the rolling C value.

    Parameters:
    T (array-like): The time series data.
    resid (array-like): The residuals of the model.
    mu (float): The mean value.
    sigma (float, optional): The standard deviation. Default is 3.
    quantile (float, optional): The quantile to use for the IQR calculation. Default is 0.25.
    window (float, optional): The window size for the rolling calculation. Default is 0.2.
    step (float, optional): The step size for the rolling calculation. Default is 1.0.

    Returns:
    C (array-like): The calculated rolling C values.
    """

    q13 = rolling_IQR_outlier(T, resid - mu, sigma, quantile, window, step)
    C = np.max(np.abs(q13), axis=0)

    return C


@numba.jit(nopython=True, error_model="numpy", cache=True)
def generalized_loss_fcn(x, a=2, a_min=LOSS_ALPHA_MIN):
    """
    This function calculates the generalized loss function based on the given parameters.

    Parameters:
    x (float): The input value for which the loss is to be calculated.
    a (float): The parameter that determines the type of loss function to be used. Default is 2.
    a_min (float): The minimum value of 'a' for which the Welsch/Leclerc loss is to be calculated. Default is LOSS_ALPHA_MIN.

    Returns:
    float: The calculated loss.

    The function supports the following types of loss functions based on the value of 'a', i.e. alpha:
    - L2 loss for a=2.0
    - Smoothed L1 loss for a=1.0
    - Charbonnier loss for a=0.0
    - Cauchy/Lorentzian loss for a=-2.0
    - Welsch/Leclerc loss for a<=a_min
    - Generalized loss for other values of 'a'
    """

    # defaults to sum of squared error
    x_2 = x**2

    if a == 2.0:  # L2
        loss = 0.5 * x_2
    elif a == 1.0:  # smoothed L1
        loss = np.sqrt(x_2 + 1) - 1
    elif a == 0.0:  # Charbonnier loss
        loss = np.log(0.5 * x_2 + 1)
    elif a == -2.0:  # Cauchy/Lorentzian loss
        loss = 2 * x_2 / (x_2 + 4)
    elif a <= a_min:  # at -infinity, Welsch/Leclerc loss
        loss = 1 - np.exp(-0.5 * x_2)
    else:
        loss = np.abs(a - 2) / a * ((x_2 / np.abs(a - 2) + 1) ** (a / 2) - 1)

    return loss


@numba.jit(nopython=True, error_model="numpy", cache=True)
def generalized_loss_derivative(x, c=1, a=2):
    """
    This function calculates the derivative of the generalized loss function.

    Parameters:
    x (float): The input value for which the derivative of the loss function is to be calculated.
    c (float, optional): The scale parameter. Default is 1.
    a (float, optional): Alpha value for the loss function. Default is 2.

    Returns:
    float: The derivative of the generalized loss function at x.

    The function supports different types of loss functions based on the value of alpha:
    - L2 loss for a=2.0
    - Smoothed L1 loss for a=1.0
    - Charbonnier loss for a=0.0
    - Cauchy/Lorentzian loss for a=-2.0
    - Welsch/Leclerc loss for a<=LOSS_ALPHA_MIN
    - Generalized loss for other values of alpha
    """

    if a == 2.0:  # L2
        dloss_dx = x / c**2
    elif a == 1.0:  # smoothed L1
        dloss_dx = x / c**2 / np.sqrt((x / c) ** 2 + 1)
    elif a == 0.0:  # Charbonnier loss
        dloss_dx = 2 * x / (x**2 + 2 * c**2)
    elif a == -2.0:  # Cauchy/Lorentzian loss
        dloss_dx = 16 * c**2 * x / (4 * c**2 + x**2) ** 2
    elif a <= LOSS_ALPHA_MIN:  # at -infinity, Welsch/Leclerc loss
        dloss_dx = x / c**2 * np.exp(-0.5 * (x / c) ** 2)
    else:
        dloss_dx = x / c**2 * ((x / c) ** 2 / np.abs(a - 2) + 1)

    return dloss_dx


@numba.jit(nopython=True, error_model="numpy", cache=True)
def generalized_loss_weights(x: np.ndarray, a: float = 2, min_weight: float = 0.00):
    """
    This function calculates the generalized loss weights for a given array of values.

    Parameters:
    x (np.ndarray): The input array for which the weights are to be calculated.
    a (float, optional): The alpha parameter for the weight calculation. Default is 2.
    min_weight (float, optional): The minimum weight to be considered. Default is 0.00.

    Returns:
    np.ndarray: The calculated weights for the input array.
    """

    dtype = numba.float64
    if numba.config.DISABLE_JIT:
        dtype = np.float64

    w = np.ones(len(x), dtype=dtype)
    for i, xi in enumerate(x):
        if a == 2 or xi <= 0:
            w[i] = 1
        elif a == 0:
            w[i] = 1 / (0.5 * xi**2 + 1)
        elif a <= LOSS_ALPHA_MIN:
            w[i] = np.exp(-0.5 * xi**2)
        else:
            w[i] = (xi**2 / np.abs(a - 2) + 1) ** (0.5 * a - 1)

    return w * (1 - min_weight) + min_weight


# penalize the loss function using approximate partition function
# default to L2 loss
def penalized_loss_fcn(x, a=2, use_penalty=True):
    """
    This function calculates the penalized loss for a given input. There is a penalty for more coefficients
    being added into the funciton being fit.

    Parameters:
    x (numpy array): The input data for which the loss is to be calculated.
    a (float, optional): The parameter for the generalized loss function. Default is 2.
    use_penalty (bool, optional): Flag to decide whether to use penalty or not. Default is True.

    Returns:
    loss (float): The calculated penalized loss.

    Raises:
    Exception: If non-finite values are found in the calculated loss.
    """

    loss = generalized_loss_fcn(x, a)

    if use_penalty:
        penalty = ln_Z(
            a, LOSS_ALPHA_MIN
        )  # approximate partition function for C=1, tau=10
        loss += penalty

        if not np.isfinite(loss).all():
            # print("a: ", a)
            # print("x: ", x)
            # print("penalty: ", penalty)
            raise Exception("non-finite values in 'penalized_loss_fcn'")

    return loss


@numba.jit(nopython=True, error_model="numpy", cache=True)
def alpha_scaled(s, a_max=2):
    """
    This function calculates the alpha value based on the input s and a_max (max alpha allowed).

    Parameters:
    s (float): The input value for which alpha is to be calculated.
    a_max (int, optional): The maximum value of alpha. Default is 2.

    Returns:
    float: The calculated alpha value.

    Note:
    The function uses two different methods to calculate alpha based on the value of a_max.
    If a_max is 2, it uses a specific formula involving logarithmic and exponential functions.
    If a_max is not 2, it uses a different formula also involving logarithmic and exponential functions.
    """

    if a_max == 2:
        a = 3
        b = 0.25

        if s < 0:
            s = 0

        if s > 1:
            s = 1

        s_max = 1 - 2 / (1 + 10**a)
        s = (1 - 2 / (1 + 10 ** (a * s**b))) / s_max

        alpha = LOSS_ALPHA_MIN + (2 - LOSS_ALPHA_MIN) * s

    else:
        x0 = 1
        k = 1.5  # 1 or 1.5, testing required

        if s >= 1:
            return 100
        elif s <= 0:
            return -100

        A = (np.exp((100 - x0) / k) + 1) / (1 - np.exp(200 / k))
        K = (1 - A) * np.exp((x0 - 100) / k) + 1

        alpha = x0 - k * np.log((K - A) / (s - A) - 1)

    return alpha


def adaptive_loss_fcn(x, mu=0, c=1, alpha="adaptive", replace_nonfinite=True):
    """
    This function calculates the adaptive loss function value and the alpha parameter.

    Parameters:
    x (numpy array): The input data.
    mu (float, optional): The mean value for standardization. Default is 0.
    c (float, optional): The scale value for standardization. Default is 1.
    alpha (str or float, optional): The alpha parameter for the loss function. If "adaptive", the function will optimize alpha. Default is "adaptive".
    replace_nonfinite (bool, optional): If True, replaces non-finite values in x with the maximum finite value. Default is True.

    Returns:
    tuple: The value of the loss function and the alpha parameter.
    """

    if np.all(mu != 0) or np.all(c != 1):
        x = (x - mu) / c  # standardized residuals

    if replace_nonfinite:
        x[~np.isfinite(x)] = np.max(x[np.isfinite(x)])

    loss_alpha_fcn = lambda alpha: penalized_loss_fcn(
        x, a=alpha, use_penalty=True
    ).sum()

    if alpha == "adaptive":  #
        res = minimize_scalar(
            lambda s: loss_alpha_fcn(alpha_scaled(s)),
            bounds=[-1e-5, 1 + 1e-5],
            method="Bounded",
            options={"xatol": 1e-5},
        )
        loss_alpha = alpha_scaled(res.x)
        # res = minimize(lambda s: loss_alpha_fcn(alpha_scaled(s[0])), x0=[0.7], bounds=[[0, 1]], method="L-BFGS-B")
        # loss_alpha = alpha_scaled(res.x[0])
        loss_fcn_val = res.fun

    else:
        loss_alpha = alpha
        loss_fcn_val = loss_alpha_fcn(alpha)

    return loss_fcn_val, loss_alpha


# Assumes that x has not been standardized
def adaptive_weights(
    x, 
    alpha="adaptive", 
    sigma=3, 
    quantile=0.25, 
    min_weight=0.00, 
    C_algo="iqr_legacy", 
    replace_nonfinite=True
):
    """
    This function calculates the adaptive weights for a given data set.

    Parameters:
    x (numpy.array): The input data.
    alpha (str or float, optional): The alpha value for the adaptive loss function. Default is "adaptive".
    sigma (float, optional): The sigma threshold for outlier removal. Default is 3.
    quantile (float, optional): The quantile for outlier removal. Default is 0.25.
    min_weight (float, optional): The minimum weight. Default is 0.00.
    replace_nonfinite (bool, optional): Whether to replace non-finite values. Default is True.

    Returns:
    tuple: A tuple containing the generalized loss weights, C value, and alpha value.
    """

    x_no_outlier, _ = remove_outliers(x, sigma_threshold=sigma, quantile=0.25)

    # TODO: Should x be abs or not?
    # likely should be abs
    # mu = np.median(np.abs(x_no_outlier))
    mu = np.median(x_no_outlier)

    C = get_C(x, mu, sigma, quantile, C_algo)
    x = (x - mu) / C

    if alpha == "adaptive":
        _, alpha = adaptive_loss_fcn(
            x, alpha=alpha, replace_nonfinite=replace_nonfinite
        )

    return generalized_loss_weights(x, a=alpha, min_weight=min_weight), C, alpha
