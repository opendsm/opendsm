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

from typing import Optional, Tuple, Union

import numba
import numpy as np
from scipy.optimize import minimize_scalar

from opendsm.common.stats.adaptive_loss_Z import ln_Z
from opendsm.common.stats.outliers import (
    remove_outliers,
    _IQR_outlier,
)
from opendsm.common.stats.basic import _median_absolute_deviation
from opendsm.common.utils import OoM_numba

# Loss function constants
LOSS_ALPHA_MIN = -100.0
LOSS_ALPHA_MAX = 100.0


@numba.jit(nopython=True, cache=True)
def sliding_window(
    arr: np.ndarray, 
    window_size: int, 
    step: int = 0,
) -> np.ndarray:
    """Create sliding windows over a time series array.

    Reference: https://giov.dev/2018/05/a-window-on-numpy-s-views.html

    Args:
        arr: Input array with time advancing along dimension 0
        window_size: Size of sliding window
        step: Step size for sliding window. If 0, uses non-overlapping
            contiguous windows (step=window_size)

    Returns:
        Array with windowed views of the input data

    Raises:
        ValueError: If window_size > array size or step < 0
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


def rolling_IQR_outlier(
    x: np.ndarray,
    y: np.ndarray,
    sigma_threshold: float = 3.0,
    quantile: float = 0.25,
    window: Union[float, int] = 0.05,
    step: float = 1.0,
) -> np.ndarray:
    """Calculate rolling outlier thresholds using IQR method.

    Args:
        x: X-values of the data (e.g., time)
        y: Y-values of the data (residuals or observations)
        sigma_threshold: Sigma threshold for IQR outlier detection
        quantile: Quantile for IQR calculation (0.25 for standard IQR)
        window: Window size. If <= 1, treated as proportion of data length
        step: Step size for rolling calculation (as proportion of window if < 1)

    Returns:
        2D array with shape (2, len(x)) where row 0 is lower threshold
        and row 1 is upper threshold
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

    # Vectorized computation of means and quantiles
    x_interp = np.mean(x_windows, axis=1)
    q13 = np.quantile(y_windows, [quantile, 1 - quantile], axis=1)
    q1 = q13[0]
    q3 = q13[1]

    # Empirical scaling factor to convert sigma threshold to IQR multiplier
    q13_scalar = 0.7413 * sigma_threshold - 0.5
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
def get_C(
    resid: np.ndarray,
    mu: float,
    sigma: float,
    quantile: float = 0.25,
    algo: str = "iqr_legacy",
) -> float:
    """Calculate scale parameter C for adaptive loss weighting.

    Computes a robust scale estimate using various methods to normalize
    residuals for adaptive loss functions.

    Args:
        resid: Residuals from model fit
        mu: Location parameter (typically median of residuals)
        sigma: Scale factor (typically sigma threshold for outliers)
        quantile: Quantile for IQR calculation (0.25 for standard IQR)
        algo: Algorithm to use - 'iqr_legacy', 'iqr', 'mad', or 'stdev'

    Returns:
        Scale parameter C for normalizing residuals
    """
    # remove non-finite values
    resid = resid[np.isfinite(resid)]

    if algo == "iqr_legacy":
        # TODO: uncertain if these C functions should use np.min, np.mean, or np.max
        # suspect we can switch to IQR below, but need to test
        bounds = _IQR_outlier(
            resid - mu, weights=None, sigma_threshold=sigma, quantile=quantile
        )
        C = np.max(np.abs(bounds))

    elif algo == "iqr":
        resid = np.abs(resid)

        bounds = _IQR_outlier(
            resid - mu, weights=None, sigma_threshold=sigma, quantile=quantile
        )
        C = np.max(np.abs(bounds))

    elif algo == "mad":
        C = sigma * _median_absolute_deviation(resid, median=None, weights=None)

    elif algo == "stdev":
        C = sigma * np.std(resid)

    if C == 0:
        C = OoM_numba(np.array([C]), method="floor")[0]

    return C


def rolling_C(
    T: np.ndarray,
    resid: np.ndarray,
    mu: float,
    sigma: float = 3.0,
    quantile: float = 0.25,
    window: Union[float, int] = 0.2,
    step: float = 1.0,
) -> np.ndarray:
    """Calculate rolling scale parameter C for adaptive loss weighting.

    Args:
        T: Time or x-axis values
        resid: Residuals from model fit
        mu: Location parameter (typically median)
        sigma: Sigma threshold for outlier detection
        quantile: Quantile for IQR calculation (0.25 for standard IQR)
        window: Window size (proportion if <= 1, absolute if > 1)
        step: Step size for rolling calculation

    Returns:
        Array of rolling C values
    """
    q13 = rolling_IQR_outlier(T, resid - mu, sigma, quantile, window, step)
    C = np.max(np.abs(q13), axis=0)

    return C


@numba.jit(nopython=True, error_model="numpy", cache=True)
def generalized_loss_fcn(
    x: Union[float, np.ndarray], 
    alpha: float = 2.0, 
    alpha_min: float = LOSS_ALPHA_MIN,
) -> Union[float, np.ndarray]:
    """Calculate generalized loss function value.

    Implements a family of robust loss functions parameterized by alpha.
    Different alpha values correspond to different well-known loss functions.

    Args:
        x: Input value(s) - typically normalized residuals
        alpha: Shape parameter determining loss function type
        alpha_min: Minimum alpha value for Welsch/Leclerc loss

    Returns:
        Loss function value(s)

    Loss function types by alpha value:
        - alpha = 2.0: L2 (squared error) loss
        - alpha = 1.0: Smoothed L1 (Pseudo-Huber) loss
        - alpha = 0.0: Charbonnier loss
        - alpha = -2.0: Cauchy/Lorentzian loss
        - alpha <= alpha_min: Welsch/Leclerc loss
        - other: Generalized Charbonnier loss
    """

    # Defaults to sum of squared error
    x_2 = x**2

    if alpha == 2.0:  # L2
        loss = 0.5 * x_2
    elif alpha == 1.0:  # smoothed L1
        loss = np.sqrt(x_2 + 1) - 1
    elif alpha == 0.0:  # Charbonnier loss
        loss = np.log(0.5 * x_2 + 1)
    elif alpha == -2.0:  # Cauchy/Lorentzian loss
        loss = 2 * x_2 / (x_2 + 4)
    elif alpha <= alpha_min:  # at -infinity, Welsch/Leclerc loss
        loss = 1 - np.exp(-0.5 * x_2)
    else:
        loss = np.abs(alpha - 2) / alpha * ((x_2 / np.abs(alpha - 2) + 1) ** (alpha / 2) - 1)

    return loss


@numba.jit(nopython=True, error_model="numpy", cache=True)
def generalized_loss_derivative(
    x: Union[float, np.ndarray], 
    scale: float = 1.0, 
    alpha: float = 2.0,
) -> Union[float, np.ndarray]:
    """Calculate derivative of generalized loss function.

    Computes the gradient of the loss function with respect to the input,
    accounting for the scale parameter.

    Args:
        x: Input value(s) - typically residuals
        scale: Scale parameter for normalization
        alpha: Shape parameter determining loss function type

    Returns:
        Derivative of loss function with respect to x

    Loss function types by alpha value:
        - alpha = 2.0: L2 loss
        - alpha = 1.0: Smoothed L1 (Pseudo-Huber) loss
        - alpha = 0.0: Charbonnier loss
        - alpha = -2.0: Cauchy/Lorentzian loss
        - alpha <= LOSS_ALPHA_MIN: Welsch/Leclerc loss
        - other: Generalized loss
    """
    if alpha == 2.0:  # L2
        dloss_dx = x / scale**2
    elif alpha == 1.0:  # smoothed L1
        dloss_dx = x / scale**2 / np.sqrt((x / scale) ** 2 + 1)
    elif alpha == 0.0:  # Charbonnier loss
        dloss_dx = 2 * x / (x**2 + 2 * scale**2)
    elif alpha == -2.0:  # Cauchy/Lorentzian loss
        dloss_dx = 16 * scale**2 * x / (4 * scale**2 + x**2) ** 2
    elif alpha <= LOSS_ALPHA_MIN:  # at -infinity, Welsch/Leclerc loss
        dloss_dx = x / scale**2 * np.exp(-0.5 * (x / scale) ** 2)
    else:
        dloss_dx = x / scale**2 * ((x / scale) ** 2 / np.abs(alpha - 2) + 1)

    return dloss_dx


@numba.jit(nopython=True, error_model="numpy", cache=True)
def generalized_loss_weights(
    x: np.ndarray, 
    alpha: float = 2.0, 
    min_weight: float = 0.0,
) -> np.ndarray:
    """Calculate adaptive weights based on generalized loss function.

    Computes observation weights that downweight outliers according to
    the loss function shape parameter alpha.

    Args:
        x: Normalized residuals (typically (residuals - mu) / scale)
        alpha: Shape parameter determining weight behavior
        min_weight: Minimum weight value (prevents complete downweighting)

    Returns:
        Array of weights in range [min_weight, 1.0]
    """

    dtype = numba.float64
    if numba.config.DISABLE_JIT:
        dtype = np.float64

    # Vectorized computation
    x_sq = x**2
    w = np.ones(len(x), dtype=dtype)

    if alpha == 2.0:
        # L2 loss: all weights are 1.0
        pass
    elif alpha == 0.0:
        # Charbonnier loss
        w = np.where(x > 0, 1.0 / (0.5 * x_sq + 1.0), 1.0)
    elif alpha <= LOSS_ALPHA_MIN:
        # Welsch/Leclerc loss
        w = np.where(x > 0, np.exp(-0.5 * x_sq), 1.0)
    else:
        # Generalized loss
        w = np.where(x > 0, (x_sq / np.abs(alpha - 2) + 1) ** (0.5 * alpha - 1), 1.0)

    return w * (1.0 - min_weight) + min_weight


def penalized_loss_fcn(
    x: np.ndarray, 
    alpha: float = 2.0, 
    use_penalty: bool = True,
) -> np.ndarray:
    """Calculate penalized loss function with partition function penalty.

    Adds a penalty term based on the approximate partition function to
    penalize more complex loss functions (lower alpha values).

    Args:
        x: Normalized input values (typically residuals)
        alpha: Shape parameter for loss function
        use_penalty: Whether to include partition function penalty

    Returns:
        Penalized loss values

    Raises:
        Exception: If non-finite values are found in calculated loss
    """
    loss = generalized_loss_fcn(x, alpha=alpha)

    if use_penalty:
        # Approximate partition function penalty for C=1, tau=10
        penalty = ln_Z(alpha, LOSS_ALPHA_MIN)
        loss += penalty

        if not np.isfinite(loss).all():
            # print("alpha: ", alpha)
            # print("x: ", x)
            # print("penalty: ", penalty)
            raise Exception("non-finite values in 'penalized_loss_fcn'")

    return loss


@numba.jit(nopython=True, error_model="numpy", cache=True)
def alpha_scaled(
    s: float, 
    alpha_max: float = 2.0,
) -> float:
    """Convert scaled parameter s to alpha value.

    Transforms a bounded input s to the alpha parameter space using
    nonlinear scaling to provide smooth optimization behavior.

    Args:
        s: Scaled input value (typically in [0, 1] for optimization)
        alpha_max: Maximum alpha value (determines scaling method)

    Returns:
        Alpha value in range [LOSS_ALPHA_MIN, alpha_max] (approximately)
    """
    if alpha_max == 2.0:
        a = 3
        b = 0.25

        # Clip s to valid range
        if s < 0:
            s = 0
        if s > 1:
            s = 1

        # Nonlinear scaling using power law
        s_max = 1 - 2 / (1 + 10**a)
        s = (1 - 2 / (1 + 10 ** (a * s**b))) / s_max

        alpha = LOSS_ALPHA_MIN + (2.0 - LOSS_ALPHA_MIN) * s

    else:
        # Alternative scaling using logistic function
        x0 = 1.0
        k = 1.5

        if s >= 1:
            return LOSS_ALPHA_MAX
        elif s <= 0:
            return LOSS_ALPHA_MIN

        A = (np.exp((LOSS_ALPHA_MAX - x0) / k) + 1) / (
            1 - np.exp(2 * LOSS_ALPHA_MAX / k)
        )
        K = (1 - A) * np.exp((x0 - LOSS_ALPHA_MAX) / k) + 1

        alpha = x0 - k * np.log((K - A) / (s - A) - 1)

    return alpha


def adaptive_loss_fcn(
    x: np.ndarray,
    mu: float = 0.0,
    scale: float = 1.0,
    alpha: Union[str, float] = "adaptive",
    replace_nonfinite: bool = True,
) -> Tuple[float, float]:
    """Calculate adaptive loss function and optimal alpha parameter.

    Computes the total loss and optionally optimizes the alpha parameter
    to minimize the penalized loss function.

    Args:
        x: Input residuals
        mu: Location parameter for normalization
        scale: Scale parameter for normalization
        alpha: Shape parameter ('adaptive' for optimization, or fixed value)
        replace_nonfinite: Replace non-finite values with max finite value

    Returns:
        Tuple of (total loss value, alpha parameter used)
    """
    # Standardize residuals if needed
    if np.all(mu != 0.0) or np.all(scale != 1.0):
        x = (x - mu) / scale

    if replace_nonfinite:
        x[~np.isfinite(x)] = np.max(x[np.isfinite(x)])

    def _loss_for_alpha(alpha_val: float) -> float:
        """Compute total penalized loss for given alpha."""
        return penalized_loss_fcn(x, alpha=alpha_val, use_penalty=True).sum()

    if alpha == "adaptive":
        # Optimize alpha parameter over scaled space
        res = minimize_scalar(
            lambda s: _loss_for_alpha(alpha_scaled(s)),
            bounds=[-1e-5, 1 + 1e-5],
            method="Bounded",
            options={"xatol": 1e-5},
        )
        loss_alpha = alpha_scaled(res.x)
        # res = minimize(lambda s: _loss_for_alpha(alpha_scaled(s[0])), x0=[0.7], bounds=[[0, 1]], method="L-BFGS-B")
        # loss_alpha = alpha_scaled(res.x[0])
        loss_fcn_val = res.fun
    else:
        loss_alpha = alpha
        loss_fcn_val = _loss_for_alpha(alpha)

    return loss_fcn_val, loss_alpha


def adaptive_weights(
    x: np.ndarray,
    alpha: Union[str, float] = "adaptive",
    sigma: float = 3.0,
    quantile: float = 0.25,
    min_weight: float = 0.0,
    C_algo: str = "iqr_legacy",
    replace_nonfinite: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """Calculate adaptive weights for robust regression.

    Computes observation weights that downweight outliers based on
    the adaptive loss function. The scale and alpha parameters are
    automatically determined from the data.

    Args:
        x: Input residuals (not standardized)
        alpha: Shape parameter ('adaptive' for optimization, or fixed value)
        sigma: Sigma threshold for outlier detection
        quantile: Quantile for IQR calculation (0.25 for standard IQR)
        min_weight: Minimum weight value (prevents complete downweighting)
        C_algo: Algorithm for scale estimation ('iqr_legacy', 'iqr', 'mad', 'stdev')
        replace_nonfinite: Replace non-finite values with max finite value

    Returns:
        Tuple of (weights array, scale parameter C, alpha parameter)
    """
    x_no_outlier, _ = remove_outliers(x, sigma_threshold=sigma, quantile=0.25)

    # TODO: Should x be abs or not?
    # likely should be abs
    # mu = np.median(np.abs(x_no_outlier))
    mu = np.median(x_no_outlier)

    C = get_C(x, mu, sigma, quantile, C_algo)
    x_normalized = (x - mu) / C

    if alpha == "adaptive":
        _, alpha = adaptive_loss_fcn(
            x_normalized, alpha=alpha, replace_nonfinite=replace_nonfinite
        )

    return generalized_loss_weights(x_normalized, alpha=alpha, min_weight=min_weight), C, alpha