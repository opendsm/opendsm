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

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from skfda.representation.grid import FDataGrid as _FDataGrid
from skfda.representation.basis import Fourier as _Fourier
from skfda.preprocessing.dim_reduction import FPCA as _FPCA

import pywt

from sklearn.decomposition import PCA, KernelPCA

from opendsm.common.stats import basic as _basic
from opendsm.common.clustering import settings as _settings



def _safe_standardize(
    data: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    threshold: float = 1e-10
) -> np.ndarray:
    """Safely standardize data by centering and scaling.

    If the scale (e.g., standard deviation or MAD) is near zero, only centers
    the data without scaling to avoid division by near-zero values.

    Parameters
    ----------
    data : np.ndarray
        Input data to standardize.
    center : np.ndarray
        Centering values (e.g., mean or median) to subtract from data.
    scale : np.ndarray
        Scaling values (e.g., std or MAD) to divide by. Can be scalar or array.
    threshold : float, optional
        Minimum threshold for scale values. If scale is below this, only
        centering is performed. Default is 1e-10.

    Returns
    -------
    np.ndarray
        Standardized data. If scale is near zero for any element, those
        elements are only centered without scaling.
    """
    centered = data - center

    # Handle scalar scale
    if np.isscalar(scale) or scale.ndim == 0:
        if scale > threshold:
            return centered / scale
        else:
            return centered

    # Handle array scale with broadcasting
    # Replace near-zero scales with 1 for safe division, but track which were replaced
    scale_safe = np.where(scale > threshold, scale, 1.0)
    result = centered / scale_safe

    # For positions where scale was near zero, use only centered value
    near_zero_mask = scale <= threshold
    if np.any(near_zero_mask):
        # Use broadcasting to apply mask
        if centered.ndim == 2 and scale.ndim == 1:
            # Expand mask to match data dimensions
            result = np.where(near_zero_mask, centered, result)
        else:
            result[near_zero_mask] = centered[near_zero_mask]

    return result


def normalize(
    data: np.ndarray,
    settings: _settings.NormalizeSettings
) -> np.ndarray:
    method = settings.method
    axis = settings.axis

    if method == _settings.NormalizeChoice.STANDARDIZE:
        mean = np.mean(data, axis=axis)
        std = np.std(data, axis=axis)
        data = _safe_standardize(data, mean, std)

    elif method == _settings.NormalizeChoice.MED_MAD:
        median = np.median(data, axis=axis)
        mad = _basic.median_absolute_deviation(data, median=median, axis=axis)
        data = _safe_standardize(data, median, mad)

    elif method == _settings.NormalizeChoice.MIN_MAX_QUANTILE:
        q = settings.quantile
        a, b = [-1, 1]  # range to normalize to

        min_val, max_val = np.quantile(data, [q, 1 - q], axis=axis)

        idx_same = np.argwhere(min_val == max_val).flatten()
        idx_diff = [idx for idx in range(data.shape[0]) if idx not in idx_same]

        if axis == 0:
            min_val = min_val[idx_diff][None, :]
            max_val = max_val[idx_diff][None, :]
        elif axis == 1:
            min_val = min_val[idx_diff][:, None]
            max_val = max_val[idx_diff][:, None]

        if len(idx_same) > 0:
            data[idx_same, :] = (a + b) / 2
        
        data[idx_diff, :] = (b - a) * (data[idx_diff, :] - min_val) / (max_val - min_val) + a

    return data


class FpcaError(Exception):
    pass

def _fpca_base(
    x: np.ndarray, 
    y: np.ndarray, 
    min_var_ratio: float
) -> np.ndarray:
    """
    applies fpca to concatenated transform loadshape dataframe values

    x -> time converted to np array taken from loadshape dataframe
    y -> transformed values

    assumes mixture_components return and fourier basis

    also may return a string as second return value. if it is not None, it implies an error occurred
    """

    if 0 >= min_var_ratio or min_var_ratio >= 1:
        raise FpcaError("min_var_ratio but be greater than 0 and less than 1")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise FpcaError("provided non finite values for fpca")

    if len(x) == 0 or len(y) == 0:
        raise FpcaError("provided empty values for fpca")

    n_min = 1
    # get maximum n components

    # smallest 1  || min(largest = number of samples - 1, # time points)
    n_max = np.min(np.array(np.shape(y)) - [1, 5])
    if n_max < n_min:
        n_max = n_min

    n_max = int(n_max)

    # get maximum principle components
    fd = _FDataGrid(grid_points=x, data_matrix=y)
    basis_fcn = _Fourier

    basis_fd = fd.to_basis(basis_fcn(n_basis=n_max + 4))
    fpca = _FPCA(n_components=n_max, components_basis=basis_fcn(n_basis=n_max + 4))
    fpca.fit(basis_fd)

    var_ratio = np.cumsum(fpca.explained_variance_ratio_) - min_var_ratio
    n = int(np.argmin(var_ratio < 0.0) + 1)

    basis_fd = fd.to_basis(basis_fcn(n_basis=n + 4))
    fpca = _FPCA(n_components=n, components_basis=basis_fcn(n_basis=n + 4))
    fpca.fit(basis_fd)

    mixture_components = fpca.transform(basis_fd)

    return mixture_components


def fpca_transform(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
) -> np.ndarray:
    min_var_ratio = settings.fpca_transform.min_var_ratio

    x = np.arange(data.shape[1]) # assumes uniform spacing
    try:
        fcpa_mixture_components = _fpca_base(
            x=x, 
            y=data, 
            min_var_ratio=min_var_ratio
        )
    except FpcaError as e:
        raise e

    return fcpa_mixture_components


def wavelet_transform(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
) -> np.ndarray:
    """
    Transforms the data using the wavelet transform settings
    """
    wavelet_settings = settings.wavelet_transform

    def _dwt_coeffs(data, wavelet="db1", wavelet_mode="periodization", n_levels=None):
        all_features = []
        # iterate through rows of numpy array
        for row in range(len(data)):
            # get max level of decomposition
            dwt_max_level = pywt.dwt_max_level(data[row].shape[0], wavelet)

            if n_levels is None: # None could be input into wavedec directly to same effect
                n_levels = dwt_max_level

            elif n_levels > dwt_max_level:
                n_levels = dwt_max_level
            
            decomp_coeffs = pywt.wavedec(
                data[row], wavelet=wavelet, mode=wavelet_mode, level=n_levels
            )

            decomp_coeffs = np.hstack(decomp_coeffs)

            all_features.append(decomp_coeffs)

        return np.vstack(all_features)

    def _pca_coeffs(features, method, min_var_ratio_explained=0.95, n_components=None):
        if min_var_ratio_explained is not None:
            n_components = min_var_ratio_explained

        # kernel pca is not fully developed
        if method == "kernel_pca":
            if n_components ==  "mle":
                pca = PCA(
                    n_components=n_components,
                    random_state=settings._seed,
                )
                pca_features = pca.fit_transform(features)

            pca = KernelPCA(n_components=None, kernel="rbf")
            pca_features = pca.fit_transform(features)

            if min_var_ratio_explained is not None:
                explained_variance_ratio = pca.eigenvalues_ / np.sum(pca.eigenvalues_)

                # get the cumulative explained variance ratio
                cumulative_explained_variance = np.cumsum(explained_variance_ratio)

                # find number of components that explain pct% of the variance
                n_components = np.argmax(cumulative_explained_variance > n_components).astype(int)

            if not isinstance(n_components, (int, np.integer)):
                raise ValueError("n_components must be an integer for kernel PCA")

            # pca = PCA(n_components=n_components)
            pca = KernelPCA(n_components=n_components, kernel="rbf")
            pca_features = pca.fit_transform(features)

        else:
            pca = PCA(
                n_components=n_components,
                random_state=settings._seed,
            )
            pca_features = pca.fit_transform(features)

        return pca_features

    # calculate wavelet coefficients
    with warnings.catch_warnings():
        features = _dwt_coeffs(
            data, 
            wavelet_settings.wavelet_name, 
            wavelet_settings.wavelet_mode, 
            wavelet_settings.wavelet_n_levels
        )

    pca_features = _pca_coeffs(
        features,
        wavelet_settings.pca_method,
        wavelet_settings.pca_min_variance_ratio_explained,
        wavelet_settings.pca_n_components,
    )

    # normalize pca features
    if settings.normalize.post_transform:
        # ignores all other values from normalize settings
        mean = pca_features.mean()
        std = pca_features.std()
        pca_features = _safe_standardize(pca_features, mean, std)

    if wavelet_settings.include_scale_feature:
        pca_features = np.hstack([pca_features, np.median(data, axis=1)[:, None]])

    return pca_features


def transform_features(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
) -> np.ndarray:
    
    # normalize the data
    if settings.normalize.pre_transform:
        data = normalize(data, settings.normalize)

    # transform the data
    if settings.transform_selection == _settings.TransformChoice.FPCA:
        data = fpca_transform(data, settings)

    elif settings.transform_selection == _settings.TransformChoice.WAVELET:
        data = wavelet_transform(data, settings)

    return data