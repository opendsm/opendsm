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

import numpy as np

from opendsm.common.stats import basic as _basic
from opendsm.common.clustering import settings as _settings


_SENTINEL = object()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

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

    if np.isscalar(scale) or scale.ndim == 0:
        if scale > threshold:
            return centered / scale
        else:
            return centered

    scale_safe = np.where(scale > threshold, scale, 1.0)
    result = centered / scale_safe

    near_zero_mask = scale <= threshold
    if np.any(near_zero_mask):
        result = np.where(near_zero_mask, centered, result)

    return result


def variance_cap(
    data: np.ndarray,
    axis: int = 1,
    percentile: float = 5.0,
) -> tuple[np.ndarray, float]:
    """Cap per-sample post-normalization variance at a percentile.

    Prevents pathological noise amplification from per-sample normalization
    on near-zero data. Samples whose variance exceeds the cap are rescaled
    so their variance equals the cap. Pattern (relative shape) is preserved;
    only amplitude is attenuated.

    Returns (capped_data, compression_ratio) where compression_ratio is
    pre-cap total variance / post-cap total variance (1.0 = no capping needed).
    """
    if data.ndim != 2:
        return data, 1.0

    sample_vars = np.var(data, axis=axis)
    pre_cap_total = float(np.sum(np.var(data, axis=0)))
    var_cap = np.percentile(sample_vars, percentile)

    if var_cap <= 0:
        return data, 1.0

    exceeds = sample_vars > var_cap
    if not np.any(exceeds):
        return data, 1.0

    scale_factors = np.ones(sample_vars.shape)
    scale_factors[exceeds] = np.sqrt(var_cap / sample_vars[exceeds])

    if axis == 1:
        data = data * scale_factors[:, np.newaxis]
    else:
        data = data * scale_factors[np.newaxis, :]

    post_cap_total = float(np.sum(np.var(data, axis=0)))
    compression = pre_cap_total / post_cap_total if post_cap_total > 0 else 1.0

    return data, compression


def normalize(
    data: np.ndarray,
    settings: _settings.NormalizeSettings,
    axis: int | None = _SENTINEL,
) -> np.ndarray:
    """Normalize data along a specified axis using various normalization methods.

    Parameters
    ----------
    data : np.ndarray
        Input data to normalize.
    settings : NormalizeSettings
        Configuration specifying normalization method and axis.
    axis : int or None, optional
        Override the axis from settings. If not provided, uses settings._axis.

    Returns
    -------
    np.ndarray
        Normalized data with same shape as input.
    """
    method = settings.method
    if axis is _SENTINEL:
        axis = settings._axis

    if method == _settings.NormalizeChoice.STANDARDIZE:
        mean = np.mean(data, axis=axis)
        std = np.std(data, axis=axis)
        if axis is not None:
            mean = np.expand_dims(mean, axis=axis)
            std = np.expand_dims(std, axis=axis)
        data = _safe_standardize(data, mean, std)

    elif method == _settings.NormalizeChoice.MED_MAD:
        median = np.median(data, axis=axis)
        mad = _basic.median_absolute_deviation(data, median=median, axis=axis)
        if axis is not None:
            median = np.expand_dims(median, axis=axis)
            mad = np.expand_dims(mad, axis=axis)
        data = _safe_standardize(data, median, mad)

    elif method == _settings.NormalizeChoice.MIN_MAX_QUANTILE:
        q = settings.quantile
        a, b = [-1, 1]

        min_val, max_val = np.quantile(data, [q, 1 - q], axis=axis)

        if axis is None:
            if np.isclose(min_val, max_val, atol=1e-6):
                data = np.full_like(data, (a + b) / 2)
            else:
                data = (b - a) * (data - min_val) / (max_val - min_val + 1e-16) + a
        else:
            min_val = np.expand_dims(min_val, axis=axis)
            max_val = np.expand_dims(max_val, axis=axis)

            const_mask = np.isclose(min_val, max_val, atol=1e-6)

            data = np.where(
                const_mask,
                (a + b) / 2,
                (b - a) * (data - min_val) / (max_val - min_val + 1e-16) + a
            )

    return data


# ---------------------------------------------------------------------------
# Magnitude features (pre-transform supplemental information)
# ---------------------------------------------------------------------------

def compute_magnitude_features(
    original_data: np.ndarray,
    magnitude_settings: _settings.FeatureMagnitudeSettings,
) -> np.ndarray:
    """Compute and standardize magnitude features from original data.

    Computes median, quantile_range, and/or baseload on the unnormalized data,
    then standardizes each feature independently using MAD for robustness.

    Parameters
    ----------
    original_data : np.ndarray
        Original (pre-normalization) data, shape (n_samples, n_features).
    magnitude_settings : FeatureMagnitudeSettings
        Configuration for which features to compute and their quantiles.

    Returns
    -------
    np.ndarray
        Magnitude features, shape (n_samples, n_features_magnitude).
        Each feature standardized independently to ~N(0, 1).
    """
    if not magnitude_settings.features:
        return np.empty((original_data.shape[0], 0))

    features = []

    MF = _settings.MagnitudeFeature
    for feat_name in magnitude_settings.features:
        if feat_name == MF.MEAN:
            values = np.mean(original_data, axis=1, keepdims=True)

        elif feat_name == MF.STDEV:
            values = np.std(original_data, axis=1, keepdims=True)

        elif feat_name == MF.MEDIAN:
            values = np.median(original_data, axis=1, keepdims=True)

        elif feat_name == MF.MAD:
            values = _basic.median_absolute_deviation(original_data, axis=1, keepdims=True)

        elif feat_name == MF.QUANTILE_RANGE:
            q = magnitude_settings.quantile_range_q
            q_low = np.quantile(original_data, q, axis=1, keepdims=True)
            q_high = np.quantile(original_data, 1.0 - q, axis=1, keepdims=True)
            values = q_high - q_low

        elif feat_name == MF.BASELOAD:
            q = magnitude_settings.baseload_q
            values = np.quantile(original_data, q, axis=1, keepdims=True)

        elif feat_name == MF.PEAK:
            q = 1 - magnitude_settings.peak_q
            values = np.quantile(original_data, q, axis=1, keepdims=True)

        # Standardize independently: center by median, scale by MAD
        center = np.median(values)
        spread = _basic.median_absolute_deviation(values.ravel(), median=center)
        values = _safe_standardize(values, center, spread)

        features.append(values)

    if not features:
        return np.empty((original_data.shape[0], 0))

    return np.hstack(features)
