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

from typing import NamedTuple

import numpy as np

from sklearn.decomposition import PCA

from opendsm.common.clustering import settings as _settings


from opendsm.common.clustering.transform.normalize import (
    normalize,
    compute_magnitude_features,
    variance_cap,
)
from opendsm.common.clustering.transform.fpca import fpca_transform
from opendsm.common.clustering.transform.wavelet import wavelet_transform


class TransformResult(NamedTuple):
    """Result of ``transform_features``.

    Attributes
    ----------
    data : np.ndarray
        Full feature vector used for clustering.
    null_test_data : np.ndarray
        Original (unnormalized) data for null tests. No normalization
        artifacts, no variance cap distortion — just the raw meter
        readings. Null tests detect whether the original data has
        genuine cluster structure.
    """
    data: np.ndarray
    null_test_data: np.ndarray


def _adaptive_weight(source_var: float, target_var: float, balance: float) -> float:
    """Compute adaptive scaling weight for feature group balancing.

    Scales the target group so its total variance contribution is
    ``balance`` times the source group's total variance.
    """
    if target_var > 0:
        return np.sqrt(source_var / target_var) * balance
    return 0.0


def transform_features(
    data: np.ndarray,
    settings: _settings.ClusteringSettings,
) -> np.ndarray:
    """Transform features: normalize → variance cap → [raw + wavelet/fpca + magnitude].

    Feature vector order: pre-normalized raw (axis=0 post-norm) | wavelet PCA | magnitude.

    The per-sample normalization includes a variance cap (5th percentile)
    that suppresses pathological noise amplification on near-zero data.
    The cap's compression ratio (pre-cap / post-cap total variance)
    adaptively boosts the magnitude balance via log2: when shape features
    required heavy compression, magnitude carries more signal.
    """
    wt = settings.feature_transform.wavelet
    norm = settings.feature_transform.normalize

    # Keep reference to original data for robust CV computation
    original_data = data

    # Magnitude features from original (unnormalized) data
    mag_features = None
    if norm.enabled:
        mag_settings = settings.feature_transform.magnitude_features
        mag_features = compute_magnitude_features(data, mag_settings)
        if mag_features.shape[1] == 0:
            mag_features = None

    # Pre-transform normalization + variance cap
    pre_normalized = None
    cap_compression = 1.0
    if norm.enabled:
        data = normalize(data, norm)
        data, cap_compression = variance_cap(data, axis=norm._axis)
        pre_normalized = data.copy()
    else:
        pre_normalized = data.copy()

    # Null test data: globally normalized (axis=0) original data.
    # Per-sample normalization distorts population-level distances that
    # null tests need — flat meters get noise amplified, normal meters
    # get structure compressed. Global normalization preserves inter-sample
    # distance structure while putting features on a common scale.
    null_test_data = normalize(original_data, norm, axis=0) if norm.enabled else original_data.copy()

    # Apply main transform (wavelet or FPCA)
    explained_ratio = 1.0
    wavelet_skipped = False
    if settings.feature_transform.fpca.enabled:
        data = fpca_transform(data, settings)
        # FPCA doesn't return explained ratio yet — treat as full retention
        explained_ratio = 1.0

    elif wt.enabled:
        data, explained_ratio, cap_fraction = wavelet_transform(data, settings)
        # Skip wavelet output when auto-cap removed > 1/3 of levels.
        # This means the wavelet couldn't decompose deeply enough to add
        # value over the raw normalized features. Use pre-normalized data
        # instead (with axis=0 post-normalization for per-feature scaling).
        if cap_fraction > 1.0 / 3.0:
            data = normalize(pre_normalized, norm, axis=0) if norm.enabled else pre_normalized
            wavelet_skipped = True

    # Optional winsorization on transform output
    if norm.winsorize_threshold is not None:
        c = norm.winsorize_threshold
        data = np.clip(data, -c, c)

    # Assemble feature vector: [raw, wavelet, magnitude]
    groups = []

    # 1. Pre-normalized raw features, weighted by information lost in PCA.
    # Only append when a transform is active AND wasn't skipped (otherwise
    # data IS the pre-normalized features and appending would duplicate).
    transform_active = (settings.feature_transform.fpca.enabled or wt.enabled) and not wavelet_skipped
    if transform_active and explained_ratio < 1.0:
        raw_weight = np.sqrt(1.0 - explained_ratio)
        raw_post = normalize(pre_normalized, norm, axis=0) if norm.enabled else pre_normalized
        wavelet_var = np.sum(np.var(data, axis=0))
        raw_var = np.sum(np.var(raw_post, axis=0))
        if raw_var > 0:
            raw_scale = _adaptive_weight(wavelet_var, raw_var, raw_weight)
            groups.append(raw_post * raw_scale)

    # 2. Wavelet/FPCA features (reference scale)
    groups.append(data)

    # 3. Magnitude features, balanced against combined shape variance.
    # The base balance is boosted by the variance cap compression ratio
    # ONLY when the original data has meaningful magnitude variation
    # (robust CV > 0.5). This prevents boosting on uniformly flat data
    # where the cap compressed normalization noise but magnitude is also
    # constant.  Robust CV = IQR(row_means) / median(row_means).
    if mag_features is not None:
        shape_data = np.hstack(groups)
        total_shape_var = np.sum(np.var(shape_data, axis=0))
        total_mag_var = np.sum(np.var(mag_features, axis=0))
        base_balance = settings.feature_transform.magnitude_features.weight
        row_means = np.mean(original_data, axis=1)
        med_row = np.median(row_means)
        if med_row > 0:
            iqr_row = (np.percentile(row_means, 75)
                       - np.percentile(row_means, 25))
            robust_cv = iqr_row / med_row
        else:
            robust_cv = 0.0
        if cap_compression > 1.0 and robust_cv > 0.5:
            boost = max(1.0, np.log2(cap_compression))
        else:
            boost = 1.0
        balance = base_balance * boost
        mag_scale = _adaptive_weight(total_shape_var, total_mag_var, balance)
        groups.append(mag_features * mag_scale)

    result = np.hstack(groups)

    # Auto-PCA fallback: when n < features, reduce to n-1 components to
    # prevent degenerate distance matrices. Applied after all feature
    # assembly including raw and magnitude appending.
    n_samples, n_features = result.shape
    if n_samples < n_features and n_samples > 1:
        n_components = n_samples - 1
        result = PCA(n_components=n_components).fit_transform(result)

    return TransformResult(result, null_test_data)
