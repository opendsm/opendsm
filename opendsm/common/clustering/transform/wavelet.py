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

import numpy as np

import pywt

from sklearn.decomposition import PCA

from opendsm.common.clustering import settings as _settings

from opendsm.common.clustering.transform.normalize import normalize
from opendsm.common.clustering.transform.parallel_analysis import (
    _parallel_analysis_n_components,
)


# ---------------------------------------------------------------------------
# Wavelet transform
# ---------------------------------------------------------------------------

# Minimum coefficients per subband for PCA to be meaningful. Levels with
# fewer coefficients than this skip PCA in _reduce_single_subband, passing
# through raw DWT coefficients that add noise. Auto-capping n_levels in
# _dwt_coeffs ensures every subband meets this minimum.
_MIN_PCA_COEFFS = 4


def _dwt_coeffs(
    data: np.ndarray,
    wavelet: str,
    wavelet_mode: str,
    n_levels: int | None,
) -> tuple[list[np.ndarray], float]:
    """Compute DWT and return the list of per-level subband arrays.

    Returns
    -------
    subbands : list of arrays
        The list produced by ``pywt.wavedec`` — [cA_n, cD_n, ..., cD_1].
    cap_fraction : float
        Fraction of levels removed by auto-cap (0.0 = no cap, 0.5 = half
        the levels removed).  Used by ``transform_features`` to decide
        whether the wavelet decomposition was deep enough to be useful.

    Auto-caps n_levels so the coarsest subband has at least _MIN_PCA_COEFFS
    coefficients, ensuring every level can be PCA'd (noise-filtered) rather
    than passing through raw DWT coefficients.
    """
    n_points = data.shape[1]
    dwt_max_level = pywt.dwt_max_level(n_points, wavelet)

    if n_levels is None:
        n_levels = dwt_max_level
    elif n_levels > dwt_max_level:
        n_levels = dwt_max_level

    original_levels = n_levels

    # Cap so coarsest subband has enough coefficients for PCA.
    while n_levels > 1:
        coarsest_len = pywt.dwt_coeff_len(n_points, pywt.Wavelet(wavelet).dec_len, mode=wavelet_mode)
        for _ in range(n_levels - 1):
            coarsest_len = pywt.dwt_coeff_len(coarsest_len, pywt.Wavelet(wavelet).dec_len, mode=wavelet_mode)
        if coarsest_len >= _MIN_PCA_COEFFS:
            break
        n_levels -= 1

    cap_fraction = (original_levels - n_levels) / original_levels if original_levels > 0 else 0.0

    subbands = pywt.wavedec(data, wavelet=wavelet, mode=wavelet_mode, level=n_levels, axis=1)
    return subbands, cap_fraction


def _reduce_single_subband(
    band: np.ndarray,
    min_var_ratio: float | None,
    n_components: int | str | None,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Apply PCA to a single subband array.

    If the subband has < _MIN_PCA_COEFFS coefficients or <= 2 samples,
    return it as-is (not enough data for meaningful PCA).

    Returns (reduced_data, explained_variance_ratio) where
    explained_variance_ratio is the fraction of total variance retained
    by the selected components (1.0 if PCA was skipped).
    """
    n_samples, n_coeffs = band.shape
    if n_coeffs < _MIN_PCA_COEFFS or n_samples <= 2:
        return band, 1.0

    random_state = int(seed) if seed is not None else None

    max_components = min(n_samples - 1, n_coeffs)
    if n_components == "parallel_analysis":
        n = _parallel_analysis_n_components(band, method="pca", seed=seed or 0)
        n = min(n, max_components)
        pca = PCA(n_components=n, random_state=random_state)
    elif isinstance(n_components, int):
        pca = PCA(n_components=min(n_components, max_components),
                   random_state=random_state)
    elif n_components == "mle":
        pca = PCA(n_components="mle", random_state=random_state)
    else:
        ratio = min_var_ratio if min_var_ratio is not None else 0.95
        pca = PCA(n_components=ratio, random_state=random_state)

    reduced = pca.fit_transform(band)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return reduced, explained


def _reduce_wavelet_subbands(
    subbands: list[np.ndarray],
    min_var_ratio: float | None,
    n_components: int | str | None,
    seed: int | None = None,
    scope: str = "per_level",
) -> tuple[list[np.ndarray] | np.ndarray, float]:
    """Apply PCA to DWT subbands and return the feature matrix.

    Returns (reduced_data, explained_variance_ratio).

    Parameters
    ----------
    subbands : list of np.ndarray
        Per-level DWT coefficient arrays from ``_dwt_coeffs``.
    min_var_ratio : float or None
        Variance-ratio threshold (used when n_components is None).
    n_components : int, "mle", "parallel_analysis", or None
        Component-count specification.
    seed : int or None
        Random seed.
    scope : {"global", "per_level"}
        "global" flattens all subbands then applies PCA (legacy behavior).
        "per_level" applies PCA independently to each wavelet level,
        preserving scale separation.
    """
    if scope == "per_level":
        reduced = []
        # Track explained variance: weight each level's ratio by its
        # share of total input variance for a meaningful aggregate.
        level_total_vars = []
        level_explained_ratios = []
        for i, band in enumerate(subbands):
            band_seed = (seed + i * 7) if seed is not None else None
            band_reduced, band_explained = _reduce_single_subband(
                band, min_var_ratio, n_components, seed=band_seed,
            )
            reduced.append(band_reduced)
            level_total_vars.append(float(np.sum(np.var(band, axis=0))))
            level_explained_ratios.append(band_explained)

        # Variance-weighted average of per-level explained ratios
        total_input_var = sum(level_total_vars)
        if total_input_var > 0:
            explained_ratio = sum(
                v * r for v, r in zip(level_total_vars, level_explained_ratios)
            ) / total_input_var
        else:
            explained_ratio = 1.0

        return reduced, explained_ratio

    # Legacy global path
    flat = np.hstack(subbands)
    random_state = int(seed) if seed is not None else None

    if n_components == "parallel_analysis":
        n = _parallel_analysis_n_components(subbands, method="pca", seed=seed or 0)
        pca = PCA(n_components=n, random_state=random_state)
    elif n_components is not None:
        pca = PCA(n_components=n_components, random_state=random_state)
    else:
        ratio = min_var_ratio if min_var_ratio is not None else 0.95
        pca = PCA(n_components=ratio, random_state=random_state)

    result = pca.fit_transform(flat)
    explained_ratio = float(np.sum(pca.explained_variance_ratio_))
    return result, explained_ratio


def wavelet_transform(
    data: np.ndarray,
    settings: _settings.ClusteringSettings,
) -> tuple[np.ndarray, float, float]:
    """Transform data using DWT followed by PCA.

    Post-normalization scope follows pca_scope:
    - 'per_level': each level's PCA output is normalized independently before
      concatenation, preventing high-energy coarse levels from dominating.
    - 'global': the full concatenated feature matrix is normalized together.

    Returns (transformed_data, explained_variance_ratio, cap_fraction) where
    explained_variance_ratio is the variance-weighted average of per-level
    PCA explained variance ratios, and cap_fraction is the fraction of wavelet
    levels removed by auto-cap (0.0 = no cap, >0.33 = wavelet struggling).
    """
    wavelet_settings = settings.feature_transform.wavelet
    norm_settings = settings.feature_transform.normalize

    with warnings.catch_warnings():
        subbands, cap_fraction = _dwt_coeffs(
            data,
            wavelet_settings.wavelet_name,
            wavelet_settings.wavelet_mode,
            wavelet_settings.wavelet_n_levels,
        )

    result, explained_ratio = _reduce_wavelet_subbands(
        subbands,
        wavelet_settings.pca_min_variance_ratio_explained,
        wavelet_settings.pca_n_components,
        seed=wavelet_settings._seed,
        scope=wavelet_settings.pca_scope,
    )

    # per_level: result is a list of per-band arrays
    if isinstance(result, list):
        # Compute per-level variance weights before normalization destroys scale
        if wavelet_settings.variance_weighted:
            level_vars = np.array([np.var(band) for band in result])
            total_var = level_vars.sum()
            if total_var > 0:
                weights = np.sqrt(level_vars / total_var)
            else:
                weights = np.ones(len(result))
        else:
            weights = np.ones(len(result))

        if norm_settings.enabled:
            result = [normalize(band, norm_settings, axis=0) for band in result]

        result = [band * w for band, w in zip(result, weights)]
        return np.hstack(result), explained_ratio, cap_fraction

    # global: result is already hstacked; normalize the full matrix
    if norm_settings.enabled:
        result = normalize(result, norm_settings, axis=0)
    return result, explained_ratio, cap_fraction
