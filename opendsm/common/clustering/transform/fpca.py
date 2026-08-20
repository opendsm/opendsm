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

from functools import partial

import numpy as np

from sklearn.decomposition import PCA

from opendsm.common.clustering import settings as _settings

from opendsm.common.clustering.transform.normalize import normalize
from opendsm.common.clustering.transform.parallel_analysis import (
    _parallel_analysis_n_components,
)



# ---------------------------------------------------------------------------
# Fourier basis
# ---------------------------------------------------------------------------

def _fourier_design(x: np.ndarray, n_basis: int) -> np.ndarray:
    """Orthonormal Fourier basis evaluated on ``x``, one column per function.

    The period is the span of ``x``.  An even ``n_basis`` is rounded up, since
    the sine/cosine terms only pair off at odd sizes.
    """
    x = np.asarray(x, dtype=float)
    n_basis += 1 - n_basis % 2
    period = x[-1] - x[0]

    columns = [np.full_like(x, 1.0 / np.sqrt(period))]
    k = 1
    while len(columns) < n_basis:
        w = 2.0 * np.pi * k / period
        columns.append(np.sqrt(2.0 / period) * np.sin(w * x))

        if len(columns) < n_basis:
            columns.append(np.sqrt(2.0 / period) * np.cos(w * x))
        k += 1
    design = np.column_stack(columns)

    return design


def _fourier_coefficient_map(x: np.ndarray, n_basis: int) -> np.ndarray:
    """Least-squares map from samples on the grid ``x`` to Fourier coefficients.

    Depends only on the grid and the basis size, so callers that transform many
    curve sets on one grid build it once and reuse it.
    """
    coefficient_map = np.linalg.pinv(_fourier_design(x, n_basis))

    return coefficient_map


def _fpca_eigenvalues(y: np.ndarray, coefficient_map: np.ndarray) -> np.ndarray:
    """Descending eigenvalues of the Fourier-coefficient covariance.

    The Fourier basis is orthonormal, so its Gram matrix is the identity and
    functional PCA reduces to ordinary PCA on the basis coefficients.  Any
    non-orthonormal basis would need the coefficients weighted by the Cholesky
    factor of the Gram matrix first.
    """
    coefficients = (coefficient_map @ y.T).T
    centered = coefficients - coefficients.mean(axis=0)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]

    return np.maximum(eigenvalues, 0.0)


def _fpca_max_components(n_samples: int, n_features: int) -> int:
    """Largest component count the basis supports for this shape."""
    return max(1, min(n_samples - 1, n_features - 5))


def _fpca_spectrum(
    data: np.ndarray,
    coefficient_map: np.ndarray,
    n_max: int,
) -> np.ndarray:
    """FPCA eigenvalue spectrum for parallel analysis, truncated to ``n_max``."""
    eigenvalues = _fpca_eigenvalues(data, coefficient_map)

    return eigenvalues[:n_max]


def _fpca_explained_variance(
    y: np.ndarray,
    x: np.ndarray,
    n_max: int,
) -> np.ndarray:
    """Fit FPCA with n_max components and return explained_variance_ratio_."""
    eigenvalues = _fpca_eigenvalues(y, _fourier_coefficient_map(x, n_max + 4))
    total = eigenvalues.sum()

    if total < 1e-10:
        return np.zeros(n_max)
    ratios = eigenvalues[:n_max] / total

    return ratios


def _fpca_transform_with_n(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Fit FPCA with exactly n components and return the transformed features.

    ``covariance_eigh`` is requested explicitly because the default ``auto``
    solver selects randomised SVD at the shapes this path produces, which is
    seeded by ``random_state`` and would make the scores non-reproducible.
    """
    coefficients = (_fourier_coefficient_map(x, n + 4) @ y.T).T
    fpca = PCA(n_components=n, svd_solver="covariance_eigh")
    scores = fpca.fit_transform(coefficients)

    return scores


# ---------------------------------------------------------------------------
# FPCA transform
# ---------------------------------------------------------------------------

class FpcaError(Exception):
    pass


def _fpca_base(
    x: np.ndarray,
    y: np.ndarray,
    min_var_ratio: float,
) -> np.ndarray:
    """FPCA with automatic n_components via variance-ratio threshold.

    Two-pass: first fit (n_max components) determines n from the cumulative
    explained variance ratio; second fit uses exactly n components so the
    Fourier basis size is appropriate for the retained dimensionality.
    """
    if 0 >= min_var_ratio or min_var_ratio >= 1:
        raise FpcaError("min_var_ratio but be greater than 0 and less than 1")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise FpcaError("provided non finite values for fpca")
    if len(x) == 0 or len(y) == 0:
        raise FpcaError("provided empty values for fpca")

    n_max = max(1, int(np.min(np.array(np.shape(y)) - [1, 5])))

    eig_ratios = _fpca_explained_variance(y, x, n_max)
    var_ratio_arr = np.cumsum(eig_ratios) - min_var_ratio
    n = int(np.argmin(var_ratio_arr < 0.0) + 1)

    return _fpca_transform_with_n(x, y, n)


def fpca_transform(
    data: np.ndarray,
    settings: _settings.ClusteringSettings,
) -> np.ndarray:
    fpca_settings = settings.feature_transform.fpca

    if not np.all(np.isfinite(data)):
        raise FpcaError("provided non finite values for fpca")
    if len(data) == 0:
        raise FpcaError("provided empty values for fpca")

    x = np.arange(data.shape[1])

    if settings._seed is None:
        seed = 0
    else:
        seed = settings._seed

    if fpca_settings.use_parallel_analysis:
        n_max = _fpca_max_components(*data.shape)
        # Bound once here: every permutation reuses the same factorisation.
        spectrum = partial(
            _fpca_spectrum,
            coefficient_map=_fourier_coefficient_map(x, n_max + 4),
            n_max=n_max,
        )
        n = _parallel_analysis_n_components(data, spectrum, seed=seed)
        result = _fpca_transform_with_n(x, data, n)
    else:
        result = _fpca_base(x, data, fpca_settings.min_var_ratio)

    norm_settings = settings.feature_transform.normalize
    if norm_settings.enabled:
        result = normalize(result, norm_settings, axis=0)

    return result
