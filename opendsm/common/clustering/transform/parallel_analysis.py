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

from skfda.representation.grid import FDataGrid as _FDataGrid
from skfda.representation.basis import FourierBasis as _FourierBasis
from skfda.preprocessing.dim_reduction import FPCA as _FPCA

from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Parallel Analysis helpers
# ---------------------------------------------------------------------------

def _sigmoid_scalar(x: float, x0: float, k: float) -> float:
    """Numerically stable scalar sigmoid: 1 / (1 + exp(-(x - x0) / k))."""
    z = (x - x0) / k
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return ez / (ez + 1.0)


def _pa_n_permutations(n_samples: int) -> int:
    """Number of PA permutations as a smooth inverse-square function of n_samples.

    Decreases from ~300 at small n, approaching 15 asymptotically at large n.
    Uses 1/(1 + (n/25)²) decay — no hard floor, continuous derivative throughout.

    Representative values:
        n=7  → ~279   n=25 → 157   n=84 → ~38   n=365 → ~16   n=2000 → 15
    """
    return int(15 + 285 / (1.0 + (float(n_samples) / 25.0) ** 2))


def _pa_percentile(n_samples: int) -> float:
    """PA null-distribution threshold as a smooth sigmoid function of n_samples.

    Smoothly increases from 75 (over-retain, small n) to 95 (standard, large n).
    Transition centred at n=40, width 12.  Well-behaved for all n >= 1.
    """
    return 75.0 + 20.0 * _sigmoid_scalar(float(n_samples), x0=40.0, k=12.0)


def _fpca_explained_variance(
    y: np.ndarray,
    x: np.ndarray,
    n_max: int,
) -> np.ndarray:
    """Fit FPCA with n_max components and return explained_variance_ratio_.

    Used for both the variance-ratio threshold in ``_fpca_base`` and the
    parallel analysis null distribution in ``_parallel_analysis_n_components``.
    """
    n_basis = n_max + 4
    fd = _FDataGrid(grid_points=x, data_matrix=y)
    basis_fd = fd.to_basis(_FourierBasis(n_basis=n_basis))
    fpca = _FPCA(n_components=n_max, components_basis=_FourierBasis(n_basis=n_basis))
    fpca.fit(basis_fd)
    return np.asarray(fpca.explained_variance_ratio_)


def _fpca_transform_with_n(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Fit FPCA with exactly n components and return the transformed features."""
    n_basis = n + 4
    fd = _FDataGrid(grid_points=x, data_matrix=y)
    basis_fd = fd.to_basis(_FourierBasis(n_basis=n_basis))
    fpca = _FPCA(n_components=n, components_basis=_FourierBasis(n_basis=n_basis))
    fpca.fit(basis_fd)
    return fpca.transform(basis_fd)


def _compute_pa_eigenvalues(
    flat: np.ndarray,
    method: str,
    grid_points: np.ndarray | None,
    n_max: int,
) -> np.ndarray:
    """Compute eigenvalues for parallel analysis, normalised to sum to 1.

    Returns an array of length n_max.  Any trailing components beyond what
    the decomposition produces are zero-padded.
    """
    if method == "pca":
        pca = PCA(n_components=None)
        pca.fit(flat)
        eigs = pca.explained_variance_
    elif method == "fpca":
        eigs = _fpca_explained_variance(flat, grid_points, n_max)
    else:
        raise ValueError(f"Unknown PA method: {method!r}")

    result = np.zeros(n_max)
    n_fill = min(len(eigs), n_max)
    result[:n_fill] = eigs[:n_fill]

    total = result.sum()
    if total < 1e-10:
        return result
    return result / total


def _block_permute_dwt(
    subbands: list[np.ndarray],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Block permutation for DWT subbands.

    Each subband (decomposition level) receives an independent row shuffle.
    This preserves within-band coefficient correlations (arising from the DWT
    filter overlap) while destroying cross-level and between-sample structure,
    giving a more accurate null model than column-wise permutation.

    Returns a flat (n_samples, total_n_coeffs) array ready for PCA.
    """
    n_samples = subbands[0].shape[0]
    return np.hstack([band[rng.permutation(n_samples)] for band in subbands])


def _parallel_analysis_n_components(
    features: list[np.ndarray] | np.ndarray,
    method: str = "pca",
    grid_points: np.ndarray | None = None,
    seed: int = 0,
) -> int:
    """Determine n_components via Parallel Analysis (Horn 1965).

    Permutes the data B times to build a null eigenvalue distribution, then
    retains components where the actual normalised eigenvalue exceeds the null
    percentile threshold.  Both B and the threshold vary smoothly with
    n_samples via sigmoid functions: more permutations and a lower
    (over-retain) threshold at small n, converging to standard values
    (B=30, 95th pct) at large n.

    Parameters
    ----------
    features : list of np.ndarray or np.ndarray
        For PCA with block permutation: list of DWT subband arrays as returned
        by ``_dwt_coeffs``.  For FPCA: flat 2-D array
        (n_samples, n_features/n_timepoints).
    method : {"pca", "fpca"}
        Decomposition method.
    grid_points : np.ndarray, optional
        Required for FPCA: uniform time-point grid (typically np.arange(T)).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    int
        Number of components to retain (at least 1).

    Notes
    -----
    Eigenvalues are normalised by their sum before comparison so the test is
    scale-invariant.

    For DWT-derived list inputs, block permutation is used: each subband
    receives an independent row shuffle, preserving within-band coefficient
    correlations while destroying cross-level structure.  For flat-array
    inputs (FPCA), column-wise permutation is used.
    """
    rng = np.random.RandomState(seed)

    flat = np.hstack(features) if isinstance(features, list) else features
    n_samples, n_features = flat.shape

    if method == "fpca":
        n_max = max(1, min(n_samples - 1, n_features - 5))
    else:
        n_max = min(n_samples - 1, n_features)

    if n_max < 1:
        return 1

    n_B = _pa_n_permutations(n_samples)
    pct = _pa_percentile(n_samples)

    actual = _compute_pa_eigenvalues(flat, method, grid_points, n_max)

    null = np.zeros((n_B, n_max))
    for i in range(n_B):
        if isinstance(features, list):
            perm_flat = _block_permute_dwt(features, rng)
        else:
            perm_flat = flat.copy()
            for col in range(perm_flat.shape[1]):
                rng.shuffle(perm_flat[:, col])

        n_perm = perm_flat.shape[0]
        if method == "fpca":
            null_n_max = max(1, min(n_perm - 1, n_features - 5))
        else:
            null_n_max = min(n_perm - 1, n_features)

        if null_n_max < 1:
            continue

        null_eigs = _compute_pa_eigenvalues(
            perm_flat, method, grid_points, null_n_max
        )
        n_fill = min(null_n_max, n_max)
        null[i, :n_fill] = null_eigs[:n_fill]

    threshold = np.percentile(null, pct, axis=0)
    return max(1, int(np.sum(actual > threshold)))
