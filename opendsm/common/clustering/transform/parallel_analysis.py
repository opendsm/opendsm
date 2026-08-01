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

from sklearn.decomposition import PCA



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


def _compute_pa_eigenvalues(
    flat: np.ndarray,
    method: str,
    grid_points: np.ndarray | None,
    n_max: int,
    coefficient_map: np.ndarray | None = None,
) -> np.ndarray:
    """Compute eigenvalues for parallel analysis, normalised to sum to 1.

    Returns an array of length n_max.  Any trailing components beyond what
    the decomposition produces are zero-padded.

    ``coefficient_map`` is the FPCA basis map for ``grid_points``; it is built
    here when omitted, and passed in by the permutation loop so the
    pseudo-inverse is factored once rather than per permutation.
    """
    if method == "pca":
        pca = PCA(n_components=None)
        pca.fit(flat)
        eigs = pca.explained_variance_
    elif method == "fpca":
        if coefficient_map is None:
            coefficient_map = _fourier_coefficient_map(grid_points, n_max + 4)
        eigs = _fpca_eigenvalues(flat, coefficient_map)
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

    if isinstance(features, list):
        flat = np.hstack(features)
    else:
        flat = features
    n_samples, n_features = flat.shape

    if method == "fpca":
        n_max = max(1, min(n_samples - 1, n_features - 5))
    else:
        n_max = min(n_samples - 1, n_features)

    if n_max < 1:
        return 1

    n_B = _pa_n_permutations(n_samples)
    pct = _pa_percentile(n_samples)

    # Column shuffling preserves the shape, so every permutation shares this map.
    coefficient_map = None
    if method == "fpca":
        coefficient_map = _fourier_coefficient_map(grid_points, n_max + 4)

    actual = _compute_pa_eigenvalues(flat, method, grid_points, n_max, coefficient_map)

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
            perm_flat, method, grid_points, null_n_max, coefficient_map
        )
        n_fill = min(null_n_max, n_max)
        null[i, :n_fill] = null_eigs[:n_fill]

    threshold = np.percentile(null, pct, axis=0)

    return max(1, int(np.sum(actual > threshold)))
