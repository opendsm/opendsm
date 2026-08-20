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

from collections.abc import Callable

import numpy as np

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


def _pca_spectrum(data: np.ndarray) -> np.ndarray:
    """Descending PCA eigenvalues of ``data``, the spectrum Horn's method assumes.

    Truncated to the largest component count PCA can support for this shape.
    """
    pca = PCA(n_components=None)
    pca.fit(data)
    n_max = min(data.shape[0] - 1, data.shape[1])

    return pca.explained_variance_[:n_max]


def _normalised_spectrum(eigenvalues: np.ndarray, n_max: int) -> np.ndarray:
    """Pad or truncate to ``n_max`` and normalise to sum 1.

    Normalising makes the comparison against the permutation null
    scale-invariant; padding covers decompositions that return fewer
    components than the shape allows.
    """
    result = np.zeros(n_max)
    n_fill = min(len(eigenvalues), n_max)
    result[:n_fill] = eigenvalues[:n_fill]

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
    spectrum: Callable[[np.ndarray], np.ndarray] = _pca_spectrum,
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
        A list of DWT subband arrays as returned by ``_dwt_coeffs``, or a flat
        2-D array (n_samples, n_features/n_timepoints).
    spectrum : callable
        Maps a data matrix to that decomposition's descending eigenvalues,
        already truncated to the component count it supports.  The retained
        count is the length of the spectrum returned for the real data, so
        anything the decomposition needs — a basis, a factorisation — is bound
        by the caller and reused across every permutation.
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
    inputs, column-wise permutation is used.
    """
    rng = np.random.RandomState(seed)

    if isinstance(features, list):
        flat = np.hstack(features)
    else:
        flat = features

    actual_eigenvalues = spectrum(flat)
    n_max = len(actual_eigenvalues)

    if n_max < 1:
        return 1

    n_samples = flat.shape[0]
    n_B = _pa_n_permutations(n_samples)
    pct = _pa_percentile(n_samples)
    actual = _normalised_spectrum(actual_eigenvalues, n_max)

    null = np.zeros((n_B, n_max))
    for i in range(n_B):
        if isinstance(features, list):
            perm_flat = _block_permute_dwt(features, rng)
        else:
            perm_flat = flat.copy()
            for col in range(perm_flat.shape[1]):
                rng.shuffle(perm_flat[:, col])

        null[i] = _normalised_spectrum(spectrum(perm_flat), n_max)

    threshold = np.percentile(null, pct, axis=0)

    return max(1, int(np.sum(actual > threshold)))
