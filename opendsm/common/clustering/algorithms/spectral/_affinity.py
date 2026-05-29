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

from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import eigsh as _eigsh, svds as _svds

from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import NearestNeighbors


_SELF_TUNING_SPARSE_THRESHOLD = 500


_MAX_EMBEDDING_COMPONENTS = 100  # eigsh cost is O(n·k²); cap to stay practical


def _auto_diffusion_time(eigenvalues: np.ndarray) -> int:
    """Select diffusion time from the spectral gap.

    Finds the largest gap in the eigenvalue spectrum (descending).
    Sets t so that the eigenvalue just below the gap decays to < 1%
    of its original value: t = ceil(log(0.01) / log(λ_below)).

    This preserves cluster-separation eigenvectors (λ ≈ 1, above the gap)
    while suppressing within-cluster noise (λ < gap, decays rapidly).

    Clamped to [2, 10]: t=1 is just self-tuning (no diffusion benefit),
    t > 10 over-smooths (crushes all but the top 1-2 eigenvectors).
    """
    if len(eigenvalues) < 2:
        return 2
    gaps = -np.diff(eigenvalues)  # positive gaps (eigenvalues are descending)
    best = int(np.argmax(gaps))
    lam_below = float(eigenvalues[best + 1])
    if lam_below <= 0.01:
        return 2  # already negligible, minimal diffusion
    if lam_below >= 0.99:
        return 10  # no clear gap, use maximum smoothing
    t = int(np.ceil(np.log(0.01) / np.log(lam_below)))
    return max(2, min(t, 10))


def _diffusion_map(
    A,
    diffusion_time: int | None,
    n_components: int = 20,
    seed: int | None = None,
    alpha: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Compute diffusion map embedding from an affinity matrix.

    Instead of materializing P^t (dense n×n), computes the top eigenvectors
    of the normalized transition matrix and scales by λ^t.  Memory is O(n·k)
    where k = min(n_components, _MAX_EMBEDDING_COMPONENTS).

    *alpha* controls the Coifman & Lafon (2006) α-normalization:
      - α=1.0: standard normalized Laplacian (removes all density dependence)
      - α=0.5: recommended default — preserves some density information while
        symmetrizing, better for varying-density clusters
      - α=0.0: unnormalized (density-weighted, rarely useful)

    Returns (embedding, t_used) where embedding has shape (n, n_components).
    Euclidean distance in this space equals diffusion distance at time t.
    When diffusion_time is None, t is auto-selected from the spectral gap.
    """
    if issparse(A):
        row_sums = np.asarray(A.sum(axis=1)).ravel()
    else:
        row_sums = A.sum(axis=1)
    row_sums = np.maximum(row_sums, 1e-20)

    # Alpha-normalization: A_α = D^{-α} A D^{-α}
    # This re-weights the affinity to reduce the influence of density.
    # α=1 fully removes density; α=0.5 partially preserves it.
    if alpha != 0.0:
        d_neg_alpha = row_sums ** (-alpha)
        if issparse(A):
            D_neg_alpha = diags(d_neg_alpha)
            A_alpha = D_neg_alpha @ A @ D_neg_alpha
        else:
            A_alpha = A * np.outer(d_neg_alpha, d_neg_alpha)
        # Recompute row sums after alpha-normalization
        if issparse(A_alpha):
            row_sums = np.asarray(A_alpha.sum(axis=1)).ravel()
        else:
            row_sums = A_alpha.sum(axis=1)
        row_sums = np.maximum(row_sums, 1e-20)
    else:
        A_alpha = A

    d_inv_sqrt = 1.0 / np.sqrt(row_sums)

    # Symmetric normalized matrix: M = D^{-1/2} A_α D^{-1/2}
    # M has the same eigenvalues as the transition matrix P = D^{-1}A_α
    # but is symmetric → real eigenvalues, stable eigsh.
    if issparse(A_alpha):
        D_inv_sqrt = diags(d_inv_sqrt)
        M = D_inv_sqrt @ A_alpha @ D_inv_sqrt
    else:
        M = A_alpha * np.outer(d_inv_sqrt, d_inv_sqrt)

    n = A.shape[0]
    n_components = min(n_components, _MAX_EMBEDDING_COMPONENTS)
    n_eig = min(n_components + 1, n - 1)

    if issparse(M) and n > 50:
        v0 = np.random.default_rng(seed).standard_normal(n)
        eigenvalues, eigenvectors = _eigsh(M, k=n_eig, which="LM", v0=v0)
    else:
        if issparse(M):
            M = M.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # eigh returns ascending; take the top n_eig
        eigenvalues = eigenvalues[-n_eig:]
        eigenvectors = eigenvectors[:, -n_eig:]

    # Sort descending by eigenvalue
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Drop the trivial first eigenvector (eigenvalue ≈ 1, constant vector)
    eigenvalues = eigenvalues[1:]
    eigenvectors = eigenvectors[:, 1:]

    # Keep eigenvectors in the symmetric basis (eigenvectors of M).
    # The transition-matrix basis (ψ = d^{-1/2} φ) would weight points
    # inversely by degree, biasing PCA splits toward low-degree outliers.
    # The symmetric basis gives equal weight to all points, matching the
    # normalized-cut objective used by spectral bisection.

    # Auto-select diffusion time from spectral gap if not specified
    if diffusion_time is None:
        diffusion_time = _auto_diffusion_time(eigenvalues)

    # Diffusion map: scale each eigenvector by λ^t
    scales = np.maximum(eigenvalues, 0.0) ** diffusion_time
    return eigenvectors * scales[None, :], diffusion_time


def _sigma_floor(sigma: np.ndarray) -> np.ndarray:
    """Apply a data-derived floor to local scaling parameters.

    Points in degenerate neighborhoods (very small sigma relative to the
    population) get their sigma raised to the floor, preventing the
    self-tuning kernel from amplifying noise-level distances into
    artificial affinity structure.

    Floor is median(sigma) * 0.1 — the median is robust to outliers,
    and 10% of median preserves genuine local scaling variation while
    preventing near-zero sigma from creating false clusters.
    """
    floor = np.median(sigma) * 0.1
    return np.maximum(sigma, floor)


def _self_tuning_affinity_dense(data: np.ndarray, k: int) -> np.ndarray:
    """Zelnik-Manor & Perona (2004) locally-scaled affinity matrix, dense path."""
    n = data.shape[0]
    k_eff = min(k, n - 1)
    D_sq = pairwise_distances(data, metric="sqeuclidean")
    sigma_sq = np.partition(D_sq, k_eff, axis=1)[:, k_eff]
    sigma = np.sqrt(np.maximum(sigma_sq, 1e-20))
    sigma = _sigma_floor(sigma)
    return np.exp(-D_sq / np.outer(sigma, sigma))


def _self_tuning_affinity_sparse(data: np.ndarray, k: int, k_connect: int):
    """Zelnik-Manor & Perona (2004) locally-scaled affinity matrix, sparse path."""
    n = data.shape[0]
    k_eff = min(k, n - 1)
    k_connect = min(k_connect, n - 2)
    k_query = max(k_eff, k_connect) + 1

    nn = NearestNeighbors(n_neighbors=k_query, algorithm="auto", metric="euclidean")
    nn.fit(data)
    dist, idx = nn.kneighbors(data)

    sigma = _sigma_floor(np.maximum(dist[:, k_eff], 1e-10))

    rows = np.repeat(np.arange(n), k_connect)
    cols = idx[:, 1:k_connect + 1].ravel()
    d_flat = dist[:, 1:k_connect + 1].ravel()

    vals = np.exp(-(d_flat ** 2) / (sigma[rows] * sigma[cols]))
    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)


def _power_iteration_fiedler(
    A,
    max_iter: int = 100,
    tol: float = 1e-8,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Lin & Cohen (2010) power iteration clustering.

    Computes the Fiedler vector (2nd eigenvector of normalized Laplacian)
    via power iteration on the normalized affinity matrix, without explicit
    eigendecomposition.  O(n * nnz_per_row * iter) — faster than eigsh for
    large sparse matrices where only the top eigenvector is needed.

    Returns (fiedler_vector, lambda2) where lambda2 is the estimated
    2nd eigenvalue of the transition matrix (1 - λ_laplacian).
    """
    n = A.shape[0]
    if issparse(A):
        row_sums = np.asarray(A.sum(axis=1)).ravel()
    else:
        row_sums = A.sum(axis=1)
    row_sums = np.maximum(row_sums, 1e-20)

    # Normalized affinity: P = D^{-1} A (row-stochastic)
    if issparse(A):
        D_inv = diags(1.0 / row_sums)
        P = D_inv @ A
    else:
        P = A / row_sums[:, None]

    # Power iteration to find the 2nd eigenvector of P.
    # The 1st eigenvector is the constant vector (eigenvalue 1).
    # Deflate by projecting out the constant direction at each step.
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    # Deflate: remove the component along the stationary distribution (∝ row_sums)
    pi = row_sums / row_sums.sum()
    v -= np.dot(v, pi) / np.dot(pi, pi) * pi
    norm = np.linalg.norm(v)
    if norm < 1e-20:
        return np.zeros(n), 0.0
    v /= norm

    lambda2 = 0.0
    for _ in range(max_iter):
        v_new = P @ v if issparse(P) else P @ v
        # Deflate constant eigenvector
        v_new -= np.dot(v_new, pi) / np.dot(pi, pi) * pi
        lambda2_new = float(np.dot(v, v_new))
        norm = np.linalg.norm(v_new)
        if norm < 1e-20:
            break
        v_new /= norm
        if abs(lambda2_new - lambda2) < tol:
            lambda2 = lambda2_new
            v = v_new
            break
        lambda2 = lambda2_new
        v = v_new

    # Convert: Laplacian eigenvalue = 1 - transition eigenvalue
    laplacian_lambda2 = max(0.0, 1.0 - abs(lambda2))
    return v, laplacian_lambda2


def _anisotropic_affinity_sparse(
    data: np.ndarray,
    k: int,
    k_connect: int,
    n_local_pcs: int = 3,
):
    """Singer (2006) anisotropic affinity with local covariance estimation.

    For each point, estimates a local covariance from its k neighbors and
    computes Mahalanobis-like distances.  Uses a low-rank approximation
    (top n_local_pcs principal components of each neighborhood) to stay
    practical at high d.

    The affinity between i and j is:
      A_ij = exp(-0.5 * (d_M(i,j)² + d_M(j,i)²) / 2)
    where d_M(i,j) is the Mahalanobis distance from j to i's local model.
    Symmetrized via the average of both directions.
    """
    n, d = data.shape
    k_eff = min(k, n - 1)
    k_connect = min(k_connect, n - 2)
    k_query = max(k_eff, k_connect) + 1
    n_pcs = min(n_local_pcs, d, k_eff - 1)

    nn = NearestNeighbors(n_neighbors=k_query, algorithm="auto", metric="euclidean")
    nn.fit(data)
    dist, idx = nn.kneighbors(data)

    # Precompute local PCA for each point's neighborhood
    # local_axes[i] has shape (n_pcs, d) — top PCs of i's neighborhood
    # local_scales[i] has shape (n_pcs,) — corresponding singular values
    local_axes = np.empty((n, n_pcs, d), dtype=np.float32)
    local_inv_scales = np.empty((n, n_pcs), dtype=np.float32)

    for i in range(n):
        neighbors = data[idx[i, 1:k_eff + 1]]  # (k_eff, d)
        centered = neighbors - data[i]
        if n_pcs >= min(k_eff, d):
            _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        else:
            # Truncated SVD for efficiency when d is large
            # svds needs k < min(m, n)
            k_svd = min(n_pcs, min(centered.shape) - 1)
            if k_svd < 1:
                local_axes[i] = 0
                local_inv_scales[i] = 0
                continue
            _, s, Vt = _svds(centered.astype(np.float64), k=k_svd)
            # svds returns in ascending order
            s = s[::-1]
            Vt = Vt[::-1]
        s = np.maximum(s[:n_pcs], 1e-10)
        local_axes[i] = Vt[:n_pcs]
        local_inv_scales[i] = 1.0 / s

    # Floor on inverse scales: prevent degenerate neighborhoods from
    # amplifying noise via huge Mahalanobis distances. Uses the same
    # principle as _sigma_floor — median-derived, robust to outliers.
    median_inv_scale = np.median(local_inv_scales)
    if median_inv_scale > 0:
        inv_scale_cap = median_inv_scale * 10.0
        np.minimum(local_inv_scales, inv_scale_cap, out=local_inv_scales)

    # Build sparse affinity: for each edge (i, j), compute anisotropic distance
    rows = np.repeat(np.arange(n), k_connect)
    cols = idx[:, 1:k_connect + 1].ravel()
    diff = data[cols] - data[rows]  # (n_edges, d)

    # Mahalanobis distance from j to i's local model:
    # d_M(i,j)² = Σ_pc (diff · axis_pc / scale_pc)²
    # Project diff onto each point's local PCs and scale
    proj_i = np.einsum('ed,ecd->ec', diff, local_axes[rows])  # (n_edges, n_pcs)
    mahal_sq_i = np.sum((proj_i * local_inv_scales[rows]) ** 2, axis=1)

    proj_j = np.einsum('ed,ecd->ec', diff, local_axes[cols])  # (n_edges, n_pcs)
    mahal_sq_j = np.sum((proj_j * local_inv_scales[cols]) ** 2, axis=1)

    # Symmetrized Mahalanobis distance
    mahal_sq = (mahal_sq_i + mahal_sq_j) / 2.0
    vals = np.exp(-0.5 * mahal_sq)

    A = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A.maximum(A.T)


def _affinity_matrix(
    data: np.ndarray,
    algo: SpectralClustering,
):
    """Compute the affinity matrix for the given data."""
    params = algo.kernel_params
    if params is None:
        params = {}
    if not callable(algo.affinity):
        params["gamma"] = algo.gamma
        params["degree"] = algo.degree
        params["coef0"] = algo.coef0

    X = pairwise_kernels(
        data, metric=algo.affinity, filter_params=True, **params
    )
    return X
