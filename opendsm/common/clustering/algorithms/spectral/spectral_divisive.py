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

import heapq

import numpy as np

from scipy.sparse import csgraph, csr_matrix as _csr
from scipy.sparse.linalg import eigsh as _sparse_eigsh

from sklearn.neighbors import NearestNeighbors

from opendsm.common.clustering.algorithms.settings import (
    AffinityMatrixOptions,
    SpectralSettings,
)
from opendsm.common.clustering.metrics.labels import ClusteringResult
from opendsm.common.clustering.metrics.settings import SmallClusterMode
from opendsm.common.clustering.algorithms.k_medians import kmedians_refine
from opendsm.common.clustering.algorithms.spectral._affinity import (
    _MAX_EMBEDDING_COMPONENTS,
    _SELF_TUNING_SPARSE_THRESHOLD,
    _anisotropic_affinity_sparse,
    _auto_diffusion_time,
    _diffusion_map,
    _power_iteration_fiedler,
    _self_tuning_affinity_dense,
    _self_tuning_affinity_sparse,
)


_ANN_THRESHOLD = 100_000  # use approximate NN only when n_query exceeds this
_FIEDLER_FALLBACK = 10_000  # Nyström sub-clusters below this use per-split Fiedler


def _knn_query(reference: np.ndarray, query: np.ndarray, k: int):
    """k-NN query: returns (distances, indices) both shape (n_query, k).

    Uses hnswlib (approximate) when installed and n_query > _ANN_THRESHOLD.
    Falls back to sklearn exact k-NN otherwise.
    """
    if query.shape[0] > _ANN_THRESHOLD:
        try:
            import hnswlib

            n_ref = reference.shape[0]
            dim = reference.shape[1]
            index = hnswlib.Index(space="l2", dim=dim)
            index.init_index(max_elements=n_ref, ef_construction=200, M=16)
            index.add_items(reference.astype(np.float32))
            index.set_ef(max(k * 2, 100))
            idx, dist_sq = index.knn_query(query.astype(np.float32), k=k)
            return np.sqrt(dist_sq), idx  # hnswlib returns squared L2
        except ImportError:
            pass

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nn.fit(reference)
    return nn.kneighbors(query)


def _nystrom_embedding(
    data: np.ndarray,
    k_st: int,
    m: int,
    n_components: int,
    diffusion_time: int | None = 1,
    seed: int | None = None,
) -> np.ndarray | None:
    """Compute a global spectral embedding via Nyström approximation.

    Builds exact eigenvectors on a random sample of m points, then extends
    to all n points using sparse k-NN affinity.  The result is an (n, n_components)
    embedding where each row is a point's coordinates in spectral space.

    All subsequent bisections can operate on this embedding directly using
    cheap dense operations — no per-split affinity or eigsh needed.

    Returns (n, n_components) float64 array, or None on failure.
    """
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    sample_idx = np.sort(rng.choice(n, m, replace=False))
    rest_idx = np.setdiff1d(np.arange(n), sample_idx)

    sample_data = data[sample_idx]
    rest_data = data[rest_idx]
    k_st_eff = max(3, min(k_st, m // 5))

    # ── Step 1: exact eigenvectors on the m×m sample ─────────────────
    if m > _SELF_TUNING_SPARSE_THRESHOLD:
        k_connect = min(2 * (k_st_eff + 2), m - 2)
        A_mm = _self_tuning_affinity_sparse(sample_data, k_st_eff, k_connect)
    else:
        A_mm = _self_tuning_affinity_dense(sample_data, k_st_eff)
        A_mm = _csr(A_mm)

    L_mm = csgraph.laplacian(A_mm, normed=True)
    n_eig = min(n_components + 1, m - 1)
    v0 = np.random.default_rng(seed).standard_normal(L_mm.shape[0])
    try:
        eigenvalues, eigenvectors = _sparse_eigsh(
            L_mm, k=n_eig, which="SM", maxiter=2000, tol=1e-6, v0=v0
        )
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    except Exception:
        return None

    # Drop the trivial first eigenvector (constant, eigenvalue ≈ 0)
    eigvecs_sample = eigenvectors[:, 1:n_eig]   # (m, n_components)

    # Diffusion scaling: λ_transition = 1 - λ_laplacian, scale by λ_t^t
    # diffusion_time=1 means no scaling (self-tuning only).
    # diffusion_time=None means auto-select from spectral gap.
    if diffusion_time != 1:
        laplacian_eigs = eigenvalues[1:n_eig]
        transition_eigs = np.clip(1.0 - laplacian_eigs, 0.0, 1.0)
        if diffusion_time is None:
            sorted_teigs = np.sort(transition_eigs)[::-1]
            t = _auto_diffusion_time(sorted_teigs)
        else:
            t = diffusion_time
        scales = transition_eigs ** t
        eigvecs_sample *= scales[None, :]

    # Canonicalize sign: make the largest-magnitude entry positive per column.
    for col in range(eigvecs_sample.shape[1]):
        if eigvecs_sample[np.argmax(np.abs(eigvecs_sample[:, col])), col] < 0:
            eigvecs_sample[:, col] *= -1

    # ── Step 2: sparse Nyström extension to all n points ─────────────
    k_nn = min(2 * (k_st_eff + 2), m - 1)

    dist_nm, idx_nm = _knn_query(sample_data, rest_data, k_nn + 1)

    # Local scales with global fallback for boundary stability.
    # Points at cluster boundaries have inflated k-NN distances (few nearby
    # sample points), producing blurry affinities.  Clamp to the median
    # sample sigma to prevent over-smoothing at boundaries.
    nn_sample = NearestNeighbors(n_neighbors=k_st_eff + 1, algorithm="auto", metric="euclidean")
    nn_sample.fit(sample_data)
    sigma_sample = nn_sample.kneighbors(sample_data)[0][:, k_st_eff]
    sigma_sample = np.maximum(sigma_sample, 1e-10)
    sigma_global = float(np.median(sigma_sample))

    sigma_rest = dist_nm[:, min(k_st_eff, k_nn)]
    sigma_rest = np.clip(sigma_rest, 1e-10, 3.0 * sigma_global)

    # Sparse affinity between rest and sample neighbors
    sigma_neighbors = sigma_sample[idx_nm]                     # (n_rest, k_nn+1)
    affinities = np.exp(
        -(dist_nm ** 2) / (sigma_rest[:, None] * sigma_neighbors)
    )                                                          # (n_rest, k_nn+1)
    row_sums = np.maximum(affinities.sum(axis=1, keepdims=True), 1e-10)
    affinities /= row_sums                                     # row-normalize in-place

    # Extend eigenvectors via sparse matmul: W @ eigvecs_sample
    # W is (n_rest, m) sparse with k_nn+1 entries per row.
    # Avoids the (n_rest, k_nn+1, n_components) 3D gather that OOMs at large n.
    n_rest = len(rest_idx)
    k_per_row = idx_nm.shape[1]
    rows = np.repeat(np.arange(n_rest), k_per_row)
    W = _csr((affinities.ravel(), (rows, idx_nm.ravel())), shape=(n_rest, m))
    eigvecs_rest = W @ eigvecs_sample

    # ── Assemble full embedding ──────────────────────────────────────
    embedding = np.empty((n, eigvecs_sample.shape[1]), dtype=np.float64)
    embedding[sample_idx] = eigvecs_sample
    embedding[rest_idx] = eigvecs_rest

    # Row-normalize for standard spectral clustering (normalized cuts).
    # Skip when diffusion scaling was applied — the eigenvalue^t magnitudes
    # carry cluster separation information that normalization would destroy.
    if diffusion_time is None or diffusion_time == 1:
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embedding /= norms

    return embedding


def _apply_fiedler(
    fiedler: np.ndarray,
    lambda2: float,
    indices: np.ndarray,
    min_split_size: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Split indices using the Fiedler vector via sweep cut.

    Sorts points by Fiedler value and sweeps all valid split positions,
    picking the one with minimum conductance (Spielman & Teng 2004).
    This finds a better cut than median or sign-based splitting,
    especially for overlapping clusters where the Fiedler vector is noisy.

    When *min_split_size* > 1 (KEEP mode), restricts the sweep to
    positions [min_split_size, n - min_split_size] to ensure both halves
    meet the size constraint.  The original lambda2 is preserved for
    heap priority and eigengap scoring.
    """
    if not np.isfinite(lambda2):
        return np.inf, indices, np.array([], dtype=np.intp)

    n = len(fiedler)
    lo = max(min_split_size, 1)
    hi = n - lo
    if lo >= hi:
        return np.inf, indices, np.array([], dtype=np.intp)

    order = np.argsort(fiedler)
    sorted_vals = fiedler[order]

    # Check if the Fiedler vector has any variation
    if sorted_vals[-1] - sorted_vals[0] < 1e-10:
        return np.inf, indices, np.array([], dtype=np.intp)

    # Sweep cut: evaluate each valid split position using a balance-weighted
    # gap score.  We want the position with the largest gap that also
    # produces a reasonably balanced partition.
    #
    # Score = gap * min(|left|, |right|) / n
    #
    # This rewards large gaps (cluster boundaries) AND balance (avoids
    # cutting off single points).  Unlike conductance (gap / min_vol),
    # this has no boundary bias — the balance factor in the NUMERATOR
    # penalizes extreme splits rather than rewarding them.
    gaps = np.diff(sorted_vals[lo - 1:hi])
    if len(gaps) == 0:
        return np.inf, indices, np.array([], dtype=np.intp)

    left_sizes = np.arange(lo, hi, dtype=np.float64)
    balance = np.minimum(left_sizes, n - left_sizes) / n
    score = gaps * balance

    best_gap_idx = int(np.argmax(score))
    best = lo + best_gap_idx

    if gaps[best_gap_idx] < 1e-10:
        return np.inf, indices, np.array([], dtype=np.intp)

    return lambda2, indices[order[:best]], indices[order[best:]]


def _sub_seed(base_seed: int | None, indices: np.ndarray) -> int:
    """Derive a deterministic per-split seed from the global seed and sub-cluster."""
    s = base_seed if base_seed is not None else 0
    return s ^ int(indices[0]) ^ (int(indices[-1]) << 16) ^ len(indices)


def _fiedler_split(
    data: np.ndarray,
    indices: np.ndarray,
    k_st: int,
    min_split_size: int = 1,
    seed: int | None = None,
    anisotropic: bool = False,
    use_pic: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Partition a sub-cluster using its Fiedler (2nd smallest) eigenvector.

    Builds affinity + Laplacian + eigsh per split. Use _embedding_split
    when operating on a pre-computed spectral embedding.
    """
    n_sub = len(indices)
    if n_sub <= 2:
        if n_sub <= 1:
            return np.inf, indices, np.array([], dtype=np.intp)
        # The normalized Laplacian of a 2-node graph always gives λ₂=2
        # regardless of edge weight, losing all distance information.
        # Use the self-tuning affinity weight directly: low affinity
        # (distant pair) → small lambda2 → high heap priority (split
        # first).  High affinity (close pair) → large lambda2 → low
        # priority (keep together).
        d_sq = float(np.sum((data[indices[0]] - data[indices[1]]) ** 2))
        if d_sq < 1e-20:
            return np.inf, indices[:1], indices[1:]
        # Self-tuning sigma for a 2-point group: each point's scale is
        # the distance to its only neighbor, so sigma_i = sigma_j = sqrt(d_sq).
        # Affinity w = exp(-d_sq / (sigma_i * sigma_j)) = exp(-1) ≈ 0.368
        # Combinatorial Laplacian λ₂ = 2w for a 2-node graph.
        w = np.exp(-1.0)
        return float(2.0 * w), indices[:1], indices[1:]

    k_st_eff = max(3, min(k_st, n_sub // 5))
    sub_data = data[indices]

    k_connect = min(2 * (k_st_eff + 2), n_sub - 2)
    if anisotropic and n_sub > 2 * k_st_eff:
        A = _anisotropic_affinity_sparse(sub_data, k_st_eff, k_connect)
    elif n_sub > _SELF_TUNING_SPARSE_THRESHOLD:
        A = _self_tuning_affinity_sparse(sub_data, k_st_eff, k_connect)
    else:
        A = _self_tuning_affinity_dense(sub_data, k_st_eff)

    if use_pic:
        # Power iteration clustering: faster than eigsh for large sparse A
        fiedler_vec, lambda2_pic = _power_iteration_fiedler(
            A, seed=_sub_seed(seed, indices)
        )
        return _apply_fiedler(fiedler_vec, lambda2_pic, indices,
                              min_split_size=min_split_size)

    L = csgraph.laplacian(A, normed=True)
    try:
        v0 = np.random.default_rng(_sub_seed(seed, indices)).standard_normal(n_sub)
        eigenvalues, eigenvectors = _sparse_eigsh(
            L, k=2, which="SM", maxiter=2000, tol=1e-6, v0=v0
        )
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    except Exception:
        return np.inf, indices, np.array([], dtype=np.intp)

    return _apply_fiedler(eigenvectors[:, 1], float(eigenvalues[1]), indices,
                          min_split_size=min_split_size)


def _embedding_split(
    embedding: np.ndarray,
    indices: np.ndarray,
    min_split_size: int = 1,
    use_gap_priority: bool = False,
    seed: int | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Split a sub-cluster using PCA on its spectral embedding rows.

    The embedding IS spectral space, so the first principal component of a
    sub-cluster corresponds to its Fiedler vector. O(n_sub * d^2) with no
    affinity construction, no k-NN, no eigsh.
    """
    n_sub = len(indices)
    if n_sub <= 2:
        if n_sub <= 1:
            return np.inf, indices, np.array([], dtype=np.intp)
        d_sq = float(np.sum((embedding[indices[0]] - embedding[indices[1]]) ** 2))
        if d_sq < 1e-20:
            return np.inf, indices[:1], indices[1:]
        # Match _fiedler_split: self-tuning 2-node affinity gives
        # constant lambda2 = 2·exp(-1) ≈ 0.736, a moderate priority.
        return float(2.0 * np.exp(-1.0)), indices[:1], indices[1:]

    sub = embedding[indices]
    mean = sub.mean(axis=0)
    centered = sub - mean

    # First principal component only — use power iteration for large sub-clusters
    # (O(n_sub * d * iters) vs O(n_sub * d²) for full SVD).
    d = centered.shape[1]
    if n_sub * d > 100_000:
        # Power iteration for top singular vector.
        rng = np.random.default_rng(_sub_seed(seed, indices))
        v = rng.standard_normal(d)
        v /= np.linalg.norm(v)
        max_iter = 50  # enough for σ₁/σ₂ ≥ 1.02 to converge
        for _ in range(max_iter):
            u = centered @ v
            u_norm = np.linalg.norm(u)
            if u_norm < 1e-20:
                return np.inf, indices, np.array([], dtype=np.intp)
            u /= u_norm
            v_new = centered.T @ u
            s1 = np.linalg.norm(v_new)
            if s1 < 1e-20:
                return np.inf, indices, np.array([], dtype=np.intp)
            v_new /= s1
            # Convergence check: cosine similarity of consecutive iterates
            if abs(np.dot(v, v_new)) > 1.0 - 1e-10:
                v = v_new
                break
            v = v_new
        fiedler = centered @ v
        pca_variance = float(s1 ** 2 / (n_sub - 1))
    else:
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        fiedler = centered @ Vt[0]
        pca_variance = float(s[0] ** 2 / (n_sub - 1))

    if use_gap_priority:
        # Gap-based priority for diffusion embeddings: a bimodal sub-cluster
        # has a large gap in sorted PC1 values; a unimodal one has small gaps.
        # lambda2 = 1 - relative_gap: clear split → low → high priority.
        # Respect min_split_size: only consider gaps that produce valid partitions.
        sorted_proj = np.sort(fiedler)
        total_range = sorted_proj[-1] - sorted_proj[0]
        if total_range < 1e-20:
            return np.inf, indices, np.array([], dtype=np.intp)
        lo = max(min_split_size, 1)
        hi = n_sub - lo
        if lo >= hi:
            return np.inf, indices, np.array([], dtype=np.intp)
        gaps = np.diff(sorted_proj[lo - 1:hi])
        if len(gaps) == 0:
            return np.inf, indices, np.array([], dtype=np.intp)
        max_gap = float(gaps.max())
        lambda2 = 1.0 - max_gap / total_range
    else:
        # PCA variance for Nyström/standard embeddings — matches the
        # Fiedler eigenvalue semantics used by _fiedler_split.
        lambda2 = pca_variance

    return _apply_fiedler(fiedler, lambda2, indices,
                          min_split_size=min_split_size)


def _lambda2_eigengap_scores(
    lambda2_by_k: dict[int, float],
    n_cluster_lower: int,
    n_cluster_upper: int,
) -> np.ndarray | None:
    """Compute eigengap-analog scores from the λ₂ sequence of divisive splits.

    The score for k is the jump in λ₂ from k to k+1: large jump means the
    next split is much more forced — a natural stopping point just before k+1.
    Score convention: lower is better (negative jump = larger gap preferred).

    Returns None if there is not enough data to compute useful scores.
    """
    n_clusters_range = np.arange(n_cluster_lower, n_cluster_upper + 1)
    lambda2_vals = np.array(
        [lambda2_by_k.get(k, np.nan) for k in n_clusters_range],
        dtype=np.float64,
    )
    # Need at least two finite values to compute differences
    finite_mask = np.isfinite(lambda2_vals)
    if finite_mask.sum() < 2:
        return None

    # Jump from k to k+1: score[i] = lambda2[i+1] - lambda2[i]
    # Pad the last entry with nan (no look-ahead for the last k).
    # Negate so that larger jumps → lower score → better rank in Schulze.
    jumps = np.empty_like(lambda2_vals)
    jumps[:-1] = np.diff(lambda2_vals)
    jumps[-1] = np.nan  # last k has no next split to compare

    return -jumps



def _spectral_divisive_single(data, settings, seed, k_st):
    """Single greedy Fiedler bisection run with a fixed local-scale k_st.

    When n exceeds nystrom_samples, computes a global spectral embedding
    via Nyström once and runs all bisections on the low-dimensional
    embedding.  Otherwise uses exact per-split affinity + eigsh.

    Returns (ClusteringResult, lambda2_by_k) where lambda2_by_k maps k →
    the Fiedler λ₂ used to produce k clusters.
    """
    algo_settings = getattr(settings, settings.algorithm_selection.value)
    n_lower = algo_settings.n_cluster.lower
    n_upper = algo_settings.n_cluster.upper
    threshold = algo_settings.split_lambda2_threshold
    nystrom = getattr(algo_settings, "nystrom_samples", None)
    refine = algo_settings.refinement_enabled
    refine_max_iter = algo_settings.refinement_max_iter

    lbl = ClusteringResult(
        data=data,
        score_settings=algo_settings.scoring,
        seed=seed,
        n_cluster_lower=n_lower,
        min_cluster_size=settings.min_cluster_size,
        small_cluster_mode=settings.small_cluster_mode,
    )
    n = data.shape[0]
    flat_labels = np.zeros(n, dtype=np.intp)
    lambda2_by_k: dict[int, float] = {}

    # Global embedding path — used when:
    #   (a) n exceeds Nyström threshold (approximate eigsh on sample), or
    #   (b) affinity='diffusion' (diffusion map IS a global embedding).
    # All bisections use _embedding_split on the embedding coordinates.
    use_diffusion = algo_settings.affinity == AffinityMatrixOptions.DIFFUSION
    use_anisotropic = algo_settings.affinity == AffinityMatrixOptions.ANISOTROPIC
    use_pic = algo_settings.use_pic
    use_embedding = False
    embedding = None

    if nystrom is not None and n > nystrom:
        n_components = min(2 * n_upper, nystrom - 1, _MAX_EMBEDDING_COMPONENTS)
        diff_t = algo_settings.diffusion_time if use_diffusion else 1
        embedding = _nystrom_embedding(
            data, k_st, nystrom, n_components, diff_t, seed,
        )
        use_embedding = embedding is not None
    elif use_diffusion:
        # Diffusion on all data (n ≤ nystrom threshold).
        # Build self-tuning affinity, compute diffusion map as embedding.
        n_components = min(2 * n_upper, n - 2, _MAX_EMBEDDING_COMPONENTS)
        if n > _SELF_TUNING_SPARSE_THRESHOLD:
            k_connect = min(2 * (k_st + 2), n - 2)
            A = _self_tuning_affinity_sparse(data, k_st, k_connect)
        else:
            A = _self_tuning_affinity_dense(data, k_st)
        embedding, _t_used = _diffusion_map(
            A, algo_settings.diffusion_time, n_components,
            seed=seed, alpha=algo_settings.diffusion_alpha)
        if embedding is not None and embedding.shape[1] > 0:
            # No row-normalization for diffusion maps — the magnitude
            # (eigenvalue^t scaling) carries cluster separation information.
            # Normalizing would project onto a hypersphere, distorting the
            # diffusion geometry.
            use_embedding = True

    # For diffusion Fiedler fallback: truncate embedding to dimensions with
    # meaningful variance.  Diffusion λ^t decay crushes higher components to
    # near-zero noise that would dominate k-NN distances.
    emb_informative = None
    if use_diffusion and use_embedding:
        col_var = np.var(embedding, axis=0)
        informative = col_var > 1e-6 * col_var.max()
        if informative.any():
            emb_informative = embedding[:, informative]
        else:
            emb_informative = embedding[:, :2]  # keep at least 2 dims

    # With KEEP mode, enforce min_cluster_size at the split level —
    # the scoring pipeline won't merge or remove small clusters, so
    # the algorithm must not produce them.  With OUTLIER/ABSORB, the
    # scoring pipeline handles small clusters via merge/penalty, so
    # let the algorithm explore freely.
    enforce_min_split = (
        settings.small_cluster_mode == SmallClusterMode.KEEP
    )
    min_cs = settings.min_cluster_size if enforce_min_split else 1
    tiebreak = 0
    heap: list = []

    def _push(indices: np.ndarray, k: int = 1) -> None:
        nonlocal tiebreak
        if len(indices) < 2 * min_cs:
            return
        # Three split strategies, chosen by context:
        #
        # 1. PCA on embedding (fast, O(n_sub*d)):
        #    Used when a global embedding exists and there are enough
        #    informative dimensions for the current bisection depth.
        #
        # 2. Fiedler on truncated embedding coords (accurate + preserves
        #    diffusion):  Builds a fresh self-tuning affinity in diffusion
        #    space.  Fiedler can find separations that PCA misses once the
        #    embedding's informative dimensions are exhausted.
        #
        # 3. Fiedler on original data (standard per-split):
        #    Used when there's no embedding, or for small Nyström
        #    sub-clusters where local re-estimation is more accurate.
        if use_embedding and (use_diffusion or len(indices) > _FIEDLER_FALLBACK):
            # Strategies 1 or 2: use PCA while the embedding has enough
            # informative directions.  Switch to Fiedler-on-embedding when
            # PCA runs out (current_k exceeds informative dims).
            n_informative = (
                emb_informative.shape[1] if emb_informative is not None
                else embedding.shape[1]
            )
            if k <= n_informative:
                # Strategy 1: PCA on full embedding
                lambda2, left_idx, right_idx = _embedding_split(
                    embedding, indices, min_split_size=min_cs,
                    use_gap_priority=use_diffusion, seed=seed)
            else:
                # Strategy 2: Fiedler on truncated embedding.
                # Multiscale diffusion (Coifman & Maggioni 2006):
                # recompute the diffusion map on the sub-cluster at a
                # reduced t, revealing finer local structure that the
                # global high-t embedding suppressed.
                sub_data = emb_informative[indices]
                if len(sub_data) > 6 and sub_data.shape[1] >= 2:
                    if len(sub_data) > _SELF_TUNING_SPARSE_THRESHOLD:
                        k_c = min(2 * (k_st + 2), len(sub_data) - 2)
                        A_local = _self_tuning_affinity_sparse(
                            sub_data.astype(np.float32), k_st, k_c)
                    else:
                        A_local = _self_tuning_affinity_dense(
                            sub_data.astype(np.float32), k_st)
                    local_emb, _ = _diffusion_map(
                        A_local, diffusion_time=2,
                        n_components=min(10, len(sub_data) - 2),
                        seed=_sub_seed(seed, indices))
                    # Use local embedding for Fiedler split
                    # Create a full-size array indexed by original indices
                    local_full = np.zeros(
                        (emb_informative.shape[0], local_emb.shape[1]),
                        dtype=local_emb.dtype)
                    local_full[indices] = local_emb
                    lambda2, left_idx, right_idx = _fiedler_split(
                        local_full, indices, k_st,
                        min_split_size=min_cs, seed=seed)
                else:
                    lambda2, left_idx, right_idx = _fiedler_split(
                        emb_informative, indices, k_st,
                        min_split_size=min_cs, seed=seed)
        else:
            # Strategy 3: Fiedler on original data
            lambda2, left_idx, right_idx = _fiedler_split(
                data, indices, k_st, min_split_size=min_cs, seed=seed,
                anisotropic=use_anisotropic, use_pic=use_pic)
        if lambda2 == np.inf:
            return
        heapq.heappush(heap, (lambda2, tiebreak, left_idx, right_idx))
        tiebreak += 1

    _push(np.arange(n))

    current_k = 1
    next_label = 1

    # k=1 is a valid candidate when n_lower allows it.
    if n_lower <= 1:
        lbl.add(1, flat_labels.copy())

    while current_k < n_upper:
        if not heap:
            break

        lambda2_node, _, left_idx, right_idx = heapq.heappop(heap)

        if lambda2_node >= threshold:
            break

        flat_labels[right_idx] = next_label
        next_label += 1
        current_k += 1
        lambda2_by_k[current_k] = float(lambda2_node)

        if current_k >= n_lower:
            if refine and current_k >= 2:
                refined = kmedians_refine(
                    data, flat_labels, max_iter=refine_max_iter,
                    min_cluster_size=min_cs, metric=settings._argmin_metric,
                )
                actual_k = len(np.unique(refined[refined >= 0]))
                if actual_k >= n_lower:
                    lbl.add(actual_k, refined)
            else:
                lbl.add(current_k, flat_labels.copy())

        if current_k < n_upper:
            _push(left_idx, current_k)
            _push(right_idx, current_k)

    return lbl, lambda2_by_k


def spectral_divisive(data, settings):
    """Greedy recursive spectral divisive clustering with optional multi-restart."""
    algo_settings = getattr(settings, settings.algorithm_selection.value)
    seed = settings._seed
    recluster_count = algo_settings.recluster_count
    k_st_base = algo_settings.local_scale_neighbors
    eigengap_weight = algo_settings.eigengap_weight

    if recluster_count == 0:
        lbl, lambda2_by_k = _spectral_divisive_single(data, settings, seed, k_st_base)
        if eigengap_weight > 0 and lambda2_by_k:
            scores = _lambda2_eigengap_scores(
                lambda2_by_k,
                algo_settings.n_cluster.lower,
                algo_settings.n_cluster.upper,
            )
            if scores is not None:
                n_lower = algo_settings.n_cluster.lower
                n_upper = algo_settings.n_cluster.upper
                lbl._eigengap_scores = {
                    k: float(scores[i])
                    for i, k in enumerate(range(n_lower, n_upper + 1))
                    if i < len(scores)
                }
                lbl._eigengap_weight = eigengap_weight
        return lbl

    all_by_k: dict[int, list] = {}
    merged_lambda2: dict[int, list[float]] = {}
    for i in range(recluster_count + 1):
        lbl_i, lambda2_i = _spectral_divisive_single(
            data, settings, seed, k_st_base + i * 3,
        )
        for k, lms in lbl_i._labels_store.items():
            all_by_k.setdefault(k, []).extend(lms)
        for k, lv in lambda2_i.items():
            merged_lambda2.setdefault(k, []).append(lv)

    lbl = ClusteringResult(
        data=data,
        score_settings=algo_settings.scoring,
        seed=seed,
        n_cluster_lower=algo_settings.n_cluster.lower,
        min_cluster_size=settings.min_cluster_size,
        small_cluster_mode=settings.small_cluster_mode,
    )
    for k in sorted(all_by_k):
        for lm in all_by_k[k]:
            lbl._add_scored(k, lm)

    if eigengap_weight > 0 and merged_lambda2:
        # Average lambda2 across restarts for each k
        lambda2_by_k = {k: float(np.mean(vs)) for k, vs in merged_lambda2.items()}
        scores = _lambda2_eigengap_scores(
            lambda2_by_k,
            algo_settings.n_cluster.lower,
            algo_settings.n_cluster.upper,
        )
        if scores is not None:
            n_lower = algo_settings.n_cluster.lower
            n_upper = algo_settings.n_cluster.upper
            lbl._eigengap_scores = {
                k: float(scores[i])
                for i, k in enumerate(range(n_lower, n_upper + 1))
                if i < len(scores)
            }
            lbl._eigengap_weight = eigengap_weight

    return lbl
