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

"""Normalized difference-in-differences (NDID) comparison-group correction kernel."""

from __future__ import annotations

import numpy as np

from opendsm.common.stats.distribution_transform.yeo_johnson import yj_transform



_MAD_SCALE = 1.4826


def _cauchy(u):
    return 1.0 / (1.0 + u**2)


def _weighted_median(values, weights):
    """Row-wise weighted median of (T, n) values; NaN on rows with no positive weight.

    The median is the first ascending value whose cumulative weight reaches half the
    row total (less a 1e-12 relative tolerance). Zero-weight entries never qualify.
    """
    values = np.where(weights > 0, values, np.inf)
    order = np.argsort(values, axis=1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=1)
    cum_weight = np.cumsum(np.take_along_axis(weights, order, axis=1), axis=1)
    total = cum_weight[:, -1]

    threshold = 0.5 * total - 1e-12 * total
    idx = np.argmax(cum_weight >= threshold[:, None], axis=1)
    median = sorted_values[np.arange(len(values)), idx]

    return np.where(total > 0, median, np.nan)


def _robust_z(values, weights):
    """Weighted-median/MAD standardization; 0 everywhere on rows whose MAD is 0."""
    med = _weighted_median(values, weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        mad = _MAD_SCALE * _weighted_median(np.abs(values - med[:, None]), weights)
        scaled = (values - med[:, None]) / mad[:, None]

    return np.where((weights > 0) & (mad > 0)[:, None], scaled, 0.0)


def _trust(e, w0, lam, trust_constant):
    """Cross-sectional trust of each meter at each timestep (one pass, no iteration)."""
    u = _robust_z(e, w0)
    y = yj_transform(np.ascontiguousarray(u).ravel(), float(lam)).reshape(u.shape)
    z = _robust_z(y, w0)
    with np.errstate(divide="ignore"):
        trust = np.minimum(1.0, trust_constant / np.abs(z))

    return np.where(w0 > 0, trust, 0.0)


def _validate(mTr, mTr_unc, s_Tr, q, oCGr, mCGr, s, S, cg_label, relevance, t_weight, laws):
    T = len(mTr)
    for name, arr in (("mTr_unc", mTr_unc), ("s_Tr", s_Tr), ("q", q)):
        if arr.shape != (T,):
            raise ValueError(f"'{name}' must have shape ({T},) to match 'mTr'; got {arr.shape}.")

    if oCGr.ndim != 2 or oCGr.shape[0] != T:
        raise ValueError(f"'oCGr' must have shape ({T}, M); got {oCGr.shape}.")

    M = oCGr.shape[1]
    for name, arr in (("mCGr", mCGr), ("s", s)):
        if arr.shape != (T, M):
            raise ValueError(f"'{name}' must have shape ({T}, {M}); got {arr.shape}.")

    for name, arr in (("S", S), ("cg_label", cg_label), ("relevance", relevance)):
        if arr.shape != (M,):
            raise ValueError(f"'{name}' must have shape ({M},); got {arr.shape}.")

    if not np.issubdtype(q.dtype, np.integer):
        raise ValueError(f"'q' must hold integer cell indices; got dtype {q.dtype}.")

    n_cells = laws.v.shape[1]
    if T and (q.min() < 0 or q.max() >= n_cells):
        raise ValueError(f"'q' must index the {n_cells} cells of the laws.")

    n_clusters = len(np.unique(cg_label[cg_label >= 0]))
    if t_weight.shape != (n_clusters,):
        raise ValueError(
            f"'t_weight' must have one entry per non-negative cluster label ({n_clusters}); "
            f"got shape {t_weight.shape}."
        )

    if laws.v.shape[0] != n_clusters:
        raise ValueError(
            f"'laws' carry {laws.v.shape[0]} clusters but 'cg_label' has {n_clusters}."
        )


def ndid_correction_matrix(
    mTr,
    mTr_unc,
    s_Tr,
    S_Tr,
    oCGr,
    mCGr,
    s,
    S,
    q,
    cg_label,
    t_weight,
    laws,
    settings,
    relevance=None,
    trust_constant=3.0,
    return_trust=False,
):
    """NDID correction of one treatment meter against a clustered comparison pool.

    Each pool meter's reporting residual is normalized by its typical load, and the
    treatment's correction at a timestep is a ``t_weight``-combination of per-cluster
    weighted means of those normalized residuals, scaled back by the treatment's typical
    load. Within a cluster, meters are weighted by how closely their predicted relative
    load matches the treatment's (Cauchy state kernel), by size relevance and the size
    law, by ``relevance``, and by a robust cross-sectional trust that discounts extreme
    residuals after a Yeo-Johnson transform at the cluster's pooled lambda.

    All arrays must already be aligned: rows are the same reporting timesteps, pool
    columns are the same admitted meters in pool order, and ``t_weight`` is already
    re-indexed to the clusters of ``laws`` (the sorted distinct non-negative labels of
    the admitted meters). Inputs are cast to float64.

    Parameters
    ----------
    mTr, mTr_unc : array-like, shape (T,)
        Treatment model prediction and its reported band, in treatment units.
    s_Tr : array-like, shape (T,)
        Treatment typical load at each timestep, in treatment units.
    S_Tr : float
        Treatment annual scale.
    oCGr, mCGr, s : array-like, shape (T, M)
        Pool observed, pool predicted and pool typical load, in each meter's units.
    S : array-like, shape (M,)
        Pool annual scales.
    q : array-like of int, shape (T,)
        Profile cell index of each timestep.
    cg_label : array-like of int, shape (M,)
        Cluster label per pool meter; negative labels are excluded.
    t_weight : array-like, shape (n_clusters,)
        Treatment cluster weights in sorted non-negative label order; a cluster with
        non-positive weight never contributes.
    laws : PooledLaws
        Frozen pool laws of the run; supplies v, spread, lam, acf, gamma_used and S_ref.
    settings : NDIDSettings
        Supplies `state_bandwidth_factor` and `size_bandwidth`.
    relevance : array-like, shape (M,), optional
        Non-negative per-meter weight multipliers; None means ones.
    trust_constant : float
        Robust-z threshold beyond which trust decays as c / |z|; ``inf`` disables trust.
    return_trust : bool
        Also return the (T, M) trust array.

    Returns
    -------
    mTrc : ndarray, shape (T,)
        Corrected treatment prediction, in treatment units.
    corrected_unc : ndarray, shape (T,)
        sqrt(mTr_unc**2 + cg_var), in treatment units.
    cg_var : ndarray, shape (T,)
        One-sigma sampling variance of the correction, in treatment units squared.
    cg_effective_n : ndarray, shape (T,)
        Cluster-weighted Kish effective number of pool meters.
    cg_acf : ndarray, shape (K_acf,)
        ``t_weight``-weighted pooled residual autocorrelation at lags 1..K_acf.
    mask : ndarray of bool, shape (T, M)
        Pool meter available at t with positive weight in a contributing cluster.
    trust : ndarray, shape (T, M)
        Only when ``return_trust``; 0 on entries outside ``mask``.

    Raises
    ------
    ValueError
        If shapes disagree, ``q`` is not an in-range integer index, or the number of
        clusters in ``cg_label``, ``t_weight`` and ``laws`` differ.
    """
    mTr = np.asarray(mTr, dtype=np.float64)
    mTr_unc = np.asarray(mTr_unc, dtype=np.float64)
    s_Tr = np.asarray(s_Tr, dtype=np.float64)
    oCGr = np.asarray(oCGr, dtype=np.float64)
    mCGr = np.asarray(mCGr, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    q = np.asarray(q)
    cg_label = np.asarray(cg_label)
    t_weight = np.asarray(t_weight, dtype=np.float64)
    if relevance is None:
        relevance = np.ones(oCGr.shape[1:])
    relevance = np.asarray(relevance, dtype=np.float64)

    _validate(mTr, mTr_unc, s_Tr, q, oCGr, mCGr, s, S, cg_label, relevance, t_weight, laws)

    T, M = oCGr.shape
    row_finite = np.isfinite(mTr) & np.isfinite(s_Tr)
    active = row_finite & (s_Tr > 0)

    available = np.isfinite(mCGr) & np.isfinite(oCGr) & np.isfinite(s) & (s > 0)
    available &= active[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(available, (mCGr - oCGr) / s, 0.0)
        x = np.where(available, mCGr / s, 0.0)
        x_Tr = np.where(active, mTr / s_Tr, 0.0)

    static = (S / laws.S_ref) ** (2.0 * laws.gamma_used) * relevance
    if settings.size_bandwidth is not None:
        static = static * _cauchy((np.log(S) - np.log(S_Tr)) / settings.size_bandwidth)

    clusters = np.unique(cg_label[cg_label >= 0])
    n_clusters = len(clusters)
    contributes = np.zeros((n_clusters, T), dtype=bool)
    e_hat = np.zeros((n_clusters, T))
    var = np.zeros((n_clusters, T))
    n_eff = np.zeros((n_clusters, T))
    mask = np.zeros((T, M), dtype=bool)
    trust = np.zeros((T, M))

    for k, label in enumerate(clusters):
        if t_weight[k] <= 0:
            continue

        cols = cg_label == label
        e_k = e[:, cols]

        h = settings.state_bandwidth_factor * laws.spread[k, q][:, None]
        d = x[:, cols] - x_Tr[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            state = np.where(h > 0, _cauchy(d / h), (d == 0).astype(float))

        w0 = np.where(available[:, cols], state * static[cols], 0.0)
        ok = w0.sum(axis=1) > 0
        w0[~ok] = 0.0

        trust_k = _trust(e_k, w0, laws.lam[k], trust_constant)
        w = w0 * trust_k
        W = w.sum(axis=1)
        W_safe = np.where(ok, W, 1.0)

        mean = (w * e_k).sum(axis=1) / W_safe
        n = np.where(ok, W**2 / np.where(ok, (w**2).sum(axis=1), 1.0), 0.0)
        se2 = (
            (w**2 * (e_k - mean[:, None]) ** 2).sum(axis=1) / W_safe**2
            * n / np.maximum(n - 1.0, 1e-9)
        )
        blend = np.clip(n - 1.0, 0.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            var_k = blend * se2 + (1.0 - blend) * laws.v[k, q] / n

        contributes[k] = ok
        e_hat[k] = np.where(ok, mean, 0.0)
        var[k] = np.where(ok, var_k, 0.0)
        n_eff[k] = n
        mask[:, cols] = w > 0
        trust[:, cols] = trust_k

    cluster_weight = t_weight[:, None] * contributes
    weight_total = cluster_weight.sum(axis=0)
    corrected = weight_total > 0
    T_k = cluster_weight / np.where(corrected, weight_total, 1.0)

    mTrc = np.full(T, np.nan)
    corrected_unc = np.full(T, np.nan)
    cg_var = np.full(T, np.nan)
    cg_effective_n = np.zeros(T)

    no_state = row_finite & ~active
    mTrc[no_state] = mTr[no_state]
    cg_var[no_state] = 0.0
    corrected_unc[no_state] = mTr_unc[no_state]

    c = corrected
    mTrc[c] = mTr[c] - s_Tr[c] * (T_k[:, c] * e_hat[:, c]).sum(axis=0)
    cg_var[c] = s_Tr[c] ** 2 * (T_k[:, c] ** 2 * var[:, c]).sum(axis=0)
    corrected_unc[c] = np.sqrt(mTr_unc[c] ** 2 + cg_var[c])
    cg_effective_n[c] = (T_k[:, c] * n_eff[:, c]).sum(axis=0)

    cg_acf = t_weight @ laws.acf

    if return_trust:
        return mTrc, corrected_unc, cg_var, cg_effective_n, cg_acf, mask, trust

    return mTrc, corrected_unc, cg_var, cg_effective_n, cg_acf, mask
