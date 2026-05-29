#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2025 OpenDSM contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


# ── internal numpy helpers ────────────────────────────────────────────────────

def _build_rank_arrays(
    score_matrix: np.ndarray,
    voter_names: list[str],
    voter_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, list[str], np.ndarray | None, np.ndarray | None]:
    """Prepare rank, weight, and normalized-score arrays from a raw score matrix.

    Parameters
    ----------
    score_matrix : (n_cand, n_voters) float64
        Raw scores; nan = abstain, inf = active worst, -inf = active best.
    voter_names : list[str]
    voter_weights : dict[str, float] | None
        Per-voter weights.  None → all 1.0.

    Returns
    -------
    rank_of_candidate : (n_cand, m) int | None
        rank_of_candidate[c, v] = rank of candidate c for kept voter v (0=best).
    weights : (m,) float64 | None
        Normalised weights for the kept voters.
    kept_names : list[str]
        Voter names that survived filtering.
    abstain_bool : (n_cand, m) bool | None
        True where the original score was nan.
    normalized_scores : (n_cand, m) float64 | None
        Per-voter scores min-max normalized to [0, 1].  inf/nan mapped to 1.0.
    """
    n_cand, n_v = score_matrix.shape

    if voter_weights is None:
        raw_w = np.ones(n_v, dtype=np.float64)
    else:
        raw_w = np.array([voter_weights.get(v, 0.0) for v in voter_names], dtype=np.float64)

    abstain_bool = np.isnan(score_matrix)
    filled = np.where(abstain_bool, np.inf, score_matrix)

    # Drop zero-weight and all-inf columns (no signal)
    keep = (raw_w > 0) & ~np.all(filled == np.inf, axis=0)
    if not keep.any():
        return None, None, [], None, None

    filled       = filled[:, keep]
    abstain_bool = abstain_bool[:, keep]
    raw_w        = raw_w[keep]
    kept_names   = [v for v, k in zip(voter_names, keep) if k]

    # Normalise weights to sum to the number of active voters (matches old code)
    m       = raw_w.size
    total_w = raw_w.sum()
    weights = raw_w * (m / total_w) if (total_w > 0 and total_w != m) else raw_w

    # rank_of_candidate[c, v] = dense rank of candidate c for voter v (0 = best)
    rank_of_candidate = np.empty((n_cand, m), dtype=np.intp)
    for v in range(m):
        col = filled[:, v]
        unique_vals = np.unique(col)
        rank_of_candidate[:, v] = np.searchsorted(unique_vals, col)

    # Normalize scores to [0, 1] per voter using MAD-clipped min-max.
    # Preserves score-gap magnitudes (unlike pure rank) while being robust
    # to extreme outliers (unlike pure min-max).  Finite scores beyond
    # median ± 3*MAD are clipped before scaling, preventing a single
    # extreme value from compressing the useful range.  Works at any
    # candidate count (MAD is robust even with 3-6 values).
    # Score semantics: -inf = active best → 0.0
    #                  +inf = active worst → 1.0
    #                  nan  = abstain (handled separately by abstain_bool)
    normalized = np.full_like(filled, 0.5)
    for v in range(m):
        col = filled[:, v]
        finite_mask = np.isfinite(col)
        n_finite = int(finite_mask.sum())
        if n_finite < 2:
            normalized[:, v] = np.where(col == -np.inf, 0.0,
                               np.where(col == np.inf, 1.0, 0.5))
            continue
        finite_vals = col[finite_mask]
        med = np.median(finite_vals)
        mad = np.median(np.abs(finite_vals - med))
        if mad > 0:
            lo = med - 3 * mad
            hi = med + 3 * mad
        else:
            lo, hi = finite_vals.min(), finite_vals.max()
        spread = hi - lo
        if spread > 0:
            clipped = np.clip(col, lo, hi)
            normalized[:, v] = np.where(
                finite_mask, (clipped - lo) / spread,
                np.where(col == -np.inf, 0.0, 1.0),
            )
        else:
            normalized[:, v] = np.where(
                finite_mask, 0.5,
                np.where(col == -np.inf, 0.0, 1.0),
            )

    return rank_of_candidate, weights, kept_names, abstain_bool, normalized


def _schulze_pairwise_preference(
    rank_of_candidate: np.ndarray,
    weights: np.ndarray,
    abstain_bool: np.ndarray,
    normalized_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorised pairwise preference matrix P[i, j] = weighted votes for i over j.

    When *normalized_scores* is provided (score-magnitude-aware mode), the
    preference contribution from each voter is proportional to the score
    gap between candidates, not just the ordinal direction.  This lets a
    voter with a strong preference (large score gap) contribute more than
    a voter with a marginal preference (tiny gap).

    When *normalized_scores* is None, falls back to rank-based voting
    where ties contribute 0.5 to each side.

    Parameters
    ----------
    rank_of_candidate : (n_cand, m) int
    weights           : (m,) float64
    abstain_bool      : (n_cand, m) bool
    normalized_scores : (n_cand, m) float64 or None
        Per-voter scores normalized to [0, 1] range.  Lower is better.

    Returns
    -------
    P : (n_cand, n_cand) float64
    """
    ab = abstain_bool[:, None, :] | abstain_bool[None, :, :]  # (n_cand, n_cand, m)

    if normalized_scores is not None:
        # Score-magnitude-aware: P[i,j,v] = max(0, norm_score_j - norm_score_i)
        # i.e. how much better i is than j on voter v, scaled by score gap.
        s_i = normalized_scores[:, None, :]          # (n_cand, 1, m)
        s_j = normalized_scores[None, :, :]          # (1, n_cand, m)
        margin = np.maximum(s_j - s_i, 0.0)         # positive when i is better (lower score)
        margin = np.where(ab, 0.0, margin)
        P = margin @ weights                         # (n_cand, n_cand)
    else:
        r_i = rank_of_candidate[:, None, :]          # (n_cand, 1, m)
        r_j = rank_of_candidate[None, :, :]          # (1, n_cand, m)
        i_beats = ((r_i < r_j) & ~ab).astype(np.float64)
        i_ties  = ((r_i == r_j) & ~ab).astype(np.float64)
        P = (i_beats + 0.5 * i_ties) @ weights      # (n_cand, n_cand)

    np.fill_diagonal(P, 0.0)
    return P


def _schulze_path_strength(P: np.ndarray) -> np.ndarray:
    """Floyd-Warshall strongest-path propagation (in-place).

    P[i, j] is updated to the strength of the strongest path from i to j.
    """
    n = P.shape[0]
    for k in range(n):
        via_k = np.minimum(P[:, k : k + 1], P[k : k + 1, :])
        np.maximum(P, via_k, out=P)
    return P


def _schulze_rank_strength(P: np.ndarray) -> np.ndarray:
    """Margin-weighted win accumulation.

    Each pairwise victory of i over j contributes ``P[i,j] - P[j,i]`` to
    candidate i's score.  This diverges from standard binary Schulze (+1 per
    win) in favour of better tie resolution — acceptable here because
    candidates are ordered k-values, not clones.
    """
    return np.sum(np.maximum(P - P.T, 0), axis=1)


# ── public API ────────────────────────────────────────────────────────────────

def build_rank_matrix(
    proxies: list,
) -> tuple[np.ndarray, list[str], dict[str, set[int]]]:
    """Extract raw score matrix and abstain mask from score proxies.

    Parameters
    ----------
    proxies : list of SimpleNamespace with a ``.score`` dict

    Returns
    -------
    score_matrix : np.ndarray (n_cand, n_voters) float64
        Raw scores; nan = abstain, inf = active worst.
    voter_names : list[str]
    abstain_mask : dict[str, set[int]]
        voter → set of candidate indices that voter abstains on.
    """
    if not proxies:
        return np.empty((0, 0), dtype=np.float64), [], {}

    voter_names = list(proxies[0].score.keys())
    score_matrix = np.array(
        [[proxy.score.get(v, np.inf) for v in voter_names] for proxy in proxies],
        dtype=np.float64,
    )
    abstain_mask = {
        v: {int(i) for i in np.where(np.isnan(score_matrix[:, vi]))[0]}
        for vi, v in enumerate(voter_names)
    }
    return score_matrix, voter_names, abstain_mask


def schulze_voting(
    score_matrix: np.ndarray,
    voter_names: list[str],
    voter_weights: dict[str, float] | None = None,
    window_size: float = 0,
    return_preference_df: bool = False,
    abstain_mask: dict | None = None,  # unused internally; kept for API compat
    candidate_k_values: list[int] | None = None,
    k_penalty_strength: float = 1.0,
    k_penalty_rate: float = 1.0,
    occam_confidence_floor: float = 0.0,
) -> int | tuple:
    """Schulze voting over a raw score matrix.

    Works entirely in numpy.  A preference DataFrame is constructed only when
    ``return_preference_df=True``.

    Parameters
    ----------
    score_matrix : (n_cand, n_voters) float64
        From :func:`build_rank_matrix` (or with extra columns injected).
    voter_names : list[str]
        Column labels for *score_matrix*.
    voter_weights : dict[str, float] | None
        Per-voter weights.  None → all 1.0.  Voters absent from the dict
        receive weight 0.0 (silenced).
    window_size : float
        Gaussian smoothing sigma applied to win scores before argmax.  0 = off.
    return_preference_df : bool
        When True also return a summary DataFrame (slow path; for debugging).

    Returns
    -------
    winner_idx : int
        Index into the *n_cand* axis of the winning candidate.
    confidence : float
        Winner's share of total win scores, in [0, 1].  0 = tie among
        all candidates; 1 = winner holds all preference strength.
    pref_df : pd.DataFrame
        Only present when *return_preference_df* is True.
    """
    n_cand = score_matrix.shape[0]
    if n_cand == 0:
        raise ValueError("No candidates.")

    rank_of_candidate, weights, kept_names, abstain_bool, normalized = _build_rank_arrays(
        score_matrix, voter_names, voter_weights
    )

    # Complexity penalty on normalized scores.
    # Lower k gets penalized: its normalized scores (0=best, 1=worst) are
    # shifted toward 1, making it harder for low-k candidates to win.
    # f(k) = 1 - strength / k^rate: asymptotically approaches 1 as k grows.
    # penalized = 1 - f(k) * (1 - normalized)
    if (candidate_k_values is not None and normalized is not None
            and k_penalty_strength > 0):
        k_arr = np.array(candidate_k_values[:n_cand], dtype=np.float64)
        penalty = k_penalty_strength / np.maximum(k_arr, 1.0) ** k_penalty_rate
        f_k = 1.0 - np.clip(penalty, 0.0, 1.0)
        for i in range(n_cand):
            if f_k[i] < 1.0:
                normalized[i, :] = 1.0 - f_k[i] * (1.0 - normalized[i, :])

    if rank_of_candidate is None:
        candidate_wins = np.zeros(n_cand)
    else:
        P = _schulze_pairwise_preference(
            rank_of_candidate, weights, abstain_bool,
            normalized_scores=normalized,
        )
        P = _schulze_path_strength(P)
        candidate_wins = _schulze_rank_strength(P)

    if window_size > 0:
        candidate_wins = gaussian_filter1d(
            candidate_wins, sigma=window_size, mode="nearest", cval=0.0
        )

    winner_idx = int(np.argmax(candidate_wins))

    # Confidence: margin between the winner's total win score and the
    # best candidate from a *different* k value, normalized to [0, 1].
    #
    # Uses total win scores (which aggregate all pairwise comparisons)
    # rather than the head-to-head path strength between two specific
    # candidates.  This avoids two issues:
    # 1. Multiple restarts at the same k produce clones that tie
    #    head-to-head, giving 0 confidence even when the winner clearly
    #    dominates other k values.
    # 2. The Schulze winner can lose head-to-head to the runner-up
    #    while still winning overall (Condorcet paradox), making
    #    pairwise confidence negative.
    if n_cand > 1 and rank_of_candidate is not None:
        winner_k = (candidate_k_values[winner_idx]
                    if candidate_k_values is not None else None)
        ranked = np.argsort(candidate_wins)[::-1]
        runner_up_idx = None
        for r_idx in ranked:
            if r_idx == winner_idx:
                continue
            r_k = (candidate_k_values[r_idx]
                   if candidate_k_values is not None else None)
            if winner_k is None or r_k is None or r_k != winner_k:
                runner_up_idx = int(r_idx)
                break
        if runner_up_idx is None:
            runner_up_idx = int(ranked[1]) if len(ranked) > 1 else winner_idx
        w_score = candidate_wins[winner_idx]
        r_score = candidate_wins[runner_up_idx]
        denom = w_score + r_score
        confidence = float((w_score - r_score) / denom) if denom > 0 else 0.0
    else:
        confidence = 1.0 if n_cand <= 1 else 0.0

    # Occam tiebreaker for close votes.  When the Schulze winner's pairwise
    # confidence against the runner-up falls below ``occam_confidence_floor``
    # (calibrated at 0.15 ~ 5th percentile of confidence over 72 representative
    # spectral scenarios), aggregate scores are statistically indistinguishable
    # from a random k selection.  In that regime we widen the "tied" set to
    # every candidate whose pairwise confidence against the current winner is
    # also below the floor, and break the tie by preferring the smallest k
    # (parsimony / BIC-MDL logic).  Default 0.0 disables the tiebreaker; opt
    # in per algorithm.
    if (
        occam_confidence_floor > 0.0
        and candidate_k_values is not None
        and n_cand > 1
        and confidence < occam_confidence_floor
        and candidate_wins[winner_idx] > 0
    ):
        w_score = candidate_wins[winner_idx]
        tied_idxs = [winner_idx]
        for i in range(n_cand):
            if i == winner_idx:
                continue
            c_score = candidate_wins[i]
            denom_i = w_score + c_score
            pair_conf = (w_score - c_score) / denom_i if denom_i > 0 else 1.0
            if pair_conf < occam_confidence_floor:
                tied_idxs.append(i)
        if len(tied_idxs) > 1:
            winner_idx = min(tied_idxs, key=lambda i: candidate_k_values[i])

    if not return_preference_df:
        return winner_idx, confidence

    # ── preference DataFrame — only built when explicitly requested ───────────

    filled = np.where(np.isnan(score_matrix), np.inf, score_matrix)
    rank_order = np.argsort(filled, axis=0)  # (n_cand, n_voters), value = cand at rank r
    df_pref = pd.DataFrame(rank_order, columns=voter_names)
    df_pref["wins"] = candidate_wins.astype(int)
    df_pref.index.name = "candidate"
    return winner_idx, confidence, df_pref
