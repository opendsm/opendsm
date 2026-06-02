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

"""EM-based regime discovery for the DailyAdaptivePSplineModel.

Discovers K temperature-energy regimes by iteratively fitting PSplines
(M-step) and reassigning days to the best-fitting curve (E-step).
Selects K via BIC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from opendsm.eemeter.models.daily_pspline.fitting import fit_segment
from opendsm.eemeter.models.daily_pspline.spline import PSpline

from .classifier import RegimeClassifier, build_features


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

@dataclass
class RegimeResult:
    """Output of regime discovery for one value of K."""

    k: int
    splines: dict[int, PSpline] = field(default_factory=dict)
    assignments: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    classifier: RegimeClassifier = field(default_factory=RegimeClassifier)
    bic: float = np.inf
    converged: bool = False


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def discover_regimes(
    df: pd.DataFrame,
    settings,
) -> RegimeResult:
    """Discover optimal regime structure via EM + BIC selection.

    Evaluates K = 1 .. ``settings.k_max`` regimes, runs the EM loop
    for each, trains an assignment classifier, and selects the K with
    the lowest BIC.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with DatetimeIndex, 'temperature', 'observed' columns.
    settings : DailyAdaptivePSplineSettings
        Full model settings.

    Returns
    -------
    RegimeResult
        Best regime structure (splines, assignments, classifier, BIC).
    """
    temp = df["temperature"].values
    obs = df["observed"].values
    N = len(df)

    # Build classifier features once (shared across all K)
    X = build_features(df, settings)

    # Lightweight PSpline settings for EM exploration
    em_pspline = settings.pspline.model_copy(update={
        "adaptive_iterations": 1,
        "max_weight_iterations": 0,
    })

    # ------------------------------------------------------------------
    # Phase 1: Fit K=1 baseline and compute per-group residual signatures
    # ------------------------------------------------------------------
    baseline_assign = np.zeros(N, dtype=int)
    baseline_splines = _fit_regimes(df, baseline_assign, settings, em_pspline)
    if baseline_splines is None:
        return RegimeResult(k=1)

    baseline_pred = baseline_splines[0].predict(temp)
    residuals = obs - baseline_pred

    # Compute residual signatures per month and per day-of-week.
    # These are cheap 1D clustering inputs that tell us which
    # calendar groups behave differently.
    months = df.index.month.values
    dows = df.index.dayofweek.values

    month_resid = np.array([
        np.median(residuals[months == m]) if np.sum(months == m) >= 5 else 0.0
        for m in range(1, 13)
    ])
    dow_resid = np.array([
        np.median(residuals[dows == d]) if np.sum(dows == d) >= 5 else 0.0
        for d in range(7)
    ])

    # ------------------------------------------------------------------
    # Phase 2: Generate informed initializations from residual signatures
    # ------------------------------------------------------------------
    best = RegimeResult(k=1)

    for K in range(1, settings.k_max + 1):
        inits = _informed_initializations(
            df, K, settings, month_resid, dow_resid, residuals,
        )

        for init_assign in inits:
            splines, assignments, converged = _em_loop(
                df, init_assign, K, settings, pspline_settings=em_pspline,
            )

            if splines is None:
                continue

            # Enforce prediction-time feasibility
            classifier = RegimeClassifier()
            classifier.fit(X, assignments, settings)
            assignments = classifier.predict(X)

            # Refit with full PSpline settings
            splines = _fit_regimes(df, assignments, settings)
            if splines is None:
                continue

            bic = _compute_bic(
                splines, assignments, N,
                settings.bic_penalty_multiplier, settings.bic_penalty_power,
            )

            result = RegimeResult(
                k=K,
                splines=splines,
                assignments=assignments,
                classifier=classifier,
                bic=bic,
                converged=converged,
            )

            if bic < best.bic:
                best = result

    return best


# ------------------------------------------------------------------
# Initialization from residual signatures
# ------------------------------------------------------------------

def _informed_initializations(
    df: pd.DataFrame,
    K: int,
    settings,
    month_resid: np.ndarray,
    dow_resid: np.ndarray,
    residuals: np.ndarray,
) -> list[np.ndarray]:
    """Generate EM initializations informed by K=1 residual signatures.

    Instead of blind temperature/doy quantiles, clusters the monthly
    and day-of-week residual medians to find which calendar groups
    behave differently.  This is a 1D clustering on 12 (or 7) values
    — essentially free — and produces initializations that align with
    the actual regime structure.
    """
    N = len(df)

    if K == 1:
        return [np.zeros(N, dtype=int)]

    months = df.index.month.values
    dows = df.index.dayofweek.values
    inits = []

    # 1. Monthly residual clustering — finds seasonal regimes
    month_init = _cluster_residual_groups(
        month_resid, K, months, N, range_vals=range(1, 13),
        min_group=settings.min_regime_days,
    )
    if month_init is not None:
        inits.append(month_init)

    # 2. Day-of-week residual clustering — finds behavioral regimes
    if K == 2:
        dow_init = _cluster_residual_groups(
            dow_resid, K, dows, N, range_vals=range(7),
            min_group=settings.min_regime_days,
        )
        if dow_init is not None:
            inits.append(dow_init)

    # 3. Combined month × dow: cluster on per-(month, dow) residuals
    #    for K >= 3 where both seasonal and behavioral splits coexist
    if K >= 3 and len(inits) == 0:
        # Fallback: use month clustering with more groups
        month_init = _cluster_residual_groups(
            month_resid, K, months, N, range_vals=range(1, 13),
            min_group=settings.min_regime_days,
        )
        if month_init is not None:
            inits.append(month_init)

    # Fallback: if no informed init produced valid assignments,
    # use temperature quantiles
    if not inits:
        inits.append(_init_temp_quantile(df["temperature"].values, K))

    return inits


def _cluster_residual_groups(
    group_resids: np.ndarray,
    K: int,
    group_labels: np.ndarray,
    N: int,
    range_vals,
    min_group: int,
) -> np.ndarray | None:
    """Cluster calendar groups (months or dows) by their residual medians.

    Uses 1D K-means on the residual medians to find which groups
    belong together.  Maps the group-level clustering back to
    per-day assignments.

    Returns None if any resulting regime has fewer than min_group days.
    """
    vals = list(range_vals)
    n_vals = len(vals)
    if K >= n_vals:
        return None

    # 1D K-means on residual medians: sort and find optimal split points
    order = np.argsort(group_resids)
    sorted_resids = group_resids[order]

    # Find K-1 split points that maximize between-group variance
    # For small n_vals (7 or 12), brute-force all combinations
    from itertools import combinations
    best_splits = None
    best_variance = -np.inf

    split_candidates = range(1, n_vals)
    for splits in combinations(split_candidates, K - 1):
        splits = (0,) + splits + (n_vals,)
        # Between-group variance
        group_means = []
        group_sizes = []
        for i in range(len(splits) - 1):
            segment = sorted_resids[splits[i]:splits[i + 1]]
            if len(segment) == 0:
                break
            group_means.append(np.mean(segment))
            group_sizes.append(len(segment))
        else:
            grand_mean = np.mean(sorted_resids)
            bgv = sum(s * (m - grand_mean) ** 2 for m, s in zip(group_means, group_sizes))
            if bgv > best_variance:
                best_variance = bgv
                best_splits = splits

    if best_splits is None:
        return None

    # Map sorted groups back to original group indices
    group_to_regime = np.zeros(n_vals, dtype=int)
    for regime_id in range(K):
        for idx in range(best_splits[regime_id], best_splits[regime_id + 1]):
            group_to_regime[order[idx]] = regime_id

    # Map group-level regimes to per-day assignments
    assignments = np.zeros(N, dtype=int)
    for i, val in enumerate(vals):
        mask = group_labels == val
        assignments[mask] = group_to_regime[i]

    # Check minimum regime sizes
    for k in range(K):
        if np.sum(assignments == k) < min_group:
            return None

    return assignments


def _init_temp_quantile(temp: np.ndarray, K: int) -> np.ndarray:
    """Fallback: split by temperature quantiles."""
    quantiles = np.linspace(0, 100, K + 1)
    edges = np.percentile(temp, quantiles)
    assignments = np.zeros(len(temp), dtype=int)
    for i in range(K):
        if i == K - 1:
            mask = temp >= edges[i]
        else:
            mask = (temp >= edges[i]) & (temp < edges[i + 1])
        assignments[mask] = i
    return assignments


# ------------------------------------------------------------------
# EM loop
# ------------------------------------------------------------------

def _em_loop(
    df: pd.DataFrame,
    assignments: np.ndarray,
    K: int,
    settings,
    pspline_settings=None,
) -> tuple[dict[int, PSpline] | None, np.ndarray, bool]:
    """Run EM: alternate between fitting PSplines and reassigning days.

    Parameters
    ----------
    pspline_settings : optional
        Lightweight PSpline settings for EM iterations. If None, uses
        ``settings.pspline``.

    Returns (splines, final_assignments, converged).
    Returns (None, assignments, False) if any regime becomes too small.
    """
    temp = df["temperature"].values
    obs = df["observed"].values
    N = len(df)

    prior_wrmse = np.inf
    converged = False
    splines: dict[int, PSpline] = {}
    prev_assignments = np.full(N, -1, dtype=int)

    for iteration in range(settings.em_max_iter):
        # M-step: only refit regimes whose assignments changed
        changed_regimes = set(range(K)) if iteration == 0 else {
            k for k in range(K)
            if not np.array_equal(
                np.where(prev_assignments == k)[0],
                np.where(assignments == k)[0],
            )
        }

        for k in changed_regimes:
            mask = assignments == k
            if np.sum(mask) < settings.min_regime_days:
                return None, assignments, False
            x_k = temp[mask]
            y_k = obs[mask]
            sort_idx = np.argsort(x_k)
            ps = pspline_settings if pspline_settings is not None else settings.pspline
            splines[k] = fit_segment(x_k[sort_idx], y_k[sort_idx], ps)

        # E-step: reassign each day to curve with smallest absolute residual.
        # Absolute (L1) distance is robust to outliers — squared residuals
        # let a single outlier dominate a regime's assignment score,
        # collapsing subtle but real shifts.
        residuals = np.full((N, K), np.inf)
        for k, spl in splines.items():
            residuals[:, k] = np.abs(obs - spl.predict(temp))

        new_assignments = np.argmin(residuals, axis=1)

        # Check for degenerate regimes (too few days)
        for k in range(K):
            if np.sum(new_assignments == k) < settings.min_regime_days:
                new_assignments = _merge_small_regime(
                    new_assignments, k, residuals, settings.min_regime_days,
                )

        # Convergence check
        wrmse = np.sqrt(np.mean(np.min(residuals, axis=1)))
        if np.array_equal(assignments, new_assignments):
            converged = True
            assignments = new_assignments
            break
        if prior_wrmse < np.inf and abs(wrmse - prior_wrmse) <= settings.em_convergence_tol * prior_wrmse:
            converged = True
            assignments = new_assignments
            break

        prior_wrmse = wrmse
        prev_assignments = assignments
        assignments = new_assignments

    # Final M-step with converged assignments (lightweight)
    splines = _fit_regimes(df, assignments, settings, pspline_settings)
    return splines, assignments, converged


def _fit_regimes(
    df: pd.DataFrame,
    assignments: np.ndarray,
    settings,
    pspline_settings=None,
) -> dict[int, PSpline] | None:
    """Fit one PSpline per regime on assigned days.

    Parameters
    ----------
    pspline_settings : optional
        Override for PSpline settings (e.g. lightweight during EM).
        If None, uses ``settings.pspline``.

    Returns None if any regime has fewer than ``min_regime_days`` days.
    """
    ps = pspline_settings if pspline_settings is not None else settings.pspline
    temp = df["temperature"].values
    obs = df["observed"].values
    regime_ids = np.unique(assignments)
    splines: dict[int, PSpline] = {}

    for k in regime_ids:
        mask = assignments == k
        if np.sum(mask) < settings.min_regime_days:
            return None

        # Sort by temperature for fit_segment
        x_k = temp[mask]
        y_k = obs[mask]
        sort_idx = np.argsort(x_k)
        x_k = x_k[sort_idx]
        y_k = y_k[sort_idx]

        splines[int(k)] = fit_segment(x_k, y_k, ps)

    return splines


def _merge_small_regime(
    assignments: np.ndarray,
    small_k: int,
    residuals: np.ndarray,
    min_days: int,
) -> np.ndarray:
    """Merge a too-small regime into the next-best regime for each day."""
    assignments = assignments.copy()
    mask = assignments == small_k
    # For days in the small regime, pick the next-best regime
    resid_copy = residuals[mask].copy()
    resid_copy[:, small_k] = np.inf
    assignments[mask] = np.argmin(resid_copy, axis=1)
    return assignments


# ------------------------------------------------------------------
# BIC scoring
# ------------------------------------------------------------------

def _compute_bic(
    splines: dict[int, PSpline],
    assignments: np.ndarray,
    N: int,
    penalty_multiplier: float = 1.0,
    penalty_power: float = 2.0,
) -> float:
    """RSS-based score with a penalty on the number of regimes K.

    score = N * ln(RSS/N) + multiplier * K^power * ln(N)

    The penalty is on K (number of splits), not on per-spline
    coefficients — spline complexity is already handled by each
    regime's internal model selection. The power term makes each
    additional regime progressively more expensive.
    """
    K = len(splines)
    rss = 0.0
    for k, spl in splines.items():
        resid = spl.y - spl.predict(spl.x)
        rss += float(np.sum(resid ** 2))

    if rss <= 0:
        rss = 1e-10

    return N * np.log(rss / N) + penalty_multiplier * (K ** penalty_power) * np.log(N)
