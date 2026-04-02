"""Adaptive fitting loop for a single temperature-energy segment.

Orchestrates breakpoint optimization, BIC knot-count scan, adaptive
weight cycling, and knot refinement to produce a PSpline.
All mutable state is local — no class-level mutation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from opendsm.common.metrics import BaselineMetrics
from opendsm.common.stats.adaptive_loss import KernelWeightCache
from opendsm.eemeter.models.daily.utilities.selection_criteria import selection_criteria

from opendsm.eemeter.models.daily_pspline.solver import PSplineSolver, pad_knots, effective_df
from opendsm.eemeter.models.daily_pspline.breakpoints import optimize_breakpoints
from opendsm.eemeter.models.daily_pspline.spline import PSpline, predict_raw
from opendsm.eemeter.models.daily_pspline.uncertainty import build_uncertainty_estimator
from opendsm.eemeter.models.daily_pspline.knots import Knots


def fit_segment(
    x: np.ndarray,
    y: np.ndarray,
    settings,
    weights: Optional[np.ndarray] = None,
    bp: Optional[np.ndarray] = None,
    time_sort: Optional[np.ndarray] = None,
) -> PSpline:
    """Fit one temperature-energy segment.

    Standardizes data, runs the adaptive fitting loop, and returns a
    fully-initialized PSpline. All intermediate state (weights,
    knots, regularization) is local to this function.

    Parameters
    ----------
    x : array-like
        Temperature values (sorted).
    y : array-like
        Energy values.
    settings : DailyPSplineSettings
        Model hyperparameters.
    weights : array-like or None
        Per-observation weights; uniform if None.
    bp : array-like or None
        Fixed breakpoints [lower, upper]; auto-estimated if None.

    Returns
    -------
    PSpline
        Fitted prediction object.
    """
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    if weights is None:
        weights = np.ones_like(x, dtype=float)
    else:
        weights = np.ascontiguousarray(weights, dtype=float)

    x_mean, x_std = np.mean(x), _clipped_std(x)
    y_mean, y_std = np.mean(y), _clipped_std(y)
    x_s = (x - x_mean) / x_std
    y_s = (y - y_mean) / y_std

    degree = settings.bspline_degree
    zone_knot_count = settings.zone.knot_count_max if settings.zone.knot_count_max is not None else 10
    bp_provided = bp is not None

    if bp_provided:
        bp_s = (np.asarray(bp, dtype=float) - x_mean) / x_std
    else:
        mid = 0.5 * (x_s[0] + x_s[-1])
        bp_s = np.array([mid, mid])

    N = len(x_s)
    bic_maxiter = min(10, settings.maxiter)

    solver, coefs, bp_s, V, final_lhs = _fit_degree(
        x_s, y_s, bp_s, bp_provided, weights, zone_knot_count, bic_maxiter, N,
        "nlopt_direct", degree, settings,
    )

    bp_orig = bp_s * x_std + x_mean
    fit_bnds = np.array([x[0], x[-1]])
    num_params = solver.n_base - solver.k + 1

    # Compute predictions before construction (no mutation)
    pred = predict_raw(
        solver.padded_knots, coefs, solver.k,
        x, x_mean, x_std, y_mean, y_std, fit_bnds, settings.bc_type,
    )

    config = {
        "n_min": settings.zone.n_min,
        "lambda_smoothing": settings.lambda_smoothing,
        "lambda_curvature": settings.lambda_curvature,
        "kappa_penalty": settings.kappa_penalty,
        "maxiter": settings.maxiter,
        "slope_threshold_pct": settings.slope_threshold_pct,
        "include_autocorr": settings.include_autocorrelation_in_uncertainty,
        "uncertainty_alpha": settings.uncertainty_alpha,
    }

    return PSpline(
        knots_std=solver.padded_knots,
        coefs_std=coefs,
        degree=solver.k,
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
        bp=bp_orig, fit_bnds=fit_bnds,
        bc_type=settings.bc_type,
        config=config,
        x=x, y=y,
        training_metrics=BaselineMetrics(
            df=pd.DataFrame({"observed": y, "predicted": pred}),
            num_model_params=num_params,
        ),
        uncertainty=build_uncertainty_estimator(
            solver, coefs, V, x, y_s,
            residuals=y - pred,
            weights=weights,
            y_std=y_std,
            x_mean=x_mean,
            x_std=x_std,
            settings=settings,
            n=N,
            ddof=num_params,
            time_sort=time_sort,
        ),
    )


# ------------------------------------------------------------------
# Core adaptive loop
# ------------------------------------------------------------------

def _fit_degree(
    x_s, y_s, bp_s, bp_provided, weights, zone_knot_count,
    bic_maxiter, N, bp_algo, degree, settings,
):
    """Adaptive fitting loop for one bspline degree.

    Iterates: bp optimization → BIC knot scan → adaptive weight update,
    until convergence. Returns (solver, coefs, bp_s, V).
    """
    s = settings
    n_min = s.zone.n_min
    kappa = s.kappa_penalty
    reg_alpha = s.regularization_alpha

    knots_obj = Knots(
        x_s, y_s, w=weights,
        spline_interp_count=1000, spline_lambda=10,
        n_min=n_min, bspline_degree=degree,
        lambda_smoothing=s.lambda_smoothing,
        kappa_penalty=kappa, maxiter=s.maxiter,
        lightweight=(zone_knot_count == 0),
    )

    B_cache: dict = {}
    prior_wrmse = np.inf
    prior_bp = None
    solver = coefs = V_final = None
    knots_refined = False
    prior_a_weights = None
    weight_iters = s.max_weight_iterations

    kernel_cache = KernelWeightCache(
        x_s, zone_knot_count,
        min_knot_spacing_pct=knots_obj.min_knot_spacing_pct,
        n_eff_min=n_min,
    )

    for iter_idx in range(s.adaptive_iterations):
        # Disable regularization after first iteration
        iter_reg = reg_alpha if iter_idx == 0 else 0.0
        iter_algo = bp_algo if iter_idx == 0 else "nlopt_sbplx"

        if not bp_provided and (prior_bp is None or not (
            s.freeze_bp_on_convergence and np.allclose(bp_s, prior_bp, atol=1e-5)
        )):
            bp_s, _ = optimize_breakpoints(
                x_s, y_s, bp_s, knots_obj, weights,
                n_min=n_min, zone_knot_count=zone_knot_count,
                degree=degree, lambda_smoothing=s.lambda_smoothing,
                bc_type=s.bc_type, kappa=kappa, maxiter=s.maxiter,
                reg_alpha=iter_reg, reg_pct_lasso=s.regularization_percent_lasso,
                allow_hdd=s.zone.allow_heating_zone,
                allow_cdd=s.zone.allow_cooling_zone,
                algorithm=iter_algo,
            )
        prior_bp = bp_s

        w_sq = weights ** 2
        sum_w_sq = float(np.sum(w_sq))
        y_wmean = float(np.dot(w_sq, y_s)) / sum_w_sq
        wtss = float(np.dot(w_sq, (y_s - y_wmean) ** 2))

        def _candidate(bp, n_hdd, n_cdd):
            knots = knots_obj.get_internal_knots(
                bp=bp, n_knots=zone_knot_count, n_min=n_min,
                n_knots_hdd=n_hdd, n_knots_cdd=n_cdd,
            )
            padded = pad_knots(knots, degree)
            padded_key = tuple(padded)
            B = B_cache.get(padded_key)
            if B is None:
                B = BSpline.design_matrix(x=x_s, t=padded, k=degree, extrapolate=True).toarray()
                B_cache[padded_key] = B
            psp = PSplineSolver(
                x_s, y_s, padded, degree, weights,
                s.lambda_smoothing, s.bc_type, kappa, B=B,
            )
            c, v, lhs = psp.solve(bp, weights, kappa, bic_maxiter)
            resid = psp.B @ c - y_s
            wssr = float(np.dot(w_sq, resid ** 2))
            edf = effective_df(psp, v, kappa, w_sq)
            score = selection_criteria(
                wssr, wtss, N, edf,
                s.zone.criteria, s.zone.penalty_multiplier, s.zone.penalty_power,
            )
            return score, psp, c, v, lhs

        hdd_max = (
            min(zone_knot_count, int(x_s.searchsorted(bp_s[0], side='left')) // n_min)
            if s.zone.allow_heating_zone else 0
        )
        cdd_max = (
            min(zone_knot_count, (N - int(x_s.searchsorted(bp_s[1], side='right'))) // n_min)
            if s.zone.allow_cooling_zone else 0
        )

        solver, coefs, V_final, final_lhs = _zone_knot_scan(bp_s, hdd_max, cdd_max, _candidate)

        residuals = (solver.B @ coefs - y_s).reshape(-1)
        wrmse = np.sqrt(np.dot(w_sq, residuals ** 2) / sum_w_sq)

        if prior_wrmse < np.inf and abs(wrmse - prior_wrmse) <= 1e-3 * prior_wrmse:
            break
        prior_wrmse = wrmse

        # Curvature-based knot refinement (degree >= 2, first iteration only)
        if not knots_refined and degree >= 2 and iter_idx == 0:
            refined = _knots_from_constrained_curvature(
                solver, coefs, bp_s, x_s, n_min,
                n_knots_hdd=hdd_max, n_knots_cdd=cdd_max,
                n_knots_tidd=zone_knot_count,
            )
            knots_obj._knot_cache.clear()
            knots_obj._refined_knots = refined
            knots_obj._refined_bp = bp_s.copy()
            B_cache.clear()
            knots_refined = True

        # Kernel adaptive weights
        if weight_iters > 0:
            a_weights, median_alpha = kernel_cache.compute_weights(residuals)
            weights = _rescale_to_range(
                (weights * a_weights) if weights is not None else a_weights,
            )
            weight_iters -= 1

            if median_alpha == 2.0:
                break
            if prior_a_weights is not None and np.max(np.abs(a_weights - prior_a_weights)) < 0.1:
                break
            prior_a_weights = a_weights
        else:
            break

    # Re-solve with curvature penalty applied (post model-selection).
    # This is done after bp/knot selection so the penalty only regularizes
    # the final coefficients without altering the model structure.
    if s.lambda_curvature > 0 and solver is not None:
        psp_final = PSplineSolver(
            x_s, y_s, solver.padded_knots, degree, weights,
            s.lambda_smoothing, s.bc_type, kappa, B=solver.B,
            lambda_curvature=s.lambda_curvature,
        )
        coefs, V_final, final_lhs = psp_final.solve(bp_s, weights, kappa, s.maxiter)
        solver = psp_final

    return solver, coefs, bp_s, V_final, final_lhs


# ------------------------------------------------------------------
# Zone knot scan
# ------------------------------------------------------------------

def _zone_knot_scan(bp_s, hdd_max, cdd_max, candidate_fn):
    """Two-axis alternating search over (n_hdd, n_cdd) knot counts.

    Exploits near-independence of HDD and CDD zones. Alternating-axis
    search with a verification pass catches coupling when it matters.

    Returns (solver, coefs, V, LHS).
    """
    hdd_max = min(hdd_max, 8)
    cdd_max = min(cdd_max, 8)
    best_score = np.inf
    best_solver = best_coefs = best_V = best_lhs = None

    def _scan(fixed_axis, fixed_val, scan_max):
        nonlocal best_score, best_solver, best_coefs, best_V, best_lhs
        best_count = 0
        for count in range(scan_max + 1):
            n_hdd = count if fixed_axis == "cdd" else fixed_val
            n_cdd = fixed_val if fixed_axis == "cdd" else count
            score, psp, c, v, lhs = candidate_fn(bp_s, n_hdd, n_cdd)
            if score < best_score:
                best_score, best_solver, best_coefs, best_V, best_lhs = score, psp, c, v, lhs
                best_count = count
        return best_count

    best_hdd = _scan("cdd", cdd_max, hdd_max)
    best_cdd = _scan("hdd", best_hdd, cdd_max)
    prev_hdd = best_hdd
    best_hdd = _scan("cdd", best_cdd, hdd_max)
    if best_hdd != prev_hdd:
        _scan("hdd", best_hdd, cdd_max)

    return best_solver, best_coefs, best_V, best_lhs


# ------------------------------------------------------------------
# Knot refinement from constrained curvature
# ------------------------------------------------------------------

def _knots_from_constrained_curvature(
    solver, coefs, bp_s, x_s, n_min,
    n_knots_hdd, n_knots_cdd, n_knots_tidd=10,
):
    """Place knots using curvature of a constrained P-spline fit.

    Uses equi-curvature integral from Yeh (2020) on the monotone-
    constrained spline rather than an unconstrained smoother.

    References
    ----------
    Yeh, R. et al. (2020). Fast Automatic Knot Placement Method for
        Accurate B-spline Curve Fitting. DOI:10.1016/j.cad.2020.102905
    """
    spl = BSpline(solver.padded_knots, coefs, solver.k)
    x_lo, x_hi = x_s[0], x_s[-1]

    def _curvature_knots(x_range_lo, x_range_hi, n_knots):
        if n_knots <= 0:
            return np.array([])
        n_eval = max(100, 20 * n_knots)
        xs = np.linspace(x_range_lo, x_range_hi, n_eval)

        deriv_order = min(2, solver.k)
        if deriv_order == 0:
            curv = np.abs(spl(xs))
        else:
            curv = np.abs(spl.derivative(deriv_order)(xs))

        curv_max = np.max(curv)
        curv = np.maximum(curv, 1e-10 * curv_max) if curv_max > 0 else np.ones_like(curv)

        dx = xs[1] - xs[0]
        cum = np.empty(len(curv))
        cum[0] = 0.0
        np.cumsum(0.5 * dx * (curv[:-1] + curv[1:]), out=cum[1:])

        targets = np.linspace(0, cum[-1], n_knots + 2)
        return np.interp(targets, cum, xs)

    # HDD zone
    hdd_knots = np.array([])
    n_hdd_data = int(x_s.searchsorted(bp_s[0], side='left'))
    if n_hdd_data >= n_min and n_knots_hdd > 0:
        n_knots_hdd = min(n_knots_hdd, n_hdd_data // n_min)
        hdd_knots = _curvature_knots(x_lo, bp_s[0], n_knots_hdd)
        hdd_knots = hdd_knots[hdd_knots < bp_s[0]]

    # TIDD zone — uniform
    if np.isclose(bp_s[0], bp_s[1]):
        tidd_knots = np.array([bp_s[0]])
    else:
        tidd_knots = np.linspace(bp_s[0], bp_s[1], n_knots_tidd + 2)

    # CDD zone
    cdd_knots = np.array([])
    n_cdd_data = len(x_s) - int(x_s.searchsorted(bp_s[1], side='right'))
    if n_cdd_data >= n_min and n_knots_cdd > 0:
        n_knots_cdd = min(n_knots_cdd, n_cdd_data // n_min)
        cdd_knots = _curvature_knots(bp_s[1], x_hi, n_knots_cdd)
        cdd_knots = cdd_knots[cdd_knots > bp_s[1]]

    return np.hstack([hdd_knots, tidd_knots, cdd_knots])


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _clipped_std(values: np.ndarray, clip_val: float = 1e-6) -> float:
    """Standard deviation clipped to avoid near-zero values."""
    std = np.std(values)
    return 1.0 if std < clip_val else std


def _rescale_to_range(
    values: np.ndarray,
    new_min: float = 1.0,
    new_max: float = 10.0,
) -> np.ndarray:
    """Min-max rescale to [new_min, new_max]."""
    values = np.asarray(values)
    old_min, old_max = np.min(values), np.max(values)
    if old_max == old_min:
        return np.full_like(values, new_min, dtype=float)
    scale = (new_max - new_min) / (old_max - old_min)
    return new_min + scale * (values - old_min)
