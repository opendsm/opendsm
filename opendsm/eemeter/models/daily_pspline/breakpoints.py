"""Breakpoint estimation and optimization for the P-spline model.

Breakpoints delineate the HDD (heating), TIDD (temperature-independent),
and CDD (cooling) zones of the energy-temperature curve.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from opendsm.eemeter.models.daily.utilities.opt_settings import OptimizationSettings
from opendsm.eemeter.models.daily.optimize import NLoptOptimizer
from opendsm.eemeter.models.daily_pspline.solver import PSplineSolver, pad_knots


def optimize_breakpoints(
    x: np.ndarray,
    y: np.ndarray,
    bp_init: np.ndarray,
    knots_obj,
    weights: Optional[np.ndarray],
    n_min: int,
    zone_knot_count: int,
    degree: int,
    lambda_smoothing: float,
    bc_type: Optional[str],
    kappa: float,
    maxiter: int,
    reg_alpha: float,
    reg_pct_lasso: float,
    allow_hdd: bool,
    allow_cdd: bool,
    algorithm: str = "nlopt_direct",
) -> tuple[np.ndarray, PSplineSolver]:
    """Optimize breakpoint positions via NLopt global/local search.

    Breakpoints are parameterized as normalized cumulative fractions
    of the data range, ensuring bp[0] <= bp[1] by construction.

    Returns
    -------
    bp : ndarray
        Optimized breakpoints [lower, upper].
    solver : PSplineSolver
        Solver built on the final knot configuration.
    """
    x_min, x_max = x[0], x[-1]
    x_range = x_max - x_min
    N = len(x)

    inner_maxiter = min(15 if N <= 150 else 5, maxiter) if maxiter else 5

    lasso_a = reg_pct_lasso * reg_alpha
    ridge_a = (1 - reg_pct_lasso) * reg_alpha
    x_bnds = np.array([x_min, x_max])
    has_reg = reg_alpha != 0

    def _X_to_bp(X):
        bp0 = X[0] * x_range + x_min
        bp1 = X[1] * (x_max - bp0) + bp0
        return np.array([bp0, bp1])

    # Build reduced parameter space for enabled zones
    remaining = x_max - bp_init[0]
    x0_full = np.clip([
        (bp_init[0] - x_min) / x_range,
        (float(bp_init[1]) - bp_init[0]) / remaining if remaining > 0 else 0.0,
    ], 0.0, 1.0)
    bnds_full = np.array([(0.0, 1.0), (0.0, 1.0)])

    if allow_hdd and allow_cdd:
        to_full = lambda X: X
        x0_opt, bnds_opt = x0_full, bnds_full
    elif allow_hdd:
        to_full = lambda X: np.array([X[0], 1.0])
        x0_opt, bnds_opt = x0_full[:1], bnds_full[:1]
    elif allow_cdd:
        to_full = lambda X: np.array([0.0, X[0]])
        x0_opt, bnds_opt = x0_full[1:], bnds_full[1:]
    else:
        # Both zones disabled — no optimization needed
        bp = np.array([x_min, x_max])
        final_knots = knots_obj.get_internal_knots(bp=bp, n_knots=zone_knot_count, n_min=n_min)
        solver = PSplineSolver(x, y, pad_knots(final_knots, degree), degree, weights, lambda_smoothing, bc_type, kappa)
        return bp, solver

    def objective(X_free, grad=None):
        trial_bp = _X_to_bp(to_full(X_free))
        trial_knots = knots_obj.get_internal_knots(bp=trial_bp, n_knots=zone_knot_count, n_min=n_min)
        trial_padded = pad_knots(trial_knots, degree)
        psp = PSplineSolver(x, y, trial_padded, degree, weights, lambda_smoothing, bc_type, kappa)
        coefs, _, _ = psp.solve(trial_bp, weights, kappa, inner_maxiter)

        resid = psp.B @ coefs - y
        loss = np.sum(resid ** 2) / N
        wrmse = np.sqrt(loss)

        if has_reg:
            penalty = trial_bp - x_bnds
            penalty *= wrmse / x_range
            if lasso_a:
                loss += lasso_a * np.linalg.norm(penalty, 1)
            if ridge_a:
                loss += ridge_a * np.linalg.norm(penalty, 2)

        return loss

    budget = int(np.clip(N // 10, 100 if "direct" in algorithm else 30,
                         400 if "direct" in algorithm else 100))
    opt_settings = OptimizationSettings(
        algorithm=algorithm, initial_step=0.025,
        stop_criteria_type="iteration maximum", stop_criteria_value=budget,
        x_tol_rel=1e-3, f_tol_rel=1e-3,
    )

    optimizer = NLoptOptimizer(objective, x0_opt, bnds_opt, opt_settings)
    result = optimizer.run()
    bp = _X_to_bp(to_full(result.x))

    # Absorb small zones into TIDD
    n_hdd = int(x.searchsorted(bp[0], side='left'))
    if 0 < n_hdd < n_min:
        bp[0] = x_min
        n_hdd = 0

    n_cdd = N - int(x.searchsorted(bp[1], side='right'))
    if 0 < n_cdd < n_min:
        bp[1] = x_max
        n_cdd = 0

    n_tidd = N - n_hdd - n_cdd
    if n_tidd < n_min:
        bp[:] = np.mean(bp)

    if not allow_hdd:
        bp[0] = x_min
    if not allow_cdd:
        bp[1] = x_max

    final_knots = knots_obj.get_internal_knots(bp=bp, n_knots=zone_knot_count, n_min=n_min)
    solver = PSplineSolver(x, y, pad_knots(final_knots, degree), degree, weights, lambda_smoothing, bc_type, kappa)
    return bp, solver
