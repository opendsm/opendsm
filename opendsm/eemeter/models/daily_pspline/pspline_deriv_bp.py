from __future__ import annotations

import json
import warnings
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils import _check_and_correct_inputs
from spline_knots import Knots

from opendsm.common.stats.adaptive_loss import adaptive_weights
from opendsm.common.metrics import BaselineMetrics, BaselineMetricsFromDict
from opendsm.eemeter.models.daily.utilities.opt_settings import OptimizationSettings
from opendsm.eemeter.models.daily.optimize import NLoptOptimizer


class MaxIterationWarning(UserWarning): ...


class DailyPSplineSchema(BaseModel):
    """Pydantic schema for DailyPSpline serialization."""

    # DailyPSpline configuration
    n_min: int = Field(..., description="Minimum points per zone")
    lambda_smoothing: float = Field(..., description="Smoothing parameter")
    kappa_penalty: float = Field(..., description="Monotonicity penalty")
    maxiter: int = Field(..., description="Maximum iterations")

    # Standardization parameters
    x_mean: float = Field(..., description="Mean of x training data")
    x_std: float = Field(..., description="Standard deviation of x training data")
    y_mean: float = Field(..., description="Mean of y training data")
    y_std: float = Field(..., description="Standard deviation of y training data")

    # BSpline parameters
    knots: List[float] = Field(..., description="Full knot vector including padding")
    coefficients: List[float] = Field(..., description="Spline coefficients")
    degree: int = Field(..., description="Spline degree")
    extrapolate: bool = Field(..., description="Extrapolation flag")

    # Fitted state
    breakpoints: List[float] = Field(..., description="Zone breakpoints [lower, upper]")
    fit_bounds: List[float] = Field(..., description="Training data bounds [min, max]")

    # Metrics
    training_metrics: Optional[dict] = Field(None, description="Training metrics")


def _pad_knots(
    internal_knots: np.ndarray,
    bspline_degree: int,
) -> np.ndarray:
    """Add repeated boundary knots for B-spline boundary stability.

    Adds ``bspline_degree`` repeated knots at each end (multiplicity k+1),
    so that exactly one basis function is non-zero at each boundary.

    See De Leeuw (2017) *Computing and Fitting Monotone Splines*.

    Parameters
    ----------
    internal_knots : array-like
        Internal knot vector (without padding).
    bspline_degree : int
        Degree of B-spline basis.

    Returns
    -------
    knots : ndarray
        Full knot vector including padding knots.
    """
    k = bspline_degree

    left_padding = np.repeat(internal_knots[0], k)
    right_padding = np.repeat(internal_knots[-1], k)

    return np.hstack([left_padding, internal_knots, right_padding])


@lru_cache(maxsize=16)
def _cached_first_diff(n: int) -> np.ndarray:
    """First-order difference matrix of size (n-1, n). Cached by n."""
    return np.diff(np.eye(n), n=1, axis=0)


def _difference_matrices(
    knots: np.ndarray,
    n_base_funcs: int,
    bspline_degree: int = 3
) -> list[np.ndarray]:
    """Compute weighted difference matrices D1, D2, D3 for non-uniform B-splines.

    These matrices approximate derivatives of the spline coefficients, accounting
    for non-uniform knot spacing via inverse weighting.

    References
    ----------
    - General P-Splines for Non-Uniform B-Splines (https://doi.org/10.48550/arXiv.2201.06808)
    - Multivariate probabilistic CRPS learning with an application to day-ahead electricity prices
      (https://doi.org/10.1016/j.ijforecast.2024.01.005)

    Parameters
    ----------
    knots : array-like
        Full knot vector.
    n_base_funcs : int
        Number of B-spline basis functions.
    bspline_degree : int
        Degree of B-spline basis (order = degree + 1).

    Returns
    -------
    D : list of ndarray
        [D1, D2, D3] difference matrices for 1st, 2nd, 3rd derivatives.
    """

    def lag_diff(arr, lag):
        return arr[lag:] - arr[:-lag]

    o = bspline_degree

    D = [None, None, None]  # D1, D2, D3
    for i in range(3):
        if o > i:
            # Weighted difference for non-uniform knots
            a = 1 / (o - i)
            diag_vals = a * lag_diff(knots[i+1:-i-1], lag=o - i)

            # Replace zeros from repeated boundary knots with 1 to avoid
            # division by zero. The corresponding rows become standard
            # (unweighted) differences, which is correct since repeated
            # knots have zero spacing and shouldn't be penalized differently.
            diag_vals = np.where(diag_vals == 0, 1.0, diag_vals)

            # W is diagonal, so inv(W) @ fd == fd / diag_vals (broadcasting)
            fd = _cached_first_diff(n_base_funcs - i)
            _D = fd / diag_vals[:, np.newaxis]

            D[i] = _D if i == 0 else _D @ D[i-1]

        else:
            # Spline order <= derivative order: use standard difference
            D[i] = np.diff(np.eye(n_base_funcs), n=i+1, axis=0)

    return D


def _derivative_zones(
    knots: np.ndarray,
    bp: np.ndarray,
    bspline_degree: int
) -> dict[str, np.ndarray]:
    """Assign derivative constraint zones to basis function intervals.

    Uses overlap-based classification: the i-th first-difference row
    corresponds to a derivative basis function with support
    [knots[i+1], knots[i+k+1]]. A derivative is assigned to TIDD if its
    support overlaps the TIDD zone [bp[0], bp[1]], ensuring the continuous
    derivative is zero everywhere inside TIDD. Derivatives entirely below
    bp[0] are HDD; entirely above bp[1] are CDD.

    hdd:  heating zone — penalize increasing derivatives
    tidd: temperature-independent zone — penalize any non-zero derivative
    cdd:  cooling zone — penalize decreasing derivatives
    """
    empty = np.array([], dtype=int)
    k = bspline_degree
    n_deriv = len(knots) - k - 2

    # Edge cases: single breakpoint at domain boundary
    rtol = 1e-4
    single_bp = np.isclose(bp[0], bp[1], rtol=rtol)

    if single_bp and np.isclose(knots[-1], bp[1], rtol=rtol):
        return {"hdd": np.arange(n_deriv), "tidd": empty, "cdd": empty}

    if single_bp and np.isclose(knots[0], bp[0], rtol=rtol):
        return {"hdd": empty, "tidd": empty, "cdd": np.arange(n_deriv)}

    # Support interval of each derivative basis function
    left = knots[1:1 + n_deriv]
    right = knots[k + 1:k + 1 + n_deriv]

    # Overlap-based: TIDD if support intersects [bp[0], bp[1]]
    # HDD if support entirely <= bp[0]; CDD if support entirely >= bp[1]
    knot_idx = {
        "hdd": np.flatnonzero(right <= bp[0]),
        "tidd": np.flatnonzero((left < bp[1]) & (right > bp[0])),
        "cdd": np.flatnonzero(left >= bp[1]),
    }

    return knot_idx


def _tidd_weights(
    x: np.ndarray,
    weights: Optional[np.ndarray],
    bp: np.ndarray,
    weight_factor: float = 10
) -> np.ndarray:
    """Upweight observations in the TIDD zone to enforce flatness.

    Parameters
    ----------
    x : array-like
        Input x-coordinates.
    weights : array-like or None
        Existing weights (if any).
    bp : array-like
        Breakpoints [lower, upper] defining the TIDD zone.
    weight_factor : float
        Multiplicative weight for TIDD observations.

    Returns
    -------
    weights : ndarray
        Updated weight vector.
    """
    tidd_idx = np.flatnonzero((x > bp[0]) & (x < bp[1]))

    if tidd_idx.size == 0:
        return weights

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = weights.copy()  # avoid compounding across repeated fit calls

    weights[tidd_idx] *= weight_factor

    return weights


def fit_daily_pspline(
    x: np.ndarray,
    y: np.ndarray,
    bp: np.ndarray,
    knots: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bspline_degree: int = 3,
    lambda_smoothing: float = 0.1,
    kappa_penalty: float = 10**6,
    maxiter: int = 30,
    bc_type: Optional[str] = None,
    _validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a penalized B-spline (P-spline) with zone-specific monotonicity constraints.

    Monotone smoothing splines implementation using penalized B-splines (aka. P-splines).

    Solves the equation

        (B'B + λD3'D3 + κD1'VD1)α = B'y

    where

        B'B: The least squares part
        λD3'D3: The smoothing part
        κD1'VD1: The monotonicity part
        α: The coefficients of the B-spline basis functions

    The algorithm was introduced in [Eilers2005]

    References:
    -----------
    [Eilers2005]: Eilers, P. H. C. 2005. Unimodal smoothing. Journal of Chemometrics
                19:317–328. DOI:10.1002/cem.935.

    Parameters
    ----------
    x: array-like
        The x-coordinates of the training data.
    y: array-like
        The y-coordinates of the training data.
    bp: array-like
        Breakpoints defining the TIDD (temperature-independent) zone as [lower, upper].
    weights: array-like, optional
        Per-observation weights for weighted least squares.
    bspline_degree: int
        The degree of the B-spline (which is also the degree of the fitted spline
        function). The order of the splines is degree + 1.
    knots: array-like
        Internal knot positions (without padding).
    lambda_smoothing: float
        The smoothing parameter. Higher values will result in smoother curves.
    kappa_penalty: float
        The penalty parameter for enforcing monotonicity. Higher values will result in
        more monotonic curves. kappa_penalty of 0 means that monotonicity is not
        enforced at all.
    maxiter: int
        Maximum number of iterations for the algorithm. If the algorithm does not
        converge within this number of iterations, a warning is issued.

    Returns
    -------
    knots : ndarray
        Full knot vector including padding knots.
    coefs : ndarray
        Fitted spline coefficients.

    Notes
    -----
    Alternative implementations investigated but found consistently slower:

    1. Warm start (CG): 2-3x slower for typical problems (~0.8 ms overhead)
       - See benchmark_sequential_fits.py and WARM_START_REMOVED.md
    2. Sparse matrices: Slower across ALL tested sizes (100 to 100,000 points)
       - 3-4x slower for typical data (100-365 points)
       - Still 1.03x slower even at 100,000 points
       - See benchmark_scaling_analysis.py and SPARSE_MATRICES_ANALYSIS.md

    Dense matrices with direct solve via np.linalg.solve is optimal across all
    data sizes. The matrix size (~12x12 basis functions) remains small regardless
    of data size, so dense BLAS/LAPACK operations are always fastest.

    https://github.com/fohrloop/penalized-splines/blob/main/penalized_splines.py
    """

    # Input validation (skipped in hot paths where inputs are already clean)
    if _validate:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        bp = np.asarray(bp, dtype=float)
        knots = np.asarray(knots, dtype=float)

        if len(x) != len(y):
            raise ValueError(f"x and y must have same length (got {len(x)} and {len(y)})")

        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise ValueError("x and y must not contain NaN or Inf values")

        if len(bp) != 2:
            raise ValueError(f"bp must have length 2 (got {len(bp)})")

        if bp[0] > bp[1]:
            raise ValueError(f"bp[0] must be <= bp[1] (got {bp[0]} > {bp[1]})")

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if len(weights) != len(x):
                raise ValueError(f"weights must match length of x (got {len(weights)} vs {len(x)})")
            if np.any(weights <= 0):
                raise ValueError("weights must be positive")

    # Add padding knots outside data range for boundary stability
    knots = _pad_knots(knots, bspline_degree)
    n_base_funcs = len(knots) - bspline_degree - 1

    B = BSpline.design_matrix(x=x, t=knots, k=bspline_degree, extrapolate=True).toarray()
    D1, D2, D3 = _difference_matrices(knots, n_base_funcs, bspline_degree=bspline_degree)
    D1_zone = _derivative_zones(knots, bp, bspline_degree)

    # Add penalty to enforce zero second derivatives at boundaries for stability
    boundary_penalty_matrix = 0.0
    if bc_type == "clamped" and bspline_degree >= 2:
        boundary_matrix = np.vstack([D1[0, :], D1[-1, :]])
        boundary_penalty_matrix = kappa_penalty * boundary_matrix.T @ boundary_matrix
    elif bc_type == "natural" and bspline_degree >= 2:
        boundary_matrix = np.vstack([D2[0, :], D2[-1, :]])
        boundary_penalty_matrix = kappa_penalty * boundary_matrix.T @ boundary_matrix

    # Cache D1.T for repeated use in loop
    D1T = D1.T

    # Smooth third derivative penalty
    if lambda_smoothing > 0:
        D3_smoothing = lambda_smoothing * D3.T @ D3
    else:
        D3_smoothing = 0.0

    # Set TIDD weights if the zone exists
    if bp[1] - bp[0] > 0:
        weights = _tidd_weights(x, weights, bp, weight_factor=50)

    if weights is None:
        B_gram = B.T @ B
        A = B_gram + D3_smoothing + boundary_penalty_matrix
        BTy = B.T @ y

    else:
        # A Weighted Least-Squares Approach for B-Spline Shape Representation
        # https://doi.org/10.1109/ISCCSP.2008.4537350
        # Use broadcasting instead of dense n×n diagonal matrix
        w_sq = np.square(weights)
        Bw = B * w_sq[:, np.newaxis]

        B_gram = B.T @ Bw
        A = B_gram + D3_smoothing + boundary_penalty_matrix
        BTy = B.T @ (y * w_sq)

    # Number of derivative intervals
    n_deriv = D1.shape[0]

    # Apply iterative derivative penalization for higher-degree splines
    delta = np.zeros(n_deriv, dtype=float)
    V = np.zeros(n_deriv, dtype=int)
    V_new = np.zeros(n_deriv, dtype=int)  # Preallocate buffer

    # Preallocate arrays for loop to avoid repeated allocations
    D1_weighted = np.empty_like(D1)
    kappa_V_delta = np.empty(n_deriv, dtype=float)

    # Hoist zone checks outside loop
    hdd_idx = D1_zone["hdd"]
    tidd_idx = D1_zone["tidd"]
    cdd_idx = D1_zone["cdd"]
    has_hdd = len(hdd_idx) > 0
    has_tidd = len(tidd_idx) > 0
    has_cdd = len(cdd_idx) > 0

    for i in range(maxiter):
        # The equation : (B'w2B + λD3'D3 + κD1'VD1)α = B'w2y + κD1'V delta
        # α = coefs
        # Since W = diag(κV), use broadcasting: W @ D1 = (κV)[:, None] * D1
        kappa_V = kappa_penalty * V

        # Dense matrix operations with in-place optimization
        np.multiply(kappa_V[:, np.newaxis], D1, out=D1_weighted)
        lhs = A + D1T @ D1_weighted

        # Compute kappa_V * delta in-place
        np.multiply(kappa_V, delta, out=kappa_V_delta)
        rhs = BTy + D1T @ kappa_V_delta

        # Solve for coefs using direct solve (fastest for small systems)
        # Falls back to lstsq for near-singular matrices
        try:
            coefs = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            coefs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)

        deriv = D1 @ coefs

        # Reuse V_new buffer instead of reallocating
        V_new.fill(0)

        # Penalize for increasing in hdd zone
        if has_hdd:
            V_new[hdd_idx] = deriv[hdd_idx] > delta[hdd_idx]

        # Penalize for non-constant in tidd zone
        if has_tidd:
            V_new[tidd_idx] = deriv[tidd_idx] != delta[tidd_idx]

        # Penalize for decreasing in cdd zone
        if has_cdd:
            V_new[cdd_idx] = deriv[cdd_idx] < delta[cdd_idx]

        # Check convergence (count_nonzero is faster than sum for boolean arrays)
        dv = np.count_nonzero(V != V_new)
        if dv == 0:
            break

        # Swap references to avoid copy
        V, V_new = V_new, V

    else:
        warnings.warn(
            "Max iteration reached. The results are not reliable.", MaxIterationWarning
        )

    return knots, coefs


def _rescale_to_range(
    values: np.ndarray,
    new_min: float = 1.0,
    new_max: float = 10.0,
) -> np.ndarray:
    """Rescale values to a specified range using min-max normalization.

    Maps values from their current range [min(values), max(values)]
    to [new_min, new_max] via linear transformation.

    Parameters
    ----------
    values : array-like
        Values to rescale.
    new_min : float
        Target minimum value. Default is 1.0.
    new_max : float
        Target maximum value. Default is 10.0.

    Returns
    -------
    rescaled : ndarray
        Values rescaled to [new_min, new_max].

    Notes
    -----
    If all values are identical (max == min), returns array of new_min values.
    """
    values = np.asarray(values)
    old_min = np.min(values)
    old_max = np.max(values)

    if old_max == old_min:
        return np.full_like(values, new_min, dtype=float)

    # Linear transformation: y = a + (b - a) * (x - x_min) / (x_max - x_min)
    scale = (new_max - new_min) / (old_max - old_min)
    rescaled = new_min + scale * (values - old_min)

    return rescaled


def _estimate_bp_from_derivative(knots_obj, threshold_pct=0.10):
    """Estimate breakpoints from smoothing spline derivative analysis.

    Walks outward from the smoothing spline minimum until the derivative
    magnitude exceeds a fraction of its maximum value. This gives a cheap
    estimate of the TIDD (temperature-independent) zone boundaries.

    Parameters
    ----------
    knots_obj : Knots
        Fitted Knots object containing the smoothing spline.
    threshold_pct : float
        Fraction of maximum |dy/dx| used as the flatness threshold.

    Returns
    -------
    bp : ndarray
        Estimated breakpoints [lower, upper] in standardized space.
    """
    spl = knots_obj.spl
    spl_x = knots_obj.spl_x

    d1 = spl.derivative(1)
    y_vals = spl(spl_x)
    abs_dy = np.abs(d1(spl_x))

    max_abs_dy = np.max(abs_dy)
    if max_abs_dy < 1e-10:
        return np.array([spl_x[0], spl_x[-1]])

    threshold = threshold_pct * max_abs_dy
    min_idx = np.argmin(y_vals)

    # Walk left from minimum: find where slope exceeds threshold
    left_exceed = np.flatnonzero(abs_dy[:min_idx + 1][::-1] > threshold)
    bp0 = spl_x[min_idx - left_exceed[0]] if len(left_exceed) > 0 else spl_x[0]

    # Walk right from minimum: find where slope exceeds threshold
    right_exceed = np.flatnonzero(abs_dy[min_idx:] > threshold)
    bp1 = spl_x[min_idx + right_exceed[0]] if len(right_exceed) > 0 else spl_x[-1]

    if bp0 > bp1:
        mid = 0.5 * (bp0 + bp1)
        bp0 = bp1 = mid

    return np.array([bp0, bp1])


class DailyPSpline(BSpline):
    """Penalized B-Spline with monotonicity constraints for energy modeling.

    Implements a P-spline (penalized B-spline) with zone-specific monotonicity
    constraints suitable for modeling energy load shapes with HDD/TIDD/CDD zones.

    Parameters
    ----------
    bspline_degree : int
        Degree of the B-spline basis. Default is 3 (cubic splines).

    Attributes
    ----------
    n_min : int
        Minimum number of data points required in HDD and CDD zones.
    lambda_smoothing : float
        Smoothing parameter for third derivative penalty.
    kappa_penalty : float
        Penalty parameter for monotonicity constraints.
    maxiter : int
        Maximum iterations for constraint optimization.
    """

    def __init__(
        self,
        bspline_degree=3,
        bc_type="natural",
        regularization_alpha=0.01,
        regularization_percent_lasso=1.0,
    ):
        # TODO move to settings class?
        self.bspline_degree = bspline_degree
        self.bc_type = bc_type
        self.n_min = 5
        self.lambda_smoothing = 0
        self.kappa_penalty = 1E9
        self.maxiter = 100

        # BP regularization: penalizes bp far from data bounds to maximize TIDD width
        self.regularization_alpha = regularization_alpha
        self.regularization_percent_lasso = regularization_percent_lasso

    def _set_bp(
        self,
        x,
        y,
        bp,
        weights=None,
        zone_knot_count=10,
        search_margin=0.2,
    ):
        """Optimize breakpoint positions via simultaneous global optimization.

        Both breakpoints are optimized simultaneously using NLopt DIRECT.
        Breakpoints are parameterized as normalized cumulative fractions of
        the data range, ensuring bp[0] <= bp[1].

        Parameters
        ----------
        x : ndarray
            Independent variable (standardized).
        y : ndarray
            Dependent variable (standardized).
        bp : array-like
            Initial breakpoints [lower, upper].
        weights : ndarray or None
            Base data weights.
        zone_knot_count : int
            Number of internal knots per zone.
        search_margin : float
            Half-width of the search region around the initial bp estimate,
            expressed as a fraction of the data range.  0.2 means ±20% of
            the range in each parameterized dimension.

        Returns
        -------
        bp : ndarray
            Optimized breakpoints [lower, upper].
        """
        # force weights_alpha to be zero, weights are updated adaptively outside of this function
        weights_alpha = 2.0
        x_min = x[0]
        x_max = x[-1]
        x_range = x_max - x_min

        N = len(x)
        inner_maxiter = min(5, self.maxiter) if self.maxiter else 5
        lasso_a = self.regularization_percent_lasso * self.regularization_alpha
        ridge_a = (1 - self.regularization_percent_lasso) * self.regularization_alpha
        x_bnds = np.array([x_min, x_max])

        def bp_penalty(trial_bp, wrmse):
            # penalty for distance from data bounds (push toward extremes)
            penalty = trial_bp - x_bnds
            # penalty = np.array([np.min(np.abs(b - x_bnds)) for b in trial_bp])

            # penalty for distance between breakpoints
            # penalty += np.abs(np.diff(trial_bp))[0] / 2

            # scale by wRMSE and range
            penalty *= wrmse / x_range

            total = 0.0
            if lasso_a != 0:
                total += lasso_a * np.linalg.norm(penalty, 1)
            if ridge_a != 0:
                total += ridge_a * np.linalg.norm(penalty, 2)

            return total

        def objective(X, grad=[]):
            trial_bp = np.array(X)
            bp_sum = np.sum(trial_bp)
            if bp_sum > 1:
                trial_bp /= bp_sum
            trial_bp = np.cumsum(trial_bp) * x_range + x_min

            trial_knots = self.knots.get_internal_knots(
                bp=trial_bp,
                n_knots=zone_knot_count,
                n_min=self.n_min,
            )

            try:
                padded, coefs = fit_daily_pspline(
                    x=x, y=y, bp=trial_bp,
                    knots=trial_knots,
                    weights=weights,
                    bspline_degree=self.bspline_degree,
                    lambda_smoothing=self.lambda_smoothing,
                    kappa_penalty=self.kappa_penalty,
                    maxiter=inner_maxiter,
                    bc_type=self.bc_type,
                    _validate=False,
                )
            except Exception:
                return np.inf

            spl = BSpline(t=padded, c=coefs, k=self.bspline_degree, extrapolate=True)
            resid = spl(x) - y

            # adaptive weights
            if weights_alpha < 2:
                a_weights, _, _ = adaptive_weights(resid, alpha=weights_alpha, C_algo="mad")
                w = a_weights.copy()
                if weights is not None:
                    w *= weights
                w /= np.sum(w)

                wsse = np.sum(w * resid ** 2)
                loss = wsse / N
                wrmse = np.sqrt(loss)

            else:
                sse = np.sum(resid ** 2)
                loss = sse / N
                wrmse = np.sqrt(loss)

            if self.regularization_alpha != 0:
                loss += bp_penalty(trial_bp, wrmse)

            return loss

        bp_init = np.asarray(bp, dtype=float)
        x0 = np.clip([
            (bp_init[0] - x_min) / x_range,
            (bp_init[1] - bp_init[0]) / x_range,
        ], 0.0, 1.0)

        # Narrow search region around initial estimate
        bnds = np.array([
            (max(0.0, x0[0] - search_margin), min(1.0, x0[0] + search_margin)),
            (max(0.0, x0[1] - search_margin), min(1.0, x0[1] + search_margin)),
        ])

        opt_settings = OptimizationSettings(
            algorithm="nlopt_sbplx",
            stop_criteria_type="iteration maximum",
            stop_criteria_value=100,
            x_tol_rel=1e-2,
            f_tol_rel=1e-1,
        )

        optimizer = NLoptOptimizer(objective, x0, bnds, opt_settings)
        result = optimizer.run()

        # Transform result back to bp space
        bp = np.array(result.x)
        bp_sum = np.sum(bp)
        if bp_sum > 1:
            bp /= bp_sum
        bp = np.cumsum(bp) * x_range + x_min

        # Absorb HDD/CDD zones with fewer than n_min data points into TIDD
        n_hdd = np.sum(x < bp[0])
        if 0 < n_hdd < self.n_min:
            bp[0] = x_min

        n_cdd = np.sum(x > bp[1])
        if 0 < n_cdd < self.n_min:
            bp[1] = x_max

        # If fewer than n_min data points between breakpoints, merge them
        n_tidd = N - np.sum(x < bp[0]) - np.sum(x > bp[1])
        if n_tidd < self.n_min:
            avg_bp = np.mean(bp)
            bp[:] = avg_bp

        return bp

    def fit(self, x, y, bp=None, weights=None, zone_knot_count=None) -> "DailyPSpline":
        """Fit the P-spline to data with zone-specific monotonicity constraints.

        This method fits a penalized B-spline with automatic zone detection and
        monotonicity constraints. The data is standardized internally for numerical
        stability, and training metrics are computed automatically.

        Parameters
        ----------
        x : array-like
            Independent variable (e.g., temperature agnostic to units).
        y : array-like
            Dependent variable (e.g., energy load in kWh or therms).
        bp : array-like or None, optional
            Breakpoints [lower, upper] defining the TIDD (temperature-independent
            demand) zone. If None, breakpoints are estimated automatically. Default is None.
        weights : array-like or None, optional
            Per-observation weights for weighted least squares. If None, uniform
            weights are used (with upweighting in TIDD zone). Default is None.
        zone_knot_count : int or None, optional
            Number of internal knots per zone. Default is 10.

        Returns
        -------
        self : DailyPSpline
            Fitted spline instance
        """
        if zone_knot_count is None:
            zone_knot_count = 10
            
        elif zone_knot_count < 0:
            raise ValueError("`zone_knot_count` must be non-negative")

        # Fit the spline with new data
        self.x, self.y, self.weights = _check_and_correct_inputs(x, y, weights)

        # Standardize x and y for better numerical stability
        self.x_mean = np.mean(self.x)
        self.x_std = np.std(self.x)
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)

        x_standardized = (self.x - self.x_mean) / self.x_std
        y_standardized = (self.y - self.y_mean) / self.y_std

        self.knots = Knots(
            x_standardized,
            y_standardized,
            w=self.weights,
            spline_interp_count=1000,
            spline_lambda=10,
            n_min=self.n_min,
            check_data=False,
            bspline_degree=self.bspline_degree,
            lambda_smoothing=self.lambda_smoothing,
            kappa_penalty=self.kappa_penalty,
            maxiter=self.maxiter,
        )

        # Initial bp guess from smoothing spline derivative, or user-provided
        if bp is not None:
            bp_standardized = (np.asarray(bp, dtype=float) - self.x_mean) / self.x_std
        else:
            bp_standardized = _estimate_bp_from_derivative(self.knots)

        prior_rmse = np.inf
        n_iter = 10
        for n in range(n_iter):
            # Optimize breakpoints
            bp_standardized = self._set_bp(
                x_standardized, 
                y_standardized,
                bp_standardized,
                weights=self.weights,
                zone_knot_count=zone_knot_count,
            )

            # Place knots with updated breakpoints
            internal_knots = self.knots.get_internal_knots(
                bp=bp_standardized,
                n_knots=zone_knot_count,
                n_min=self.n_min,
            )

            knots, coefs = fit_daily_pspline(
                x_standardized,
                y_standardized,
                bp_standardized,
                internal_knots,
                weights=self.weights,
                bspline_degree=self.bspline_degree,
                lambda_smoothing=self.lambda_smoothing,
                kappa_penalty=self.kappa_penalty,
                maxiter=self.maxiter,
                bc_type=self.bc_type,
            )

            spl = BSpline(t=knots, c=coefs, k=self.bspline_degree, extrapolate=True)
            residuals = (y_standardized - spl(x_standardized)).reshape(-1)
            rmse = np.sqrt(np.mean(residuals ** 2))

            if np.isclose(rmse, prior_rmse, rtol=1e-3):
                break

            prior_rmse = rmse

            # Update weights using adaptive loss
            a_weights, _, weight_alpha = adaptive_weights(
                residuals, alpha="adaptive", C_algo="mad",
            )

            if weight_alpha != 2.0:
                if self.weights is None:
                    self.weights = a_weights
                else:
                    self.weights = self.weights * a_weights

                self.weights = _rescale_to_range(self.weights, new_min=1.0, new_max=100.0)

        super().__init__(t=knots, c=coefs, k=self.bspline_degree, extrapolate=True)

        # Store breakpoints and bounds in original space
        self.bp = bp_standardized * self.x_std + self.x_mean
        self.fit_bnds = np.array([self.x[0], self.x[-1]])

        # Compute training metrics
        self.training_metrics = BaselineMetrics(
            df=pd.DataFrame({
                'observed': self.y,
                'predicted': self.predict(self.x)
            }),
            num_model_params=len(internal_knots)
        )

        return self

    def __call__(self, x):
        """Evaluate the spline with automatic standardization/de-standardization.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the spline.

        Returns
        -------
        y : ndarray
            Predicted values at the given points (in original scale).
        """
        x = np.asarray(x, dtype=float)

        # Standardize input
        x_standardized = (x - self.x_mean) / self.x_std

        # Call parent in standardized space
        y_standardized = super().__call__(x_standardized)

        # De-standardize output
        y = y_standardized * self.y_std + self.y_mean

        # Extrapolation outside fit bounds based on boundary conditions
        fit_bnds = getattr(self, 'fit_bnds', None)
        bc_type = getattr(self, 'bc_type', None)
        if fit_bnds is not None and bc_type is not None:
            lo = x < fit_bnds[0]
            hi = x > fit_bnds[1]
            x_lo_std = (fit_bnds[0] - self.x_mean) / self.x_std
            x_hi_std = (fit_bnds[1] - self.x_mean) / self.x_std
            y_lo = float(super().__call__(x_lo_std)) * self.y_std + self.y_mean
            y_hi = float(super().__call__(x_hi_std)) * self.y_std + self.y_mean

            if bc_type == "clamped":
                y = np.where(lo, y_lo, np.where(hi, y_hi, y))

            elif bc_type == "natural" and self.k >= 1:
                spl = BSpline(self.t, self.c, self.k)
                deriv_spl = spl.derivative(1)
                dy_lo = float(deriv_spl(x_lo_std)) * self.y_std / self.x_std
                dy_hi = float(deriv_spl(x_hi_std)) * self.y_std / self.x_std
                lo_extrap = y_lo + dy_lo * (x - fit_bnds[0])
                hi_extrap = y_hi + dy_hi * (x - fit_bnds[1])
                y = np.where(lo, lo_extrap, np.where(hi, hi_extrap, y))

        return y

    def predict(self, x):
        """Predict values using the fitted spline.

        Parameters
        ----------
        x : array-like
            Points at which to evaluate the spline.

        Returns
        -------
        y : ndarray
            Predicted values at the given points.
        """
        return self(x)

    def derivative(self, nu=1):
        """Return the derivative of the spline in original space.

        Uses chain rule to account for x standardization:
        dy/dx = (y_std / x_std) * (dy_std/dx_std)

        Parameters
        ----------
        nu : int
            Order of derivative. Default is 1.

        Returns
        -------
        derivative_spline : DailyPSpline
            A new DailyPSpline representing the derivative.
        """
        # Get derivative in standardized space
        deriv_std = super().derivative(nu=nu)

        # Create a wrapper DailyPSpline that applies chain rule scaling
        deriv_pspline = DailyPSpline.__new__(DailyPSpline)

        # Copy the standardized derivative's BSpline parameters
        BSpline.__init__(
            deriv_pspline,
            t=deriv_std.t,
            c=deriv_std.c,
            k=deriv_std.k,
            extrapolate=deriv_std.extrapolate
        )

        # Store standardization parameters from parent
        deriv_pspline.x_mean = self.x_mean
        deriv_pspline.x_std = self.x_std
        deriv_pspline.y_mean = 0  # Derivative doesn't have y offset
        deriv_pspline.y_std = self.y_std / (self.x_std ** nu)  # Chain rule scaling
        deriv_pspline.bc_type = getattr(self, 'bc_type', None)
        deriv_pspline.fit_bnds = getattr(self, 'fit_bnds', None)

        return deriv_pspline

    def antiderivative(self, nu=1):
        """Return the antiderivative of the spline in original space.

        The antiderivative accounts for standardization:
        ∫y dx = y_std * x_std * ∫y_std dx_std + y_mean * x + C

        Parameters
        ----------
        nu : int
            Order of antiderivative. Default is 1.

        Returns
        -------
        antideriv_spline : DailyPSpline
            A DailyPSpline object that evaluates the antiderivative.
        """
        # Get antiderivative in standardized space
        antideriv_std = super().antiderivative(nu=nu)

        # Create a DailyPSpline wrapper
        antideriv_pspline = DailyPSpline.__new__(DailyPSpline)

        # Copy the standardized antiderivative's BSpline parameters
        BSpline.__init__(
            antideriv_pspline,
            t=antideriv_std.t,
            c=antideriv_std.c,
            k=antideriv_std.k,
            extrapolate=antideriv_std.extrapolate
        )

        # Store standardization parameters
        antideriv_pspline.x_mean = self.x_mean
        antideriv_pspline.x_std = self.x_std
        antideriv_pspline.y_std = self.y_std * (self.x_std ** nu)
        antideriv_pspline.y_mean = 0  # Linear term handled separately

        # Capture the linear coefficient for the y_mean * x term
        linear_coef = self.y_mean

        # Store reference to DailyPSpline's __call__ method
        pspline_call = DailyPSpline.__call__

        # Define custom __call__ that adds the linear term
        def call_with_linear(x):
            # Evaluate the spline part with standardization
            result = pspline_call(antideriv_pspline, x)
            # Add the linear term: y_mean * x
            return result + linear_coef * np.asarray(x)

        # Replace __call__ method on this instance
        antideriv_pspline.__call__ = call_with_linear

        return antideriv_pspline

    def integrate(self, a, b, extrapolate=None):
        """Compute the definite integral in original space.

        Accounts for standardization when computing ∫_a^b y dx.

        Parameters
        ----------
        a : float
            Lower integration bound (in original space).
        b : float
            Upper integration bound (in original space).
        extrapolate : bool, optional
            Whether to extrapolate beyond data bounds.

        Returns
        -------
        integral : float
            The definite integral from a to b.
        """
        # Standardize bounds
        a_std = (a - self.x_mean) / self.x_std
        b_std = (b - self.x_mean) / self.x_std

        # Integrate in standardized space
        integral_std = super().integrate(a_std, b_std, extrapolate=extrapolate)

        # Scale back to original space:
        # ∫_a^b y dx = y_std * x_std * ∫_a_std^b_std y_std dx_std + y_mean * (b - a)
        integral_orig = self.y_std * self.x_std * integral_std + self.y_mean * (b - a)

        return integral_orig

    def to_dict(self) -> dict:
        """Serialize PSpline to dictionary.

        Returns
        -------
        dict
            Dictionary containing all PSpline state needed for reconstruction.
        """
        # Prepare data dictionary
        data = {
            # DailyPSpline configuration
            "n_min": self.n_min,
            "lambda_smoothing": self.lambda_smoothing,
            "kappa_penalty": self.kappa_penalty,
            "maxiter": self.maxiter,
            # Standardization parameters
            "x_mean": float(self.x_mean),
            "x_std": float(self.x_std),
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
            # BSpline parameters
            "knots": self.t.tolist(),
            "coefficients": self.c.tolist(),
            "degree": self.k,
            "extrapolate": self.extrapolate,
            # Fitted state
            "breakpoints": self.bp.tolist(),
            "fit_bounds": self.fit_bnds.tolist(),
            # Training metrics
            "training_metrics": self.training_metrics if self.training_metrics is None else self.training_metrics.model_dump(),
        }

        # Validate with Pydantic schema
        schema = DailyPSplineSchema(**data)
        return schema.model_dump()

    def to_json(self) -> str:
        """Serialize DailyPSpline to JSON string.

        Returns
        -------
        str
            JSON string containing DailyPSpline state.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "DailyPSpline":
        """Deserialize DailyPSpline from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing DailyPSpline state (e.g., from to_dict method).

        Returns
        -------
        DailyPSpline
            Reconstructed DailyPSpline instance.
        """
        # Validate with Pydantic schema
        schema = DailyPSplineSchema(**data)

        # Create new instance without calling __init__
        pspline = cls.__new__(cls)

        # Initialize as BSpline with knots and coefficients
        knots = np.array(data["knots"])
        coefficients = np.array(data["coefficients"])
        degree = data["degree"]
        extrapolate = data["extrapolate"]

        BSpline.__init__(
            pspline,
            t=knots,
            c=coefficients,
            k=degree,
            extrapolate=extrapolate
        )

        # Set standardization parameters
        pspline.x_mean = data["x_mean"]
        pspline.x_std = data["x_std"]
        pspline.y_mean = data["y_mean"]
        pspline.y_std = data["y_std"]

        # Set DailyPSpline configuration
        pspline.bspline_degree = degree  # Use degree from BSpline initialization
        pspline.n_min = data["n_min"]
        pspline.lambda_smoothing = data["lambda_smoothing"]
        pspline.kappa_penalty = data["kappa_penalty"]
        pspline.maxiter = data["maxiter"]

        # Set fitted state
        pspline.bp = np.array(data["breakpoints"])
        pspline.fit_bnds = np.array(data["fit_bounds"])

        # Set training metrics
        if data["training_metrics"] is not None:
            pspline.training_metrics = BaselineMetricsFromDict(data["training_metrics"])
        else:
            pspline.training_metrics = None

        return pspline

    @classmethod
    def from_json(cls, json_str: str) -> "DailyPSpline":
        """Deserialize DailyPSpline from JSON string.

        Parameters
        ----------
        json_str : str
            JSON string containing DailyPSpline state.

        Returns
        -------
        DailyPSpline
            Reconstructed DailyPSpline instance.
        """
        return cls.from_dict(json.loads(json_str))

