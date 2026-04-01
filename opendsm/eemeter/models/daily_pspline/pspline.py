from __future__ import annotations

import json
import warnings
from functools import lru_cache
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from scipy.interpolate import BSpline

from opendsm.eemeter.models.daily_pspline.spline_knots import Knots

from opendsm.common.stats.adaptive_loss import adaptive_weights, kernel_adaptive_weights, KernelWeightCache
from opendsm.common.metrics import BaselineMetrics, BaselineMetricsFromDict
from opendsm.eemeter.models.daily.utilities.opt_settings import OptimizationSettings
from opendsm.eemeter.models.daily.optimize import NLoptOptimizer
from opendsm.eemeter.models.daily.utilities.selection_criteria import selection_criteria


# Ridge added to every LHS matrix to reduce ill-conditioning.
# Typical B'B diagonal entries are O(n/m) ≈ 15-20; 1e-4 is ~5e-6 relative —
# negligible for the fit but prevents singular fallbacks to lstsq.
_EPS_RIDGE = 1e-4

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


class _BasePSpline:
    """Penalized B-spline building block with precomputed, bp-independent matrices.

    Caches the design matrix B, difference matrices D1/D3, and the base Gram
    matrix for a fixed padded knot vector.  Reused across multiple bp evaluations
    (e.g., during breakpoint optimization) to avoid redundant computation.

    Parameters
    ----------
    x : ndarray
        Input coordinates (standardized).
    y : ndarray
        Response values (standardized).
    padded_knots : ndarray
        Full knot vector including boundary-repetition padding.
    k : int
        B-spline degree.
    weights : ndarray or None
        Base per-observation weights (without TIDD upweighting).
    lambda_smoothing : float
        Third-derivative smoothing penalty weight.
    bc_type : str or None
        Boundary condition type ("natural", "clamped", or None).
    kappa : float
        Penalty weight used for boundary conditions and monotonicity.
    """

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_knots(internal_knots: np.ndarray, bspline_degree: int) -> np.ndarray:
        """Add repeated boundary knots for B-spline boundary stability.

        Adds ``bspline_degree`` repeated knots at each end (multiplicity k+1),
        so that exactly one basis function is non-zero at each boundary.
        Ensures the padded vector has at least ``2*k + 2`` elements (the
        minimum for ``BSpline.design_matrix``).

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

    @staticmethod
    @lru_cache(maxsize=16)
    def _cached_first_diff(n: int) -> np.ndarray:
        """First-order difference matrix of size (n-1, n). Cached by n."""
        result = np.diff(np.eye(n), n=1, axis=0)
        result.flags.writeable = False
        return result

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        padded_knots: np.ndarray,
        k: int,
        weights: Optional[np.ndarray],
        lambda_smoothing: float,
        bc_type: Optional[str],
        kappa: float,
        B: Optional[np.ndarray] = None,
    ):
        self.x = x
        self.y = y
        self.k = k
        self.padded_knots = padded_knots
        self.n_base = len(padded_knots) - k - 1

        self.B = BSpline.design_matrix(x=x, t=padded_knots, k=k, extrapolate=True).toarray() if B is None else B

        need_D2 = (bc_type == "natural" and k >= 2)
        need_D3 = (lambda_smoothing > 0)
        D1, D2, D3 = _BasePSpline._difference_matrices(
            tuple(self.padded_knots), self.k, self.n_base,
            need_D2=need_D2, need_D3=need_D3,
        )
        self.D1 = D1
        self.D1T = D1.T
        self.n_deriv = D1.shape[0]

        D3_smoothing = lambda_smoothing * D3.T @ D3 if lambda_smoothing > 0 and D3 is not None else 0.0

        boundary_penalty = 0.0
        if bc_type == "clamped" and k >= 2:
            bm = np.vstack([D1[0, :], D1[-1, :]])
            boundary_penalty = kappa * bm.T @ bm
        elif bc_type == "natural" and k >= 2 and D2 is not None:
            bm = np.vstack([D2[0, :], D2[-1, :]])
            boundary_penalty = kappa * bm.T @ bm

        self._penalty_sum = D3_smoothing + boundary_penalty
        self._ridge = _EPS_RIDGE * np.eye(self.n_base)

        # Preallocated solve-loop buffers (shared across solve calls)
        self._D1_weighted_buf = np.empty_like(D1)
        self._kappa_V_buf = np.empty(self.n_deriv, dtype=float)
        self._kVd_buf = np.empty(self.n_deriv, dtype=float)
        self._lhs_buf = np.empty((self.n_base, self.n_base), dtype=float)

        # Precompute base Gram matrix for given weights (no TIDD upweighting)
        self._base_weights = weights
        self._A_base, self._BTy_base = self._gram(weights)

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=256)
    def _difference_matrices(padded_knots_tuple: tuple, k: int, n_base: int,
                             need_D2: bool = True, need_D3: bool = True):
        """Cached difference matrices keyed on knot vector, degree, and basis count.

        Results are reused across BIC-scan candidates that share the same knot
        configuration, avoiding redundant O(n²) matrix construction.

        When ``need_D2`` or ``need_D3`` is False, the corresponding matrix is
        not computed (returned as None), avoiding the chain matrix
        multiplication for unused higher-order differences.
        """
        knots = np.asarray(padded_knots_tuple)

        def lag_diff(arr, lag):
            return arr[lag:] - arr[:-lag]

        def _compute_D(i, D_prev):
            if k > i:
                a = 1 / (k - i)
                diag_vals = a * lag_diff(knots[i + 1:-i - 1], lag=k - i)
                diag_vals = np.where(diag_vals == 0, 1.0, diag_vals)
                fd = _BasePSpline._cached_first_diff(n_base - i)
                _D = fd / diag_vals[:, np.newaxis]
                return _D if D_prev is None else _D @ D_prev
            else:
                return np.diff(np.eye(n_base), n=i + 1, axis=0)

        D1 = _compute_D(0, None)
        D1.flags.writeable = False

        D2 = None
        if need_D2 or need_D3:  # D3 requires D2 as input
            D2 = _compute_D(1, D1)
            D2.flags.writeable = False

        D3 = None
        if need_D3:
            D3 = _compute_D(2, D2)
            D3.flags.writeable = False

        return D1, D2, D3

    @staticmethod
    @lru_cache(maxsize=256)
    def _derivative_zones(padded_knots_tuple: tuple, k: int, bp0: float, bp1: float) -> dict:
        """Cached derivative zone assignment, keyed on (knot vector, degree, bp0, bp1).

        Uses inline scalar comparisons instead of np.isclose to avoid per-call
        numpy dispatch overhead.
        """
        knots = np.asarray(padded_knots_tuple)
        n_deriv = len(knots) - k - 2

        empty = np.array([], dtype=int)
        empty.flags.writeable = False

        rtol = 1e-4
        atol = 1e-12
        single_bp = abs(bp1 - bp0) <= rtol * (abs(bp1) + atol)

        if single_bp and abs(knots[-1] - bp1) <= rtol * (abs(bp1) + atol):
            full = np.arange(n_deriv)
            full.flags.writeable = False
            return {"hdd": full, "tidd": empty, "cdd": empty}

        if single_bp and abs(knots[0] - bp0) <= rtol * (abs(bp0) + atol):
            full = np.arange(n_deriv)
            full.flags.writeable = False
            return {"hdd": empty, "tidd": empty, "cdd": full}

        left = knots[1:1 + n_deriv]
        right = knots[k + 1:k + 1 + n_deriv]

        zones = {
            "hdd": np.flatnonzero(right <= bp0),
            "tidd": np.flatnonzero((left < bp1) & (right > bp0)),
            "cdd": np.flatnonzero(left >= bp1),
        }

        for v in zones.values():
            v.flags.writeable = False
        return zones

    def _tidd_weights(
        self,
        weights: Optional[np.ndarray],
        bp: np.ndarray,
        weight_factor: float = 50,
    ) -> np.ndarray:
        """Upweight observations in the TIDD zone to enforce flatness.

        Parameters
        ----------
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
        x = self.x
        tidd_idx = np.flatnonzero((x > bp[0]) & (x < bp[1]))

        if tidd_idx.size == 0:
            return weights

        if weights is None:
            weights = np.ones_like(x)
        else:
            weights = weights.copy()  # avoid compounding across repeated fit calls

        weights[tidd_idx] *= weight_factor
        return weights

    def _gram(self, weights: Optional[np.ndarray]):
        """Compute LHS Gram matrix and B'y for given weights."""
        if weights is None:
            A = self.B.T @ self.B + self._penalty_sum + self._ridge
            BTy = self.B.T @ self.y
        else:
            w_sq = np.square(weights)
            A = self.B.T @ (self.B * w_sq[:, np.newaxis]) + self._penalty_sum + self._ridge
            BTy = self.B.T @ (self.y * w_sq)
        return A, BTy

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _fit_iterative(
        self,
        A: np.ndarray,
        BTy: np.ndarray,
        D1_zone: dict,
        kappa: float,
        maxiter: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run iterative monotonicity-constrained least-squares solve.

        Solves ``(A + κD1'VD1)α = BTy + κD1'Vδ`` iteratively until the
        active-set matrix V converges or *maxiter* is reached.

        The first iteration is unrolled as a cheap unconstrained solve (V=0,
        no penalty matrix construction needed).  If the unconstrained solution
        already satisfies all monotonicity constraints, returns immediately
        without entering the iterative loop.

        Parameters
        ----------
        A : ndarray
            Base LHS matrix (Gram + smoothing + boundary penalties).
        BTy : ndarray
            Base RHS vector (B'y or B'W²y).
        D1_zone : dict
            Zone index arrays {"hdd", "tidd", "cdd"} from _BasePSpline._derivative_zones.
        kappa : float
            Monotonicity penalty weight.
        maxiter : int
            Maximum iterations before issuing MaxIterationWarning.

        Returns
        -------
        coefs : ndarray
            Fitted spline coefficients.
        V : ndarray
            Converged active-set diagonal (1 where monotonicity constraint is
            active, 0 elsewhere).  Used by callers for effective-degrees-of-freedom
            computation and constrained-derivative breakpoint extraction.
        """
        D1 = self.D1
        D1T = self.D1T
        D1_weighted_buf = self._D1_weighted_buf
        kappa_V_buf = self._kappa_V_buf
        kVd_buf = self._kVd_buf
        lhs_buf = self._lhs_buf
        n_deriv = self.n_deriv

        delta = np.zeros(n_deriv, dtype=float)
        V = np.zeros(n_deriv, dtype=int)
        V_new = np.zeros(n_deriv, dtype=int)

        hdd_idx = D1_zone["hdd"]
        tidd_idx = D1_zone["tidd"]
        cdd_idx = D1_zone["cdd"]
        has_hdd = len(hdd_idx) > 0
        has_tidd = len(tidd_idx) > 0
        has_cdd = len(cdd_idx) > 0

        # --- Iteration 0: unconstrained (V=0, skip penalty construction) ---
        try:
            coefs = np.linalg.solve(A, BTy)
        except np.linalg.LinAlgError:
            coefs, _, _, _ = np.linalg.lstsq(A, BTy, rcond=None)

        deriv = D1 @ coefs

        if has_hdd:
            V[hdd_idx] = deriv[hdd_idx] > delta[hdd_idx]
        if has_tidd:
            V[tidd_idx] = deriv[tidd_idx] != delta[tidd_idx]
        if has_cdd:
            V[cdd_idx] = deriv[cdd_idx] < delta[cdd_idx]

        if np.sum(V) == 0:
            return coefs, V  # unconstrained solution satisfies all constraints

        # --- Iterations 1+: penalized solve with preallocated buffers ---
        for i in range(1, maxiter):
            np.multiply(kappa, V, out=kappa_V_buf)

            np.multiply(kappa_V_buf[:, np.newaxis], D1, out=D1_weighted_buf)
            np.dot(D1T, D1_weighted_buf, out=lhs_buf)
            lhs_buf += A
            np.multiply(kappa_V_buf, delta, out=kVd_buf)
            rhs = BTy + D1T @ kVd_buf

            try:
                coefs = np.linalg.solve(lhs_buf, rhs)
            except np.linalg.LinAlgError:
                coefs, _, _, _ = np.linalg.lstsq(lhs_buf, rhs, rcond=None)

            deriv = D1 @ coefs
            V_new.fill(0)

            if has_hdd:
                V_new[hdd_idx] = deriv[hdd_idx] > delta[hdd_idx]
            if has_tidd:
                V_new[tidd_idx] = deriv[tidd_idx] != delta[tidd_idx]
            if has_cdd:
                V_new[cdd_idx] = deriv[cdd_idx] < delta[cdd_idx]

            if np.array_equal(V, V_new):
                break

            V, V_new = V_new, V

        else:
            warnings.warn(
                "Max iteration reached. The results are not reliable.", MaxIterationWarning
            )

        return coefs, V

    def _fit(
        self,
        bp: np.ndarray,
        base_weights: Optional[np.ndarray],
        kappa: float,
        maxiter: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit spline coefficients for the given breakpoints.

        Applies TIDD upweighting for observations inside ``bp``, then
        delegates to :meth:`_fit_iterative` with precomputed matrices.

        Parameters
        ----------
        bp : ndarray
            Breakpoints [lower, upper] defining the TIDD zone.
        base_weights : ndarray or None
            Per-observation weights *before* TIDD upweighting.
        kappa : float
            Monotonicity penalty weight.
        maxiter : int
            Maximum solver iterations.

        Returns
        -------
        coefs : ndarray
            Fitted spline coefficients.
        V : ndarray
            Converged active-set diagonal from the monotonicity solve.
        """
        if bp[1] - bp[0] > 0:
            eff_w = self._tidd_weights(base_weights, bp)
        else:
            eff_w = base_weights

        # Reuse precomputed base matrices when weights are unchanged
        if eff_w is base_weights and base_weights is self._base_weights:
            A, BTy = self._A_base, self._BTy_base
        else:
            A, BTy = self._gram(eff_w)

        D1_zone = _BasePSpline._derivative_zones(
            tuple(self.padded_knots), self.k, float(bp[0]), float(bp[1])
        )
        return self._fit_iterative(A, BTy, D1_zone, kappa, maxiter)

    # ------------------------------------------------------------------
    # Public factory
    # ------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
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

        padded_knots = cls._pad_knots(knots, bspline_degree)
        instance = cls(
            x, y, padded_knots, bspline_degree, weights,
            lambda_smoothing, bc_type, kappa_penalty,
        )
        coefs, _V = instance._fit(bp, weights, kappa_penalty, maxiter)

        return padded_knots, coefs


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


def _estimate_bp_from_segmented_regression(x, y, n_min=5, grid_size=50):
    """Estimate breakpoints by scanning a 3-piece constrained linear model.

    Fits the model: decreasing line (HDD) + constant (TIDD) + increasing
    line (CDD) for each candidate (bp0, bp1) on a coarse grid, using
    precomputed cumulative statistics for O(1) per evaluation.

    This directly optimizes the functional form used by degree-1 P-splines
    with ``knot_count_max=0``, giving a better initial estimate than the
    smoothing-spline derivative walk.

    Parameters
    ----------
    x : ndarray
        Sorted independent variable (standardized temperatures).
    y : ndarray
        Dependent variable (standardized energy).
    n_min : int
        Minimum data points per zone.
    grid_size : int
        Number of candidate values per breakpoint dimension.

    Returns
    -------
    bp : ndarray
        Estimated breakpoints [lower, upper].
    """
    n = len(x)
    if n < 3 * n_min:
        return np.array([x[n // 3], x[2 * n // 3]])

    # Precompute cumulative statistics for O(1) per-segment regression
    cum_n = np.arange(1, n + 1, dtype=float)
    cum_x = np.cumsum(x)
    cum_y = np.cumsum(y)
    cum_xx = np.cumsum(x * x)
    cum_xy = np.cumsum(x * y)
    cum_yy = np.cumsum(y * y)
    total_x = cum_x[-1]
    total_y = cum_y[-1]
    total_xx = cum_xx[-1]
    total_xy = cum_xy[-1]
    total_yy = cum_yy[-1]

    def _seg_sse(i_start, i_end, constrained_sign=None):
        """SSE of a linear fit on x[i_start:i_end+1].

        constrained_sign: None=unconstrained, -1=slope<=0, +1=slope>=0.
        If constraint is violated, fits a constant (slope=0) instead.
        """
        if i_end < i_start:
            return 0.0
        nn = i_end - i_start + 1
        if nn < 2:
            return 0.0

        # Cumulative sums for the segment [i_start, i_end]
        if i_start == 0:
            sx = cum_x[i_end]
            sy = cum_y[i_end]
            sxx = cum_xx[i_end]
            sxy = cum_xy[i_end]
            syy = cum_yy[i_end]
        else:
            sx = cum_x[i_end] - cum_x[i_start - 1]
            sy = cum_y[i_end] - cum_y[i_start - 1]
            sxx = cum_xx[i_end] - cum_xx[i_start - 1]
            sxy = cum_xy[i_end] - cum_xy[i_start - 1]
            syy = cum_yy[i_end] - cum_yy[i_start - 1]

        Sxx = sxx - sx * sx / nn
        Sxy = sxy - sx * sy / nn
        Syy = syy - sy * sy / nn

        if Sxx < 1e-12:
            return max(0.0, Syy)

        slope = Sxy / Sxx

        # Check monotonicity constraint
        if constrained_sign is not None:
            if constrained_sign < 0 and slope > 0:
                return max(0.0, Syy)  # constant fit
            if constrained_sign > 0 and slope < 0:
                return max(0.0, Syy)  # constant fit

        return max(0.0, Syy - Sxy * Sxy / Sxx)

    # Grid of candidate breakpoint indices
    # Use data indices spaced across the range, ensuring n_min at boundaries
    idx_candidates = np.linspace(n_min, n - n_min - 1, grid_size, dtype=int)
    idx_candidates = np.unique(idx_candidates)

    best_sse = np.inf
    best_i = best_j = n // 3

    for i in idx_candidates:
        sse_hdd = _seg_sse(0, i - 1, constrained_sign=-1)
        for j in idx_candidates:
            if j <= i:
                continue
            if j - i < n_min:
                continue

            sse_tidd = _seg_sse(i, j - 1, constrained_sign=None)
            sse_cdd = _seg_sse(j, n - 1, constrained_sign=+1)
            total_sse = sse_hdd + sse_tidd + sse_cdd

            if total_sse < best_sse:
                best_sse = total_sse
                best_i = i
                best_j = j

    return np.array([x[best_i], x[best_j]])


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


def _clipped_std(values: np.ndarray, clip_val: float = 1e-6) -> float:
    """Compute standard deviation with clipping to avoid near-zero values.

    Parameters
    ----------
    values : array-like
        Input values to compute standard deviation on.
    clip_val : float
        Minimum allowed standard deviation to prevent numerical instability.

    Returns
    -------
    std : float
        Standard deviation of the input values, clipped to at least clip_val.
    """
    std = np.std(values)

    if std < clip_val:
        return 1.0

    return std


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
        bspline_degree: int = 3,
        bc_type: Optional[str] = "natural",
        n_min: int = 5,
        lambda_smoothing: float = 0.0,
        kappa_penalty: float = 1e9,
        maxiter: int = 100,
        adaptive_iterations: int = 10,
        zone_knot_count_max: Optional[int] = None,
        allow_heating_zone: bool = True,
        allow_cooling_zone: bool = True,
        zone_criteria: str = "bic",
        zone_penalty_multiplier: float = 1.0,
        zone_penalty_power: float = 1.0,
        regularization_alpha: float = 0.01,
        regularization_percent_lasso: float = 1.0,
        freeze_bp_on_convergence: bool = False,
        max_weight_iterations: int = 2,
        zone_knot_scan: bool = True,
    ):
        self.bspline_degree = bspline_degree
        self.bc_type = bc_type
        self.n_min = n_min
        self.lambda_smoothing = lambda_smoothing
        self.kappa_penalty = kappa_penalty
        self.maxiter = maxiter
        self.adaptive_iterations = adaptive_iterations
        self.zone_knot_count_max = zone_knot_count_max
        self.allow_heating_zone = allow_heating_zone
        self.allow_cooling_zone = allow_cooling_zone
        self.zone_criteria = zone_criteria
        self.zone_penalty_multiplier = zone_penalty_multiplier
        self.zone_penalty_power = zone_penalty_power

        # BP regularization: penalizes bp far from data bounds to maximize TIDD width
        self.regularization_alpha = regularization_alpha
        self.regularization_percent_lasso = regularization_percent_lasso

        # Skip re-optimizing bp in later iterations when it has already converged
        self.freeze_bp_on_convergence = freeze_bp_on_convergence

        # Adaptive weight iteration control
        self.max_weight_iterations = max_weight_iterations

        # When False, skip the BIC knot-count scan and use maximum knot counts directly
        self.zone_knot_scan = zone_knot_scan

        self.weights_alpha = [2.0]

    def _set_bp(
        self,
        x,
        y,
        bp,
        weights=None,
        zone_knot_count=10,
        algorithm="nlopt_direct",
        degree=None,
    ):
        """Optimize breakpoint positions via simultaneous global optimization.

        Both breakpoints are optimized simultaneously using NLopt DIRECT.
        Breakpoints are parameterized as normalized cumulative fractions of
        the data range, ensuring bp[0] <= bp[1].

        The knot vector is fixed from the initial bp estimate so that the
        design matrix B and difference matrices D can be precomputed once
        and reused across all objective evaluations.  Only the zone
        assignments and TIDD weights (which depend on bp) are recomputed
        per evaluation.

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

        Returns
        -------
        bp : ndarray
            Optimized breakpoints [lower, upper].
        psp : _BasePSpline
            Spline matrices built on the final optimized knot configuration.
        """
        x_min = x[0]
        x_max = x[-1]
        x_range = x_max - x_min

        N = len(x)
        # Small segments (e.g. wd/we splits with ~130 points) have noisier
        # objective surfaces — give the active-set solve more iterations so
        # bp evaluations are more stable.
        if N <= 150:
            inner_maxiter = min(15, self.maxiter) if self.maxiter else 15
        else:
            inner_maxiter = min(5, self.maxiter) if self.maxiter else 5
        lasso_a = self.regularization_percent_lasso * self.regularization_alpha
        ridge_a = (1 - self.regularization_percent_lasso) * self.regularization_alpha
        x_bnds = np.array([x_min, x_max])
        kappa = self.kappa_penalty
        k = degree if degree is not None else self.bspline_degree
        has_reg = self.regularization_alpha != 0

        bp_init = np.asarray(bp, dtype=float)
        _knots_obj = self.knots
        _lambda = self.lambda_smoothing
        _bc = self.bc_type
        _n_min = self.n_min

        def bp_penalty(trial_bp, wrmse):
            penalty = trial_bp - x_bnds
            penalty *= wrmse / x_range

            total = 0.0
            if lasso_a != 0:
                total += lasso_a * np.linalg.norm(penalty, 1)
            if ridge_a != 0:
                total += ridge_a * np.linalg.norm(penalty, 2)

            return total

        # Build reduced parameter space based on which zones are free to optimize.
        # Parameterization: X0 = (bp[0]-x_min)/x_range, X1 = (bp[1]-bp[0])/(x_max-bp[0])
        # bp[0] = X0*x_range + x_min
        # bp[1] = X1*(x_max - bp[0]) + bp[0]   — ensures bp[0] <= bp[1] <= x_max by construction
        allow_hdd = self.allow_heating_zone
        allow_cdd = self.allow_cooling_zone

        remaining = x_max - bp_init[0]
        x0_full = np.clip([
            (bp_init[0] - x_min) / x_range,
            (float(bp_init[1]) - bp_init[0]) / remaining if remaining > 0 else 0.0,
        ], 0.0, 1.0)

        bnds_full = np.array([(0.0, 1.0), (0.0, 1.0)])

        if allow_hdd and allow_cdd:
            to_full = lambda X: X
            x0_opt = x0_full
            bnds_opt = bnds_full
        elif allow_hdd:   # cooling fixed: bp[1] = x_max  →  X1 = 1.0
            to_full = lambda X: np.array([X[0], 1.0])
            x0_opt = x0_full[:1]
            bnds_opt = bnds_full[:1]
        elif allow_cdd:   # heating fixed: bp[0] = x_min  →  X0 = 0
            to_full = lambda X: np.array([0.0, X[0]])
            x0_opt = x0_full[1:]
            bnds_opt = bnds_full[1:]
        else:             # both fixed: no free parameters, skip optimizer
            to_full = None
            x0_opt = None
            bnds_opt = None

        def _X_to_bp(X):
            bp0 = X[0] * x_range + x_min
            bp1 = X[1] * (x_max - bp0) + bp0
            return np.array([bp0, bp1])

        def objective(X_free, grad=[]):
            trial_bp = _X_to_bp(to_full(X_free))

            # Recompute knots for this bp so the B-spline basis matches
            # the zone boundaries.
            trial_knots = _knots_obj.get_internal_knots(
                bp=trial_bp, n_knots=zone_knot_count, n_min=_n_min,
            )
            trial_padded = _BasePSpline._pad_knots(trial_knots, k)
            psp = _BasePSpline(
                x, y, trial_padded, k, weights, _lambda, _bc, kappa,
            )

            coefs, _V = psp._fit(trial_bp, weights, kappa, inner_maxiter)

            resid = psp.B @ coefs - y
            sse = np.sum(resid ** 2)
            loss = sse / N
            wrmse = np.sqrt(loss)

            if has_reg:
                loss += bp_penalty(trial_bp, wrmse)

            return loss

        if algorithm == "nlopt_direct":
            adaptive_budget = int(np.clip(N // 10, 100, 400))
        else:
            adaptive_budget = int(np.clip(N // 10, 30, 100))

        opt_settings = OptimizationSettings(
            algorithm=algorithm,
            initial_step=0.025,
            stop_criteria_type="iteration maximum",
            stop_criteria_value=adaptive_budget,
            x_tol_rel=1e-3,
            f_tol_rel=1e-3,
        )

        if x0_opt is not None:
            optimizer = NLoptOptimizer(objective, x0_opt, bnds_opt, opt_settings)
            result = optimizer.run()
            bp = _X_to_bp(to_full(result.x))
        else:
            bp = np.array([x_min, x_max])

        # Absorb HDD/CDD zones with fewer than n_min data points into TIDD
        n_hdd = int(x.searchsorted(bp[0], side='left'))
        if 0 < n_hdd < self.n_min:
            bp[0] = x_min
            n_hdd = 0

        n_cdd = N - int(x.searchsorted(bp[1], side='right'))
        if 0 < n_cdd < self.n_min:
            bp[1] = x_max
            n_cdd = 0

        # If fewer than n_min data points between breakpoints, merge them
        n_tidd = N - n_hdd - n_cdd
        if n_tidd < self.n_min:
            avg_bp = np.mean(bp)
            bp[:] = avg_bp

        # Enforce disabled zones: pin breakpoints to data bounds
        if not self.allow_heating_zone:
            bp[0] = x_min
        if not self.allow_cooling_zone:
            bp[1] = x_max

        # Build final _BasePSpline with the optimized bp's knot configuration
        final_knots = self.knots.get_internal_knots(bp=bp, n_knots=zone_knot_count, n_min=self.n_min)
        padded_final = _BasePSpline._pad_knots(final_knots, k)
        psp = _BasePSpline(x, y, padded_final, k, weights, self.lambda_smoothing, self.bc_type, kappa)

        return bp, psp

    # ------------------------------------------------------------------
    # Fit helpers (called by fit)
    # ------------------------------------------------------------------

    def _zone_knot_scan(self, bp_std, hdd_max, cdd_max, candidate_fn):
        """Two-axis grid search over (n_hdd, n_cdd) knot counts.

        Exploits the near-independence of HDD and CDD zones (they fit
        different data segments) to search each axis separately:

        1. Scan HDD count 0→hdd_max with CDD fixed at cdd_max → best_hdd
        2. Scan CDD count 0→cdd_max with HDD fixed at best_hdd → best_cdd
        3. Verify: re-scan HDD with CDD=best_cdd — if unchanged, done;
           otherwise repeat once more.

        Typical evaluations: hdd_max + cdd_max + hdd_max ≈ 15-25 vs
        hdd_max × cdd_max ≈ 20-80 for an exhaustive grid.  Gives the same
        result when the axes are independent (which they nearly are — the
        only coupling is the shared BIC penalty denominator).

        Parameters
        ----------
        bp_std : ndarray
            Current breakpoints (standardised).
        hdd_max : int
            Maximum HDD knot count to try.
        cdd_max : int
            Maximum CDD knot count to try.
        candidate_fn : callable
            ``candidate_fn(bp, n_hdd, n_cdd) -> (score, psp, coefs)``

        Returns
        -------
        psp : _BasePSpline
        coefs : ndarray
        """
        hdd_max = min(hdd_max, 8)
        cdd_max = min(cdd_max, 8)

        best_score = np.inf
        best_psp = best_coefs = None

        def _scan_axis(fixed_axis, fixed_val, scan_max):
            """Scan one axis, return best count and its (score, psp, coefs)."""
            nonlocal best_score, best_psp, best_coefs
            best_count = 0
            for count in range(scan_max + 1):
                if fixed_axis == "cdd":
                    score, psp, coefs = candidate_fn(bp_std, count, fixed_val)
                else:
                    score, psp, coefs = candidate_fn(bp_std, fixed_val, count)
                if score < best_score:
                    best_score = score
                    best_psp, best_coefs = psp, coefs
                    best_count = count
            return best_count

        # Pass 1: scan HDD with CDD at max
        best_hdd = _scan_axis("cdd", cdd_max, hdd_max)

        # Pass 2: scan CDD with HDD at best
        best_cdd = _scan_axis("hdd", best_hdd, cdd_max)

        # Pass 3: verify HDD hasn't changed given best CDD
        prev_hdd = best_hdd
        best_hdd = _scan_axis("cdd", best_cdd, hdd_max)

        if best_hdd != prev_hdd:
            # Coupling mattered — re-scan CDD once more
            best_cdd = _scan_axis("hdd", best_hdd, cdd_max)

        return best_psp, best_coefs

    @staticmethod
    def _bp_from_constrained_derivative(psp, coefs, x_std):
        """Extract breakpoints from the monotone-constrained fit's derivative.

        The active-set solver forces ``D1 @ coefs ≈ 0`` in the TIDD zone.
        Identifies the leftmost and rightmost zero-derivative knot spans to
        recover the breakpoint region, then maps to continuous x-positions
        via knot-span midpoints.

        This provides a *coarse basin localization* — not a precise bp — to
        warm-start the next iteration's local optimizer (Sbplx).  Resolution
        is limited by knot spacing, but the signal is physically grounded:
        it reflects the constrained fit's actual flat zone rather than an
        unconstrained smoothing spline's derivative.

        Parameters
        ----------
        psp : _BasePSpline
            Fitted spline matrices.
        coefs : ndarray
            Converged spline coefficients.
        x_std : ndarray
            Standardized x data (sorted).

        Returns
        -------
        bp_refined : ndarray or None
            Refined breakpoints ``[lower, upper]`` in standardized space.
            Returns ``None`` if the derivative structure is degenerate
            (entirely flat or no flat region at all).
        """
        d1 = psp.D1 @ coefs
        n_deriv = len(d1)
        if n_deriv == 0:
            return None
        knots = psp.padded_knots
        k = psp.k

        # Scale threshold to derivative magnitude
        d1_scale = np.max(np.abs(d1))
        if d1_scale < 1e-12:
            return None  # entirely flat — no bp to extract

        tol = 1e-4 * d1_scale
        is_flat = np.abs(d1) < tol

        if not np.any(is_flat):
            # No flat region — model is monotone throughout.
            # Return collapsed bp at the minimum of the fitted curve.
            y_eval = psp.B @ coefs
            min_idx = np.argmin(y_eval)
            mid = x_std[min_idx]
            return np.array([mid, mid])

        # Map derivative indices to knot-span midpoints (continuous positions)
        span_left = knots[1:1 + n_deriv]
        span_right = knots[k + 1:k + 1 + n_deriv]
        span_mids = 0.5 * (span_left + span_right)

        flat_indices = np.flatnonzero(is_flat)
        bp_lower = span_mids[flat_indices[0]]
        bp_upper = span_mids[flat_indices[-1]]

        # Clamp to data range
        bp_lower = np.clip(bp_lower, x_std[0], x_std[-1])
        bp_upper = np.clip(bp_upper, bp_lower, x_std[-1])

        return np.array([bp_lower, bp_upper])

    @staticmethod
    def _effective_df(psp, V, kappa, w_sq):
        """Compute effective degrees of freedom: ``tr(H)``.

        Uses the identity ``edf = tr((LHS)⁻¹ A)`` where both matrices are
        m×m (number of basis functions), making this O(m³) — negligible for
        typical m ≈ 15-25.

        The LHS includes the converged monotonicity penalty ``κD1'VD1`` so
        that constrained coefficients are correctly counted as having reduced
        freedom.

        Parameters
        ----------
        psp : _BasePSpline
            Fitted spline matrices.
        V : ndarray
            Converged active-set diagonal from ``_fit_iterative``.
        kappa : float
            Monotonicity penalty weight.
        w_sq : ndarray
            Squared observation weights, shape ``(n,)``.

        Returns
        -------
        edf : float
            Effective degrees of freedom (trace of the hat matrix).
        """
        # Data Gram matrix (without penalties)
        A = psp.B.T @ (psp.B * w_sq[:, np.newaxis])

        # Full LHS including all penalties
        mono_penalty = kappa * (psp.D1.T @ (psp.D1 * V.astype(float)[:, np.newaxis]))
        LHS = A + psp._penalty_sum + psp._ridge + mono_penalty

        # edf = tr(LHS⁻¹ @ A), both m×m
        try:
            LHS_inv_A = np.linalg.solve(LHS, A)
        except np.linalg.LinAlgError:
            # Singular LHS (e.g., very large kappa with many active constraints).
            # Fall back to lstsq which handles rank-deficient systems.
            LHS_inv_A, _, _, _ = np.linalg.lstsq(LHS, A, rcond=None)
        return float(np.trace(LHS_inv_A))

    def _weighted_stats(self, y):
        """Compute w², sum(w²), and weighted total sum of squares for self.weights."""
        w_sq = self.weights ** 2
        sum_w_sq = float(np.sum(w_sq))
        y_wmean = float(np.dot(w_sq, y)) / sum_w_sq
        wtss = float(np.dot(w_sq, (y - y_wmean) ** 2))

        return w_sq, sum_w_sq, wtss

    def _score(self, wssr, wtss, N, edf):
        """Compute zone selection criterion using configured settings.

        Parameters
        ----------
        wssr : float
            Weighted sum of squared residuals.
        wtss : float
            Weighted total sum of squares.
        N : int
            Number of observations.
        edf : float
            Effective degrees of freedom (from ``_effective_df``).
        """
        return selection_criteria(
            wssr,
            wtss,
            N,
            edf,
            self.zone_criteria,
            self.zone_penalty_multiplier,
            self.zone_penalty_power,
        )

    def _knots_from_constrained_curvature(
        self,
        psp,
        coefs,
        bp_std,
        x_std,
        n_knots_hdd,
        n_knots_cdd,
        n_knots_tidd=10,
    ):
        """Place knots using curvature of a previously fitted constrained P-spline.

        Uses the same equi-curvature integral as Yeh (2020), but evaluated on
        the monotone-constrained spline rather than an unconstrained smoother.
        For HDD/CDD zones, knots concentrate where the constrained curve bends
        most (the elbows).  The TIDD zone gets uniform spacing since the curve
        is flat there by construction.

        Parameters
        ----------
        psp : _BasePSpline
            Previously fitted spline (provides knot vector and degree).
        coefs : ndarray
            Fitted spline coefficients.
        bp_std : ndarray
            Current breakpoints in standardized space.
        x_std : ndarray
            Standardized x data (sorted).
        n_knots_hdd, n_knots_cdd : int
            Target knot counts per zone.
        n_knots_tidd : int
            Number of uniform knots in the TIDD zone.

        Returns
        -------
        internal_knots : ndarray
            Full internal knot vector (without boundary padding).
        """
        spl = BSpline(psp.padded_knots, coefs, psp.k)
        x_lo, x_hi = x_std[0], x_std[-1]
        n_min = self.n_min

        def _curvature_knots(x_range_lo, x_range_hi, n_knots):
            """Place n_knots via equi-curvature integral on the given range."""
            if n_knots <= 0:
                return np.array([])

            n_eval = max(100, 20 * n_knots)
            xs = np.linspace(x_range_lo, x_range_hi, n_eval)

            # Use second derivative of the constrained spline as curvature proxy.
            # For degree < 2 the second derivative is zero; fall back to first.
            deriv_order = min(2, psp.k)
            if deriv_order == 0:
                curv = np.abs(spl(xs))
            else:
                d = spl.derivative(deriv_order)
                curv = np.abs(d(xs))

            # Avoid zero-curvature regions collapsing the integral
            curv_max = np.max(curv)
            if curv_max > 0:
                curv = np.maximum(curv, 1e-10 * curv_max)
            else:
                curv = np.ones_like(curv)

            # Equi-curvature integral (Yeh 2020)
            dx = xs[1] - xs[0]
            cum = np.empty(len(curv))
            cum[0] = 0.0
            np.cumsum(0.5 * dx * (curv[:-1] + curv[1:]), out=cum[1:])

            targets = np.linspace(0, cum[-1], n_knots + 2)
            return np.interp(targets, cum, xs)

        # HDD zone
        hdd_knots = np.array([])
        n_hdd_data = int(x_std.searchsorted(bp_std[0], side='left'))
        if n_hdd_data >= n_min and n_knots_hdd > 0:
            n_knots_hdd = min(n_knots_hdd, n_hdd_data // n_min)
            hdd_knots = _curvature_knots(x_lo, bp_std[0], n_knots_hdd)
            hdd_knots = hdd_knots[hdd_knots < bp_std[0]]

        # TIDD zone — uniform, since curve is flat by construction
        if np.isclose(bp_std[0], bp_std[1]):
            tidd_knots = np.array([bp_std[0]])
        else:
            tidd_knots = np.linspace(bp_std[0], bp_std[1], n_knots_tidd + 2)

        # CDD zone
        cdd_knots = np.array([])
        n_cdd_data = len(x_std) - int(x_std.searchsorted(bp_std[1], side='right'))
        if n_cdd_data >= n_min and n_knots_cdd > 0:
            n_knots_cdd = min(n_knots_cdd, n_cdd_data // n_min)
            cdd_knots = _curvature_knots(bp_std[1], x_hi, n_knots_cdd)
            cdd_knots = cdd_knots[cdd_knots > bp_std[1]]

        return np.hstack([hdd_knots, tidd_knots, cdd_knots])

    def _fit_degree(self, x_std, y_std, bp_std, bp_provided, zone_knot_count, bic_maxiter, N, bp_opt_algo, degree):
        """Adaptive fitting loop for one bspline degree.

        Initialises ``self.knots`` for *degree*, then iterates: optimise bp →
        BIC knot scan → adaptive weight update, until convergence.

        After the first iteration, two refinements are applied:
        1. **bp warm-start**: The constrained fit's derivative reveals the
           actual flat zone; this warm-starts the next bp optimisation.
        2. **Curvature-based knot refinement** (degree ≥ 2): Knots are
           re-placed using the constrained spline's curvature rather than
           the unconstrained smoothing spline's, giving self-consistent
           placement.

        Mutates ``self.weights`` and ``self.knots``.

        Parameters
        ----------
        x_std, y_std : ndarray
            Standardised independent and dependent variables.
        bp_std : ndarray
            Initial breakpoints (standardised).
        bp_provided : bool
            If True, skip bp optimisation entirely.
        zone_knot_count : int
            Target knots per zone.
        bic_maxiter : int
            Max solver iterations used inside each candidate evaluation.
        N : int
            Number of observations.
        bp_opt_algo : str
            NLopt algorithm name for ``_set_bp``.
        degree : int
            B-spline degree for this iteration.

        Returns
        -------
        psp : _BasePSpline
        coefs : ndarray
        bp_std : ndarray
            Converged breakpoints.
        """
        # Skip the expensive robust smoothing spline when zone_knot_count=0
        # (no Yeh knot placement needed — get_internal_knots only returns bp
        # values and data boundaries).
        self.knots = Knots(
            x_std, y_std,
            w=self.weights,
            spline_interp_count=1000,
            spline_lambda=10,
            n_min=self.n_min,
            bspline_degree=degree,
            lambda_smoothing=self.lambda_smoothing,
            kappa_penalty=self.kappa_penalty,
            maxiter=self.maxiter,
            lightweight=(zone_knot_count == 0),
        )
        _B_cache: dict = {}
        prior_wrmse = np.inf
        prior_bp = None
        psp = coefs = V_final = None
        _knots_refined = False
        _prior_a_weights = None
        _weight_iters_remaining = self.max_weight_iterations
        _kernel_cache = KernelWeightCache(
            x_std, zone_knot_count,
            min_knot_spacing_pct=self.knots.min_knot_spacing_pct if hasattr(self.knots, 'min_knot_spacing_pct') else 0.025,
            n_eff_min=self.n_min,
        )

        saved_reg_alpha = self.regularization_alpha

        for iter_idx in range(self.adaptive_iterations):
            # Disable regularization after first iteration: adaptive weights
            # handle outlier robustness, so the regularizer is no longer needed
            # and would compete with the weights for bp placement authority.
            if iter_idx > 0:
                self.regularization_alpha = 0.0

            # First iteration: use the caller's algorithm (DIRECT for global
            # search with uniform weights).  Subsequent iterations: switch to
            # Sbplx (local refinement) so adaptive weights can nudge the bp
            # without sending it to a completely different basin.
            iter_algo = bp_opt_algo if iter_idx == 0 else "nlopt_sbplx"

            if not bp_provided and (prior_bp is None or not (
                self.freeze_bp_on_convergence
                and np.allclose(bp_std, prior_bp, atol=1e-5)
            )):
                bp_std, _ = self._set_bp(
                    x_std, y_std, bp_std,
                    weights=self.weights,
                    zone_knot_count=zone_knot_count,
                    algorithm=iter_algo,
                    degree=degree,
                )
            prior_bp = bp_std

            # Hoist weight-dependent constants so _candidate doesn't recompute per call
            # (~81 candidates × up to 10 outer iterations would be expensive).
            _w_sq, _sum_w_sq, _wtss = self._weighted_stats(y_std)
            _kappa = self.kappa_penalty

            def _candidate(bp, n_hdd, n_cdd, _w_sq=_w_sq, _wtss=_wtss):
                knots = self.knots.get_internal_knots(
                    bp=bp, n_knots=zone_knot_count, n_min=self.n_min,
                    n_knots_hdd=n_hdd, n_knots_cdd=n_cdd,
                )
                padded = _BasePSpline._pad_knots(knots, degree)
                padded_key = tuple(padded)
                B = _B_cache.get(padded_key)
                if B is None:
                    B = BSpline.design_matrix(x=x_std, t=padded, k=degree, extrapolate=True).toarray()
                    _B_cache[padded_key] = B
                cand_psp = _BasePSpline(
                    x_std, y_std, padded, degree,
                    self.weights, self.lambda_smoothing, self.bc_type, _kappa,
                    B=B,
                )
                cand_coefs, cand_V = cand_psp._fit(bp, self.weights, _kappa, bic_maxiter)
                resid = cand_psp.B @ cand_coefs - y_std
                wssr = float(np.dot(_w_sq, resid ** 2))
                edf = DailyPSpline._effective_df(cand_psp, cand_V, _kappa, _w_sq)
                score = self._score(wssr, _wtss, N, edf)
                return score, cand_psp, cand_coefs, cand_V

            hdd_max = (
                min(zone_knot_count, int(x_std.searchsorted(bp_std[0], side='left')) // self.n_min)
                if self.allow_heating_zone else 0
            )
            cdd_max = (
                min(zone_knot_count, (N - int(x_std.searchsorted(bp_std[1], side='right'))) // self.n_min)
                if self.allow_cooling_zone else 0
            )

            # Adapt _zone_knot_scan's candidate_fn to unpack the 4-tuple
            def _candidate_3tuple(bp, n_hdd, n_cdd, _w_sq=_w_sq, _wtss=_wtss):
                score, cand_psp, cand_coefs, cand_V = _candidate(bp, n_hdd, n_cdd, _w_sq, _wtss)
                # Stash V on the psp so we can retrieve it after the scan
                cand_psp._last_V = cand_V
                return score, cand_psp, cand_coefs

            if self.zone_knot_scan:
                psp, coefs = self._zone_knot_scan(bp_std, hdd_max, cdd_max, _candidate_3tuple)
            else:
                _, psp, coefs, _ = _candidate(bp_std, hdd_max, cdd_max)

            V_final = getattr(psp, '_last_V', None)
            if V_final is None:
                # Fallback: re-derive V from a full solve (only happens if zone_knot_scan=False)
                coefs, V_final = psp._fit(bp_std, self.weights, _kappa, bic_maxiter)

            residuals = (psp.B @ coefs - y_std).reshape(-1)
            wrmse = np.sqrt(np.dot(_w_sq, residuals ** 2) / _sum_w_sq)

            if prior_wrmse < np.inf and abs(wrmse - prior_wrmse) <= 1e-3 * prior_wrmse:
                break
            prior_wrmse = wrmse

            # --- Refinements after first converged iteration ---

            # (a) Warm-start bp from the constrained fit's derivative.
            #     Only apply if bp is already collapsed (single point) — on
            #     V-shaped meters where bp[0] < bp[1], the derivative walk
            #     would collapse the TIDD gap, destroying a good two-bp solution.
            if not bp_provided and np.abs(bp_std[1] - bp_std[0]) < 1e-6:
                bp_candidate = DailyPSpline._bp_from_constrained_derivative(psp, coefs, x_std)
                if bp_candidate is not None:
                    bp_std = bp_candidate

            # (b) Curvature-based knot refinement from constrained fit (degree >= 2 only).
            #     Runs once: re-places knots using the constrained spline's curvature
            #     instead of the unconstrained smoothing spline's.
            if not _knots_refined and degree >= 2 and iter_idx == 0:
                refined_knot_vector = self._knots_from_constrained_curvature(
                    psp, coefs, bp_std, x_std,
                    n_knots_hdd=hdd_max, n_knots_cdd=cdd_max,
                    n_knots_tidd=zone_knot_count,
                )
                # Replace the Knots object's cache so subsequent iterations use
                # the refined positions.  We create a minimal Knots with the
                # same robust smoothing spline but override get_internal_knots
                # to return the refined vector for the current bp.
                self.knots._knot_cache.clear()
                self.knots._refined_knots = refined_knot_vector
                self.knots._refined_bp = bp_std.copy()
                _B_cache.clear()  # padded knots will change
                _knots_refined = True

            # Compute kernel adaptive weights.  Skip if we've exceeded the
            # weight iteration cap.
            if _weight_iters_remaining > 0:
                a_weights, median_alpha = kernel_adaptive_weights(
                    x_std, residuals, _cache=_kernel_cache,
                )
                self.weights_alpha.append(median_alpha)

                self.weights = (self.weights * a_weights) if self.weights is not None else a_weights
                self.weights = _rescale_to_range(self.weights, new_min=1.0, new_max=100.0)
                _weight_iters_remaining -= 1

                if median_alpha == 2.0:
                    break  # weighted residuals are Gaussian, no further reweighting needed
                if _prior_a_weights is not None and np.max(np.abs(a_weights - _prior_a_weights)) < 0.1:
                    break  # weights stabilized
                _prior_a_weights = a_weights
            else:
                break

        self.regularization_alpha = saved_reg_alpha
        return psp, coefs, bp_std, V_final

    def fit(self, x, y, bp=None, weights=None, bspline_degree=None) -> "DailyPSpline":
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

        Returns
        -------
        self : DailyPSpline
            Fitted spline instance
        """
        self.weights_alpha = [2.0]

        zone_knot_count = self.zone_knot_count_max if self.zone_knot_count_max is not None else 10
        if zone_knot_count < 0:
            raise ValueError("`zone_knot_count` must be non-negative")

        if weights is None:
            weights = np.ones_like(x, dtype=float)
        else:
            weights = np.ascontiguousarray(weights, dtype=float)
            if np.any(weights <= 0):
                raise ValueError('Invalid vector of weights')
            if x.shape[0] != weights.shape[0]:
                raise ValueError(f'``x`` and ``weights`` should have the same length, but got {x.shape[0]} and {weights.shape[0]}')

        self.x = np.ascontiguousarray(x, dtype=float)
        self.y = np.ascontiguousarray(y, dtype=float)
        self.weights = weights

        self.x_mean = np.mean(self.x)
        self.x_std = _clipped_std(self.x)
        self.y_mean = np.mean(self.y)
        self.y_std = _clipped_std(self.y)

        x_std = (self.x - self.x_mean) / self.x_std
        y_std = (self.y - self.y_mean) / self.y_std

        # Initial bp estimate.  When bp is user-provided, convert to
        # standardized space.  Otherwise use the midpoint — DIRECT ignores
        # x0 and searches the full space regardless.
        if bp is not None:
            bp_std_init = (np.asarray(bp, dtype=float) - self.x_mean) / self.x_std
        else:
            mid = 0.5 * (x_std[0] + x_std[-1])
            bp_std_init = np.array([mid, mid])

        N = len(x_std)
        bic_maxiter = min(10, self.maxiter)
        bp_provided = bp is not None

        # Fit at the target degree using DIRECT for global bp search.
        # Previously a degree-0 surrogate was used to seed bp, but the
        # step-function loss landscape can have different optima than the
        # target degree (e.g., V-shaped meters where the linear vertex
        # differs from the step-function split point), causing Sbplx to
        # get trapped in the wrong basin.
        target_degree = bspline_degree if bspline_degree is not None else self.bspline_degree

        base_weights = self.weights.copy()
        bp_std = bp_std_init.copy()

        psp, coefs, bp_std, V_final = self._fit_degree(
            x_std, y_std, bp_std, bp_provided,
            zone_knot_count, bic_maxiter, N,
            "nlopt_direct", target_degree,
        )

        # BIC score on final adaptive weights, using edf
        w_sq, _, wtss = self._weighted_stats(y_std)
        wssr = float(np.dot(w_sq, (psp.B @ coefs - y_std).reshape(-1) ** 2))
        edf = DailyPSpline._effective_df(psp, V_final, self.kappa_penalty, w_sq)
        score = self._score(wssr, wtss, N, edf)

        best_psp, best_coefs = psp, coefs
        best_bp_std = bp_std.copy()
        best_knots = self.knots
        best_weights = self.weights.copy()

        self.knots = best_knots
        self.weights = best_weights
        super().__init__(t=best_psp.padded_knots, c=best_coefs, k=best_psp.k, extrapolate=True)
        self.bp = best_bp_std * self.x_std + self.x_mean
        self.fit_bnds = np.array([self.x[0], self.x[-1]])

        # Store effective degrees of freedom for use in split selection (model.py)
        self.edf = edf

        self.training_metrics = BaselineMetrics(
            df=pd.DataFrame({'observed': self.y, 'predicted': self.predict(self.x)}),
            num_model_params=best_psp.n_base - best_psp.k + 1,
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

            # Cache boundary values on first extrapolation call
            if not hasattr(self, '_y_lo'):
                x_lo_std = (fit_bnds[0] - self.x_mean) / self.x_std
                x_hi_std = (fit_bnds[1] - self.x_mean) / self.x_std
                self._y_lo = float(super().__call__(x_lo_std)) * self.y_std + self.y_mean
                self._y_hi = float(super().__call__(x_hi_std)) * self.y_std + self.y_mean
                if bc_type == "natural" and self.k >= 1:
                    deriv_spl = BSpline(self.t, self.c, self.k).derivative(1)
                    self._dy_lo = float(deriv_spl(x_lo_std)) * self.y_std / self.x_std
                    self._dy_hi = float(deriv_spl(x_hi_std)) * self.y_std / self.x_std

            if bc_type == "clamped":
                y = np.where(lo, self._y_lo, np.where(hi, self._y_hi, y))

            elif bc_type == "natural" and self.k >= 1:
                lo_extrap = self._y_lo + self._dy_lo * (x - fit_bnds[0])
                hi_extrap = self._y_hi + self._dy_hi * (x - fit_bnds[1])
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
            "training_metrics": None if self.training_metrics is None else self.training_metrics.model_dump(),
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
        # Validate and coerce via Pydantic schema
        schema = DailyPSplineSchema(**data)

        # Create new instance without calling __init__
        pspline = cls.__new__(cls)

        # Initialize as BSpline with knots and coefficients
        BSpline.__init__(
            pspline,
            t=np.array(schema.knots),
            c=np.array(schema.coefficients),
            k=schema.degree,
            extrapolate=schema.extrapolate,
        )

        # Set standardization parameters
        pspline.x_mean = schema.x_mean
        pspline.x_std = schema.x_std
        pspline.y_mean = schema.y_mean
        pspline.y_std = schema.y_std

        # Set DailyPSpline configuration
        pspline.bspline_degree = schema.degree
        pspline.n_min = schema.n_min
        pspline.lambda_smoothing = schema.lambda_smoothing
        pspline.kappa_penalty = schema.kappa_penalty
        pspline.maxiter = schema.maxiter

        # Set fitted state
        pspline.bp = np.array(schema.breakpoints)
        pspline.fit_bnds = np.array(schema.fit_bounds)

        # Set training metrics
        if schema.training_metrics is not None:
            pspline.training_metrics = BaselineMetricsFromDict(schema.training_metrics)
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