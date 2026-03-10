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

from opendsm.eemeter.models.daily_pspline.spline_knots import Knots

from opendsm.common.stats.adaptive_loss import adaptive_weights
from opendsm.common.metrics import BaselineMetrics, BaselineMetricsFromDict
from opendsm.eemeter.models.daily.utilities.opt_settings import OptimizationSettings
from opendsm.eemeter.models.daily.optimize import NLoptOptimizer


# Ridge added to every LHS matrix to reduce ill-conditioning.
# Typical B'B diagonal entries are O(n/m) ≈ 15-20; 1e-6 is negligible (~5e-8
# relative) but large enough to regularise near-singular configurations.
_EPS_RIDGE = 1e-6

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
        return np.diff(np.eye(n), n=1, axis=0)

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
    ):
        self.x = x
        self.y = y
        self.k = k
        self.padded_knots = padded_knots
        self.n_base = len(padded_knots) - k - 1

        self.B = BSpline.design_matrix(x=x, t=padded_knots, k=k, extrapolate=True).toarray()
        D1, D2, D3 = self._difference_matrices()
        self.D1 = D1
        self.D1T = D1.T
        self.n_deriv = D1.shape[0]

        D3_smoothing = lambda_smoothing * D3.T @ D3 if lambda_smoothing > 0 else 0.0

        boundary_penalty = 0.0
        if bc_type == "clamped" and k >= 2:
            bm = np.vstack([D1[0, :], D1[-1, :]])
            boundary_penalty = kappa * bm.T @ bm
        elif bc_type == "natural" and k >= 2:
            bm = np.vstack([D2[0, :], D2[-1, :]])
            boundary_penalty = kappa * bm.T @ bm

        self._penalty_sum = D3_smoothing + boundary_penalty
        self._ridge = _EPS_RIDGE * np.eye(self.n_base)

        # Preallocated solve-loop buffers (shared across solve calls)
        self._D1_weighted_buf = np.empty_like(D1)
        self._kVd_buf = np.empty(self.n_deriv, dtype=float)

        # Precompute base Gram matrix for given weights (no TIDD upweighting)
        self._base_weights = weights
        self._A_base, self._BTy_base = self._gram(weights)

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------

    def _difference_matrices(self) -> list[np.ndarray]:
        """Compute weighted difference matrices D1, D2, D3 for non-uniform B-splines.

        These matrices approximate derivatives of the spline coefficients, accounting
        for non-uniform knot spacing via inverse weighting.

        References
        ----------
        - General P-Splines for Non-Uniform B-Splines (https://doi.org/10.48550/arXiv.2201.06808)
        - Multivariate probabilistic CRPS learning with an application to day-ahead electricity prices
          (https://doi.org/10.1016/j.ijforecast.2024.01.005)

        Returns
        -------
        D : list of ndarray
            [D1, D2, D3] difference matrices for 1st, 2nd, 3rd derivatives.
        """
        knots = self.padded_knots
        n_base_funcs = self.n_base
        o = self.k

        def lag_diff(arr, lag):
            return arr[lag:] - arr[:-lag]

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
                fd = self._cached_first_diff(n_base_funcs - i)
                _D = fd / diag_vals[:, np.newaxis]

                D[i] = _D if i == 0 else _D @ D[i-1]

            else:
                # Spline order <= derivative order: use standard difference
                D[i] = np.diff(np.eye(n_base_funcs), n=i+1, axis=0)

        return D

    def _derivative_zones(self, bp: np.ndarray) -> dict[str, np.ndarray]:
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
        knots = self.padded_knots
        k = self.k
        n_deriv = len(knots) - k - 2
        empty = np.array([], dtype=int)

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
        return {
            "hdd": np.flatnonzero(right <= bp[0]),
            "tidd": np.flatnonzero((left < bp[1]) & (right > bp[0])),
            "cdd": np.flatnonzero(left >= bp[1]),
        }

    def _tidd_weights(
        self,
        weights: Optional[np.ndarray],
        bp: np.ndarray,
        weight_factor: float = 10,
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
    ) -> np.ndarray:
        """Run iterative monotonicity-constrained least-squares solve.

        Solves ``(A + κD1'VD1)α = BTy + κD1'Vδ`` iteratively until the
        active-set matrix V converges or *maxiter* is reached.

        Parameters
        ----------
        A : ndarray
            Base LHS matrix (Gram + smoothing + boundary penalties).
        BTy : ndarray
            Base RHS vector (B'y or B'W²y).
        D1_zone : dict
            Zone index arrays {"hdd", "tidd", "cdd"} from _derivative_zones.
        kappa : float
            Monotonicity penalty weight.
        maxiter : int
            Maximum iterations before issuing MaxIterationWarning.

        Returns
        -------
        coefs : ndarray
            Fitted spline coefficients.
        """
        D1 = self.D1
        D1T = self.D1T
        D1_weighted_buf = self._D1_weighted_buf
        kVd_buf = self._kVd_buf
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

        for i in range(maxiter):
            kappa_V = kappa * V

            np.multiply(kappa_V[:, np.newaxis], D1, out=D1_weighted_buf)
            lhs = A + D1T @ D1_weighted_buf
            np.multiply(kappa_V, delta, out=kVd_buf)
            rhs = BTy + D1T @ kVd_buf

            try:
                coefs = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                coefs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)

            deriv = D1 @ coefs
            V_new.fill(0)

            if has_hdd:
                V_new[hdd_idx] = deriv[hdd_idx] > delta[hdd_idx]
            if has_tidd:
                V_new[tidd_idx] = deriv[tidd_idx] != delta[tidd_idx]
            if has_cdd:
                V_new[cdd_idx] = deriv[cdd_idx] < delta[cdd_idx]

            if np.count_nonzero(V != V_new) == 0:
                break

            V, V_new = V_new, V

        else:
            warnings.warn(
                "Max iteration reached. The results are not reliable.", MaxIterationWarning
            )

        return coefs

    def _fit(
        self,
        bp: np.ndarray,
        base_weights: Optional[np.ndarray],
        kappa: float,
        maxiter: int,
    ) -> np.ndarray:
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
        """
        if bp[1] - bp[0] > 0:
            eff_w = self._tidd_weights(base_weights, bp, weight_factor=50)
        else:
            eff_w = base_weights

        # Reuse precomputed base matrices when weights are unchanged
        if eff_w is base_weights and base_weights is self._base_weights:
            A, BTy = self._A_base, self._BTy_base
        else:
            A, BTy = self._gram(eff_w)

        D1_zone = self._derivative_zones(bp)
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
        coefs = instance._fit(bp, weights, kappa_penalty, maxiter)
        
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
        regularization_alpha: float = 0.01,
        regularization_percent_lasso: float = 1.0,
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

        # BP regularization: penalizes bp far from data bounds to maximize TIDD width
        self.regularization_alpha = regularization_alpha
        self.regularization_percent_lasso = regularization_percent_lasso

        self.weights_alpha = [2.0]

    def _set_bp(
        self,
        x,
        y,
        bp,
        weights=None,
        zone_knot_count=10,
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
        search_margin : float
            Half-width of the search region around the initial bp estimate,
            expressed as a fraction of the data range.  0.2 means ±20% of
            the range in each parameterized dimension.

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
        inner_maxiter = min(5, self.maxiter) if self.maxiter else 5
        lasso_a = self.regularization_percent_lasso * self.regularization_alpha
        ridge_a = (1 - self.regularization_percent_lasso) * self.regularization_alpha
        x_bnds = np.array([x_min, x_max])
        kappa = self.kappa_penalty
        k = self.bspline_degree
        has_reg = self.regularization_alpha != 0

        # --- Precompute bp-independent matrices (fixed knot vector) ---
        bp_init = np.asarray(bp, dtype=float)
        fixed_knots = self.knots.get_internal_knots(
            bp=bp_init, n_knots=zone_knot_count, n_min=self.n_min,
        )
        padded_knots = _BasePSpline._pad_knots(fixed_knots, k)

        cache = _BasePSpline(
            x, y, padded_knots, k, weights,
            self.lambda_smoothing, self.bc_type, kappa,
        )

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
        # Parameterization: X0 = (bp[0]-x_min)/x_range, X1 = (bp[1]-bp[0])/x_range
        allow_hdd = self.allow_heating_zone
        allow_cdd = self.allow_cooling_zone

        x0_full = np.clip([
            (bp_init[0] - x_min) / x_range,
            (bp_init[1] - bp_init[0]) / x_range,
        ], 0.0, 1.0)

        if allow_hdd and allow_cdd:
            def to_full(X): 
                return X
            
            x0_opt = x0_full
            bnds_opt = np.array([(0.0, 1.0), (0.0, 1.0)])

        elif allow_hdd:   # cooling fixed: bp[1] = x_max  →  X1 = 1 - X0
            def to_full(X): 
                return np.array([X[0], 1.0 - X[0]])
            
            x0_opt = x0_full[:1]
            bnds_opt = np.array([(0.0, 1.0)])

        elif allow_cdd:   # heating fixed: bp[0] = x_min  →  X0 = 0
            def to_full(X): 
                return np.array([0.0, X[0]])
            
            x0_opt = x0_full[1:]
            bnds_opt = np.array([(0.0, 1.0)])

        else:             # both fixed: no free parameters, skip optimizer
            def to_full(X): 
                return np.array([0.0, 1.0])
            
            x0_opt = None

        def objective(X_free, grad=[]):
            trial_bp = to_full(X_free).copy()
            bp_sum = np.sum(trial_bp)
            if bp_sum > 1:
                trial_bp /= bp_sum
            trial_bp = np.cumsum(trial_bp) * x_range + x_min

            coefs = cache._fit(trial_bp, weights, kappa, inner_maxiter)

            resid = cache.B @ coefs - y
            sse = np.sum(resid ** 2)
            loss = sse / N
            wrmse = np.sqrt(loss)

            if has_reg:
                loss += bp_penalty(trial_bp, wrmse)

            return loss

        # Scale NLopt budget with segment size: larger segments warrant more iterations,
        # but the relationship is sub-linear (diminishing returns).  Floor at 30 so
        # small segments still converge; cap at 100 to match the previous fixed budget.
        adaptive_budget = int(np.clip(N // 10, 30, 100))

        opt_settings = OptimizationSettings(
            algorithm="nlopt_sbplx",
            initial_step=0.20,
            stop_criteria_type="iteration maximum",
            stop_criteria_value=adaptive_budget,
            x_tol_rel=1e-2,
            f_tol_rel=1e-1,
        )

        if x0_opt is not None:
            optimizer = NLoptOptimizer(objective, x0_opt, bnds_opt, opt_settings)
            result = optimizer.run()
            X_result = to_full(result.x)
        else:
            X_result = np.array([0.0, 1.0])

        # Transform result back to bp space
        bp = np.array(X_result)
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

    def fit(self, x, y, bp=None, weights=None) -> "DailyPSpline":
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
        zone_knot_count = self.zone_knot_count_max
        if zone_knot_count is None:
            zone_knot_count = 10
            
        elif zone_knot_count < 0:
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
        for n in range(self.adaptive_iterations):
            # Optimize breakpoints and build final spline matrices in one step
            bp_standardized, psp = self._set_bp(
                x_standardized,
                y_standardized,
                bp_standardized,
                weights=self.weights,
                zone_knot_count=zone_knot_count,
            )

            coefs = psp._fit(bp_standardized, self.weights, self.kappa_penalty, self.maxiter)
            residuals = (psp.B @ coefs - y_standardized).reshape(-1)
            rmse = np.sqrt(np.mean(residuals ** 2))

            if np.isclose(rmse, prior_rmse, rtol=1e-3):
                break

            prior_rmse = rmse

            # Update weights using adaptive loss
            a_weights, _, weight_alpha = adaptive_weights(
                residuals, alpha="adaptive", C_algo="mad",
            )
            self.weights_alpha.append(weight_alpha)

            if weight_alpha == 2.0:
                break  # Gaussian residuals: weights unchanged, next iteration identical

            if self.weights is None:
                self.weights = a_weights
            else:
                self.weights = self.weights * a_weights

            self.weights = _rescale_to_range(self.weights, new_min=1.0, new_max=100.0)

        super().__init__(t=psp.padded_knots, c=coefs, k=self.bspline_degree, extrapolate=True)

        # Store breakpoints and bounds in original space
        self.bp = bp_standardized * self.x_std + self.x_mean
        self.fit_bnds = np.array([self.x[0], self.x[-1]])

        # Compute training metrics
        self.training_metrics = BaselineMetrics(
            df=pd.DataFrame({
                'observed': self.y,
                'predicted': self.predict(self.x)
            }),
            num_model_params=psp.n_base - psp.k + 1
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


# ---------------------------------------------------------------------------
# Numba JIT warmup
# ---------------------------------------------------------------------------
# adaptive_weights uses a Numba-JIT function (penalized_loss_fcn) that is
# compiled on first call.  Calling it once at import time avoids ~53 ms of
# per-model-fit compilation overhead observed during profiling.
def _warmup_numba() -> None:
    try:
        _dummy = np.ones(50, dtype=np.float64)
        adaptive_weights(_dummy, alpha="adaptive", C_algo="mad")
    except Exception:
        pass


_warmup_numba()
