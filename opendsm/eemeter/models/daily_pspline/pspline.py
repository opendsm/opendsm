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

from opendsm.common.stats.adaptive_loss import adaptive_weights
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
        D1, D2, D3 = _BasePSpline._difference_matrices(tuple(self.padded_knots), self.k, self.n_base)
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
    def _difference_matrices(padded_knots_tuple: tuple, k: int, n_base: int):
        """Cached difference matrices keyed on knot vector, degree, and basis count.

        Results are reused across BIC-scan candidates that share the same knot
        configuration, avoiding redundant O(n²) matrix construction.
        """
        knots = np.asarray(padded_knots_tuple)

        def lag_diff(arr, lag):
            return arr[lag:] - arr[:-lag]

        D = [None, None, None]
        for i in range(3):
            if k > i:
                a = 1 / (k - i)
                diag_vals = a * lag_diff(knots[i + 1:-i - 1], lag=k - i)
                diag_vals = np.where(diag_vals == 0, 1.0, diag_vals)
                fd = _BasePSpline._cached_first_diff(n_base - i)
                _D = fd / diag_vals[:, np.newaxis]
                D[i] = _D if i == 0 else _D @ D[i - 1]
            else:
                D[i] = np.diff(np.eye(n_base), n=i + 1, axis=0)

        for d in D:
            d.flags.writeable = False
        return D[0], D[1], D[2]

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
            Zone index arrays {"hdd", "tidd", "cdd"} from _BasePSpline._derivative_zones.
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

        for i in range(maxiter):
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
        inner_maxiter = min(5, self.maxiter) if self.maxiter else 5
        lasso_a = self.regularization_percent_lasso * self.regularization_alpha
        ridge_a = (1 - self.regularization_percent_lasso) * self.regularization_alpha
        x_bnds = np.array([x_min, x_max])
        kappa = self.kappa_penalty
        k = degree if degree is not None else self.bspline_degree
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
        
        # TODO: improve bp optimization, could iterate back and forth
        adaptive_budget = int(np.clip(N // 10, 30, 100))
        # adaptive_budget = 1000
        algorithm = "nlopt_sbplx"

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
        """Two-pass greedy knot-count scan using BIC-like selection criteria.

        Pass 1 scans HDD knot count (CDD fixed at max).
        Pass 2 scans CDD knot count (HDD fixed at best from pass 1).
        Each pass stops after two consecutive non-improving steps.

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
        psp = coefs = None

        # Pass 1: scan HDD knot count, CDD fixed at max
        best_hdd = hdd_max
        best_score = np.inf
        n_worse = 0
        for count in range(hdd_max, -1, -1):
            score, cand_psp, cand_coefs = candidate_fn(bp_std, count, cdd_max)
            if score < best_score:
                best_score = score
                best_hdd = count
                psp, coefs = cand_psp, cand_coefs
                n_worse = 0
            else:
                n_worse += 1
                if n_worse >= 2:
                    break

        # Pass 2: scan CDD knot count, HDD fixed at best from pass 1
        best_score = np.inf
        n_worse = 0
        for count in range(cdd_max, -1, -1):
            score, cand_psp, cand_coefs = candidate_fn(bp_std, best_hdd, count)
            if score < best_score:
                best_score = score
                psp, coefs = cand_psp, cand_coefs
                n_worse = 0
            else:
                n_worse += 1
                if n_worse >= 2:
                    break

        return psp, coefs

    def _weighted_stats(self, y):
        """Compute w², sum(w²), and weighted total sum of squares for self.weights."""
        w_sq = self.weights ** 2
        sum_w_sq = float(np.sum(w_sq))
        y_wmean = float(np.dot(w_sq, y)) / sum_w_sq
        wtss = float(np.dot(w_sq, (y - y_wmean) ** 2))

        return w_sq, sum_w_sq, wtss

    def _score(self, wssr, wtss, N, n_base):
        """Compute zone selection criterion using configured settings."""
        return selection_criteria(
            wssr, 
            wtss,
            N, 
            n_base,
            self.zone_criteria, 
            self.zone_penalty_multiplier, 
            self.zone_penalty_power,
        )

    def _fit_degree(self, x_std, y_std, bp_std, bp_provided, zone_knot_count, bic_maxiter, N, bp_opt_algo, degree):
        """Adaptive fitting loop for one bspline degree.

        Initialises ``self.knots`` for *degree*, then iterates: optimise bp →
        BIC knot scan → adaptive weight update, until convergence.
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
        )
        _B_cache: dict = {}
        prior_wrmse = np.inf
        prior_bp = None
        psp = coefs = None

        for _ in range(self.adaptive_iterations):
            if not bp_provided and (prior_bp is None or not (
                self.freeze_bp_on_convergence
                and np.allclose(bp_std, prior_bp, atol=1e-5)
            )):
                bp_std, _ = self._set_bp(
                    x_std, y_std, bp_std,
                    weights=self.weights,
                    zone_knot_count=zone_knot_count,
                    algorithm=bp_opt_algo,
                    degree=degree,
                )
            prior_bp = bp_std

            # Hoist weight-dependent constants so _candidate doesn't recompute per call
            # (~20 candidates × up to 10 outer iterations would be expensive).
            _w_sq, _sum_w_sq, _wtss = self._weighted_stats(y_std)

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
                    self.weights, self.lambda_smoothing, self.bc_type, self.kappa_penalty,
                    B=B,
                )
                cand_coefs = cand_psp._fit(bp, self.weights, self.kappa_penalty, bic_maxiter)
                resid = cand_psp.B @ cand_coefs - y_std
                wssr = float(np.dot(_w_sq, resid ** 2))
                score = self._score(wssr, _wtss, N, cand_psp.n_base)
                return score, cand_psp, cand_coefs

            hdd_max = (
                min(zone_knot_count, int(x_std.searchsorted(bp_std[0], side='left')) // self.n_min)
                if self.allow_heating_zone else 0
            )
            cdd_max = (
                min(zone_knot_count, (N - int(x_std.searchsorted(bp_std[1], side='right'))) // self.n_min)
                if self.allow_cooling_zone else 0
            )
            if self.zone_knot_scan:
                psp, coefs = self._zone_knot_scan(bp_std, hdd_max, cdd_max, _candidate)
            else:
                _, psp, coefs = _candidate(bp_std, hdd_max, cdd_max)

            residuals = (psp.B @ coefs - y_std).reshape(-1)
            wrmse = np.sqrt(np.dot(_w_sq, residuals ** 2) / _sum_w_sq)

            if abs(wrmse - prior_wrmse) <= 1e-3 * prior_wrmse:
                break
            prior_wrmse = wrmse

            a_weights, _, weight_alpha = adaptive_weights(residuals, alpha="adaptive", C_algo="mad")
            self.weights_alpha.append(weight_alpha)
            if weight_alpha == 2.0:
                break  # Gaussian residuals: weights unchanged, next iteration identical

            self.weights = (self.weights * a_weights) if self.weights is not None else a_weights
            self.weights = _rescale_to_range(self.weights, new_min=1.0, new_max=100.0)

        return psp, coefs, bp_std

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

        # Initial Knots used only for the bp derivative estimate; rebuilt per-degree inside _fit_degree
        self.knots = Knots(
            x_std, y_std, w=self.weights,
            spline_interp_count=1000, spline_lambda=10, n_min=self.n_min,
            bspline_degree=self.bspline_degree,
            lambda_smoothing=self.lambda_smoothing,
            kappa_penalty=self.kappa_penalty, maxiter=self.maxiter,
        )
        bp_std_init = (
            (np.asarray(bp, dtype=float) - self.x_mean) / self.x_std
            if bp is not None
            else _estimate_bp_from_derivative(self.knots)
        )

        N = len(x_std)
        bic_maxiter = min(10, self.maxiter)
        bp_provided = bp is not None

        # Outer loop: degree 0 seeds bp via global nlopt_direct; higher degrees refine
        # locally with nlopt_sbplx using the prior degree's converged bp as warm-start.
        # The degree with the best BIC score is selected; degree 0 is excluded unless
        # it is the explicit target (fit(bspline_degree=0) or self.bspline_degree == 0).
        target_degree = bspline_degree if bspline_degree is not None else self.bspline_degree
        degree_sequence = [0, target_degree] if target_degree > 0 else [0]

        base_weights = self.weights.copy()
        best_score = np.inf
        best_psp = best_coefs = best_bp_std = best_knots = None
        best_weights = base_weights
        outer_prior_bp = None
        for deg_idx, current_degree in enumerate(degree_sequence):
            bp_opt_algo = "nlopt_direct" if deg_idx == 0 else "nlopt_sbplx"
            bp_std = outer_prior_bp.copy() if outer_prior_bp is not None else bp_std_init.copy()
            self.weights = base_weights.copy()

            deg_zone_knot_count = 20 if current_degree == 0 else zone_knot_count
            psp, coefs, bp_std = self._fit_degree(
                x_std, y_std, bp_std, bp_provided,
                deg_zone_knot_count, bic_maxiter, N,
                bp_opt_algo, current_degree,
            )

            # BIC score for this degree on final adaptive weights
            w_sq, _, wtss = self._weighted_stats(y_std)
            wssr = float(np.dot(w_sq, (psp.B @ coefs - y_std).reshape(-1) ** 2))
            score = self._score(wssr, wtss, N, psp.n_base)

            if (current_degree > 0 or target_degree == 0) and score < best_score:
                best_score = score
                best_psp, best_coefs = psp, coefs
                best_bp_std = bp_std.copy()
                best_knots = self.knots
                best_weights = self.weights.copy()

            outer_prior_bp = bp_std.copy()

        self.knots = best_knots
        self.weights = best_weights
        super().__init__(t=best_psp.padded_knots, c=best_coefs, k=best_psp.k, extrapolate=True)
        self.bp = best_bp_std * self.x_std + self.x_mean
        self.fit_bnds = np.array([self.x[0], self.x[-1]])
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