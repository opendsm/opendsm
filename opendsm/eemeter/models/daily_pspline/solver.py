"""Penalized B-spline solver with zone-specific monotonicity constraints.

Implements the iterative active-set method from Eilers (2005) for fitting
monotone P-splines. Precomputes design and penalty matrices for a fixed
knot vector; the solve can be repeated for different breakpoints/weights.

References
----------
Eilers, P. H. C. (2005). Unimodal smoothing. Journal of Chemometrics,
    19, 317-328. DOI:10.1002/cem.935

De Leeuw, J. (2017). Computing and Fitting Monotone Splines.
    UCLA Statistics Preprints.

General P-Splines for Non-Uniform B-Splines.
    DOI:10.48550/arXiv.2201.06808
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
from scipy.interpolate import BSpline


# Ridge regularization for numerical stability.
_EPS_RIDGE = 1e-4


class MaxIterationWarning(UserWarning): ...


class PSplineSolver:
    """Precomputed spline geometry for a fixed knot vector.

    Construct once per knot configuration, then call ``solve()`` for
    different breakpoints and weights without rebuilding matrices.

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
        Per-observation weights (without TIDD upweighting).
    lambda_smoothing : float
        Third-derivative smoothing penalty weight.
    lambda_curvature : float
        Second-derivative curvature penalty weight; prevents rapid slope changes.
    lambda_slope : float
        First-derivative slope penalty weight; prevents steep slopes in sparse zones.
    bc_type : str or None
        Boundary condition type ('natural', 'clamped', or None).
    kappa : float
        Penalty weight for boundary conditions and monotonicity.
    B : ndarray or None
        Pre-built design matrix (avoids recomputation in hot loops).
    """

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
        lambda_curvature: float = 0.0,
        lambda_slope: float = 0.0,
    ):
        self.x = x
        self.y = y
        self.k = k
        self.padded_knots = padded_knots
        self.n_base = len(padded_knots) - k - 1

        self.B = (
            BSpline.design_matrix(x=x, t=padded_knots, k=k, extrapolate=True).toarray()
            if B is None else B
        )

        need_D2 = (bc_type == "natural" and k >= 2) or (lambda_curvature > 0 and k >= 2)
        need_D3 = (lambda_smoothing > 0)
        D1, D2, D3 = _difference_matrices(
            tuple(self.padded_knots), self.k, self.n_base,
            need_D2=need_D2, need_D3=need_D3,
        )
        self.D1 = D1
        self.D1T = D1.T
        self.n_deriv = D1.shape[0]

        def _density_weighted_penalty(D_pen, lam, span):
            """Build density-weighted penalty matrix lam * Dw'Dw.

            Column sums of B measure how much data each coefficient "sees";
            averaging ``span`` adjacent columns gives per-row support.
            Weights are normalized so median support maps to 1.0, making
            lambda interpretable as the penalty at typical data density.
            """
            col_support = self.B.sum(axis=0)  # (n_base,)
            row_support = np.array([
                np.mean(col_support[max(0, i):min(self.n_base, i + span)])
                for i in range(D_pen.shape[0])
            ])
            median_support = np.median(row_support)
            density_w = median_support / np.maximum(row_support, 1e-10)
            Dw = D_pen * density_w[:, np.newaxis]
            return lam * Dw.T @ Dw

        D2_penalty = 0.0
        if lambda_curvature > 0 and D2 is not None:
            D2_penalty = _density_weighted_penalty(D2, lambda_curvature, span=3)

        D1_penalty = 0.0
        if lambda_slope > 0 and k == 1:
            D1_penalty = _density_weighted_penalty(D1, lambda_slope, span=2)
        D3_penalty = lambda_smoothing * D3.T @ D3 if lambda_smoothing > 0 and D3 is not None else 0.0

        bc_penalty = 0.0
        if bc_type == "clamped" and k >= 2:
            bm = np.vstack([D1[0, :], D1[-1, :]])
            bc_penalty = kappa * bm.T @ bm
        elif bc_type == "natural" and k >= 2 and D2 is not None:
            bm = np.vstack([D2[0, :], D2[-1, :]])
            bc_penalty = kappa * bm.T @ bm

        self._penalty_sum = D1_penalty + D2_penalty + D3_penalty + bc_penalty
        self._ridge = _EPS_RIDGE * np.eye(self.n_base)

        # Preallocated solve-loop buffers
        self._D1w_buf = np.empty_like(D1)
        self._kV_buf = np.empty(self.n_deriv, dtype=float)
        self._kVd_buf = np.empty(self.n_deriv, dtype=float)
        self._lhs_buf = np.empty((self.n_base, self.n_base), dtype=float)

        # Precompute base Gram matrix
        self._base_weights = weights
        self._A_base, self._BTy_base = self._gram(weights)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self,
        bp: np.ndarray,
        weights: Optional[np.ndarray],
        kappa: float,
        maxiter: int,
        residuals: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit coefficients for given breakpoints.

        Applies TIDD upweighting, computes zone assignments, then runs
        the iterative monotonicity-constrained solve.

        Parameters
        ----------
        residuals : array, optional
            Residuals from a prior fit, used to estimate per-zone
            variance for TIDD upweighting.  On the first call, pass
            None to fall back on raw-y MAD.

        Returns (coefs, V, LHS) where V is the converged active-set
        diagonal and LHS is the final penalized Gram matrix.
        """
        if bp[1] - bp[0] > 0:
            eff_w = self._tidd_weights(weights, bp, residuals=residuals)
        else:
            eff_w = weights

        if eff_w is weights and weights is self._base_weights:
            A, BTy = self._A_base, self._BTy_base
        else:
            A, BTy = self._gram(eff_w)

        zones = derivative_zones(tuple(self.padded_knots), self.k, float(bp[0]), float(bp[1]))
        return self._solve_iterative(A, BTy, zones, kappa, maxiter)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _gram(self, weights: Optional[np.ndarray]):
        """Weighted Gram matrix A = B'W²B + penalties, and B'W²y."""
        if weights is None:
            A = self.B.T @ self.B + self._penalty_sum + self._ridge
            BTy = self.B.T @ self.y
        else:
            w_sq = np.square(weights)
            A = self.B.T @ (self.B * w_sq[:, np.newaxis]) + self._penalty_sum + self._ridge
            BTy = self.B.T @ (self.y * w_sq)
        return A, BTy

    def _tidd_weights(
        self,
        weights: Optional[np.ndarray],
        bp: np.ndarray,
        residuals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Upweight TIDD observations to equalize standard errors across zones.

        The standard error of a zone's level estimate is
        ``sigma_zone / sqrt(N_zone)``.  Equalizing SEs across zones
        gives each zone influence proportional to its estimation
        precision.  The factor is::

            factor = sqrt((sigma_adj / sigma_tidd)^2 * (N_adj / N_tidd))

        Per-zone sigma is estimated from residuals when available
        (after the first fit iteration), falling back to MAD of raw y
        on the initial call.  Using residuals is more accurate because
        raw y variance in the CDD/HDD zones includes the slope signal,
        overstating the noise.
        """
        tidd_mask = (self.x > bp[0]) & (self.x < bp[1])
        n_tidd = int(np.sum(tidd_mask))
        if n_tidd < 2:
            return weights

        hdd_mask = self.x <= bp[0]
        cdd_mask = self.x >= bp[1]
        n_hdd = int(np.sum(hdd_mask))
        n_cdd = int(np.sum(cdd_mask))

        # Choose the data source for variance estimation
        vals = residuals if residuals is not None else self.y

        _MAD_K = 1.4826
        v_tidd = vals[tidd_mask]
        mad_tidd = np.median(np.abs(v_tidd - np.median(v_tidd))) * _MAD_K
        mad_tidd = max(mad_tidd, 1e-10)

        # Use the largest adjacent zone for comparison
        if n_hdd >= n_cdd and n_hdd >= 2:
            v_adj = vals[hdd_mask]
            n_adj = n_hdd
        elif n_cdd >= 2:
            v_adj = vals[cdd_mask]
            n_adj = n_cdd
        else:
            return weights

        mad_adj = np.median(np.abs(v_adj - np.median(v_adj))) * _MAD_K
        mad_adj = max(mad_adj, 1e-10)

        variance_ratio = (mad_adj / mad_tidd) ** 2
        size_ratio = n_adj / n_tidd
        factor = np.sqrt(variance_ratio * size_ratio)

        if factor <= 1.0:
            return weights

        w = np.ones_like(self.x) if weights is None else weights.copy()
        w[tidd_mask] *= factor
        return w

    def _solve_iterative(
        self,
        A: np.ndarray,
        BTy: np.ndarray,
        zones: dict,
        kappa: float,
        maxiter: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Iterative active-set monotonicity-constrained solve.

        Solves (A + κD1'VD1)α = BTy + κD1'Vδ until V converges.
        First iteration is unconstrained (V=0) with early exit.

        Returns (coefs, V, LHS) where LHS is the final penalized Gram matrix.
        """
        D1, D1T = self.D1, self.D1T
        n_deriv = self.n_deriv

        delta = np.zeros(n_deriv, dtype=float)
        V = np.zeros(n_deriv, dtype=int)
        V_new = np.zeros(n_deriv, dtype=int)

        hdd_idx, tidd_idx, cdd_idx = zones["hdd"], zones["tidd"], zones["cdd"]
        has_hdd, has_tidd, has_cdd = len(hdd_idx) > 0, len(tidd_idx) > 0, len(cdd_idx) > 0

        # Iteration 0: unconstrained solve
        coefs = _safe_solve(A, BTy)
        deriv = D1 @ coefs

        if has_hdd:
            V[hdd_idx] = deriv[hdd_idx] > delta[hdd_idx]
        if has_tidd:
            V[tidd_idx] = deriv[tidd_idx] != delta[tidd_idx]
        if has_cdd:
            V[cdd_idx] = deriv[cdd_idx] < delta[cdd_idx]

        if np.sum(V) == 0:
            return coefs, V, A

        # Iterations 1+: penalized solve with preallocated buffers
        D1w_buf, kV_buf, kVd_buf, lhs_buf = (
            self._D1w_buf, self._kV_buf, self._kVd_buf, self._lhs_buf,
        )

        for _ in range(1, maxiter):
            np.multiply(kappa, V, out=kV_buf)
            np.multiply(kV_buf[:, np.newaxis], D1, out=D1w_buf)
            np.dot(D1T, D1w_buf, out=lhs_buf)
            lhs_buf += A
            np.multiply(kV_buf, delta, out=kVd_buf)
            rhs = BTy + D1T @ kVd_buf

            coefs = _safe_solve(lhs_buf, rhs)
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
                "Max iteration reached. The results are not reliable.",
                MaxIterationWarning,
            )

        # Enforce exact TIDD flatness: average coefficients in the TIDD
        # zone so derivatives are structurally zero.  The penalty-based
        # solve gets close (~1e-4) but not exact.
        if has_tidd:
            coefs = _enforce_tidd_flat(coefs, tidd_idx, D1)

        return coefs, V, lhs_buf.copy()


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------

def pad_knots(internal_knots: np.ndarray, degree: int) -> np.ndarray:
    """Add repeated boundary knots (multiplicity k+1).

    See De Leeuw (2017) *Computing and Fitting Monotone Splines*.
    """
    k = degree
    return np.hstack([
        np.repeat(internal_knots[0], k),
        internal_knots,
        np.repeat(internal_knots[-1], k),
    ])


def effective_df(
    solver: PSplineSolver,
    V: np.ndarray,
    kappa: float,
    w_sq: np.ndarray,
) -> float:
    """Effective degrees of freedom: tr((LHS)⁻¹ A).

    Includes the converged monotonicity penalty so constrained
    coefficients are correctly counted as having reduced freedom.
    """
    A = solver.B.T @ (solver.B * w_sq[:, np.newaxis])
    mono = kappa * (solver.D1.T @ (solver.D1 * V.astype(float)[:, np.newaxis]))
    LHS = A + solver._penalty_sum + solver._ridge + mono
    LHS_inv_A = _safe_solve(LHS, A)
    return float(np.trace(LHS_inv_A))


@lru_cache(maxsize=256)
def derivative_zones(
    padded_knots_tuple: tuple,
    k: int,
    bp0: float,
    bp1: float,
) -> dict:
    """Assign derivative indices to HDD / TIDD / CDD zones.

    The i-th first-difference row has support [knots[i+1], knots[i+k+1]].
    Zones are assigned by overlap with [bp0, bp1].
    """
    knots = np.asarray(padded_knots_tuple)
    n_deriv = len(knots) - k - 2
    empty = np.array([], dtype=int)
    empty.flags.writeable = False

    rtol, atol = 1e-4, 1e-12
    single_bp = abs(bp1 - bp0) <= rtol * (abs(bp1) + atol)

    if single_bp and abs(knots[-1] - bp1) <= rtol * (abs(bp1) + atol):
        full = np.arange(n_deriv); full.flags.writeable = False
        return {"hdd": full, "tidd": empty, "cdd": empty}
    if single_bp and abs(knots[0] - bp0) <= rtol * (abs(bp0) + atol):
        full = np.arange(n_deriv); full.flags.writeable = False
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


@lru_cache(maxsize=16)
def _cached_first_diff(n: int) -> np.ndarray:
    result = np.diff(np.eye(n), n=1, axis=0)
    result.flags.writeable = False
    return result


@lru_cache(maxsize=256)
def _difference_matrices(
    padded_knots_tuple: tuple,
    k: int,
    n_base: int,
    need_D2: bool = True,
    need_D3: bool = True,
):
    """Weighted difference matrices for non-uniform B-splines.

    References
    ----------
    General P-Splines for Non-Uniform B-Splines.
        DOI:10.48550/arXiv.2201.06808
    """
    knots = np.asarray(padded_knots_tuple)

    def lag_diff(arr, lag):
        return arr[lag:] - arr[:-lag]

    def _compute_D(i, D_prev):
        if k > i:
            a = 1 / (k - i)
            diag_vals = a * lag_diff(knots[i + 1:-i - 1], lag=k - i)
            diag_vals = np.where(diag_vals == 0, 1.0, diag_vals)
            fd = _cached_first_diff(n_base - i)
            _D = fd / diag_vals[:, np.newaxis]
            return _D if D_prev is None else _D @ D_prev
        return np.diff(np.eye(n_base), n=i + 1, axis=0)

    D1 = _compute_D(0, None)
    D1.flags.writeable = False

    D2 = None
    if need_D2 or need_D3:
        D2 = _compute_D(1, D1)
        D2.flags.writeable = False

    D3 = None
    if need_D3:
        D3 = _compute_D(2, D2)
        D3.flags.writeable = False

    return D1, D2, D3


def _enforce_tidd_flat(
    coefs: np.ndarray,
    tidd_idx: np.ndarray,
    D1: np.ndarray,
) -> np.ndarray:
    """Set TIDD-zone derivatives to exactly zero by averaging coefficients.

    Each D1 row i connects coefficients i and i+1.  For TIDD rows, we
    replace the linked coefficients with their mean so the first
    difference is structurally zero.
    """
    coefs = coefs.copy()
    # Collect all coefficient indices touched by TIDD derivative rows
    tidd_coef_idx = set()
    for i in tidd_idx:
        # D1 row i links coefs i and i+1 (for standard first-difference)
        tidd_coef_idx.add(i)
        tidd_coef_idx.add(i + 1)
    tidd_coef_idx = sorted(tidd_coef_idx)
    if len(tidd_coef_idx) > 0:
        mean_val = np.mean(coefs[tidd_coef_idx])
        coefs[tidd_coef_idx] = mean_val
    return coefs


def _safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """np.linalg.solve with lstsq fallback for singular systems."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return result
