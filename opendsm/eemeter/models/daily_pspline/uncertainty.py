"""Prediction uncertainty estimation for fitted P-splines.

Model covariance uses a soft monotonicity penalty (κ=N) instead of the
hard constraint (κ=1e9) from fitting, treating monotonicity as a
Bayesian prior rather than a hard constraint.  This gives meaningful
model variance that widens at data-sparse edges.

Noise scale is a global empirical quantile of |residuals|, calibrated
for target coverage without distributional assumptions.  Heteroscedastic
σ(T) is deferred to the planned YJ transform, which will stabilize
variance and make a global σ correct by construction.

Bartlett VIF for temporal autocorrelation is applied to the model
variance component (curve uncertainty), not to the noise component
(individual observation variance is independent of training ACF).

References
----------
Ruppert, D., Wand, M. P. & Carroll, R. J. (2003). *Semiparametric
    Regression*. Cambridge University Press.  (Soft-penalty covariance.)
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import t as t_dist



class UncertaintyEstimator:
    """Computes prediction interval half-widths for a fitted P-spline.

    Created by ``fit_segment`` and attached to the ``PSpline`` instance.
    PSpline delegates ``prediction_uncertainty`` to this object.

    Parameters
    ----------
    sandwich_cov : ndarray, shape (m, m)
        HC3 sandwich covariance of spline coefficients in original y-space.
    knots_std : ndarray
        Padded knot vector in standardized x-space (for basis evaluation).
    degree : int
        B-spline degree.
    x_mean, x_std, y_std : float
        Standardization parameters for basis evaluation and scaling.
    sigma_local_T : ndarray
        Training temperatures (sorted, original space).
    sigma_local : ndarray
        Local noise scale σ(T) at each training temperature.
    vif : float
        Bartlett variance inflation factor from residual autocorrelation.
    include_autocorr : bool
        Whether to apply VIF to the noise term.
    alpha : float
        Significance level for prediction intervals (e.g. 0.1 for 90% PI).
    n : int
        Number of training observations.
    ddof : int
        Model degrees of freedom (for t-distribution dof = n - ddof).
    """

    def __init__(
        self,
        sandwich_cov: np.ndarray,
        knots_std: np.ndarray,
        degree: int,
        x_mean: float,
        x_std: float,
        sigma_local_T: np.ndarray,
        sigma_local: np.ndarray,
        vif: float,
        dof: int,
    ):
        self._sandwich_cov = sandwich_cov
        self._knots_std = knots_std
        self._degree = degree
        self._x_mean = x_mean
        self._x_std = x_std
        self._sigma_local_T = sigma_local_T
        self._sigma_local = sigma_local
        self._vif = vif
        self._dof = max(1, dof)

    def to_dict(self) -> dict:
        return {
            "sandwich_cov": self._sandwich_cov.tolist(),
            "knots_std": self._knots_std.tolist(),
            "degree": self._degree,
            "x_mean": self._x_mean,
            "x_std": self._x_std,
            "sigma_local_T": self._sigma_local_T.tolist(),
            "sigma_local": self._sigma_local.tolist(),
            "vif": self._vif,
            "dof": self._dof,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UncertaintyEstimator:
        return cls(
            sandwich_cov=np.array(data["sandwich_cov"]),
            knots_std=np.array(data["knots_std"]),
            degree=data["degree"],
            x_mean=data["x_mean"],
            x_std=data["x_std"],
            sigma_local_T=np.array(data["sigma_local_T"]),
            sigma_local=np.array(data["sigma_local"]),
            vif=data["vif"],
            dof=data["dof"],
        )

    def __call__(
        self,
        x: np.ndarray,
        include_autocorr: bool = True,
        alpha: float = 0.1,
    ) -> np.ndarray:
        """Prediction interval half-width at each temperature.

        Parameters
        ----------
        x : ndarray
            Temperatures at which to evaluate uncertainty.
        include_autocorr : bool
            Whether to apply Bartlett VIF to the noise term.
        alpha : float
            Significance level (e.g. 0.1 for 90% PI).
        """
        x = np.asarray(x, dtype=float)

        # Model uncertainty: b(T)' Σ b(T) per observation
        x_std = (x - self._x_mean) / self._x_std
        B = BSpline.design_matrix(
            x_std, self._knots_std, self._degree, extrapolate=True,
        ).toarray()
        model_var = np.sum((B @ self._sandwich_cov) * B, axis=1)

        # Noise: interpolated local σ²(T) with log-linear tail extrapolation
        sigma_sq = _interpolate_sigma_sq(x, self._sigma_local_T, self._sigma_local)

        # VIF inflates model variance (autocorrelation means less info about
        # the curve), not noise variance (a new observation's deviation from
        # the true curve is independent of training autocorrelation).
        vif = self._vif if include_autocorr else 1.0
        total_var = vif * model_var + sigma_sq

        t_crit = t_dist.ppf(1 - alpha / 2, df=self._dof)
        return t_crit * np.sqrt(np.maximum(total_var, 0.0))


# ------------------------------------------------------------------
# Construction helper (called by fit_segment)
# ------------------------------------------------------------------

def build_uncertainty_estimator(
    solver,
    coefs: np.ndarray,
    V: np.ndarray,
    x: np.ndarray,
    y_s: np.ndarray,
    residuals: np.ndarray,
    weights: np.ndarray,
    y_std: float,
    x_mean: float,
    x_std: float,
    settings,
    n: int,
    ddof: int,
    time_sort: np.ndarray | None = None,
) -> UncertaintyEstimator:
    """Build an UncertaintyEstimator from solver output.

    Model covariance uses a soft monotonicity penalty (κ=N) instead of
    the hard constraint (κ=1e9) used for fitting.  This treats the
    monotonicity constraint as a Bayesian prior rather than a hard
    constraint, giving meaningful model variance that increases at
    edges where data is sparse.

    Noise scale uses a piecewise-linear spline on |residuals| to capture
    heteroscedasticity (HDD noisier than TIDD noisier than CDD) without
    the edge instability of kernel-weighted estimates.
    """
    B = solver.B
    w_sq = weights ** 2

    # --- Model covariance with soft monotonicity penalty ---
    # For fitting, κ=1e9 enforces hard monotonicity → Var(ŷ) ≈ 0.
    # For uncertainty, we use κ_unc=N (Bayesian interpretation: the
    # monotonicity prior is weighted proportionally to the data).
    # This gives meaningful model variance: wider at edges (less data
    # support), moderate in the interior, reasonable for extrapolation.
    A_data = B.T @ (B * w_sq[:, np.newaxis])
    kappa_unc = float(n)
    mono_penalty = kappa_unc * (solver.D1.T @ (solver.D1 * V.astype(float)[:, np.newaxis]))
    LHS_soft = A_data + solver._penalty_sum + solver._ridge + mono_penalty
    LHS_soft_inv = _safe_inv(LHS_soft)

    resid_std = y_s - B @ coefs
    sigma2_std = float(np.dot(w_sq, resid_std ** 2) / np.sum(w_sq))
    sandwich_cov = LHS_soft_inv @ A_data @ LHS_soft_inv * sigma2_std

    # Scale to original y-space
    sandwich_cov_orig = sandwich_cov * (y_std ** 2)

    # --- Global noise scale ---
    # Empirical quantile of |residuals| calibrated to the target coverage.
    # No distributional assumption — directly reflects the residual tail
    # behavior.  Divided by t_crit since the caller multiplies by it.
    # Heteroscedastic σ(T) is deferred to the YJ transform, which will
    # stabilize variance and make a global σ correct by construction.
    alpha = settings.uncertainty_alpha
    t_crit_cal = t_dist.ppf(1 - alpha / 2, df=max(n - ddof, 1))
    q_global = np.percentile(np.abs(residuals), 100 * (1 - alpha / 2))
    sigma_global = q_global / t_crit_cal
    sigma_local = np.full(n, max(sigma_global, 1e-10))

    # --- Bartlett VIF for residual autocorrelation ---
    if time_sort is not None:
        resid_time = residuals[time_sort]
    else:
        resid_time = residuals
    acf_K = _compute_acf_K(resid_time)
    vif = _bartlett_vif(resid_time, acf_K)

    return UncertaintyEstimator(
        sandwich_cov=sandwich_cov_orig,
        knots_std=solver.padded_knots,
        degree=solver.k,
        x_mean=x_mean,
        x_std=x_std,
        sigma_local_T=x,
        sigma_local=sigma_local,
        vif=vif,
        dof=n - ddof,
    )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _interpolate_sigma_sq(
    x_new: np.ndarray,
    x_train: np.ndarray,
    sigma_train: np.ndarray,
) -> np.ndarray:
    """Interpolate local σ² with log-linear tail extrapolation.

    Within training range: linear interpolation of σ².
    Beyond training range: log-linear extrapolation using slope from
    the last 10 training points (clamped to non-negative growth).
    """
    sigma_sq_train = sigma_train ** 2
    sigma_sq = np.interp(x_new, x_train, sigma_sq_train)

    n_tail = min(10, len(x_train) // 3)
    if n_tail < 2:
        return sigma_sq

    lo_mask = x_new < x_train[0]
    if np.any(lo_mask):
        log_s = np.log(np.maximum(sigma_sq_train[:n_tail], 1e-20))
        slope = min(np.polyfit(x_train[:n_tail], log_s, 1)[0], 0.0)
        log_extrap = np.log(max(sigma_sq_train[0], 1e-20)) + slope * (x_new[lo_mask] - x_train[0])
        sigma_sq[lo_mask] = np.exp(log_extrap)

    hi_mask = x_new > x_train[-1]
    if np.any(hi_mask):
        log_s = np.log(np.maximum(sigma_sq_train[-n_tail:], 1e-20))
        slope = max(np.polyfit(x_train[-n_tail:], log_s, 1)[0], 0.0)
        log_extrap = np.log(max(sigma_sq_train[-1], 1e-20)) + slope * (x_new[hi_mask] - x_train[-1])
        sigma_sq[hi_mask] = np.exp(log_extrap)

    return sigma_sq


def _compute_acf_K(residuals: np.ndarray, max_K: int = 14) -> int:
    """Lag at which ACF drops below significance, capped at max_K."""
    n = len(residuals)
    if n < 10:
        return 0
    threshold = 1.96 / np.sqrt(n)
    r = residuals - np.mean(residuals)
    var = np.dot(r, r)
    if var < 1e-12:
        return 0
    for k in range(1, min(max_K + 1, n // 3)):
        if abs(np.dot(r[k:], r[:-k]) / var) < threshold:
            return k - 1
    return max_K


def _bartlett_vif(residuals: np.ndarray, K: int) -> float:
    """Variance inflation factor via Bartlett formula.

    VIF = 1 + 2 Σ_{k=1}^{K} (1 - k/n) ρ(k)
    """
    n = len(residuals)
    if K == 0 or n < 10:
        return 1.0
    r = residuals - np.mean(residuals)
    var = np.dot(r, r)
    if var < 1e-12:
        return 1.0
    vif = 1.0
    for k in range(1, K + 1):
        vif += 2 * (1 - k / n) * np.dot(r[k:], r[:-k]) / var
    return max(1.0, vif)


def _safe_inv(A: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A)
