"""Prediction uncertainty estimation for fitted P-splines.

Model covariance uses a soft monotonicity penalty (κ=N) instead of the
hard constraint (κ=1e9) from fitting, treating monotonicity as a
Bayesian prior rather than a hard constraint.  This gives meaningful
model variance that widens at data-sparse edges.

Noise quantiles are estimated via robust Yeo-Johnson transform on the
residuals.  In transformed space the residuals are approximately normal,
so symmetric z·σ intervals are correct; back-transforming produces
asymmetric intervals that are wider upward (usage spikes) and tighter
downward — matching the physical reality of energy data.

Bartlett VIF for temporal autocorrelation is applied to the model
variance component (curve uncertainty), not to the noise component
(individual observation variance is independent of training ACF).

References
----------
Ruppert, D., Wand, M. P. & Carroll, R. J. (2003). *Semiparametric
    Regression*. Cambridge University Press.  (Soft-penalty covariance.)

Raymaekers, J. & Rousseeuw, P. J. (2021). Transforming variables to
    central normality. *Machine Learning* 110, 2375–2415.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.stats import t as t_dist

from opendsm.common.stats.distribution_transform import YeoJohnson



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
        vif: float,
        dof: int,
        yj: YeoJohnson,
        sigma_scale_T: np.ndarray,
        sigma_scale: np.ndarray,
    ):
        self._sandwich_cov = sandwich_cov
        self._knots_std = knots_std
        self._degree = degree
        self._x_mean = x_mean
        self._x_std = x_std
        self._vif = vif
        self._dof = max(1, dof)
        self._yj = yj
        self._sigma_scale_T = sigma_scale_T
        self._sigma_scale = sigma_scale

    def to_dict(self) -> dict:
        return {
            "sandwich_cov": self._sandwich_cov.tolist(),
            "knots_std": self._knots_std.tolist(),
            "degree": self._degree,
            "x_mean": self._x_mean,
            "x_std": self._x_std,
            "vif": self._vif,
            "dof": self._dof,
            "yj": self._yj.to_dict(),
            "sigma_scale_T": self._sigma_scale_T.tolist(),
            "sigma_scale": self._sigma_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> UncertaintyEstimator:
        return cls(
            sandwich_cov=np.array(data["sandwich_cov"]),
            knots_std=np.array(data["knots_std"]),
            degree=data["degree"],
            x_mean=data["x_mean"],
            x_std=data["x_std"],
            vif=data["vif"],
            dof=data["dof"],
            yj=YeoJohnson.from_dict(data["yj"]),
            sigma_scale_T=np.array(data["sigma_scale_T"]),
            sigma_scale=np.array(data["sigma_scale"]),
        )

    def __call__(
        self,
        x: np.ndarray,
        predicted: np.ndarray,
        include_autocorr: bool = True,
        alpha: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Asymmetric prediction interval bounds via YJ back-transform.

        In YJ-transformed residual space, the interval is symmetric:
        [μ_t - z·σ_total, μ_t + z·σ_total].  Back-transforming produces
        asymmetric bounds in original space.

        Parameters
        ----------
        x : ndarray
            Temperatures at which to evaluate uncertainty.
        predicted : ndarray
            Predicted values at each temperature.
        include_autocorr : bool
            Whether to apply Bartlett VIF to model variance.
        alpha : float
            Significance level (e.g. 0.1 for 90% PI).

        Returns
        -------
        lower, upper : ndarray, ndarray
            Lower and upper prediction interval bounds.
        """
        x = np.asarray(x, dtype=float)

        # Model uncertainty: b(T)' Σ b(T) per observation
        x_std = (x - self._x_mean) / self._x_std
        B = BSpline.design_matrix(
            x_std, self._knots_std, self._degree, extrapolate=True,
        ).toarray()
        model_var = np.sum((B @ self._sandwich_cov) * B, axis=1)

        # VIF inflates model variance (autocorrelation means less info about
        # the curve), not noise variance.
        vif = self._vif if include_autocorr else 1.0
        model_std = np.sqrt(np.maximum(vif * model_var, 0.0))

        t_crit = t_dist.ppf(1 - alpha / 2, df=self._dof)

        # Noise shape from YJ (asymmetric, global).
        # The interval is centered at YJ(0) — a residual of zero (perfect
        # prediction) — not at the origin of transformed space.
        center = self._yj.transform(np.array([[0.0]]))[0, 0]
        noise_lower = self._yj.inverse_transform(np.array([[center - t_crit]]))[0, 0]
        noise_upper = self._yj.inverse_transform(np.array([[center + t_crit]]))[0, 0]

        # Noise scale from smoothing spline (heteroscedastic)
        scale = np.interp(x, self._sigma_scale_T, self._sigma_scale)

        # Total: prediction + scaled noise + model uncertainty
        lower = predicted + scale * noise_lower - t_crit * model_std
        upper = predicted + scale * noise_upper + t_crit * model_std

        return lower, upper


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

    # --- Noise shape via YJ transform ---
    # Fit robust Yeo-Johnson on residuals to normalize them.  The class
    # post-standardizes so the output is approximately N(0, 1).  Symmetric
    # ±z intervals in transformed space back-transform to asymmetric
    # intervals in original space (wider upward for right-skewed data).
    yj = YeoJohnson()
    yj.fit(residuals.reshape(-1, 1))

    # Clamp λ to [-0.5, 1.0].  Outside this range the transform is too
    # aggressive — very negative λ compresses the positive tail so
    # heavily that the inverse becomes non-monotone in the standardized
    # domain.  λ=1 is identity; λ=0 is log(1+x); λ<0 is strong
    # compression.  The range [-0.5, 1.0] captures meaningful asymmetry
    # without numerical instability.
    lam = yj.lambdas_[0]
    lam_clamped = float(np.clip(lam, -0.5, 1.0))
    if lam_clamped != lam:
        yj.lambdas_ = np.array([lam_clamped])
        # Re-fit post-standardization with clamped lambda
        from opendsm.common.stats.distribution_transform.yeo_johnson import yj_transform
        d = yj.to_dict()
        x_pre = (residuals - d["pre_mu"][0]) / d["pre_sigma"][0]
        rt = yj_transform(x_pre, lam_clamped)
        yj.post_mu_ = np.array([float(np.median(rt))])
        yj.post_sigma_ = np.array([float(np.std(rt))])

    # --- Noise scale via smoothing spline on |residuals| ---
    # The YJ provides the shape (asymmetry); the scale spline provides
    # the heteroscedasticity (noise level varies with temperature).
    # We store the ratio σ_local/σ_global so the YJ-derived noise bounds
    # get scaled up/down by local noise level.
    sigma_scale = _fit_sigma_scale(x, residuals)

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
        vif=vif,
        dof=n - ddof,
        yj=yj,
        sigma_scale_T=x,
        sigma_scale=sigma_scale,
    )


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _fit_sigma_scale(
    x: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """Relative noise scale σ(T)/σ_global via smoothing spline on |r|.

    Returns the ratio at each training temperature. A value > 1 means
    noisier than average; < 1 means quieter.  The smoothing spline
    captures the broad heteroscedastic trend without chasing individual
    residuals.  The ratio is clipped to [0.5, 2.0] to prevent extreme
    scaling at sparse edges.
    """
    from scipy.interpolate import UnivariateSpline

    abs_r = np.abs(residuals)
    sigma_global = np.median(abs_r) * 1.4826
    if sigma_global < 1e-10:
        return np.ones_like(x)

    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    abs_r_sorted = abs_r[sort_idx]

    n = len(x)
    try:
        # Heavy smoothing: s = 2 * n * var(|r|) produces a very gentle
        # trend with few effective knots, preventing sharp transitions.
        spl = UnivariateSpline(
            x_sorted, abs_r_sorted,
            k=3, s=2 * n * np.var(abs_r_sorted),
        )
        sigma_local = np.maximum(spl(x), 1e-10) * np.sqrt(np.pi / 2)
        scale = sigma_local / sigma_global
    except Exception:
        return np.ones_like(x)

    return np.clip(scale, 0.5, 2.0)


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
