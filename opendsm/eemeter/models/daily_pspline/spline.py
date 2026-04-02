"""Lightweight fitted P-spline for prediction and serialization.

Uses composition with scipy.interpolate.BSpline. The internal BSpline
operates in original (destandardized) space — knots and coefficients
are transformed at construction time so that __call__, derivative,
antiderivative, and integrate all delegate directly without scaling.
"""

from __future__ import annotations

import json
from functools import cached_property
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy.interpolate import BSpline

from opendsm.common.metrics import BaselineMetrics, BaselineMetricsFromDict
from opendsm.eemeter.models.daily_pspline.uncertainty import UncertaintyEstimator


class PSplineSchema(BaseModel):
    """Pydantic schema for PSpline serialization.

    Stores standardization parameters and standardized knots/coefficients
    so the destandardized BSpline can be reconstructed exactly.
    """

    n_min: int = Field(..., description="Minimum points per zone")
    lambda_smoothing: float = Field(..., description="Smoothing parameter")
    lambda_curvature: float = Field(0.0, description="Second-derivative curvature penalty")
    kappa_penalty: float = Field(..., description="Monotonicity penalty")
    maxiter: int = Field(..., description="Maximum iterations")

    x_mean: float = Field(..., description="Mean of x training data")
    x_std: float = Field(..., description="Std of x training data")
    y_mean: float = Field(..., description="Mean of y training data")
    y_std: float = Field(..., description="Std of y training data")

    knots: List[float] = Field(..., description="Standardized knot vector including padding")
    coefficients: List[float] = Field(..., description="Standardized spline coefficients")
    degree: int = Field(..., description="Spline degree")
    extrapolate: bool = Field(..., description="Extrapolation flag")
    bc_type: Optional[str] = Field(None, description="Boundary condition type")

    breakpoints: List[float] = Field(..., description="Zone breakpoints [lower, upper]")
    fit_bounds: List[float] = Field(..., description="Training data bounds [min, max]")

    slope_threshold_pct: float = Field(0.05, description="Effective balance point slope threshold")

    training_metrics: Optional[dict] = Field(None, description="Training metrics")
    uncertainty: Optional[dict] = Field(None, description="UncertaintyEstimator state")



def _destandardize(t_std, c_std, x_mean, x_std, y_mean, y_std):
    """Transform standardized knots/coefficients to original space.

    B-splines are affine-invariant: B_i((x - μ_x)/σ_x; t_std) = B_i(x; t_std*σ_x + μ_x).
    Partition of unity gives sum(B_i(x)) = 1, so a constant y_mean can be
    absorbed into the coefficients: c_orig = c_std * y_std + y_mean.
    """
    t_orig = np.asarray(t_std) * x_std + x_mean
    c_orig = np.asarray(c_std) * y_std + y_mean
    return t_orig, c_orig


def predict_raw(
    knots_std: np.ndarray,
    coefs_std: np.ndarray,
    degree: int,
    x: np.ndarray,
    x_mean: float,
    x_std: float,
    y_mean: float,
    y_std: float,
    fit_bnds: np.ndarray,
    bc_type: str | None,
) -> np.ndarray:
    """Evaluate a destandardized P-spline with extrapolation.

    Standalone function used by ``fit_segment`` to compute predictions
    before PSpline construction.
    """
    t_orig, c_orig = _destandardize(knots_std, coefs_std, x_mean, x_std, y_mean, y_std)
    spl = BSpline(t_orig, c_orig, degree, extrapolate=True)
    x = np.asarray(x, dtype=float)
    y = spl(x)

    if bc_type is None:
        return y

    lo_mask = x < fit_bnds[0]
    hi_mask = x > fit_bnds[1]
    y_lo = float(spl(fit_bnds[0]))
    y_hi = float(spl(fit_bnds[1]))

    if bc_type == "clamped":
        y = np.where(lo_mask, y_lo, np.where(hi_mask, y_hi, y))
    elif bc_type == "natural" and degree >= 1:
        dspl = spl.derivative(1)
        dy_lo = float(dspl(fit_bnds[0]))
        dy_hi = float(dspl(fit_bnds[1]))
        lo_extrap = y_lo + dy_lo * (x - fit_bnds[0])
        hi_extrap = y_hi + dy_hi * (x - fit_bnds[1])
        y = np.where(lo_mask, lo_extrap, np.where(hi_mask, hi_extrap, y))

    return y


class PSpline:
    """Fitted penalized B-spline for energy load prediction.

    Wraps a scipy BSpline operating in original (destandardized) space.
    Derivative, antiderivative, and integrate delegate directly to the
    internal BSpline — no scaling or linear correction needed.

    Constructed by ``fit_segment`` — not by users directly.

    Parameters
    ----------
    knots_std : ndarray
        Standardized knot vector (with boundary padding).
    coefs_std : ndarray
        Standardized spline coefficients.
    degree : int
        B-spline degree.
    x_mean, x_std, y_mean, y_std : float
        Standardization parameters (stored for serialization).
    bp : ndarray
        Zone breakpoints [lower, upper] in original space.
    fit_bnds : ndarray
        Training data bounds [min, max] in original space.
    bc_type : str or None
        Boundary condition type ('natural', 'clamped', or None).
    config : dict
        Fitting hyperparameters (n_min, lambda_smoothing, kappa_penalty, maxiter).
    training_metrics : BaselineMetrics or None
        Fit quality metrics computed on training data.
    """

    def __init__(
        self,
        knots_std: np.ndarray,
        coefs_std: np.ndarray,
        degree: int,
        x_mean: float,
        x_std: float,
        y_mean: float,
        y_std: float,
        bp: np.ndarray,
        fit_bnds: np.ndarray,
        bc_type: str | None,
        config: dict,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        training_metrics: BaselineMetrics | None = None,
        uncertainty: UncertaintyEstimator | None = None,
    ):
        # Store standardized params for serialization
        self._knots_std = np.asarray(knots_std)
        self._coefs_std = np.asarray(coefs_std)
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

        # Destandardized BSpline in original space
        t_orig, c_orig = _destandardize(knots_std, coefs_std, x_mean, x_std, y_mean, y_std)
        self._spl = BSpline(t_orig, c_orig, degree, extrapolate=True)

        self.t = t_orig
        self.c = c_orig
        self.k = degree
        self.bp = np.asarray(bp)
        self.fit_bnds = np.asarray(fit_bnds)
        self.bc_type = bc_type
        self.config = config
        self.x = x
        self.y = y
        self.training_metrics = training_metrics
        self.uncertainty = uncertainty

        # Precompute boundary values for extrapolation
        if bc_type is not None:
            lo, hi = fit_bnds
            self._y_lo = float(self._spl(lo))
            self._y_hi = float(self._spl(hi))
            if bc_type == "natural" and degree >= 1:
                dspl = self._spl.derivative(1)
                self._dy_lo = float(dspl(lo))
                self._dy_hi = float(dspl(hi))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = self._spl(x)

        if self.bc_type is None:
            return y

        x_lo, x_hi = self.fit_bnds
        lo = x < x_lo
        hi = x > x_hi

        if self.bc_type == "clamped":
            y = np.where(lo, self._y_lo, np.where(hi, self._y_hi, y))

        elif self.bc_type == "natural" and self.k >= 1:
            lo_extrap = self._y_lo + self._dy_lo * (x - x_lo)
            hi_extrap = self._y_hi + self._dy_hi * (x - x_hi)
            y = np.where(lo, lo_extrap, np.where(hi, hi_extrap, y))

        elif self.bc_type == "antideriv":
            # Antiderivative of extrapolated regions:
            #   Clamped: ∫c dx = c*(x - x_lo) + AD(x_lo)
            #   Natural: ∫(a + b*dx) dx = a*dx + b*dx²/2 + AD(x_lo)
            dx_lo = x - x_lo
            dx_hi = x - x_hi
            if self._ad_parent_bc == "clamped":
                lo_val = self._ad_val_lo + self._ad_y_lo * dx_lo
                hi_val = self._ad_val_hi + self._ad_y_hi * dx_hi
            else:  # natural
                lo_val = self._ad_val_lo + self._ad_y_lo * dx_lo + 0.5 * self._ad_dy_lo * dx_lo ** 2
                hi_val = self._ad_val_hi + self._ad_y_hi * dx_hi + 0.5 * self._ad_dy_hi * dx_hi ** 2
            y = np.where(lo, lo_val, np.where(hi, hi_val, y))

        return y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self(x)

    @cached_property
    def eff_bp(self) -> np.ndarray:
        """Effective balance points where slope becomes physically meaningful.

        Scans outward from fitted BPs to find where |f'(T)| first exceeds
        ``slope_threshold_pct * max(|f'|)``.
        """
        pct = self.config.get("slope_threshold_pct", 0.05)
        dspl = self.derivative()
        lo, hi = self.fit_bnds
        bp = self.bp

        T_eval = np.linspace(lo, hi, 500)
        abs_slopes = np.abs(dspl(T_eval))
        max_slope = np.max(abs_slopes)

        if max_slope < 1e-12:
            return bp.copy()

        threshold = pct * max_slope

        # HDD: scan from bp[0] toward lo, find first T where |f'| > threshold
        eff_bp0 = bp[0]
        if bp[0] - lo > 1e-6:
            hdd_mask = T_eval < bp[0]
            if np.any(hdd_mask):
                hdd_T = T_eval[hdd_mask][::-1]  # bp[0] toward lo
                hdd_slopes = abs_slopes[hdd_mask][::-1]
                above = np.where(hdd_slopes > threshold)[0]
                if len(above) > 0:
                    eff_bp0 = float(hdd_T[above[0]])

        # CDD: scan from bp[1] toward hi, find first T where |f'| > threshold
        eff_bp1 = bp[1]
        if hi - bp[1] > 1e-6:
            cdd_mask = T_eval > bp[1]
            if np.any(cdd_mask):
                cdd_T = T_eval[cdd_mask]  # bp[1] toward hi
                cdd_slopes = abs_slopes[cdd_mask]
                above = np.where(cdd_slopes > threshold)[0]
                if len(above) > 0:
                    eff_bp1 = float(cdd_T[above[0]])

        return np.array([eff_bp0, eff_bp1])

    def prediction_uncertainty(
        self,
        x: np.ndarray,
        include_autocorr: bool | None = None,
        alpha: float | None = None,
    ) -> np.ndarray:
        """Prediction interval half-width at each temperature.

        Parameters
        ----------
        x : ndarray
            Temperatures.
        include_autocorr : bool or None
            Whether to apply Bartlett VIF. Defaults to config value.
        alpha : float or None
            Significance level. Defaults to config value.
        """
        if self.uncertainty is None:
            raise RuntimeError(
                "prediction_uncertainty requires an UncertaintyEstimator from fit_segment"
            )
        if include_autocorr is None:
            include_autocorr = self.config.get("include_autocorr", True)
        if alpha is None:
            alpha = self.config.get("uncertainty_alpha", 0.1)
        return self.uncertainty(x, include_autocorr=include_autocorr, alpha=alpha)

    def derivative(self, nu: int = 1) -> PSpline:
        """Return the nu-th derivative, respecting extrapolation.

        Clamped: 0 outside fit bounds (constant extrapolation).
        Natural: dy_lo / dy_hi (constant) for nu=1, 0 for nu>=2.
        """
        result = PSpline.__new__(PSpline)
        result._spl = self._spl.derivative(nu=nu)
        result.t = result._spl.t
        result.c = result._spl.c
        result.k = result._spl.k
        result.bp = self.bp
        result.fit_bnds = self.fit_bnds
        result.config = self.config
        result.training_metrics = None

        if self.bc_type == "clamped":
            # d/dx(constant) = 0
            result.bc_type = "clamped"
            result._y_lo = 0.0
            result._y_hi = 0.0
        elif self.bc_type == "natural" and nu == 1:
            # d/dx(a + b*x) = b (constant)
            result.bc_type = "clamped"
            result._y_lo = self._dy_lo
            result._y_hi = self._dy_hi
        elif self.bc_type == "natural" and nu >= 2:
            # d²/dx²(a + b*x) = 0
            result.bc_type = "clamped"
            result._y_lo = 0.0
            result._y_hi = 0.0
        elif self.bc_type == "antideriv" and nu == 1:
            # Derivative of antiderivative recovers original extrapolation
            if self._ad_parent_bc == "clamped":
                # d/dx(y_lo * x + C) = y_lo (constant)
                result.bc_type = "clamped"
                result._y_lo = self._ad_y_lo
                result._y_hi = self._ad_y_hi
            else:
                # d/dx(y_lo*x + dy_lo*x²/2 + C) = y_lo + dy_lo*x (linear)
                result.bc_type = "natural"
                result._y_lo = self._ad_y_lo
                result._y_hi = self._ad_y_hi
                result._dy_lo = self._ad_dy_lo
                result._dy_hi = self._ad_dy_hi
        else:
            result.bc_type = None

        return result

    def antiderivative(self, nu: int = 1) -> PSpline:
        """Return the nu-th antiderivative, respecting extrapolation.

        The antiderivative is anchored so that AD(fit_bnds[0]) matches
        the interior spline's antiderivative at that point.
        """
        result = PSpline.__new__(PSpline)
        result._spl = self._spl.antiderivative(nu=nu)
        result.t = result._spl.t
        result.c = result._spl.c
        result.k = result._spl.k
        result.bp = self.bp
        result.fit_bnds = self.fit_bnds
        result.config = self.config
        result.training_metrics = None

        if self.bc_type in ("clamped", "natural"):
            # Store boundary info so __call__ can integrate the extrapolation
            result.bc_type = "antideriv"
            result._ad_parent_bc = self.bc_type
            result._ad_y_lo = self._y_lo
            result._ad_y_hi = self._y_hi
            result._ad_dy_lo = getattr(self, "_dy_lo", 0.0)
            result._ad_dy_hi = getattr(self, "_dy_hi", 0.0)
            # Anchor: value of interior antiderivative at boundaries
            result._ad_val_lo = float(result._spl(self.fit_bnds[0]))
            result._ad_val_hi = float(result._spl(self.fit_bnds[1]))
        else:
            result.bc_type = None

        return result

    def integrate(self, a: float, b: float) -> float:
        """Definite integral from a to b, respecting extrapolation."""
        ad = self.antiderivative()
        return float(ad(np.array([b]))[0] - ad(np.array([a]))[0])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        data = {
            "n_min": self.config["n_min"],
            "lambda_smoothing": self.config["lambda_smoothing"],
            "lambda_curvature": self.config.get("lambda_curvature", 0.0),
            "kappa_penalty": self.config["kappa_penalty"],
            "maxiter": self.config["maxiter"],
            "slope_threshold_pct": self.config.get("slope_threshold_pct", 0.05),
            "x_mean": float(self.x_mean),
            "x_std": float(self.x_std),
            "y_mean": float(self.y_mean),
            "y_std": float(self.y_std),
            "knots": self._knots_std.tolist(),
            "coefficients": self._coefs_std.tolist(),
            "degree": self.k,
            "extrapolate": True,
            "bc_type": self.bc_type,
            "breakpoints": self.bp.tolist(),
            "fit_bounds": self.fit_bnds.tolist(),
            "training_metrics": (
                self.training_metrics.model_dump()
                if self.training_metrics is not None else None
            ),
            "uncertainty": (
                self.uncertainty.to_dict()
                if self.uncertainty is not None else None
            ),
        }
        return PSplineSchema(**data).model_dump()

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> PSpline:
        schema = PSplineSchema(**data)
        spl = cls(
            knots_std=np.array(schema.knots),
            coefs_std=np.array(schema.coefficients),
            degree=schema.degree,
            x_mean=schema.x_mean,
            x_std=schema.x_std,
            y_mean=schema.y_mean,
            y_std=schema.y_std,
            bp=np.array(schema.breakpoints),
            fit_bnds=np.array(schema.fit_bounds),
            bc_type=schema.bc_type,
            config={
                "n_min": schema.n_min,
                "lambda_smoothing": schema.lambda_smoothing,
                "lambda_curvature": schema.lambda_curvature,
                "kappa_penalty": schema.kappa_penalty,
                "maxiter": schema.maxiter,
                "slope_threshold_pct": schema.slope_threshold_pct,
            },
            training_metrics=(
                BaselineMetricsFromDict(schema.training_metrics)
                if schema.training_metrics is not None else None
            ),
        )
        if schema.uncertainty is not None:
            spl.uncertainty = UncertaintyEstimator.from_dict(schema.uncertainty)
        return spl

    @classmethod
    def from_json(cls, json_str: str) -> PSpline:
        return cls.from_dict(json.loads(json_str))



