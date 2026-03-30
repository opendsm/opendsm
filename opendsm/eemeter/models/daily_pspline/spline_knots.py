from __future__ import annotations

import numpy as np

from scipy.interpolate import UnivariateSpline
from scipy.integrate import cumulative_trapezoid

from opendsm.common.stats.adaptive_loss import adaptive_weights


def check_n_knots(n_knots):
    if not isinstance(n_knots, int):
        raise ValueError("n_knots must be an integer.")
    if n_knots < 0:
        raise ValueError("n_knots must be non-negative.")


# Number of iterative-reweighting passes for the robust smoothing spline.
# 3 passes are sufficient to suppress outlier-driven curvature artifacts
# without adding meaningful latency (~1-2 ms total for typical data).
_ROBUST_SPLINE_ITERATIONS = 3


class Knots:
    def __init__(
            self,
            x,
            y,
            w=None,
            spline_interp_count=200,  # Reduced from 1000 for 5x speedup
            spline_lambda=10,
            n_min=5,
            min_knot_spacing_pct=0.025,
            # Optional parameters used by optimize_knots
            bspline_degree=None,
            lambda_smoothing=None,
            kappa_penalty=None,
            maxiter=None,
            lightweight=False,
        ):

        self.spline_interp_count = spline_interp_count
        self.spline_lambda = spline_lambda
        self.n_min = n_min
        self.min_knot_spacing_pct = min_knot_spacing_pct
        self.spl = None

        self._x_data = x
        self._y_data = y
        self.weights = w

        if lightweight:
            # Lightweight mode: skip smoothing spline entirely.
            # Used when the Knots object is only needed for get_internal_knots
            # with zone_knot_count=0 (TIDD uniform knots only — no Yeh placement).
            self.spl_x = np.linspace(x[0], x[-1], spline_interp_count)
        else:
            # Fit a robust smoothing spline via iterative reweighting.
            # The unconstrained UnivariateSpline can oscillate through noisy data,
            # creating spurious curvature peaks that attract knots to noise rather
            # than to the true elbows of the heating/cooling curve.  Reweighting
            # with adaptive_weights (MAD-based) downweights outliers before
            # curvature is extracted, producing a cleaner signal for Yeh placement.
            s_param = len(x) * self.spline_lambda
            robust_w = w.copy() if w is not None else np.ones(len(x), dtype=float)
            spl = UnivariateSpline(x, y, w=robust_w, s=s_param, k=3)
            for _ in range(_ROBUST_SPLINE_ITERATIONS):
                resid = y - spl(x)
                a_w, _, _ = adaptive_weights(resid, alpha="adaptive", C_algo="mad")
                robust_w = robust_w * a_w
                robust_w = np.clip(robust_w, 0.01, None)
                spl = UnivariateSpline(x, y, w=robust_w, s=s_param, k=3)
            self.spl = spl
            self.spl_x = np.linspace(x[0], x[-1], spline_interp_count)

        # Store fit parameters for optimize_knots
        self.bspline_degree = bspline_degree
        self.lambda_smoothing = lambda_smoothing
        self.kappa_penalty = kappa_penalty
        self.maxiter = maxiter

        # Cache derivative spline for _yeh_knot_placement (avoids reconstruction per call)
        p = int(min(3, bspline_degree)) if bspline_degree is not None else 3
        self._yeh_p = p
        self._deriv_spl = self.spl.derivative(p) if (self.spl is not None and p > 0) else None

        # Cache for get_internal_knots keyed on (bp, n_knots, n_min, n_knots_hdd, n_knots_cdd)
        self._knot_cache: dict = {}


    def _yeh_knot_placement_og(self, n_knots, spl_x=None):
        """Fast Automatic Knot Placement Method for Accurate B-spline Curve Fitting
        https://doi.org/10.1016/j.cad.2020.102905

        Ensures that each interval between consecutive knots contains at least
        n_min data points. If not, reduces n_knots and tries again.

        Args:
            n_knots (_type_): number of interior knots to place
            spl_x: optional x values for spline evaluation

        Returns:
            x_knot: _array of knot x-positions
        """

        if self.spl is None:
            raise ValueError("Smoothing spline has not been fit yet.")

        if spl_x is None:
            spl_x = self.spl_x

        p = min(3.0, self.bspline_degree)

        # UnivariateSpline uses derivative(n) instead of derivative(nu=n)
        features = np.power(np.abs(self.spl.derivative(int(p))(spl_x)), 1/p)
        cum_features = cumulative_trapezoid(features, spl_x, initial=0)

        y_knot = np.linspace(0, cum_features[-1], n_knots + 2)
        x_knot = np.interp(y_knot, cum_features, spl_x)

        return np.array(x_knot)


    def _yeh_knot_placement(self, n_knots, spl_x=None):
        """Fast Automatic Knot Placement Method for Accurate B-spline Curve Fitting
        https://doi.org/10.1016/j.cad.2020.102905

        Ensures that each interval between consecutive knots contains at least
        n_min data points. If not, reduces n_knots and tries again.

        Args:
            n_knots: number of interior knots to place
            spl_x: optional x values for spline evaluation

        Returns:
            x_knot: array of knot x-positions
        """

        if self.spl is None:
            raise ValueError("Smoothing spline has not been fit yet.")

        if spl_x is None:
            spl_x = self.spl_x

        p = self._yeh_p

        # Compute curvature-based cumulative feature distribution
        if p == 0:
            features = np.abs(self.spl(spl_x))
        else:
            features = np.power(np.abs(self._deriv_spl(spl_x)), 1.0 / p)

        # Trapezoidal cumsum on uniform grid (avoids cumulative_trapezoid overhead)
        dx = spl_x[1] - spl_x[0]
        cum_features = np.empty(len(features))
        cum_features[0] = 0.0
        np.cumsum(0.5 * dx * (features[:-1] + features[1:]), out=cum_features[1:])

        # Vectorized initial placement via equal-area subdivision
        y_targets = np.linspace(0, cum_features[-1], n_knots + 2)
        x_knots_init = np.interp(y_targets, cum_features, spl_x)

        # Post-process: enforce n_min data points between knots and min spacing
        x_data = self._x_data
        n_data = len(x_data)
        min_spacing = self.min_knot_spacing_pct * (x_data[-1] - x_data[0])
        n_min = self.n_min
        x_end = spl_x[-1]

        # Batch searchsorted for all initial candidates (one vectorized call)
        init_pos = x_data.searchsorted(x_knots_init, side='left')

        # Pre-allocate output array instead of growing a Python list
        x_knots = np.empty(len(x_knots_init))
        x_knots[0] = x_knots_init[0]
        n_out = 1
        left_idx = init_pos[0]

        for i in range(1, len(x_knots_init) - 1):
            x_cand = x_knots_init[i]
            right_idx = init_pos[i]
            n_points = right_idx - left_idx
            adjusted = False

            if n_points < n_min:
                n_remaining = n_data - left_idx
                if n_remaining < n_min:
                    break

                x_data_min = x_data[left_idx + n_min - 1]
                if x_data_min > x_cand:
                    x_cand = x_data_min
                    adjusted = True

            prev_plus_spacing = x_knots[n_out - 1] + min_spacing
            if x_cand < prev_plus_spacing:
                x_cand = prev_plus_spacing
                adjusted = True

            if x_cand >= x_end:
                break

            x_knots[n_out] = x_cand
            n_out += 1

            # Only recompute searchsorted when candidate was adjusted
            if adjusted:
                left_idx = x_data.searchsorted(x_cand, side='left')
            else:
                left_idx = right_idx

        x_knots[n_out] = x_end
        n_out += 1

        return x_knots[:n_out]
    

    def initial_guess(self, n_knots=None, spl_x=None):
        check_n_knots(n_knots)

        self.x = self._yeh_knot_placement(n_knots, spl_x=spl_x)
        self.n = n_knots

        return self


    def get_internal_knots(self, bp=None, n_knots=10, n_min=5, n_knots_hdd=None, n_knots_cdd=None):
        """Generate internal knots with zone-specific density.

        Places knots adaptively based on data density in HDD/TIDD/CDD zones.
        Ensures that HDD and CDD zones each have at least n_min data points,
        otherwise expands TIDD zone to include them.

        If ``_refined_knots`` has been set (by constrained-curvature refinement),
        returns that vector when the requested bp matches the refinement bp.

        Parameters
        ----------
        bp : array-like or None
            Breakpoints [lower, upper] defining the TIDD zone.
            If None, will be estimated automatically.
        n_knots : int
            Target number of knots per zone (TIDD and fallback for HDD/CDD).
        n_min : int
            Minimum number of data points required in HDD and CDD zones.
            Zones with fewer points will be absorbed into TIDD.
        n_knots_hdd : int or None
            Override for the HDD zone knot count. Defaults to ``n_knots``.
        n_knots_cdd : int or None
            Override for the CDD zone knot count. Defaults to ``n_knots``.

        Returns
        -------
        knots : ndarray
            Internal knot vector (without padding).
        """
        # If constrained-curvature refinement has provided a knot vector and
        # the requested bp matches, use it directly.
        refined = getattr(self, '_refined_knots', None)
        if refined is not None:
            refined_bp = getattr(self, '_refined_bp', None)
            if refined_bp is not None and np.allclose(bp, refined_bp, atol=1e-8):
                return refined

        if n_knots_hdd is None:
            n_knots_hdd = n_knots
        if n_knots_cdd is None:
            n_knots_cdd = n_knots

        cache_key = (round(bp[0], 10), round(bp[1], 10), n_knots, n_min, n_knots_hdd, n_knots_cdd)
        cached = self._knot_cache.get(cache_key)
        if cached is not None:
            return cached

        n_refinement_pts = max(50, 20 * max(n_knots_hdd, n_knots_cdd, n_knots))
        n_hdd_points = np.sum(self._x_data < bp[0])
        n_cdd_points = np.sum(self._x_data > bp[1])

        # HDD zone knots
        hdd_knots = np.array([])
        if n_hdd_points >= n_min:
            n_knots_hdd = min(n_knots_hdd, n_hdd_points // n_min)
            hdd_spl_x = np.linspace(self._x_data[0], bp[0], n_refinement_pts)
            hdd_knots = self._yeh_knot_placement(n_knots_hdd, spl_x=hdd_spl_x)
            hdd_knots = hdd_knots[hdd_knots < bp[0]]

        # TIDD zone knots (uniform spacing; need interior knots so derivative
        # positions fall inside the zone for monotonicity enforcement)
        if np.isclose(bp[0], bp[1]):
            tidd_knots = np.array([bp[0]])
        else:
            tidd_knots = np.linspace(bp[0], bp[1], n_knots + 2)

        # CDD zone knots
        cdd_knots = np.array([])
        if n_cdd_points >= n_min:
            n_knots_cdd = min(n_knots_cdd, n_cdd_points // n_min)
            cdd_spl_x = np.linspace(bp[1], self._x_data[-1], n_refinement_pts)
            cdd_knots = self._yeh_knot_placement(n_knots_cdd, spl_x=cdd_spl_x)
            cdd_knots = cdd_knots[cdd_knots > bp[1]]

        internal_knots = np.hstack([hdd_knots, tidd_knots, cdd_knots])

        # Ensure the knot vector spans the data range and has at least 2
        # distinct values.  BSpline requires unique(t[k:n+1]) >= 2.
        # This can happen when bp is collapsed (no TIDD zone) and no
        # HDD/CDD knots are requested — physically valid (direct heating-
        # to-cooling transition) but needs data-bound anchors.
        x_lo, x_hi = self._x_data[0], self._x_data[-1]
        if len(internal_knots) == 0 or np.ptp(internal_knots) < 1e-8:
            # Build from data bounds + bp, deduplicate with tolerance
            candidates = np.sort(np.concatenate([[x_lo], internal_knots, [x_hi]]))
            # Remove near-duplicates (within 1e-8)
            mask = np.concatenate([[True], np.diff(candidates) > 1e-8])
            internal_knots = candidates[mask]
            # If still only 1 unique value, nudge to create a span
            if len(internal_knots) < 2:
                eps = max(abs(internal_knots[0]) * 1e-6, 1e-10)
                internal_knots = np.array([internal_knots[0] - eps, internal_knots[0] + eps])

        self._knot_cache[cache_key] = internal_knots
        return internal_knots