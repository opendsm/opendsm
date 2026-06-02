"""Tests for breakpoint optimization."""

import numpy as np
import pytest

from opendsm.eemeter.models.daily_pspline.breakpoints import optimize_breakpoints
from opendsm.eemeter.models.daily_pspline.knots import Knots
from opendsm.eemeter.models.daily_pspline.solver import PSplineSolver, pad_knots


class TestOptimizeBreakpoints:
    def test_returns_bp_and_solver(self, rng):
        x = np.linspace(-2, 2, 60)
        y = np.abs(x) + rng.normal(0, 0.1, 60)
        bp_init = np.array([0.0, 0.0])
        knots_obj = Knots(
            x, y, n_min=5, bspline_degree=3,
            lambda_smoothing=0.0, kappa_penalty=1e6, maxiter=30,
        )
        bp, solver = optimize_breakpoints(
            x, y, bp_init, knots_obj,
            weights=None, n_min=5, zone_knot_count=5,
            degree=3, lambda_smoothing=0.0, bc_type=None,
            kappa=1e6, maxiter=30,
            reg_alpha=0.0, reg_pct_lasso=1.0,
            allow_hdd=True, allow_cdd=True,
        )
        assert len(bp) == 2
        assert bp[0] <= bp[1], f"bp[0]={bp[0]:.3f} > bp[1]={bp[1]:.3f}"
        assert isinstance(solver, PSplineSolver)

    def test_disabled_zones_pins_bp_to_bounds(self, rng):
        x = np.linspace(0, 1, 40)
        y = rng.standard_normal(40)
        knots_obj = Knots(
            x, y, n_min=5, bspline_degree=3,
            lambda_smoothing=0.0, kappa_penalty=1e6, maxiter=30,
        )
        bp, _ = optimize_breakpoints(
            x, y, np.array([0.5, 0.5]), knots_obj,
            weights=None, n_min=5, zone_knot_count=5,
            degree=3, lambda_smoothing=0.0, bc_type=None,
            kappa=1e6, maxiter=30,
            reg_alpha=0.0, reg_pct_lasso=1.0,
            allow_hdd=False, allow_cdd=False,
        )
        np.testing.assert_allclose(
            bp, [x[0], x[-1]], atol=1e-10,
            err_msg="Disabled zones should pin bp to data bounds",
        )
