"""Tests for PSplineSolver — the monotonicity-constrained penalized B-spline solver."""

import numpy as np
import pytest

from opendsm.eemeter.models.daily_pspline.solver import (
    PSplineSolver,
    pad_knots,
    derivative_zones,
    effective_df,
    _difference_matrices,
)


class TestPadKnots:
    def test_degree_3_adds_3_each_side(self):
        knots = np.array([0.0, 1.0, 2.0, 3.0])
        padded = pad_knots(knots, degree=3)
        assert padded[0] == 0.0 and padded[1] == 0.0 and padded[2] == 0.0
        assert padded[-1] == 3.0 and padded[-2] == 3.0 and padded[-3] == 3.0
        assert len(padded) == len(knots) + 6

    def test_degree_0_adds_nothing(self):
        knots = np.array([1.0, 2.0, 3.0])
        padded = pad_knots(knots, degree=0)
        np.testing.assert_array_equal(padded, knots)

    def test_degree_1_adds_1_each_side(self):
        knots = np.array([0.0, 5.0])
        padded = pad_knots(knots, degree=1)
        assert len(padded) == 4
        assert padded[0] == 0.0 and padded[-1] == 5.0


class TestDerivativeZones:
    def test_hdd_only_when_bp_at_domain_end(self):
        knots = tuple(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]))
        zones = derivative_zones(knots, k=3, bp0=3.0, bp1=3.0)
        assert len(zones["cdd"]) == 0, "Expected no CDD zone when bp at right boundary"
        assert len(zones["hdd"]) > 0

    def test_cdd_only_when_bp_at_domain_start(self):
        knots = tuple(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]))
        zones = derivative_zones(knots, k=3, bp0=0.0, bp1=0.0)
        assert len(zones["hdd"]) == 0, "Expected no HDD zone when bp at left boundary"
        assert len(zones["cdd"]) > 0

    def test_three_zones_with_interior_bp(self):
        knots = tuple(np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]))
        zones = derivative_zones(knots, k=3, bp0=2.0, bp1=3.5)
        total = len(zones["hdd"]) + len(zones["tidd"]) + len(zones["cdd"])
        n_deriv = len(knots) - 3 - 2
        assert total == n_deriv, (
            f"Zone assignment should cover all {n_deriv} derivatives, got {total}"
        )


class TestDifferenceMatrices:
    def test_d1_shape(self):
        knots = tuple(pad_knots(np.linspace(0, 1, 6), degree=3))
        n_base = len(knots) - 3 - 1
        D1, _, _ = _difference_matrices(knots, 3, n_base, need_D2=False, need_D3=False)
        assert D1.shape == (n_base - 1, n_base)

    def test_d3_is_none_when_not_needed(self):
        knots = tuple(pad_knots(np.linspace(0, 1, 6), degree=3))
        n_base = len(knots) - 3 - 1
        _, _, D3 = _difference_matrices(knots, 3, n_base, need_D2=False, need_D3=False)
        assert D3 is None


class TestSolverSolve:
    @pytest.fixture
    def simple_problem(self):
        """Monotone decreasing data with known structure."""
        x = np.linspace(-2, 2, 50)
        y = -0.5 * x  # Linear decreasing
        internal_knots = np.linspace(-2, 2, 8)
        padded = pad_knots(internal_knots, degree=3)
        weights = np.ones(50)
        return x, y, padded, weights

    def test_solve_returns_coefs_and_V(self, simple_problem):
        x, y, padded, weights = simple_problem
        solver = PSplineSolver(x, y, padded, 3, weights, 0.0, None, 1e6)
        bp = np.array([0.0, 0.0])
        coefs, V, _ = solver.solve(bp, weights, 1e6, 30)
        assert len(coefs) == solver.n_base
        assert len(V) == solver.n_deriv

    def test_unconstrained_fit_matches_least_squares(self, simple_problem):
        x, y, padded, weights = simple_problem
        # kappa=0 disables monotonicity → unconstrained WLS
        solver = PSplineSolver(x, y, padded, 3, weights, 0.0, None, 0.0)
        bp = np.array([-2.0, 2.0])
        coefs, V, _ = solver.solve(bp, weights, 0.0, 30)
        pred = solver.B @ coefs
        rmse = np.sqrt(np.mean((y - pred) ** 2))
        assert rmse < 0.05, f"Unconstrained fit should be nearly exact, got RMSE={rmse:.4f}"

    def test_monotone_constraint_enforced(self, rng):
        """HDD zone derivatives should be non-negative (decreasing energy)."""
        x = np.linspace(-2, 2, 80)
        y = np.exp(-x)  # Monotone decreasing
        internal_knots = np.linspace(-2, 2, 10)
        padded = pad_knots(internal_knots, degree=3)
        solver = PSplineSolver(x, y, padded, 3, None, 0.0, None, 1e9)
        bp = np.array([2.0, 2.0])  # All HDD
        coefs, V, _ = solver.solve(bp, None, 1e9, 100)
        deriv = solver.D1 @ coefs
        # HDD zone: penalize increasing (positive) derivatives
        violations = np.sum(deriv > 1e-6)
        assert violations == 0, f"Monotone constraint violated: {violations} positive derivatives"


class TestEffectiveDf:
    def test_edf_less_than_n_base(self, rng):
        x = np.linspace(0, 1, 50)
        y = rng.standard_normal(50)
        internal_knots = np.linspace(0, 1, 6)
        padded = pad_knots(internal_knots, degree=3)
        solver = PSplineSolver(x, y, padded, 3, None, 0.0, None, 1e6)
        bp = np.array([0.5, 0.5])
        coefs, V, _ = solver.solve(bp, None, 1e6, 30)
        w_sq = np.ones(50)
        edf = effective_df(solver, V, 1e6, w_sq)
        assert 0 < edf <= solver.n_base, f"edf={edf} should be in (0, {solver.n_base}]"
