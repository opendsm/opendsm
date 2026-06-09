#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2014-2025 OpenDSM contributors
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from opendsm.common.stats.adaptive_loss import (
    LOSS_ALPHA_MIN,
    KernelWeightCache,
    _numba_brent_alpha,
    _numba_penalized_loss,
    adaptive_loss_fcn,
    adaptive_weights,
    alpha_scaled,
    generalized_loss_derivative,
    generalized_loss_fcn,
    generalized_loss_weights,
    get_C,
    penalized_loss_fcn,
    remove_outliers,
    rolling_C,
    rolling_IQR_outlier,
    sliding_window,
)



# ---------------------------------------------------------------------------
# generalized_loss_fcn — analytic values at the named alpha regimes
# ---------------------------------------------------------------------------

# At x=2: x^2=4. Each entry is the closed form of the loss family member.
_LOSS_AT_X2 = {
    2.0: 0.5 * 4.0,                       # L2
    1.0: np.sqrt(4.0 + 1.0) - 1.0,        # smoothed L1 (pseudo-Huber)
    0.0: np.log(0.5 * 4.0 + 1.0),         # Charbonnier: log(3)
    -2.0: 2 * 4.0 / (4.0 + 4.0),          # Cauchy/Lorentzian: 1.0
    -200.0: 1.0 - np.exp(-0.5 * 4.0),     # Welsch/Leclerc (alpha <= alpha_min)
}


@pytest.mark.parametrize("alpha,expected", list(_LOSS_AT_X2.items()))
def test_generalized_loss_fcn_analytic_values(alpha, expected):
    """Loss at x=2 matches the closed form of each named loss family."""
    result = generalized_loss_fcn(np.array([2.0]), alpha=alpha)[0]

    assert result == pytest.approx(expected, rel=1e-12)


def test_generalized_loss_fcn_zero_at_zero():
    """Every loss family is zero at x=0 (no penalty for a perfect fit)."""
    for alpha in _LOSS_AT_X2:
        loss = generalized_loss_fcn(np.array([0.0]), alpha=alpha)[0]

        assert loss == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("alpha", [2.0, 1.0, 0.0, -2.0, -200.0, 3.0, 1.5, 0.5, -0.7, -10.0])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_generalized_loss_derivative_matches_finite_difference(alpha, scale):
    """The analytic derivative agrees with a central finite difference.

    generalized_loss_fcn takes the already-normalized argument, so the loss at
    raw x under a given scale is generalized_loss_fcn(x / scale). Covers the
    canonical families and the general-alpha branch, over a spread of x and
    several scales so a single lucky point cannot hide a formula error.
    """
    x = np.array([-5.0, -1.7, -0.3, 0.3, 0.8, 1.3, 2.5, 5.0])
    h = 1e-6

    finite_diff = (
        generalized_loss_fcn((x + h) / scale, alpha=alpha)
        - generalized_loss_fcn((x - h) / scale, alpha=alpha)
    ) / (2 * h)
    analytic = generalized_loss_derivative(x, scale=scale, alpha=alpha)

    assert np.allclose(analytic, finite_diff, atol=1e-5)


# ---------------------------------------------------------------------------
# generalized_loss_weights — downweighting behavior
# ---------------------------------------------------------------------------

def test_weights_all_one_at_alpha_2():
    """L2 loss (alpha=2) weights every observation equally at 1.0."""
    x = np.array([0.0, 0.5, 1.0, 5.0, -10.0])

    weights = generalized_loss_weights(x, alpha=2.0)

    assert np.allclose(weights, 1.0)


def test_weights_downweight_large_residuals_below_alpha_2():
    """Below alpha=2 the weight strictly decreases with |residual|."""
    x = np.array([0.0, 1.0, 3.0, 10.0])

    weights = generalized_loss_weights(x, alpha=0.0)

    assert weights[0] == pytest.approx(1.0)
    assert np.all(np.diff(weights) < 0)
    assert np.all(weights[1:] < 1.0)


def test_weights_min_weight_floor_is_respected():
    """min_weight sets the floor a heavily-downweighted point cannot drop below."""
    x = np.array([0.0, 100.0])

    weights = generalized_loss_weights(x, alpha=-200.0, min_weight=0.2)

    assert weights.min() >= 0.2 - 1e-12
    assert weights[1] == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# alpha_scaled — bounded, monotone reparametrization of alpha
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha_max", [2.0, 100.0])
def test_alpha_scaled_monotone_increasing(alpha_max):
    """alpha_scaled maps the unit interval monotonically onto alpha space."""
    s = np.linspace(0.0, 1.0, 25)
    alphas = np.array([alpha_scaled(si, alpha_max) for si in s])

    assert np.all(np.diff(alphas) > 0)


@pytest.mark.parametrize("alpha_max", [2.0, 100.0])
def test_alpha_scaled_bounds_and_clipping(alpha_max):
    """Endpoints hit (alpha_min, alpha_max); out-of-range s is clipped to them."""
    assert alpha_scaled(0.0, alpha_max) == pytest.approx(LOSS_ALPHA_MIN)
    assert alpha_scaled(1.0, alpha_max) == pytest.approx(alpha_max)
    assert alpha_scaled(-5.0, alpha_max) == pytest.approx(LOSS_ALPHA_MIN)
    assert alpha_scaled(5.0, alpha_max) == pytest.approx(alpha_max)


# ---------------------------------------------------------------------------
# _numba_brent_alpha — matches scipy's bounded Brent on the same objective
# ---------------------------------------------------------------------------

def test_numba_brent_matches_scipy_bounded():
    """The hand-ported Brent minimizer reproduces scipy's bounded result."""
    rng = np.random.default_rng(7)
    x_sq = rng.standard_normal(80) ** 2
    w = np.full(80, 1.0 / 80)

    def objective(s):
        return _numba_penalized_loss(x_sq, w, alpha_scaled(s), LOSS_ALPHA_MIN)

    scipy_res = minimize_scalar(
        objective, method="Bounded", bounds=[-1e-5, 1 + 1e-5], options={"xatol": 1e-5}
    )
    expected = alpha_scaled(scipy_res.x)

    assert _numba_brent_alpha(x_sq, w, LOSS_ALPHA_MIN) == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# penalized_loss_fcn — partition-function penalty and the failure path
# ---------------------------------------------------------------------------

def test_penalized_loss_adds_partition_penalty():
    """The penalty raises the bare loss by ln_Z(alpha) at every point."""
    x = np.array([0.5, 1.0])

    bare = generalized_loss_fcn(x, alpha=1.0)
    penalized = penalized_loss_fcn(x, alpha=1.0, use_penalty=True)

    assert np.all(penalized > bare)
    assert np.allclose(penalized - bare, (penalized - bare)[0])


def test_penalized_loss_no_penalty_equals_bare_loss():
    """use_penalty=False returns the unpenalized generalized loss."""
    x = np.array([0.5, 1.0, 2.0])

    assert np.allclose(
        penalized_loss_fcn(x, alpha=1.0, use_penalty=False),
        generalized_loss_fcn(x, alpha=1.0),
    )


def test_penalized_loss_raises_on_nonfinite_input():
    """Non-finite normalized residuals produce non-finite loss → explicit raise."""
    with pytest.raises(Exception, match="non-finite"):
        penalized_loss_fcn(np.array([np.inf, 1.0]), alpha=1.0, use_penalty=True)


# ---------------------------------------------------------------------------
# adaptive_loss_fcn — optimization vs fixed alpha
# ---------------------------------------------------------------------------

def test_adaptive_loss_fcn_fixed_alpha_passthrough():
    """A fixed alpha is returned unchanged alongside its loss."""
    x = np.array([0.1, -0.2, 0.3, 0.05])

    loss, alpha = adaptive_loss_fcn(x, alpha=1.0)

    assert alpha == 1.0
    assert np.isfinite(loss)


def test_adaptive_loss_fcn_clean_gaussian_selects_l2():
    """Tight, outlier-free residuals drive the adaptive alpha to L2 (alpha≈2)."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal(400) * 0.01

    _, alpha = adaptive_loss_fcn(x, alpha="adaptive")

    assert alpha == pytest.approx(2.0, abs=1e-3)


def test_adaptive_loss_fcn_outliers_push_alpha_negative():
    """Heavy outliers move the adaptive alpha well below L2 to downweight them."""
    x = np.array([0.0, 0.1, -0.1, 0.05, -0.05, 50.0, -60.0])

    _, alpha = adaptive_loss_fcn(x, alpha="adaptive")

    assert alpha < 2.0


# ---------------------------------------------------------------------------
# adaptive_weights — pinned regression values (deterministic) + properties
# ---------------------------------------------------------------------------

def test_adaptive_weights_no_standardization():
    """Pinned weights/scale/alpha for a clean ramp (adaptive selects L2)."""
    x = np.array([1, 2, 3, 4, 5])

    weights, C, alpha = adaptive_weights(x)

    assert np.allclose(weights, np.array([1, 1, 1, 0.99993894, 0.99992852]), atol=1e-3)
    assert np.isclose(C, 4.4478)
    assert np.isclose(alpha, 2.0)


def test_adaptive_weights_fixed_alpha_above_two_upweights():
    """At alpha=3 the generalized weights exceed 1 and grow with |residual|."""
    x = np.array([1, 2, 3, 4, 5])
    x = (x - np.mean(x)) / np.std(x)

    weights, C, alpha = adaptive_weights(x, alpha=3)

    assert np.allclose(
        weights, np.array([1.09644634, 1.02496275, 1, 1.02496275, 1.09644634]), atol=1e-3
    )
    assert np.isclose(C, 3.1450695413615257)
    assert np.isclose(alpha, 3.0)


def test_adaptive_weights_handles_nonfinite():
    """A NaN entry is replaced rather than poisoning the weights."""
    x = np.array([1, 2, np.nan, 4, 5])

    weights, C, alpha = adaptive_weights(x)

    assert np.allclose(
        weights, np.array([1, 1, 0.99993282, 0.99994308, 0.99993282]), atol=1e-3
    )
    assert np.isfinite(weights).all()


def test_adaptive_weights_outlier_gets_near_zero_weight():
    """A gross outlier collapses to ~0 weight while inliers stay near 1."""
    x = np.array([1, 2, 3, 4, 5, 100])

    weights, C, alpha = adaptive_weights(x)

    assert weights[-1] < 0.01
    assert np.all(weights[:-1] > 0.9)
    assert alpha < 2.0


# ---------------------------------------------------------------------------
# remove_outliers
# ---------------------------------------------------------------------------

def test_remove_outliers_keeps_clean_data():
    """Clean data passes through with all indices retained."""
    data = np.array([1, 2, 3, 4, 5])

    kept, idx = remove_outliers(data)

    assert np.array_equal(kept, data)
    assert np.array_equal(idx, np.arange(len(data)))


def test_remove_outliers_drops_extreme_value():
    """A single extreme value is dropped; the rest survive."""
    data = np.array([1, 2, 3, 4, 5, 100])

    kept, idx = remove_outliers(data)

    assert np.array_equal(kept, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(idx, np.arange(5))


def test_remove_outliers_identical_values_returns_all():
    """All-identical input has no IQR spread; every index is kept."""
    data = np.array([5.0, 5.0, 5.0, 5.0])

    kept, idx = remove_outliers(data)

    assert np.array_equal(kept, data)
    assert np.array_equal(idx, np.arange(4))


# ---------------------------------------------------------------------------
# rolling helpers — shapes and window-as-proportion
# ---------------------------------------------------------------------------

def test_rolling_IQR_outlier_shape_and_window_proportion():
    """Returns (2, n) lower/upper thresholds; window<=1 is a fraction of n."""
    x = np.arange(100.0)
    y = np.random.default_rng(3).standard_normal(100)

    thresholds = rolling_IQR_outlier(x, y, window=0.1)

    assert thresholds.shape == (2, 100)
    assert np.all(thresholds[0] <= thresholds[1])


def test_rolling_C_returns_per_point_scale():
    """rolling_C yields one positive scale value per observation."""
    T = np.arange(100.0)
    resid = np.random.default_rng(4).standard_normal(100)

    C = rolling_C(T, resid, mu=0.0, window=0.2)

    assert C.shape == (100,)
    assert np.all(C >= 0)


# ---------------------------------------------------------------------------
# KernelWeightCache — local adaptive weighting over a covariate
# ---------------------------------------------------------------------------

def test_kernel_cache_degenerate_returns_unit_weights():
    """Fewer than 3 points (no geometry) → unit weights, alpha=2."""
    cache = KernelWeightCache(np.array([1.0, 2.0]))

    weights, alpha = cache.compute_weights(np.array([5.0, 9.0]))

    assert np.allclose(weights, 1.0)
    assert alpha == 2.0


def test_kernel_cache_clean_residuals_select_near_l2():
    """Clean residuals over the covariate keep the local alpha near L2.

    alpha is computed on residuals normalized by the local scale, so it is
    scale-invariant; clean Gaussian noise lands just below 2 (no heavy tails
    to downweight) and weights stay bounded by 1.
    """
    rng = np.random.default_rng(21)
    x = np.sort(rng.standard_normal(200) + 10.0)
    residuals = rng.standard_normal(200)

    cache = KernelWeightCache(x)
    weights, median_alpha = cache.compute_weights(residuals)

    assert 1.8 < median_alpha <= 2.0
    assert weights.max() <= 1.0 + 1e-9
    assert len(weights) == 200


def test_kernel_cache_downweights_local_outlier():
    """An injected local outlier is weighted far below the surrounding inliers."""
    rng = np.random.default_rng(22)
    x = np.sort(rng.standard_normal(200) + 10.0)
    residuals = rng.standard_normal(200) * 0.1
    residuals[100] = 25.0

    cache = KernelWeightCache(x)
    weights, _ = cache.compute_weights(residuals)

    assert weights[100] < 0.5
    assert weights[100] < np.median(weights)


def test_kernel_cache_return_local_scale_shape():
    """return_local_scale adds a per-observation scale array of length N."""
    rng = np.random.default_rng(23)
    x = np.sort(rng.standard_normal(150) + 5.0)
    residuals = rng.standard_normal(150)

    cache = KernelWeightCache(x)
    weights, alpha, c_local = cache.compute_weights(residuals, return_local_scale=True)

    assert c_local.shape == (150,)
    assert np.all(c_local > 0)


# ---------------------------------------------------------------------------
# get_C — robust scale estimate, and sliding_window validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ["iqr_legacy", "iqr", "mad", "stdev"])
def test_get_C_positive_for_each_algo(algo):
    """Every scale-estimation algorithm returns a finite positive C."""
    resid = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -2.0, -1.0])

    C = get_C(resid, mu=0.0, sigma=3.0, quantile=0.25, algo=algo)

    assert np.isfinite(C)
    assert C > 0


def test_get_C_stdev_equals_sigma_times_std():
    """The 'stdev' algorithm returns exactly sigma * std(resid)."""
    resid = np.array([1.0, 2.0, 3.0, 4.0, 5.0, -2.0, -1.0])

    C = get_C(resid, mu=0.0, sigma=3.0, algo="stdev")

    assert C == pytest.approx(3.0 * np.std(resid))


def test_get_C_grows_with_residual_scale():
    """A more-dispersed residual set yields a larger scale estimate."""
    tight = np.array([-1.0, 0.0, 1.0, 0.5, -0.5])
    wide = tight * 10.0

    assert get_C(wide, mu=0.0, sigma=3.0, algo="mad") > get_C(tight, mu=0.0, sigma=3.0, algo="mad")


def test_sliding_window_shape():
    """sliding_window tiles the array into overlapping windows of the given size."""
    windows = sliding_window(np.arange(10.0), 3, step=2)

    assert windows.shape == (4, 3)
    assert np.array_equal(windows[0], [0, 1, 2])
    assert np.array_equal(windows[1], [2, 3, 4])


def test_sliding_window_rejects_oversized_window():
    """A window larger than the array raises ValueError."""
    with pytest.raises(ValueError, match="Window size"):
        sliding_window(np.arange(5.0), 10)


def test_sliding_window_rejects_negative_step():
    """A negative step raises ValueError."""
    with pytest.raises(ValueError, match="Step must be positive"):
        sliding_window(np.arange(5.0), 2, step=-1)
