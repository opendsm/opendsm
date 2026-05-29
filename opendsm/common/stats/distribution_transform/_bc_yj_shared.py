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

"""Shared helpers for robust power-transform fitting (Raymaekers & Rousseeuw 2021).

Constants, optimisers, and outlier-weight computation used by both
:mod:`.yeo_johnson` and :mod:`.box_cox`.
"""

import functools

import numpy as np
import numba

from scipy.stats import norm
from scipy.optimize import minimize_scalar

from opendsm.common.stats.distribution_transform.mu_sigma import robust_mu_sigma


_WANT_VALUE = False
_WANT_DERIV = True

_C_HUBER  = 1.5   # Huber M-estimator constant; insensitive over [1.0, 2.5]
_TUKEY_C  = 0.5   # Tukey bisquare half-width; insensitive over [0.25, 2.0]
_LAM_EPS  = 1e-7  # neighbourhood of λ=0 and λ=2 where limit formulae apply


# ---------------------------------------------------------------------------
# Python helpers
# ---------------------------------------------------------------------------

def _huber_std(x):
    """Robustly standardise x with Huber M-estimator (falls back to IQR)."""
    mu, sigma = robust_mu_sigma(x, "huber_m_estimate", c=_C_HUBER, tol=1e-08)
    if sigma == 0.0:
        sigma = 1.0
    return (x - mu) / sigma


def _brent_min(obj_fn, bounds=(-4.0, 4.0)):
    """Bounded Brent minimisation over lambda."""
    res = minimize_scalar(obj_fn, bounds=bounds, method="bounded", options={"xatol": 1e-4})
    return res.x


@functools.lru_cache(maxsize=16)
def _normal_scores(n):
    """Theoretical normal order statistics (Blom approximation), cached by n."""
    return norm.ppf((np.arange(n) + 2/3) / (n + 1/3))


# ---------------------------------------------------------------------------
# Numba helpers
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, error_model="numpy", cache=True)
def _bisquare_weights(vals, mu, sigma_inv, threshold):
    """Tukey bisquare outlier weights with inline standardisation.

    wᵢ = (1 − (|zᵢ|/t)²)²  if |zᵢ| ≤ t,  else 0.
    """
    weight = np.empty_like(vals)
    W = 0.0
    for i in range(len(vals)):
        u = abs((vals[i] - mu) * sigma_inv) / threshold
        if u <= 1.0:
            s = 1.0 - u * u;  w = s * s
            weight[i] = w;  W += w
        else:
            weight[i] = 0.0
    return weight, W


# ---------------------------------------------------------------------------
# Secant optimizer
# ---------------------------------------------------------------------------

def _secant(grad_fn, lam0, bounds=(-4.0, 4.0), xatol=1e-4, max_iter=10):
    """Secant root-finding on grad_fn, warm-started at lam0.

    Returns None on degeneracy or non-convergence — caller falls back to Brent.
    """
    lo, hi = bounds
    x_curr = float(np.clip(lam0, lo, hi))
    perturb = 0.05
    x_prev = x_curr - perturb if x_curr > lo + perturb else x_curr + perturb

    g_prev = grad_fn(x_prev)
    g_curr = grad_fn(x_curr)

    for _ in range(max_iter):
        if abs(g_curr) < xatol:
            break

        dg = g_curr - g_prev
        dx = x_curr - x_prev

        if abs(dg) < 1e-10 or abs(dx) < 1e-14:
            return None

        step = -g_curr * dx / dg
        if abs(step) > 0.5 * (hi - lo):
            step = -np.sign(g_curr) * 0.1 * (hi - lo)

        x_new  = float(np.clip(x_curr + step, lo, hi))
        x_prev, g_prev = x_curr, g_curr
        x_curr = x_new
        g_curr = grad_fn(x_curr)

        if abs(x_curr - x_prev) < xatol:
            break
    else:
        return None

    return float(np.clip(x_curr, lo, hi))


# ---------------------------------------------------------------------------
# Shared fitting algorithm
# ---------------------------------------------------------------------------

def _fit_lambda(
    x,
    pre_std_fn,
    initial_obj_fn,
    refinement_fns_fn,
    apply_fn,
    Q_perc=0.40,
    outlier_alpha=0.010,
    pre_standardize=True,
    post_standardize=True,
):
    """Fit a robust power-transform lambda (Raymaekers & Rousseeuw 2021).

    Parameters
    ----------
    x               : 1-D array of finite values
    pre_std_fn      : callable(x) → standardised x  (YJ or BC specific)
    initial_obj_fn  : callable(xs, Q) → objective function for Brent
    refinement_fns_fn : callable(xs, lam, Q, alpha) → (obj_fn, grad_fn)
    apply_fn        : callable(xs, lam, idx) → transformed in original order
    Q_perc          : quantile for rectification boundary
    outlier_alpha   : tail probability for outlier threshold
    pre_standardize : robustly centre and scale before fitting
    post_standardize: robustly centre and scale the output

    Returns
    -------
    xt  : transformed array, original order
    lam : fitted lambda
    """
    if pre_standardize:
        x = pre_std_fn(x)

    idx = np.argsort(x)
    xs = x[idx]
    _n = len(xs)
    Q = np.array([xs[min(int(Q_perc * _n), _n - 1)],
                  xs[min(int((1 - Q_perc) * _n), _n - 1)]])

    lam = _brent_min(initial_obj_fn(xs, Q))

    for _ in range(2):
        obj_fn, grad_fn = refinement_fns_fn(xs, lam, Q, outlier_alpha)
        lam_new = _secant(grad_fn, lam0=lam)
        if lam_new is None:
            warm_bounds = (max(-4.0, lam - 1.5), min(4.0, lam + 1.5))
            lam_new = _brent_min(obj_fn, bounds=warm_bounds)
            if obj_fn(lam_new) > obj_fn(lam):
                lam_new = lam
        if abs(lam_new - lam) < 1e-4:
            lam = lam_new
            break
        lam = lam_new

    xt = apply_fn(xs, lam, idx)
    if post_standardize:
        xt = _huber_std(xt)

    return xt, lam
