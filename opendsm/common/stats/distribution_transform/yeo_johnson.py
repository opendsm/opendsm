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

"""Robust Yeo-Johnson power transform (Raymaekers & Rousseeuw 2021).

Class-based API for per-dimension Yeo-Johnson transforms with full
invertibility, serialisation, and graceful handling of edge cases.
"""

import numpy as np
import numba

from scipy.stats import norm, yeojohnson as _scipy_yeojohnson

from opendsm.common.stats.distribution_transform.mu_sigma import robust_mu_sigma
from opendsm.common.stats.distribution_transform._base import TransformBase, PowerTransformMixin
from opendsm.common.stats.distribution_transform._bc_yj_shared import (
    _WANT_VALUE,
    _WANT_DERIV,
    _C_HUBER,
    _TUKEY_C,
    _LAM_EPS,
    _huber_std,
    _normal_scores,
    _bisquare_weights,
    _fit_lambda,
)


# ---------------------------------------------------------------------------
# Scalar kernels
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, error_model="numpy", cache=True)
def _yj(x, lam, deriv):
    """Scalar Yeo-Johnson value or x-derivative at a single point."""
    if not deriv:
        if   (lam != 0) and (x >= 0): return ((1 + x)**lam - 1) / lam
        elif (lam == 0) and (x >= 0): return np.log(1 + x)
        elif (lam != 2) and (x <  0): return -((1 - x)**(2 - lam) - 1) / (2 - lam)
        elif (lam == 2) and (x <  0): return -np.log(1 - x)
        else:                          return np.nan
    else:
        if   (lam != 0) and (x >= 0): return (x + 1)**(lam - 1)
        elif (lam == 0) and (x >= 0): return 1 / (1 + x)
        elif (lam != 2) and (x <  0): return (1 - x)**(1 - lam)
        elif (lam == 2) and (x <  0): return 1 / (1 - x)
        else:                          return np.nan


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _yj_inverse(y, lam):
    """Scalar inverse Yeo-Johnson: given y = YJ(x, lam), return x."""
    if y >= 0:
        if -_LAM_EPS < lam < _LAM_EPS:
            return np.expm1(y)
        inner = y * lam + 1.0
        if inner <= 0.0:
            return -1.0
        return inner ** (1.0 / lam) - 1.0
    else:
        p = 2.0 - lam
        if -_LAM_EPS < p < _LAM_EPS:
            return 1.0 - np.exp(-y)
        inner = 1.0 - y * p
        if inner <= 0.0:
            return 1.0
        return 1.0 - inner ** (1.0 / p)


# ---------------------------------------------------------------------------
# Vectorized transforms
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, error_model="numpy", cache=True)
def yj_transform(x, lam):
    """Forward Yeo-Johnson transform of array x."""
    h = np.empty_like(x)
    for i in range(len(x)):
        h[i] = _yj(x[i], lam, _WANT_VALUE)
    return h


@numba.jit(nopython=True, error_model="numpy", cache=True)
def yj_inverse_transform(y, lam):
    """Inverse Yeo-Johnson transform of array y."""
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = _yj_inverse(y[i], lam)
    return out


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _yj_rectified_transform(x, lam, Q):
    """Tail-rectified YJ transform of sorted array x (Def. 2, R&R 2021)."""
    q1, q3 = Q[0], Q[1]
    n = len(x)

    hq1 = _yj(q1, lam, _WANT_VALUE);  dq1 = _yj(q1, lam, _WANT_DERIV)
    hq3 = _yj(q3, lam, _WANT_VALUE);  dq3 = _yj(q3, lam, _WANT_DERIV)

    h = np.empty_like(x)

    lo = 0
    while lo < n and x[lo] < q1: lo += 1
    hi = lo
    while hi < n and x[hi] < q3: hi += 1

    for i in range(lo):      h[i] = hq1 + (x[i] - q1) * dq1
    for i in range(lo, hi):  h[i] = _yj(x[i], lam, _WANT_VALUE)
    for i in range(hi, n):   h[i] = hq3 + (x[i] - q3) * dq3

    return h


# ---------------------------------------------------------------------------
# Fitting kernels
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, error_model="numpy", cache=True)
def _yj_hg(x, lam):
    """Scalar YJ(x,λ) and ∂YJ/∂λ, sharing pow/log intermediates."""
    if x >= 0.0:
        if -_LAM_EPS < lam < _LAM_EPS:
            l = np.log(1.0 + x)
            return l, 0.5 * l * l
        v = 1.0 + x;  u = v ** lam;  l = np.log(v)
        h = (u - 1.0) / lam
        g = (u * (lam * l - 1.0) + 1.0) / (lam * lam)
        return h, g
    else:
        p = 2.0 - lam;  v = 1.0 - x
        if -_LAM_EPS < p < _LAM_EPS:
            l = np.log(v)
            return -l, 0.5 * l * l
        u = v ** p;  l = np.log(v)
        h = -(u - 1.0) / p
        g = (l * u * p - u + 1.0) / (p * p)
        return h, g


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _yj_var_dvar(x, lam, weight, W):
    """Weighted variance V and dV/dλ in a single pass."""
    S1 = 0.0;  S2 = 0.0;  S3 = 0.0;  S4 = 0.0
    for i in range(len(x)):
        h, g = _yj_hg(x[i], lam)
        w = weight[i];  wh = w * h
        S1 += wh;  S2 += wh * h;  S3 += w * g;  S4 += wh * g
    mu_w = S1 / W
    V  = S2 / W - mu_w * mu_w
    dV = 2.0 * (S4 / W - mu_w * S3 / W)
    return V, dV


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _yj_apply(x_sorted, lam, idx):
    """Apply YJ(x,λ) to sorted x and return results in original order."""
    out = np.empty(len(x_sorted))
    for i in range(len(x_sorted)):
        out[idx[i]] = _yj(x_sorted[i], lam, _WANT_VALUE)
    return out


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _yj_log_jacobian(x_sorted, weight):
    """Weighted YJ log-Jacobian: Σwᵢ·sign(xᵢ)·log(1 + |xᵢ|)."""
    J = 0.0;  i = 0;  n = len(x_sorted)
    while i < n and x_sorted[i] < 0.0:
        J -= weight[i] * np.log(1.0 - x_sorted[i]);  i += 1
    while i < n:
        J += weight[i] * np.log(1.0 + x_sorted[i]);  i += 1
    return J


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _yj_mle_obj(x_sorted, lam, weight, W, J):
    """Weighted negative log-likelihood for YJ: ½W·log V − (λ−1)·J."""
    S1 = 0.0;  S2 = 0.0
    for i in range(len(x_sorted)):
        h = _yj(x_sorted[i], lam, _WANT_VALUE);  wh = weight[i] * h
        S1 += wh;  S2 += wh * h
    mu_w = S1 / W
    V = S2 / W - mu_w * mu_w
    return 0.5 * np.log(max(V, 1e-300)) * W - (lam - 1) * J


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _yj_tukey_loss(x, lam, Q, phi, inv_c):
    """Tukey bisquare loss for YJ."""
    q1, q3 = Q[0], Q[1];  n = len(x)

    hq1 = _yj(q1, lam, _WANT_VALUE);  dq1 = _yj(q1, lam, _WANT_DERIV)
    hq3 = _yj(q3, lam, _WANT_VALUE);  dq3 = _yj(q3, lam, _WANT_DERIV)

    lo = 0
    while lo < n and x[lo] < q1: lo += 1
    hi = lo
    while hi < n and x[hi] < q3: hi += 1

    total = 0.0
    for i in range(lo):
        t = (hq1 + (x[i] - q1) * dq1 - phi[i]) * inv_c
        if -1.0 < t < 1.0: s = 1.0 - t * t;  total += 1.0 - s * s * s
        else:               total += 1.0
    for i in range(lo, hi):
        t = (_yj(x[i], lam, _WANT_VALUE) - phi[i]) * inv_c
        if -1.0 < t < 1.0: s = 1.0 - t * t;  total += 1.0 - s * s * s
        else:               total += 1.0
    for i in range(hi, n):
        t = (hq3 + (x[i] - q3) * dq3 - phi[i]) * inv_c
        if -1.0 < t < 1.0: s = 1.0 - t * t;  total += 1.0 - s * s * s
        else:               total += 1.0
    return total


# ---------------------------------------------------------------------------
# Fitting algorithm
# ---------------------------------------------------------------------------

def _yj_initial_obj(x_sorted, Q):
    """Iteration-0 objective for YJ: Tukey bisquare quantile-matching."""
    phi   = _normal_scores(len(x_sorted))
    inv_c = 1.0 / _TUKEY_C

    def obj_fn(lam):
        return _yj_tukey_loss(x_sorted, lam, Q, phi, inv_c)

    return obj_fn


def _yj_refinement_fns(x_sorted, lam_prev, Q, outlier_alpha):
    """Build YJ refinement objective and gradient at the current lambda estimate."""
    h_rect = _yj_rectified_transform(x_sorted, lam_prev, Q)
    mu, sigma = robust_mu_sigma(h_rect, "huber_m_estimate", c=_C_HUBER, tol=1e-08)
    mu    = float(np.asarray(mu).flat[0])
    sigma = float(np.asarray(sigma).flat[0])
    if sigma == 0.0:
        sigma = 1.0

    threshold = norm.ppf(1 - outlier_alpha)
    weight, W = _bisquare_weights(h_rect, mu, 1.0 / sigma, threshold)

    if W < len(x_sorted) * 0.05:
        weight = np.ones(len(x_sorted), dtype=float)
        W = float(len(x_sorted))

    J = _yj_log_jacobian(x_sorted, weight)

    def obj_fn(lam):
        return _yj_mle_obj(x_sorted, lam, weight, W, J)

    def grad_fn(lam):
        V, dV = _yj_var_dvar(x_sorted, lam, weight, W)
        if V < 1e-300:
            return -J
        return 0.5 * W * dV / V - J

    return obj_fn, grad_fn


def _fit_yj_lambda(x, **kwargs):
    """Fit a robust YJ lambda (Raymaekers & Rousseeuw 2021)."""
    return _fit_lambda(
        x,
        pre_std_fn=_huber_std,
        initial_obj_fn=_yj_initial_obj,
        refinement_fns_fn=_yj_refinement_fns,
        apply_fn=_yj_apply,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------

class YeoJohnson(PowerTransformMixin, TransformBase):
    """Per-dimension Yeo-Johnson power transform with invertibility.

    When ``robust=True`` (default), uses the Raymaekers & Rousseeuw (2021)
    algorithm which downweights outliers.  When ``robust=False``, uses
    scipy's maximum-likelihood estimation.

    Parameters
    ----------
    robust : bool, default True
    Q_perc : float, default 0.40
        Only used when ``robust=True``.
    outlier_alpha : float, default 0.010
        Only used when ``robust=True``.
    min_variance : float, default 1e-10
    min_samples : int, default 5
    """

    def __init__(
        self,
        robust=True,
        Q_perc=0.40,
        outlier_alpha=0.010,
        min_variance=1e-10,
        min_samples=5,
    ):
        super().__init__(min_variance=min_variance, min_samples=min_samples)
        self.robust = robust
        self.Q_perc = Q_perc
        self.outlier_alpha = outlier_alpha

    def _fit_dim(self, d, col_f, fm, X, return_transformed, out):
        mu_pre, sigma_pre = robust_mu_sigma(
            col_f, "huber_m_estimate", c=_C_HUBER, tol=1e-08,
        )
        mu_pre = float(np.asarray(mu_pre).flat[0])
        sigma_pre = float(np.asarray(sigma_pre).flat[0])
        if sigma_pre < self.min_variance:
            self.skip_dims_[d] = True
            return
        col_std = (col_f - mu_pre) / sigma_pre

        if self.robust:
            raw_yj, lam = _fit_yj_lambda(
                col_std,
                Q_perc=self.Q_perc,
                outlier_alpha=self.outlier_alpha,
                pre_standardize=False,
                post_standardize=False,
            )
        else:
            raw_yj, lam = _scipy_yeojohnson(col_std, lmbda=None)

        mu_post, sigma_post = robust_mu_sigma(
            raw_yj, "huber_m_estimate", c=_C_HUBER, tol=1e-08,
        )
        mu_post = float(np.asarray(mu_post).flat[0])
        sigma_post = float(np.asarray(sigma_post).flat[0])
        if sigma_post < self.min_variance:
            sigma_post = 1.0

        self.lambdas_[d] = lam
        self.pre_mu_[d] = mu_pre
        self.pre_sigma_[d] = sigma_pre
        self.post_mu_[d] = mu_post
        self.post_sigma_[d] = sigma_post
        if return_transformed:
            out[fm, d] = (raw_yj - mu_post) / sigma_post

    def _transform_dim(self, v, d):
        col_std = (v - self.pre_mu_[d]) / self.pre_sigma_[d]
        yj_vals = yj_transform(col_std, self.lambdas_[d])
        return (yj_vals - self.post_mu_[d]) / self.post_sigma_[d]

    def _inverse_transform_dim(self, v, d):
        col_f = v * self.post_sigma_[d] + self.post_mu_[d]
        col_f = yj_inverse_transform(col_f, self.lambdas_[d])
        return col_f * self.pre_sigma_[d] + self.pre_mu_[d]

    # _init_params, _serialise_params, _deserialise_params,
    # _serialise_hyperparams inherited from PowerTransformMixin.
