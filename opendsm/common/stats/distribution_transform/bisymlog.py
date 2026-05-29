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

"""Bi-symmetric logarithmic (bisymlog) transform.

The bisymlog transform is: sign(x) · log_b(|x/C| + 1)
where C controls the linear-to-log crossover and b is the base.
"""

import numpy as np
import numba

from scipy.optimize import minimize_scalar
from scipy.stats import skew

from opendsm.common.stats.distribution_transform.mu_sigma import robust_mu_sigma
from opendsm.common.stats.distribution_transform._base import TransformBase
from opendsm.common.stats.outliers import IQR_outlier
from opendsm.common.utils import OoM


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, error_model="numpy", cache=True)
def _bisymlog_forward(x, C, log_base_inv):
    """Vectorized bisymlog forward: sign(x) · log10(|x/C| + 1) / log10(base)."""
    out = np.empty_like(x)
    for i in range(len(x)):
        xi = x[i]
        if xi >= 0:
            out[i] = np.log10(xi / C + 1.0) * log_base_inv
        else:
            out[i] = -np.log10(-xi / C + 1.0) * log_base_inv
    return out


@numba.jit(nopython=True, error_model="numpy", cache=True)
def _bisymlog_inverse(y, C, base):
    """Vectorized bisymlog inverse: sign(y) · C · (base^|y| - 1)."""
    out = np.empty_like(y)
    for i in range(len(y)):
        yi = y[i]
        if yi >= 0:
            out[i] = C * (base ** yi - 1.0)
        else:
            out[i] = -C * (base ** (-yi) - 1.0)
    return out


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------

class Bisymlog(TransformBase):
    """Per-dimension bi-symmetric logarithmic transform with invertibility.

    Fits the C parameter per feature (by minimising skewness when
    ``robust=True``, or heuristically from data range when ``robust=False``).

    Parameters
    ----------
    robust : bool, default True
    base : int or float, default 10
    heuristic_scaling_factor : float, default 0.5
        Controls the linear-to-log crossover for heuristic C.
    rescale_quantile : float or None, default None
        If set, rescale transformed data to preserve inter-quantile range.
    min_variance : float, default 1e-10
    min_samples : int, default 3
    """

    _HYPERPARAM_KEYS = (
        "robust", "base", "heuristic_scaling_factor",
        "rescale_quantile", "min_variance", "min_samples",
    )

    def __init__(
        self,
        robust=True,
        base=10,
        heuristic_scaling_factor=0.5,
        rescale_quantile=None,
        min_variance=1e-10,
        min_samples=3,
    ):
        super().__init__(min_variance=min_variance, min_samples=min_samples)
        self.robust = robust
        self.base = base
        self.heuristic_scaling_factor = heuristic_scaling_factor
        self.rescale_quantile = rescale_quantile
        self._log_base_inv = 1.0 / np.log10(base)

        if rescale_quantile is not None and (rescale_quantile <= 0 or rescale_quantile >= 0.5):
            raise ValueError("rescale_quantile must be in (0, 0.5)")

    # -- C estimation --------------------------------------------------------

    @staticmethod
    def _heuristic_C(x, scaling_factor=0.5):
        """Estimate C from the data range (fast, non-robust)."""
        min_x, max_x = x.min(), x.max()
        if min_x == max_x:
            return None

        if np.sign(max_x) != np.sign(min_x):
            parts = [x[x >= 0], x[x <= 0]]
            C = 0.0
            for part in parts:
                r = np.abs(part.max() - part.min())
                if r > C:
                    C = r
                    max_x = part.max()
        else:
            C = np.abs(max_x - min_x)

        s_fcn = lambda v: np.power(10, np.power(v, 2))
        s_range = s_fcn(np.array([0.0, 1.0]))
        sf = s_fcn(scaling_factor)
        s_bnds = np.array([-1.0, 6.0])
        s = (sf - s_range[0]) / np.diff(s_range) * np.diff(s_bnds) + s_bnds[0]
        C *= 10 ** (OoM(max_x) + float(s[0]))
        return float(np.asarray(C).flat[0])

    def _robust_C(self, x):
        """Fit C by minimising skewness of the transformed data."""
        log_base_inv = self._log_base_inv

        def obj(log_C):
            C = 10 ** log_C
            xt = _bisymlog_forward(x, C, log_base_inv)
            mu, sigma = robust_mu_sigma(
                xt, "adaptive_weighted",
                use_mean=False, rel_err=1e-4, abs_err=1e-4,
            )
            xt = (xt - mu) / sigma
            bounds = IQR_outlier(xt, sigma_threshold=3, quantile=0.05)
            xt = xt[(bounds[0] < xt) & (xt < bounds[1])]
            return np.abs(skew(xt))

        res = minimize_scalar(obj, bounds=(-14, 6), method="bounded")
        return 10 ** res.x

    # -- TransformBase hooks -------------------------------------------------

    def _init_params(self, D):
        self.C_ = np.ones(D)
        self.rescale_slope_ = np.ones(D)
        self.rescale_offset_ = np.zeros(D)

    def _fit_dim(self, d, col_f, fm, X, return_transformed, out):
        if self.robust:
            C = self._robust_C(col_f)
        else:
            C = self._heuristic_C(col_f, self.heuristic_scaling_factor)

        if C is None or C <= 0:
            self.skip_dims_[d] = True
            return

        self.C_[d] = C
        transformed = _bisymlog_forward(col_f, C, self._log_base_inv)

        if self.rescale_quantile is not None:
            pq = np.quantile(col_f, [self.rescale_quantile, 1 - self.rescale_quantile])
            cq = np.quantile(transformed, [self.rescale_quantile, 1 - self.rescale_quantile])
            denom = np.diff(cq)
            if abs(denom) > 1e-15:
                slope = float(np.diff(pq) / denom)
                offset = float(pq[0] - cq[0] * slope)
            else:
                slope, offset = 1.0, 0.0
            self.rescale_slope_[d] = slope
            self.rescale_offset_[d] = offset
            if return_transformed:
                out[fm, d] = transformed * slope + offset
        elif return_transformed:
            out[fm, d] = transformed

    def _transform_dim(self, v, d):
        out = _bisymlog_forward(v, self.C_[d], self._log_base_inv)
        if self.rescale_quantile is not None:
            out = out * self.rescale_slope_[d] + self.rescale_offset_[d]
        return out

    def _inverse_transform_dim(self, v, d):
        if self.rescale_quantile is not None:
            v = (v - self.rescale_offset_[d]) / self.rescale_slope_[d]
        return _bisymlog_inverse(v, self.C_[d], self.base)

    def _serialise_params(self):
        return {
            "C": self.C_.tolist(),
            "rescale_slope": self.rescale_slope_.tolist(),
            "rescale_offset": self.rescale_offset_.tolist(),
        }

    def _deserialise_params(self, d):
        self.C_ = np.array(d["C"])
        self.rescale_slope_ = np.array(d["rescale_slope"])
        self.rescale_offset_ = np.array(d["rescale_offset"])

    def _serialise_hyperparams(self):
        hp = super()._serialise_hyperparams()
        hp.update({
            "robust": self.robust,
            "base": self.base,
            "heuristic_scaling_factor": self.heuristic_scaling_factor,
            "rescale_quantile": self.rescale_quantile,
        })
        return hp
