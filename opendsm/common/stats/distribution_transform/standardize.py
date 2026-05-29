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

"""Robust standardisation (location-scale) transform."""

import numpy as np

from opendsm.common.stats.distribution_transform.mu_sigma import robust_mu_sigma
from opendsm.common.stats.distribution_transform._base import TransformBase


class Standardize(TransformBase):
    """Per-dimension robust standardisation with invertibility.

    Fits robust location (mu) and scale (sigma) per feature, then
    standardises as ``(x - mu) / sigma``.

    Parameters
    ----------
    robust_type : str, default "iqr"
        Options: ``"iqr"``, ``"huber_m_estimate"``, ``"adaptive_weighted"``.
    min_variance : float, default 1e-10
    min_samples : int, default 3
    """

    _HYPERPARAM_KEYS = ("robust_type", "min_variance", "min_samples")

    def __init__(self, robust_type="iqr", min_variance=1e-10, min_samples=3):
        super().__init__(min_variance=min_variance, min_samples=min_samples)
        self.robust_type = robust_type

    def _init_params(self, D):
        self.mu_ = np.zeros(D)
        self.sigma_ = np.ones(D)

    def _fit_dim(self, d, col_f, fm, X, return_transformed, out):
        mu, sigma = robust_mu_sigma(col_f, self.robust_type)
        mu = float(np.asarray(mu).flat[0])
        sigma = float(np.asarray(sigma).flat[0])
        if sigma < self.min_variance:
            sigma = 1.0
        self.mu_[d] = mu
        self.sigma_[d] = sigma
        if return_transformed:
            out[fm, d] = (col_f - mu) / sigma

    def _transform_dim(self, v, d):
        return (v - self.mu_[d]) / self.sigma_[d]

    def _inverse_transform_dim(self, v, d):
        return v * self.sigma_[d] + self.mu_[d]

    def _serialise_params(self):
        return {"mu": self.mu_.tolist(), "sigma": self.sigma_.tolist()}

    def _deserialise_params(self, d):
        self.mu_ = np.array(d["mu"])
        self.sigma_ = np.array(d["sigma"])

    def _serialise_hyperparams(self):
        hp = super()._serialise_hyperparams()
        hp["robust_type"] = self.robust_type
        return hp
