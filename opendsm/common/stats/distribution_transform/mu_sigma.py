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

from copy import deepcopy as copy
import numpy as np
from statsmodels.robust.scale import Huber as huber_m_estimate

from opendsm.common.stats.adaptive_loss import adaptive_weights
from opendsm.common.stats.basic import (
    MAD_k, 
    weighted_quantile,
    median_absolute_deviation,
)


def adaptive_weighted_mu_sigma(x, use_mean=False, rel_err=1E-4, abs_err=1E-4):
    mu = np.median(x)
    sigma = median_absolute_deviation(x, median=mu)

    for n in range(10):
        mu_prior = copy(mu)
        sigma_prior = copy(sigma)
        weight = adaptive_weights(x)[0]
        if use_mean:
            mu = np.sum(weight*x)/np.sum(weight)
            sigma = np.sum(weight*(x - mu)**2)/np.sum(weight)

        else:
            mu = float(np.asarray(weighted_quantile(x, 0.5, weights=weight)).flat[0])
            sigma = median_absolute_deviation(x, median=mu, weights=weight)

        max_abs_err = np.max(np.abs([(mu - mu_prior), (sigma - sigma_prior)]))
        max_rel_err = np.max(np.abs([(mu - mu_prior)/mu_prior, (sigma - sigma_prior)/sigma_prior]))

        if (max_rel_err < rel_err) | (max_abs_err < abs_err):
            break

    if sigma == 0:
        sigma = 1

    return mu, sigma


def ransac_mu_sigma(x, n_iter=100, n_sample=100, seed=None):
    mu = np.median(x)
    sigma = median_absolute_deviation(x, median=mu)

    for _ in range(n_iter):
        np.random.seed(seed)
        idx = np.random.choice(x.size, n_sample)
        x_sample = x[idx]

        mu_sample = np.median(x_sample)
        sigma_sample = median_absolute_deviation(x_sample, median=mu_sample)

        if sigma_sample < sigma:
            mu = mu_sample
            sigma = sigma_sample

    return mu, sigma


def robust_mu_sigma(x, robust_type="huber_m_estimate", **kwargs):
    if (len(x) <= 3) and (robust_type != "iqr"):
        robust_type = "iqr"

    if robust_type == "iqr":
        mu = weighted_quantile(x, 0.5)
        sigma = weighted_quantile(np.abs(x - mu), 0.5)*MAD_k

    elif robust_type == "huber_m_estimate":
        try:
            if "maxiter" not in kwargs:
                kwargs["maxiter"] = 30

            # raise RuntimeWarning to error
            with np.errstate(all='raise'):
                mu, sigma = huber_m_estimate(**kwargs)(x)

        except Exception as e:
            mu, sigma = robust_mu_sigma(x, robust_type="iqr")

    elif robust_type == "adaptive_weighted": # slow
        mu, sigma = adaptive_weighted_mu_sigma(x, **kwargs)

    elif robust_type == "ransac":
        mu, sigma = ransac_mu_sigma(x, **kwargs)

    else:
        raise ValueError(
            f"Unknown robust_type {robust_type!r}. Must be one of "
            "'iqr', 'huber_m_estimate', 'adaptive_weighted', 'ransac'."
        )

    return mu, sigma