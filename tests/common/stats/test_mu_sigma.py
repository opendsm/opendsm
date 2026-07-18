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

from opendsm.common.stats.basic import median_absolute_deviation
from opendsm.common.stats.distribution_transform.mu_sigma import robust_mu_sigma



def _scalar(v):
    """Coerce iqr's size-1-array returns and scalar returns to a Python float."""
    return float(np.asarray(v).ravel()[0])


@pytest.fixture
def standard_normal():
    return np.random.default_rng(0).normal(0.0, 1.0, 4000)


@pytest.mark.parametrize("robust_type", ["iqr", "huber_m_estimate", "adaptive_weighted"])
def test_robust_mu_sigma_recovers_location_and_scale(robust_type, standard_normal):
    """The population estimators recover mu≈0, sigma≈1 on a standard-normal sample.

    RANSAC is excluded: it keeps the minimum-scale subsample by design, so it
    estimates the densest core rather than the population scale (see
    test_ransac_selects_most_compact_subsample).
    """
    mu, sigma = robust_mu_sigma(standard_normal, robust_type=robust_type)

    assert _scalar(mu) == pytest.approx(0.0, abs=0.1)
    assert _scalar(sigma) == pytest.approx(1.0, abs=0.1)


def test_ransac_selects_most_compact_subsample(standard_normal):
    """RANSAC's scale never exceeds the full-sample MAD (it minimizes scatter).

    Because it is seeded with the full-data MAD and only ever replaces it with a
    smaller subsample scale, the result is bounded above by the full MAD — and
    on clean data it lands below the population scale, not at it.
    """
    full_mad = median_absolute_deviation(standard_normal)
    _, sigma = robust_mu_sigma(standard_normal, robust_type="ransac", seed=0)

    assert _scalar(sigma) <= full_mad
    assert _scalar(sigma) < 1.0


def test_unknown_robust_type_raises():
    """An unrecognized robust_type raises a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unknown robust_type"):
        robust_mu_sigma(np.arange(10.0), robust_type="not_a_method")


def test_short_input_forces_iqr():
    """A 3-element input routes to iqr, matching an explicit iqr call (median 2)."""
    x = np.array([1.0, 2.0, 3.0])

    forced = robust_mu_sigma(x, robust_type="huber_m_estimate")
    explicit = robust_mu_sigma(x, robust_type="iqr")

    assert _scalar(forced[0]) == pytest.approx(_scalar(explicit[0]))
    assert _scalar(forced[1]) == pytest.approx(_scalar(explicit[1]))
    assert _scalar(forced[0]) == pytest.approx(2.0)


def test_huber_failure_falls_back_to_iqr(standard_normal):
    """When Huber fails to converge (maxiter=1) the result matches the iqr path."""
    fallback = robust_mu_sigma(standard_normal, robust_type="huber_m_estimate", maxiter=1)
    iqr = robust_mu_sigma(standard_normal, robust_type="iqr")

    assert _scalar(fallback[0]) == pytest.approx(_scalar(iqr[0]))
    assert _scalar(fallback[1]) == pytest.approx(_scalar(iqr[1]))


def test_ransac_is_deterministic_under_seed(standard_normal):
    """A fixed seed makes the RANSAC estimator reproducible run-to-run."""
    a = robust_mu_sigma(standard_normal, robust_type="ransac", seed=5)
    b = robust_mu_sigma(standard_normal, robust_type="ransac", seed=5)

    assert _scalar(a[0]) == _scalar(b[0])
    assert _scalar(a[1]) == _scalar(b[1])


def test_ransac_actually_iterates_over_distinct_subsamples(standard_normal):
    """More iterations lower the selected scale, proving each draw is distinct.

    RANSAC keeps the minimum-scale subsample, so drawing more distinct
    subsamples can only lower the result. If every iteration drew the same
    subsample (a seeding bug), n_iter would have no effect and the scales
    would be equal.
    """
    one = robust_mu_sigma(standard_normal, robust_type="ransac", seed=0, n_iter=1)
    many = robust_mu_sigma(standard_normal, robust_type="ransac", seed=0, n_iter=200)

    assert _scalar(many[1]) < _scalar(one[1])


def test_ransac_does_not_pollute_global_rng(standard_normal):
    """The estimator draws from a local Generator, leaving the global RNG intact."""
    np.random.seed(12345)
    before = np.random.random()

    np.random.seed(12345)
    robust_mu_sigma(standard_normal, robust_type="ransac", seed=7)
    after = np.random.random()

    assert before == after


def test_robust_estimators_resist_outliers():
    """A single gross outlier barely moves the robust location/scale estimate."""
    rng = np.random.default_rng(3)
    clean = rng.normal(0.0, 1.0, 500)
    contaminated = np.append(clean, 1e6)

    mu_clean, sigma_clean = robust_mu_sigma(clean, robust_type="iqr")
    mu_dirty, sigma_dirty = robust_mu_sigma(contaminated, robust_type="iqr")

    assert _scalar(mu_dirty) == pytest.approx(_scalar(mu_clean), abs=0.05)
    assert _scalar(sigma_dirty) == pytest.approx(_scalar(sigma_clean), abs=0.05)
