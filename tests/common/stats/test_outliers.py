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

from opendsm.common.stats.outliers import (
    IQR_outlier,
    remove_outliers,
)



def test_iqr_outlier_analytic_bounds():
    """Bounds match the closed form: [q1 - k·IQR, q3 + k·IQR], k=0.7413σ-0.5."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    lower, upper = IQR_outlier(x, sigma_threshold=3, quantile=0.25)

    q1, q3 = 2.0, 4.0
    scalar = 0.7413 * 3 - 0.5
    spread = (q3 - q1) * scalar

    assert lower == pytest.approx(q1 - spread)
    assert upper == pytest.approx(q3 + spread)


def test_iqr_outlier_widens_with_sigma_threshold():
    """A larger sigma_threshold produces strictly wider outlier bounds."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    narrow = IQR_outlier(x, sigma_threshold=2)
    wide = IQR_outlier(x, sigma_threshold=5)

    assert wide[0] < narrow[0]
    assert wide[1] > narrow[1]


def test_iqr_outlier_weights_shift_bounds():
    """Supplying weights routes through the weighted-quantile path and moves bounds."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    unweighted = IQR_outlier(x)
    weighted = IQR_outlier(x, weights=np.array([5.0, 1.0, 1.0, 1.0, 1.0]))

    assert not np.allclose(unweighted, weighted)


def test_iqr_outlier_ignores_non_finite():
    """NaN/inf entries are dropped before the quantiles are computed."""
    clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dirty = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.inf, 5.0])

    assert np.allclose(IQR_outlier(dirty), IQR_outlier(clean))


def test_remove_outliers_keeps_clean_data():
    """Data inside the bounds passes through unchanged with all indices."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    kept, idx = remove_outliers(x)

    assert np.array_equal(kept, x)
    assert np.array_equal(idx, np.arange(5))


def test_remove_outliers_drops_extreme_and_reports_index():
    """A gross outlier is removed; surviving indices point back into the input."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])

    kept, idx = remove_outliers(x)

    assert np.array_equal(kept, x[:5])
    assert np.array_equal(idx, np.arange(5))


def test_remove_outliers_identical_values_short_circuits():
    """All-identical input has no spread, so every index is kept."""
    x = np.array([7.0, 7.0, 7.0, 7.0])

    kept, idx = remove_outliers(x)

    assert np.array_equal(kept, x)
    assert np.array_equal(idx, np.arange(4))


def test_remove_outliers_respects_weights():
    """A downweighted extreme value is still excluded from the kept set."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.1])

    kept, idx = remove_outliers(x, weights)

    assert np.array_equal(kept, x[:5])
    assert np.array_equal(idx, np.arange(5))


def test_remove_outliers_extreme_finite_value_keeps_inliers_not_fallback():
    """A lone extreme finite value never empties the in-bounds set.

    The 25th/75th percentiles sit on the cluster of zeros, so the zeros are
    always inside the bounds and are returned by the normal path — the
    closest-point fallback is not reached for finite data.
    """
    x = np.array([0.0, 0.0, 0.0, 0.0, 1e9])

    kept, idx = remove_outliers(x, sigma_threshold=3)

    assert np.array_equal(kept, np.zeros(4))
    assert np.array_equal(idx, np.arange(4))


# The closest-point fallback in remove_outliers is unreachable for finite input
# (a 20k-trial search found no trigger) and the only path to it — all-non-finite
# input — raises in _IQR_outlier under NUMBA_DISABLE_JIT, so it cannot be covered
# portably. It carries a `# pragma: no cover` in the source rather than a
# JIT-mode-dependent test.
