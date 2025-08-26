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

from opendsm.common.adaptive_loss import (
    remove_outliers,
    adaptive_weights,
    adaptive_loss_fcn,
)


def test_remove_outliers():
    # Test case 1: No outliers
    data = np.array([1, 2, 3, 4, 5])
    data_no_outliers, idx_no_outliers = remove_outliers(data)
    assert np.array_equal(data, data_no_outliers)
    assert np.array_equal(idx_no_outliers, np.arange(len(data)))

    # Test case 2: Outliers present
    data = np.array([1, 2, 3, 4, 5, 100])
    data_no_outliers, idx_no_outliers = remove_outliers(data)
    assert np.array_equal(data_no_outliers, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(idx_no_outliers, np.arange(len(data) - 1))

    # Test case 3: Weights provided
    data = np.array([1.0, 2, 3, 4, 5, 100])
    weights = np.array([1, 1, 1, 1, 1, 0.1])
    data_no_outliers, idx_no_outliers = remove_outliers(data, weights)
    assert np.array_equal(data_no_outliers, np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(idx_no_outliers, np.arange(len(data) - 1))


def test_adaptive_loss_fcn():
    # Test case 1: Test with all finite values
    x = np.array([1, 2, 3, 4, 5])
    loss_fcn_val, loss_alpha = adaptive_loss_fcn(x)
    assert np.isfinite(loss_fcn_val)
    assert np.isfinite(loss_alpha)

    # Test case 3: Test with zero values
    x = np.array([0, 0, 0, 0, 0])
    loss_fcn_val, loss_alpha = adaptive_loss_fcn(x)
    assert np.isfinite(loss_fcn_val)
    assert np.isfinite(loss_alpha)

    # Test case 4: Test with negative values
    x = np.array([-1, -2, -3, -4, -5])
    loss_fcn_val, loss_alpha = adaptive_loss_fcn(x)
    assert np.isfinite(loss_fcn_val)
    assert np.isfinite(loss_alpha)

    # Test case 5: Test with large values
    x = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
    loss_fcn_val, loss_alpha = adaptive_loss_fcn(x)
    assert np.isfinite(loss_fcn_val)
    assert np.isfinite(loss_alpha)


def test_adaptive_weights():
    # Test case 1: x has not been standardized
    x = np.array([1, 2, 3, 4, 5])
    weights, C, alpha = adaptive_weights(x)
    assert np.allclose(weights, np.array([1, 1, 1, 0.99993894, 0.99992852]), atol=1e-3)
    assert np.isclose(C, 4.4478)
    assert np.isclose(alpha, 2.0)

    # Test case 2: x has been standardized
    x = np.array([1, 2, 3, 4, 5])
    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    weights, C, alpha = adaptive_weights(x, alpha=3)
    assert np.allclose(weights, np.array([1, 1, 1, 1.02496275, 1.09644634]), atol=1e-3)
    assert np.isclose(C, 3.1450695413615257)
    assert np.isclose(alpha, 3.0)

    # Test case 3: x contains non-finite values
    x = np.array([1, 2, np.nan, 4, 5])
    weights, C, alpha = adaptive_weights(x)
    assert np.allclose(
        weights, np.array([1, 1, 0.99993282, 0.99994308, 0.99993282]), atol=1e-3
    )
    assert np.isclose(C, 5.55975)
    assert np.isclose(alpha, 2.0)

    # Test case 4: x contains outliers
    x = np.array([1, 2, 3, 4, 5, 100])
    weights, C, alpha = adaptive_weights(x)
    assert np.allclose(weights, np.array([1, 1, 1, 0.9865, 0.9479, 0.0011]), atol=1e-3)
    assert np.isclose(C, 6.05975)
    assert np.isclose(alpha, -1.0928, atol=1e-2)
