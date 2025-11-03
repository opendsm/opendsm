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
from opendsm.eemeter.models.daily.utilities.selection_criteria import (
    neg_log_likelihood,
    selection_criteria,
)


def test_neg_log_likelihood():
    # Test case 1: Test with a simple loss and N
    loss = 1.0
    N = 10
    result = neg_log_likelihood(loss, N)
    expected = -2.6764598670764994
    assert np.allclose(result, expected)

    # Test case 2: Test with a larger loss and N
    loss = 10.0
    N = 100
    result = neg_log_likelihood(loss, N)
    expected = -26.764598670764993
    assert np.allclose(result, expected)

    # Test case 3: Test with a loss of zero (should return infinity)
    loss = 0.0
    N = 10
    result = neg_log_likelihood(loss, N)
    expected = np.inf
    assert np.allclose(result, expected)

    # Test case 4: Test with a negative loss (should raise ValueError)
    loss = -1.0
    N = 10
    try:
        neg_log_likelihood(loss, N)
    except ValueError as e:
        assert str(e) == "loss must be non-negative"

    # Test case 5: Test with a negative N (should raise ValueError)
    loss = 1.0
    N = -10
    try:
        neg_log_likelihood(loss, N)
    except ValueError as e:
        assert str(e) == "N must be positive"


def test_selection_criteria():
    # Test case 1: Test with RMSE criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="rmse"
    )
    expected = np.sqrt(loss / N)
    assert np.allclose(result, expected)

    # Test case 2: Test with RMSE adjusted criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="rmse_adj"
    )
    expected = np.sqrt(loss / (N - num_coeffs - 1))
    assert np.allclose(result, expected)

    # Test case 3: Test with R-squared criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="r_squared"
    )
    expected = (1 - (1 - loss / TSS)) * 10
    assert np.allclose(result, expected)

    # Test case 4: Test with adjusted R-squared criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="r_squared_adj"
    )
    expected = 1.2857142857142856
    assert np.allclose(result, expected)

    # Test case 5: Test with FPE criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="fpe"
    )
    expected = 0.18571428571428572
    assert np.allclose(result, expected)

    # Test case 6: Test with AIC criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="aic"
    )
    expected = 0.9352919734152998
    assert np.allclose(result, expected)

    # Test case 7: Test with AICc criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="aicc"
    )
    expected = 1.1067205448438713
    assert np.allclose(result, expected)

    # Test case 8: Test with CAIC criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="caic"
    )
    expected = 1.195808992014109
    assert np.allclose(result, expected)

    # Test case 9: Test with BIC criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="bic"
    )
    expected = 0.9958089920141091
    assert np.allclose(result, expected)

    # Test case 10: Test with SABIC criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    result = selection_criteria(
        loss, TSS, N, num_coeffs, model_selection_criteria="sabic"
    )
    expected = 0.3966625373033108
    assert np.allclose(result, expected)

    # Test case 11: Test with invalid criterion -> Not possible since we are using the settings config
    # loss = 1.0
    # TSS = 10.0
    # N = 10
    # num_coeffs = 2
    # try:
    #     selection_criteria(loss, TSS, N, num_coeffs, model_selection_criteria="invalid")
    # except ValueError as e:
    #     assert str(e) == "Invalid model selection criterion"

    # Remove the below test cases once the following criteria are implemented

    # Test case 12: Test with DIC criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    try:
        selection_criteria(loss, TSS, N, num_coeffs, model_selection_criteria="dic")
    except NotImplementedError as e:
        assert str(e) == "DIC has not been implmented as a model selection criterion"

    # Test case 13: Test with WAIC criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    try:
        selection_criteria(loss, TSS, N, num_coeffs, model_selection_criteria="waic")
    except NotImplementedError as e:
        assert str(e) == "WAIC has not been implmented as a model selection criterion"

    # Test case 14: Test with WBIC criterion
    loss = 1.0
    TSS = 10.0
    N = 10
    num_coeffs = 2
    try:
        selection_criteria(loss, TSS, N, num_coeffs, model_selection_criteria="wbic")
    except NotImplementedError as e:
        assert str(e) == "WBIC has not been implmented as a model selection criterion"
