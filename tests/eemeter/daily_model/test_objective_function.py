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

from opendsm.eemeter.models.daily.objective_function import (
    get_idx,
    no_weights_obj_fcn,
    obj_fcn_decorator,
)

from opendsm.eemeter.models.daily.base_models.hdd_tidd_cdd import (
    evaluate_hdd_tidd_cdd_smooth,
    _hdd_tidd_cdd_smooth_weight,
)

from opendsm.eemeter.models.daily.utilities.settings import DailySettings as Settings


def test_get_idx():
    # Test case 1: Test with empty lists
    A = []
    B = []
    assert get_idx(A, B) == []

    # Test case 2: Test with one empty list
    A = ["a", "b", "c"]
    B = []
    assert get_idx(A, B) == []

    # Test case 3: Test with one non-empty list
    A = ["a", "b", "c"]
    B = ["a", "b", "c", "d", "e"]
    assert get_idx(A, B) == [0, 1, 2]

    # Test case 4: Test with two non-empty lists
    A = ["a", "b", "c"]
    B = ["a1", "b2", "c3", "d4", "e5"]
    assert get_idx(A, B) == [0, 1, 2]

    # Test case 5: Test with two non-empty lists with duplicates
    A = ["a", "b", "c"]
    B = ["a1", "b2", "c3", "a4", "e5"]
    assert get_idx(A, B) == [0, 1, 2, 3]


def test_no_weights_obj_fcn():
    # Test case 1: Test with X, obs and idx_bp as None, should raise an error
    X = None
    obs = None
    idx_bp = None
    model_fcn = lambda x: x
    aux_inputs = (model_fcn, obs, idx_bp)
    with pytest.raises(TypeError):
        no_weights_obj_fcn(X, aux_inputs)

    # Test case 2: Test with X, obs and idx_bp as empty arrays
    X = np.array([])
    obs = np.array([])
    idx_bp = np.array([])
    model_fcn = lambda x: x
    aux_inputs = (model_fcn, obs, idx_bp)
    assert no_weights_obj_fcn(X, aux_inputs) == 0

    # Test case 3: Test with X, obs and idx_bp as non-empty arrays
    X = np.array([1, 2, 3])
    obs = np.array([2, 4, 6])
    idx_bp = np.array([0, 2])
    model_fcn = lambda x: x * 2
    aux_inputs = (model_fcn, obs, idx_bp)
    assert no_weights_obj_fcn(X, aux_inputs) == 0

    # Test case 4: Test with X, obs and idx_bp as non-empty arrays with negative values
    X = np.array([-1, -2, -3])
    obs = np.array([-2, -4, -6])
    idx_bp = np.array([0, 2])
    model_fcn = lambda x: x * -2
    aux_inputs = (model_fcn, obs, idx_bp)
    assert no_weights_obj_fcn(X, aux_inputs) == 192


@pytest.fixture
def obj_fcn_test(self):
    # Create an instance of obj_fcn_decorator for testing

    # Define inputs
    T = np.array([1, 2, 3, 4, 5])
    obs = np.array([1, 2, 3, 4, 5])
    settings = Settings()
    coef_id = ["dd_k", "dd_beta", "dd_bp"]
    return obj_fcn_decorator(
        lambda x: x,
        lambda x: x,
        lambda x: x,
        T,
        obs,
        settings,
        alpha=2.0,
        coef_id=coef_id,
        initial_fit=True,
    )


def test_obj_fcn_decorator():
    # Test case 1: Test with minimum input values
    model_fcn_full = evaluate_hdd_tidd_cdd_smooth
    weight_fcn = _hdd_tidd_cdd_smooth_weight
    TSS_fcn = None
    T = np.array([1, 2, 3]).astype(float)
    obs = np.array([4, 5, 6]).astype(float)
    weights = None
    settings = type(
        "Settings",
        (object,),
        {
            "segment_minimum_count": 1,
            "regularization_percent_lasso": 0.5,
            "regularization_alpha": 0.1,
        },
    )()
    alpha = 2.0
    coef_id = ["dd_k", "dd_beta", "dd_bp"]
    initial_fit = True
    obj_fcn = obj_fcn_decorator(
        model_fcn_full,
        weight_fcn,
        TSS_fcn,
        T,
        obs,
        weights,
        settings,
        alpha,
        coef_id,
        initial_fit,
    )
    assert (
        obj_fcn(
            [73.48349431, 0.0, 0.0, 81.66823342, 5.34878023, 0.0, 50.85274713],
            [model_fcn_full, obs, [0, 1]],
        )
        == 2105.4340868444824
    )

    # Test case 2: Test with initial fit set as False
    model_fcn_full = evaluate_hdd_tidd_cdd_smooth
    weight_fcn = _hdd_tidd_cdd_smooth_weight
    TSS_fcn = None
    T = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(float)
    obs = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]).astype(float)
    settings = type(
        "Settings",
        (object,),
        {
            "segment_minimum_count": 2,
            "regularization_percent_lasso": 0.2,
            "regularization_alpha": 0.5,
        },
    )()
    alpha = 3.0
    coef_id = ["dd_k", "dd_beta"]
    initial_fit = False
    obj_fcn = obj_fcn_decorator(
        model_fcn_full,
        weight_fcn,
        TSS_fcn,
        T,
        obs,
        weights,
        settings,
        alpha,
        coef_id,
        initial_fit,
    )
    assert (
        obj_fcn(
            [1.5, 0.0, 0.0, 85.5, 10.254, 0.0, 50.85], [model_fcn_full, obs, [0, 1]]
        )
        == 1295.1226641177577
    )
