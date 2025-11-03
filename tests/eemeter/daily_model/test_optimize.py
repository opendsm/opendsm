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
from opendsm.eemeter.models.daily.optimize import obj_fcn_dec, Optimizer

from opendsm.eemeter.models.daily.objective_function import obj_fcn_decorator

from opendsm.eemeter.models.daily.base_models.hdd_tidd_cdd import (
    evaluate_hdd_tidd_cdd_smooth,
    _hdd_tidd_cdd_smooth_weight,
)

from opendsm.eemeter.models.daily.fit_base_models import _get_opt_settings
from opendsm.eemeter.models.daily.utilities.settings import DailySettings as Settings


def test_obj_fcn_dec():
    # Test case 1: Check if the function returns the expected output for valid input
    def obj_fcn(x):
        return np.sum(x**2)

    x0 = np.array([1, 2, 3])
    bnds = np.array([[0, 1], [1, 2], [2, 3]])
    obj_fcn_eval, idx_opt = obj_fcn_dec(obj_fcn, x0, bnds)

    x = np.array([0.5, 1.5, 2.5])
    expected_output = 5
    assert obj_fcn_eval(x) == expected_output
    assert idx_opt == [0, 1, 2]


def get_obj_fcn(settings):
    # Test case 1: Test with minimum input values
    model_fcn_full = evaluate_hdd_tidd_cdd_smooth
    weight_fcn = _hdd_tidd_cdd_smooth_weight
    TSS_fcn = None
    T = np.array([1, 2, 3, 4, 5, 6, 7]).astype(float)
    obs = np.array([2, 4, 6, 8, 10, 12, 14]).astype(float)
    base_weights = None
    alpha = 2.0
    coef_id = [
        "hdd_bp",
        "hdd_beta",
        "hdd_k",
        "cdd_bp",
        "cdd_beta",
        "cdd_k",
        "intercept",
    ]
    initial_fit = True
    return obj_fcn_decorator(
        model_fcn_full,
        weight_fcn,
        TSS_fcn,
        T,
        obs,
        base_weights,
        settings,
        alpha,
        coef_id,
        initial_fit,
    )


@pytest.fixture
def get_x0():
    return np.array([73.48349431, 0.0, 0.0, 81.66823342, 5.34878023, 0.0, 50.85274713])


@pytest.fixture
def get_bnds():
    return np.array(
        [
            [59.79057581, 86.91811908],
            [0.0, 16.0463407],
            [0.0, 1.0],
            [59.79057581, 86.91811908],
            [0.0, 16.0463407],
            [0.0, 1.0],
            [20.13393783, 79.33486982],
        ]
    )


def test_optimizer_run(get_x0, get_bnds):
    x0 = get_x0
    bnds = get_bnds
    coef_id = [
        "hdd_bp",
        "hdd_beta",
        "hdd_k",
        "cdd_bp",
        "cdd_beta",
        "cdd_k",
        "intercept",
    ]

    # Test case 1: Test with empty options
    settings = Settings()
    opt_settings = _get_opt_settings(settings)
    obj_fcn = get_obj_fcn(settings)
    optimizer = Optimizer(obj_fcn, x0, bnds, coef_id, settings, opt_settings)
    res = optimizer.run()
    assert np.allclose(res.x, np.array([20.13393]), rtol=1e-5, atol=1e-5)

    # Test case 2: Test with scipy algorithm
    settings = Settings(developer_mode=True, algorithm_choice="scipy_Nelder-Mead")
    opt_settings = _get_opt_settings(settings)
    obj_fcn = get_obj_fcn(settings)
    optimizer = Optimizer(obj_fcn, x0, bnds, coef_id, settings, opt_settings)
    res = optimizer.run()
    assert np.allclose(res.x, np.array([20.13393]), rtol=1e-5, atol=1e-5)

    # Test case 3: Test with nlopt algorithm
    settings = Settings(developer_mode=True, algorithm_choice="nlopt_sbplx")
    opt_settings = _get_opt_settings(settings)
    obj_fcn = get_obj_fcn(settings)
    optimizer = Optimizer(obj_fcn, x0, bnds, coef_id, settings, opt_settings)
    res = optimizer.run()
    assert np.allclose(res.x, np.array([20.13393]), rtol=1e-5, atol=1e-5)