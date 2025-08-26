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
from opendsm.eemeter.models.daily.parameters import ModelCoefficients
from opendsm.eemeter.models.daily.parameters import ModelType
from opendsm.eemeter.models.daily.utilities.settings import DailySettings as Settings
from opendsm.eemeter.models.daily.base_models.c_hdd_tidd import fit_c_hdd_tidd
from opendsm.eemeter.models.daily.fit_base_models import _get_opt_settings


def test_fit_c_hdd_tidd_smooth():
    # Test case 1: Test with initial_fit=True
    T = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(float)
    obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(float)
    settings = Settings(
        developer_mode=True,
        alpha_selection=0.1,
        alpha_final=0.2,
        segment_minimum_count=5,
        maximum_slope_OoM_scaler=1,
    )
    opt_options = _get_opt_settings(settings)
    x0 = ModelCoefficients(
        model_type=ModelType.HDD_TIDD_SMOOTH,
        intercept=0.0,
        hdd_bp=0.0,
        hdd_beta=0.0,
        hdd_k=0.0,
        cdd_bp=None,
        cdd_beta=None,
        cdd_k=None,
    )
    bnds = None
    weights = None
    initial_fit = True
    smooth = True
    res = fit_c_hdd_tidd(T, obs, weights, settings, opt_options, smooth, x0, bnds, initial_fit)
    assert res.success == True

    # Test case 2: Test with initial_fit=False
    T = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(float)
    obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(float)
    settings = Settings(
        developer_mode=True,
        alpha_selection=0.1,
        alpha_final=0.2,
        segment_minimum_count=5,
        maximum_slope_OoM_scaler=1,
    )
    opt_options = _get_opt_settings(settings)
    x0 = ModelCoefficients(
        model_type=ModelType.HDD_TIDD_SMOOTH,
        intercept=0.0,
        hdd_bp=0.0,
        hdd_beta=0.0,
        hdd_k=0.0,
        cdd_bp=None,
        cdd_beta=None,
        cdd_k=None,
    )
    bnds = None
    initial_fit = False
    smooth = True
    res = fit_c_hdd_tidd(T, obs, weights, settings, opt_options, smooth, x0, bnds, initial_fit)
    assert res.success == True

    # Test case 3: Test with x0=None
    T = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).astype(float)
    obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(float)
    settings = Settings(
        developer_mode=True,
        alpha_selection=0.1,
        alpha_final=0.2,
        segment_minimum_count=5,
        maximum_slope_OoM_scaler=1,
    )
    opt_options = _get_opt_settings(settings)
    x0 = None
    bnds = None
    initial_fit = True
    smooth = True
    res = fit_c_hdd_tidd(T, obs, weights, settings, opt_options, smooth, x0, bnds, initial_fit)
    assert res.success == True