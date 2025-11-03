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
from opendsm.eemeter.models.daily.base_models.full_model import full_model


def test_full_model_import():
    hdd_bp = 50
    hdd_beta = 0.01
    hdd_k = 0.001
    cdd_bp = 80
    cdd_beta = 0.02
    cdd_k = 0.002
    intercept = 100
    T_fit_bnds = np.array([10, 100]).astype(np.double)
    T = np.linspace(10, 100, 130).astype(np.double)

    res = full_model(
        hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept, T_fit_bnds, T
    )
    assert res.size == T.size
