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

from opendsm.eemeter.models.hourly.regressors import (
    SafeElasticNet,
    SafeLasso,
    SafeLinearRegression,
    SafeRidge,
)



SAFE_REGRESSORS = [SafeLinearRegression, SafeRidge, SafeLasso, SafeElasticNet]


@pytest.fixture
def design_matrix():
    return np.random.default_rng(0).normal(size=(50, 3))


@pytest.mark.parametrize("cls", SAFE_REGRESSORS)
def test_null_fit_zero_target_single(cls, design_matrix):
    """An all-zero single target null-fits to zero coefficients and intercept."""
    model = cls().fit(design_matrix, np.zeros(50))

    assert np.all(model.coef_ == 0.0)
    assert model.intercept_ == 0.0


@pytest.mark.parametrize("cls", SAFE_REGRESSORS)
def test_null_fit_zero_target_multi(cls, design_matrix):
    """An all-zero 24-hour multi-target null-fits to all-zero coefficients.

    This is the hourly fabrication guard: a stuck/zero-usage target must not
    produce an invented load shape across the 24 hourly models.
    """
    model = cls().fit(design_matrix, np.zeros((50, 24)))

    assert model.coef_.shape == (24, 3)
    assert not np.any(model.coef_)
    assert not np.any(model.intercept_)


@pytest.mark.parametrize("cls", SAFE_REGRESSORS)
def test_nonzero_target_does_real_fit(cls, design_matrix):
    """A non-zero target falls through to the real estimator (non-trivial fit)."""
    y = design_matrix @ np.array([1.0, -2.0, 0.5]) + 3.0

    model = cls().fit(design_matrix, y)

    assert np.any(model.coef_ != 0.0)
