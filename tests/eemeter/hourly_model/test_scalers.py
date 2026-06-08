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

from opendsm.eemeter.models.hourly.scalers import (
    SafeRobustScaler,
    SafeStandardScaler,
)



SAFE_SCALERS = [SafeStandardScaler, SafeRobustScaler]


@pytest.mark.parametrize("cls", SAFE_SCALERS)
def test_constant_feature_scale_clamped_to_one(cls):
    """A constant (zero-spread) feature gets scale_ clamped to 1.0, not 0.

    Only the degenerate column is clamped; the well-behaved column keeps its
    real scale, and transforming the constant feature stays finite.
    """
    rng = np.random.default_rng(0)
    data = np.column_stack([np.full(50, 7.0), rng.normal(0, 3, 50)])

    scaler = cls().fit(data)

    assert scaler.scale_[0] == 1.0
    assert scaler.scale_[1] != 1.0
    assert np.isfinite(scaler.transform(data)).all()


@pytest.mark.parametrize("cls", SAFE_SCALERS)
def test_direct_scale_assignment_is_clamped(cls):
    """Assigning a near-zero scale_ directly is clamped on the way in."""
    scaler = cls()

    scaler.scale_ = np.array([1e-9, 2.0])

    assert scaler.scale_[0] == 1.0
    assert scaler.scale_[1] == 2.0


@pytest.mark.parametrize("cls", SAFE_SCALERS)
def test_well_scaled_data_unaffected(cls):
    """Features with real spread keep their computed scale (no spurious clamp)."""
    rng = np.random.default_rng(1)
    data = rng.normal(0, 5, size=(100, 2))

    scaler = cls().fit(data)

    assert np.all(scaler.scale_ > 1e-6)
    assert np.all(scaler.scale_ != 1.0)
