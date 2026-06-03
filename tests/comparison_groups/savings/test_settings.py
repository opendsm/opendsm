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

import pytest

from opendsm.comparison_groups.savings.settings import (
    CGCorrectionSettings,
    CorrectionCapSettings,
    CorrectionCapChoice,
    OutlierRejectionSettings,
)


def test_correction_cap_defaults_are_valid():
    cap = CorrectionCapSettings()

    assert cap.type == CorrectionCapChoice.SOLAR
    assert cap.solar_threshold is not None


def test_solar_cap_requires_threshold():
    with pytest.raises(ValueError):
        CorrectionCapSettings(type=CorrectionCapChoice.SOLAR, solar_threshold=None)


def test_global_cap_rejects_solar_threshold():
    with pytest.raises(ValueError):
        CorrectionCapSettings(type=CorrectionCapChoice.GLOBAL)


def test_global_cap_with_null_threshold_is_valid():
    cap = CorrectionCapSettings(type=CorrectionCapChoice.GLOBAL, solar_threshold=None)

    assert cap.solar_threshold is None


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        CGCorrectionSettings(alpha=1.5)


def test_outlier_quantile_out_of_range_rejected():
    with pytest.raises(ValueError):
        OutlierRejectionSettings(quantile=0.6)
