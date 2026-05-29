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

from __future__ import annotations

import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler


_MIN_SCALE = 1e-6


class SafeStandardScaler(StandardScaler):
    """StandardScaler that clamps near-zero scale_ values to 1.0.

    Prevents numerical explosion when transforming near-constant features
    (where std dev is zero or near-zero). Clamping is applied automatically
    whenever scale_ is set, whether via fit() or direct assignment.
    """

    def __setattr__(self, name, value):
        if name == "scale_" and value is not None:
            value = np.asarray(value)
            value = np.where(np.abs(value) < _MIN_SCALE, 1.0, value)
        super().__setattr__(name, value)


class SafeRobustScaler(RobustScaler):
    """RobustScaler that clamps near-zero scale_ values to 1.0.

    Prevents numerical explosion when transforming features where the IQR
    is zero or near-zero (e.g. >50% identical values). Clamping is applied
    automatically whenever scale_ is set, whether via fit() or direct assignment.
    """

    def __setattr__(self, name, value):
        if name == "scale_" and value is not None:
            value = np.asarray(value)
            value = np.where(np.abs(value) < _MIN_SCALE, 1.0, value)
        super().__setattr__(name, value)
