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
import numpy as np

from opendsm.common.metrics import acf


# TODO: this is incomplete, need to add more tests
def test_acf():
    # Test case 1: Test with a simple input array
    x = np.array([1, 2, 3, 4, 5])
    expected_output = np.array([1.0, 0.4, -0.1, -0.4])
    assert np.allclose(acf(x), expected_output)

    # Test case 3: Test with a moving mean and standard deviation
    x = np.array([1, 2, 3, 4, 5])
    expected_output = np.array([1.0, 1.0, 1.0, 1.0])
    assert np.allclose(acf(x, ac_type="moving_stats"), expected_output)

    # Test case 4: Test with a specific lag_n
    x = np.array([1, 2, 3, 4, 5])
    expected_output = np.array([1.0, 0.4])
    assert np.allclose(acf(x, lag_n=1), expected_output)