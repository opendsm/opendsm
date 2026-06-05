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



def test_acf():
    """ACF per method on a linear ramp; pins each method's defined output."""
    x = np.array([1, 2, 3, 4, 5])

    # Default is MOVING_STATS: every shifted slice of a linear ramp is perfectly
    # correlated, so each lag is 1.0.
    assert np.allclose(acf(x), [1.0, 1.0, 1.0, 1.0])
    assert np.allclose(acf(x, ac_type="moving_stats"), [1.0, 1.0, 1.0, 1.0])

    # STATIONARY_CORRELATE is the standard biased ACF of [1, 2, 3, 4, 5].
    assert np.allclose(acf(x, ac_type="stationary_correlate"), [1.0, 0.4, -0.1, -0.4])

    # lag_n caps the number of lags returned (lags 0..lag_n).
    assert np.allclose(acf(x, lag_n=1), [1.0, 1.0])
    assert np.allclose(acf(x, ac_type="stationary_correlate", lag_n=1), [1.0, 0.4])


@pytest.mark.xfail(
    reason="STATIONARY_STATS_FFT omits zero-padding, so it computes the circular "
    "(not linear) autocorrelation and diverges from STATIONARY_CORRELATE. Fix "
    "tracked for the common/metrics work.",
    strict=True,
)
def test_acf_stationary_fft_matches_correlate():
    """The FFT and correlate stationary methods should agree (both = linear ACF)."""
    x = np.array([1, 2, 3, 4, 5])

    assert np.allclose(
        acf(x, ac_type="stationary_stats_fft"),
        acf(x, ac_type="stationary_correlate"),
    )