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

from opendsm.eemeter.models.daily.base_models.full_model import (
    full_model,
    get_full_model_x,
)

T_MIN = 50.0
T_MIN_SEG = 55.0
T_MAX_SEG = 85.0
T_MAX = 90.0
T_FIT_BNDS = np.array([T_MIN, T_MAX])
INTERCEPT = 100.0


def _c_hdd_tidd_bps(bp, beta):
    x = np.array([bp, beta, INTERCEPT])
    hdd_bp, _, _, cdd_bp, _, _, _ = get_full_model_x(
        "c_hdd_tidd", x, T_MIN, T_MAX, T_MIN_SEG, T_MAX_SEG
    )

    return hdd_bp, cdd_bp


@pytest.mark.parametrize(
    ("bp", "beta"),
    [(T_MIN, 2.0), (T_MAX, -2.0)],
    ids=["cooling_from_T_min", "heating_from_T_max"],
)
def test_single_slope_balance_point_is_kept(bp, beta):
    """A cooling slope from T_min or a heating slope from T_max keeps its fitted balance point."""
    bps = _c_hdd_tidd_bps(bp, beta)

    assert bps == (bp, bp), f"expected balance point {bp}, got (hdd, cdd) = {bps}"


@pytest.mark.parametrize(
    ("bp", "beta", "expected"),
    [
        (52.0, 2.0, T_MIN_SEG),
        (88.0, -2.0, T_MAX_SEG),
        (T_MIN, -2.0, T_MIN_SEG),
        (T_MAX, 2.0, T_MAX_SEG),
    ],
    ids=[
        "cooling_in_min_buffer",
        "heating_in_max_buffer",
        "heating_at_T_min",
        "cooling_at_T_max",
    ],
)
def test_balance_point_in_temperature_buffer_moves_to_segment_bound(bp, beta, expected):
    """A balance point between a data edge and its segment bound, other than a single slope's
    own edge, moves to the segment bound."""
    bps = _c_hdd_tidd_bps(bp, beta)

    assert bps == (expected, expected), f"expected balance point {expected}, got {bps}"


@pytest.mark.parametrize(
    ("hdd_beta", "cdd_beta", "bp", "T"),
    [
        (0.0, 2.0, T_MIN, [30.0, 49.0, T_MIN]),
        (2.0, 0.0, T_MAX, [T_MAX, 91.0, 110.0]),
    ],
    ids=["cooling_below_T_min", "heating_above_T_max"],
)
def test_single_slope_is_flat_beyond_its_balance_point(hdd_beta, cdd_beta, bp, T):
    """Past the balance point at the edge of the fitted temperatures, a single slope predicts
    its intercept."""
    E = full_model(
        bp, hdd_beta, 0.0, bp, cdd_beta, 0.0, INTERCEPT, T_FIT_BNDS, np.array(T)
    )

    np.testing.assert_allclose(E, INTERCEPT, err_msg=f"temperatures {T} gave {E}")


@pytest.mark.parametrize(
    ("hdd_beta", "cdd_beta", "bp", "T", "expected"),
    [
        (0.0, 2.0, T_MIN, [95.0, 100.0], [190.0, 200.0]),
        (2.0, 0.0, T_MAX, [45.0, 30.0], [190.0, 220.0]),
    ],
    ids=["cooling_above_T_max", "heating_below_T_min"],
)
def test_single_slope_extends_beyond_the_data_on_its_slope_side(
    hdd_beta, cdd_beta, bp, T, expected
):
    """Beyond the fitted temperatures on the sloped side, a single slope keeps its trend."""
    E = full_model(
        bp, hdd_beta, 0.0, bp, cdd_beta, 0.0, INTERCEPT, T_FIT_BNDS, np.array(T)
    )

    np.testing.assert_allclose(E, expected, err_msg=f"temperatures {T} gave {E}")
