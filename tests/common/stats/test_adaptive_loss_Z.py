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

from opendsm.common.stats.adaptive_loss_Z import (
    ln_Z,
    ln_Z_inf,
    ln_Z_numba,
)



# The numba De Boor evaluator must reproduce the scipy BSpline reference used
# by the pure-Python ln_Z; the two are used interchangeably in hot loops.

@pytest.mark.parametrize(
    "alpha",
    [-100.0, -99.0, -50.0, -10.0, -2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 1.9, 2.0,
     2.1, 5.0, 50.0, 99.0, 99.9],
)
def test_ln_Z_numba_matches_bspline(alpha):
    """ln_Z_numba (De Boor) equals the scipy BSpline ln_Z across the knot range."""
    assert ln_Z_numba(alpha) == pytest.approx(ln_Z(alpha), abs=1e-9)


def test_ln_Z_asymptote_below_alpha_min():
    """At/below alpha_min both implementations return the analytic limit."""
    assert ln_Z(-100.0) == ln_Z_inf
    assert ln_Z(-150.0) == ln_Z_inf
    assert ln_Z_numba(-100.0) == pytest.approx(ln_Z_inf)
    assert ln_Z_numba(-150.0) == pytest.approx(ln_Z_inf)


def test_ln_Z_minimized_near_l2():
    """The partition penalty is smallest near alpha=2 (L2), the prior mode."""
    grid = np.linspace(-50.0, 2.0, 200)
    values = np.array([ln_Z(a) for a in grid])

    assert grid[np.argmin(values)] == pytest.approx(2.0, abs=0.5)
