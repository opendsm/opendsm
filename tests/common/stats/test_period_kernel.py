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
import pandas as pd
import pytest

from opendsm.common.stats.period_kernel import (
    N_CELLS,
    SCHEMES,
    cauchy_kernel,
    cell_index,
    cell_means,
    cell_scheme,
    circular_distance,
    smooth,
)



TZ = "America/Los_Angeles"
CADENCES = ["hourly", "daily", "billing"]


def _kernel(cadence, bandwidth=None, **overrides):
    """Kernel with every bandwidth at `bandwidth` (defaults when None), then overrides."""
    if bandwidth is None:
        bandwidths = {
            "hour_bandwidth_h": 2.0,
            "day_of_week_bandwidth_d": 1.0,
            "calendar_bandwidth_d": 28.0,
        }
    else:
        bandwidths = {
            "hour_bandwidth_h": bandwidth,
            "day_of_week_bandwidth_d": bandwidth,
            "calendar_bandwidth_d": bandwidth,
        }

    bandwidths.update(overrides)
    G = cauchy_kernel(cadence, **bandwidths)

    return G


# ---------------------------------------------------------------------------
# cell index
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "timestamp, expected",
    [
        # Monday 2024-01-01 00:00: the first cell of every scheme
        ("2024-01-01 00:00", {"hourly": 0, "daily": 0, "billing": 0}),
        # Wednesday 2023-07-12 14:00: month 7, weekday 2, hour 14
        ("2023-07-12 14:00", {"hourly": 6 * 168 + 2 * 24 + 14, "daily": 6 * 7 + 2, "billing": 6}),
        # Sunday 2023-12-31 23:00: the last cell of every scheme
        ("2023-12-31 23:00", {"hourly": 2015, "daily": 83, "billing": 11}),
    ],
)
@pytest.mark.parametrize("cadence", CADENCES)
def test_cell_index_of_known_local_timestamps(timestamp, expected, cadence):
    index = pd.DatetimeIndex([pd.Timestamp(timestamp, tz=TZ)])

    q = cell_index(index, cadence)

    assert q.tolist() == [expected[cadence]], f"{cadence} {timestamp}: got {q}"


def test_cell_index_uses_local_time_not_utc():
    """2024-01-01 07:30 UTC is Sunday 2023-12-31 23:30 in Los Angeles: the last hourly cell."""
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-01 07:30", tz="UTC")]).tz_convert(TZ)

    assert cell_index(index, "hourly").tolist() == [2015]


@pytest.mark.parametrize("cadence", CADENCES)
def test_cell_index_covers_every_cell_of_a_year(cadence):
    index = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h", tz=TZ)

    q = cell_index(index, cadence)

    assert np.array_equal(np.unique(q), np.arange(N_CELLS[SCHEMES[cadence]]))


def test_unknown_cadence_raises():
    with pytest.raises(ValueError, match="Unknown cadence"):
        cell_scheme("15min")


# ---------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------

def test_circular_distance_wraps_around_the_circle():
    a = np.array([23, 0, 6, 0])
    b = np.array([0, 12, 0, 334])
    period = np.array([24, 24, 7, 365])

    d = circular_distance(a, b, period)

    assert d.tolist() == [1, 12, 1, 31]


@pytest.mark.parametrize("cadence", CADENCES)
def test_kernel_shape_symmetry_and_unit_diagonal(cadence):
    G = _kernel(cadence)
    n_cells = N_CELLS[SCHEMES[cadence]]

    assert G.shape == (n_cells, n_cells)
    assert np.allclose(G, G.T, rtol=0, atol=1e-15)
    assert np.allclose(np.diag(G), 1.0, rtol=0, atol=1e-15)
    assert np.all(G > 0)


def test_kernel_hourly_entries_match_hand_computed_factors():
    G = _kernel("hourly")

    # q 0 and q 23: hours 0 and 23 of a January Monday, one hour apart on the 24 h circle
    assert G[0, 23] == pytest.approx(1 / (1 + (1 / 2.0) ** 2), rel=1e-12)
    # q 0 and q 24: Monday and Tuesday at hour 0 in January
    assert G[0, 24] == pytest.approx(1 / (1 + (1 / 1.0) ** 2), rel=1e-12)
    # January and December, same weekday and hour: 31 days apart on the 365 d circle
    assert G[0, 11 * 168] == pytest.approx(1 / (1 + (31 / 28.0) ** 2), rel=1e-12)


@pytest.mark.parametrize(
    "cadence, dropped, kept",
    [
        ("hourly", [], ["hour_bandwidth_h", "day_of_week_bandwidth_d", "calendar_bandwidth_d"]),
        ("daily", ["hour_bandwidth_h"], ["day_of_week_bandwidth_d", "calendar_bandwidth_d"]),
        ("billing", ["hour_bandwidth_h", "day_of_week_bandwidth_d"], ["calendar_bandwidth_d"]),
    ],
)
def test_kernel_drop_rule_per_cadence(cadence, dropped, kept):
    """A dropped factor's bandwidth has no effect on the kernel; a kept one does."""
    G = _kernel(cadence)

    for name in dropped:
        assert np.array_equal(_kernel(cadence, **{name: 1e6}), G), f"{cadence} kept {name}"

    for name in kept:
        assert not np.allclose(_kernel(cadence, **{name: 1e6}), G), f"{cadence} dropped {name}"


# ---------------------------------------------------------------------------
# cell means and smoothing
# ---------------------------------------------------------------------------

def test_cell_means_groups_by_cell_and_leaves_empty_cells_zero():
    q = np.array([0, 0, 2, 2, 2])
    values = np.array([1.0, 3.0, 2.0, 4.0, 6.0])

    means = cell_means(q, values, 4)

    assert means.tolist() == [2.0, 0.0, 4.0, 0.0]


@pytest.mark.parametrize("cadence", CADENCES)
def test_smooth_narrow_bandwidths_return_the_raw_cell_values(cadence):
    """At 0.01 bandwidths each cell's off-cell weight is below 1e-3 of its own."""
    n_cells = N_CELLS[SCHEMES[cadence]]
    values = np.random.default_rng(0).uniform(1.0, 2.0, n_cells)
    counts = np.ones(n_cells, dtype=int)

    smoothed = smooth(values, counts, _kernel(cadence, bandwidth=0.01))

    rel = np.abs(smoothed / values - 1).max()
    assert rel < 1e-3, f"max relative deviation from raw {rel:.2e}"


@pytest.mark.parametrize("cadence", CADENCES)
def test_smooth_wide_bandwidths_return_the_count_weighted_global_mean(cadence):
    rng = np.random.default_rng(1)
    n_cells = N_CELLS[SCHEMES[cadence]]
    values = rng.uniform(1.0, 2.0, n_cells)
    counts = rng.integers(0, 5, n_cells)
    counts[0] = 3
    expected = np.sum(counts * values) / np.sum(counts)

    smoothed = smooth(values, counts, _kernel(cadence, bandwidth=1e6))

    assert np.allclose(smoothed, expected, rtol=1e-6, atol=0)


def test_smooth_fills_empty_cells_with_finite_values():
    """A six-week hourly baseline fills every cell of the year, and empty cells' raw values
    (here NaN) never enter the average."""
    n_cells = N_CELLS["hour_of_week_x_month"]
    index = pd.date_range("2023-06-01", periods=6 * 168, freq="h", tz=TZ)
    q = cell_index(index, "hourly")
    counts = np.bincount(q, minlength=n_cells)
    values = np.full(n_cells, np.nan)
    values[counts > 0] = 2.0

    smoothed = smooth(values, counts, _kernel("hourly"))

    assert (counts == 0).sum() > 1500
    assert np.all(np.isfinite(smoothed))
    assert np.allclose(smoothed, 2.0, rtol=1e-12, atol=0)
