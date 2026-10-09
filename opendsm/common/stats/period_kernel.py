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

"""Calendar cell grids and the Cauchy product kernel that smooths values across them.

A cadence maps each timestep to an integer cell q of a fixed grid:

    hourly   "hour_of_week_x_month"   q = (month - 1) * 168 + weekday * 24 + hour   2016 cells
    daily    "day_of_week_x_month"    q = (month - 1) * 7 + weekday                  84 cells
    billing  "month"                  q = month - 1                                  12 cells

with weekday 0 = Monday, hour and month in local time. Each cell has a position on up to three
circles (hour-of-day on 24 h, weekday on 7 d, day-of-year on 365 d, a month sitting at its day
15 in a non-leap year). The kernel between two cells is the product of Cauchy factors
1 / (1 + (distance / bandwidth)^2) on those circles, dropping a factor whenever the cadence step
is not shorter than the factor's period: daily drops hour-of-day, billing drops hour-of-day and
weekday.
"""

import numpy as np



SCHEMES = {
    "hourly": "hour_of_week_x_month",
    "daily": "day_of_week_x_month",
    "billing": "month",
}

N_CELLS = {
    "hour_of_week_x_month": 2016,
    "day_of_week_x_month": 84,
    "month": 12,
}

_HOURS_PER_DAY = 24
_DAYS_PER_WEEK = 7
_DAYS_PER_YEAR = 365

# day-of-year of day 15 of each month in a non-leap year
_MONTH_MID_DAY = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]) + 15


def cell_scheme(cadence: str) -> str:
    """Name of the cell scheme used at a cadence ("hourly", "daily" or "billing")."""
    if cadence not in SCHEMES:
        raise ValueError(f"Unknown cadence {cadence!r}; expected one of {list(SCHEMES)}")

    return SCHEMES[cadence]


def cell_index(index, cadence: str) -> np.ndarray:
    """Cell index q of each timestamp of a tz-aware DatetimeIndex in local time.

    At billing cadence the index is the read periods' midpoints.
    """
    scheme = cell_scheme(cadence)
    month = np.asarray(index.month, dtype=int) - 1

    if scheme == "hour_of_week_x_month":
        weekday = np.asarray(index.dayofweek, dtype=int)
        hour = np.asarray(index.hour, dtype=int)
        q = month * 168 + weekday * 24 + hour
    elif scheme == "day_of_week_x_month":
        weekday = np.asarray(index.dayofweek, dtype=int)
        q = month * 7 + weekday
    else:
        q = month

    return q


def _circle_factor(positions: np.ndarray, period: float, bandwidth: float) -> np.ndarray:
    """Cauchy factor 1 / (1 + (circular distance / bandwidth)^2) between all position pairs."""
    u = circular_distance(positions[:, None], positions[None, :], period) / bandwidth
    factor = 1.0 / (1.0 + u**2)

    return factor


def circular_distance(a: np.ndarray, b: np.ndarray, period: float) -> np.ndarray:
    """Shortest distance between positions a and b on a circle of the given period."""
    d = np.abs(a - b) % period
    d = np.minimum(d, period - d)

    return d


def cauchy_kernel(
    cadence: str,
    hour_bandwidth_h: float,
    day_of_week_bandwidth_d: float,
    calendar_bandwidth_d: float,
) -> np.ndarray:
    """(n_cells, n_cells) product of Cauchy factors over the cadence's kept circles.

    The cells are the Cartesian product month x weekday x hour (outer to inner, matching q),
    so the product kernel is the Kronecker product of the per-circle factors.
    """
    scheme = cell_scheme(cadence)
    G = _circle_factor(_MONTH_MID_DAY, _DAYS_PER_YEAR, calendar_bandwidth_d)

    if scheme in ("hour_of_week_x_month", "day_of_week_x_month"):
        weekday = _circle_factor(np.arange(_DAYS_PER_WEEK), _DAYS_PER_WEEK, day_of_week_bandwidth_d)
        G = np.kron(G, weekday)

    if scheme == "hour_of_week_x_month":
        hour = _circle_factor(np.arange(_HOURS_PER_DAY), _HOURS_PER_DAY, hour_bandwidth_h)
        G = np.kron(G, hour)

    return G


def cell_means(q: np.ndarray, values: np.ndarray, n_cells: int) -> np.ndarray:
    """Per-cell mean of values grouped by cell index; empty cells are 0."""
    counts = np.bincount(q, minlength=n_cells)
    sums = np.bincount(q, weights=values, minlength=n_cells)
    means = np.divide(sums, counts, out=np.zeros(n_cells), where=counts > 0)

    return means


def smooth(values: np.ndarray, counts: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Count-weighted kernel average over the non-empty cells.

    smoothed[q] = sum_q' G[q, q'] N[q'] v[q'] / sum_q' G[q, q'] N[q'] over cells with N > 0.
    Every cell, empty or not, receives a value when at least one cell is non-empty.
    """
    weighted = np.where(counts > 0, counts * values, 0.0)
    smoothed = (G @ weighted) / (G @ counts)

    return smoothed
