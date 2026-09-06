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

from opendsm.comparison_groups.savings.read_periods import derive_read_periods



# Fixed-offset zone: every day is 24h, so a constant per-read spread stays a
# constant normalized rate. The DST test uses a real zone explicitly.
_TZ = "Etc/GMT+5"
_DST_TZ = "America/New_York"


def _build_daily(specs, start="2021-01-01", tz=_TZ, dst_proportional=False):
    """Contiguous daily frame from ``[(rate, n_days), ...]``. The daily observed
    is the per-read rate, optionally scaled by the day's true length so a
    23/25-hour DST day carries proportionally less/more energy (as the real
    billing daily substrate does)."""
    total = sum(n for _, n in specs)
    index = pd.date_range(start, periods=total, freq="D", tz=tz)

    rates = np.empty(total, dtype=np.float64)
    pos = 0
    for rate, n in specs:
        rates[pos : pos + n] = rate
        pos += n

    if dst_proportional:
        hours = np.append(np.diff(index.as_unit("ns").asi8) / 3.6e12, 24.0)
        observed = rates * hours / 24.0
    else:
        observed = rates.copy()

    frame = pd.DataFrame({"observed": observed}, index=index)
    frame.index.name = "datetime"

    return frame


def _length(daily, period):
    start, end = period
    length = int(daily.index.get_loc(end)) - int(daily.index.get_loc(start)) + 1

    return length


def _lengths(daily, periods):
    values = [_length(daily, period) for period in periods]

    return values


def test_explicit_boundaries_win_over_inference():
    daily = _build_daily([(10.0, 90)])

    result = derive_read_periods(
        daily, read_boundaries=[daily.index[30], daily.index[60]]
    )

    assert result.ledger_reason is None
    starts = [period[0] for period in result.periods]
    ends = [period[1] for period in result.periods]
    assert starts == [daily.index[0], daily.index[30], daily.index[60]]
    assert ends == [daily.index[29], daily.index[59], daily.index[89]]


def test_regular_monthly_cycle_detected():
    specs = [(float(10 + i), 30) for i in range(12)]
    daily = _build_daily(specs)

    result = derive_read_periods(daily)

    assert result.ledger_reason is None
    assert _lengths(daily, result.periods) == [30] * 12


def test_bimonthly_cycle_detected():
    specs = [(float(10 + i), 60) for i in range(6)]
    daily = _build_daily(specs)

    result = derive_read_periods(daily)

    assert result.ledger_reason is None
    assert _lengths(daily, result.periods) == [60] * 6


def test_irregular_cycles_within_median_tolerance_detected():
    specs = [(10.0, 28), (20.0, 35), (30.0, 42), (40.0, 30), (50.0, 38)]
    daily = _build_daily(specs)

    result = derive_read_periods(daily)

    assert result.ledger_reason is None
    assert _lengths(daily, result.periods) == [28, 35, 42, 30, 38]


def test_single_valid_read_returns_one_period():
    daily = _build_daily([(10.0, 45)])

    result = derive_read_periods(daily)

    assert result.ledger_reason is None
    assert len(result.periods) == 1
    assert _length(daily, result.periods[0]) == 45


def test_dst_spanning_reads_have_no_spurious_boundaries():
    """Day-length normalization is load-bearing at a read boundary coinciding
    with a DST transition. The two reads straddling the fall-back boundary
    (Nov 3, a 25-hour day that opens the final read) differ in rate by exactly the
    25/24 day-length ratio: WITHOUT normalization the 25-hour day carries the same
    raw daily energy as the preceding read and is misassigned backward across the
    boundary, shifting the read lengths from 46/46 to 47/45. Normalization
    recovers the constant per-read rate and keeps every boundary where it belongs.
    The interior spring-forward day (Mar 10, 23 hours) likewise stays within its
    read. Removing normalization makes the pinned lengths below fail."""
    r_post = 40.0
    r_pre = r_post * 25.0 / 24.0  # the fall-back day at read 6's start hides this jump
    specs = [(10.0, 45), (15.0, 46), (20.0, 46), (25.0, 46), (30.0, 46), (r_pre, 46), (r_post, 46)]
    daily = _build_daily(specs, start="2019-02-01", tz=_DST_TZ, dst_proportional=True)

    hours = np.append(np.diff(daily.index.as_unit("ns").asi8) / 3.6e12, 24.0)
    assert hours.min() < 23.5  # spring-forward day (interior to read 1) is exercised
    assert hours.max() > 24.5  # fall-back day (opening the final read) is exercised

    result = derive_read_periods(daily)

    assert result.ledger_reason is None
    assert _lengths(daily, result.periods) == [45, 46, 46, 46, 46, 46, 46]


def test_nan_gap_between_reads_yields_no_extra_period():
    daily = _build_daily([(10.0, 34), (20.0, 30)])
    daily.iloc[30:34, 0] = np.nan

    result = derive_read_periods(daily)

    assert result.ledger_reason is None
    assert len(result.periods) == 2
    assert result.periods[0] == (daily.index[0], daily.index[29])
    assert result.periods[1] == (daily.index[34], daily.index[63])


def test_nan_gap_within_a_read_stays_one_period():
    daily = _build_daily([(10.0, 40)])
    daily.iloc[18:23, 0] = np.nan

    result = derive_read_periods(daily)

    assert result.ledger_reason is None
    assert len(result.periods) == 1
    assert result.periods[0] == (daily.index[0], daily.index[39])


def test_single_day_spread_blip_is_absorbed():
    daily = _build_daily([(10.0, 40)])
    daily.iloc[20, 0] = 3.0

    result = derive_read_periods(daily)

    assert result.ledger_reason is None
    assert len(result.periods) == 1
    assert _length(daily, result.periods[0]) == 40


def test_median_backup_subdivides_merged_identical_reads_into_equal_periods():
    specs = [(10.0, 30), (20.0, 30), (30.0, 30), (40.0, 30), (40.0, 30)]
    daily = _build_daily(specs)

    result = derive_read_periods(daily)

    assert result.ledger_reason is None
    assert len(result.periods) == 5
    assert _lengths(daily, result.periods) == [30, 30, 30, 30, 30]


def test_flat_reporting_reads_ledger():
    daily = _build_daily([(10.0, 90)])

    result = derive_read_periods(daily)

    assert result.periods is None
    assert "cadence" in result.ledger_reason


def test_single_day_frame_ledgers_as_too_short():
    index = pd.date_range("2021-01-01", periods=1, freq="D", tz=_TZ)
    daily = pd.DataFrame({"observed": [10.0]}, index=index)

    result = derive_read_periods(daily)

    assert result.periods is None
    assert "spans 1 days" in result.ledger_reason


def test_leading_blip_is_absorbed_into_its_only_neighbour():
    """A sub-guard run at the start of the series has one neighbour, so it joins
    the following read rather than becoming a period of its own."""
    daily = _build_daily([(1.0, 2), (2.0, 30), (3.0, 30)])

    result = derive_read_periods(daily)

    lengths = [(end - start).days + 1 for start, end in result.periods]
    assert lengths == [32, 30]


def test_interior_blip_joins_the_closer_rate_neighbour():
    """A sub-guard run between two reads joins the neighbour whose rate is
    nearer: here the right one."""
    daily = _build_daily([(2.0, 30), (2.9, 3), (3.0, 30)])

    result = derive_read_periods(daily)

    lengths = [(end - start).days + 1 for start, end in result.periods]
    assert lengths == [30, 33]


def test_truncated_final_period_ledgers():
    specs = [(10.0, 30), (20.0, 30), (30.0, 30), (40.0, 15)]
    daily = _build_daily(specs)

    result = derive_read_periods(daily)

    assert result.periods is None
    assert "25-70" in result.ledger_reason


def test_observed_all_nan_ledgers():
    daily = _build_daily([(10.0, 60)])
    daily["observed"] = np.nan

    result = derive_read_periods(daily)

    assert result.periods is None
    assert "absent or all NaN" in result.ledger_reason


def test_empty_frame_ledgers():
    index = pd.DatetimeIndex([], tz=_TZ, name="datetime")
    daily = pd.DataFrame({"observed": pd.Series([], dtype=np.float64)}, index=index)

    result = derive_read_periods(daily)

    assert result.periods is None
    assert "absent or all NaN" in result.ledger_reason


def test_extension_with_prior_redraws_only_trailing_period():
    prior_daily = _build_daily([(10.0, 30), (20.0, 30), (30.0, 26)])
    prior = derive_read_periods(prior_daily)
    assert prior.ledger_reason is None
    assert _lengths(prior_daily, prior.periods) == [30, 30, 26]

    extended = _build_daily([(10.0, 30), (20.0, 30), (30.0, 30), (40.0, 30)])
    result = derive_read_periods(extended, prior_periods=prior.periods)

    assert result.ledger_reason is None
    assert len(result.periods) == 4
    # earlier periods frozen from the prior
    assert result.periods[0] == prior.periods[0]
    assert result.periods[1] == prior.periods[1]
    # trailing period re-drawn from 26 to 30 days, plus the new read
    assert result.periods[2] == (extended.index[60], extended.index[89])
    assert result.periods[3] == (extended.index[90], extended.index[119])


def test_freeze_prevents_earlier_period_redraw_on_extension():
    prior_daily = _build_daily([(10.0, 60), (20.0, 30), (30.0, 30)])
    prior = derive_read_periods(prior_daily)
    assert prior.ledger_reason is None
    # the merged 60-day span was subdivided into two 30-day periods
    assert _lengths(prior_daily, prior.periods) == [30, 30, 30, 30]

    extended = _build_daily(
        [(10.0, 60), (20.0, 30), (30.0, 30), (40.0, 42), (50.0, 42), (60.0, 42), (70.0, 42)]
    )

    # a global-median recompute over the extended data leaves the leading span
    # as ONE 60-day period (its median rises above the 1.5x subdivide trigger)
    naive = derive_read_periods(extended)
    assert naive.ledger_reason is None
    assert _length(extended, naive.periods[0]) == 60

    # freezing the prior keeps the leading span as its two 30-day periods
    frozen = derive_read_periods(extended, prior_periods=prior.periods)
    assert frozen.ledger_reason is None
    assert _length(extended, frozen.periods[0]) == 30
    assert _length(extended, frozen.periods[1]) == 30
    assert frozen.periods[0] == prior.periods[0]
    assert frozen.periods[1] == prior.periods[1]
