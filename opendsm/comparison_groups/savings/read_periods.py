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

"""Read-period inference for billing treatment meters.

A billing meter rides a contiguous daily substrate internally; the raw read
datetimes are consumed during data processing and do not survive onto the
processed frame. This module recovers the read periods that partition that
substrate, so the correction can aggregate the meter back to its received
cadence.

A caller may pass explicit ``read_boundaries`` (period start datetimes), which
always win. Otherwise the periods are inferred from the daily observed spread:
within one read the per-minute spread rate is constant, so day-length-normalized
observed values are equal within a read and jump between reads. Boundaries are
drawn only between finite days (NaN days belong to no period); spurious short
segments are absorbed into a neighbor; and a merged span of near-identical reads
is subdivided into equal periods at the detected cadence. When a prior result is
supplied its confirmed periods are frozen and only the trailing region is
re-inferred, so an extension can never re-draw an earlier period.

Meters whose cadence cannot be recovered (no observed, a constant spread with no
resolvable cadence, or an inferred period outside the valid day range) are
reported with a ledger reason for the caller to record and drop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd



# Valid read length in calendar days; periods outside this range after inference
# are not credible reads and route the meter to the ledger.
_MIN_READ_DAYS = 25
_MAX_READ_DAYS = 70

# A detected segment shorter than this is treated as a single-day spread blip
# (a re-estimated or partial-coverage day) and absorbed into a neighbor rather
# than emitted as its own read period.
_MIN_PERIOD_DAYS = 5

# Relative tolerance below which two day-length-normalized spread rates are the
# same read. Adjacent reads differ by whole-percent consumption changes; within
# a read the rate is constant up to float noise.
_RATE_RTOL = 1e-2

# A span at least this multiple of the detected median cadence is treated as
# merged near-identical reads and subdivided into equal periods.
_SUBDIVIDE_RATIO = 1.5


class ReadPeriodResult:
    """Inferred read periods for one billing treatment meter.

    ``periods`` is the ordered list of ``(start, end)`` tz-aware daily datetimes
    (both inclusive) partitioning the meter's daily substrate, or ``None`` when
    the cadence could not be recovered. ``ledger_reason`` is the human string for
    the reporting_data ledger entry when ``periods`` is ``None`` (``None``
    otherwise).
    """

    def __init__(self, periods, ledger_reason):
        self.periods = periods
        self.ledger_reason = ledger_reason


def derive_read_periods(daily, read_boundaries=None, prior_periods=None):
    """Derive read periods for one billing treatment meter's daily substrate.

    Args:
        daily: DataFrame with a tz-aware ``DatetimeIndex`` and an ``observed``
            column at daily cadence.
        read_boundaries: optional explicit period start datetimes; when given
            they win outright and no inference runs.
        prior_periods: optional ``[(start, end), ...]`` from a prior result. Its
            earlier periods are frozen and only the trailing region is
            re-inferred; the median cadence is not recomputed over the extended
            data.

    Returns:
        ``ReadPeriodResult``. ``periods`` is ``None`` with a ``ledger_reason``
        when the cadence cannot be recovered.
    """
    index = daily.index

    if read_boundaries is not None:
        periods = _periods_from_boundaries(index, read_boundaries)

        return ReadPeriodResult(periods, None)

    observed = daily["observed"].to_numpy(dtype=np.float64)

    if observed.size == 0 or not np.isfinite(observed).any():
        return ReadPeriodResult(None, "reporting observed is absent or all NaN")

    if prior_periods:
        return _infer_with_prior(index, observed, prior_periods)

    runs = _segment(index, observed)
    periods, reason = _finalize(index, runs)

    return ReadPeriodResult(periods, reason)


def _infer_with_prior(index, observed, prior_periods):
    """Freeze the prior's confirmed periods and re-infer only the trailing
    region. The subdivision median is taken from the frozen periods, so
    extending the data can never move an earlier boundary."""
    frozen = list(prior_periods[:-1])
    trailing_start = prior_periods[-1][0]
    start_pos = int(index.get_loc(trailing_start))

    sub_runs = _segment(index[start_pos:], observed[start_pos:])
    sub_runs = [(start + start_pos, end + start_pos) for start, end in sub_runs]

    if not sub_runs:
        return ReadPeriodResult(None, "reporting observed is absent or all NaN")

    frozen_lengths = [_frozen_length(index, start, end) for start, end in frozen]
    if frozen_lengths:
        median = float(np.median(frozen_lengths))
        sub_runs = _subdivide_runs(sub_runs, median)

    periods, reason = _check_ranges(index, sub_runs)
    if reason is not None:
        return ReadPeriodResult(None, reason)

    combined = frozen + periods

    return ReadPeriodResult(combined, None)


def _finalize(index, runs):
    """Turn detected runs into validated read periods or a ledger reason."""
    if not runs:
        reason = "reporting observed is absent or all NaN"

        return None, reason

    if len(runs) == 1:
        return _finalize_single(index, runs[0])

    median = float(np.median([_run_length(run) for run in runs]))
    runs = _subdivide_runs(runs, median)

    return _check_ranges(index, runs)


def _finalize_single(index, run):
    """A lone detected run is a clean single read only inside the valid range; a
    longer constant spread has no resolvable cadence and a shorter one is a
    truncated read — both route to the ledger."""
    length = _run_length(run)

    if _MIN_READ_DAYS <= length <= _MAX_READ_DAYS:
        periods = [_period(index, run)]

        return periods, None

    if length > _MAX_READ_DAYS:
        reason = (
            f"reporting spread is constant over {length} days with no recoverable read "
            "cadence; pass explicit read_boundaries"
        )

        return None, reason

    reason = (
        f"the only inferred read period spans {length} days, outside the valid "
        f"{_MIN_READ_DAYS}-{_MAX_READ_DAYS} day range"
    )

    return None, reason


def _check_ranges(index, runs):
    """Validate every run against the valid day range; any offender ledgers the
    meter."""
    offenders = [_run_length(run) for run in runs if not _valid_length(run)]

    if offenders:
        reason = (
            f"inferred read period(s) span {offenders} days, outside the valid "
            f"{_MIN_READ_DAYS}-{_MAX_READ_DAYS} day range"
        )

        return None, reason

    periods = [_period(index, run) for run in runs]

    return periods, None


def _segment(index, observed):
    """Group finite days into runs of equal day-length-normalized spread rate,
    then absorb spurious short segments. Returns ``[(start_pos, end_pos), ...]``
    over the given index (both inclusive)."""
    finite = np.flatnonzero(np.isfinite(observed))

    if finite.size == 0:
        return []

    hours = _day_hours(index)
    norm = observed / hours

    runs = []
    for pos in finite:
        rate = float(norm[pos])

        if runs and _close(rate, runs[-1][2]):
            run = runs[-1]
            run[1] = int(pos)
            run[3] += 1
            run[2] += (rate - run[2]) / run[3]
        else:
            runs.append([int(pos), int(pos), rate, 1])

    runs = _absorb(runs)

    return [(run[0], run[1]) for run in runs]


def _absorb(runs):
    """Coalesce adjacent same-rate runs and absorb sub-guard runs into their
    closest-rate neighbor until stable. Absorbing a mid-read blip lets the two
    surrounding runs coalesce back into one read."""
    changed = True

    while changed:
        changed = False

        i = 0
        while i < len(runs) - 1:
            if _close(runs[i][2], runs[i + 1][2]):
                merged_rate = (runs[i][2] + runs[i + 1][2]) / 2.0
                runs[i] = [runs[i][0], runs[i + 1][1], merged_rate, runs[i][3] + runs[i + 1][3]]
                del runs[i + 1]
                changed = True
            else:
                i += 1

        if len(runs) < 2:
            break

        for i, run in enumerate(runs):
            if run[1] - run[0] + 1 >= _MIN_PERIOD_DAYS:
                continue

            target = _closest_neighbor(runs, i)
            keep = runs[target]
            runs[target] = [min(keep[0], run[0]), max(keep[1], run[1]), keep[2], keep[3]]
            del runs[i]
            changed = True
            break

    return runs


def _closest_neighbor(runs, i):
    """Index of the neighbor whose rate is nearest run ``i`` (left on a tie)."""
    if i == 0:
        return i + 1

    if i == len(runs) - 1:
        return i - 1

    left = i - 1
    right = i + 1

    rate = runs[i][2]
    if abs(runs[right][2] - rate) < abs(runs[left][2] - rate):
        return right

    return left


def _subdivide_runs(runs, median):
    """Split any run at least ``_SUBDIVIDE_RATIO`` times the median cadence into
    ``round(length / median)`` equal periods, leaving no sub-minimum sliver."""
    if median <= 0.0:
        return runs

    out = []
    for start, end in runs:
        length = end - start + 1

        if length >= _SUBDIVIDE_RATIO * median:
            n = int(np.floor(length / median + 0.5))
            n = max(n, 2)
            edges = np.linspace(start, end + 1, n + 1)
            edges = np.floor(edges + 0.5).astype(int)

            for k in range(n):
                out.append((int(edges[k]), int(edges[k + 1] - 1)))
        else:
            out.append((start, end))

    return out


def _periods_from_boundaries(index, boundaries):
    """Build ``(start, end)`` periods from explicit period start datetimes, each
    snapped to the daily row containing it. The frame start always opens the
    first period and the frame end always closes the last."""
    starts = {0}

    for boundary in boundaries:
        stamp = pd.Timestamp(boundary)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(index.tz)
        else:
            stamp = stamp.tz_convert(index.tz)

        pos = int(index.get_indexer([stamp.normalize()], method="ffill")[0])
        starts.add(max(pos, 0))

    starts = sorted(starts)
    periods = _starts_to_periods(index, starts)

    return periods


def _starts_to_periods(index, starts):
    periods = []

    for i, start in enumerate(starts):
        if i + 1 < len(starts):
            end = starts[i + 1] - 1
        else:
            end = len(index) - 1

        periods.append((index[start], index[end]))

    return periods


def _day_hours(index):
    """Elapsed hours spanned by each daily row (23/25 across DST transitions);
    the final row reuses the prior span."""
    if len(index) == 1:
        return np.array([24.0])

    deltas = np.diff(index.as_unit("ns").asi8) / 3.6e12
    hours = np.append(deltas, deltas[-1])

    return hours


def _close(a, b):
    scale = max(abs(a), abs(b))

    if scale == 0.0:
        return True

    return abs(a - b) <= _RATE_RTOL * scale


def _run_length(run):
    return run[1] - run[0] + 1


def _valid_length(run):
    length = _run_length(run)

    return _MIN_READ_DAYS <= length <= _MAX_READ_DAYS


def _period(index, run):
    period = (index[run[0]], index[run[1]])

    return period


def _frozen_length(index, start, end):
    length = int(index.get_loc(end)) - int(index.get_loc(start)) + 1

    return length
