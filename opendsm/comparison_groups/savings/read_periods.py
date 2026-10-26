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

Periods are inferred from the daily observed spread: within one read the
per-minute spread rate is constant, so day-length-normalized observed values are
equal within a read and jump between reads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd



# Valid read length in calendar days; periods outside this range after inference
# are not credible reads and route the meter to the ledger.
_MIN_READ_DAYS = 25
_MAX_READ_DAYS = 70

# A detected segment shorter than this is a spread blip (re-estimated or
# partial-coverage days) and is absorbed into a neighbor rather than emitted as
# its own read period.
_MIN_PERIOD_DAYS = 5

# Relative tolerance below which two day-length-normalized spread rates are the
# same read. Adjacent reads differ by whole-percent consumption changes; within
# a read the rate is constant up to float noise.
_RATE_RTOL = 1e-2

# A span at least this multiple of the detected median cadence is treated as
# merged near-identical reads and subdivided into equal periods.
_SUBDIVIDE_RATIO = 1.5

_NO_OBSERVED_REASON = "reporting observed is absent or all NaN"


class ReadPeriodResult:
    """Read periods for one billing treatment meter.

    ``periods`` is the ordered, non-overlapping list of ``(start, end)`` tz-aware
    daily datetimes (both inclusive), or ``None`` when no credible periods could
    be drawn. ``ledger_reason`` is the human string for the reporting_data ledger
    entry when ``periods`` is ``None`` (``None`` otherwise).
    """

    def __init__(self, periods, ledger_reason):
        self.periods = periods
        self.ledger_reason = ledger_reason


def periods_from_boundaries(index, read_boundaries):
    """Build read periods from explicit period start datetimes.

    Each boundary is snapped to the daily row containing it; a tz-naive boundary
    is read in the index's zone. The first row always opens the first period and
    the last row always closes the last, so a boundary before the first row or
    after the last row's day adds no period. The periods are not checked against
    the valid read length.

    Args:
        index: tz-aware daily ``DatetimeIndex`` of the meter's substrate.
        read_boundaries: period start datetimes, in any order.

    Returns:
        ``ReadPeriodResult`` with ``periods`` always set.
    """
    starts = {0}

    for boundary in read_boundaries:
        stamp = pd.Timestamp(boundary)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(index.tz)
        else:
            stamp = stamp.tz_convert(index.tz)

        day = stamp.normalize()
        if day > index[-1]:
            continue

        pos = int(index.get_indexer([day], method="ffill")[0])
        starts.add(max(pos, 0))

    periods = _starts_to_periods(index, sorted(starts))
    result = ReadPeriodResult(periods, None)

    return result


def infer_read_periods(daily, prior_periods=None):
    """Infer read periods for one billing treatment meter's daily substrate.

    Boundaries are drawn only between finite days, so NaN days before the first
    read, after the last, or between two reads of different rate belong to no
    period. Spurious short segments are absorbed into a neighbor, and a span of
    merged near-identical reads is subdivided into equal periods at the median
    cadence.

    Args:
        daily: DataFrame with a tz-aware ``DatetimeIndex`` and an ``observed``
            column at daily cadence.
        prior_periods: optional ``[(start, end), ...]`` from a prior result on a
            prefix of ``daily``. Every period but the last is kept unchanged and
            only the region from the last period's start is re-inferred, with
            the subdivision median taken from the kept periods. With a single
            prior period the region from its start is inferred afresh.

    Returns:
        ``ReadPeriodResult``. ``periods`` is ``None`` with a ``ledger_reason``
        when observed is absent or all NaN, when a lone run longer than the
        valid read length leaves no recoverable cadence, or when any period
        falls outside the valid read length.
    """
    index = daily.index
    observed = daily["observed"].to_numpy(dtype=np.float64)

    if observed.size == 0 or not np.isfinite(observed).any():
        result = ReadPeriodResult(None, _NO_OBSERVED_REASON)

        return result

    if prior_periods:
        result = _infer_with_prior(index, observed, prior_periods)

        return result

    runs = _segment(index, observed)
    result = _finalize(index, runs)

    return result


def _infer_with_prior(index, observed, prior_periods):
    frozen = list(prior_periods[:-1])
    start_pos = int(index.get_loc(prior_periods[-1][0]))

    sub_runs = _segment(index[start_pos:], observed[start_pos:])
    runs = [(start + start_pos, end + start_pos) for start, end in sub_runs]

    if not frozen:
        result = _finalize(index, runs)

        return result

    if not runs:
        result = ReadPeriodResult(None, _NO_OBSERVED_REASON)

        return result

    frozen_runs = [(int(index.get_loc(start)), int(index.get_loc(end))) for start, end in frozen]
    median = float(np.median([_run_length(run) for run in frozen_runs]))
    runs = _subdivide_runs(runs, median)

    result = _check_ranges(index, runs)
    if result.periods is None:
        return result

    result = ReadPeriodResult(frozen + result.periods, None)

    return result


def _finalize(index, runs):
    """Turn detected runs into validated read periods or a ledger reason."""
    if not runs:
        result = ReadPeriodResult(None, _NO_OBSERVED_REASON)

        return result

    if len(runs) == 1:
        return _finalize_single(index, runs[0])

    median = float(np.median([_run_length(run) for run in runs]))
    runs = _subdivide_runs(runs, median)

    return _check_ranges(index, runs)


def _finalize_single(index, run):
    """A lone run longer than the valid read length is a constant spread with no
    resolvable cadence; any other lone run is validated like a set of runs."""
    length = _run_length(run)

    if length > _MAX_READ_DAYS:
        reason = (
            f"reporting spread is constant over {length} days with no recoverable read "
            "cadence; pass explicit read_boundaries"
        )
        result = ReadPeriodResult(None, reason)

        return result

    return _check_ranges(index, [run])


def _check_ranges(index, runs):
    """Validate every run against the valid day range; any offender ledgers the
    meter."""
    offenders = [_run_length(run) for run in runs if not _valid_length(run)]

    if offenders:
        reason = (
            f"inferred read period(s) span {offenders} days, outside the valid "
            f"{_MIN_READ_DAYS}-{_MAX_READ_DAYS} day range"
        )
        result = ReadPeriodResult(None, reason)

        return result

    periods = [_period(index, run) for run in runs]
    result = ReadPeriodResult(periods, None)

    return result


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
    spans = [(run[0], run[1]) for run in runs]

    return spans


def _absorb(runs):
    """Coalesce adjacent same-rate runs and absorb sub-guard runs into their
    closest-rate neighbor until stable. Absorbing a mid-read blip lets the two
    surrounding runs coalesce back into one read."""
    changed = True

    while changed:
        changed = False

        i = 0
        while i < len(runs) - 1:
            left = runs[i]
            right = runs[i + 1]

            if _close(left[2], right[2]):
                count = left[3] + right[3]
                rate = (left[2] * left[3] + right[2] * right[3]) / count
                runs[i] = [left[0], right[1], rate, count]
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
    out = []
    for start, end in runs:
        length = end - start + 1

        if length >= _SUBDIVIDE_RATIO * median:
            n = int(np.floor(length / median + 0.5))
            edges = np.linspace(start, end + 1, n + 1)
            edges = np.floor(edges + 0.5).astype(int)

            for k in range(n):
                out.append((int(edges[k]), int(edges[k + 1] - 1)))
        else:
            out.append((start, end))

    return out


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

    close = abs(a - b) <= _RATE_RTOL * scale

    return close


def _run_length(run):
    return run[1] - run[0] + 1


def _valid_length(run):
    length = _run_length(run)

    return _MIN_READ_DAYS <= length <= _MAX_READ_DAYS


def _period(index, run):
    period = (index[run[0]], index[run[1]])

    return period

