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

"""Seeded perturbation scenarios for compliance and regression testing.

Frame scenarios take a baseline and reporting frame and return a perturbed copy of
each; unless the scenario name says otherwise only the baseline is altered. Billing
schedule scenarios operate on a monthly frame, whose daily index carries the period
value on the read date and NaN in between. Every scenario copies its inputs and is
deterministic under its default seed.
"""

import numpy as np
import pandas as pd



# ===== Frame scenarios =====


def single_hour_gaps(baseline, reporting, seed=0):
    """Isolated missing hours scattered through the baseline; hits data coverage."""
    baseline = baseline.copy()
    reporting = reporting.copy()
    n_slots = len(baseline) // 2
    n_gaps = max(1, n_slots // 10)
    slots = np.random.default_rng(seed).choice(n_slots, n_gaps, replace=False)
    baseline.iloc[slots * 2, baseline.columns.get_loc("observed")] = np.nan

    return baseline, reporting


def multi_day_gap(baseline, reporting, seed=0):
    """Three consecutive missing days in the baseline; hits the consecutive gap limit."""
    baseline = baseline.copy()
    reporting = reporting.copy()
    start = _random_block_start(baseline.index, 3, seed)
    baseline.loc[_block_mask(baseline.index, start, 3), "observed"] = np.nan

    return baseline, reporting


def zeros(baseline, reporting, seed=0):
    """A fortnight of zero baseline readings, which electricity data counts as missing."""
    baseline = baseline.copy()
    reporting = reporting.copy()
    start = _random_block_start(baseline.index, 14, seed)
    baseline.loc[_block_mask(baseline.index, start, 14), "observed"] = 0.0

    return baseline, reporting


def negative_export(baseline, reporting, seed=0):
    """A week of negative baseline readings, as net export does; hits the negative rule."""
    baseline = baseline.copy()
    reporting = reporting.copy()
    start = _random_block_start(baseline.index, 7, seed)
    mask = _block_mask(baseline.index, start, 7)
    baseline.loc[mask, "observed"] = -baseline.loc[mask, "observed"].abs()

    return baseline, reporting


def dst_week_gap(baseline, reporting, seed=0):
    """The seven days around spring forward are missing; hits coverage across the shift."""
    baseline = baseline.copy()
    reporting = reporting.copy()
    transition = _spring_forward(baseline.index)
    start = transition.normalize() - pd.DateOffset(days=3)
    baseline.loc[_block_mask(baseline.index, start, 7), "observed"] = np.nan

    return baseline, reporting


# ===== Billing schedule scenarios =====


def offcycle_read(baseline, reporting, seed=0):
    """An extra read splits one period to 20 days; hits the minimum period length."""
    baseline = baseline.copy()
    reporting = reporting.copy()
    reads = _read_dates(baseline)
    rng = np.random.default_rng(seed)
    i = rng.integers(1, len(reads) - 2)
    start, end = reads[i], reads[i + 1]
    split = start + pd.DateOffset(days=20)
    fraction = 20 / ((end - start) / pd.Timedelta(days=1))
    total = baseline.loc[start, "observed"]
    baseline.loc[start, "observed"] = total * fraction
    baseline.loc[split, "observed"] = total * (1 - fraction)

    return baseline, reporting


def bimonthly_reads(baseline, reporting, seed=0):
    """Every second read over half the baseline is merged into the previous period."""
    baseline = baseline.copy()
    reporting = reporting.copy()
    reads = _read_dates(baseline)
    phase = np.random.default_rng(seed).integers(0, 2)

    for i in range(1 + phase, len(reads) // 2, 2):
        kept, dropped = reads[i - 1], reads[i]
        baseline.loc[kept, "observed"] += baseline.loc[dropped, "observed"]
        baseline.loc[dropped, "observed"] = np.nan

    return baseline, reporting


def estimated_read(baseline, reporting, seed=0):
    """One period reads zero and trues up in the next. Electricity data counts the zero read as
    missing; model input carries no estimated-read flag, so the estimated-read rule cannot see it."""
    baseline = baseline.copy()
    reporting = reporting.copy()
    reads = _read_dates(baseline)
    i = np.random.default_rng(seed).integers(1, len(reads) - 2)
    estimated, trued_up = reads[i], reads[i + 1]
    baseline.loc[trued_up, "observed"] += baseline.loc[estimated, "observed"]
    baseline.loc[estimated, "observed"] = 0.0

    return baseline, reporting


# ===== Scenario helpers =====


def _block_mask(index, start, days):
    end = start + pd.DateOffset(days=days)

    return (index >= start) & (index < end)


def _random_block_start(index, days, seed):
    starts = index.normalize().unique()[1:-days]
    i = np.random.default_rng(seed).integers(0, len(starts))

    return starts[i]


def _spring_forward(index):
    offsets = np.array([ts.utcoffset().total_seconds() for ts in index])
    increases = np.flatnonzero(np.diff(offsets) > 0)

    if len(increases) == 0:
        raise ValueError("index does not span a spring-forward transition")

    return index[increases[0] + 1]


def _read_dates(df):
    return df.index[df["observed"].notna()]


# ===== Synthetic data perturbations =====


def create_repeated_values(df, repeat_pct):
    """Reduce uniqueness by repeating values

    Args:
        df: DataFrame with hourly data
        repeat_pct: float - percentage of values to repeat (e.g., 0.75 for 75%)
    """
    df = df.copy()
    n_repeat = int(len(df) * repeat_pct)
    repeated_value = df["observed"].iloc[0]
    df.loc[df.index[:n_repeat], "observed"] = repeated_value

    return df


def create_extreme_values(df, n_extreme, column="observed", seed=0):
    """Add outliers beyond 3x IQR

    Args:
        df: DataFrame with hourly data
        n_extreme: int - number of extreme values to add
        column: str - column to add extremes to
    """
    df = df.copy()
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 3 * iqr

    # Add values beyond upper bound
    extreme_indices = np.random.default_rng(seed).choice(len(df), n_extreme, replace=False)
    df.loc[df.index[extreme_indices], column] = upper_bound * 2

    return df
