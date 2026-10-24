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

Each scenario takes one frame and returns a perturbed copy; the random draws use a fixed
seed, so every scenario is deterministic. Billing schedule scenarios operate on a monthly
frame, whose daily index carries the period value on the read date and NaN in between.
"""

import numpy as np
import pandas as pd



_SEED = 0


# ===== Frame scenarios =====


def isolated_gaps(df):
    """Isolated missing rows scattered through the frame; hits data coverage."""
    df = df.copy()
    n_slots = len(df) // 2
    n_gaps = max(1, n_slots // 10)
    slots = np.random.default_rng(_SEED).choice(n_slots, n_gaps, replace=False)
    df.iloc[slots * 2, df.columns.get_loc("observed")] = np.nan

    return df


def multi_day_gap(df):
    """Three consecutive missing days; hits the consecutive gap limit."""
    df = df.copy()
    start = _random_block_start(df.index, 3)
    df.loc[_block_mask(df.index, start, 3), "observed"] = np.nan

    return df


def zeros(df):
    """A fortnight of zero readings, which electricity data counts as missing."""
    df = df.copy()
    start = _random_block_start(df.index, 14)
    df.loc[_block_mask(df.index, start, 14), "observed"] = 0.0

    return df


def negative_export(df):
    """A week of negative readings, as net export does; hits the negative rule."""
    df = df.copy()
    start = _random_block_start(df.index, 7)
    mask = _block_mask(df.index, start, 7)
    df.loc[mask, "observed"] = -df.loc[mask, "observed"].abs()

    return df


def dst_week_gap(df):
    """The seven days around spring forward are missing; hits coverage across the shift."""
    df = df.copy()
    transition = _spring_forward(df.index)
    start = transition.normalize() - pd.DateOffset(days=3)
    df.loc[_block_mask(df.index, start, 7), "observed"] = np.nan

    return df


# ===== Frame scenario table =====


FRAME_SCENARIOS = {
    "isolated_gaps": isolated_gaps,
    "multi_day_gap": multi_day_gap,
    "zeros": zeros,
    "negative_export": negative_export,
    "dst_week_gap": dst_week_gap,
}


# ===== Billing schedule scenarios =====


def offcycle_read(df):
    """An extra read splits one period to 20 days; hits the minimum period length."""
    df = df.copy()
    reads = _read_dates(df)
    i = np.random.default_rng(_SEED).integers(1, len(reads) - 2)
    start, end = reads[i], reads[i + 1]
    split = start + pd.DateOffset(days=20)
    fraction = 20 / ((end - start) / pd.Timedelta(days=1))
    total = df.loc[start, "observed"]
    df.loc[start, "observed"] = total * fraction
    df.loc[split, "observed"] = total * (1 - fraction)

    return df


def bimonthly_reads(df):
    """Every second read over half the frame is merged into the previous period."""
    df = df.copy()
    reads = _read_dates(df)
    phase = np.random.default_rng(_SEED).integers(0, 2)

    for i in range(1 + phase, len(reads) // 2, 2):
        kept, dropped = reads[i - 1], reads[i]
        df.loc[kept, "observed"] += df.loc[dropped, "observed"]
        df.loc[dropped, "observed"] = np.nan

    return df


def estimated_read(df):
    """One period reads zero and trues up in the next. Electricity data counts the zero read as
    missing; model input carries no estimated-read flag, so the estimated-read rule cannot see it."""
    df = df.copy()
    reads = _read_dates(df)
    i = np.random.default_rng(_SEED).integers(1, len(reads) - 2)
    estimated, trued_up = reads[i], reads[i + 1]
    df.loc[trued_up, "observed"] += df.loc[estimated, "observed"]
    df.loc[estimated, "observed"] = 0.0

    return df


# ===== Scenario helpers =====


def _block_mask(index, start, days):
    end = start + pd.DateOffset(days=days)

    return (index >= start) & (index < end)


def _random_block_start(index, days):
    starts = index.normalize().unique()[1:-days]
    i = np.random.default_rng(_SEED).integers(0, len(starts))

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

    extreme_indices = np.random.default_rng(seed).choice(len(df), n_extreme, replace=False)
    df.loc[df.index[extreme_indices], column] = upper_bound * 2

    return df
