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

"""The sufficiency rules and the orchestrator that runs them.

A rule is a pure function of the sufficiency context: it reads the frame, the settings and
the data report, mutates nothing, and returns the entries it found. The orchestrator
decides which rules run for a family and phase, and records what they return.
"""

from __future__ import annotations

import pandas as pd

from typing import Callable, Literal, Optional

from opendsm.eemeter.common.sufficiency_criteria import SufficiencyCriteria, _round_sig
from opendsm.eemeter.common.sufficiency_registry import (
    Column,
    Family,
    Phase,
    RULE_SPECS,
    RuleSpec,
    register,
    registered_rules,
)
from opendsm.eemeter.common.warnings import EEMeterWarning



# These rules record judgments the data path made before sufficiency runs, so their entries
# precede the rules that judge the frame.
_DATA_REPORT_RULES = (
    "offcycle_reads",
    "high_frequency_meter_coverage",
    "high_frequency_observed_coverage",
    "estimated_reads",
)

_DECLARATION_ORDER = {name: order for order, name in enumerate(RULE_SPECS)}


@register("no_data")
def no_data(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    """Report the infeasible entry when the frame cannot support a fit at all.

    A baseline needs at least two periods carrying both a temperature and an observed
    value; a reporting period needs at least one temperature.
    """

    if context.is_reporting_data:
        min_usable = 1
        usable = context.data.temperature.notna()
    elif "observed" in context.data.columns:
        min_usable = 2
        usable = context.data.temperature.notna() & context.data.observed.notna()
    else:
        min_usable = 2
        usable = pd.Series(False, index=context.data.index)

    if int(usable.sum()) >= min_usable:
        return []

    entry = EEMeterWarning(
        qualified_name="eemeter.sufficiency_criteria.no_data",
        description="No data available.",
        severity="infeasible",
        data={},
    )

    return [entry]


@register("baseline_day_length")
def baseline_day_length(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    min_length = context.settings.min_baseline_length
    max_length = context.settings.max_baseline_length

    # a frame with no non-null row spans NaN days, which the no_data rule owns
    # rather than this one, so compare outward and let a NaN span pass
    n_days_total = context.n_days_total
    out_of_range = n_days_total < min_length or n_days_total > max_length
    if not out_of_range:
        return []

    entry = EEMeterWarning(
        qualified_name="eemeter.sufficiency_criteria.incorrect_number_of_total_days",
        description=(
            f"Baseline length is not within the expected range of {min_length}-{max_length} days."
        ),
        severity="disqualified",
        member="data",
        data={"n_days_total": context.n_days_total},
    )

    return [entry]


@register("negative_observed_values")
def negative_observed_values(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    # electricity goes negative under net export
    if context.is_electricity_data:
        return []

    n_negative = int((context.data.observed < 0).sum())
    if n_negative == 0:
        return []

    # CalTrack 2.3.5
    entry = EEMeterWarning(
        qualified_name="eemeter.sufficiency_criteria.negative_observed_values",
        description="Found negative Observed values",
        severity="disqualified",
        member="data",
        data={"n_negative_observed_values": n_negative},
    )

    return [entry]


@register("unique_values")
def unique_values(context: SufficiencyCriteria, col: Column) -> list[EEMeterWarning]:
    min_pct_unique = context._col_settings(col).min_pct_unique_values

    values = context.data[col].dropna()

    if values.empty:
        return []

    n_total = len(values)
    n_unique = values.nunique()
    unique_percentage = n_unique / n_total

    if unique_percentage >= min_pct_unique:
        return []

    name = context._col_display_name(col)
    entry = EEMeterWarning(
        qualified_name=f"eemeter.sufficiency_criteria.insufficient_unique_{col}_values",
        description=(
            f"Less than {min_pct_unique*100:.0f}% of {name} values are unique. "
            f"Data contains too many repeated values."
        ),
        severity="disqualified",
        member="data",
        data={
            "n_unique_values": n_unique,
            "n_total_values": n_total,
            "unique_percentage": round(unique_percentage*100, 1),
            "min_required_percentage": round(min_pct_unique*100, 1),
        },
    )

    return [entry]


@register("daily_coverage")
def daily_coverage(context: SufficiencyCriteria, col: Column) -> list[EEMeterWarning]:
    n_days_total = float(context.n_days_total)
    name = context._col_display_name(col)

    if col == "temperature":
        valid_days = context.n_valid_temperature_days
    elif col == "ghi":
        raise NotImplementedError("GHI valid days percentage check not implemented yet")
    elif col == "observed":
        valid_days = context.n_valid_observed_days
    elif col == "joint":
        valid_days = context.n_valid_days
    else:
        raise ValueError(f"unknown sufficiency column: {col}")

    min_pct = context._col_settings(col).min_pct_daily_coverage

    if n_days_total > 0:
        valid_pct = valid_days / n_days_total
    else:
        valid_pct = 0

    if valid_pct >= min_pct:
        return []

    entry = EEMeterWarning(
        qualified_name=(
            "eemeter.sufficiency_criteria"
            f".too_many_days_with_missing_{col}_data"
        ),
        description=f"Too many days in data have missing {name} data.",
        severity="disqualified",
        member="data",
        data={
            f"n_valid_{col}_data_days": valid_days,
            "n_days_total": n_days_total,
        },
    )

    return [entry]


@register("monthly_coverage")
def monthly_coverage(context: SufficiencyCriteria, col: Column) -> list[EEMeterWarning]:
    if col == "joint":
        raise NotImplementedError("Joint monthly coverage check not implemented yet")

    name = context._col_display_name(col)
    min_pct = context._col_settings(col).min_pct_monthly_coverage

    non_null_pct_per_month = (
        context.data[col]
        .groupby(context.data.index.month)
        .apply(lambda x: x.notna().mean())
    )

    if not (non_null_pct_per_month < min_pct).any():
        return []

    entry = EEMeterWarning(
        qualified_name=f"eemeter.sufficiency_criteria.missing_monthly_{col}_data",
        description=(
            f"More than {(1-min_pct)*100:.0f}% of the monthly {name} data is missing."
        ),
        severity="disqualified",
        member="data",
        data={
            "lowest_monthly_coverage": _round_sig(non_null_pct_per_month.min()),
        },
    )

    return [entry]


@register("extreme_values")
def extreme_values(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    observed = context.data.observed.dropna()
    if observed.empty:
        return []

    lower_quantile = observed.quantile(0.25)
    upper_quantile = observed.quantile(0.75)
    iqr = upper_quantile - lower_quantile
    lower_bound = lower_quantile - (3 * iqr)
    upper_bound = upper_quantile + (3 * iqr)
    n_extreme_values = int(((observed < lower_bound) | (observed > upper_bound)).sum())

    if n_extreme_values == 0:
        return []

    # Inspired by CalTRACK 2.3.6
    entry = EEMeterWarning(
        qualified_name="eemeter.sufficiency_criteria.extreme_values_detected",
        description="Extreme values (outside 3x IQR) must be flagged for manual review.",
        severity="advisory",
        data={
            "n_extreme_values": n_extreme_values,
            "median": _round_sig(observed.median()),
            "upper_quantile": _round_sig(upper_quantile),
            "lower_quantile": _round_sig(lower_quantile),
            "lower_bound": _round_sig(lower_bound),
            "upper_bound": _round_sig(upper_bound),
            "min_value": _round_sig(observed.min()),
            "max_value": _round_sig(observed.max()),
        },
    )

    return [entry]


@register("boundary_gap")
def boundary_gap(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    """Report data reaching outside the requested window, at either boundary."""

    entries = []
    for gap_type in ("start", "end"):
        entry = _boundary_gap_entry(context, gap_type)

        if entry is not None:
            entries.append(entry)

    return entries


def _boundary_gap_entry(
    context: SufficiencyCriteria, gap_type: Literal["start", "end"]
) -> Optional[EEMeterWarning]:
    if gap_type == "start":
        user_boundary = context.settings.requested_start
    else:
        user_boundary = context.settings.requested_end

    if user_boundary is None:
        return None

    non_null_index = context.data.dropna().index
    if gap_type == "start":
        data_boundary = non_null_index.min()
        gap = (data_boundary - user_boundary).days
    else:
        data_boundary = non_null_index.max()
        gap = (user_boundary - data_boundary).days

    if gap >= 0:
        return None

    # CalTRACK 2.2.4
    if gap_type == "start":
        err = "before"
    else:
        err = "after"

    entry = EEMeterWarning(
        qualified_name=(
            "eemeter.sufficiency_criteria"
            f".extra_data_{err}_requested_{gap_type}_date"
        ),
        description=f"Extra data found {err} requested {gap_type} date.",
        severity="disqualified",
        member="data",
        data={
            f"requested_{gap_type}": user_boundary.isoformat(),
            f"data_{gap_type}": data_boundary.isoformat(),
        },
    )

    return entry


@register("consecutive_missing_temperature_hours")
def consecutive_missing_temperature_hours(
    context: SufficiencyCriteria,
) -> list[EEMeterWarning]:
    # TODO : Check implementation wrt Caltrack 2.2.4.1
    hourly_data = context.data["temperature"].resample("h").first()
    mask = hourly_data.isna()
    grouped = mask.groupby((mask != mask.shift()).cumsum())
    max_consecutive_nans = grouped.sum().max()
    allowed_consecutive_nans = context.settings.temperature.max_consecutive_hours_missing

    if max_consecutive_nans <= allowed_consecutive_nans:
        return []

    entry = EEMeterWarning(
        qualified_name="eemeter.sufficiency_criteria.too_many_consecutive_hours_temperature_data_missing",
        description=(
            f"More than {allowed_consecutive_nans} hours of consecutive hourly Temperature data is missing."
        ),
        severity="disqualified",
        member="data",
        data={"Max_consecutive_hours_missing": int(max_consecutive_nans)},
    )

    return [entry]


@register("high_frequency_observed_coverage")
def high_frequency_observed_coverage(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    """Report hourly days whose observed coverage fell below the settings-declared floor.

    The rows themselves are already dropped by the cleaning path; this rule only reports
    what `context.data_report.low_coverage_days` names.
    """

    low_coverage_days = context.data_report.low_coverage_days
    if not low_coverage_days:
        return []

    min_pct = context.settings.observed.min_pct_hourly_coverage

    # CalTRACK 2.2.2.1
    entry = EEMeterWarning(
        qualified_name="eemeter.sufficiency_criteria.missing_high_frequency_observed_data",
        description=(
            f"More than {(1-min_pct)*100:.0f}% of the high frequency Observed data is missing."
        ),
        severity="advisory",
        data=[timestamp.isoformat() for timestamp in low_coverage_days],
    )

    return [entry]


@register("high_frequency_meter_coverage")
def high_frequency_meter_coverage(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    """Report daily or billing days whose meter coverage fell below the settings-declared floor."""

    low_coverage_days = context.data_report.low_coverage_days
    if not low_coverage_days:
        return []

    min_pct = context.settings.observed.min_pct_hourly_coverage

    entry = EEMeterWarning(
        qualified_name="eemeter.sufficiency_criteria.missing_high_frequency_meter_data",
        description=(
            f"More than {(1-min_pct)*100:.0f}% of the high frequency Meter data is missing."
        ),
        severity="advisory",
        data=[timestamp.isoformat() for timestamp in low_coverage_days],
    )

    return [entry]


@register("offcycle_reads")
def offcycle_reads(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    """Report billing periods whose length fell outside the settings-declared bounds."""

    offcycle_periods = context.data_report.offcycle_periods
    if not offcycle_periods:
        return []

    data_frequency = context.data_report.data_frequency
    min_days = context.settings.min_days_in_period
    if data_frequency == "billing_monthly":
        period_type = "monthly"
        max_days = context.settings.max_days_in_monthly_period
    elif data_frequency == "billing_bimonthly":
        period_type = "bimonthly"
        max_days = context.settings.max_days_in_bimonthly_period
    else:
        return []

    # CalTRACK 2.2.3.4, 2.2.3.5
    entry = EEMeterWarning(
        qualified_name=f"eemeter.sufficiency_criteria.offcycle_reads_in_billing_{period_type}_data",
        description=(
            f"Off-cycle reads found in billing {period_type} data having a duration "
            f"less than {min_days} days or greater than {max_days} days."
        ),
        severity="disqualified",
        member="data",
        data=[timestamp.isoformat() for timestamp in offcycle_periods],
    )

    return [entry]


@register("estimated_reads")
def estimated_reads(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    """Report billing reads that absorbed a preceding estimated read."""

    estimated_merges = context.data_report.estimated_merges
    if not estimated_merges:
        return []

    # CalTRACK 2.2.3.1
    entry = EEMeterWarning(
        qualified_name="eemeter.sufficiency_criteria.estimated_reads_merged",
        description=(
            f"{len(estimated_merges)} billing read(s) absorbed a preceding estimated read."
        ),
        severity="advisory",
        data={
            timestamp.isoformat(): _round_sig(value)
            for timestamp, value in estimated_merges.items()
        },
    )

    return [entry]


def _rule_order(name: str, spec: RuleSpec) -> tuple[int, int]:
    """Sort key placing infeasible rules first and report-fed rules next."""

    if spec.severity == "infeasible":
        group, position = 0, _DECLARATION_ORDER[name]
    elif name in _DATA_REPORT_RULES:
        group, position = 1, _DATA_REPORT_RULES.index(name)
    else:
        group, position = 2, _DECLARATION_ORDER[name]

    return group, position


def _applicable_rules(family: Family, phase: Phase) -> list[tuple[str, RuleSpec, Callable]]:
    """The registered rules that may run for a family and phase, in execution order."""

    rules = [
        (name, spec, function)
        for name, (spec, function) in registered_rules().items()
        if family in spec.families and phase in spec.phases
    ]

    return sorted(rules, key=lambda rule: _rule_order(rule[0], rule[1]))


def _toggled_on(
    context: SufficiencyCriteria, name: str, spec: RuleSpec, phase: Phase, col: Optional[Column]
) -> bool:
    if not spec.toggleable:
        return True

    if col is None:
        block = context.settings.rules
    else:
        block = context._col_settings(col)

    return getattr(getattr(block, name), phase)


def _column_applies(context: SufficiencyCriteria, col: Column) -> bool:
    """Whether a column-scoped rule has both a settings block and usable data for a column."""

    if context._col_settings(col) is None:
        return False

    return not context._should_skip_col(col)


def run_sufficiency(
    context: SufficiencyCriteria, phase: Phase
) -> dict[tuple[str, Optional[Column]], int]:
    """Run every enabled sufficiency rule and record what it found on the context.

    Returns the run report: (rule name, column) -> number of entries the rule produced,
    with a `None` column for the rules that judge the whole frame. Entries are recorded,
    never emitted; the caller that owns the compliance object emits them.
    """

    report = {}
    for name, spec, function in _applicable_rules(context.family, phase):
        if context.data.empty and spec.severity != "infeasible":
            break

        if not spec.columns:
            if not _toggled_on(context, name, spec, phase, None):
                continue

            report[(name, None)] = _record(context, function(context))
            continue

        for col in spec.columns:
            if not _column_applies(context, col):
                continue

            if not _toggled_on(context, name, spec, phase, col):
                continue

            report[(name, col)] = _record(context, function(context, col))

    return report


def _record(context: SufficiencyCriteria, entries: list[EEMeterWarning]) -> int:
    for entry in entries:
        context.compliance.record(entry)

    return len(entries)
