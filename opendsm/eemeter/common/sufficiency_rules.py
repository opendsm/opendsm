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

from typing import Callable, Optional

from opendsm.eemeter.common.data_processor_utilities import billing_period_bounds
from opendsm.eemeter.common.sufficiency_criteria import SufficiencyCriteria, _round_sig
from opendsm.eemeter.common.sufficiency_registry import (
    Column,
    Family,
    Phase,
    RULE_SPECS,
    RuleSpec,
)
from opendsm.eemeter.common.warnings import EEMeterWarning



def no_data(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    """Report the infeasible entry when the frame cannot support a fit at all.

    A baseline needs at least two periods carrying both a temperature and an observed
    value; a reporting period needs at least one temperature.
    """

    if context.is_reporting_data:
        min_usable = 1
        usable = context.data.temperature.notna()
    else:
        min_usable = 2
        # An input that collapses to no periods has no observed column at all
        usable = context.data.reindex(columns=["temperature", "observed"]).notna().all(axis=1)

    if int(usable.sum()) >= min_usable:
        return []

    entry = EEMeterWarning(
        qualified_name="eemeter.sufficiency_criteria.no_data",
        description="No data available.",
        severity="infeasible",
        data={},
    )

    return [entry]


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


def daily_coverage(context: SufficiencyCriteria, col: Column) -> list[EEMeterWarning]:
    n_days_total = float(context.n_days_total)
    name = context._col_display_name(col)

    if col == "temperature":
        valid_days = context.n_valid_temperature_days
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


def monthly_coverage(context: SufficiencyCriteria, col: Column) -> list[EEMeterWarning]:
    if col not in RULE_SPECS["monthly_coverage"].columns:
        raise ValueError(f"monthly_coverage does not judge the {col} column")

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


def high_frequency_meter_coverage(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    """Report each day the data report names as blanked for low meter coverage."""

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


def offcycle_reads(context: SufficiencyCriteria) -> list[EEMeterWarning]:
    """Report billing periods whose length fell outside the settings-declared bounds."""

    offcycle_periods = context.data_report.offcycle_periods
    if not offcycle_periods:
        return []

    period_type, min_days, max_days = billing_period_bounds(
        context.settings, context.data_report.data_frequency
    )

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


RULE_FUNCTIONS: dict[str, Callable] = {
    "no_data": no_data,
    "baseline_day_length": baseline_day_length,
    "negative_observed_values": negative_observed_values,
    "extreme_values": extreme_values,
    "consecutive_missing_temperature_hours": consecutive_missing_temperature_hours,
    "high_frequency_meter_coverage": high_frequency_meter_coverage,
    "offcycle_reads": offcycle_reads,
    "estimated_reads": estimated_reads,
    "unique_values": unique_values,
    "daily_coverage": daily_coverage,
    "monthly_coverage": monthly_coverage,
}


def _check_rule_binding(specs: dict[str, RuleSpec], functions: dict[str, Callable]) -> None:
    """Raise ValueError unless every declared rule has exactly one function and vice versa."""

    missing = sorted(set(specs) - set(functions))
    extra = sorted(set(functions) - set(specs))
    if missing or extra:
        raise ValueError(
            f"sufficiency rule table mismatch: specs without a function {missing}, "
            f"functions without a spec {extra}"
        )


_check_rule_binding(RULE_SPECS, RULE_FUNCTIONS)


def _applicable_rules(family: Family, phase: Phase) -> list[tuple[str, RuleSpec, Callable]]:
    """The rules that may run for a family and phase, in the table's execution order."""

    rules = [
        (name, spec, RULE_FUNCTIONS[name])
        for name, spec in RULE_SPECS.items()
        if family in spec.families and phase in spec.phases
    ]

    return rules


def _toggled_on(
    context: SufficiencyCriteria, name: str, spec: RuleSpec, phase: Phase, col: Optional[Column]
) -> bool:
    if spec.severity == "infeasible":
        return True

    if col is None:
        block = context.settings.rules
    else:
        block = context._col_settings(col)

    toggle = getattr(block, name)
    enabled = getattr(toggle, phase)

    return enabled


def _column_applies(context: SufficiencyCriteria, col: Column) -> bool:
    """Whether a column-scoped rule has both a settings block and usable data for a column."""

    if context._col_settings(col) is None:
        return False

    applies = not context._should_skip_col(col)

    return applies


def run_sufficiency(context: SufficiencyCriteria, phase: Phase) -> None:
    """Run every enabled rule that applies to the context's family, phase and columns, and
    record what each finds on `context.compliance`.

    On an empty frame only the infeasible rules run. A column rule skips a column with no
    settings block or no usable data. Entries are recorded, never emitted; the caller that
    owns the compliance object emits them.
    """

    for name, spec, function in _applicable_rules(context.family, phase):
        if context.data.empty and spec.severity != "infeasible":
            break

        if not spec.columns:
            if not _toggled_on(context, name, spec, phase, None):
                continue

            _record(context, function(context))
            continue

        for col in spec.columns:
            if not _column_applies(context, col):
                continue

            if not _toggled_on(context, name, spec, phase, col):
                continue

            _record(context, function(context, col))


def _record(context: SufficiencyCriteria, entries: list[EEMeterWarning]) -> None:
    for entry in entries:
        context.compliance.record(entry)
