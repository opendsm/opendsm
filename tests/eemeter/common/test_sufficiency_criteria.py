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

from functools import partial

import numpy as np
import pandas as pd
import pytest

from opendsm.eemeter.common.data_processor_utilities import DataReport
from opendsm.eemeter.common.data_settings import (
    BillingDataSufficiencySettings,
    DailyDataSufficiencySettings,
    HourlyDataSufficiencySettings,
)
from opendsm.eemeter.common.sufficiency_criteria import SufficiencyCriteria
from opendsm.eemeter.common.sufficiency_rules import (
    RULE_FUNCTIONS,
    _applicable_rules,
    baseline_day_length,
    consecutive_missing_temperature_hours,
    daily_coverage,
    estimated_reads,
    extreme_values,
    high_frequency_meter_coverage,
    monthly_coverage,
    negative_observed_values,
    no_data,
    offcycle_reads,
    run_sufficiency,
    unique_values,
)
from opendsm.eemeter.common.warnings import EEMeterWarning



_SETTINGS_BY_FAMILY = {
    "daily": DailyDataSufficiencySettings,
    "hourly": HourlyDataSufficiencySettings,
    "billing": BillingDataSufficiencySettings,
}

_FAMILIES = list(_SETTINGS_BY_FAMILY)


def _daily_frame(n_days=365, start="2020-01-01", tz="UTC"):
    """A clean daily aggregated frame that passes every sufficiency rule.

    Carries both the value columns (`temperature`, `observed`) and the
    per-period coverage counts (`temperature_not_null`/`temperature_null`) the
    criteria read; full temperature coverage and varied, non-negative observed.
    """
    index = pd.date_range(start, periods=n_days, freq="D", tz=tz)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "temperature": rng.normal(60.0, 15.0, n_days),
            "temperature_not_null": np.full(n_days, 24.0),
            "temperature_null": np.zeros(n_days),
            "observed": rng.normal(30.0, 5.0, n_days),
        },
        index=index,
    )

    return df


def _criteria(
    df,
    is_electricity_data=True,
    is_reporting_data=False,
    family="daily",
    settings=None,
    data_report=None,
    span_start=None,
    span_end=None,
):
    if settings is None:
        settings = _SETTINGS_BY_FAMILY[family]()

    kwargs = {}
    if data_report is not None:
        kwargs["data_report"] = data_report

    sc = SufficiencyCriteria(
        data=df,
        is_electricity_data=is_electricity_data,
        is_reporting_data=is_reporting_data,
        family=family,
        settings=settings,
        span_start=span_start,
        span_end=span_end,
        **kwargs,
    )

    return sc


def _names(entries):
    return {entry.qualified_name for entry in entries}


def _dq_names(sc):
    return {d.qualified_name for d in sc.compliance.disqualification}


def _warning_names(sc):
    return {w.qualified_name for w in sc.compliance.warnings}


def _find(entries, qualified_name):
    """The single entry named `qualified_name` among `entries`."""
    matches = [e for e in entries if e.qualified_name == qualified_name]
    names = [e.qualified_name for e in entries]
    assert len(matches) == 1, f"expected one {qualified_name}, got {names}"

    return matches[0]


# ---------------------------------------------------------------------------
# no_data
# ---------------------------------------------------------------------------

def test_no_data_is_infeasible_when_all_nan():
    """An all-NaN frame records no_data as infeasible."""
    df = _daily_frame()
    df["observed"] = np.nan
    df["temperature"] = np.nan
    sc = _criteria(df)

    entries = no_data(sc)

    record = _find(entries, "eemeter.sufficiency_criteria.no_data")
    assert record.severity == "infeasible"
    assert record.member is None


def test_no_data_passes_on_clean_frame():
    """A populated frame raises no no_data disqualification."""
    sc = _criteria(_daily_frame())

    assert no_data(sc) == []


def test_no_data_passes_when_only_auxiliary_column_is_nan():
    """Usable temperature/observed rows pass no_data even if another column is all-NaN."""
    df = _daily_frame()
    df["temperature_not_null"] = np.nan
    sc = _criteria(df)

    entries = no_data(sc)

    assert "eemeter.sufficiency_criteria.no_data" not in _names(entries)


def test_rules_record_nothing_on_the_context():
    """A rule function is pure: it returns its entries and records none of them."""
    df = _daily_frame()
    df["observed"] = np.nan
    df["temperature"] = np.nan
    sc = _criteria(df)

    entries = no_data(sc)

    assert len(entries) == 1
    assert sc.compliance.all == []


# ---------------------------------------------------------------------------
# baseline_day_length
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_days", [328, 367])
def test_baseline_day_length_disqualifies_out_of_range(n_days):
    """Below the 329-day floor or above the 366-day ceiling disqualifies."""
    sc = _criteria(_daily_frame(n_days=n_days))

    entries = baseline_day_length(sc)

    record = _find(entries, "eemeter.sufficiency_criteria.incorrect_number_of_total_days")
    assert record.severity == "disqualified"
    assert record.member == "data"
    assert record.data == {"n_days_total": n_days}


@pytest.mark.parametrize("n_days", [329, 365, 366])
def test_baseline_day_length_passes_in_range(n_days):
    """Lengths within [329, 366] pass; 329 is the exact lower boundary."""
    sc = _criteria(_daily_frame(n_days=n_days))

    assert baseline_day_length(sc) == []


# ---------------------------------------------------------------------------
# negative_observed_values
# ---------------------------------------------------------------------------

def test_negative_observed_disqualifies_gas():
    """Negative observed values disqualify gas data."""
    df = _daily_frame()
    df.iloc[5, df.columns.get_loc("observed")] = -1.0
    sc = _criteria(df, is_electricity_data=False)

    entries = negative_observed_values(sc)

    record = _find(entries, "eemeter.sufficiency_criteria.negative_observed_values")
    assert record.severity == "disqualified"
    assert record.member == "data"
    assert record.data == {"n_negative_observed_values": 1}


def test_negative_observed_allowed_for_electricity():
    """Electricity may be negative (net metering), so the rule is skipped."""
    df = _daily_frame()
    df.iloc[5, df.columns.get_loc("observed")] = -1.0
    sc = _criteria(df, is_electricity_data=True)

    assert negative_observed_values(sc) == []


def test_negative_observed_passes_clean_gas():
    """Non-negative gas observed raises no disqualification."""
    sc = _criteria(_daily_frame().abs(), is_electricity_data=False)

    assert negative_observed_values(sc) == []


# ---------------------------------------------------------------------------
# unique_values
# ---------------------------------------------------------------------------

def test_unique_values_disqualifies_repeated_observed():
    """A constant observed series fails the 10%-unique floor with the right payload."""
    df = _daily_frame()
    df["observed"] = 5.0
    sc = _criteria(df)

    entries = unique_values(sc, "observed")

    name = "eemeter.sufficiency_criteria.insufficient_unique_observed_values"
    record = _find(entries, name)
    assert record.severity == "disqualified"
    assert record.member == "data"
    payload = record.data
    assert payload["n_unique_values"] == 1
    assert payload["n_total_values"] == 365
    assert payload["unique_percentage"] == 0.3
    assert payload["min_required_percentage"] == 10.0


def test_unique_values_passes_varied_observed():
    """A varied observed series clears the unique-values floor."""
    sc = _criteria(_daily_frame())

    assert unique_values(sc, "observed") == []


# ---------------------------------------------------------------------------
# daily_coverage
# ---------------------------------------------------------------------------

def test_daily_coverage_disqualifies_low_temperature_coverage():
    """Too many days below the per-period temperature coverage fail the rule."""
    df = _daily_frame()
    # zero out temperature coverage on 20% of days (well below the 90% floor)
    df.iloc[:80, df.columns.get_loc("temperature_not_null")] = 0.0
    df.iloc[:80, df.columns.get_loc("temperature_null")] = 24.0
    sc = _criteria(df)

    entries = daily_coverage(sc, "temperature")

    name = "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data"
    record = _find(entries, name)
    assert record.severity == "disqualified"
    assert record.member == "data"
    assert record.data["n_valid_temperature_data_days"] == 285


def test_daily_coverage_disqualifies_low_observed_coverage():
    """Too many null observed days fail the observed coverage rule.

    The nulls are interior, so n_days_total stays at the frame's 365 days; only the
    valid-day fraction drops below 90%.
    """
    df = _daily_frame()
    df.iloc[50:110, df.columns.get_loc("observed")] = np.nan
    sc = _criteria(df)

    entries = daily_coverage(sc, "observed")

    name = "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data"
    record = _find(entries, name)
    assert record.severity == "disqualified"
    assert record.member == "data"
    assert record.data["n_valid_observed_data_days"] == 305


@pytest.mark.parametrize("col", ["temperature", "observed"])
def test_daily_coverage_passes_full_coverage(col):
    """Full coverage passes the daily-coverage rule for each column."""
    sc = _criteria(_daily_frame())

    assert daily_coverage(sc, col) == []


@pytest.mark.parametrize("n_days", [1, 7, 9])
def test_valid_days_count_every_day_of_a_short_window(n_days):
    """A fully covered window of any length counts each of its days, the last one
    included, so a short reporting window passes the 90% daily-coverage rule."""
    sc = _criteria(_daily_frame(n_days), is_reporting_data=True)

    entries = daily_coverage(sc, "joint")

    assert sc.n_valid_days == n_days
    assert sc.n_days_total == n_days
    assert entries == []


def _hourly_local_days_frame(n_days, start="2020-03-05", tz="America/Chicago"):
    """A clean hourly frame over whole local calendar days; the window may cross a DST change."""
    end = pd.Timestamp(start) + pd.Timedelta(days=n_days)
    index = pd.date_range(start, end, freq="h", tz=tz, inclusive="left")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "temperature": rng.normal(60.0, 15.0, len(index)),
            "temperature_not_null": np.ones(len(index)),
            "temperature_null": np.zeros(len(index)),
            "observed": rng.normal(30.0, 5.0, len(index)),
        },
        index=index,
    )

    return df


def test_hourly_rows_count_by_their_share_of_a_day_across_a_dst_change():
    """Each hour counts for one twenty-fourth of a day and the span is read off the wall
    clock: the spring-forward week spans seven days, its 167 hours count as 167/24 valid
    days, and a missing hour costs one share."""
    df = _hourly_local_days_frame(7)
    df.iloc[30, df.columns.get_loc("observed")] = np.nan
    sc = _criteria(df)

    assert sc.n_days_total == 7
    np.testing.assert_allclose(sc.n_valid_temperature_days, 167 / 24)
    np.testing.assert_allclose(sc.n_valid_observed_days, 166 / 24)


def test_off_midnight_hourly_window_spans_its_elapsed_days():
    """A year of hours from 06:00 touches 366 calendar days but spans 365 and holds 365
    valid days, so a start inside a day does not lengthen the window."""
    sc = _criteria(_hourly_local_days_frame(365, start="2021-01-01 06:00", tz="UTC"))

    assert sc.n_days_total == 365
    assert sc.n_valid_days == 365


@pytest.mark.parametrize("start, n_days", [("2020-03-01", 31), ("2020-11-01", 7)])
def test_n_days_total_counts_calendar_days_across_a_dst_change(start, n_days):
    """The span is counted in calendar days of the data's clock, so a window
    holding one DST change is neither a day short nor a day long."""
    sc = _criteria(_daily_frame(n_days, start=start, tz="America/Chicago"))

    assert sc.n_days_total == n_days
    assert sc.n_valid_days == n_days


def test_n_days_total_counts_edge_rows_without_data():
    """The span runs from the frame's first row to its last, so edge rows without data count
    toward the length and against the valid-day fraction."""
    df = _daily_frame(365)
    df.iloc[:10, df.columns.get_loc("observed")] = np.nan
    df.iloc[-30:, df.columns.get_loc("observed")] = np.nan
    sc = _criteria(df)

    entries = daily_coverage(sc, "joint")

    assert sc.n_days_total == 365
    assert sc.n_valid_days == 325
    assert _names(entries) == {"eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data"}


def test_n_days_total_counts_a_given_span_beyond_the_frame():
    """A span wider than the frame counts its days outside the frame as days without data:
    300 days inside a 365-day span pass the length rule and fail daily coverage."""
    df = _daily_frame(300, start="2020-02-01")
    sc = _criteria(
        df,
        span_start=pd.Timestamp("2020-01-01", tz="UTC"),
        span_end=pd.Timestamp("2020-12-30", tz="UTC"),
    )

    assert sc.n_days_total == 365
    assert baseline_day_length(sc) == []
    assert _names(daily_coverage(sc, "joint")) == {
        "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data"
    }


@pytest.mark.parametrize(
    "family, span_end, n_days",
    [
        ("daily", "2020-01-10 23:00", 10),
        ("billing", "2020-01-10 23:00", 10),
        ("hourly", "2020-01-10 23:45", 10),
    ],
)
def test_n_days_total_counts_the_row_holding_a_span_bound(family, span_end, n_days):
    """A span bound inside a row counts that whole row: a day for daily and billing, an hour
    for hourly."""
    sc = _criteria(
        _daily_frame(10),
        family=family,
        span_start=pd.Timestamp("2020-01-01", tz="UTC"),
        span_end=pd.Timestamp(span_end, tz="UTC"),
    )

    assert sc.n_days_total == n_days


def test_n_days_total_is_nan_for_a_frame_with_no_rows():
    """A frame with no rows has a NaT span, whose NaN length the baseline-length rule passes."""
    sc = _criteria(_daily_frame().iloc[:0])

    assert np.isnan(sc.n_days_total)
    assert baseline_day_length(sc) == []


# ---------------------------------------------------------------------------
# monthly_coverage
# ---------------------------------------------------------------------------

def test_monthly_coverage_disqualifies_sparse_month():
    """A month with mostly-missing temperature fails the monthly-coverage rule."""
    df = _daily_frame()
    january = df.index.month == 1
    df.loc[january, "temperature"] = np.nan
    sc = _criteria(df)

    entries = monthly_coverage(sc, "temperature")

    name = "eemeter.sufficiency_criteria.missing_monthly_temperature_data"
    record = _find(entries, name)
    assert record.severity == "disqualified"
    assert record.member == "data"
    assert record.data == {"lowest_monthly_coverage": 0.0}


@pytest.mark.parametrize(
    "rule, col",
    [(daily_coverage, "ghi"), (monthly_coverage, "joint")],
)
def test_column_rule_raises_on_a_column_it_does_not_judge(rule, col):
    """A coverage rule called on a column outside its spec raises rather than judging it."""
    sc = _criteria(_daily_frame(), family="hourly")

    with pytest.raises(ValueError, match=col):
        rule(sc, col)


def test_monthly_coverage_passes_full_year():
    """Every month fully present raises no monthly-coverage disqualification."""
    sc = _criteria(_daily_frame())

    assert monthly_coverage(sc, "temperature") == []


# ---------------------------------------------------------------------------
# extreme_values
# ---------------------------------------------------------------------------

def test_extreme_values_warns_on_outlier():
    """A single value outside 3x IQR raises an extreme-values warning counting one."""
    df = _daily_frame()
    df.iloc[10, df.columns.get_loc("observed")] = 1e6
    sc = _criteria(df)

    entries = extreme_values(sc)

    name = "eemeter.sufficiency_criteria.extreme_values_detected"
    record = _find(entries, name)
    assert record.severity == "advisory"
    assert record.member is None
    assert record.data["n_extreme_values"] == 1


def test_extreme_values_quiet_on_clean_data():
    """A tight observed distribution raises no extreme-values warning."""
    sc = _criteria(_daily_frame())

    assert extreme_values(sc) == []


# ---------------------------------------------------------------------------
# consecutive_missing_temperature_hours
# ---------------------------------------------------------------------------

def _hourly_frame(n_hours=48, start="2020-01-01"):
    """An hourly frame with full temperature coverage."""
    index = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "temperature": rng.normal(60.0, 15.0, n_hours),
            "temperature_not_null": np.ones(n_hours),
            "temperature_null": np.zeros(n_hours),
            "observed": rng.normal(30.0, 5.0, n_hours),
        },
        index=index,
    )

    return df


def test_consecutive_missing_temperature_hours_disqualifies_a_run_over_the_allowance():
    """A temperature gap longer than the allowance is disqualified and reports its length."""
    df = _hourly_frame()
    df.iloc[10:18, df.columns.get_loc("temperature")] = np.nan
    sc = _criteria(df, family="hourly")

    entries = consecutive_missing_temperature_hours(sc)

    name = (
        "eemeter.sufficiency_criteria"
        ".too_many_consecutive_hours_temperature_data_missing"
    )
    record = _find(entries, name)
    assert record.severity == "disqualified"
    assert record.member == "data"
    assert record.data == {"Max_consecutive_hours_missing": 8}


def test_consecutive_missing_temperature_hours_passes_within_the_allowance():
    """A temperature gap within the allowance raises no disqualification."""
    df = _hourly_frame()
    df.iloc[10:13, df.columns.get_loc("temperature")] = np.nan
    sc = _criteria(df, family="hourly")

    assert consecutive_missing_temperature_hours(sc) == []


# ---------------------------------------------------------------------------
# offcycle_reads
# ---------------------------------------------------------------------------

def test_offcycle_reads_disqualifies_monthly_data_frequency():
    """A monthly-frequency report names the monthly qualified name and the off-cycle days."""
    start = pd.Timestamp("2020-02-01", tz="UTC")
    data_report = DataReport(
        data_frequency="billing_monthly",
        offcycle_periods=(start,),
    )
    sc = _criteria(_daily_frame(), family="billing", data_report=data_report)

    entries = offcycle_reads(sc)

    name = "eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data"
    record = _find(entries, name)
    assert record.severity == "disqualified"
    assert record.member == "data"
    assert record.data == [start.isoformat()]


def test_offcycle_reads_disqualifies_bimonthly_data_frequency():
    """A bimonthly-frequency report names the bimonthly qualified name, not the monthly one."""
    start = pd.Timestamp("2020-02-01", tz="UTC")
    data_report = DataReport(
        data_frequency="billing_bimonthly",
        offcycle_periods=(start,),
    )
    sc = _criteria(_daily_frame(), family="billing", data_report=data_report)

    entries = offcycle_reads(sc)

    names = _names(entries)
    assert "eemeter.sufficiency_criteria.offcycle_reads_in_billing_bimonthly_data" in names
    assert "eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data" not in names


def test_offcycle_reads_passes_on_an_empty_report():
    """No off-cycle periods in the data report raises no disqualification."""
    sc = _criteria(_daily_frame(), family="billing", data_report=DataReport())

    assert offcycle_reads(sc) == []


# ---------------------------------------------------------------------------
# high_frequency_meter_coverage
# ---------------------------------------------------------------------------

def test_high_frequency_meter_coverage_warns_on_low_coverage_days():
    """Days the data report names as dropped for low meter coverage raise the advisory."""
    day = pd.Timestamp("2020-03-01", tz="UTC")
    data_report = DataReport(low_coverage_days=(day,))
    sc = _criteria(_daily_frame(), data_report=data_report)

    entries = high_frequency_meter_coverage(sc)

    name = "eemeter.sufficiency_criteria.missing_high_frequency_meter_data"
    record = _find(entries, name)
    assert record.severity == "advisory"
    assert record.member is None
    assert record.data == [day.isoformat()]
    assert record.description == "More than 50% of the high frequency Meter data is missing."


def test_high_frequency_meter_coverage_passes_on_an_empty_report():
    """No low-coverage days in the data report raises no advisory."""
    sc = _criteria(_daily_frame(), data_report=DataReport())

    assert high_frequency_meter_coverage(sc) == []


# ---------------------------------------------------------------------------
# estimated_reads
# ---------------------------------------------------------------------------

def test_estimated_reads_warns_on_merged_reads():
    """A merged estimated read raises the advisory naming the read and its merged value."""
    read = pd.Timestamp("2020-04-01", tz="UTC")
    data_report = DataReport(estimated_merges={read: 123.456})
    sc = _criteria(_daily_frame(), family="billing", data_report=data_report)

    entries = estimated_reads(sc)

    name = "eemeter.sufficiency_criteria.estimated_reads_merged"
    record = _find(entries, name)
    assert record.severity == "advisory"
    assert record.member is None
    assert record.data == {read.isoformat(): 123.5}


def test_estimated_reads_passes_when_nothing_was_merged():
    """No estimated merges in the data report raises no advisory."""
    sc = _criteria(_daily_frame(), family="billing", data_report=DataReport())

    assert estimated_reads(sc) == []


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", _FAMILIES)
def test_applicable_rules_put_every_infeasible_rule_first(family):
    """Every infeasible rule precedes every other rule in the execution order."""
    severities = [spec.severity for _, spec, _ in _applicable_rules(family, "baseline")]

    leading = [i for i, severity in enumerate(severities) if severity == "infeasible"]

    assert leading == list(range(len(leading))), (
        f"{family} baseline order has a non-infeasible rule ahead of an infeasible one: "
        f"{severities}"
    )


@pytest.mark.parametrize(
    "family, expected",
    [
        ("billing", ["no_data", "offcycle_reads", "estimated_reads", "baseline_day_length"]),
        ("daily", ["no_data", "high_frequency_meter_coverage", "baseline_day_length"]),
    ],
)
def test_applicable_rules_run_the_report_fed_rules_between_no_data_and_the_frame_rules(
    family, expected
):
    """The rules fed by the data report run after no_data and ahead of every frame rule."""
    names = [name for name, _, _ in _applicable_rules(family, "baseline")]

    leading = names[: len(expected)]

    assert leading == expected, f"{family} baseline order: {names}"


def _marker_entries(name, context, col=None):
    """One advisory entry named for the rule and column that produced it."""
    entry = EEMeterWarning(
        qualified_name=f"{name}.{col}",
        description="marker",
        severity="advisory",
        data={},
    )

    return [entry]


@pytest.fixture
def rule_markers(monkeypatch):
    """Replace every rule function with one that records a `<rule>.<column>` entry."""
    for name in RULE_FUNCTIONS:
        monkeypatch.setitem(RULE_FUNCTIONS, name, partial(_marker_entries, name))


def _recorded_names(sc):
    names = [entry.qualified_name for entry in sc.compliance.all]

    return names


@pytest.mark.parametrize(
    "key, present",
    [
        ("baseline_day_length.None", True),
        ("unique_values.temperature", True),
        ("unique_values.observed", True),
        ("unique_values.joint", False),
        ("monthly_coverage.ghi", False),
        ("monthly_coverage.observed", False),
    ],
    ids=[
        "whole_frame_rule_enabled",
        "temperature_column_enabled",
        "observed_column_enabled",
        "column_the_rule_does_not_judge",
        "column_block_absent",
        "column_toggled_off",
    ],
)
def test_run_sufficiency_runs_column_rules_only_where_they_are_enabled(
    rule_markers, key, present
):
    """The daily baseline run reaches a rule and column only when the rule judges that
    column, the family carries its block, and its toggle is on."""
    sc = _criteria(_daily_frame())

    run_sufficiency(sc, "baseline")

    recorded = _recorded_names(sc)
    assert (key in recorded) is present, f"expected {key} present={present}, ran {recorded}"


def test_run_sufficiency_records_nothing_for_a_clean_frame():
    """A clean frame leaves the compliance object with zero entries."""
    sc = _criteria(_daily_frame())

    run_sufficiency(sc, "baseline")

    assert sc.compliance.all == [], f"unexpected entries {_recorded_names(sc)}"


def test_run_sufficiency_records_what_the_rules_return():
    """Entries returned by a rule land on the context's compliance object."""
    df = _daily_frame(n_days=328)
    sc = _criteria(df)

    run_sufficiency(sc, "baseline")

    assert "eemeter.sufficiency_criteria.incorrect_number_of_total_days" in _dq_names(sc)


def test_run_sufficiency_stops_after_no_data_on_an_empty_frame(rule_markers):
    """An empty frame runs no_data and no other rule."""
    sc = _criteria(_daily_frame(n_days=0))

    run_sufficiency(sc, "baseline")

    recorded = _recorded_names(sc)
    assert recorded == ["no_data.None"], f"rules run on an empty frame: {recorded}"


def test_run_sufficiency_skips_a_rule_toggled_off():
    """A rule switched off for the phase neither runs nor records."""
    df = _daily_frame()
    df.iloc[10, df.columns.get_loc("observed")] = 1e6
    settings = DailyDataSufficiencySettings(rules={"extreme_values": {"baseline": False}})
    sc = _criteria(df, settings=settings)

    run_sufficiency(sc, "baseline")

    assert sc.compliance.warnings == []


def test_run_sufficiency_runs_a_rule_toggled_on():
    """An extreme observed value records the extreme-values advisory when the rule is left on."""
    df = _daily_frame()
    df.iloc[10, df.columns.get_loc("observed")] = 1e6
    sc = _criteria(df)

    run_sufficiency(sc, "baseline")

    assert "eemeter.sufficiency_criteria.extreme_values_detected" in _warning_names(sc)


def test_fresh_instances_have_independent_result_lists():
    """Each SufficiencyCriteria starts empty and does not share its result lists."""
    first = _criteria(_daily_frame())
    second = _criteria(_daily_frame())

    first.compliance.all.append("sentinel")

    assert second.compliance.all == []


# ---------------------------------------------------------------------------
# reporting path
# ---------------------------------------------------------------------------

def _reporting_context(df, family):
    sc = _criteria(df, is_reporting_data=True, family=family)

    return sc


@pytest.mark.parametrize("family", _FAMILIES)
def test_reporting_flags_observed_daily_coverage_when_insufficient(family):
    """Reporting observed below the daily-coverage floor is disqualified once observed is present."""
    df = _daily_frame()
    df.iloc[50:110, df.columns.get_loc("observed")] = np.nan
    sc = _reporting_context(df, family)

    run_sufficiency(sc, "reporting")

    assert "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data" in _dq_names(sc)


@pytest.mark.parametrize("family", ["daily", "hourly"])
def test_reporting_flags_unique_values_when_observed_flat(family):
    """A constant reporting observed series fails the unique-values floor once observed is present."""
    df = _daily_frame()
    df["observed"] = 5.0
    sc = _reporting_context(df, family)

    run_sufficiency(sc, "reporting")

    assert "eemeter.sufficiency_criteria.insufficient_unique_observed_values" in _dq_names(sc)


@pytest.mark.parametrize("family", _FAMILIES)
def test_reporting_skips_observed_rules_for_all_nan_observed(family):
    """An all-NaN (prediction-only) reporting observed column triggers neither observed rule."""
    df = _daily_frame()
    df["observed"] = np.nan
    sc = _reporting_context(df, family)

    run_sufficiency(sc, "reporting")

    dq_names = _dq_names(sc)
    assert "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data" not in dq_names
    assert "eemeter.sufficiency_criteria.insufficient_unique_observed_values" not in dq_names


def _frame_long_with_extreme_observed(n_days=400):
    df = _daily_frame(n_days=n_days)
    df.iloc[10, df.columns.get_loc("observed")] = 1e6

    return df


@pytest.mark.parametrize("family", _FAMILIES)
def test_baseline_still_runs_day_length_and_extreme_values(family):
    """Control: the reporting-path frame fails baseline day-length and extreme-values."""
    sc = _criteria(_frame_long_with_extreme_observed(), family=family)

    run_sufficiency(sc, "baseline")

    assert "eemeter.sufficiency_criteria.incorrect_number_of_total_days" in _dq_names(sc)
    assert "eemeter.sufficiency_criteria.extreme_values_detected" in _warning_names(sc)


@pytest.mark.parametrize("family", _FAMILIES)
def test_reporting_skips_day_length_and_extreme_values(family):
    """A large, extreme-valued reporting frame is not disqualified by baseline-only rules."""
    sc = _reporting_context(_frame_long_with_extreme_observed(), family)

    run_sufficiency(sc, "reporting")

    assert "eemeter.sufficiency_criteria.incorrect_number_of_total_days" not in _dq_names(sc)
    assert "eemeter.sufficiency_criteria.extreme_values_detected" not in _warning_names(sc)
