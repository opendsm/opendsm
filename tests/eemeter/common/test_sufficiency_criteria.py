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

from opendsm.eemeter.common.data_processor_utilities import DataReport
from opendsm.eemeter.common.data_settings import (
    BillingDataSufficiencySettings,
    DailyDataSufficiencySettings,
    HourlyDataSufficiencySettings,
)
from opendsm.eemeter.common.sufficiency_criteria import SufficiencyCriteria
from opendsm.eemeter.common.sufficiency_registry import rule_spec
from opendsm.eemeter.common.sufficiency_rules import (
    _DATA_REPORT_RULES,
    _applicable_rules,
    baseline_day_length,
    boundary_gap,
    consecutive_missing_temperature_hours,
    daily_coverage,
    estimated_reads,
    extreme_values,
    high_frequency_meter_coverage,
    high_frequency_observed_coverage,
    monthly_coverage,
    negative_observed_values,
    no_data,
    offcycle_reads,
    run_sufficiency,
    unique_values,
)



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
        **kwargs,
    )

    return sc


def _names(entries):
    return {entry.qualified_name for entry in entries}


def _dq_names(sc):
    return {d.qualified_name for d in sc.compliance.disqualification}


def _warning_names(sc):
    return {w.qualified_name for w in sc.compliance.warnings}


def _find(records, qualified_name):
    return next(r for r in records if r.qualified_name == qualified_name)


# ---------------------------------------------------------------------------
# no_data
# ---------------------------------------------------------------------------

def test_no_data_disqualifies_when_all_nan():
    """An all-NaN frame is disqualified as no_data."""
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
# baseline_day_length  (min 329, max 366 for daily)
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


def test_baseline_day_length_never_runs_in_the_reporting_phase():
    """Reporting data is exempt from the baseline-length rule, by its declared phases."""

    assert rule_spec("baseline_day_length").phases == ("baseline",)


# ---------------------------------------------------------------------------
# negative_observed_values  (gas only)
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
# unique_values  (observed needs >= 10% unique)
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
# daily_coverage  (>= 90% daily coverage)
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

    The nulls are interior (endpoints kept) so the requested window — and thus
    n_days_total — is unchanged; only the valid-day fraction drops below 90%.
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


def test_requested_window_gaps_count_calendar_days_across_a_dst_change():
    """The gaps between the requested bounds and the data are read off the data's own
    clock, so a DST change inside a gap does not lose a day."""
    settings = DailyDataSufficiencySettings(
        requested_start=pd.Timestamp("2020-03-05", tz="America/Chicago"),
        requested_end=pd.Timestamp("2020-11-05", tz="America/Chicago"),
    )
    sc = _criteria(_daily_frame(20, start="2020-03-10", tz="America/Chicago"), settings=settings)

    assert sc.n_days_total == (pd.Timestamp("2020-11-05") - pd.Timestamp("2020-03-05")).days + 1


@pytest.mark.parametrize("start, n_days", [("2020-03-01", 31), ("2020-11-01", 7)])
def test_n_days_total_counts_calendar_days_across_a_dst_change(start, n_days):
    """The span is counted in calendar days of the data's clock, so a window
    holding one DST change is neither a day short nor a day long."""
    sc = _criteria(_daily_frame(n_days, start=start, tz="America/Chicago"))

    assert sc.n_days_total == n_days
    assert sc.n_valid_days == n_days


# ---------------------------------------------------------------------------
# monthly_coverage  (>= 90% of each month present)
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


def test_monthly_coverage_passes_full_year():
    """Every month fully present raises no monthly-coverage disqualification."""
    sc = _criteria(_daily_frame())

    assert monthly_coverage(sc, "temperature") == []


# ---------------------------------------------------------------------------
# extreme_values  (warning, outside 3x IQR)
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


def test_extreme_values_never_runs_in_the_reporting_phase():
    """Reporting data is exempt from the extreme-values rule, by its declared phases."""

    assert rule_spec("extreme_values").phases == ("baseline",)


# ---------------------------------------------------------------------------
# boundary_gap  (extra data beyond the requested window)
# ---------------------------------------------------------------------------

def test_boundary_gap_disqualifies_extra_data_before_start():
    """Data extending before the requested start date is flagged."""
    df = _daily_frame()
    requested_start = df.index.min() + pd.Timedelta(days=10)
    settings = DailyDataSufficiencySettings(requested_start=requested_start)
    sc = _criteria(df, settings=settings)

    entries = boundary_gap(sc)

    name = "eemeter.sufficiency_criteria.extra_data_before_requested_start_date"
    record = _find(entries, name)
    assert record.severity == "disqualified"
    assert record.member == "data"


def test_boundary_gap_flags_both_boundaries_in_one_run():
    """One call covers the start and the end of the requested window."""
    df = _daily_frame()
    settings = DailyDataSufficiencySettings(
        requested_start=df.index.min() + pd.Timedelta(days=10),
        requested_end=df.index.max() - pd.Timedelta(days=10),
    )
    sc = _criteria(df, settings=settings)

    entries = boundary_gap(sc)

    assert _names(entries) == {
        "eemeter.sufficiency_criteria.extra_data_before_requested_start_date",
        "eemeter.sufficiency_criteria.extra_data_after_requested_end_date",
    }


def test_boundary_gap_passes_when_aligned():
    """No disqualification when the data starts at the requested start date."""
    df = _daily_frame()
    settings = DailyDataSufficiencySettings(requested_start=df.index.min())
    sc = _criteria(df, settings=settings)

    assert boundary_gap(sc) == []


# ---------------------------------------------------------------------------
# consecutive_missing_temperature_hours  (hourly baseline, disabled by default)
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


def test_consecutive_missing_temperature_hours_is_disabled_by_default():
    """The rule is registered for the baseline phase but switched off by default."""

    assert rule_spec("consecutive_missing_temperature_hours").toggleable

    settings = HourlyDataSufficiencySettings()
    toggle = settings.rules.consecutive_missing_temperature_hours
    assert toggle.baseline is False
    assert toggle.reporting is False


# ---------------------------------------------------------------------------
# offcycle_reads  (billing periods outside the settings-declared duration bounds)
# ---------------------------------------------------------------------------

def test_offcycle_reads_disqualifies_monthly_data_frequency():
    """A monthly-frequency report names the monthly qualified name and the off-cycle days."""
    start = pd.Timestamp("2020-02-01", tz="UTC")
    data_report = DataReport(
        data_frequency="billing_monthly",
        offcycle_periods={start: 10.0},
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
        offcycle_periods={start: 10.0},
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
# high_frequency_meter_coverage  (daily/billing, advisory)
# ---------------------------------------------------------------------------

def test_high_frequency_meter_coverage_warns_on_low_coverage_days():
    """Days the data report names as dropped for low meter coverage raise the advisory."""
    day = pd.Timestamp("2020-03-01", tz="UTC")
    data_report = DataReport(low_coverage_days={day: 0.25})
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
# estimated_reads  (billing, advisory, estimated_reads_merged qualified name)
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
# high_frequency_observed_coverage  (hourly baseline, advisory, disabled by default)
# ---------------------------------------------------------------------------

def test_high_frequency_observed_coverage_warns_on_low_coverage_days():
    """Days the data report names as dropped for low observed coverage raise the advisory."""
    hour = pd.Timestamp("2020-05-01T03:00:00", tz="UTC")
    data_report = DataReport(low_coverage_days={hour: 0.1})
    sc = _criteria(_daily_frame(), family="hourly", data_report=data_report)

    entries = high_frequency_observed_coverage(sc)

    name = "eemeter.sufficiency_criteria.missing_high_frequency_observed_data"
    record = _find(entries, name)
    assert record.severity == "advisory"
    assert record.member is None
    assert record.data == [hour.isoformat()]
    assert record.description == "More than 50% of the high frequency Observed data is missing."


def test_high_frequency_observed_coverage_passes_on_an_empty_report():
    """No low-coverage rows in the data report raises no advisory."""
    sc = _criteria(_daily_frame(), family="hourly", data_report=DataReport())

    assert high_frequency_observed_coverage(sc) == []


def test_high_frequency_observed_coverage_is_disabled_by_default():
    """The rule is registered but switched off for both phases by default."""

    assert rule_spec("high_frequency_observed_coverage").toggleable

    settings = HourlyDataSufficiencySettings()
    toggle = settings.rules.high_frequency_observed_coverage
    assert toggle.baseline is False
    assert toggle.reporting is False


# ---------------------------------------------------------------------------
# the orchestrator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", _FAMILIES)
def test_applicable_rules_put_every_infeasible_rule_first(family):
    """run_sufficiency stops at the first non-infeasible rule on an empty frame."""
    severities = [spec.severity for _, spec, _ in _applicable_rules(family, "baseline")]

    leading = [i for i, severity in enumerate(severities) if severity == "infeasible"]

    assert leading == list(range(len(leading))), (
        f"{family} baseline order has a non-infeasible rule ahead of an infeasible one: "
        f"{severities}"
    )


@pytest.mark.parametrize(
    "family, expected",
    [
        ("billing", ["offcycle_reads", "high_frequency_meter_coverage", "estimated_reads"]),
        ("hourly", ["high_frequency_observed_coverage"]),
    ],
)
def test_applicable_rules_keep_the_report_fed_rules_in_declared_order(family, expected):
    """The report-fed rules run in _DATA_REPORT_RULES order, ahead of the frame rules."""
    names = [name for name, _, _ in _applicable_rules(family, "baseline")]

    report_fed = [name for name in names if name in _DATA_REPORT_RULES]

    assert report_fed == expected, f"{family} baseline order: {names}"
    assert names.index(report_fed[-1]) < names.index("baseline_day_length"), (
        f"a report-fed rule runs after a frame rule: {names}"
    )


def test_run_sufficiency_runs_column_rules_only_where_they_are_enabled():
    """The daily baseline run covers the enabled columns and skips the disabled ones."""
    sc = _criteria(_daily_frame())

    report = run_sufficiency(sc, "baseline")

    assert ("baseline_day_length", None) in report
    assert ("unique_values", "temperature") in report
    assert ("unique_values", "observed") in report
    # joint carries no unique-value threshold, and daily has no ghi block at all
    assert ("unique_values", "joint") not in report
    assert ("monthly_coverage", "ghi") not in report
    # daily never judges observed monthly coverage
    assert ("monthly_coverage", "observed") not in report


def test_run_sufficiency_reports_zero_entries_for_a_clean_frame():
    """A clean frame runs the rules and every one of them reports nothing."""
    sc = _criteria(_daily_frame())

    report = run_sufficiency(sc, "baseline")

    assert set(report.values()) == {0}
    assert sc.compliance.all == []


def test_run_sufficiency_records_what_the_rules_return():
    """Entries returned by a rule land on the context's compliance object."""
    df = _daily_frame(n_days=328)
    sc = _criteria(df)

    report = run_sufficiency(sc, "baseline")

    assert report[("baseline_day_length", None)] == 1
    assert "eemeter.sufficiency_criteria.incorrect_number_of_total_days" in _dq_names(sc)


def test_run_sufficiency_stops_after_no_data_on_an_empty_frame():
    """An empty frame reports no_data and nothing else."""
    sc = _criteria(_daily_frame(n_days=0))

    report = run_sufficiency(sc, "baseline")

    assert report == {("no_data", None): 1}


def test_run_sufficiency_skips_a_rule_toggled_off():
    """A rule switched off for the phase neither runs nor records."""
    df = _daily_frame()
    df.iloc[10, df.columns.get_loc("observed")] = 1e6
    settings = DailyDataSufficiencySettings(rules={"extreme_values": {"baseline": False}})
    sc = _criteria(df, settings=settings)

    report = run_sufficiency(sc, "baseline")

    assert ("extreme_values", None) not in report
    assert sc.compliance.warnings == []


def test_run_sufficiency_runs_a_rule_toggled_on():
    """The same frame with the rule left on records the advisory."""
    df = _daily_frame()
    df.iloc[10, df.columns.get_loc("observed")] = 1e6
    sc = _criteria(df)

    report = run_sufficiency(sc, "baseline")

    assert report[("extreme_values", None)] == 1
    assert "eemeter.sufficiency_criteria.extreme_values_detected" in _warning_names(sc)


def test_fresh_instances_have_independent_result_lists():
    """Each SufficiencyCriteria starts empty and does not share its result lists."""
    first = _criteria(_daily_frame())
    second = _criteria(_daily_frame())

    first.compliance.all.append("sentinel")

    assert second.compliance.disqualification == []
    assert second.compliance.warnings == []


# ---------------------------------------------------------------------------
# reporting path: observed rules run when observed is present
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

    report = run_sufficiency(sc, "reporting")

    assert ("daily_coverage", "observed") not in report
    assert ("unique_values", "observed") not in report

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
