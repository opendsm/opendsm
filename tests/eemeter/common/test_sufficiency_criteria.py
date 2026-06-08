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

from opendsm.eemeter.common.data_settings import (
    BillingDataSufficiencySettings,
    DailyDataSufficiencySettings,
)
from opendsm.eemeter.common.sufficiency_criteria import (
    BillingSufficiencyCriteria,
    SufficiencyCriteria,
)



def _daily_frame(n_days=365, start="2020-01-01"):
    """A clean daily aggregated frame that passes every sufficiency rule.

    Carries both the value columns (`temperature`, `observed`) and the
    per-period coverage counts (`temperature_not_null`/`temperature_null`) the
    criteria read; full temperature coverage and varied, non-negative observed.
    """
    index = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
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


def _criteria(df, is_electricity_data=True, is_reporting_data=False, settings=None):
    if settings is None:
        settings = DailyDataSufficiencySettings()
    sc = SufficiencyCriteria(
        data=df,
        is_electricity_data=is_electricity_data,
        is_reporting_data=is_reporting_data,
        settings=settings,
    )
    # Defensive: start each instance with its own empty result lists.
    sc.disqualification = []
    sc.warnings = []

    return sc


def _dq_names(sc):
    return {d.qualified_name for d in sc.disqualification}


def _warning_names(sc):
    return {w.qualified_name for w in sc.warnings}


# ---------------------------------------------------------------------------
# no_data
# ---------------------------------------------------------------------------

def test_no_data_disqualifies_when_all_nan():
    """An all-NaN frame is disqualified as no_data."""
    df = _daily_frame()
    df["observed"] = np.nan
    df["temperature"] = np.nan
    sc = _criteria(df)

    sc._check_no_data()

    assert "eemeter.sufficiency_criteria.no_data" in _dq_names(sc)


def test_no_data_passes_on_clean_frame():
    """A populated frame raises no no_data disqualification."""
    sc = _criteria(_daily_frame())

    sc._check_no_data()

    assert sc.disqualification == []


# ---------------------------------------------------------------------------
# baseline_day_length  (min 329, max 366 for daily)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_days", [328, 367])
def test_baseline_day_length_disqualifies_out_of_range(n_days):
    """Below the 329-day floor or above the 366-day ceiling disqualifies."""
    sc = _criteria(_daily_frame(n_days=n_days))

    sc._check_baseline_day_length()

    assert "eemeter.sufficiency_criteria.incorrect_number_of_total_days" in _dq_names(sc)


@pytest.mark.parametrize("n_days", [329, 365, 366])
def test_baseline_day_length_passes_in_range(n_days):
    """Lengths within [329, 366] pass; 329 is the exact lower boundary."""
    sc = _criteria(_daily_frame(n_days=n_days))

    sc._check_baseline_day_length()

    assert sc.disqualification == []


def test_baseline_day_length_skipped_for_reporting_data():
    """Reporting data is exempt from the baseline-length rule."""
    sc = _criteria(_daily_frame(n_days=10), is_reporting_data=True)

    sc._check_baseline_day_length()

    assert sc.disqualification == []


# ---------------------------------------------------------------------------
# negative_observed_values  (gas only)
# ---------------------------------------------------------------------------

def test_negative_observed_disqualifies_gas():
    """Negative observed values disqualify gas data."""
    df = _daily_frame()
    df.iloc[5, df.columns.get_loc("observed")] = -1.0
    sc = _criteria(df, is_electricity_data=False)

    sc._check_negative_observed_values()

    assert "eemeter.sufficiency_criteria.negative_observed_values" in _dq_names(sc)


def test_negative_observed_allowed_for_electricity():
    """Electricity may be negative (net metering), so the rule is skipped."""
    df = _daily_frame()
    df.iloc[5, df.columns.get_loc("observed")] = -1.0
    sc = _criteria(df, is_electricity_data=True)

    sc._check_negative_observed_values()

    assert sc.disqualification == []


def test_negative_observed_passes_clean_gas():
    """Non-negative gas observed raises no disqualification."""
    sc = _criteria(_daily_frame().abs(), is_electricity_data=False)

    sc._check_negative_observed_values()

    assert sc.disqualification == []


# ---------------------------------------------------------------------------
# unique_values  (observed needs >= 10% unique)
# ---------------------------------------------------------------------------

def test_unique_values_disqualifies_repeated_observed():
    """A near-constant observed series fails the 10%-unique floor."""
    df = _daily_frame()
    df["observed"] = 5.0
    sc = _criteria(df)

    sc._check_unique_values(col="observed")

    assert "eemeter.sufficiency_criteria.insufficient_unique_observed_values" in _dq_names(sc)


def test_unique_values_passes_varied_observed():
    """A varied observed series clears the unique-values floor."""
    sc = _criteria(_daily_frame())

    sc._check_unique_values(col="observed")

    assert sc.disqualification == []


# ---------------------------------------------------------------------------
# valid_days_percentage  (>= 90% daily coverage)
# ---------------------------------------------------------------------------

def test_valid_days_percentage_disqualifies_low_temperature_coverage():
    """Too many days below the per-period temperature coverage fail the rule."""
    df = _daily_frame()
    # zero out temperature coverage on 20% of days (well below the 90% floor)
    df.iloc[:80, df.columns.get_loc("temperature_not_null")] = 0.0
    df.iloc[:80, df.columns.get_loc("temperature_null")] = 24.0
    sc = _criteria(df)

    sc._check_valid_days_percentage(col="temperature")

    assert "eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data" in _dq_names(sc)


def test_valid_days_percentage_disqualifies_low_observed_coverage():
    """Too many null observed days fail the observed coverage rule.

    The nulls are interior (endpoints kept) so the requested window — and thus
    n_days_total — is unchanged; only the valid-day fraction drops below 90%.
    """
    df = _daily_frame()
    df.iloc[50:110, df.columns.get_loc("observed")] = np.nan
    sc = _criteria(df)

    sc._check_valid_days_percentage(col="observed")

    assert "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data" in _dq_names(sc)


@pytest.mark.parametrize("col", ["temperature", "observed"])
def test_valid_days_percentage_passes_full_coverage(col):
    """Full coverage passes the daily-coverage rule for each column."""
    sc = _criteria(_daily_frame())

    sc._check_valid_days_percentage(col=col)

    assert sc.disqualification == []


# ---------------------------------------------------------------------------
# valid_monthly_coverage  (>= 90% of each month present)
# ---------------------------------------------------------------------------

def test_valid_monthly_coverage_disqualifies_sparse_month():
    """A month with mostly-missing temperature fails the monthly-coverage rule."""
    df = _daily_frame()
    january = df.index.month == 1
    df.loc[january, "temperature"] = np.nan
    sc = _criteria(df)

    sc._check_valid_monthly_coverage(col="temperature")

    assert "eemeter.sufficiency_criteria.missing_monthly_temperature_data" in _dq_names(sc)


def test_valid_monthly_coverage_passes_full_year():
    """Every month fully present raises no monthly-coverage disqualification."""
    sc = _criteria(_daily_frame())

    sc._check_valid_monthly_coverage(col="temperature")

    assert sc.disqualification == []


# ---------------------------------------------------------------------------
# extreme_values  (warning, outside 3x IQR)
# ---------------------------------------------------------------------------

def test_extreme_values_warns_on_outlier():
    """An observed value outside 3x IQR raises an extreme-values warning."""
    df = _daily_frame()
    df.iloc[10, df.columns.get_loc("observed")] = 1e6
    sc = _criteria(df)

    sc._check_extreme_values()

    assert "eemeter.sufficiency_criteria.extreme_values_detected" in _warning_names(sc)


def test_extreme_values_quiet_on_clean_data():
    """A tight observed distribution raises no extreme-values warning."""
    sc = _criteria(_daily_frame())

    sc._check_extreme_values()

    assert sc.warnings == []


# ---------------------------------------------------------------------------
# n_days_boundary_gap  (extra data beyond the requested window)
# ---------------------------------------------------------------------------

def test_boundary_gap_disqualifies_extra_data_before_start():
    """Data extending before the requested start date is flagged."""
    df = _daily_frame()
    requested_start = df.index.min() + pd.Timedelta(days=10)
    settings = DailyDataSufficiencySettings(requested_start=requested_start)
    sc = _criteria(df, settings=settings)

    sc._check_n_days_boundary_gap("start")

    assert "eemeter.sufficiency_criteria.extra_data_before_requested_start_date" in _dq_names(sc)


def test_boundary_gap_passes_when_aligned():
    """No disqualification when the data starts at the requested start date."""
    df = _daily_frame()
    settings = DailyDataSufficiencySettings(requested_start=df.index.min())
    sc = _criteria(df, settings=settings)

    sc._check_n_days_boundary_gap("start")

    assert sc.disqualification == []


# ---------------------------------------------------------------------------
# billing off-cycle reads  (period must be within [min_days, max_days])
# ---------------------------------------------------------------------------

def _billing_criteria(df):
    sc = BillingSufficiencyCriteria(
        data=df,
        is_electricity_data=True,
        is_reporting_data=False,
        settings=BillingDataSufficiencySettings(),
    )
    sc.disqualification = []
    sc.warnings = []

    return sc


def test_billing_offcycle_read_disqualifies():
    """A short off-cycle period (below the 25-day floor) is flagged."""
    index = pd.to_datetime(
        ["2020-01-01", "2020-02-01", "2020-03-01", "2020-03-10", "2020-04-10"]
    ).tz_localize("UTC")
    df = pd.DataFrame({"value": [10.0, 12.0, 11.0, 3.0, 9.0]}, index=index)
    sc = _billing_criteria(df)

    sc._check_observed_data_billing_monthly()

    assert "eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data" in _dq_names(sc)


def test_billing_regular_monthly_cadence_passes():
    """A regular monthly cadence has no off-cycle reads."""
    index = pd.date_range("2020-01-01", periods=12, freq="MS", tz="UTC")
    df = pd.DataFrame({"value": np.arange(12.0)}, index=index)
    sc = _billing_criteria(df)

    sc._check_observed_data_billing_monthly()

    assert sc.disqualification == []
