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

import copy

from functools import partial, reduce
from typing import Callable, NamedTuple, Optional

import numpy as np
import pandas as pd
import pytest

from opendsm.eemeter.common.compliance import Compliance
from opendsm.eemeter.common.data_settings import (
    BillingDataSettings,
    DailyDataSettings,
    HourlyDataSettings,
)
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    NonstandardSettingsError,
)
from opendsm.eemeter.common.sufficiency_registry import RULE_SPECS
from opendsm.eemeter.common.sufficiency_rules import RULE_FUNCTIONS
from opendsm.eemeter.models.billing.data import BillingBaselineData, BillingReportingData
from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.daily.data import DailyBaselineData, DailyReportingData
from opendsm.eemeter.models.daily.model import DailyModel
from opendsm.eemeter.models.hourly.data import HourlyBaselineData, HourlyReportingData
from opendsm.eemeter.models.hourly.model import HourlyModel



_MODEL_CLASSES = {
    "daily": DailyModel,
    "billing": BillingModel,
    "hourly": HourlyModel,
}

_COMSTOCK_FIXTURES = {
    "daily": "comstock_daily",
    "billing": "comstock_monthly",
    "hourly": "comstock_hourly",
}


@pytest.fixture(scope="module")
def baseline_model_hourly(comstock_hourly):
    """The hourly model fit once on the ComStock baseline meter; tests read it and never
    mutate it."""
    df_b, _ = comstock_hourly
    model = HourlyModel().fit(df_b.copy(), is_electricity_data=True, enforce_compliance=False)

    return model


@pytest.fixture(params=list(_MODEL_CLASSES))
def model_name(request):
    return request.param


@pytest.fixture
def fitted_model(model_name, request):
    """The shared fitted model for `model_name`; copy it before mutating."""
    model = request.getfixturevalue(f"baseline_model_{model_name}")

    return model


@pytest.fixture
def comstock_frames(model_name, request):
    """The ComStock baseline and reporting frames at `model_name`'s frequency."""
    frames = request.getfixturevalue(_COMSTOCK_FIXTURES[model_name])

    return frames


def _plain(entries) -> list[dict]:
    """Compliance records as JSON-ready mappings."""
    records = [
        {
            "qualified_name": entry.qualified_name,
            "severity": entry.severity,
            "member": entry.member,
            "data": entry.data,
        }
        for entry in entries
    ]

    return records


def _one_step_past(threshold: float) -> float:
    step = float(np.nextafter(threshold, np.inf))

    return step


# Fit-metric check


def _fit_metric_outcome(model, cvrmse_adj: float, pnrmse_adj: float) -> dict:
    """Run the model's fit-metric check on a copy of `model` whose baseline metrics carry
    these values; `model` is left unchanged."""
    checked = copy.copy(model)
    checked.compliance = Compliance()
    checked.baseline_metrics = copy.copy(model.baseline_metrics)
    # the metrics are cached computed fields, read from the instance dict; pydantic 2.4 drops
    # an attribute assignment to them, so the values go into the dict as a loaded model's do
    checked.baseline_metrics.__dict__.update(cvrmse_adj=cvrmse_adj, pnrmse_adj=pnrmse_adj)
    checked._check_model_fit()
    outcome = {
        "passed": checked.compliance.qualified,
        "records": _plain(checked.compliance.all),
    }

    return outcome


@pytest.mark.slow
def test_fit_metric_check_passes_while_either_metric_is_within_its_threshold(
    fitted_model, snapshot
):
    """The fit-metric check passes when `cvrmse_adj` or `pnrmse_adj` is at most its default
    threshold, and disqualifies the model only when both are one step past."""
    cvrmse_threshold = fitted_model.settings.cvrmse_threshold
    pnrmse_threshold = fitted_model.settings.pnrmse_threshold
    cvrmse_past = _one_step_past(cvrmse_threshold)
    pnrmse_past = _one_step_past(pnrmse_threshold)

    outcomes = {
        "both_at_threshold": _fit_metric_outcome(
            fitted_model, cvrmse_threshold, pnrmse_threshold
        ),
        "cvrmse_past_threshold": _fit_metric_outcome(
            fitted_model, cvrmse_past, pnrmse_threshold
        ),
        "pnrmse_past_threshold": _fit_metric_outcome(
            fitted_model, cvrmse_threshold, pnrmse_past
        ),
        "both_past_threshold": _fit_metric_outcome(fitted_model, cvrmse_past, pnrmse_past),
    }

    assert outcomes == snapshot


# Timezone check


def _prediction_outcome(model, df) -> dict:
    """Predict on `df` with data compliance enforced, returning whether the prediction is
    allowed and the reporting data's compliance records."""
    data = model._reporting_data(df)
    try:
        model._predict_data(data, enforce_compliance=["data"])
    except DataSufficiencyError:
        passed = False
    else:
        passed = True

    outcome = {"passed": passed, "records": _plain(data.compliance.all)}

    return outcome


@pytest.mark.slow
def test_timezone_check_refuses_reporting_data_in_another_timezone(
    fitted_model, comstock_frames, snapshot
):
    """Prediction passes on reporting data in the model's timezone and is refused on the same
    data in another timezone, which carries a timezone-mismatch record. The mismatched frame
    keeps the local wall-clock timestamps so the mismatch is its only disqualification."""
    _, df_r = comstock_frames
    wall_clock_in_utc = df_r.tz_localize(None).tz_localize("UTC")

    outcomes = {
        "matching": _prediction_outcome(fitted_model, df_r.copy()),
        "mismatched": _prediction_outcome(fitted_model, wall_clock_in_utc),
    }

    assert outcomes == snapshot


# Settings-deviation check


def _settings_fit_outcome(model, df) -> dict:
    """Fit `model` on `df` with settings compliance enforced, returning whether the fit is
    allowed and its settings records."""
    try:
        model.fit(df, is_electricity_data=True, enforce_compliance=["settings"])
    except NonstandardSettingsError as error:
        outcome = {"passed": False, "records": _plain(error.entries)}
    else:
        outcome = {
            "passed": True,
            "records": _plain(model.compliance.disqualified_in({"settings"})),
        }

    return outcome


@pytest.mark.slow
def test_settings_deviation_check_refuses_a_changed_developer_setting(
    model_name, comstock_frames, snapshot
):
    """A settings-enforced fit passes with default settings and is refused with a changed
    developer-tier setting, carrying the deviation as a settings record."""
    model_class = _MODEL_CLASSES[model_name]
    df_b, _ = comstock_frames
    changed = {"cvrmse_threshold": 2 * model_class().settings.cvrmse_threshold}

    outcomes = {
        "default": _settings_fit_outcome(model_class(), df_b.copy()),
        "changed": _settings_fit_outcome(model_class(settings=changed), df_b.copy()),
    }

    assert outcomes == snapshot


# Sufficiency thresholds


# no DST, so every day holds 24 hours, and not UTC, so no utc_index advisory is recorded
_START = pd.Timestamp("2019-01-01", tz="America/Phoenix")
# a tenth of 330 days, and of 330 * 24 hours, is whole, so a 90% coverage threshold is hit
# exactly; the length also sits inside the default baseline length range
_N_DAYS = 330
# April has 30 days, so a 90% monthly coverage threshold is hit exactly
_MONTH = 4
# the day a single-day input alters, clear of the frame edges
_DAY = 100

_DATA_SETTINGS = {
    "daily": DailyDataSettings,
    "billing": BillingDataSettings,
    "hourly": HourlyDataSettings,
}

_DATA_CLASSES = {
    ("daily", "baseline"): DailyBaselineData,
    ("daily", "reporting"): DailyReportingData,
    ("billing", "baseline"): BillingBaselineData,
    ("billing", "reporting"): BillingReportingData,
    ("hourly", "baseline"): HourlyBaselineData,
    ("hourly", "reporting"): HourlyReportingData,
}

_UNTHRESHOLDED_RULES = {
    "no_data",
    "negative_observed_values",
    "extreme_values",
    "estimated_reads",
}


def _hourly_frame(n_days: int) -> pd.DataFrame:
    """Hourly temperature and observed values over `n_days` whole days from `_START`, varied
    enough to pass every sufficiency rule."""
    index = pd.date_range(_START, periods=24 * n_days, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "temperature": rng.uniform(20.0, 90.0, len(index)),
            "observed": rng.uniform(1.0, 2.0, len(index)),
        },
        index=index,
    )

    return df


def _billing_frame(period_days: list[int], n_unread_days: int = 0) -> pd.DataFrame:
    """Hourly temperature over `n_unread_days` days without a read followed by the billing
    periods, with one observed read at the start of each period."""
    df = _hourly_frame(n_unread_days + sum(period_days))
    read_days = n_unread_days + np.cumsum([0, *period_days[:-1]])
    reads = _START + pd.to_timedelta(read_days, unit="D")
    df["observed"] = np.nan
    df.loc[reads, "observed"] = np.linspace(300.0, 600.0, len(period_days))

    return df


def _monthly_periods(n_days: int) -> list[int]:
    """Billing periods within a day of each other, about 30 days long, spanning `n_days`."""
    n_periods = round(n_days / 30)
    base, remainder = divmod(n_days, n_periods)
    periods = [base + 1] * remainder + [base] * (n_periods - remainder)

    return periods


def _daily_frame(n_days: int) -> pd.DataFrame:
    """Hourly temperature over `n_days` days, with one observed value at the start of each
    day."""
    df = _hourly_frame(n_days)
    df["observed"] = np.nan
    df.iloc[::24, df.columns.get_loc("observed")] = np.linspace(24.0, 48.0, n_days)

    return df


def _frame(family: str, n_days: int) -> pd.DataFrame:
    """A frame of `n_days` days for `family` that passes every sufficiency rule."""
    if family == "hourly":
        df = _hourly_frame(n_days)
    elif family == "daily":
        df = _daily_frame(n_days)
    elif family == "billing":
        df = _billing_frame(_monthly_periods(n_days))
    else:
        raise ValueError(f"unknown family: {family}")

    return df


def _units(family: str, n_days: int) -> int:
    """The rows a coverage rule counts over `n_days` days: hours for hourly, days otherwise."""
    if family == "hourly":
        n_units = 24 * n_days
    else:
        n_units = n_days

    return n_units


def _spread(n_units: int, count: int, start: int = 0) -> np.ndarray:
    """`count` distinct unit positions spread evenly over [start, n_units)."""
    positions = np.linspace(start, n_units - 1, count).astype(int)

    return positions


def _hours(days, n_hours: int = 24) -> np.ndarray:
    """Row positions of the first `n_hours` hours of each day in `days` of an hourly frame
    from `_START`."""
    hours = (np.asarray(days)[:, None] * 24 + np.arange(n_hours)).ravel()

    return hours


def _observed_rows(family: str, units) -> np.ndarray:
    """Row positions of the observed values at `units`: the hours themselves for hourly
    data, the read at the start of each day for daily data."""
    if family == "hourly":
        rows = np.asarray(units)
    elif family == "daily":
        rows = _hours(units, 1)
    else:
        raise ValueError(f"no observed row per unit for the {family} family")

    return rows


def _blank(df: pd.DataFrame, column: str, rows) -> None:
    df.iloc[np.asarray(rows, dtype=int), df.columns.get_loc(column)] = np.nan


def _short_day_hours(sufficiency) -> int:
    """The fewest missing hours that leave a day short of the temperature period coverage,
    which still leaves the day a mean temperature."""
    n_hours = int((1 - sufficiency.temperature.min_pct_period_coverage) * 24) + 1

    return n_hours


def _baseline_length_frame(family, sufficiency, threshold, step) -> pd.DataFrame:
    """A frame of `threshold + step` days, so a negative step goes below the threshold."""
    df = _frame(family, threshold + step)

    return df


def _daily_coverage_frame(column, family, sufficiency, threshold, step) -> pd.DataFrame:
    """A frame whose `column` days fall short of the daily coverage threshold by `step` units.

    A unit short of temperature is an hour without a value for hourly data and a day short of
    the period coverage otherwise; a unit short of observed is an hour or a day without a
    value, or a leading day before the first billing read. The joint column alternates its
    units between the two, so that neither column alone falls short.
    """
    n_units = _units(family, _N_DAYS)
    positions = _spread(n_units, round((1 - threshold) * n_units) + step)

    if column == "temperature":
        observed_positions, temperature_positions = positions[:0], positions
    elif column == "observed":
        observed_positions, temperature_positions = positions, positions[:0]
    elif column == "joint":
        observed_positions, temperature_positions = positions[0::2], positions[1::2]
    else:
        raise ValueError(f"no daily coverage frame for the {column} column")

    if family == "billing":
        n_unread_days = len(observed_positions)
        df = _billing_frame(_monthly_periods(_N_DAYS - n_unread_days), n_unread_days)
        temperature_positions = _spread(n_units, len(temperature_positions), n_unread_days)
    else:
        df = _frame(family, _N_DAYS)
        _blank(df, "observed", _observed_rows(family, observed_positions))

    if family == "hourly":
        _blank(df, "temperature", temperature_positions)
    else:
        _blank(df, "temperature", _hours(temperature_positions, _short_day_hours(sufficiency)))

    return df


def _monthly_coverage_frame(column, family, sufficiency, threshold, step) -> pd.DataFrame:
    """A frame whose `column` misses enough of `_MONTH` to sit `step` units short of the
    monthly coverage threshold: hours for hourly data, whole days otherwise."""
    df = _frame(family, _N_DAYS)
    if column == "ghi":
        df["ghi"] = np.random.default_rng(1).uniform(0.0, 800.0, len(df))

    if family == "hourly":
        month_units = np.flatnonzero(df.index.month == _MONTH)
    else:
        month_units = np.flatnonzero(df.index[::24].month == _MONTH)

    n_missing = round((1 - threshold) * len(month_units)) + step
    missing = month_units[_spread(len(month_units), n_missing)]
    if family == "hourly":
        _blank(df, column, missing)
    else:
        _blank(df, column, _hours(missing))

    return df


def _unique_values_frame(column, family, sufficiency, threshold, step) -> pd.DataFrame:
    """A frame whose `column` cycles through `step` fewer distinct values than the unique
    values threshold's share of its rows, one level per hour for hourly data and per day
    otherwise."""
    df = _frame(family, _N_DAYS)
    n_units = _units(family, _N_DAYS)
    n_unique = round(threshold * n_units) - step
    levels = np.arange(n_units) % n_unique

    if column == "temperature" and family == "hourly":
        df["temperature"] = 40.0 + levels
    elif column == "temperature":
        df["temperature"] = 40.0 + np.repeat(levels, 24)
    elif column == "observed":
        rows = _observed_rows(family, np.arange(n_units))
        df.iloc[rows, df.columns.get_loc("observed")] = 1.0 + levels / n_unique
    else:
        raise ValueError(f"no unique values frame for the {column} column")

    return df


def _partial_meter_day_frame(family, sufficiency, threshold, step) -> pd.DataFrame:
    """A frame of hourly observed values where day `_DAY` keeps `step` hours more than the
    hourly coverage threshold's share of the day."""
    df = _hourly_frame(_N_DAYS)
    n_read = round(threshold * 24) + step
    _blank(df, "observed", _hours([_DAY])[n_read:])

    return df


def _temperature_gap_frame(family, sufficiency, threshold, step) -> pd.DataFrame:
    """A frame with one run of `threshold + step` missing temperature hours."""
    df = _hourly_frame(_N_DAYS)
    _blank(df, "temperature", 24 * _DAY + np.arange(threshold + step))

    return df


def _billing_periods_frame(other_periods, family, sufficiency, threshold, step):
    """A billing frame whose first period lasts `threshold + step` days."""
    df = _billing_frame([threshold + step, *other_periods])

    return df


class _ThresholdCase(NamedTuple):
    """One settings threshold of a sufficiency rule, applied to one family.

    `frame(family, sufficiency, threshold, step)` builds the input at the threshold for
    step 0 and one unit past it for `past_step`, the unit across the edge where the verdict
    flips; `flags_at_threshold` says which side the rule records its entry on. `toggles`
    overrides the family's default rule toggles.
    """

    rule: str
    column: Optional[str]
    family: str
    threshold: str
    frame: Callable
    past_step: int
    flags_at_threshold: bool
    toggles: Optional[dict] = None

    @property
    def id(self) -> str:
        case_id = f"{self.rule}-{self.threshold}-{self.family}"

        return case_id


_MONTHLY_CYCLE = (30,) * 11
_BIMONTHLY_CYCLE = (60, 60, 60, 60, 45)

# Hourly gaps spread evenly at the daily coverage threshold leave some month short of the equal
# monthly coverage threshold, so these cases switch the monthly rule off to isolate the daily one.
_DAILY_COVERAGE_TOGGLES = {
    ("temperature", "hourly"): {
        "temperature": {"monthly_coverage": {"baseline": False, "reporting": False}},
    },
    ("observed", "hourly"): {"observed": {"monthly_coverage": {"baseline": False}}},
}

_THRESHOLD_CASES = [
    *(
        _ThresholdCase(
            "baseline_day_length", None, family, "min_baseline_length",
            _baseline_length_frame, -1, False,
        )
        for family in ("daily", "billing", "hourly")
    ),
    *(
        _ThresholdCase(
            "baseline_day_length", None, family, "max_baseline_length",
            _baseline_length_frame, 1, False,
        )
        for family in ("daily", "billing", "hourly")
    ),
    _ThresholdCase(
        "offcycle_reads", None, "billing", "min_days_in_period",
        partial(_billing_periods_frame, _MONTHLY_CYCLE), -1, False,
    ),
    _ThresholdCase(
        "offcycle_reads", None, "billing", "max_days_in_monthly_period",
        partial(_billing_periods_frame, _MONTHLY_CYCLE), 1, False,
    ),
    _ThresholdCase(
        "offcycle_reads", None, "billing", "max_days_in_bimonthly_period",
        partial(_billing_periods_frame, _BIMONTHLY_CYCLE), 1, False,
    ),
    _ThresholdCase(
        "high_frequency_meter_coverage", None, "daily", "observed.min_pct_hourly_coverage",
        _partial_meter_day_frame, 1, True,
    ),
    _ThresholdCase(
        "consecutive_missing_temperature_hours", None, "hourly",
        "temperature.max_consecutive_hours_missing", _temperature_gap_frame, 1, False,
        toggles={"rules": {"consecutive_missing_temperature_hours": {"baseline": True}}},
    ),
    *(
        _ThresholdCase(
            "unique_values", column, family, f"{column}.min_pct_unique_values",
            partial(_unique_values_frame, column), 1, False,
        )
        for column, family in (
            ("temperature", "daily"),
            ("temperature", "billing"),
            ("observed", "daily"),
            ("observed", "hourly"),
        )
    ),
    *(
        _ThresholdCase(
            "daily_coverage", column, family, f"{column}.min_pct_daily_coverage",
            partial(_daily_coverage_frame, column), 1, False,
            toggles=_DAILY_COVERAGE_TOGGLES.get((column, family)),
        )
        for column in ("temperature", "observed", "joint")
        for family in ("daily", "billing", "hourly")
    ),
    *(
        _ThresholdCase(
            "monthly_coverage", column, family, f"{column}.min_pct_monthly_coverage",
            partial(_monthly_coverage_frame, column), 1, False,
        )
        for column, family in (
            ("temperature", "daily"),
            ("temperature", "billing"),
            ("temperature", "hourly"),
            ("observed", "hourly"),
            ("ghi", "hourly"),
        )
    ),
]


def _case_settings(case: _ThresholdCase):
    """The family's default data settings, with the case's toggle overrides applied."""
    settings_class = _DATA_SETTINGS[case.family]
    if case.toggles is None:
        settings = settings_class()
    else:
        settings = settings_class(sufficiency=case.toggles)

    return settings


def _phases_switched_on(case: _ThresholdCase, sufficiency) -> list[str]:
    """The phases in which the case's rule may run and its toggle is on."""
    if case.column is None:
        block = sufficiency.rules
    else:
        block = getattr(sufficiency, case.column)

    toggle = getattr(block, case.rule)
    phases = [phase for phase in RULE_SPECS[case.rule].phases if getattr(toggle, phase)]

    return phases


def _data_outcome(family: str, phase: str, df: pd.DataFrame, settings) -> dict:
    """Build the family's data object for `phase` from `df`, returning whether its data
    qualifies and its compliance records."""
    data = _DATA_CLASSES[family, phase](df, is_electricity_data=True, settings=settings)
    outcome = {"passed": data.compliance.qualified, "records": _plain(data.compliance.all)}

    return outcome


# The daily data path drops missing hourly reads before rolling hours up to days, so the read
# before a gap is spread across the gap and every day reaches full hourly coverage.
_UNREACHABLE_FROM_HOURLY_GAPS = pytest.mark.xfail(
    strict=True,
    reason="high_frequency_meter_coverage cannot see a gap in hourly input to the daily model",
)


def _threshold_param(case: _ThresholdCase):
    marks = _UNREACHABLE_FROM_HOURLY_GAPS if case.rule == "high_frequency_meter_coverage" else ()
    param = pytest.param(case, id=case.id, marks=marks)

    return param


@pytest.mark.parametrize("case", [_threshold_param(case) for case in _THRESHOLD_CASES])
def test_sufficiency_threshold_verdicts_at_and_one_step_past_the_default(case, snapshot):
    """Data built exactly at a sufficiency rule's default threshold, and one step past it,
    carries the recorded verdicts and records in each phase where the rule is on, and the side
    `flags_at_threshold` names carries a record the other side does not."""
    settings = _case_settings(case)
    sufficiency = settings.sufficiency
    threshold = reduce(getattr, case.threshold.split("."), sufficiency)
    at_threshold = case.frame(case.family, sufficiency, threshold, 0)
    one_step_past = case.frame(case.family, sufficiency, threshold, case.past_step)

    outcomes = {
        phase: {
            "at_threshold": _data_outcome(case.family, phase, at_threshold, settings),
            "one_step_past": _data_outcome(case.family, phase, one_step_past, settings),
        }
        for phase in _phases_switched_on(case, sufficiency)
    }

    assert outcomes == snapshot
    for phase, sides in outcomes.items():
        if case.flags_at_threshold:
            flagged, unflagged = sides["at_threshold"], sides["one_step_past"]
        else:
            flagged, unflagged = sides["one_step_past"], sides["at_threshold"]
        flagged_names = {record["qualified_name"] for record in flagged["records"]}
        unflagged_names = {record["qualified_name"] for record in unflagged["records"]}
        assert flagged_names - unflagged_names, f"{phase}: no record appears across the edge"


def test_every_sufficiency_rule_is_classified_by_its_threshold():
    """Every rule in RULE_FUNCTIONS either has threshold cases, each pairing an input the rule
    flags with one it does not, or is declared to have no settings threshold."""
    thresholded = {case.rule for case in _THRESHOLD_CASES}
    unclassified = set(RULE_FUNCTIONS) - thresholded - _UNTHRESHOLDED_RULES
    contradictory = thresholded & _UNTHRESHOLDED_RULES

    assert unclassified == set(), (
        f"rules with neither threshold cases nor a no-threshold declaration: {sorted(unclassified)}"
    )
    assert contradictory == set(), (
        f"rules declared without a threshold that have threshold cases: {sorted(contradictory)}"
    )
