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

"""The context a sufficiency rule reads.

The rules themselves live in `sufficiency_rules.py`; this module holds the frame, the
settings, the data report and the cached day counts they share.
"""

from __future__ import annotations

import math
from typing import Literal

import pandas as pd

import pydantic

from opendsm.common.base_settings import BaseSettings
from opendsm.common.pydantic_utils import computed_field_cached_property

from opendsm.eemeter.common.compliance import Compliance
from opendsm.eemeter.common.data_processor_utilities import DataReport
from opendsm.eemeter.common.data_settings import BaseSufficiencySettings



def _round_sig(x, sig=4):
    """Round a scalar to `sig` significant figures; returns a plain float."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return x

    return round(x, sig - 1 - int(math.floor(math.log10(abs(x)))))


_DAY = pd.Timedelta(days=1)

# the span a lone row covers, since one row leaves no spacing to read
_SINGLE_ROW_STEP = {
    "hourly": pd.Timedelta(hours=1),
    "daily": _DAY,
    "billing": _DAY,
}


def _wall_clock(timestamp: pd.Timestamp) -> pd.Timestamp:
    """A timestamp as its own wall-clock reading, without timezone; NaT stays NaT."""
    if pd.isna(timestamp):
        return timestamp

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    return timestamp


def _in_zone_of(timestamp: pd.Timestamp, like: pd.Timestamp) -> pd.Timestamp:
    """``timestamp`` expressed in the timezone of ``like``, when both carry one."""
    if timestamp.tzinfo is not None and getattr(like, "tzinfo", None) is not None:
        timestamp = timestamp.tz_convert(like.tzinfo)

    return timestamp


def _valid_days(valid_rows: pd.Series, step: pd.Timedelta) -> float:
    """Days of valid data: each valid row counts for its share of a day, so the last row counts
    like the rest and a partial first or last day counts by the rows it has."""
    valid_days = valid_rows.sum() * (step / _DAY)

    return float(valid_days)


class SufficiencyCriteria(BaseSettings):
    """Everything a sufficiency rule may read about one frame."""

    model_config = pydantic.ConfigDict(
        frozen = False,
        arbitrary_types_allowed=True,
        str_to_lower = True,
        str_strip_whitespace = True,
    )

    data: pd.DataFrame
    is_electricity_data: bool
    is_reporting_data: bool
    family: Literal["hourly", "daily", "billing"]
    settings: BaseSufficiencySettings

    _n_valid_observed_days = None
    _n_valid_days = None
    _n_valid_temperature_days = None

    data_report: DataReport = pydantic.Field(default_factory=DataReport)
    compliance: Compliance = pydantic.Field(default_factory=Compliance)

    def _should_skip_col(self, col: str) -> bool:
        """Check if a column-based check should be skipped."""
        if self.is_reporting_data and col == "observed" and not self._has_observed_values:
            return True
        if col == "ghi" and not self._has_ghi:
            return True

        return False

    def _col_display_name(self, col: str) -> str:
        """Get display name for a column (e.g. 'temperature' -> 'Temperature', 'ghi' -> 'GHI')."""
        if col == "ghi":
            return "GHI"

        return col.capitalize()

    def _col_settings(self, col: str):
        """Get the sufficiency settings object for a column."""
        return getattr(self.settings, col)

    @computed_field_cached_property()
    def _has_ghi(self) -> bool:
        return "ghi" in self.data.columns

    @computed_field_cached_property()
    def _has_observed_values(self) -> bool:
        return "observed" in self.data.columns and bool(self.data.observed.notnull().any())

    @computed_field_cached_property()
    def n_days_total(self) -> float:
        """Days the data spans on its own wall clock: first to last row with data plus one row's
        step, extended to the requested bounds. A DST change or an off-midnight start does not
        shift it."""
        requested_start = self.settings.requested_start
        requested_end = self.settings.requested_end

        non_null_data = self.data.dropna()
        data_start = non_null_data.index.min()
        data_end = non_null_data.index.max()
        step = self._grid_step()
        n_days_data = (_wall_clock(data_end) - _wall_clock(data_start) + step) / _DAY

        n_days_start_gap = 0.0
        if requested_start is not None:
            requested_start = _in_zone_of(requested_start, data_start)
            n_days_start_gap = (_wall_clock(data_start) - _wall_clock(requested_start)) / _DAY

        n_days_end_gap = 0.0
        if requested_end is not None:
            requested_end = _in_zone_of(requested_end, data_end)
            n_days_end_gap = (_wall_clock(requested_end) - _wall_clock(data_end)) / _DAY

        return float(n_days_data + n_days_start_gap + n_days_end_gap)

    def _grid_step(self) -> pd.Timedelta:
        """The regular spacing of the data's rows (one hour or one day)."""
        index = self.data.index
        if len(index) < 2:
            return _SINGLE_ROW_STEP[self.family]

        step = pd.Series(index[1:] - index[:-1]).median()

        return step

    def _compute_valid_day_counts(self):
        step = self._grid_step()
        min_pct = self.settings.temperature.min_pct_period_coverage
        valid_temperature_rows = (
            self.data.temperature_not_null
            / (self.data.temperature_not_null + self.data.temperature_null)
        ) > min_pct

        valid_rows = valid_temperature_rows

        if "observed" in self.data.columns and (not self.is_reporting_data or self._has_observed_values):
            valid_observed_rows = self.data.observed.notnull()
            valid_rows = valid_rows & valid_observed_rows
            self._n_valid_observed_days = _valid_days(valid_observed_rows, step)
        else:
            self._n_valid_observed_days = None

        self._n_valid_temperature_days = _valid_days(valid_temperature_rows, step)
        self._n_valid_days = _valid_days(valid_rows, step)

    @computed_field_cached_property()
    def n_valid_temperature_days(self) -> float:
        if self._n_valid_temperature_days is None:
            self._compute_valid_day_counts()

        return self._n_valid_temperature_days

    @computed_field_cached_property()
    def n_valid_observed_days(self) -> float:
        if self._n_valid_observed_days is None:
            self._compute_valid_day_counts()

        return self._n_valid_observed_days

    @computed_field_cached_property()
    def n_valid_days(self) -> float:
        if self._n_valid_days is None:
            self._compute_valid_day_counts()

        return self._n_valid_days
