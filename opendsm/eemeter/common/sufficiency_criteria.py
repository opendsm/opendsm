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

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytz

import pydantic

from opendsm.common.base_settings import BaseSettings
from opendsm.common.pydantic_utils import computed_field_cached_property

from opendsm.eemeter.common.data_processor_utilities import day_counts
from opendsm.eemeter.common.data_settings import BaseSufficiencySettings

from opendsm.eemeter.common.warnings import EEMeterWarning


# TODO implement as registered functions rather than needing to call everything manually
# probably easiest to use two decorators, can be stacked, for baseline/reporting


class SufficiencyCriteria(BaseSettings):
    model_config = pydantic.ConfigDict(
        frozen = False,
        arbitrary_types_allowed=True,
        str_to_lower = True,
        str_strip_whitespace = True,
    )

    data: pd.DataFrame
    is_electricity_data: bool
    is_reporting_data: bool
    settings: BaseSufficiencySettings

    _n_valid_observed_days = None
    _n_valid_days = None
    _n_valid_temperature_days = None

    disqualification: list = []
    warnings: list = []

    @computed_field_cached_property()
    def _has_ghi(self) -> bool:
        return "ghi" in self.data.columns

    @computed_field_cached_property()
    def n_days_total(self) -> int:
        requested_start = self.settings.requested_start
        requested_end = self.settings.requested_end

        data_end = self.data.dropna().index.max()
        data_start = self.data.dropna().index.min()
        n_days_data = (
            data_end - data_start
        ).days + 1  # TODO confirm. no longer using last row nan

        if requested_start is not None:
            # check for gap at beginning
            requested_start = requested_start.astimezone(pytz.UTC)
            n_days_start_gap = (data_start - requested_start).days
        else:
            n_days_start_gap = 0

        if requested_end is not None:
            # check for gap at end
            requested_end = requested_end.astimezone(pytz.UTC)
            n_days_end_gap = (requested_end - data_end).days
        else:
            n_days_end_gap = 0

        n_days_total = n_days_data + n_days_start_gap + n_days_end_gap

        return n_days_total

    def _compute_valid_day_counts(self):
        min_pct = self.settings.temperature.min_pct_period_coverage
        valid_temperature_rows = (
            self.data.temperature_not_null
            / (self.data.temperature_not_null + self.data.temperature_null)
        ) > min_pct

        # get number of days per period - for daily this should be a series of ones
        row_day_counts = day_counts(self.data.index)

        # get valid rows
        valid_rows = valid_temperature_rows

        if not self.is_reporting_data:
            valid_observed_rows = self.data.observed.notnull()
            valid_rows = valid_rows & valid_observed_rows

            n_valid_observed_days = (valid_observed_rows * row_day_counts).sum()
            self._n_valid_observed_days = int(n_valid_observed_days)
        else:
            self._n_valid_observed_days = None

        self._n_valid_temperature_days = int((valid_temperature_rows * row_day_counts).sum())
        self._n_valid_days = int((valid_rows * row_day_counts).sum())
    
    @computed_field_cached_property()
    def n_valid_temperature_days(self) -> int:
        if self._n_valid_temperature_days is None:
            self._compute_valid_day_counts()

        return self._n_valid_temperature_days

    @computed_field_cached_property()
    def n_valid_observed_days(self) -> int:
        if self._n_valid_observed_days is None:
            self._compute_valid_day_counts()

        return self._n_valid_observed_days

    @computed_field_cached_property()
    def n_valid_days(self) -> int:
        if self._n_valid_days is None:
            self._compute_valid_day_counts()

        return self._n_valid_days

    def _check_no_data(self):
        if self.data.dropna().empty:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.no_data",
                    description=("No data available."),
                    data={},
                )
            )
            return False
        return True

    def _check_n_days_boundary_gap(self, gap_type: Literal["start", "end"]):
        gap = 0
        if gap_type == "start":
            user_boundary = self.settings.requested_start
            
            if user_boundary is not None:
                data_boundary = self.data.dropna().index.min()
                gap = (data_boundary - user_boundary).days
        else:
            if user_boundary is not None:
                data_boundary = self.data.dropna().index.max()
                gap = (user_boundary - data_boundary).days

        if gap < 0:
            # CalTRACK 2.2.4
            if gap_type == "start":
                err = "before"
            else:
                err = "after"

            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria"
                        f".extra_data_{err}_requested_{gap_type}_date"
                    ),
                    description=(f"Extra data found {err} requested {gap_type} date."),
                    data={
                        f"requested_{gap_type}": user_boundary.isoformat(),
                        f"data_{gap_type}": data_boundary.isoformat(),
                    },
                )
            )

    def _check_baseline_day_length(self):
        min_length = self.settings.min_baseline_length
        max_length = self.settings.max_baseline_length

        if self.is_reporting_data:
            return
        
        if self.n_days_total < min_length or self.n_days_total > max_length:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria" ".incorrect_number_of_total_days"
                    ),
                    description=(
                        f"Baseline length is not within the expected range of {min_length}-{max_length} days."
                    ),
                    data={"n_days_total": self.n_days_total},
                )
            )

    def _check_negative_observed_values(self):
        if self.is_reporting_data:
            return
        elif self.is_electricity_data:
            return

        n_negative_observed_values = self.data.observed[self.data.observed < 0].shape[0]

        if n_negative_observed_values > 0:
            # CalTrack 2.3.5
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria" ".negative_observed_values"
                    ),
                    description=("Found negative Observed values"),
                    data={"n_negative_observed_values": n_negative_observed_values},
                )
            )
    
    def _check_valid_days_percentage(self, col: Literal["temperature", "ghi", "observed", "joint"]):
        if self.is_reporting_data and col == "observed":
            return
        elif col == "ghi" and not self._has_ghi:
            return
        
        n_days_total = float(self.n_days_total)

        if col == "temperature":
            name = col.capitalize()
            valid_days = self.n_valid_temperature_days
            min_pct = self.settings.temperature.min_pct_daily_coverage
        elif col == "ghi":
            name = col.upper()
            raise NotImplementedError("GHI valid days percentage check not implemented yet")
            valid_days = self.n_valid_ghi_days
            min_pct = self.settings.ghi.min_pct_daily_coverage
        elif col == "observed":
            name = col.capitalize()
            valid_days = self.n_valid_observed_days
            min_pct = self.settings.observed.min_pct_daily_coverage
        elif col == "joint":
            name = col.capitalize()
            valid_days = self.n_valid_days
            min_pct = self.settings.joint.min_pct_daily_coverage

        valid_pct = 0
        if n_days_total > 0:
            valid_pct = valid_days / n_days_total

        if valid_pct < min_pct:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria"
                        f".too_many_days_with_missing_{col}_data"
                    ),
                    description=(
                        f"Too many days in data have missing {name} data."
                    ),
                    data={
                        f"n_valid_{col}_data_days": valid_days,
                        "n_days_total": n_days_total,
                    },
                )
            )

    def _check_valid_monthly_coverage(self, col: Literal["temperature", "ghi", "observed", "joint"]):
        if self.is_reporting_data and col == "observed":
            return
        elif col == "ghi" and not self._has_ghi:
            return

        if col == "temperature":
            name = col.capitalize()
            min_pct = self.settings.temperature.min_pct_monthly_coverage
        elif col == "ghi":
            name = col.upper()
            min_pct = self.settings.ghi.min_pct_monthly_coverage
        elif col == "observed":
            name = col.capitalize()
            min_pct = self.settings.observed.min_pct_monthly_coverage
        elif col == "joint":
            name = col.capitalize()
            raise NotImplementedError("Joint monthly coverage check not implemented yet")
            min_pct = self.settings.joint.min_pct_monthly_coverage
        
        non_null_pct_per_month = (
            self.data[col]
            .groupby(self.data.index.month)
            .apply(lambda x: x.notna().mean())
        )

        if (non_null_pct_per_month < min_pct).any():
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=f"eemeter.sufficiency_criteria.missing_monthly_{col}_data",
                    description=(
                        f"More than {(1-min_pct)*100}% of the monthly {name} data is missing."
                    ),
                    data={
                        "lowest_monthly_coverage": non_null_pct_per_month.min(),
                    },
                )
            )

    def _check_season_weekday_weekend_availability(self):
        raise NotImplementedError(
            "90% of season and weekday/weekend check not implemented yet"
        )

    def _check_extreme_values(self):
        if self.is_reporting_data:
            return    
        elif self.data["observed"].dropna().empty:
            return
        
        if not self.is_reporting_data:
            median = self.data.observed.median()
            lower_quantile = self.data.observed.quantile(0.25)
            upper_quantile = self.data.observed.quantile(0.75)
            iqr = upper_quantile - lower_quantile
            lower_bound = lower_quantile - (3 * iqr)
            upper_bound = upper_quantile + (3 * iqr)
            n_extreme_values = self.data.observed[
                (self.data.observed < lower_bound) | (self.data.observed > upper_bound)
            ].shape[0]
            min_value = float(self.data.observed.min())
            max_value = float(self.data.observed.max())

            if n_extreme_values > 0:
                # Inspired by CalTRACK 2.3.6
                self.warnings.append(
                    EEMeterWarning(
                        qualified_name=(
                            "eemeter.sufficiency_criteria" ".extreme_values_detected"
                        ),
                        description=(
                            "Extreme values (outside 3x IQR) must be flagged for manual review."
                        ),
                        data={
                            "n_extreme_values": n_extreme_values,
                            "median": median,
                            "upper_quantile": upper_quantile,
                            "lower_quantile": lower_quantile,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                            "min_value": min_value,
                            "max_value": max_value,
                        },
                    )
                )

    def _check_high_frequency_temperature_values(self):
        # TODO broken as written
        # If high frequency data check for 50% data coverage in rollup
        min_pct = self.settings.temperature.min_pct_hourly_coverage
        if len(temperature_features[temperature_features.coverage <= min_pct]) > 0:
            self.warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.missing_high_frequency_temperature_data",
                    description=(
                        f"More than {(1-min_pct)*100}% of the high frequency Temperature data is missing."
                    ),
                    data={
                        "high_frequency_data_missing_count": len(
                            temperature_features[
                                temperature_features.coverage <= min_pct
                            ].index.to_list()
                        )
                    },
                )
            )

        # Set missing high frequency data to NaN
        temperature_features.value[temperature_features.coverage > min_pct] = (
            temperature_features[temperature_features.coverage > min_pct].value
            / temperature_features[temperature_features.coverage > min_pct].coverage
        )

        temperature_features = (
            temperature_features[temperature_features.coverage > min_pct]
            .reindex(temperature_features.index)[["value"]]
            .rename(columns={"value": "temperature_mean"})
        )

        if "coverage" in temperature_features.columns:
            temperature_features = temperature_features.drop(columns=["coverage"])

    def _check_high_frequency_observed_values(self):
        min_pct = self.settings.observed.min_pct_hourly_coverage
        if not self.data[self.data.coverage <= min_pct].empty:
            self.warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.missing_high_frequency_observed_data",
                    description=(
                        f"More than {(1-min_pct)*100}% of the high frequency Observed data is missing."
                    ),
                    data=(self.data[self.data.coverage <= min_pct].index.to_list()),
                )
            )

        # CalTRACK 2.2.2.1 - interpolate with average of non-null values
        self.data.value[self.data.coverage > min_pct] = (
            self.data[self.data.coverage > min_pct].value
            / self.data[self.data.coverage > min_pct].coverage
        )

    def check_sufficiency_baseline(self):
        self._check_no_data()
        self._check_baseline_day_length()
        self._check_negative_observed_values()

        self._check_valid_days_percentage(col="temperature")
        self._check_valid_days_percentage(col="observed")
        self._check_valid_days_percentage(col="joint")
        self._check_valid_monthly_coverage(col="temperature")

        self._check_extreme_values()

    def check_sufficiency_reporting(self):
        self._check_no_data()

        self._check_valid_days_percentage(col="temperature")
        self._check_valid_days_percentage(col="joint")
        self._check_valid_monthly_coverage(col="temperature")
        # self._check_high_frequency_temperature_values()


class DailySufficiencyCriteria(SufficiencyCriteria):
    """
    Sufficiency Criteria class for daily models
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_sufficiency_baseline(self):
        super().check_sufficiency_baseline()

        # self._check_n_days_boundary_gap("start")
        # self._check_n_days_boundary_gap("end")
        # TODO : Maybe make these checks static? To work with the current data class
        # self._check_high_frequency_meter_values()
        # self._check_high_frequency_temperature_values()

    def check_sufficiency_reporting(self):
        super().check_sufficiency_reporting()


class BillingSufficiencyCriteria(SufficiencyCriteria):
    """
    Sufficiency Criteria class for billing models - monthly / bimonthly
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_observed_data_billing_monthly(self):
        if self.data["value"].dropna().empty:
            return

        diff = list((data.index[1:] - data.index[:-1]).days)
        filter_ = pd.Series(diff + [np.nan], index=data.index)

        min_days = self.settings.min_days_in_period
        max_days = self.settings.max_days_in_monthly_period

        # CalTRACK 2.2.3.4, 2.2.3.5
        # Billing Monthly data frequency check
        data = data[(min_days <= filter_) & (filter_ <= max_days)].reindex(  # keep these, inclusive
            data.index
        )

        if len(data[(max_days < filter_) | (filter_ < min_days)]) > 0:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data",
                    description=(
                        f"Off-cycle reads found in billing monthly data having a duration less than {min_days} days or greater than {max_days} days"
                    ),
                    data=(data[(max_days < filter_) | (filter_ < min_days)].index.to_list()),
                )
            )

    def _check_observed_data_billing_bimonthly(self):
        if self.data["value"].dropna().empty:
            return

        diff = list((data.index[1:] - data.index[:-1]).days)
        filter_ = pd.Series(diff + [np.nan], index=data.index)

        min_days = self.settings.min_days_in_period
        max_days = self.settings.max_days_in_bimonthly_period

        # CalTRACK 2.2.3.4, 2.2.3.5
        data = data[(min_days <= filter_) & (filter_ <= max_days)].reindex(  # keep these, inclusive
            data.index
        )

        if len(data[(max_days < filter_) | (filter_ < min_days)]) > 0:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.offcycle_reads_in_billing_bimonthly_data",
                    description=(
                        f"Off-cycle reads found in billing bimonthly data having a duration less than {min_days} days or greater than {max_days} days"
                    ),
                    data=(data[(max_days < filter_) | (filter_ < min_days)].index.to_list()),
                )
            )

    def _check_estimated_observed_values(self):
        # CalTRACK 2.2.3.1
        """
        Adds estimate to subsequent read if there aren't more than one estimate in a row
        and then removes the estimated row.

        Input:
        index   value   estimated
        1       2       False
        2       3       False
        3       5       True
        4       4       False
        5       6       True
        6       3       True
        7       4       False
        8       NaN     NaN

        Output:
        index   value
        1       2
        2       3
        4       9
        5       NaN
        7       7
        8       NaN
        """
        add_estimated = []
        remove_estimated_fixed_rows = []
        data = self.data
        if "estimated" in data.columns:
            data["unestimated_value"] = (
                data[:-1].value[(data[:-1].estimated == False)].reindex(data.index)
            )
            data["estimated_value"] = (
                data[:-1].value[(data[:-1].estimated)].reindex(data.index)
            )
            for i, (index, row) in enumerate(data[:-1].iterrows()):
                # ensures there is a prev_row and previous row value is null
                if i > 0 and pd.isnull(prev_row["unestimated_value"]):
                    # current row value is not null
                    add_estimated.append(prev_row["estimated_value"])
                    if not pd.isnull(row["unestimated_value"]):
                        # get all rows that had only estimated reads that will be
                        # added to the subsequent row meaning this row
                        # needs to be removed
                        remove_estimated_fixed_rows.append(prev_index)
                else:
                    add_estimated.append(0)
                prev_row = row
                prev_index = index
            add_estimated.append(np.nan)
            data["value"] = data["unestimated_value"] + add_estimated
            data = data[~data.index.isin(remove_estimated_fixed_rows)]
            data = data[["value"]]  # remove the estimated column

    def check_sufficiency_baseline(self):
        super().check_sufficiency_baseline()

        # self._check_n_days_boundary_gap("start")
        # self._check_n_days_boundary_gap("end")
        # if self.median_granularity == "billing_monthly":
        #     self._check_observed_data_billing_monthly()
        # else :
        #     self._check_observed_data_billing_bimonthly()
        self._check_estimated_observed_values()
        # self._check_high_frequency_temperature_values()

    def check_sufficiency_reporting(self):
        super().check_sufficiency_reporting()


class HourlySufficiencyCriteria(SufficiencyCriteria):
    """
    Sufficiency Criteria class for hourly models
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_baseline_length_hourly_model(self):
        pass

    def _check_hourly_consecutive_temperature_data(self):
        # TODO : Check implementation wrt Caltrack 2.2.4.1
        # Resample to hourly by taking the first non NaN value
        hourly_data = self.data["temperature"].resample("H").first()
        mask = hourly_data.isna().any(axis=1)
        grouped = mask.groupby((mask != mask.shift()).cumsum())
        max_consecutive_nans = grouped.sum().max()
        allowed_consecutive_nans = self.settings.temperature.max_consecutive_hours_missing
        if max_consecutive_nans > allowed_consecutive_nans:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.too_many_consecutive_hours_temperature_data_missing",
                    description=(
                        f"More than {allowed_consecutive_nans} hours of consecutive hourly Temperature data is missing."
                    ),
                    data={"Max_consecutive_hours_missing": int(max_consecutive_nans)},
                )
            )

    def check_sufficiency_baseline(self):
        super().check_sufficiency_baseline()
        
        # TODO : add caltrack check number on top of each method
        # self._check_n_days_boundary_gap("start")
        # self._check_n_days_boundary_gap("end")
        self._check_valid_monthly_coverage(col="ghi")
        self._check_valid_monthly_coverage(col="observed")
        # TODO these will only apply to legacy, and currently do not work
        # self._check_high_frequency_observed_values()
        # self._check_high_frequency_temperature_values()
        # self._check_hourly_consecutive_temperature_data()

    def check_sufficiency_reporting(self):
        super().check_sufficiency_reporting()
        
        self._check_valid_monthly_coverage(col="ghi")