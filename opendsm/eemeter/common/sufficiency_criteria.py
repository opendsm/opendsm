#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2025 OpenDSM contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
from math import ceil

import numpy as np
import pandas as pd
import pytz
from typing import Optional

from pydantic import BaseModel, ConfigDict

from opendsm.eemeter.common.warnings import EEMeterWarning
from opendsm.eemeter.common.data_processor_utilities import day_counts

# TODO implement as registered functions rather than needing to call everything manually
# probably easiest to use two decorators, can be stacked, for baseline/reporting


class SufficiencyCriteria(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame
    requested_start: Optional[pd.Timestamp] = None
    requested_end: Optional[pd.Timestamp] = None
    num_days: int = 365
    min_fraction_daily_coverage: float = 0.9
    min_fraction_hourly_temperature_coverage_per_period: float = 0.9
    is_reporting_data: bool = False
    is_electricity_data: bool = True
    # note that pydantic creates mutable defaults on each init
    # so this does not run into the issue that mutable default params in functions have
    disqualification: list = []
    warnings: list = []
    n_days_total: Optional[int] = None

    n_valid_meter_value_days: Optional[int] = None
    n_valid_days: Optional[int] = None
    n_valid_temperature_days: Optional[int] = None

    # TODO probably use a model validator rather than this blanket call
    def model_post_init(self, __context):
        self._compute_n_days_total()
        self._compute_valid_meter_temperature_days()

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

    def _check_n_days_end_gap(self):
        data_end = self.data.dropna().index.max()

        if self.requested_end is not None:
            # check for gap at end
            self.requested_end = self.requested_end.astimezone(pytz.UTC)
            n_days_end_gap = (self.requested_end - data_end).days
        else:
            n_days_end_gap = 0

        if n_days_end_gap < 0:
            # CalTRACK 2.2.4
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria"
                        ".extra_data_after_requested_end_date"
                    ),
                    description=("Extra data found after requested end date."),
                    data={
                        "requested_end": self.requested_end.isoformat(),
                        "data_end": data_end.isoformat(),
                    },
                )
            )
        n_days_end_gap = 0

    def _check_n_days_start_gap(self):
        data_start = self.data.dropna().index.min()

        if self.requested_start is not None:
            # check for gap at beginning
            self.requested_start = self.requested_start.astimezone(pytz.UTC)
            n_days_start_gap = (data_start - self.requested_start).days
        else:
            n_days_start_gap = 0

        if n_days_start_gap < 0:
            # CalTRACK 2.2.4
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria"
                        ".extra_data_before_requested_start_date"
                    ),
                    description=("Extra data found before requested start date."),
                    data={
                        "requested_start": self.requested_start.isoformat(),
                        "data_start": data_start.isoformat(),
                    },
                )
            )
            n_days_start_gap = 0

    def _check_negative_meter_values(self):
        if not self.is_reporting_data and not self.is_electricity_data:
            n_negative_meter_values = self.data.observed[self.data.observed < 0].shape[
                0
            ]

            if n_negative_meter_values > 0:
                # CalTrack 2.3.5
                self.disqualification.append(
                    EEMeterWarning(
                        qualified_name=(
                            "eemeter.sufficiency_criteria" ".negative_meter_values"
                        ),
                        description=("Found negative meter data values"),
                        data={"n_negative_meter_values": n_negative_meter_values},
                    )
                )

    def _compute_n_days_total(self):
        data_end = self.data.dropna().index.max()
        data_start = self.data.dropna().index.min()
        n_days_data = (
            data_end - data_start
        ).days + 1  # TODO confirm. no longer using last row nan

        if self.requested_start is not None:
            # check for gap at beginning
            self.requested_start = self.requested_start.astimezone(pytz.UTC)
            n_days_start_gap = (data_start - self.requested_start).days
        else:
            n_days_start_gap = 0

        if self.requested_end is not None:
            # check for gap at end
            self.requested_end = self.requested_end.astimezone(pytz.UTC)
            n_days_end_gap = (self.requested_end - data_end).days
        else:
            n_days_end_gap = 0

        n_days_total = n_days_data + n_days_start_gap + n_days_end_gap

        self.n_days_total = n_days_total

    def _check_baseline_length_daily_billing_model(self):
        MAX_BASELINE_LENGTH = 365
        MIN_BASELINE_LENGTH = ceil(0.9 * MAX_BASELINE_LENGTH)
        if (
            not self.is_reporting_data
            # TODO can throw with Optional. check if actually optional vs using -1 as default?
            and self.n_days_total > MAX_BASELINE_LENGTH
            or self.n_days_total < MIN_BASELINE_LENGTH
        ):
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria" ".incorrect_number_of_total_days"
                    ),
                    description=(
                        f"Baseline length is not within the expected range of {MIN_BASELINE_LENGTH}-{MAX_BASELINE_LENGTH} days."
                    ),
                    data={"num_days": self.num_days, "n_days_total": self.n_days_total},
                )
            )

    def _compute_valid_meter_temperature_days(self):
        if not self.is_reporting_data:
            valid_meter_value_rows = self.data.observed.notnull()
        valid_temperature_rows = (
            self.data.temperature_not_null
            / (self.data.temperature_not_null + self.data.temperature_null)
        ) > self.min_fraction_hourly_temperature_coverage_per_period

        if not self.is_reporting_data:
            valid_rows = valid_meter_value_rows & valid_temperature_rows
        else:
            valid_rows = valid_temperature_rows

        # get number of days per period - for daily this should be a series of ones
        row_day_counts = day_counts(self.data.index)

        # apply masks, giving total
        if not self.is_reporting_data:
            self.n_valid_meter_value_days = int(
                (valid_meter_value_rows * row_day_counts).sum()
            )
        n_valid_temperature_days = int((valid_temperature_rows * row_day_counts).sum())
        n_valid_days = int((valid_rows * row_day_counts).sum())

        self.n_valid_days = n_valid_days
        self.n_valid_temperature_days = n_valid_temperature_days

    def _check_valid_days_percentage(self):
        if self.n_days_total > 0:
            fraction_valid_days = self.n_valid_days / float(self.n_days_total)
        else:
            fraction_valid_days = 0

        if fraction_valid_days < self.min_fraction_daily_coverage:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria"
                        ".too_many_days_with_missing_data"
                    ),
                    description=(
                        "Too many days in data have missing meter data or"
                        " temperature data."
                    ),
                    data={
                        "n_valid_days": self.n_valid_days,
                        "n_days_total": self.n_days_total,
                    },
                )
            )

    def _check_valid_meter_readings_percentage(self):
        if self.n_days_total > 0:
            if not self.is_reporting_data:
                fraction_valid_meter_value_days = self.n_valid_meter_value_days / float(
                    self.n_days_total
                )
        else:
            fraction_valid_meter_value_days = 0

        if (
            not self.is_reporting_data
            and fraction_valid_meter_value_days < self.min_fraction_daily_coverage
        ):
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria"
                        ".too_many_days_with_missing_meter_data"
                    ),
                    description=("Too many days in data have missing meter data."),
                    data={
                        "n_valid_meter_data_days": self.n_valid_meter_value_days,
                        "n_days_total": self.n_days_total,
                    },
                )
            )

    def _check_valid_temperature_values_percentage(self):
        if self.n_days_total > 0:
            fraction_valid_temperature_days = self.n_valid_temperature_days / float(
                self.n_days_total
            )
        else:
            fraction_valid_temperature_days = 0

        if fraction_valid_temperature_days < self.min_fraction_daily_coverage:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name=(
                        "eemeter.sufficiency_criteria"
                        ".too_many_days_with_missing_temperature_data"
                    ),
                    description=(
                        "Too many days in data have missing temperature data."
                    ),
                    data={
                        "n_valid_temperature_data_days": self.n_valid_temperature_days,
                        "n_days_total": self.n_days_total,
                    },
                )
            )

    def _check_monthly_temperature_values_percentage(self):
        non_null_temp_percentage_per_month = (
            self.data["temperature"]
            .groupby(self.data.index.month)
            .apply(lambda x: x.notna().mean())
        )
        if (
            non_null_temp_percentage_per_month < self.min_fraction_daily_coverage
        ).any():
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.missing_monthly_temperature_data",
                    description=(
                        "More than 10% of the monthly temperature data is missing."
                    ),
                    data={
                        # TODO report percentage
                    },
                )
            )

    def _check_season_weekday_weekend_availability(self):
        raise NotImplementedError(
            "90% of season and weekday/weekend check not implemented yet"
        )

    def _check_extreme_values(self):
        if self.data["observed"].dropna().empty:
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
        if len(temperature_features[temperature_features.coverage <= 0.5]) > 0:
            self.warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.missing_high_frequency_temperature_data",
                    description=(
                        "More than 50% of the high frequency Temperature data is missing."
                    ),
                    data={
                        "high_frequency_data_missing_count": len(
                            temperature_features[
                                temperature_features.coverage <= 0.5
                            ].index.to_list()
                        )
                    },
                )
            )

        # Set missing high frequency data to NaN
        temperature_features.value[temperature_features.coverage > 0.5] = (
            temperature_features[temperature_features.coverage > 0.5].value
            / temperature_features[temperature_features.coverage > 0.5].coverage
        )

        temperature_features = (
            temperature_features[temperature_features.coverage > 0.5]
            .reindex(temperature_features.index)[["value"]]
            .rename(columns={"value": "temperature_mean"})
        )

        if "coverage" in temperature_features.columns:
            temperature_features = temperature_features.drop(columns=["coverage"])

    def _check_high_frequency_meter_values(self):
        if not self.data[self.data.coverage <= 0.5].empty:
            self.warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.missing_high_frequency_meter_data",
                    description=(
                        "More than 50% of the high frequency Meter data is missing."
                    ),
                    data=(self.data[self.data.coverage <= 0.5].index.to_list()),
                )
            )

        # CalTRACK 2.2.2.1 - interpolate with average of non-null values
        self.data.value[self.data.coverage > 0.5] = (
            self.data[self.data.coverage > 0.5].value
            / self.data[self.data.coverage > 0.5].coverage
        )

    def check_sufficiency_baseline(self):
        raise NotImplementedError(
            "Use Hourly / Daily / Billing SufficiencyCriteria class for concrete implementation"
        )

    def check_sufficiency_reporting(self):
        raise NotImplementedError(
            "Use Hourly / Daily / Billing SufficiencyCriteria class for concrete implementation"
        )


class HourlySufficiencyCriteria(SufficiencyCriteria):
    """
    Sufficiency Criteria class for hourly models
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_baseline_length_hourly_model(self):
        pass

    def _check_monthly_meter_readings_percentage(self):
        if not self.is_reporting_data:
            non_null_meter_percentage_per_month = (
                self.data["observed"]
                .groupby(self.data.index.month)
                .apply(lambda x: x.notna().mean())
            )
            if (
                non_null_meter_percentage_per_month < self.min_fraction_daily_coverage
            ).any():
                self.disqualification.append(
                    EEMeterWarning(
                        qualified_name="eemeter.sufficiency_criteria.missing_monthly_meter_data",
                        description=(
                            "More than 10% of the monthly meter data is missing."
                        ),
                        data={
                            "lowest_monthly_coverage": non_null_meter_percentage_per_month.min(),
                        },
                    )
                )

    def _check_hourly_consecutive_temperature_data(self):
        # TODO : Check implementation wrt Caltrack 2.2.4.1
        # Resample to hourly by taking the first non NaN value
        hourly_data = self.data["temperature"].resample("H").first()
        mask = hourly_data.isna().any(axis=1)
        grouped = mask.groupby((mask != mask.shift()).cumsum())
        max_consecutive_nans = grouped.sum().max()
        if max_consecutive_nans > 6:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.too_many_consecutive_hours_temperature_data_missing",
                    description=(
                        "More than 6 hours of consecutive hourly data is missing."
                    ),
                    data={"Max_consecutive_hours_missing": int(max_consecutive_nans)},
                )
            )

    def _check_monthly_ghi_percentage(self):
        if "ghi" not in self.data.columns:
            return
        non_null_temp_ghi_per_month = (
            self.data["ghi"]
            .groupby(self.data.index.month)
            .apply(lambda x: x.notna().mean())
        )
        if (non_null_temp_ghi_per_month < self.min_fraction_daily_coverage).any():
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.missing_monthly_ghi_data",
                    description=("At least one month is missing over 10% of GHI data."),
                    data={
                        "lowest_monthly_coverage": non_null_temp_ghi_per_month.min(),
                    },
                )
            )

    def check_sufficiency_baseline(self):
        # TODO : add caltrack check number on top of each method
        self._check_no_data()
        self._check_negative_meter_values()
        self._check_baseline_length_daily_billing_model()
        self._check_valid_days_percentage()
        self._check_valid_meter_readings_percentage()
        self._check_valid_temperature_values_percentage()
        self._check_monthly_temperature_values_percentage()
        self._check_monthly_meter_readings_percentage()
        self._check_extreme_values()
        self._check_monthly_ghi_percentage()
        # TODO these will only apply to legacy, and currently do not work
        # self._check_high_frequency_meter_values()
        # self._check_high_frequency_temperature_values()
        # self._check_hourly_consecutive_temperature_data()

    def check_sufficiency_reporting(self):
        self._check_no_data()
        self._check_valid_days_percentage()
        self._check_valid_temperature_values_percentage()
        self._check_monthly_temperature_values_percentage()
        self._check_monthly_ghi_percentage()
        # self._check_high_frequency_temperature_values()


class DailySufficiencyCriteria(SufficiencyCriteria):
    """
    Sufficiency Criteria class for daily models
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check_sufficiency_baseline(self):
        self._check_no_data()
        self._check_negative_meter_values()
        self._check_baseline_length_daily_billing_model()
        self._check_valid_days_percentage()
        self._check_valid_meter_readings_percentage()
        self._check_valid_temperature_values_percentage()
        self._check_monthly_temperature_values_percentage()
        self._check_extreme_values()
        # TODO : Maybe make these checks static? To work with the current data class
        # self._check_high_frequency_meter_values()
        # self._check_high_frequency_temperature_values()

    def check_sufficiency_reporting(self):
        self._check_no_data()
        self._check_valid_days_percentage()
        self._check_valid_temperature_values_percentage()
        self._check_monthly_temperature_values_percentage()
        # self._check_high_frequency_temperature_values()


class BillingSufficiencyCriteria(SufficiencyCriteria):
    """
    Sufficiency Criteria class for billing models - monthly / bimonthly
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_meter_data_billing_monthly(self):
        if self.data["value"].dropna().empty:
            return

        diff = list((data.index[1:] - data.index[:-1]).days)
        filter_ = pd.Series(diff + [np.nan], index=data.index)

        # CalTRACK 2.2.3.4, 2.2.3.5
        # Billing Monthly data frequency check
        data = data[(filter_ <= 35) & (filter_ >= 25)].reindex(  # keep these, inclusive
            data.index
        )

        if len(data[(filter_ > 35) | (filter_ < 25)]) > 0:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data",
                    description=(
                        "Off-cycle reads found in billing monthly data having a duration of less than 25 days"
                    ),
                    data=(data[(filter_ > 35) | (filter_ < 25)].index.to_list()),
                )
            )

    def _check_meter_data_billing_bimonthly(self):
        if self.data["value"].dropna().empty:
            return

        diff = list((data.index[1:] - data.index[:-1]).days)
        filter_ = pd.Series(diff + [np.nan], index=data.index)

        # CalTRACK 2.2.3.4, 2.2.3.5
        data = data[(filter_ <= 70) & (filter_ >= 25)].reindex(  # keep these, inclusive
            data.index
        )

        if len(data[(filter_ > 70) | (filter_ < 25)]) > 0:
            self.disqualification.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data",
                    description=(
                        "Off-cycle reads found in billing monthly data having a duration of less than 25 days"
                    ),
                    data=(data[(filter_ > 70) | (filter_ < 25)].index.to_list()),
                )
            )

    def _check_estimated_meter_values(self):
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
        self._check_no_data()
        self._check_negative_meter_values()
        # if self.median_granularity == "billing_monthly":
        #     self._check_meter_data_billing_monthly()
        # else :
        #     self._check_meter_data_billing_bimonthly()
        self._check_baseline_length_daily_billing_model()
        self._check_valid_days_percentage()
        self._check_valid_meter_readings_percentage()
        self._check_valid_temperature_values_percentage()
        self._check_monthly_temperature_values_percentage()
        self._check_extreme_values()
        self._check_estimated_meter_values()
        # self._check_high_frequency_temperature_values()

    def check_sufficiency_reporting(self):
        self._check_no_data()
        self._check_valid_days_percentage()
        self._check_valid_temperature_values_percentage()
        self._check_monthly_temperature_values_percentage()
        # self._check_high_frequency_temperature_values()
