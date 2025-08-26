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

import datetime

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd

from opendsm.eemeter.common.data_processor_utilities import (
    as_freq,
    clean_billing_daily_data,
    compute_minimum_granularity,
)
from opendsm.eemeter.common.features import compute_temperature_features
from opendsm.eemeter.common.data_settings import BillingDataSettings
from opendsm.eemeter.common.sufficiency_criteria import BillingSufficiencyCriteria
from opendsm.eemeter.models.daily.data import _DailyData

from opendsm.eemeter.common.warnings import EEMeterWarning



"""TODO there is still a ton of unecessarily duplicated code between billing+daily.
    we should be able to perform a few transforms within the billing baseclass, and then call super() for the rest

    unsure whether we should inherit from the public classes because we'll have to take care to use type(data)
    instead of isinstance(data,  _) when doing the checks in the model/wrapper to avoid unintentionally allowing a mix of data/model type
"""


class _BillingData(_DailyData):
    """Baseline data processor for billing data.

    2.2.3.4. Off-cycle reads (spanning less than 25 days) should be dropped from analysis.
    These readings typically occur due to meter reading problems or changes in occupancy.

    2.2.3.5. For pseudo-monthly billing cycles, periods spanning more than 35 days should be dropped from analysis.
    For bi-monthly billing cycles, periods spanning more than 70 days should be dropped from the analysis.
    """

    _settings_class = BillingDataSettings

    def _compute_meter_value_df(self, df: pd.DataFrame):
        """
        Computes the meter value DataFrame by cleaning and processing the observed meter data.
        1. The minimum granularity is computed from the non null rows. If the billing cycle is mixed between monthly and bimonthly, then the minimum granularity is bimonthly
        2. The meter data is cleaned and downsampled/upsampled into the correct frequency using clean_billing_daily_data()
        3. Add missing days as NaN by merging with a full year daily index.

        Parameters
        ----------

            df (pd.DataFrame): The DataFrame containing the observed meter data.

        Returns
        -------
            pd.DataFrame: The cleaned and processed meter value DataFrame.
        """
        meter_series_full = df["observed"]
        meter_series = meter_series_full.dropna()
        if meter_series.empty:
            return meter_series_full.resample("D").first().to_frame()

        start_date = meter_series_full.index.min()
        end_date = meter_series_full.index.max().replace(
            hour=meter_series.index[-1].hour
        )  # assume final period ends on same hour

        # ensure we adjust backwards to normalize hour, never adding time
        if end_date > meter_series_full.index.max():
            end_date = end_date - pd.Timedelta(days=1)

        min_granularity = compute_minimum_granularity(
            meter_series.index, default_granularity="billing_bimonthly"
        )

        # Ensure higher frequency data is aggregated to the monthly model
        if not min_granularity.startswith("billing"):
            # MS is so that the date for Month Start
            meter_series = meter_series.resample("MS").sum(min_count=1)
            # normalize to midnight since we're picking an arbitrary day to represent period start anyway
            end_date = end_date.normalize()
            self.warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.sufficiency_criteria.inferior_model_usage",
                    description=(
                        "Daily data is provided but the model used is monthly. Are you sure this is the intended model?"
                    ),
                    data={},
                )
            )
            min_granularity = "billing_monthly"

        # Adjust index to follow final nan convention--without this, final period will be short one day
        meter_series[end_date + pd.Timedelta(days=1)] = np.nan

        # This checks for offcycle reads. That is a disqualification if the billing cycle is less than 25 days
        meter_value_df = clean_billing_daily_data(
            meter_series.to_frame("value"), min_granularity, self.disqualification
        )

        # Spread billing data to daily
        meter_value_df = as_freq(meter_value_df["value"], "D").to_frame("value")
        meter_value_df = meter_value_df[:-1]
        meter_value_df = meter_value_df.rename(columns={"value": "observed"})

        # This will ensure that the missing days are kept in the dataframe
        # Create an index with all the days from the start and end date of 'meter_value_df'
        if len(meter_value_df) > 0:
            all_days_index = pd.date_range(
                start=start_date,
                end=end_date,
                freq="D",
                tz=df.index.tz,
                ambiguous=True,
                nonexistent="shift_forward",
            )
            all_days_df = pd.DataFrame(index=all_days_index)
            meter_value_df = meter_value_df.merge(
                all_days_df, left_index=True, right_index=True, how="outer"
            )

        return meter_value_df

    def _compute_temperature_features(
        self, df: pd.DataFrame, meter_index: pd.DatetimeIndex
    ):
        """
        Compute temperature features for the given DataFrame and meter index.
        1. The frequency of the temperature data is inferred and set to hourly if not already. If frequency is not inferred or its lower than hourly, a warning is added.
        2. The temperature data is downsampled/upsampled into the daily frequency using as_freq()
        3. High frequency temperature data is checked for missing values and a warning is added if more than 50% of the data is missing, and those rows are set to NaN.
        4. If frequency was already hourly, compute_temperature_features() is used to recompute the temperature to match with the meter index.

        Parameters
        ----------

            df (pd.DataFrame): The DataFrame containing temperature data.
            meter_index (pd.DatetimeIndex): The meter index.

        Returns
        -------

            pd.Series: The computed temperature values.
            pd.DataFrame: The computed temperature features.
        """
        temp_series = df["temperature"]
        temp_series.index.freq = temp_series.index.inferred_freq
        if temp_series.index.freq != "h":
            if (
                temp_series.index.freq is None
                or isinstance(temp_series.index.freq, MonthEnd)
                or isinstance(temp_series.index.freq, MonthBegin)
                or temp_series.index.freq > pd.Timedelta(hours=1)
            ):
                # Add warning for frequencies longer than 1 hour
                self.warnings.append(
                    EEMeterWarning(
                        qualified_name="eemeter.sufficiency_criteria.unable_to_confirm_daily_temperature_sufficiency",
                        description=(
                            "Cannot confirm that pre-aggregated temperature data had sufficient hours kept"
                        ),
                        data={},
                    )
                )
            # TODO consider disallowing this until a later patch
            if temp_series.index.freq != "D":
                # Downsample / Upsample the temperature data to daily
                temperature_features = as_freq(
                    temp_series, "D", series_type="instantaneous", include_coverage=True
                )
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
                temperature_features.loc[
                    temperature_features.coverage > 0.5, "value"
                ] = (
                    temperature_features[temperature_features.coverage > 0.5].value
                    / temperature_features[temperature_features.coverage > 0.5].coverage
                )

                temperature_features = (
                    temperature_features[temperature_features.coverage > 0.5]
                    .reindex(temperature_features.index)[["value"]]
                    .rename(columns={"value": "temperature_mean"})
                )

                if "coverage" in temperature_features.columns:
                    temperature_features = temperature_features.drop(
                        columns=["coverage"]
                    )
            else:
                temperature_features = temp_series.to_frame(name="temperature_mean")

            temperature_features["temperature_null"] = temp_series.isnull().astype(int)
            temperature_features["temperature_not_null"] = temp_series.notnull().astype(
                int
            )
            temperature_features["n_days_kept"] = 0  # unused
            temperature_features["n_days_dropped"] = 0  # unused
        else:
            if not meter_index.empty:
                buffer_idx = meter_index.max() + pd.Timedelta(days=1)
                meter_index = meter_index.union([buffer_idx])

            temperature_features = compute_temperature_features(
                meter_index,
                temp_series,
                data_quality=True,
            )
            temperature_features = temperature_features[:-1]
            # Only check for high frequency temperature data if it exists
            # TODO this check causes weird behavior with very sparse temp data.
            # will still get DQ'd, but final df receives non-nan temperatures
            median_samples = (
                temperature_features.temperature_not_null
                + temperature_features.temperature_null
            ).median()
            if median_samples > 1:
                invalid_temperature_rows = (
                    temperature_features.temperature_not_null
                    / (
                        temperature_features.temperature_not_null
                        + temperature_features.temperature_null
                    )
                ) <= 0.5
                # check against median in case start/end of data does not cover a full period
                invalid_temperature_rows |= (
                    temperature_features.temperature_not_null <= median_samples * 0.5
                )

                if invalid_temperature_rows.any():
                    self.warnings.append(
                        EEMeterWarning(
                            qualified_name="eemeter.sufficiency_criteria.missing_high_frequency_temperature_data",
                            description=(
                                "More than 50% of the high frequency temperature data is missing."
                            ),
                            data=[
                                timestamp.isoformat()
                                for timestamp in invalid_temperature_rows.index
                            ],
                        )
                    )
                    temperature_features.loc[
                        invalid_temperature_rows, "temperature_mean"
                    ] = np.nan

        temp = temperature_features["temperature_mean"].rename("temperature")
        features = temperature_features.drop(columns=["temperature_mean"])
        return temp, features

    # TODO: DELETE THIS after making real billing data class
    @property
    def billing_df(self) -> pd.DataFrame | None:
        """Get the corrected input data stored in the class. The actual dataframe is immutable, this returns a copy."""

        df = self._df.copy()

        # find indices where observed changes from prior
        observed_change = df["observed"].diff()
        observed_change = observed_change[observed_change != 0].index
        obs_change_idx = df.index.get_indexer(observed_change)
        obs_change_idx = np.append(obs_change_idx, len(df))
        obs_change_idx = np.delete(obs_change_idx, np.where(np.diff(obs_change_idx) < 15)[0])

        if obs_change_idx[0] != 0:
            obs_change_idx = np.insert(obs_change_idx, 0, 0)

        # create vector where value increases at each observed change
        group = []
        for i in range(1, len(obs_change_idx)):
            idx_range = obs_change_idx[i] - obs_change_idx[i-1]

            group.extend([i] * idx_range)

        df["group"] = group

        # get median delta

        # get first datetime, average temperature, sum of observed for each group and make new df
        df_temp = df.reset_index()
        df_temp = df_temp.rename(columns={"index": "datetime"})

        df_grouped = df_temp.groupby("group").agg({
            "datetime": "first",
            "season": "first",
            "weekday_weekend": "first",
            "temperature": "mean",
            "observed": "mean",
        }).set_index("datetime")

        # create days column for number of days between current and previous index
        df_grouped["days"] = df_grouped.index.to_series().diff().dt.days

        df_grouped = df_grouped.dropna()

        # create weights from days column
        df_grouped["weights"] = df_grouped["days"] / df_grouped["days"].sum()

        df_grouped = df_grouped.drop(columns=["days"])

        if self._df is None:
            return None
        else:
            return df_grouped.copy()


class BillingBaselineData(_BillingData):
    """
    Data class to represent Billing Baseline Data.

    Only baseline data should go into the dataframe input, no blackout data should be input.
    Checks sufficiency for the data provided as input depending on OpenEEMeter specifications and populates disqualifications and warnings based on it.

    Billing data should have an extra month's data appended at the to denote end of period. (Do not append NaN, any other value would work.)

    Args:
        df (DataFrame): A dataframe having a datetime index or a datetime column with the timezone also being set.
            It also requires 2 more columns - 'observed' for meter data, and 'temperature' for temperature data.
            The temperature column should have values in Fahrenheit. Please convert your temperatures accordingly.

        is_electricity_data (bool): Flag to ascertain if this is electricity data or not. Electricity data values of 0 are set to NaN.

    Attributes:
        df (DataFrame): Immutable dataframe that contains the meter and temperature values for the baseline data period.
        disqualification (list[EEMeterWarning]): A list of serious issues with the data that can degrade the quality of the model. If you want to go ahead with building the model while ignoring them, set the ignore_disqualification = True flag in the model. By default disqualifications are not ignored.
        warnings (list[EEMeterWarning]): A list of ssues with the data, but none that will severely reduce the quality of the model built.
    """

    def _check_data_sufficiency(self, sufficiency_df):
        """
        Private method which checks the sufficiency of the data for billing baseline calculations using the predefined OpenEEMeter sufficiency criteria.

        Args:
            sufficiency_df (pandas.DataFrame): DataFrame containing the data for sufficiency check. Should have features such as -
            temperature_null: number of temperature null periods in each aggregation step
            temperature_not_null: number of temperature non null periods in each aggregation step

        Returns:
            disqualification (List): List of disqualifications
            warnings (list): List of warnings

        """
        bsc = BillingSufficiencyCriteria(
            data=sufficiency_df, 
            is_electricity_data=self.is_electricity_data,
            is_reporting_data=False,
            settings=self.settings.sufficiency,
        )
        bsc.check_sufficiency_baseline()
        disqualification = bsc.disqualification
        warnings = bsc.warnings

        # _, disqualification, warnings = sufficiency_criteria_baseline(
        #     sufficiency_df,
        #     is_reporting_data=False,
        #     is_electricity_data=self.is_electricity_data,
        # )
        return disqualification, warnings


class BillingReportingData(_BillingData):
    """Data class to represent Billing Reporting Data.

    Only reporting data should go into the dataframe input, no blackout data should be input.
    Checks sufficiency for the data provided as input depending on OpenEEMeter specifications and populates disqualifications and warnings based on it.

    Meter data input is optional for the reporting class.

    Args:
        df (DataFrame): A dataframe having a datetime index or a datetime column with the timezone also being set.
            It also requires 2 more columns - 'observed' for meter data, and 'temperature' for temperature data.
            The temperature column should have values in Fahrenheit. Please convert your temperatures accordingly.

        is_electricity_data (bool): Flag to ascertain if this is electricity data or not. Electricity data values of 0 are set to NaN.

    Attributes:
        df (DataFrame): Immutable dataframe that contains the meter and temperature values for the baseline data period.
        disqualification (list[EEMeterWarning]): A list of serious issues with the data that can degrade the quality of the model. If you want to go ahead with building the model while ignoring them, set the ignore_disqualification = True flag in the model. By default disqualifications are not ignored.
        warnings (list[EEMeterWarning]): A list of ssues with the data, but none that will severely reduce the quality of the model built.
    """

    def __init__(
        self,
        df: pd.DataFrame, 
        is_electricity_data: bool, 
        settings: dict | None = None
    ):
        df = df.copy()
        if "observed" not in df.columns:
            df["observed"] = np.nan

        super().__init__(df, is_electricity_data, settings=settings)

    @classmethod
    def from_series(
        cls,
        meter_data: pd.Series | pd.DataFrame | None,
        temperature_data: pd.Series | pd.DataFrame,
        is_electricity_data: bool,
        tzinfo: datetime.tzinfo | None = None,
        settings: dict | None = None,
    ):
        """Create a BillingReportingData instance from meter data and temperature data.

        Args:
            meter_data: The meter data to be used for the BillingReportingData instance.
            temperature_data: The temperature data to be used for the BillingReportingData instance.
            is_electricity_data: Flag indicating whether the meter data represents electricity data.
            tzinfo: Timezone information to be used for the meter data.

        Returns:
            An instance of the Data class.
        """
        if tzinfo and meter_data is not None:
            raise ValueError(
                "When passing meter data to BillingReportingData, convert its DatetimeIndex to local timezone first; `tzinfo` param should only be used in the absence of reporting meter data."
            )
        if is_electricity_data is None and meter_data is not None:
            raise ValueError(
                "Must specify is_electricity_data when passing meter data."
            )
        if meter_data is None:
            meter_data = pd.DataFrame(
                {"observed": np.nan}, index=temperature_data.index
            )
            if tzinfo:
                meter_data = meter_data.tz_convert(tzinfo)
        if meter_data.empty:
            raise ValueError(
                "Pass meter_data=None to explicitly create a temperature-only reporting data instance."
            )
        return super().from_series(meter_data, temperature_data, is_electricity_data, settings=settings)

    def _check_data_sufficiency(self, sufficiency_df):
        """
        Private method which checks the sufficiency of the data for billing reporting calculations using the predefined OpenEEMeter sufficiency criteria.

        Parameters
        ----------
        1. sufficiency_df (pandas.DataFrame): DataFrame containing the data for sufficiency check. Should have features such as -
            - temperature_null: number of temperature null periods in each aggregation step
            - temperature_not_null: number of temperature non null periods in each aggregation step

        Returns
        -------
            disqualification (List): List of disqualifications
            warnings (list): List of warnings

        """
        bsc = BillingSufficiencyCriteria(
            data=sufficiency_df, 
            is_electricity_data=self.is_electricity_data,
            is_reporting_data=True,
            settings=self.settings.sufficiency,
        )
        bsc.check_sufficiency_reporting()
        disqualification = bsc.disqualification
        warnings = bsc.warnings

        # _, disqualification, warnings = sufficiency_criteria_baseline(
        #     sufficiency_df,
        #     is_reporting_data=True,
        #     is_electricity_data=self.is_electricity_data,
        # )
        return disqualification, warnings
