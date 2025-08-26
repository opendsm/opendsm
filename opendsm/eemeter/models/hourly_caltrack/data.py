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

from typing import Optional, Union

import numpy as np
import pandas as pd

from opendsm.eemeter.common.data_processor_utilities import compute_minimum_granularity
from opendsm.eemeter.common.features import compute_temperature_features, merge_features
from opendsm.eemeter.models.hourly_caltrack.usage_per_day import (
    caltrack_sufficiency_criteria,
)


class HourlyReportingData:
    def __init__(self, df: pd.DataFrame, is_electricity_data: bool):
        if "observed" not in df.columns:
            df["observed"] = np.nan

        if is_electricity_data:
            df.loc[df["observed"] == 0, "observed"] = np.nan

        df = self._correct_frequency(df)

        self.df = df
        self.warnings = []
        self.disqualification = []

    def _correct_frequency(self, df: pd.DataFrame):
        meter = df["observed"]
        temp = df["temperature"]

        # unknown for weirdly large frequencies. Anything higher frequency than hourly frequency still comes up as hourly
        min_granularity = compute_minimum_granularity(meter.dropna().index, "unknown")

        if meter.index.inferred_freq is None and min_granularity != "hourly":
            raise ValueError(
                f"Meter Data must be atleast hourly, but is {min_granularity}."
            )
        else:
            # TODO : Add the high frequency check for meter data
            meter = meter.resample("h").sum(min_count=1)
            meter.index.freq = "h"

        # TODO : Add the high frequency check for temperature data and add NaNs
        temp = temp.resample("h").mean()
        temp.index.freq = "h"

        return merge_features([meter, temp], keep_partial_nan_rows=True)

    @classmethod
    def from_series(
        cls,
        meter_data: Optional[pd.Series],
        temperature_data: pd.Series,
        is_electricity_data: bool,
    ):
        # TODO verify
        if meter_data is None:
            meter_data = temperature_data.copy().rename("observed") * np.nan
        df = merge_features([meter_data, temperature_data], keep_partial_nan_rows=True)
        df = df.rename(
            {
                df.columns[0]: "observed",
                df.columns[1]: "temperature",
            },
            axis=1,
        )
        return cls(df, is_electricity_data)


class HourlyBaselineData(HourlyReportingData):
    def __init__(self, df: pd.DataFrame, is_electricity_data: bool):
        if is_electricity_data:
            df.loc[df["observed"] == 0, "observed"] = np.nan

        df = self._correct_frequency(df)

        self.df = df
        self.warnings = self._check_data_sufficiency()
        self.disqualification = []

    def _check_data_sufficiency(self):
        meter = self.df["observed"].rename("meter_value")
        temp = self.df["temperature"]

        temperature_features = compute_temperature_features(
            meter.index,
            temp,
            data_quality=True,
        )

        sufficiency_df = merge_features([meter, temperature_features])
        sufficiency = caltrack_sufficiency_criteria(
            sufficiency_df, requested_start=None, requested_end=None
        )
        return sufficiency.warnings

    @classmethod
    def from_series(
        cls,
        meter_data: Union[pd.Series, pd.DataFrame],
        temperature_data: Union[pd.Series, pd.DataFrame],
        is_electricity_data: bool,
    ):
        if isinstance(meter_data, pd.Series):
            meter_data = meter_data.to_frame()
        if isinstance(temperature_data, pd.Series):
            temperature_data = temperature_data.to_frame()
        meter_data = meter_data.rename(columns={meter_data.columns[0]: "observed"})
        temperature_data = temperature_data.rename(
            columns={temperature_data.columns[0]: "temperature"}
        )
        temperature_data.index = temperature_data.index.tz_convert(
            meter_data.index.tzinfo
        )
        df = pd.concat([meter_data, temperature_data], axis=1).dropna()
        return cls(df, is_electricity_data)
