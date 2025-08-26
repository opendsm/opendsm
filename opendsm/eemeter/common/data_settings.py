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

import numpy as np
import pandas as pd

import pydantic
import datetime

from typing import Optional, Union

from opendsm.common.base_settings import MutableBaseSettings


# TODO: use this in future for all columns
class ColumnSufficiencySettings(MutableBaseSettings):
    min_pct_hourly_coverage: float = pydantic.Field(
        default=0.5,
        gt=0,
        le=1,
        description="Minimum percentage of hourly coverage.",
    )
    
    min_pct_daily_coverage: float = pydantic.Field(
        default=0.9,
        gt=0,
        le=1,
        description="Minimum percentage of daily coverage.",
    )
    
    min_pct_monthly_coverage: float = pydantic.Field(
        default=0.9,
        gt=0,
        le=1,
        description="Minimum percentage of monthly coverage.",
    )

    min_pct_period_coverage: float = pydantic.Field(
        default=0.9,
        gt=0,
        le=1,
        description="Minimum percentage of period coverage.",
    )


class TemperatureSufficiencySettings(ColumnSufficiencySettings):
    pass


class GhiSufficiencySettings(MutableBaseSettings):
    min_pct_monthly_coverage: float = pydantic.Field(
        default=0.9,
        gt=0,
        le=1,
        description="Minimum percentage of monthly coverage.",
    )


class ObservedSufficiencySettings(MutableBaseSettings):
    min_pct_hourly_coverage: float = pydantic.Field(
        default=0.5,
        gt=0,
        le=1,
        description="Minimum percentage of hourly coverage.",
    )
    
    min_pct_daily_coverage: float = pydantic.Field(
        default=0.9,
        gt=0,
        le=1,
        description="Minimum percentage of daily coverage.",
    )
    
    min_pct_monthly_coverage: float = pydantic.Field(
        default=0.9,
        gt=0,
        le=1,
        description="Minimum percentage of monthly coverage.",
    )


class JointSufficiencySettings(MutableBaseSettings):
    min_pct_daily_coverage: float = pydantic.Field(
        default=0.9,
        gt=0,
        le=1,
        description="Minimum percentage of daily coverage.",
    )


class BaseSufficiencySettings(MutableBaseSettings):
    requested_start: Optional[pd.Timestamp] = pydantic.Field(
        default=None,
        description="Requested start date for the data. If None, use the data start date."
    )
    
    requested_end: Optional[pd.Timestamp] = pydantic.Field(
        default=None,
        description="Requested end date for the data. If None, use the data end date."
    )

    min_baseline_length: int = pydantic.Field(
        default=np.ceil(0.9 * 365),
        ge=1,
        description="Minimum number of days in the baseline.",
    )

    max_baseline_length: int = pydantic.Field(
        default=366, # 366 for leap year
        ge=2,
        description="Maximum number of days in the baseline.",
    )

    temperature: TemperatureSufficiencySettings = pydantic.Field(
        default_factory=TemperatureSufficiencySettings,
    )

    ghi: GhiSufficiencySettings = pydantic.Field(
        default_factory=GhiSufficiencySettings,
    )

    observed: ObservedSufficiencySettings = pydantic.Field(
        default_factory=ObservedSufficiencySettings,
    )

    joint: JointSufficiencySettings = pydantic.Field(
        default_factory=JointSufficiencySettings,
    )

    @pydantic.field_validator("min_baseline_length", "max_baseline_length", mode="before")
    @classmethod
    def convert_float_to_int(cls, v):
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    @pydantic.model_validator(mode="after")
    def check_baseline_lengths(self):
        max_baseline_length = self.max_baseline_length
        min_baseline_length = self.min_baseline_length
        if max_baseline_length <= min_baseline_length:
            raise ValueError(
                f"max_baseline_length ({max_baseline_length}) must be greater than min_baseline_length ({min_baseline_length})"
            )
        
        return self


class DailyDataSufficiencySettings(BaseSufficiencySettings):
    ghi: None = None
    

class BillingDataSufficiencySettings(BaseSufficiencySettings):
    ghi: None = None

    min_days_in_period: int = pydantic.Field(
        default=25,
        ge=1,
        description="Minimum number of days in a billing period.",
    )

    max_days_in_monthly_period: int = pydantic.Field(
        default=70,
        ge=1,
        description="Maximum number of days in a billing period.",
    )

    max_days_in_bimonthly_period: int = pydantic.Field(
        default=70,
        ge=1,
        description="Maximum number of days in a billing period.",
    )

    @pydantic.field_validator("min_days_in_period", "max_days_in_monthly_period", "max_days_in_bimonthly_period", mode="before")
    @classmethod
    def convert_float_to_int(cls, v):
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    
class HourlyTemperatureSufficiencySettings(TemperatureSufficiencySettings):
    max_consecutive_hours_missing: int = pydantic.Field(
        default=6,
        ge=0,
        description="Maximum number of consecutive missing hours to declare the day as missing.",
    )

    @pydantic.field_validator("max_consecutive_hours_missing", mode="before")
    @classmethod
    def convert_float_to_int(cls, v):
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

class HourlyDataSufficiencySettings(BaseSufficiencySettings):
    temperature: HourlyTemperatureSufficiencySettings = pydantic.Field(
        default_factory=HourlyTemperatureSufficiencySettings,
    )


class BaseDataSettings(MutableBaseSettings):
    """is electricity data"""
    is_electricity_data: bool = pydantic.Field(
        default=True, # TODO: if is_electricity_data removed from data, this needs to be required
        description="Boolean flag to specify if the data is electricity data or not.",
    )

    time_zone: Optional[datetime.timezone] = pydantic.Field(
        default=None,
        description="Time zone for the data, e.g., 'America/Los_Angeles'. If None, time zone is not set."
    )

class DailyDataSettings(BaseDataSettings):
    sufficiency: DailyDataSufficiencySettings = pydantic.Field(
        default_factory=DailyDataSufficiencySettings,
    )

class BillingDataSettings(BaseDataSettings):
    sufficiency: BillingDataSufficiencySettings = pydantic.Field(
        default_factory=BillingDataSufficiencySettings,
    )

class HourlyDataSettings(BaseDataSettings):
    pv_start: Optional[Union[datetime.date, str]] = pydantic.Field(
        default=None,
        description="Date of the solar installation. If None, assume solar status is static."
    )

    sufficiency: HourlyDataSufficiencySettings = pydantic.Field(
        default_factory=HourlyDataSufficiencySettings,
    )