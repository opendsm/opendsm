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

from enum import Enum

from opendsm.common.const import (
    default_season_def,
    default_weekday_weekend_def,
)


class DistanceMetric(str, Enum):
    """
    what distance method to use
    """

    EUCLIDEAN = "euclidean"
    SEUCLIDEAN = "seuclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class AggType(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


"""data_settings constants"""


class LoadshapeType(str, Enum):
    OBSERVED = "observed"
    MODELED = "modeled"
    ERROR = "error"
    MODEL_ERROR = "error"  # an alias for ERROR


class TimePeriod(str, Enum):
    HOUR = "hour"
    DAY_OF_WEEK = "day_of_week"
    HOURLY_DAY_OF_WEEK = "hourly_day_of_week"
    WEEKDAY_WEEKEND = "weekday_weekend"
    HOURLY_WEEKDAY_WEEKEND = "hourly_weekday_weekend"
    MONTH = "month"
    HOURLY_MONTH = "hourly_month"
    SEASONAL_DAY_OF_WEEK = "seasonal_day_of_week"
    SEASONAL_HOURLY_DAY_OF_WEEK = "seasonal_hourly_day_of_week"
    SEASONAL_WEEKDAY_WEEKEND = "seasonal_weekday_weekend"
    SEASONAL_HOURLY_WEEKDAY_WEEKEND = "seasonal_hourly_weekday_weekend"


datetime_types = ["datetime", "datetime64", "datetime64[ns]", "datetimetz"]


season_num = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


weekday_num = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

max_data_interval_minutes_per_time_period = {
    # All the values are in minutes
    "hour": 60,
    "month": 60 * 24 * 28, # this is not used since we can have a different day per month
    "hourly_month": 60,
    "day_of_week": 60 * 24 * 7,
    "hourly_day_of_week": 60,
    "weekday_weekend":  60 * 24 * 7,
    "hourly_weekday_weekend": 60,
    "seasonal_day_of_week": 60 * 24 * 7,
    "seasonal_hourly_day_of_week": 60,
    "seasonal_weekday_weekend": 60 * 24 * 7,
    "seasonal_hourly_weekday_weekend": 60,
}

"""
    The order is the loadshape time layout order, outermost first: Data_Settings.time_layout
    and the cell keys in Data._convert_timeseries_to_loadshape follow it, so for
    seasonal_hourly_day_of_week the times run by season, then day_of_week, then hour.
"""
unique_time_periods = [
    "season",
    "month",
    "day_of_week",
    "weekday_weekend",
    "hour",
]
