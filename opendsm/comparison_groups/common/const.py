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
    DAY_OF_YEAR = "day_of_year"
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


default_season_def = {
    "options": ["summer", "shoulder", "winter"],
    "January": "winter",
    "February": "winter",
    "March": "shoulder",
    "April": "shoulder",
    "May": "shoulder",
    "June": "summer",
    "July": "summer",
    "August": "summer",
    "September": "summer",
    "October": "shoulder",
    "November": "winter",
    "December": "winter",
}


default_weekday_weekend_def = {
    "options": ["weekday", "weekend"],
    "Monday": "weekday",
    "Tuesday": "weekday",
    "Wednesday": "weekday",
    "Thursday": "weekday",
    "Friday": "weekday",
    "Saturday": "weekend",
    "Sunday": "weekend",
}


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

time_period_row_counts = {
    "hourly": 24,
    "month": 12,
    "hourly_month": 24 * 12,
    "day_of_week": 7,
    "day_of_year": 365,
    "hourly_day_of_week": 24 * 7,
    "weekday_weekend": 2,
    "hourly_weekday_weekend": 24 * 2,
    "seasonal_day_of_week": 3 * 7,
    "seasonal_hourly_day_of_week": 3 * 24 * 7,
    "seasonal_weekday_weekend": 3 * 2,
    "seasonal_hourly_weekday_weekend": 3 * 24 * 2,
}


min_granularity_per_time_period = {
    # All the values are in minutes
    "hourly": 60,
    "month": 60 * 24 * 28, # this is not used since we can have a different day per month
    "hourly_month": 60,
    "day_of_week": 60 * 24 * 7,
    "day_of_year": 60 * 24 * 7,
    "hourly_day_of_week": 60,
    "weekday_weekend":  60 * 24 * 7,
    "hourly_weekday_weekend": 60,
    "seasonal_day_of_week": 60 * 24 * 7,
    "seasonal_hourly_day_of_week": 60,
    "seasonal_weekday_weekend": 60 * 24 * 7,
    "seasonal_hourly_weekday_weekend": 60,
}

"""
    This list ordering is important for the groupby columns (refer _find_groupby_columns in data_processing.py)
    The sorting is done on the basis of this ordering in the final dataframe. First the dataframe is sorted by season, then by day_of_week, then by hour in the seasonal_hourly_day_of_week case.
    Similarly for other combinations.
"""
unique_time_periods = [
    "season",
    "month",
    "day_of_week",
    "day_of_year",
    "weekday_weekend",
    "hour",
]
