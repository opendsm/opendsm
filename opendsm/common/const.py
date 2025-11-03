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


class CAlgoChoice(str, Enum):
    IQR_LEGACY = "iqr_legacy"
    IQR = "iqr"
    MAD = "mad"
    STDEV = "stdev"


class TutorialDataChoice(str, Enum):
    """
    Options for the tutorial data to load.
    """

    FEATURES = "features"
    SEASONAL_HOUR_DAY_WEEK_LOADSHAPE = "seasonal_hourly_day_of_week_loadshape".replace(
        "_", ""
    )
    SEASONAL_DAY_WEEK_LOADSHAPE = "seasonal_day_of_week_loadshape".replace("_", "")
    MONTH_LOADSHAPE = "month_loadshape".replace("_", "")
    HOURLY_COMPARISON_GROUP_DATA = "hourly_comparison_group_data".replace("_", "")
    HOURLY_TREATMENT_DATA = "hourly_treatment_data".replace("_", "")
    DAILY_COMPARISON_GROUP_DATA = "daily_comparison_group_data".replace("_", "")
    DAILY_TREATMENT_DATA = "daily_treatment_data".replace("_", "")
    MONTHLY_COMPARISON_GROUP_DATA = "monthly_comparison_group_data".replace("_", "")
    MONTHLY_TREATMENT_DATA = "monthly_treatment_data".replace("_", "")