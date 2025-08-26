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


# TODO: this is copy-pasted from gridmeter branch, need to merge


"""data_settings constants"""
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
    "monday": 1,
    "tuesday": 2,
    "wednesday": 3,
    "thursday": 4,
    "friday": 5,
    "saturday": 6,
    "sunday": 7,
}