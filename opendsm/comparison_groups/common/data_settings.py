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

import pydantic

from typing import Optional,Union

import opendsm.comparison_groups.common.const as _const
from opendsm.common.base_settings import BaseSettings


min_data_pct = 0.8


# Note: Options list order defines how seasons will be ordered in the loadshape
class Season_Definition(BaseSettings):
    january: str = pydantic.Field(default="winter")
    february: str = pydantic.Field(default="winter")
    march: str = pydantic.Field(default="shoulder")
    april: str = pydantic.Field(default="shoulder")
    may: str = pydantic.Field(default="shoulder")
    june: str = pydantic.Field(default="summer")
    july: str = pydantic.Field(default="summer")
    august: str = pydantic.Field(default="summer")
    september: str = pydantic.Field(default="summer")
    october: str = pydantic.Field(default="shoulder")
    november: str = pydantic.Field(default="winter")
    december: str = pydantic.Field(default="winter")

    options: list[str] = pydantic.Field(default=["summer", "shoulder", "winter"])

    """Set dictionaries of seasons"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> Season_Definition:
        season_dict = {}
        for month, num in _const.season_num.items():
            val = getattr(self, month)
            if val not in self.options:
                raise ValueError(f"SeasonDefinition: {val} is not a valid option. Valid options are {self.options}")

            season_dict[num] = val
        
        self._num_dict = season_dict
        self._order = {val: i for i, val in enumerate(self.options)}

        return self


class Weekday_Weekend_Definition(BaseSettings):
    monday: str = pydantic.Field(default="weekday")
    tuesday: str = pydantic.Field(default="weekday")
    wednesday: str = pydantic.Field(default="weekday")
    thursday: str = pydantic.Field(default="weekday")
    friday: str = pydantic.Field(default="weekday")
    saturday: str = pydantic.Field(default="weekend")
    sunday: str = pydantic.Field(default="weekend")

    options: list[str] = pydantic.Field(default=["weekday", "weekend"])

    """Set dictionaries of weekday/weekend"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> Weekday_Weekend_Definition:
        weekday_dict = {}
        for day, num in _const.weekday_num.items():
            val = getattr(self, day)
            if val not in self.options:
                raise ValueError(f"WeekdayWeekendDefinition: {val} is not a valid option. Valid options are {self.options}")
            
            weekday_dict[num] = val
        
        self._num_dict = weekday_dict
        self._order = {val: i for i, val in enumerate(self.options)}

        return self
    

class Data_Settings(BaseSettings):
    """maximum number of meters to be used in the comparison pool"""
    max_pool_size: int = pydantic.Field(
        default=10000,
        ge=1,
        validate_default=True,
    )

    """aggregation type for the loadshape"""
    agg_type: Optional[_const.AggType] = pydantic.Field(
        default=_const.AggType.MEAN,
        validate_default=True,
    )
    
    """type of loadshape to be used"""
    loadshape_type: Optional[_const.LoadshapeType] = pydantic.Field(
        default=_const.LoadshapeType.MODELED, 
        validate_default=True,
    )

    """time period to be used for the loadshape"""
    time_period: Optional[_const.TimePeriod] = pydantic.Field(
        default=_const.TimePeriod.SEASONAL_HOURLY_DAY_OF_WEEK, 
        validate_default=True,
    )

    """interpolate missing values"""
    interpolate_missing: bool = pydantic.Field(
        default=True, 
        validate_default=True,
    )

    """minimum percentage of data required for a meter to be included"""
    min_data_pct_required: Optional[float] = pydantic.Field(
        default=min_data_pct, 
        validate_default=True,
    )

    @pydantic.field_validator("min_data_pct_required")
    @classmethod
    def validate_min_data_pct_required(cls, value):
        if value is None:
            pass

        elif value != min_data_pct:
            raise ValueError(f"min_data_pct_required must be {min_data_pct}")
        
        return value

    """season definition to be used for the loadshape"""
    season: Union[dict, Season_Definition] = pydantic.Field(
        default=_const.default_season_def, 
    )

    """weekday/weekend definition to be used for the loadshape"""
    weekday_weekend: Union[dict, Weekday_Weekend_Definition] = pydantic.Field(
        default=_const.default_weekday_weekend_def, 
    )

    """set season and weekday_weekend classes with given dictionaries"""
    @pydantic.model_validator(mode="after")
    def _set_nested_classes(self):
        self.model_config["frozen"] = False
        
        if isinstance(self.season, dict):
            self.season = Season_Definition(**self.season)

        if isinstance(self.weekday_weekend, dict):
            self.weekday_weekend = Weekday_Weekend_Definition(**self.weekday_weekend)

        self.model_config["frozen"] = True

        return self
    
    """validate loadshape/time series settings"""
    @pydantic.model_validator(mode="after")
    def _validate_loadshape_time_series_settings(self):
        ls_dict = {"agg_type": self.agg_type, "loadshape_type": self.loadshape_type, "time_period": self.time_period}
        is_set = {k: v is not None for k, v in ls_dict.items()}
        if any(is_set.values()):
            for k, v in is_set.items():
                if v is False:
                    raise ValueError(f"{k} must be set if any of the following are set: {list(is_set.keys())}")

        return self

    """set min_data_pct_required"""
    @pydantic.model_validator(mode="after")
    def _set_min_data_pct_on_interpolate(self):
        self.model_config["frozen"] = False

        if self.interpolate_missing:
            self.min_data_pct_required = min_data_pct
        else:
            self.min_data_pct_required = None

        self.model_config["frozen"] = True

        return self
    

if __name__ == "__main__":
    # Test SeasonDefinition
    # Note: Options list order defines how seasons will be orderd in the loadshape
    season_dict = {
        "options":  ["summer", "shoulder", "winter"],
        "January":   "winter", 
        "February":  "winter", 
        "March":     "shoulder", 
        "April":     "shoulder", 
        "May":       "shoulder", 
        "June":      "summer", 
        "July":      "summer", 
        "August":    "summer", 
        "September": "summer", 
        "October":   "shoulder", 
        "November":  "winter", 
        "December":  "winter",
        }

    # season = SeasonDefinition(**season_def)
    # print(season.model_dump_json())

    # Test WeekdayWeekendDefinition
    weekday_weekend_dict = {
        "options":  ["weekday", "weekend", "oops"],
        "Monday":    "weekday",
        "Tuesday":   "weekday",
        "Wednesday": "weekday",
        "Thursday":  "weekday",
        "Friday":    "weekend",
        "Saturday":  "weekend",
        "Sunday":    "weekday",
        }
    
    # weekday_weekend = WeekdayWeekendDefinition(**weekday_weekend_def)
    # weekday_weekend = WeekdayWeekendDefinition()
    # print(weekday_weekend.model_dump_json())

    # Test DataSettings
    settings = Data_Settings(
        agg_type="median",
        season=season_dict, 
        weekday_weekend=weekday_weekend_dict,
    )
    print(settings.model_dump_json())
    print(settings.season._num_dict)
    print(settings.season._order)