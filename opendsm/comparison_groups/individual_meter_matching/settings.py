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

import opendsm.comparison_groups.common.const as _const
from opendsm.common.base_settings import BaseSettings

from enum import Enum
from typing import Optional


class SelectionMethod(str, Enum):
    MINIMIZE_METER_DISTANCE = "minimize_meter_distance"
    MINIMIZE_LOADSHAPE_DISTANCE = "minimize_loadshape_distance"


class Settings(BaseSettings):
    """Settings for individual meter matching"""

    """distance metric to determine best comparison pool matches"""
    distance_metric: _const.DistanceMetric = pydantic.Field(
        default=_const.DistanceMetric.EUCLIDEAN, 
        validate_default=True,
    )

    """selection method for comparison group matching"""
    selection_method: SelectionMethod = pydantic.Field(
        default=SelectionMethod.MINIMIZE_METER_DISTANCE, 
        validate_default=True,
    )

    """number of comparison pool matches to each treatment meter"""
    n_matches_per_treatment: int = pydantic.Field(
        default=4, 
        ge=1, 
        validate_default=True,
    )
    
    """number of treatments to be calculated per chunk to prevent memory issues"""
    n_treatments_per_chunk: int = pydantic.Field(
        default=10000, 
        ge=1, 
        validate_default=True,
    )
    
    """allow duplicate matches in comparison group"""
    allow_duplicate_matches: bool = pydantic.Field(
        default=False, 
        validate_default=True,
    )
    
    """The maximum distance that a comparison group match can have with a given
       treatment meter. These meters are filtered out after all matching has completed."""
    max_distance_threshold: Optional[float] = pydantic.Field(
        default=None, 
        validate_default=True,
    )

    """Check if valid settings for treatment meter match loss"""
    @pydantic.model_validator(mode="after")
    def _check_allow_duplicates(self):
        if self.allow_duplicate_matches:
            if self.selection_method != SelectionMethod.MINIMIZE_METER_DISTANCE:
                distance = SelectionMethod.MINIMIZE_METER_DISTANCE.value
                raise ValueError(f"If `allow_duplicate_matches` is True then `selection_method` must be '{distance}'")

        return self
    

if __name__ == "__main__":
    s = Settings()

    print(s.model_dump_json())
