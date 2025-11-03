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
from typing import Optional

import pydantic

from opendsm.common.base_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for random sampling"""
    
    """number meters to randomly sample from comparison pool"""
    n_meters_total: Optional[int] = pydantic.Field(
        default=None, 
        validate_default=True,
    )

    """number of meters to randomly sample per treatment"""
    n_meters_per_treatment: Optional[int] = pydantic.Field(
        default=4, 
        validate_default=True,
    )

    seed: Optional[int] = pydantic.Field(
        default=None, 
        validate_default=True,
    )

    """Check if valid settings"""
    @pydantic.model_validator(mode="after")
    def _check_n_meters_choice(self):
        if self.n_meters_total is None and self.n_meters_per_treatment is None:
            raise ValueError("`n_meters_total` or `n_meters_per_treatment` must be defined")
        
        elif self.n_meters_total is not None and self.n_meters_per_treatment is not None:
            raise ValueError("`n_meters_total` and `n_meters_per_treatment` cannot be defined together")
        
        elif self.n_meters_total is not None and self.n_meters_total < 1:
            raise ValueError("`n_meters_total` must be greater than or equal to 1")

        elif self.n_meters_per_treatment is not None and self.n_meters_per_treatment < 1:
            raise ValueError("`n_meters_per_treatment` must be greater than or equal to 1")

        return self
    

if __name__ == "__main__":
    s = Settings()

    print(s.model_dump_json())
