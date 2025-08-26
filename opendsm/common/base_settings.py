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

from typing import Any


class BaseSettings(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        frozen = True,
        arbitrary_types_allowed=True,
        str_to_lower = True,
        str_strip_whitespace = True,
    )

    """Make all property keys lowercase and strip whitespace"""
    @pydantic.model_validator(mode="before")
    def __lowercase_property_keys__(cls, values: Any) -> Any:
        def __lower__(value: Any) -> Any:
            if isinstance(value, dict):
                return {k.lower().strip() if isinstance(k, str) else k: __lower__(v) for k, v in value.items()}
            return value

        return __lower__(values)

    """Make all property values lowercase and strip whitespace before validation"""
    @pydantic.field_validator("*", mode="before")
    def lowercase_values(cls, v):
        if isinstance(v, str):
            return v.lower().strip()
        return v


class MutableBaseSettings(BaseSettings):
    model_config = pydantic.ConfigDict(
        frozen = False,
        arbitrary_types_allowed=True,
        str_to_lower = True,
        str_strip_whitespace = True,
    )


# add developer field to pydantic Field
def CustomField(developer=False, *args, **kwargs):
    field = pydantic.Field(json_schema_extra={"developer": developer}, *args, **kwargs)
    return field