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

from pydantic_core import to_jsonable_python

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


# add developer field to pydantic Field
def CustomField(developer=True, *args, **kwargs):
    field = pydantic.Field(json_schema_extra={"developer": developer}, *args, **kwargs)
    return field


def _collect_deviations(settings: BaseSettings, reference: BaseSettings, prefix: str) -> list[tuple[str, Any, Any]]:
    deviations = []
    for name, field in type(settings).model_fields.items():
        extra = field.json_schema_extra or {}
        developer = extra.get("developer", True)
        if not developer:
            continue

        value = getattr(settings, name)
        default = getattr(reference, name)
        path = f"{prefix}{name}"

        if isinstance(value, BaseSettings) and isinstance(default, BaseSettings):
            deviations.extend(_collect_deviations(value, default, f"{path}."))
        elif value != default:
            deviations.append((path, to_jsonable_python(value), to_jsonable_python(default)))

    return deviations


def settings_deviations(settings: BaseSettings, reference: BaseSettings) -> list[tuple[str, Any, Any]]:
    """Return (path, value, default) for every developer-tier leaf that differs from the reference."""

    return _collect_deviations(settings, reference, "")


def settings_deviation_report(settings: BaseSettings, reference: BaseSettings) -> dict:
    """Report the developer-tier settings that differ from a reference.

    Args:
        settings: The settings to judge.
        reference: The settings they are judged against.

    Returns:
        A dictionary keyed by dotted field path, each entry holding the field's ``value``
        and the reference's ``default``. Empty when the two agree on every developer-tier
        field.
    """
    deviations = settings_deviations(settings, reference)
    report = {
        path: {"value": value, "default": default} for path, value, default in deviations
    }

    return report
