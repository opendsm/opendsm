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



def lowercase_keys(value: Any) -> Any:
    """Return a copy of a dict with every string key lowercased and stripped, nested dicts included.

    Values that are not dicts are returned as given, and lists are not descended into.

    Raises:
        ValueError: If two keys of one dict normalize to the same key.
    """
    if not isinstance(value, dict):
        return value

    lowered = {}
    for key, item in value.items():
        if isinstance(key, str):
            key = key.lower().strip()

        if key in lowered:
            raise ValueError(f"settings key '{key}' is given more than once in different spellings")

        lowered[key] = lowercase_keys(item)

    return lowered


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
        return lowercase_keys(values)

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


def _tiered_field(developer: bool, *args, **kwargs):
    extra = {**(kwargs.pop("json_schema_extra", None) or {}), "developer": developer}
    field = pydantic.Field(*args, json_schema_extra=extra, **kwargs)

    return field


def UserField(*args, **kwargs):
    """Return a pydantic ``Field`` that a deviation report never lists, nor recurses into."""
    return _tiered_field(False, *args, **kwargs)


def DeveloperField(*args, **kwargs):
    """Return a pydantic ``Field`` that a deviation report lists or recurses into."""
    return _tiered_field(True, *args, **kwargs)


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


def settings_deviation_report(settings: BaseSettings, reference: BaseSettings) -> dict:
    """Report the developer-tier settings that differ from a reference.

    Args:
        settings: The settings to judge.
        reference: The settings they are judged against.

    Returns:
        A dictionary keyed by dotted field path, each entry holding the field's ``value``
        and the reference's ``default``, both converted to JSON-compatible Python. Empty
        when the two agree on every developer-tier field.
    """
    report = {
        path: {"value": value, "default": default}
        for path, value, default in _collect_deviations(settings, reference, "")
    }

    return report
