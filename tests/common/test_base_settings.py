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

import pydantic
import pytest

from typing import Optional

from opendsm.common.base_settings import (
    BaseSettings,
    DeveloperField,
    UserField,
    lowercase_keys,
    settings_deviation_report,
)
from opendsm.eemeter.models.billing.settings import BillingSettings
from opendsm.eemeter.models.daily.utilities.opt_settings import OptimizationSettings
from opendsm.eemeter.models.daily.utilities.settings import DailySettings, Season_Definition
from opendsm.eemeter.models.hourly.settings import (
    BaseHourlySettings,
    HourlyNonSolarSettings,
    HourlySolarSettings,
)



class _Block(BaseSettings):
    multiplier: int = DeveloperField(default=1)
    threshold: int = DeveloperField(default=10)


class _Outer(BaseSettings):
    user_value: int = UserField(default=5)
    developer_value: int = DeveloperField(default=7)
    bare_value: int = pydantic.Field(default=3)
    block: _Block = DeveloperField(default_factory=_Block)
    optional_block: Optional[_Block] = DeveloperField(default=None)


_MODEL_SETTINGS_MODULES = (
    "opendsm.eemeter.models.daily.utilities.settings",
    "opendsm.eemeter.models.billing.settings",
    "opendsm.eemeter.models.hourly.settings",
)


def _mismarked_field_paths(settings: BaseSettings, prefix: str = "") -> list[str]:
    """Return the dotted paths of unmarked fields and of user-tier settings containers."""
    mismarked = []
    for name, field in type(settings).model_fields.items():
        path = f"{prefix}{name}"
        extra = field.json_schema_extra or {}
        value = getattr(settings, name)
        if "developer" not in extra:
            mismarked.append(f"{path} (unmarked)")
        elif isinstance(value, BaseSettings) and not extra["developer"]:
            mismarked.append(f"{path} (user-tier container)")

        if isinstance(value, BaseSettings) and type(value).__module__ in _MODEL_SETTINGS_MODULES:
            mismarked.extend(_mismarked_field_paths(value, f"{path}."))

    return mismarked


def test_report_is_empty_when_settings_match_the_reference():
    report = settings_deviation_report(_Outer(), _Outer())

    assert report == {}, f"identical settings must report no deviations, got {report}"


def test_report_keys_each_deviation_by_path_with_value_and_default():
    report = settings_deviation_report(_Outer(developer_value=99, block=_Block(multiplier=2)), _Outer())

    assert report == {
        "developer_value": {"value": 99, "default": 7},
        "block.multiplier": {"value": 2, "default": 1},
    }, f"expected both deviations keyed by path, got {report}"


def test_report_omits_user_tier_change():
    report = settings_deviation_report(_Outer(user_value=1), _Outer())

    assert report == {}, f"user-tier changes must not be reported, got {report}"


def test_report_treats_a_field_with_no_marker_as_developer_tier():
    report = settings_deviation_report(_Outer(bare_value=42), _Outer())

    assert report == {"bare_value": {"value": 42, "default": 3}}, (
        f"a field with no developer marker should default to developer-tier, got {report}"
    )


def test_report_lists_optional_block_none_versus_instance_as_one_entry():
    report = settings_deviation_report(_Outer(optional_block=_Block()), _Outer())

    expected = {"optional_block": {"value": {"multiplier": 1, "threshold": 10}, "default": None}}
    assert report == expected, f"expected a single entry at the block's path, got {report}"


def test_tiered_field_keeps_caller_schema_extra_beside_the_marker():
    field = UserField(default=1, json_schema_extra={"units": "kWh"})

    assert field.json_schema_extra == {"units": "kWh", "developer": False}, (
        f"expected the caller's extra merged with the marker, got {field.json_schema_extra}"
    )


@pytest.mark.parametrize(
    "settings",
    [
        DailySettings(),
        DailySettings(preset="legacy"),
        BillingSettings(),
        BaseHourlySettings(),
        HourlySolarSettings(),
        HourlyNonSolarSettings(),
        OptimizationSettings(),
    ],
    ids=[
        "daily",
        "daily_legacy",
        "billing",
        "base_hourly",
        "hourly_solar",
        "hourly_nonsolar",
        "optimization",
    ],
)
def test_every_model_settings_field_carries_a_tier_marker(settings):
    """Every field of a settings class in the daily, billing and hourly settings modules has a
    tier and their settings containers are developer-tier; containers from other modules, such
    as the clustering settings, keep the unmarked developer-tier default."""
    mismarked = _mismarked_field_paths(settings)

    assert mismarked == [], f"fields needing UserField/DeveloperField: {mismarked}"


def test_daily_season_and_weekday_changes_report_no_deviation():
    """Season and weekday leaves are user-tier inside developer-tier containers."""
    season = Season_Definition(january="summer")
    weekday_weekend = {"saturday": "weekday"}
    settings = DailySettings(season=season, weekday_weekend=weekday_weekend)

    report = settings_deviation_report(settings, DailySettings())

    assert report == {}, f"season and weekday leaves must not be reported, got {report}"


def test_lowercase_keys_lowercases_and_strips_keys_at_every_level():
    """String keys are lowercased and stripped at every level; values and other keys are kept."""
    value = {" Threshold ": 1, "BLOCK": {"THRESHOLD": "Kept", 3: None}}

    lowered = lowercase_keys(value)

    expected = {"threshold": 1, "block": {"threshold": "Kept", 3: None}}
    assert lowered == expected, f"expected {expected}, got {lowered}"


@pytest.mark.parametrize(
    "value",
    [
        {"threshold": 1, "THRESHOLD": 2},
        {"block": {"threshold": 1, " Threshold": 2}},
    ],
    ids=["top_level", "nested"],
)
def test_lowercase_keys_raises_naming_a_key_given_in_two_spellings(value):
    """Two keys of one dict that normalize to the same key raise ValueError naming it."""
    with pytest.raises(ValueError, match="'threshold'"):
        lowercase_keys(value)


def test_settings_accept_uppercase_keys_including_a_nested_block():
    """BaseSettings construction normalizes keys of the top level and of nested blocks."""
    settings = _Outer(**{"DEVELOPER_VALUE": 8, "Block": {"MULTIPLIER": 2}})

    assert settings.developer_value == 8
    assert settings.block.multiplier == 2
    assert settings.block.threshold == 10
