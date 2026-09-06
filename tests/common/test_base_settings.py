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

from typing import Optional

from opendsm.common.base_settings import BaseSettings, CustomField, settings_deviations



class _Block(BaseSettings):
    multiplier: int = CustomField(default=1, developer=True)
    threshold: int = CustomField(default=10, developer=True)


class _Outer(BaseSettings):
    user_value: int = CustomField(default=5, developer=False)
    developer_value: int = CustomField(default=7, developer=True)
    bare_value: int = pydantic.Field(default=3)
    block: _Block = CustomField(default_factory=_Block, developer=True)
    optional_block: Optional[_Block] = CustomField(default=None, developer=True)


def test_defaults_against_defaults_returns_no_deviations():
    settings = _Outer()
    reference = _Outer()

    deviations = settings_deviations(settings, reference)

    assert deviations == [], f"expected no deviations between identical defaults, got {deviations}"


def test_developer_tier_change_is_listed_with_path_value_and_default():
    settings = _Outer(developer_value=99)
    reference = _Outer()

    deviations = settings_deviations(settings, reference)

    assert deviations == [("developer_value", 99, 7)], (
        f"expected exactly one deviation for developer_value, got {deviations}"
    )


def test_nested_block_change_carries_dotted_path():
    settings = _Outer(block=_Block(multiplier=2))
    reference = _Outer()

    deviations = settings_deviations(settings, reference)

    assert deviations == [("block.multiplier", 2, 1)], (
        f"expected a dotted path for the nested change, got {deviations}"
    )


def test_user_tier_change_is_not_listed():
    settings = _Outer(user_value=1)
    reference = _Outer()

    deviations = settings_deviations(settings, reference)

    assert deviations == [], f"user-tier changes must not be reported, got {deviations}"


def test_bare_field_with_no_marker_is_treated_as_developer_tier():
    settings = _Outer(bare_value=42)
    reference = _Outer()

    deviations = settings_deviations(settings, reference)

    assert deviations == [("bare_value", 42, 3)], (
        f"a field with no developer marker should default to developer-tier, got {deviations}"
    )


def test_optional_block_none_versus_instance_is_a_single_entry():
    settings = _Outer(optional_block=_Block())
    reference = _Outer()

    deviations = settings_deviations(settings, reference)

    assert deviations == [("optional_block", {"multiplier": 1, "threshold": 10}, None)], (
        f"expected a single entry at the block's path, got {deviations}"
    )
