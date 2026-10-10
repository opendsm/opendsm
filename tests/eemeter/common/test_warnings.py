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

import json

from opendsm.common.base_settings import BaseSettings, CustomField
from opendsm.eemeter.common.warnings import EEMeterWarning, nonstandard_settings_warning




class _DummySettings(BaseSettings):
    developer_value: float = CustomField(default=1.0, developer=True)
    fixed_value: float = CustomField(default=2.0, developer=False)


def test_eemeter_warning():
    eemeter_warning = EEMeterWarning(
        qualified_name="qualified_name", description="description", data={}
    )
    assert eemeter_warning.qualified_name == "qualified_name"
    assert eemeter_warning.description == "description"
    assert eemeter_warning.data == {}
    assert str(eemeter_warning).startswith("EEMeterWarning")
    assert eemeter_warning.json() == {
        "data": {},
        "description": "description",
        "qualified_name": "qualified_name",
    }


def test_nonstandard_settings_warning_returns_none_for_identical_settings():
    settings = _DummySettings()
    reference = _DummySettings()

    assert nonstandard_settings_warning(settings, reference) is None


def test_nonstandard_settings_warning_reports_developer_tier_deviation():
    settings = _DummySettings(developer_value=3.0)
    reference = _DummySettings()

    warning = nonstandard_settings_warning(settings, reference)

    assert warning.qualified_name == "eemeter.settings.nonstandard"
    assert warning.data["deviations"] == {
        "developer_value": {"value": 3.0, "default": 1.0},
    }


def test_nonstandard_settings_warning_ignores_non_developer_tier_deviation():
    settings = _DummySettings(fixed_value=5.0)
    reference = _DummySettings()

    assert nonstandard_settings_warning(settings, reference) is None


def test_nonstandard_settings_warning_payload_is_json_serializable():
    settings = _DummySettings(developer_value=3.0)
    reference = _DummySettings()

    warning = nonstandard_settings_warning(settings, reference)

    assert json.dumps(warning.json())
