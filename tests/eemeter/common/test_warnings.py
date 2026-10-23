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

from opendsm.eemeter.common.warnings import EEMeterWarning, nonstandard_settings_warning



_REPORT = {"developer_value": {"value": 3.0, "default": 1.0}}


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


def test_nonstandard_settings_warning_returns_none_for_an_empty_report():
    assert nonstandard_settings_warning({}) is None


def test_nonstandard_settings_warning_has_the_nonstandard_qualified_name():
    warning = nonstandard_settings_warning(_REPORT)

    assert warning.qualified_name == "eemeter.settings.nonstandard"


def test_nonstandard_settings_warning_carries_the_report_as_its_deviations():
    warning = nonstandard_settings_warning(_REPORT)

    assert warning.data["deviations"] == _REPORT, (
        f"expected the report as the deviations payload, got {warning.data}"
    )


def test_nonstandard_settings_warning_payload_is_json_serializable():
    warning = nonstandard_settings_warning(_REPORT)

    assert json.dumps(warning.json())
