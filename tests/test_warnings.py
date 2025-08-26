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
from opendsm.eemeter.common.warnings import EEMeterWarning


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
