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

import pytest

from opendsm.eemeter.common.exceptions import ModelLoadError
from opendsm.eemeter.common.model_choice import check_model_choice



def test_payload_of_the_loading_class_passes():
    check_model_choice({"model_choice": "daily"}, "daily")


def test_untagged_payload_passes():
    check_model_choice({}, "daily")


def test_payload_of_another_class_raises_naming_both():
    with pytest.raises(ModelLoadError, match="'billing' model; this class loads 'daily'"):
        check_model_choice({"model_choice": "billing"}, "daily")
