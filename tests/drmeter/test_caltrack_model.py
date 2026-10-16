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

from opendsm.drmeter.models.caltrack.model import Model
from opendsm.eemeter.common.exceptions import ModelLoadError
from opendsm.eemeter.models.hourly_caltrack import HourlyModel



def test_drmeter_model_is_single_segment_hourly_model():
    """The DR caltrack Model is an HourlyModel configured for a single segment."""
    model = Model()

    assert isinstance(model, HourlyModel)
    assert model.segment_type == "single"
    assert model.alpha == 0.1


def test_drmeter_model_rejects_an_eemeter_caltrack_payload():
    with pytest.raises(ModelLoadError, match="'hourly_caltrack' model; this class loads 'drmeter_caltrack'"):
        Model.from_dict({"model_choice": "hourly_caltrack", "model": {}})
