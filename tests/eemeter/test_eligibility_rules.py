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

from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.daily.model import DailyModel
from opendsm.eemeter.models.hourly.model import HourlyModel



def _compliance_settings(settings) -> dict:
    """The compliance thresholds and toggles of a model's settings: the model-level fit and
    training thresholds, and the data-sufficiency settings, where a None threshold turns its
    check off."""
    model_thresholds = {
        name: getattr(settings, name)
        for name in type(settings).model_fields
        if name.endswith("_threshold") or name.startswith(("min_", "max_"))
    }
    compliance_settings = {
        "model": model_thresholds,
        "data": settings.data.model_dump(mode="json"),
    }

    return compliance_settings


# Default thresholds and toggles


@pytest.mark.parametrize(
    "model_class",
    [DailyModel, BillingModel, HourlyModel],
    ids=["daily", "billing", "hourly"],
)
def test_default_settings_compliance_thresholds(model_class, snapshot):
    """Each model's default settings carry the recorded compliance thresholds and toggles."""
    compliance_settings = _compliance_settings(model_class().settings)

    assert compliance_settings == snapshot
