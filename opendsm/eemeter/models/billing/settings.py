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

from typing import Literal

from opendsm.common.base_settings import CustomField
from opendsm.eemeter.common.data_settings import BillingDataSettings
from opendsm.eemeter.models.daily.utilities.settings import DailySettings



class BillingSettings(DailySettings):
    """Daily settings on the legacy preset with the billing data block."""

    preset: Literal["current", "legacy"] = CustomField(
        default="legacy",
        developer=False,
        description="Named set of default values that all other settings fall back to",
    )

    data: BillingDataSettings = pydantic.Field(default_factory=BillingDataSettings)
