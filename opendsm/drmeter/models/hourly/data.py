#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

from pathlib import Path
import copy
from typing import Optional, Union

import numpy as np
import pandas as pd

from opendsm.eemeter.models.hourly.data import _HourlyData
from opendsm.eemeter.common.sufficiency_criteria import HourlySufficiencyCriteria


class HourlyBaselineData(_HourlyData):
    def _check_data_sufficiency(self):
        return [], []
        hsc = HourlySufficiencyCriteria(
            data=sufficiency_df, is_electricity_data=self.is_electricity_data
        )
        hsc.check_sufficiency_baseline()
        disqualification = hsc.disqualification
        warnings = hsc.warnings

        return disqualification, warnings


class HourlyReportingData(_HourlyData):
    def _check_data_sufficiency(self):
        return [], []
        hsc = HourlySufficiencyCriteria(
            data=sufficiency_df, is_electricity_data=self.is_electricity_data
        )
        hsc.check_sufficiency_reporting()
        disqualification = hsc.disqualification
        warnings = hsc.warnings

        return disqualification, warnings