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

"""Comparison-group savings: matrix correction kernel, reporting correction, and
avoided-energy computation."""

from opendsm.comparison_groups.savings.model_correction import (
    model_correction,
    model_correction_matrix,
)
from opendsm.comparison_groups.savings.correction import (
    CorrectionResult,
    correct_reporting,
)
from opendsm.comparison_groups.savings.savings import (
    SavingsAggregation,
    SavingsResult,
    compute_savings,
)
from opendsm.comparison_groups.savings.settings import CGCorrectionSettings



__all__ = [
    "CGCorrectionSettings",
    "CorrectionResult",
    "SavingsAggregation",
    "SavingsResult",
    "compute_savings",
    "correct_reporting",
    "model_correction",
    "model_correction_matrix",
]
