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

from opendsm.common.exceptions import OpenDSMError
from opendsm.comparison_groups.cg_clustering.treatment_fit import (
    ClusterTreatmentMatchError,
)
from opendsm.comparison_groups.exceptions import ComparisonGroupError
from opendsm.comparison_groups.exclusions import MeterCorrectionError
from opendsm.comparison_groups.individual_meter_matching.distance_calc_selection import (
    DistanceMatchingError,
)
from opendsm.comparison_groups.stratified_sampling.bins import ModelSamplingException



COMPARISON_GROUP_ERRORS = [
    MeterCorrectionError,
    ModelSamplingException,
    DistanceMatchingError,
    ClusterTreatmentMatchError,
]

HIERARCHY = [(ComparisonGroupError, OpenDSMError)] + [
    (error_cls, base)
    for error_cls in COMPARISON_GROUP_ERRORS
    for base in (ComparisonGroupError, OpenDSMError)
]


@pytest.mark.parametrize(
    "error_cls, base",
    HIERARCHY,
    ids=[f"{cls.__name__}-{base.__name__}" for cls, base in HIERARCHY],
)
def test_error_descends_from_its_base(error_cls, base):
    assert issubclass(error_cls, base), f"{error_cls.__name__} does not subclass {base.__name__}"
