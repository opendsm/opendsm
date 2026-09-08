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

from opendsm.common.clustering.exceptions import ClusteringError
from opendsm.common.clustering.transform.fpca import FpcaError
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


def test_opendsm_error_is_an_exception_subclass():
    assert issubclass(OpenDSMError, Exception)


def test_opendsm_error_carries_its_message():
    with pytest.raises(OpenDSMError, match="something went wrong"):
        raise OpenDSMError("something went wrong")


@pytest.mark.parametrize(
    "error_cls", COMPARISON_GROUP_ERRORS, ids=lambda cls: cls.__name__
)
def test_comparison_group_error_descends_from_comparison_group_error(error_cls):
    assert issubclass(error_cls, ComparisonGroupError)


@pytest.mark.parametrize(
    "error_cls", COMPARISON_GROUP_ERRORS, ids=lambda cls: cls.__name__
)
def test_comparison_group_error_descends_from_opendsm_error(error_cls):
    assert issubclass(error_cls, OpenDSMError)


def test_comparison_group_error_is_catchable_as_opendsm_error():
    with pytest.raises(OpenDSMError):
        raise DistanceMatchingError("matching failed")


def test_comparison_group_error_is_an_opendsm_error():
    assert issubclass(ComparisonGroupError, OpenDSMError)


def test_clustering_error_is_an_opendsm_error():
    assert issubclass(ClusteringError, OpenDSMError)


def test_fpca_error_is_catchable_as_a_clustering_error():
    with pytest.raises(ClusteringError):
        raise FpcaError


def test_fpca_error_is_catchable_as_an_opendsm_error():
    with pytest.raises(OpenDSMError):
        raise FpcaError
