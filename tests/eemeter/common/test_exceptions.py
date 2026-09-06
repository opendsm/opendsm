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

from opendsm.eemeter.common import exceptions
from opendsm.eemeter.common.exceptions import (
    EEMeterError,
    NoBaselineDataError,
    NoReportingDataError,
    MissingModelParameterError,
    UnrecognizedModelTypeError,
    DataSufficiencyError,
    DisqualifiedModelError,
)

import pytest



SUBCLASSES = [
    NoBaselineDataError,
    NoReportingDataError,
    MissingModelParameterError,
    UnrecognizedModelTypeError,
    DataSufficiencyError,
    DisqualifiedModelError,
]


def test_eemeter_error_is_catchable_as_exception():
    with pytest.raises(Exception):
        raise EEMeterError


def test_every_exported_error_is_covered():
    exported = set(exceptions.__all__)
    covered = {EEMeterError.__name__} | {cls.__name__ for cls in SUBCLASSES}

    assert exported == covered, f"uncovered exports: {sorted(exported - covered)}"


@pytest.mark.parametrize("error_cls", SUBCLASSES, ids=lambda cls: cls.__name__)
def test_subclass_is_catchable_as_eemeter_error(error_cls):
    with pytest.raises(EEMeterError):
        raise error_cls


@pytest.mark.parametrize("error_cls", SUBCLASSES, ids=lambda cls: cls.__name__)
def test_subclass_raises_without_arguments(error_cls):
    with pytest.raises(error_cls):
        raise error_cls
