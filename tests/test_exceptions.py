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
from opendsm.eemeter.common.exceptions import (
    EEMeterError,
    NoBaselineDataError,
    NoReportingDataError,
    MissingModelParameterError,
    UnrecognizedModelTypeError,
)

import pytest


def test_eemeter_error():
    with pytest.raises(EEMeterError):
        raise EEMeterError


def test_no_baseline_data_error():
    with pytest.raises(NoBaselineDataError):
        raise NoBaselineDataError
    assert isinstance(NoBaselineDataError(), EEMeterError)


def test_no_reporting_data_error():
    with pytest.raises(NoReportingDataError):
        raise NoReportingDataError
    assert isinstance(NoReportingDataError(), EEMeterError)


def test_missing_model_parameter_error():
    with pytest.raises(MissingModelParameterError):
        raise MissingModelParameterError
    assert isinstance(MissingModelParameterError(), EEMeterError)


def test_unrecognized_model_type_error():
    with pytest.raises(UnrecognizedModelTypeError):
        raise UnrecognizedModelTypeError
    assert isinstance(UnrecognizedModelTypeError(), EEMeterError)
