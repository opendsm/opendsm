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

from opendsm.common.exceptions import OpenDSMError
from opendsm.common.warnings import OpenDSMWarning
from opendsm.eemeter.common import exceptions
from opendsm.eemeter.common.exceptions import (
    ComplianceError,
    DataSufficiencyError,
    DisqualifiedModelError,
    EEMeterError,
    InfeasibleDataError,
    ModelFitError,
    ModelLoadError,
    NoBaselineDataError,
    NoReportingDataError,
    NonstandardSettingsError,
)

import pytest



SUBCLASSES = [
    ComplianceError,
    DataSufficiencyError,
    InfeasibleDataError,
    DisqualifiedModelError,
    NonstandardSettingsError,
    ModelFitError,
    ModelLoadError,
    NoBaselineDataError,
    NoReportingDataError,
]

COMPLIANCE_LEAVES = [
    DataSufficiencyError,
    InfeasibleDataError,
    DisqualifiedModelError,
    NonstandardSettingsError,
]


def _record(qualified_name="eemeter.test.record"):
    record = OpenDSMWarning(
        qualified_name=qualified_name,
        description="a record",
        severity="advisory",
    )

    return record


def test_eemeter_error_is_an_opendsm_error():
    assert issubclass(EEMeterError, OpenDSMError)


def test_every_exported_error_is_covered():
    exported = set(exceptions.__all__)
    covered = {EEMeterError.__name__} | {cls.__name__ for cls in SUBCLASSES}

    assert exported == covered, f"uncovered exports: {sorted(exported - covered)}"


@pytest.mark.parametrize("error_cls", SUBCLASSES, ids=lambda cls: cls.__name__)
def test_subclass_is_catchable_as_eemeter_error(error_cls):
    with pytest.raises(EEMeterError):
        raise error_cls


@pytest.mark.parametrize("error_cls", COMPLIANCE_LEAVES, ids=lambda cls: cls.__name__)
def test_compliance_leaf_is_catchable_as_compliance_error(error_cls):
    with pytest.raises(ComplianceError):
        raise error_cls("rejected", entries=[_record()])


def test_model_fit_error_is_not_a_compliance_error():
    assert not issubclass(ModelFitError, ComplianceError)


@pytest.mark.parametrize("error_cls", COMPLIANCE_LEAVES, ids=lambda cls: cls.__name__)
def test_compliance_entries_default_to_an_empty_list(error_cls):
    error = error_cls("rejected")

    assert error.entries == []


def test_compliance_entries_round_trip_the_records_given():
    first = _record("eemeter.test.first")
    second = _record("eemeter.test.second")

    error = DataSufficiencyError("rejected", entries=[first, second])

    assert error.entries == [first, second]


def test_compliance_entries_copy_the_list_given():
    records = [_record()]
    error = DataSufficiencyError("rejected", entries=records)
    records.append(_record("eemeter.test.later"))

    assert len(error.entries) == 1
