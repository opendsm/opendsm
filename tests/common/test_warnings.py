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

import copy
import pickle
import warnings

import pytest

from opendsm.common.warnings import (
    _WARNING_REGISTRY,
    ClusteringWarning,
    OpenDSMWarning,
    StatsWarning,
)



@pytest.fixture
def advisory_warning():
    warning = OpenDSMWarning(
        qualified_name="opendsm.test.advisory",
        description="An advisory record.",
        severity="advisory",
        data={"count": 3},
    )

    return warning


@pytest.fixture
def disqualified_warning():
    warning = ClusteringWarning(
        qualified_name="opendsm.test.disqualified",
        description="A disqualifying record.",
        severity="disqualified",
        data=[1, 2],
        member="data",
    )

    return warning


@pytest.mark.parametrize("severity", ["infeasible", "advisory"])
def test_constructs_with_each_severity_that_takes_no_member(severity):
    warning = OpenDSMWarning(
        qualified_name="opendsm.test.severity",
        description="A record.",
        severity=severity,
    )

    assert warning.severity == severity, f"expected severity {severity}, got {warning.severity}"
    assert warning.member is None, f"expected member None, got {warning.member}"


def test_constructs_with_disqualified_severity_and_member(disqualified_warning):
    assert disqualified_warning.severity == "disqualified"
    assert disqualified_warning.member == "data"


def test_unknown_severity_raises_value_error():
    with pytest.raises(ValueError, match="severity must be one of"):
        OpenDSMWarning(
            qualified_name="opendsm.test.bad",
            description="A record.",
            severity="critical",
        )


def test_disqualified_without_member_raises_value_error():
    with pytest.raises(ValueError, match="member must be one of"):
        OpenDSMWarning(
            qualified_name="opendsm.test.bad",
            description="A record.",
            severity="disqualified",
        )


def test_disqualified_with_unknown_member_raises_value_error():
    with pytest.raises(ValueError, match="member must be one of"):
        OpenDSMWarning(
            qualified_name="opendsm.test.bad",
            description="A record.",
            severity="disqualified",
            member="weather",
        )


@pytest.mark.parametrize("severity", ["infeasible", "advisory"])
def test_member_on_non_disqualified_severity_raises_value_error(severity):
    with pytest.raises(ValueError, match="member must be None"):
        OpenDSMWarning(
            qualified_name="opendsm.test.bad",
            description="A record.",
            severity=severity,
            member="data",
        )


def test_str_is_one_line_and_omits_the_data_payload(advisory_warning):
    message = str(advisory_warning)

    assert message == "[advisory] opendsm.test.advisory: An advisory record."
    assert "count" not in message, f"data payload leaked into the message: {message}"


def test_repr_is_distinct_from_str(advisory_warning):
    assert repr(advisory_warning) == "OpenDSMWarning(opendsm.test.advisory)"
    assert repr(advisory_warning) != str(advisory_warning)


def test_json_carries_category_severity_and_member(disqualified_warning):
    assert disqualified_warning.json() == {
        "category": "ClusteringWarning",
        "severity": "disqualified",
        "member": "data",
        "qualified_name": "opendsm.test.disqualified",
        "description": "A disqualifying record.",
        "data": [1, 2],
    }


def test_from_json_round_trips_to_an_equal_instance_of_the_same_subclass(
    disqualified_warning,
):
    rebuilt = OpenDSMWarning.from_json(disqualified_warning.json())

    assert type(rebuilt) is ClusteringWarning, f"expected ClusteringWarning, got {type(rebuilt)}"
    assert rebuilt.json() == disqualified_warning.json()


def test_from_json_legacy_payload_rebuilds_as_eemeter_warning_with_passed_severity(
    monkeypatch,
):
    # the legacy branch resolves "EEMeterWarning" out of the registry; StatsWarning stands in
    # for it so the base-class contract is tested without importing the eemeter package
    monkeypatch.setitem(_WARNING_REGISTRY, "EEMeterWarning", StatsWarning)

    legacy_payload = {
        "qualified_name": "eemeter.legacy.record",
        "description": "Written before the taxonomy.",
        "data": {"n": 1},
    }

    rebuilt = OpenDSMWarning.from_json(legacy_payload, legacy_severity="advisory")

    assert type(rebuilt) is StatsWarning, f"expected the registered class, got {type(rebuilt)}"
    assert rebuilt.severity == "advisory"
    assert rebuilt.member is None
    assert rebuilt.qualified_name == "eemeter.legacy.record"
    assert rebuilt.data == {"n": 1}


def test_pickle_round_trip_preserves_every_field(disqualified_warning):
    rebuilt = pickle.loads(pickle.dumps(disqualified_warning))

    assert type(rebuilt) is ClusteringWarning
    assert rebuilt.json() == disqualified_warning.json()
    assert str(rebuilt) == str(disqualified_warning)


def test_deepcopy_preserves_every_field(disqualified_warning):
    rebuilt = copy.deepcopy(disqualified_warning)

    assert type(rebuilt) is ClusteringWarning
    assert rebuilt.json() == disqualified_warning.json()


def test_warn_emits_the_instance_on_the_warnings_channel(advisory_warning):
    with pytest.warns(OpenDSMWarning) as recorded:
        advisory_warning.warn()

    assert len(recorded) == 1, f"expected 1 emission, got {len(recorded)}"
    assert recorded[0].message is advisory_warning


def test_repeated_identical_warnings_all_surface(advisory_warning):
    with pytest.warns(OpenDSMWarning) as recorded:
        advisory_warning.warn()
        advisory_warning.warn()

    assert len(recorded) == 2, f"expected both emissions, got {len(recorded)}"


@pytest.mark.parametrize("warning_cls", [ClusteringWarning, StatsWarning])
def test_subclasses_register_themselves_by_class_name(warning_cls):
    assert _WARNING_REGISTRY[warning_cls.__name__] is warning_cls


def test_appended_filter_yields_to_a_host_error_policy(advisory_warning):
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter("error")
        warnings.filterwarnings("always", category=OpenDSMWarning, append=True)

        with pytest.raises(OpenDSMWarning):
            advisory_warning.warn()


def test_appended_filter_still_surfaces_repeats_with_no_host_policy(advisory_warning):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.resetwarnings()
        warnings.filterwarnings("always", category=OpenDSMWarning, append=True)

        advisory_warning.warn()
        advisory_warning.warn()

    assert len(recorded) == 2, f"expected both emissions, got {len(recorded)}"
