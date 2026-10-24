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
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

import opendsm
from opendsm.common.warnings import (
    _WARNING_REGISTRY,
    ClusteringWarning,
    OpenDSMWarning,
    StatsWarning,
)



# pytest restores its own filter list around collection and each test, which drops the
# filter the module installs at import; a fresh interpreter observes that filter as a host
# application would
_FILTER_PROBE = """
import json
import warnings

from opendsm.common.warnings import OpenDSMWarning



record =OpenDSMWarning(
    qualified_name="opendsm.test.probe", description="A record.", severity="advisory"
)

with warnings.catch_warnings(record=True) as recorded:
    for _ in range(2):
        record.warn()

with warnings.catch_warnings():
    warnings.simplefilter("error")
    try:
        record.warn()
        host_error_raises = False
    except OpenDSMWarning:
        host_error_raises = True

print(json.dumps({"repeats": len(recorded), "host_error_raises": host_error_raises}))
"""


@pytest.fixture(scope="module")
def filter_probe():
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    env["PYTHONPATH"] = str(Path(opendsm.__file__).parents[1])
    completed = subprocess.run(
        [sys.executable, "-c", _FILTER_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)

    return result


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


@pytest.mark.parametrize("missing", ["category", "severity"])
def test_from_json_payload_without_category_or_severity_raises_key_error(
    advisory_warning, missing
):
    payload = advisory_warning.json()
    del payload[missing]

    with pytest.raises(KeyError, match=missing):
        OpenDSMWarning.from_json(payload)


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


@pytest.mark.parametrize("warning_cls", [ClusteringWarning, StatsWarning])
def test_subclasses_register_themselves_by_class_name(warning_cls):
    assert _WARNING_REGISTRY[warning_cls.__name__] is warning_cls


@pytest.mark.slow
def test_repeated_identical_records_all_surface_with_no_host_policy(filter_probe):
    """With only the interpreter's default filters, the module's own filter surfaces every
    emission of a record warned twice from one call site."""

    assert filter_probe["repeats"] == 2, f"expected both emissions, got {filter_probe['repeats']}"


@pytest.mark.slow
def test_module_filter_yields_to_a_host_error_policy(filter_probe):
    """A host filter added after import takes precedence over the module's own filter."""

    assert filter_probe["host_error_raises"], "a host 'error' filter did not raise the record"
