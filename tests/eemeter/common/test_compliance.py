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

from opendsm.common.warnings import OpenDSMWarning
from opendsm.eemeter.common.compliance import Compliance, normalize_enforce_compliance



def _warning(severity, member=None, qualified_name="qn"):
    warning = OpenDSMWarning(
        qualified_name=qualified_name,
        description="description",
        severity=severity,
        member=member,
    )

    return warning


def test_compliance_null_control_is_empty_and_qualified():
    compliance = Compliance()

    assert compliance.all == []
    assert compliance.disqualification == []
    assert compliance.warnings == []
    assert compliance.status == []
    assert compliance.qualified is True


def test_compliance_record_preserves_order_across_mixed_severities():
    compliance = Compliance()
    advisory = _warning("advisory", qualified_name="a")
    disqualified = _warning("disqualified", member="data", qualified_name="b")
    infeasible = _warning("infeasible", qualified_name="c")

    compliance.record(advisory)
    compliance.record(disqualified)
    compliance.record(infeasible)

    assert compliance.all == [advisory, disqualified, infeasible]
    assert compliance.disqualification == [disqualified]
    assert compliance.warnings == [advisory]


def test_compliance_status_deduplicates_and_sorts_members():
    compliance = Compliance()
    compliance.record(_warning("disqualified", member="settings", qualified_name="a"))
    compliance.record(_warning("disqualified", member="data", qualified_name="b"))
    compliance.record(_warning("disqualified", member="data", qualified_name="c"))

    assert compliance.status == ["data", "settings"]


def test_compliance_single_disqualified_member_is_unqualified():
    compliance = Compliance()
    compliance.record(_warning("disqualified", member="model"))

    assert compliance.status == ["model"]
    assert compliance.qualified is False


def test_compliance_infeasible_entry_is_recorded_but_not_derived():
    compliance = Compliance()
    infeasible = _warning("infeasible")
    compliance.record(infeasible)

    assert compliance.all == [infeasible]
    assert compliance.disqualification == []
    assert compliance.warnings == []
    assert compliance.qualified is True


def test_normalize_enforce_compliance_true_returns_all_members():
    assert normalize_enforce_compliance(True) == frozenset({"data", "model", "settings"})


def test_normalize_enforce_compliance_false_returns_empty_set():
    assert normalize_enforce_compliance(False) == frozenset()


@pytest.mark.parametrize("member", ["data", "model", "settings"])
def test_normalize_enforce_compliance_single_member_list(member):
    assert normalize_enforce_compliance([member]) == frozenset({member})


def test_normalize_enforce_compliance_two_member_tuple():
    assert normalize_enforce_compliance(("data", "settings")) == frozenset({"data", "settings"})


def test_normalize_enforce_compliance_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="unknown_member"):
        normalize_enforce_compliance(["unknown_member"])


def test_normalize_enforce_compliance_bare_string_raises_type_error():
    with pytest.raises(TypeError):
        normalize_enforce_compliance("data")


def test_normalize_enforce_compliance_none_raises_type_error():
    with pytest.raises(TypeError):
        normalize_enforce_compliance(None)
