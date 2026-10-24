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

import datetime
import warnings

import pytest

from opendsm.common.warnings import OpenDSMWarning
from opendsm.eemeter.common.compliance import (
    SCHEMA_VERSION,
    Compliance,
    check_schema_version,
    normalize_enforce_compliance,
    prior_format_entries,
    raise_if_baseline_disqualified,
    raise_if_infeasible,
    raise_if_model_disqualified,
    record_timezone_mismatch,
)
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
    InfeasibleDataError,
)
from opendsm.eemeter.common.warnings import EEMeterWarning



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


def test_compliance_infeasible_lists_only_infeasible_entries_in_order():
    compliance = Compliance()
    first = _warning("infeasible", qualified_name="first")
    second = _warning("infeasible", qualified_name="second")
    for entry in (first, _warning("advisory"), second, _warning("disqualified", "data")):
        compliance.record(entry)

    assert compliance.infeasible == [first, second]


def test_compliance_infeasible_is_empty_without_infeasible_entries():
    compliance = Compliance()
    compliance.record(_warning("disqualified", "data"))
    compliance.record(_warning("advisory"))

    assert compliance.infeasible == []


def test_compliance_disqualified_in_filters_disqualifications_by_member():
    compliance = Compliance()
    data = _warning("disqualified", "data")
    model = _warning("disqualified", "model")
    settings = _warning("disqualified", "settings")
    for entry in (data, model, _warning("advisory"), settings):
        compliance.record(entry)

    assert compliance.disqualified_in({"data", "settings"}) == [data, settings]
    assert compliance.disqualified_in(frozenset()) == []


def test_compliance_record_and_warn_records_and_emits_the_entry():
    compliance = Compliance()
    entry = _warning("advisory")

    with pytest.warns(OpenDSMWarning) as emitted:
        compliance.record_and_warn(entry)

    assert compliance.all == [entry]
    assert [w.message for w in emitted] == [entry]


def test_compliance_record_and_warn_records_before_emitting():
    """The entry is stored even when the warnings filter turns its emission into an error."""

    compliance = Compliance()
    entry = _warning("advisory")

    with warnings.catch_warnings():
        warnings.simplefilter("error", OpenDSMWarning)
        with pytest.raises(OpenDSMWarning):
            compliance.record_and_warn(entry)

    assert compliance.all == [entry]


def test_compliance_record_and_warn_attributes_the_warning_to_its_caller():
    """The emitted warning's filename is the module that called `record_and_warn`."""

    compliance = Compliance()

    with pytest.warns(OpenDSMWarning) as emitted:
        compliance.record_and_warn(_warning("advisory"))

    assert emitted[0].filename == __file__, emitted[0].filename


def test_record_timezone_mismatch_records_and_emits_one_data_disqualification():
    compliance = Compliance()

    with pytest.warns(OpenDSMWarning) as emitted:
        record_timezone_mismatch(compliance, "America/Los_Angeles", "UTC")

    assert len(compliance.all) == 1, compliance.all
    entry = compliance.all[0]
    assert entry.qualified_name == "eemeter.data_quality.timezone_mismatch"
    assert (entry.severity, entry.member) == ("disqualified", "data")
    assert entry.data == {
        "baseline_timezone": "America/Los_Angeles",
        "reporting_timezone": "UTC",
    }
    assert [w.message for w in emitted] == [entry]


def test_record_timezone_mismatch_compares_timezones_as_strings():
    """A tzinfo and a timezone name with the same string form are one timezone."""

    compliance = Compliance()

    with warnings.catch_warnings():
        warnings.simplefilter("error", OpenDSMWarning)
        record_timezone_mismatch(compliance, datetime.timezone.utc, "UTC")

    assert compliance.all == []


def test_raise_if_infeasible_carries_every_infeasible_entry():
    compliance = Compliance()
    first = _warning("infeasible", qualified_name="first")
    second = _warning("infeasible", qualified_name="second")
    for entry in (first, _warning("disqualified", "data"), second):
        compliance.record(entry)

    with pytest.raises(InfeasibleDataError) as raised:
        raise_if_infeasible(compliance)

    assert raised.value.entries == [first, second]


def test_raise_if_infeasible_passes_without_infeasible_entries():
    compliance = Compliance()
    compliance.record(_warning("disqualified", "data"))

    raise_if_infeasible(compliance)


def test_raise_if_baseline_disqualified_carries_data_disqualifications():
    compliance = Compliance()
    data = _warning("disqualified", "data")
    compliance.record(data)
    compliance.record(_warning("disqualified", "model"))

    with pytest.raises(DataSufficiencyError) as raised:
        raise_if_baseline_disqualified(compliance, frozenset({"data"}))

    assert raised.value.entries == [data]


@pytest.mark.parametrize(
    "enforced, member",
    [
        (frozenset({"model", "settings"}), "data"),
        (frozenset({"data", "model", "settings"}), "model"),
    ],
)
def test_raise_if_baseline_disqualified_passes_without_enforced_data_disqualification(
    enforced, member
):
    compliance = Compliance()
    compliance.record(_warning("disqualified", member))

    raise_if_baseline_disqualified(compliance, enforced)


def test_raise_if_model_disqualified_carries_enforced_members_only():
    compliance = Compliance()
    model = _warning("disqualified", "model")
    compliance.record(_warning("disqualified", "data"))
    compliance.record(model)

    with pytest.raises(DisqualifiedModelError) as raised:
        raise_if_model_disqualified(compliance, frozenset({"model"}))

    assert raised.value.entries == [model]


def test_raise_if_model_disqualified_passes_when_no_member_is_enforced():
    compliance = Compliance()
    compliance.record(_warning("disqualified", "model"))

    raise_if_model_disqualified(compliance, frozenset())


@pytest.mark.parametrize(
    "enforce, expected",
    [
        (True, frozenset({"data", "model", "settings"})),
        (False, frozenset()),
        (["data"], frozenset({"data"})),
        (["model"], frozenset({"model"})),
        (["settings"], frozenset({"settings"})),
        (("data", "settings"), frozenset({"data", "settings"})),
    ],
    ids=["true", "false", "data", "model", "settings", "two_member_tuple"],
)
def test_normalize_enforce_compliance_returns_the_enforced_members(enforce, expected):
    observed = normalize_enforce_compliance(enforce)

    assert observed == expected, f"expected {expected}, got {observed}"


def test_normalize_enforce_compliance_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="unknown_member"):
        normalize_enforce_compliance(["unknown_member"])


@pytest.mark.parametrize("enforce", ["data", None], ids=["bare_string", "none"])
def test_normalize_enforce_compliance_non_collection_raises_type_error(enforce):
    with pytest.raises(TypeError):
        normalize_enforce_compliance(enforce)


@pytest.mark.parametrize(
    ("qualified_name", "member"),
    [
        ("eemeter.model_fit_metrics", "model"),
        ("eemeter.settings.nonstandard", "settings"),
        ("eemeter.sufficiency_criteria.too_many_days_with_missing_data", "data"),
    ],
)
def test_prior_format_disqualification_takes_its_member_from_the_qualified_name(
    qualified_name, member
):
    info = {
        "disqualification": [
            {"qualified_name": qualified_name, "description": "d", "data": {"n": 1}}
        ]
    }

    (entry,) = prior_format_entries(info)

    assert type(entry) is EEMeterWarning, f"expected EEMeterWarning, got {type(entry)}"
    assert entry.severity == "disqualified"
    assert entry.member == member, f"expected {member!r}, got {entry.member!r}"
    assert entry.qualified_name == qualified_name
    assert entry.data == {"n": 1}


def test_prior_format_warnings_rebuild_as_advisories_after_disqualifications():
    info = {
        "warnings": [{"qualified_name": "eemeter.advisory", "description": "a"}],
        "disqualification": [{"qualified_name": "eemeter.dq", "description": "d"}],
    }

    entries = prior_format_entries(info)

    observed = [(e.qualified_name, e.severity, e.member, e.data) for e in entries]
    expected = [
        ("eemeter.dq", "disqualified", "data", None),
        ("eemeter.advisory", "advisory", None, None),
    ]
    assert observed == expected, f"expected {expected}, got {observed}"


@pytest.mark.parametrize("info", [{}, {"disqualification": None, "warnings": None}])
def test_prior_format_info_without_records_rebuilds_nothing(info):
    assert prior_format_entries(info) == []


def test_check_schema_version_returns_a_readable_version():
    assert check_schema_version(SCHEMA_VERSION) == SCHEMA_VERSION


def test_check_schema_version_newer_than_readable_raises_value_error():
    with pytest.raises(ValueError, match="newer"):
        check_schema_version(SCHEMA_VERSION + 1)
