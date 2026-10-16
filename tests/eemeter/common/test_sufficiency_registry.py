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

import pydantic
import pytest

from opendsm.eemeter.common.sufficiency_registry import (
    ALL_COLUMNS,
    ALL_FAMILIES,
    ALL_PHASES,
    RULE_SPECS,
    register,
    registered_rules,
    restore_registry,
    rule_spec,
    snapshot_registry,
    specs_for,
)



EXPECTED_RULE_NAMES = {
    "no_data",
    "baseline_day_length",
    "negative_observed_values",
    "extreme_values",
    "boundary_gap",
    "consecutive_missing_temperature_hours",
    "high_frequency_observed_coverage",
    "high_frequency_meter_coverage",
    "offcycle_reads",
    "estimated_reads",
    "unique_values",
    "daily_coverage",
    "monthly_coverage",
}


@pytest.fixture
def empty_registry():
    """Run the test against an empty registry and put the bindings back afterwards."""
    snapshot = snapshot_registry()
    restore_registry({})
    yield

    restore_registry(snapshot)


def _noop_rule(*args, **kwargs):
    return None


def test_rule_specs_table_holds_the_thirteen_declared_rules():
    assert set(RULE_SPECS) == EXPECTED_RULE_NAMES


def test_every_spec_is_keyed_by_its_own_name():
    mismatched = {name for name, spec in RULE_SPECS.items() if spec.name != name}
    assert mismatched == set(), f"specs keyed under the wrong name: {mismatched}"


def test_every_spec_declares_at_least_one_family_and_one_phase():
    empty = {
        name
        for name, spec in RULE_SPECS.items()
        if len(spec.families) == 0 or len(spec.phases) == 0
    }
    assert empty == set(), f"specs missing a family or a phase: {empty}"


def test_only_no_data_is_non_toggleable():
    fixed = {name for name, spec in RULE_SPECS.items() if not spec.toggleable}
    assert fixed == {"no_data"}


def test_column_scoped_specs_use_only_known_column_names():
    unknown = {
        name
        for name, spec in RULE_SPECS.items()
        if not set(spec.columns).issubset(set(ALL_COLUMNS))
    }
    assert unknown == set(), f"specs with unknown columns: {unknown}"


def test_column_scoped_specs_are_the_three_coverage_rules():
    scoped = {name for name, spec in RULE_SPECS.items() if spec.columns}
    assert scoped == {"unique_values", "daily_coverage", "monthly_coverage"}


def test_no_data_is_infeasible_everywhere():
    spec = rule_spec("no_data")
    assert spec.severity == "infeasible"
    assert spec.families == ALL_FAMILIES
    assert spec.phases == ALL_PHASES
    assert spec.columns == ()


def test_offcycle_reads_is_a_billing_only_disqualification():
    spec = rule_spec("offcycle_reads")
    assert spec.families == ("billing",)
    assert spec.phases == ALL_PHASES
    assert spec.severity == "disqualified"
    assert spec.caltrack == "2.2.3.4, 2.2.3.5"


def test_rule_spec_raises_for_an_unknown_name():
    with pytest.raises(KeyError):
        rule_spec("not_a_rule")


def test_rule_specs_are_frozen():
    with pytest.raises(pydantic.ValidationError):
        rule_spec("no_data").severity = "advisory"


def test_specs_for_hourly_reporting_excludes_baseline_and_billing_rules():
    matched = specs_for("hourly", "reporting")
    assert set(matched) == {
        "no_data",
        "negative_observed_values",
        "boundary_gap",
        "unique_values",
        "daily_coverage",
        "monthly_coverage",
    }


def test_specs_for_billing_baseline_excludes_the_hourly_only_rules():
    matched = specs_for("billing", "baseline")
    assert set(matched) == EXPECTED_RULE_NAMES - {
        "consecutive_missing_temperature_hours",
        "high_frequency_observed_coverage",
    }


def test_register_raises_for_an_unknown_rule_name():
    with pytest.raises(KeyError, match="not_a_rule"):
        register("not_a_rule")


def test_register_raises_when_a_rule_is_bound_twice(empty_registry):
    register("no_data")(_noop_rule)

    with pytest.raises(ValueError, match="no_data"):
        register("no_data")(_noop_rule)


def test_registered_rules_pairs_each_bound_function_with_its_spec(empty_registry):
    register("no_data")(_noop_rule)
    bound = registered_rules()
    assert bound == {"no_data": (rule_spec("no_data"), _noop_rule)}


def test_restore_registry_drops_rules_registered_after_the_snapshot(empty_registry):
    before = snapshot_registry()
    register("no_data")(_noop_rule)
    assert "no_data" in registered_rules()

    restore_registry(before)
    assert "no_data" not in registered_rules()
