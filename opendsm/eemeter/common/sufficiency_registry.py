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

"""Declared table of sufficiency rules and the registry that binds them to functions.

This module is a leaf: it imports nothing from ``opendsm.eemeter`` so that settings
can consult the table without depending on the rule implementations.
"""

from __future__ import annotations

import pydantic

from typing import Callable, Literal



Family = Literal["hourly", "daily", "billing"]
Phase = Literal["baseline", "reporting"]
Column = Literal["temperature", "observed", "ghi", "joint"]
Severity = Literal["infeasible", "disqualified", "advisory"]

ALL_FAMILIES: tuple[Family, ...] = ("hourly", "daily", "billing")
ALL_PHASES: tuple[Phase, ...] = ("baseline", "reporting")
ALL_COLUMNS: tuple[Column, ...] = ("temperature", "observed", "ghi", "joint")


class RuleSpec(pydantic.BaseModel):
    """Where a sufficiency rule may run and the worst verdict it may reach.

    ``families`` and ``phases`` are the union over the rule's columns: they say where the
    rule can run, not where it is switched on.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    families: tuple[Family, ...] = pydantic.Field(min_length=1)
    phases: tuple[Phase, ...] = pydantic.Field(min_length=1)
    columns: tuple[Column, ...] = ()
    caltrack: str | None = None
    severity: Severity
    toggleable: bool = True


_SPECS = (
    RuleSpec(
        name="no_data",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        severity="infeasible",
        toggleable=False,
    ),
    RuleSpec(
        name="baseline_day_length",
        families=ALL_FAMILIES,
        phases=("baseline",),
        severity="disqualified",
    ),
    RuleSpec(
        name="negative_observed_values",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        caltrack="2.3.5",
        severity="disqualified",
    ),
    RuleSpec(
        name="extreme_values",
        families=ALL_FAMILIES,
        phases=("baseline",),
        caltrack="2.3.6",
        severity="advisory",
    ),
    RuleSpec(
        name="boundary_gap",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        caltrack="2.2.4",
        severity="disqualified",
    ),
    RuleSpec(
        name="consecutive_missing_temperature_hours",
        families=("hourly",),
        phases=("baseline",),
        caltrack="2.2.4.1",
        severity="disqualified",
    ),
    RuleSpec(
        name="high_frequency_observed_coverage",
        families=("hourly",),
        phases=("baseline",),
        caltrack="2.2.2.1",
        severity="advisory",
    ),
    RuleSpec(
        name="high_frequency_meter_coverage",
        families=("daily", "billing"),
        phases=ALL_PHASES,
        severity="advisory",
    ),
    RuleSpec(
        name="offcycle_reads",
        families=("billing",),
        phases=ALL_PHASES,
        caltrack="2.2.3.4, 2.2.3.5",
        severity="disqualified",
    ),
    RuleSpec(
        name="estimated_reads",
        families=("billing",),
        phases=ALL_PHASES,
        caltrack="2.2.3.1",
        severity="advisory",
    ),
    RuleSpec(
        name="unique_values",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        columns=ALL_COLUMNS,
        severity="disqualified",
    ),
    RuleSpec(
        name="daily_coverage",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        columns=ALL_COLUMNS,
        severity="disqualified",
    ),
    RuleSpec(
        name="monthly_coverage",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        columns=ALL_COLUMNS,
        severity="disqualified",
    ),
)

RULE_SPECS: dict[str, RuleSpec] = {spec.name: spec for spec in _SPECS}

_RULE_FUNCTIONS: dict[str, Callable] = {}


def register(name: str) -> Callable[[Callable], Callable]:
    if name not in RULE_SPECS:
        raise KeyError(f"unknown sufficiency rule: {name}")

    if name in _RULE_FUNCTIONS:
        raise ValueError(f"sufficiency rule is already registered: {name}")

    def decorator(function: Callable) -> Callable:
        _RULE_FUNCTIONS[name] = function

        return function

    return decorator


def rule_spec(name: str) -> RuleSpec:
    """Return the spec declared for a rule, raising KeyError if there is none."""

    return RULE_SPECS[name]


def registered_rules() -> dict[str, tuple[RuleSpec, Callable]]:
    """Return name -> (spec, function) for every rule with a bound function."""
    bound = {
        name: (RULE_SPECS[name], function) for name, function in _RULE_FUNCTIONS.items()
    }

    return bound


def specs_for(family: Family, phase: Phase) -> dict[str, RuleSpec]:
    """Return the specs that may run for a data family in a given phase."""
    matched = {
        name: spec
        for name, spec in RULE_SPECS.items()
        if family in spec.families and phase in spec.phases
    }

    return matched


def snapshot_registry() -> dict[str, Callable]:
    """Return a copy of the bound rule functions, suitable for restore_registry."""

    return dict(_RULE_FUNCTIONS)


def restore_registry(snapshot: dict[str, Callable]) -> None:
    """Set the bound rule functions to the contents of a snapshot."""
    _RULE_FUNCTIONS.clear()
    _RULE_FUNCTIONS.update(snapshot)
