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

"""Declared table of sufficiency rules: where each rule may run and its worst verdict.

This module is a leaf: it imports nothing from ``opendsm.eemeter`` so that settings
can consult the table without depending on the rule implementations.
"""

from __future__ import annotations

import pydantic

from typing import Literal



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
    rule can run, not where it is switched on. An infeasible rule always runs and has no
    toggle in the settings.
    """

    model_config = pydantic.ConfigDict(frozen=True)

    name: str
    families: tuple[Family, ...] = pydantic.Field(min_length=1)
    phases: tuple[Phase, ...] = pydantic.Field(min_length=1)
    columns: tuple[Column, ...] = ()
    severity: Severity


# Declared in execution order: the infeasible rule first, since an empty frame stops the run
# at the first rule that is not infeasible, then the data-path judgments, then the frame rules.
_SPECS = (
    RuleSpec(
        name="no_data",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        severity="infeasible",
    ),
    RuleSpec(
        name="offcycle_reads",
        families=("billing",),
        phases=ALL_PHASES,
        severity="disqualified",
    ),
    RuleSpec(
        name="high_frequency_meter_coverage",
        families=("daily",),
        phases=ALL_PHASES,
        severity="advisory",
    ),
    RuleSpec(
        name="estimated_reads",
        families=("billing",),
        phases=ALL_PHASES,
        severity="advisory",
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
        severity="disqualified",
    ),
    RuleSpec(
        name="extreme_values",
        families=ALL_FAMILIES,
        phases=("baseline",),
        severity="advisory",
    ),
    RuleSpec(
        name="consecutive_missing_temperature_hours",
        families=("hourly",),
        phases=("baseline",),
        severity="disqualified",
    ),
    RuleSpec(
        name="unique_values",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        columns=("temperature", "observed", "ghi"),
        severity="disqualified",
    ),
    RuleSpec(
        name="daily_coverage",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        columns=("temperature", "observed", "joint"),
        severity="disqualified",
    ),
    RuleSpec(
        name="monthly_coverage",
        families=ALL_FAMILIES,
        phases=ALL_PHASES,
        columns=("temperature", "observed", "ghi"),
        severity="disqualified",
    ),
)

RULE_SPECS: dict[str, RuleSpec] = {spec.name: spec for spec in _SPECS}
