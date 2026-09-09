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

"""Shared disqualification-ledger conventions.

Every meter dropped anywhere in the comparison-group stream is recorded in a
ledger frame with columns ``[id, stage, origin, reason, detail]`` (all str;
``detail`` may be empty). ``stage`` is the pipeline stage that dropped the
meter, ``origin`` names the source of the problem, ``reason`` is a short human
string, and ``detail`` carries verbatim eemeter warning qualified names /
descriptions or exception text.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd

from opendsm.eemeter.common.exceptions import EEMeterError



COLUMNS = ["id", "stage", "origin", "reason", "detail"]

STAGES = ("population", "selection", "correction")


class MeterCorrectionError(EEMeterError):
    """A treatment meter cannot be corrected at all.

    ``exclusions`` carries the ledger rows explaining the drop in the shared
    ``[id, stage, origin, reason, detail]`` schema, so a caller looping over
    treatment meters can record why this one failed and move on. Subclassing
    ``EEMeterError`` lets that caller catch a guard and a raw model failure
    together.
    """

    def __init__(self, message, exclusions):
        super().__init__(message)
        self.exclusions = exclusions


def empty_ledger():
    """An empty ledger frame with the shared columns (str dtype)."""
    frame = pd.DataFrame({column: pd.Series(dtype=str) for column in COLUMNS})

    return frame


def append(ledger, ids, stage, origin, reason, detail=""):
    """Return ``ledger`` with one row per id sharing ``stage``/``origin``/
    ``reason``/``detail``."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {list(STAGES)}, got {stage!r}")

    ids = [str(x) for x in ids]
    if not ids:
        return ledger

    rows = pd.DataFrame(
        {
            "id": ids,
            "stage": stage,
            "origin": origin,
            "reason": reason,
            "detail": detail,
        }
    )
    if ledger.empty:
        return rows

    combined = pd.concat([ledger, rows], ignore_index=True)

    return combined


def merge(*ledgers):
    """Merge ledger frames into one view ordered by stage (population,
    selection, correction) then id; rows sharing a stage and id keep their
    incoming relative order."""
    frames = [ledger for ledger in ledgers if not ledger.empty]

    if not frames:
        return empty_ledger()

    combined = pd.concat(frames, ignore_index=True)
    stage_rank = {stage: rank for rank, stage in enumerate(STAGES)}
    combined["_stage_rank"] = combined["stage"].map(stage_rank)
    combined = combined.sort_values(["_stage_rank", "id"])
    combined = combined.drop(columns="_stage_rank").reset_index(drop=True)

    return combined


def format_warnings(warnings):
    """Format an eemeter warnings list as ledger detail: ``; ``-joined
    qualified names, each with its description when present."""
    entries = []

    for warning in warnings:
        if warning.description:
            entries.append(f"{warning.qualified_name}: {warning.description}")
        else:
            entries.append(warning.qualified_name)

    text = "; ".join(entries)

    return text


def read_table_json(payload):
    """Rebuild a ledger from ``to_json(orient="table")`` output, pinning str
    dtypes."""
    frame = pd.read_json(StringIO(payload), orient="table")

    if frame.empty:
        return empty_ledger()

    ledger = frame[COLUMNS].astype(str).reset_index(drop=True)

    return ledger
