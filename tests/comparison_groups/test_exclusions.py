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

import pandas as pd
import pytest

from opendsm.comparison_groups import exclusions
from opendsm.eemeter.common.exceptions import EEMeterError
from opendsm.eemeter.common.warnings import EEMeterWarning



def test_empty_ledger_has_shared_columns_and_no_rows():
    ledger = exclusions.empty_ledger()

    assert list(ledger.columns) == ["id", "stage", "origin", "reason", "detail"]
    assert len(ledger) == 0


def test_append_adds_one_row_per_id_with_str_ids_and_empty_detail_default():
    ledger = exclusions.append(
        exclusions.empty_ledger(), [101, "m2"], "population", "pool_trim", "trimmed"
    )

    assert list(ledger["id"]) == ["101", "m2"]
    assert (ledger["stage"] == "population").all()
    assert (ledger["origin"] == "pool_trim").all()
    assert (ledger["reason"] == "trimmed").all()
    assert (ledger["detail"] == "").all()


def test_append_with_no_ids_returns_ledger_unchanged():
    ledger = exclusions.append(
        exclusions.empty_ledger(), ["m1"], "selection", "data_validation", "r"
    )

    unchanged = exclusions.append(ledger, [], "selection", "data_validation", "r")

    pd.testing.assert_frame_equal(unchanged, ledger)


def test_append_rejects_unknown_stage():
    with pytest.raises(ValueError, match="stage must be one of"):
        exclusions.append(exclusions.empty_ledger(), ["m1"], "cleaning", "model", "r")


def test_merge_orders_stages_population_selection_correction_then_id():
    correction = exclusions.append(
        exclusions.empty_ledger(), ["b", "a"], "correction", "correction_guard", "r-corr"
    )
    population = exclusions.append(
        exclusions.empty_ledger(), ["z"], "population", "pool_trim", "r-pop"
    )
    selection = exclusions.append(
        exclusions.empty_ledger(), ["m"], "selection", "treatment_fit", "r-sel"
    )

    merged = exclusions.merge(correction, population, selection)

    assert list(merged["stage"]) == ["population", "selection", "correction", "correction"]
    assert list(merged["id"]) == ["z", "m", "a", "b"]


def test_merge_is_stable_for_rows_sharing_stage_and_id():
    first = exclusions.append(
        exclusions.empty_ledger(), ["m1"], "correction", "reporting_data", "first"
    )
    second = exclusions.append(
        exclusions.empty_ledger(), ["m1"], "correction", "model", "second"
    )

    merged = exclusions.merge(first, second)

    assert list(merged["reason"]) == ["first", "second"]


def test_merge_of_empty_ledgers_returns_typed_empty_frame():
    merged = exclusions.merge(exclusions.empty_ledger(), exclusions.empty_ledger())

    assert list(merged.columns) == ["id", "stage", "origin", "reason", "detail"]
    assert merged.empty


def test_format_warnings_joins_qualified_names_with_descriptions():
    warnings_list = [
        EEMeterWarning(
            qualified_name="eemeter.sufficiency_criteria.too_few_days",
            description="Baseline is too short.",
            data=None,
        ),
        EEMeterWarning(qualified_name="eemeter.other", description="", data=None),
    ]

    detail = exclusions.format_warnings(warnings_list)

    assert detail == (
        "eemeter.sufficiency_criteria.too_few_days: Baseline is too short.; eemeter.other"
    )


def test_format_warnings_empty_list_is_empty_string():
    assert exclusions.format_warnings([]) == ""


def test_table_json_roundtrip_preserves_numeric_looking_ids():
    ledger = exclusions.append(
        exclusions.empty_ledger(), ["0042"], "selection", "data_validation", "r", detail="d"
    )

    rebuilt = exclusions.read_table_json(ledger.to_json(orient="table"))

    pd.testing.assert_frame_equal(rebuilt, ledger)
    assert rebuilt["id"].iloc[0] == "0042"


def test_meter_correction_error_is_catchable_as_an_eemeter_error():
    """A caller looping over treatment meters catches the correction guard and a
    raw model failure with a single ``except EEMeterError``."""
    ledger = exclusions.append(
        exclusions.empty_ledger(), ["m1"], "correction", "correction_guard", "all-NaN cluster weights"
    )

    with pytest.raises(EEMeterError, match="cannot be corrected"):
        raise exclusions.MeterCorrectionError("treatment meter m1 cannot be corrected", ledger)


def test_meter_correction_error_carries_the_ledger_rows_explaining_the_drop():
    """The rows ride on ``.exclusions`` in the shared ledger schema, so a caller
    can record why the meter failed and move on."""
    ledger = exclusions.append(
        exclusions.empty_ledger(),
        ["m1", "p3"],
        "correction",
        "model",
        "model prediction failed",
        detail="disqualified model",
    )

    error = exclusions.MeterCorrectionError("treatment meter m1 cannot be corrected", ledger)

    assert list(error.exclusions.columns) == ["id", "stage", "origin", "reason", "detail"]
    assert list(error.exclusions["id"]) == ["m1", "p3"]
    pd.testing.assert_frame_equal(error.exclusions, ledger)


def test_table_json_roundtrip_of_empty_ledger():
    rebuilt = exclusions.read_table_json(exclusions.empty_ledger().to_json(orient="table"))

    pd.testing.assert_frame_equal(rebuilt, exclusions.empty_ledger())
