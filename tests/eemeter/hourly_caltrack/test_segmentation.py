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

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from opendsm.eemeter.models.hourly_caltrack import segmentation as segmentation_module
from opendsm.eemeter.models.hourly_caltrack.segmentation import (
    CalTRACKSegmentModel,
    SegmentedModel,
    segment_time_series,
    iterate_segmented_dataset,
)



@pytest.fixture
def index_8760():
    return pd.date_range("2017-01-01", periods=365 * 24, freq="h", tz="UTC")


def test_segment_time_series_invalid_type(index_8760):
    with pytest.raises(ValueError):
        segment_time_series(index_8760, segment_type="unknown")


@pytest.mark.parametrize(
    "segment_type, index_slice, drop_zero_weight_segments, expected_columns, expected_shape, expected_sum",
    [
        ("single", slice(None), False, ["all"], (8760, 1), 8760.0),
        (
            "one_month",
            slice(None),
            False,
            [
                "jan",
                "feb",
                "mar",
                "apr",
                "may",
                "jun",
                "jul",
                "aug",
                "sep",
                "oct",
                "nov",
                "dec",
            ],
            (8760, 12),
            8760.0,
        ),
        (
            "three_month",
            slice(None),
            False,
            [
                "dec-jan-feb",
                "jan-feb-mar",
                "feb-mar-apr",
                "mar-apr-may",
                "apr-may-jun",
                "may-jun-jul",
                "jun-jul-aug",
                "jul-aug-sep",
                "aug-sep-oct",
                "sep-oct-nov",
                "oct-nov-dec",
                "nov-dec-jan",
            ],
            (8760, 12),
            26280.0,
        ),
        (
            "three_month_weighted",
            slice(None),
            False,
            [
                "dec-jan-feb-weighted",
                "jan-feb-mar-weighted",
                "feb-mar-apr-weighted",
                "mar-apr-may-weighted",
                "apr-may-jun-weighted",
                "may-jun-jul-weighted",
                "jun-jul-aug-weighted",
                "jul-aug-sep-weighted",
                "aug-sep-oct-weighted",
                "sep-oct-nov-weighted",
                "oct-nov-dec-weighted",
                "nov-dec-jan-weighted",
            ],
            (8760, 12),
            17520.0,
        ),
        ("one_month", slice(0, 100), True, ["jan"], (100, 1), 100.0),
    ],
    ids=[
        "single",
        "one_month",
        "three_month",
        "three_month_weighted",
        "drop_zero_weight_segments",
    ],
)
def test_segment_time_series(
    index_8760,
    segment_type,
    index_slice,
    drop_zero_weight_segments,
    expected_columns,
    expected_shape,
    expected_sum,
):
    weights = segment_time_series(
        index_8760[index_slice],
        segment_type=segment_type,
        drop_zero_weight_segments=drop_zero_weight_segments,
    )
    assert list(weights.columns) == expected_columns
    assert weights.shape == expected_shape
    assert weights.sum().sum() == expected_sum


@pytest.fixture
def dataset():
    index = pd.date_range("2017-01-01", periods=1000, freq="h", tz="UTC")
    return pd.DataFrame({"a": 1, "b": 2}, index=index, columns=["a", "b"])


def test_iterate_segmented_dataset_no_segmentation(dataset):
    iterator = iterate_segmented_dataset(dataset, segmentation=None)
    segment_name, data = next(iterator)
    assert segment_name is None
    assert list(data.columns) == ["a", "b", "weight"]
    assert data.shape == (1000, 3)
    assert data.sum().sum() == 4000

    with pytest.raises(StopIteration):
        next(iterator)


@pytest.fixture
def segmentation(dataset):
    return segment_time_series(dataset.index, segment_type="one_month")


def test_iterate_segmented_dataset_with_segmentation(dataset, segmentation):
    iterator = iterate_segmented_dataset(dataset, segmentation=segmentation)
    segment_name, data = next(iterator)
    assert segment_name == "jan"
    assert list(data.columns) == ["a", "b", "weight"]
    assert data.shape == (744, 3)
    assert data.sum().sum() == 2976.0

    segment_name, data = next(iterator)
    assert segment_name == "feb"
    assert list(data.columns) == ["a", "b", "weight"]
    assert data.shape == (256, 3)
    assert data.sum().sum() == 1024.0

    segment_name, data = next(iterator)
    assert segment_name == "mar"
    assert list(data.columns) == ["a", "b", "weight"]
    assert data.shape == (0, 3)
    assert data.sum().sum() == 0.0


def test_iterate_segmented_dataset_with_processor(dataset, segmentation):
    feature_processor_segment_names = []

    def feature_processor(
        segment_name, dataset, column_mapping=None
    ):  # rename some columns
        feature_processor_segment_names.append(segment_name)
        return dataset.rename(columns=column_mapping).assign(weight=1)

    iterator = iterate_segmented_dataset(
        dataset,
        segmentation=segmentation,
        feature_processor=feature_processor,
        feature_processor_kwargs={"column_mapping": {"a": "c", "b": "d"}},
        feature_processor_segment_name_mapping={"jan": "jan2", "feb": "feb2"},
    )
    segment_name, data = next(iterator)
    assert feature_processor_segment_names == ["jan2"]
    assert segment_name == "jan"
    assert list(data.columns) == ["c", "d", "weight"]
    assert data.shape == (1000, 3)
    assert data.sum().sum() == 4000.0

    segment_name, data = next(iterator)
    assert feature_processor_segment_names == ["jan2", "feb2"]
    assert segment_name == "feb"
    assert list(data.columns) == ["c", "d", "weight"]
    assert data.shape == (1000, 3)
    assert data.sum().sum() == 4000.0


def test_segment_model():
    segment_model = CalTRACKSegmentModel(
        segment_name="segment",
        model=None,
        formula="meter_value ~ C(hour_of_week) + a - 1",
        model_params={"C(hour_of_week)[1]": 1, "a": 1},
        warnings=None,
    )
    index = pd.date_range("2017-01-01", periods=2, freq="h", tz="UTC")
    data = pd.DataFrame({"a": [1, 1], "hour_of_week": [1, 1]}, index=index)
    prediction = segment_model.predict(data)
    assert prediction.sum() == 4


def test_segmented_model():
    segment_model = CalTRACKSegmentModel(
        segment_name="jan",
        model=None,
        formula="meter_value ~ C(hour_of_week) + a- 1",
        model_params={"C(hour_of_week)[1]": 1, "a": 1},
        warnings=None,
    )

    def fake_feature_processor(segment_name, segment_data):
        features = {"hour_of_week": 1, "a": 1, "weight": segment_data.weight}

        return pd.DataFrame(features, index=segment_data.index)

    segmented_model = SegmentedModel(
        segment_models=[segment_model],
        prediction_segment_type="one_month",
        prediction_segment_name_mapping=None,
        prediction_feature_processor=fake_feature_processor,
        prediction_feature_processor_kwargs=None,
    )

    # make this cover jan and feb but only supply jan model
    index = pd.date_range("2017-01-01", periods=24 * 50, freq="h", tz="UTC")
    temps = pd.Series(np.linspace(0, 100, 24 * 50), index=index)
    prediction = segmented_model.predict(temps.index, temps).result.predicted_usage
    assert prediction.sum() == 1488.0


def test_segment_model_serialized():
    segment_model = CalTRACKSegmentModel(
        segment_name="jan",
        model=None,
        formula="meter_value ~ a + b - 1",
        model_params={"a": 1, "b": 1},
        warnings=None,
    )
    assert segment_model.json()["formula"] == "meter_value ~ a + b - 1"
    assert segment_model.json()["model_params"] == {"a": 1, "b": 1}
    assert segment_model.json()["warnings"] == []
    assert json.dumps(segment_model.json())


def test_segmented_model_serialized():
    segment_model = CalTRACKSegmentModel(
        segment_name="jan",
        model=None,
        formula="meter_value ~ a + b - 1",
        model_params={"a": 1, "b": 1},
        warnings=None,
    )

    def fake_feature_processor(segment_name, segment_data):  # pragma: no cover
        features = {"a": 1, "b": 1, "weight": segment_data.weight}

        return pd.DataFrame(features, index=segment_data.index)

    segmented_model = SegmentedModel(
        segment_models=[segment_model],
        prediction_segment_type="one_month",
        prediction_segment_name_mapping=None,
        prediction_feature_processor=fake_feature_processor,
        prediction_feature_processor_kwargs=None,
    )
    assert segmented_model.json()["prediction_segment_type"] == "one_month"
    assert (
        segmented_model.json()["prediction_feature_processor"]
        == "fake_feature_processor"
    )
    assert json.dumps(segmented_model.json())


@pytest.fixture
def dmatrix_must_not_run(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("a formula outside the grammar reached patsy")

    monkeypatch.setattr(segmentation_module, "dmatrix", fail)


@pytest.mark.parametrize(
    "formula",
    [
        None,
        "meter_value ~ C(hour_of_week) - 1",
        "meter_value ~ C(hour_of_week) - 1 + bin_0",
        "meter_value ~ C(hour_of_week) - 1 + bin_0_occupied + bin_0_unoccupied",
        "meter_value ~ C(hour_of_week) + a - 1",
        "meter_value ~ C(hour_of_week) + a- 1",
        "meter_value ~ a + b - 1",
    ],
)
def test_segment_model_accepts_linear_formulas(formula):
    segment_model = CalTRACKSegmentModel("jan", None, formula, {})
    assert segment_model.formula == formula


REJECTED_FORMULAS = [
    "meter_value ~ __import__('os').getpid()",
    "meter_value ~ C(hour_of_week) - 1 + bin_0_occupied + open('x')",
    "meter_value ~ a.b",
    "meter_value ~ a[0]",
    "meter_value ~ a * b",
    "meter_value ~ a:b",
    "meter_value ~ 'a'",
    "usage ~ a",
    "meter_value ~",
    "",
    42,
    "meter_value ~ C(hour_of_week)\u00a0- 1",
    "meter_value ~ C(hour_of_week)\n- 1",
]


@pytest.mark.parametrize("formula", REJECTED_FORMULAS)
def test_segment_model_rejects_formulas_outside_the_grammar(
    formula, dmatrix_must_not_run
):
    with pytest.raises(ValueError, match="formula"):
        CalTRACKSegmentModel("jan", None, formula, {})


@pytest.mark.parametrize("formula", REJECTED_FORMULAS)
def test_segment_model_rejects_assignment_outside_the_grammar(
    formula, dmatrix_must_not_run
):
    segment_model = CalTRACKSegmentModel(
        "jan", None, "meter_value ~ C(hour_of_week) - 1", {}
    )
    with pytest.raises(ValueError, match="formula"):
        segment_model.formula = formula
    assert segment_model.formula == "meter_value ~ C(hour_of_week) - 1"


def test_segment_model_from_json_rejects_formula_outside_the_grammar(
    dmatrix_must_not_run,
):
    data = {
        "segment_name": "jan",
        "formula": "meter_value ~ __import__('os').getpid()",
        "model_params": {},
        "warnings": [],
    }
    with pytest.raises(ValueError, match="formula"):
        CalTRACKSegmentModel.from_json(data)


def test_segment_model_predicts_builder_formula_in_empty_environment():
    segment_model = CalTRACKSegmentModel(
        segment_name="segment",
        model=None,
        formula="meter_value ~ C(hour_of_week) - 1 + bin_0_occupied",
        model_params={"C(hour_of_week)[1]": 1, "bin_0_occupied": 2},
    )
    index = pd.date_range("2017-01-01", periods=2, freq="h", tz="UTC")
    data = pd.DataFrame({"bin_0_occupied": [1, 1], "hour_of_week": [1, 1]}, index=index)
    prediction = segment_model.predict(data)
    assert prediction.tolist() == [3, 3]


def _legacy_pickle(formula):
    """A pickle of a segment model as 1.x wrote it: formula as a plain attribute."""
    legacy = CalTRACKSegmentModel.__new__(CalTRACKSegmentModel)
    legacy.__dict__.update(
        {
            "segment_name": "jan",
            "model": None,
            "formula": formula,
            "model_params": {},
            "warnings": [],
        }
    )

    return pickle.dumps(legacy)


def test_segment_model_unpickles_legacy_formula_attribute():
    restored = pickle.loads(_legacy_pickle("meter_value ~ C(hour_of_week) - 1"))

    assert restored.formula == "meter_value ~ C(hour_of_week) - 1"
    assert pickle.loads(pickle.dumps(restored)).formula == restored.formula


def test_segment_model_rejects_legacy_pickle_outside_the_grammar(dmatrix_must_not_run):
    with pytest.raises(ValueError, match="formula"):
        pickle.loads(_legacy_pickle("meter_value ~ __import__('os').getpid()"))
