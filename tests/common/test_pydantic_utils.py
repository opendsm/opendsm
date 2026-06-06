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

import math

import pandas as pd
import pydantic
import pytest

from opendsm.common.pydantic_utils import (
    ArbitraryPydanticModel,
    PydanticDf,
    PydanticFromDict,
)



class _SpecialFloatModel(ArbitraryPydanticModel):
    x: float
    y: float
    z: float


def test_special_floats_serialize_to_strings_in_json():
    """nan/inf/-inf become their string tokens only in JSON mode."""
    model = _SpecialFloatModel(x=float("nan"), y=float("inf"), z=float("-inf"))

    json_data = model.model_dump(mode="json")

    assert json_data == {"x": "nan", "y": "inf", "z": "-inf"}


def test_special_floats_preserved_in_python_mode():
    """Python-mode dumps keep native float values, not string tokens."""
    model = _SpecialFloatModel(x=float("nan"), y=1.0, z=-2.0)

    python_data = model.model_dump()

    assert math.isnan(python_data["x"])
    assert python_data["y"] == 1.0


def test_special_floats_roundtrip_through_json():
    """Re-validating the JSON dump restores nan/inf/-inf as floats."""
    model = _SpecialFloatModel(x=float("nan"), y=float("inf"), z=float("-inf"))

    restored = _SpecialFloatModel(**model.model_dump(mode="json"))

    assert math.isnan(restored.x)
    assert math.isinf(restored.y) and restored.y > 0
    assert math.isinf(restored.z) and restored.z < 0


def test_pydantic_from_dict_roundtrips_special_floats():
    """PydanticFromDict builds a model whose JSON dump restores special floats."""
    model = PydanticFromDict({"a": float("inf"), "b": 3.0})

    restored = type(model)(**model.model_dump(mode="json"))

    assert math.isinf(restored.a)
    assert restored.b == 3.0


def test_pydantic_df_coerces_numeric_dtype():
    """A numeric column is coerced to the requested numeric dtype."""
    validated = PydanticDf(df=pd.DataFrame({"a": [1, 2, 3]}), column_types={"a": "float"})

    assert validated.df["a"].dtype == "float64"


def test_pydantic_df_wrong_columns_raise():
    """A column set that differs from the schema raises ValidationError."""
    with pytest.raises(pydantic.ValidationError, match="Expected columns"):
        PydanticDf(df=pd.DataFrame({"b": [1]}), column_types={"a": "float"})


def test_pydantic_df_non_coercible_dtype_raises():
    """A non-numeric column that cannot coerce to the numeric type raises."""
    with pytest.raises(pydantic.ValidationError, match="to be of type"):
        PydanticDf(df=pd.DataFrame({"a": ["x", "y"]}), column_types={"a": "float"})


def test_pydantic_df_no_column_types_skips_validation():
    """With column_types=None any frame is accepted unchanged."""
    df = pd.DataFrame({"anything": [1, 2], "goes": [3, 4]})

    validated = PydanticDf(df=df)

    assert list(validated.df.columns) == ["anything", "goes"]
