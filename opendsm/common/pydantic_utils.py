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

import numpy as np
import pandas as pd
import pydantic

from typing import Any, Optional

from functools import cached_property  # TODO: This requires Python 3.8


class PydanticDf(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    df: pd.DataFrame

    """list of required column types"""
    column_types: Optional[dict[str, Any]] = None

    @pydantic.model_validator(mode="after")
    def _check_columns(self):
        if self.column_types is not None:
            expected_columns = list(self.column_types.keys())
            if not set(self.df.columns) == set(expected_columns):
                raise ValueError(
                    f"Expected columns {expected_columns} but got {self.df.columns}"
                )

            for col, col_type in self.column_types.items():
                if col_type is None or col_type is Any:
                    continue

                if self.df[col].dtype != col_type:
                    # attempt to coerce numeric columns
                    if np.issubdtype(col_type, np.number) and np.issubdtype(
                        self.df[col].dtype, np.number
                    ):
                        self.df[col] = self.df[col].astype(col_type)
                    else:
                        raise ValueError(
                            f"Expected column {col} to be of type {col_type} but got {self.df[col].dtype}"
                        )
        return self


class ArbitraryPydanticModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="allow")


def PydanticFromDict(input_dict, name="PydanticModel"):
    """Creates a Pydantic model from a dictionary.

    Args:
        input_dict (dictionary): Dictionary and values to be used to create the Pydantic model.
        name (str, optional): Name of the Pydantic model. Defaults to "PydanticModel".

    Returns:
        Pydantic.BaseModel: Instantiated Pydantic model from input dictionary.
    """

    model = pydantic.create_model(
        name,
        **{name: (type(value), ...) for name, value in input_dict.items()},
        __base__=ArbitraryPydanticModel,
    )

    return model(**input_dict)


def computed_field_cached_property():
    decs = [pydantic.computed_field, cached_property]

    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco