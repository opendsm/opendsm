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
import pytest


@pytest.fixture
def col_name():
    return 'col1'


@pytest.fixture
def df_treatment(col_name):
    return pd.DataFrame(
        [
            {"id": f"id_treatment_{x}", col_name: x}
            for x in (
                list(np.arange(0, 2, 0.1))
                + list(np.arange(2, 4, 0.5))
                + list(np.arange(4, 6, 1))
                + list(np.arange(6, 10, 0.2))
            )
        ]
    )


@pytest.fixture
def df_pool(col_name):
    return pd.DataFrame(
        [
            {"id": f"id_pool_{x}", col_name: x}
            for x in np.arange(0, 20, 0.01)
        ]
    )



@pytest.fixture
def df_equiv(df_treatment, df_pool):
    df_treatment_records = pd.DataFrame(
        [
            {
                "id": dim_project_site_meter_id,
                "month": month,
                "baseline_predicted_usage": month*i,
            }
            for month in range(1, 13)
            for i, dim_project_site_meter_id in enumerate(df_treatment["id"].values)
        ]
    )
    df_pool_records = pd.DataFrame(
        [
            {
                "id": dim_project_site_meter_id,
                "month": month,
                "baseline_predicted_usage": (13 - month) * i * 0.1,
            }
            for month in range(1, 13)
            for i, dim_project_site_meter_id in enumerate(df_pool["id"].values)
        ]
    )
    return pd.concat([df_treatment_records, df_pool_records])


@pytest.fixture
def equivalence_feature_matrix(df_equiv):
    df = df_equiv.pivot(index="id", columns=["month"], values="baseline_predicted_usage")
    return df.to_numpy()


@pytest.fixture
def equivalence_feature_ids(df_equiv):
    df = df_equiv.pivot(index="id", columns=["month"], values="baseline_predicted_usage")
    return df.index.unique()