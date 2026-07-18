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

from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const



def _hour_loadshape_data(df_baseline, meter_ids):
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )
    time_series = df_baseline.loc[meter_ids].reset_index()[["id", "datetime", "observed"]]
    data = Data(time_series_df=time_series, settings=settings)

    return data


@pytest.fixture(scope="session")
def cg_loadshape_data(_comstock_hourly_all):
    """(treatment_data, comparison_pool_data) Data objects with 24-hour load
    shapes built from real ComStock hourly observations. Disjoint meter sets."""
    df_baseline, _ = _comstock_hourly_all
    ids = sorted(df_baseline.index.get_level_values("id").unique())

    treatment_data = _hour_loadshape_data(df_baseline, ids[:8])
    comparison_pool_data = _hour_loadshape_data(df_baseline, ids[8:40])

    return treatment_data, comparison_pool_data


@pytest.fixture(scope="session")
def stratified_feature_loadshape_data():
    """(treatment_data, comparison_pool_data) Data objects carrying both
    stratification features (summer_usage, winter_usage) and load shapes, for
    the public Stratified_Sampling flow. No real dataset pairs both, so these
    are constructed deterministically."""
    rng = np.random.default_rng(0)

    def _data(prefix, n):
        ids = [f"{prefix}{i}" for i in range(n)]
        features = pd.DataFrame(
            {
                "id": ids,
                "summer_usage": rng.uniform(3000, 6000, n),
                "winter_usage": rng.uniform(3000, 6000, n),
            }
        )
        rows = [
            {"id": meter_id, "time": t, "loadshape": float(rng.normal(10, 1))}
            for meter_id in ids
            for t in range(1, 25)
        ]
        loadshape = pd.DataFrame(rows)
        settings = Data_Settings(agg_type=None, loadshape_type=None, time_period=None)
        data = Data(loadshape_df=loadshape, features_df=features, settings=settings)

        return data

    treatment_data = _data("t", 40)
    comparison_pool_data = _data("p", 400)

    return treatment_data, comparison_pool_data


@pytest.fixture(scope="session")
def cg_clustering_data(_comstock_hourly_all):
    """(treatment_data, comparison_pool_data) with a comparison pool large enough
    to cluster at the default min_cluster_size of 15."""
    df_baseline, _ = _comstock_hourly_all
    ids = sorted(df_baseline.index.get_level_values("id").unique())

    treatment_data = _hour_loadshape_data(df_baseline, ids[:8])
    comparison_pool_data = _hour_loadshape_data(df_baseline, ids[8:98])

    return treatment_data, comparison_pool_data


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