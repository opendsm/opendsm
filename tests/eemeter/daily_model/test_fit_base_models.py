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
from opendsm.eemeter.models.daily.parameters import ModelCoefficients

from opendsm.eemeter.models.daily.utilities.settings import DailySettings as Settings
from opendsm.eemeter.models.daily.parameters import ModelType
from opendsm.eemeter.models.daily.fit_base_models import (
    fit_initial_models_from_full_model,
    fit_model,
    fit_final_model,
    _get_opt_settings,
)

from opendsm.eemeter.models.daily.optimize_results import OptimizedResult


@pytest.fixture
def meter_data():
    df_meter = pd.DataFrame(
        {
            "temperature": [
                10.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                65.0,
                70.0,
                75.0,
                80.0,
                85.0,
                90.0,
                95.0,
                100.0,
            ],
            "observed": [
                100.0,
                150.0,
                200.0,
                250.0,
                300.0,
                350.0,
                400.0,
                450.0,
                500.0,
                550.0,
                600.0,
                650.0,
                700.0,
                750.0,
                800.0,
                850.0,
                900.0,
                950.0,
                1000.0,
            ],
            "datetime": [
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
                "2022-01-06",
                "2022-01-07",
                "2022-01-08",
                "2022-01-09",
                "2022-01-10",
                "2022-01-11",
                "2022-01-12",
                "2022-01-13",
                "2022-01-14",
                "2022-01-15",
                "2022-01-16",
                "2022-01-17",
                "2022-01-18",
                "2022-01-19",
            ],
        }
    )
    df_meter["datetime"] = pd.to_datetime(df_meter["datetime"])
    df_meter.set_index("datetime", inplace=True)
    return df_meter


@pytest.fixture
def get_settings():
    return Settings()


@pytest.fixture
def get_optimized_result(get_settings):
    return OptimizedResult(
        x=np.array([1, 2, 3, 4]),
        bnds=[
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [0, 7],
            [0, 8],
            [0, 9],
            [0, 10],
        ],
        coef_id=["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"],
        loss_alpha=0.1,
        C=0.5,
        T=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        model=np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]),
        weight=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        resid=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        jac=None,
        mean_loss=0.2,
        TSS=0.3,
        success=True,
        message="Optimization successful.",
        nfev=10,
        time_elapsed=0.5,
        settings=get_settings,
    )


def test_fit_initial_models_from_full_model(meter_data, get_settings):
    # Test case 1: Test the function with a sample dataset
    model_res = fit_initial_models_from_full_model(meter_data, get_settings)
    assert isinstance(model_res, OptimizedResult)

    # Test case 3: Test the function with a dataset that has missing values
    T = np.array([10, 20, 30, 40, 50])
    obs = np.array([1, 2, np.nan, 4, 5])
    model_res = fit_initial_models_from_full_model(meter_data, get_settings)
    assert isinstance(model_res, OptimizedResult)

    # Test case 4: Test the function with a dataset that has negative values
    T = np.array([10, 20, 30, 40, 50])
    obs = np.array([-1, 2, 3, 4, 5])
    model_res = fit_initial_models_from_full_model(meter_data, get_settings)
    assert isinstance(model_res, OptimizedResult)


def test_fit_model(meter_data, get_settings):
    # Test case 1: Test for model_key = "hdd_tidd_cdd_smooth"
    T = np.array(meter_data["temperature"])
    obs = np.array(meter_data["observed"])
    weights = None

    fit_input = [T, obs, weights, get_settings, _get_opt_settings(get_settings)]
    res = fit_model("hdd_tidd_cdd_smooth", fit_input, None, None)
    assert isinstance(res, OptimizedResult)

    # Test case 2: Test for model_key = "hdd_tidd_cdd"
    res = fit_model("hdd_tidd_cdd", fit_input, None, None)
    assert isinstance(res, OptimizedResult)

    # Test case 3: Test for model_key = "c_hdd_tidd_smooth"
    res = fit_model("c_hdd_tidd_smooth", fit_input, None, None)
    assert isinstance(res, OptimizedResult)

    # Test case 4: Test for model_key = "c_hdd_tidd"
    x0 = ModelCoefficients(
        model_type=ModelType.HDD_TIDD,
        cdd_beta=1.747475624458497,
        cdd_bp=74.69216148926878,
        cdd_k=0.2548934690459498,
        hdd_beta=1.308196391571347,
        hdd_bp=63.332029669746596,
        hdd_k=0.0,
        intercept=49.97929032502437,
    )
    res = fit_model("c_hdd_tidd", fit_input, x0, None)
    assert isinstance(res, OptimizedResult)

    # Test case 5: Test for model_key = "tidd"
    res = fit_model("tidd", fit_input, None, None)
    assert isinstance(res, OptimizedResult)

    # TODO: add more data specific cases


def test_fit_final_model(meter_data, get_settings, get_optimized_result):
    # Test case 1: Test if the function returns an instance of OptimizedResult
    HoF = get_optimized_result

    res = fit_final_model(meter_data, HoF, get_settings)
    assert isinstance(res, OptimizedResult)

    # Test case 2: Test if the function raises a TypeError when the input arguments are of the wrong type
    with pytest.raises(TypeError):
        fit_final_model("not a dataframe", "not an OptimizedResult", "not a dictionary")
