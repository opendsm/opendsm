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
import pytest
import numpy as np

from opendsm.eemeter.models.daily.optimize_results import (
    get_k,
    reduce_model,
    OptimizedResult,
)

from opendsm.eemeter.models.daily.parameters import ModelCoefficients
from opendsm.eemeter.models.daily.utilities.settings import DailySettings as Settings


def test_get_k():
    # Test case 1: Test when all values are within the bounds
    X = [60, 0.5, 80, 0.3]
    T_min_seg = 50
    T_max_seg = 90
    assert get_k(X, T_min_seg, T_max_seg) == [70.0, 10.0, 74.0, 6.0]

    # Test case 2: Test when hdd_bp is greater than T_max_seg
    X = [100, 0.5, 80, 0.3]
    T_min_seg = 50
    T_max_seg = 90
    assert get_k(X, T_min_seg, T_max_seg) == [100, 0.0, 86.0, -6.0]

    # Test case 3: Test when cdd_bp is less than T_min_seg
    X = [60, 0.5, 20, 0.3]
    T_min_seg = 50
    T_max_seg = 90
    assert get_k(X, T_min_seg, T_max_seg) == [40.0, -20.0, 20, 0.0]

    # Test case 4: Test when both hdd_k and cdd_k are zero
    X = [100, 0.5, 20, 0.3]
    T_min_seg = 50
    T_max_seg = 90
    assert get_k(X, T_min_seg, T_max_seg) == [20, 0.0, 20, 0.0]


@pytest.mark.parametrize(
    "hdd_bp, hdd_beta, pct_hdd_k, cdd_bp, cdd_beta, pct_cdd_k, intercept, T_min, T_max, T_min_seg, T_max_seg, model_key, expected_coef_id, expected_x",
    [
        # Test case 1
        (
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            0,
            100,
            20,
            80,
            "hdd_tidd_cdd_smooth",
            ["hdd_bp", "hdd_beta", "hdd_k", "cdd_bp", "cdd_beta", "cdd_k", "intercept"],
            [10, 20, 30, 40, 50, 60, 70],
        ),
        # Test case 2
        (
            10,
            20,
            0,
            40,
            50,
            60,
            70,
            0,
            100,
            20,
            80,
            "hdd_tidd_cdd_smooth",
            ["hdd_bp", "hdd_beta", "hdd_k", "cdd_bp", "cdd_beta", "cdd_k", "intercept"],
            [10, 20, 0, 40, 50, 60, 70],
        ),
        # Test case 3
        (
            10,
            0,
            30,
            40,
            50,
            60,
            70,
            0,
            100,
            20,
            80,
            "hdd_tidd_cdd_smooth",
            ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"],
            [20.0, 50.0, 20.0, 70.0],
        ),
        # Test case 4
        (
            10,
            20,
            0,
            40,
            50,
            0,
            70,
            0,
            100,
            20,
            80,
            "hdd_tidd_cdd_smooth",
            ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"],
            [10, 20, 40, 50, 70],
        ),
        # Test case 5
        (
            10,
            0,
            0,
            40,
            0,
            0,
            70,
            0,
            100,
            20,
            80,
            "hdd_tidd_cdd_smooth",
            ["intercept"],
            [70],
        ),
    ],
)
def test_reduce_model(
    hdd_bp,
    hdd_beta,
    pct_hdd_k,
    cdd_bp,
    cdd_beta,
    pct_cdd_k,
    intercept,
    T_min,
    T_max,
    T_min_seg,
    T_max_seg,
    model_key,
    expected_coef_id,
    expected_x,
):
    coef_id, x = reduce_model(
        hdd_bp,
        hdd_beta,
        pct_hdd_k,
        cdd_bp,
        cdd_beta,
        pct_cdd_k,
        intercept,
        T_min,
        T_max,
        T_min_seg,
        T_max_seg,
        model_key,
    )

    assert coef_id == expected_coef_id
    assert np.allclose(x, expected_x)


class TestOptimizeResult:
    @pytest.fixture
    def optimize_result(self):
        # create an instance of OptimizeResult for testing
        x = np.array([1, 2, 3, 4, 5, 6, 7])
        bnds = [[0, 10] * 7]
        coef_id = [
            "hdd_bp",
            "hdd_beta",
            "hdd_k",
            "cdd_bp",
            "cdd_beta",
            "cdd_k",
            "intercept",
        ]
        loss_alpha = 2.0
        C = np.array([1, 2, 3, 4, 5, 6, 7] * 2)
        T = np.array([1, 2, 3, 4, 5, 6, 7] * 2)
        model = np.array([1, 2, 3, 4, 5, 6, 7])
        resid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        weight = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        settings = Settings()
        jac = None
        mean_loss = 0.0
        TSS = 0.0
        success = True
        message = "Optimization terminated successfully."
        nfev = 10
        time_elapsed = 1.0
        return OptimizedResult(
            x,
            bnds,
            coef_id,
            loss_alpha,
            T,
            C,
            model,
            weight,
            resid,
            jac,
            mean_loss,
            TSS,
            success,
            message,
            nfev,
            time_elapsed,
            settings,
        )

    def test_named_coeffs(self, optimize_result):
        # test that named_coeffs is an instance of ModelCoefficients
        assert isinstance(optimize_result.named_coeffs, ModelCoefficients)

    def test_prediction_uncertainty(self, optimize_result):
        # test that _prediction_uncertainty sets f_unc correctly
        optimize_result._prediction_uncertainty()
        assert optimize_result.f_unc == pytest.approx(2.1556496051287013)

    def test_set_model_key(self, optimize_result):
        # test that _set_model_key sets model_key and model_name correctly
        optimize_result._set_model_key()
        assert optimize_result.model_key == "c_hdd_tidd"
        assert optimize_result.model_name == "cdd_tidd"

    def test_refine_model(self, optimize_result):
        # test that _refine_model sets coef_id and x correctly
        optimize_result._refine_model()
        assert optimize_result.coef_id == ["c_hdd_bp", "c_hdd_beta", "intercept"]
        assert optimize_result.x == pytest.approx(np.array([4, 5, 7]))

    def test_eval(self, optimize_result):
        # test that eval returns the correct values
        T = np.array([1, 2, 3, 4, 5])
        model, f_unc, hdd_load, cdd_load = optimize_result.eval(T)
        assert model == pytest.approx(np.array([7, 7, 7, 7, 12]))
        assert f_unc == pytest.approx(
            np.array([2.15564961, 2.15564961, 2.15564961, 2.15564961, 2.15564961])
        )
        assert hdd_load == pytest.approx(np.array([0, 0, 0, 0, 0]))
        assert cdd_load == pytest.approx(np.array([0, 0, 0, 0, 5]))
