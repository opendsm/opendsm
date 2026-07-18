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
import pytest

from opendsm.comparison_groups.random_sampling.create_comparison_groups import Random_Sampling
from opendsm.comparison_groups.random_sampling.settings import Settings


@pytest.fixture
def fitted_algorithm(cg_loadshape_data):
    """A Comparison_Group_Algorithm subclass with loadshape attributes populated,
    exercising the shared base-class loadshape methods."""
    treatment_data, comparison_pool_data = cg_loadshape_data
    algorithm = Random_Sampling(Settings(n_meters_total=10, n_meters_per_treatment=None, seed=1))
    algorithm.get_comparison_group(treatment_data, comparison_pool_data)

    return algorithm


def test_get_loadshapes_returns_three_aggregate_rows(fitted_algorithm):
    loadshapes = fitted_algorithm.get_loadshapes()

    assert len(loadshapes) == 3  # treatment, comparison group, comparison pool
    assert loadshapes.shape[1] == 24


def test_get_comparison_pool_loadshape_is_single_labeled_row(fitted_algorithm):
    pool_loadshape = fitted_algorithm.get_comparison_pool_loadshape()

    assert list(pool_loadshape.index) == ["Comparison Pool"]


def test_plot_loadshapes_returns_a_figure(fitted_algorithm):
    figure = fitted_algorithm.plot_loadshapes()

    assert figure is not None


def test_validate_ls_weights_none_returns_none(fitted_algorithm):
    assert fitted_algorithm._validate_ls_weights(None) is None


def test_validate_ls_weights_equal_returns_none(fitted_algorithm):
    n_features = fitted_algorithm.treatment_loadshape.shape[1]

    assert fitted_algorithm._validate_ls_weights([1.0] * n_features) is None


def test_validate_ls_weights_normalizes_to_one(fitted_algorithm):
    n_features = fitted_algorithm.treatment_loadshape.shape[1]
    weights = np.arange(1, n_features + 1, dtype=float)

    normalized = fitted_algorithm._validate_ls_weights(weights)

    np.testing.assert_allclose(np.sum(normalized), 1.0)


def test_validate_ls_weights_wrong_length_raises(fitted_algorithm):
    with pytest.raises(ValueError):
        fitted_algorithm._validate_ls_weights([1.0, 2.0, 3.0])


def test_get_loadshapes_before_fit_raises():
    """get_loadshapes() before get_comparison_group() has no loadshapes to return.

    The contract is fit-then-query; calling it on an unfitted instance currently
    surfaces an opaque AttributeError (the treatment data is never populated)
    rather than a guard with a clear "fit first" message.
    """
    unfitted = Random_Sampling(Settings(n_meters_total=10, n_meters_per_treatment=None, seed=1))

    with pytest.raises(AttributeError):
        unfitted.get_loadshapes()
