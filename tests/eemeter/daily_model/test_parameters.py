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

from opendsm.eemeter.models.daily.parameters import ModelCoefficients, ModelType



# (construction kwargs, coefficient_ids) per model type. The c_hdd_* id family
# is shared by heating/cooling-only models; from_np_arrays disambiguates by the
# sign of beta (negative -> heating), so the kwargs choose signs accordingly.
# Full-model breakpoints keep hdd_bp < cdd_bp to avoid the reorder branch.
_ROUNDTRIP = {
    ModelType.HDD_TIDD_CDD_SMOOTH: (
        dict(hdd_bp=40.0, hdd_beta=-1.5, hdd_k=0.1, cdd_bp=70.0, cdd_beta=2.0, cdd_k=0.2, intercept=10.0),
        ["hdd_bp", "hdd_beta", "hdd_k", "cdd_bp", "cdd_beta", "cdd_k", "intercept"],
    ),
    ModelType.HDD_TIDD_CDD: (
        dict(hdd_bp=40.0, hdd_beta=-1.5, cdd_bp=70.0, cdd_beta=2.0, intercept=10.0),
        ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"],
    ),
    ModelType.HDD_TIDD_SMOOTH: (
        dict(hdd_bp=50.0, hdd_beta=-1.5, hdd_k=0.1, intercept=10.0),
        ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"],
    ),
    ModelType.TIDD_CDD_SMOOTH: (
        dict(cdd_bp=70.0, cdd_beta=2.0, cdd_k=0.2, intercept=10.0),
        ["c_hdd_bp", "c_hdd_beta", "c_hdd_k", "intercept"],
    ),
    ModelType.HDD_TIDD: (
        dict(hdd_bp=50.0, hdd_beta=-1.5, intercept=10.0),
        ["c_hdd_bp", "c_hdd_beta", "intercept"],
    ),
    ModelType.TIDD_CDD: (
        dict(cdd_bp=70.0, cdd_beta=2.0, intercept=10.0),
        ["c_hdd_bp", "c_hdd_beta", "intercept"],
    ),
    ModelType.TIDD: (
        dict(intercept=10.0),
        ["intercept"],
    ),
}


@pytest.mark.parametrize("model_type", list(_ROUNDTRIP))
def test_to_np_array_from_np_arrays_roundtrip(model_type):
    """to_np_array -> from_np_arrays reconstructs the same model type and values."""
    kwargs, ids = _ROUNDTRIP[model_type]
    original = ModelCoefficients(model_type=model_type, **kwargs)

    array = original.to_np_array()
    restored = ModelCoefficients.from_np_arrays(array, ids)

    assert restored.model_type == model_type
    assert np.allclose(restored.to_np_array(), array)


def test_every_model_type_is_covered():
    """The roundtrip table exercises all defined model types."""
    assert set(_ROUNDTRIP) == set(ModelType)


def test_from_np_arrays_unknown_ids_raises():
    """Coefficient ids that match no known pattern raise ValueError."""
    with pytest.raises(ValueError):
        ModelCoefficients.from_np_arrays(np.array([1.0, 2.0]), ["bogus", "ids"])


def test_full_model_reorders_when_breakpoints_swapped():
    """from_np_arrays swaps heating/cooling when cdd_bp < hdd_bp (full model)."""
    ids = ["hdd_bp", "hdd_beta", "cdd_bp", "cdd_beta", "intercept"]
    # hdd_bp (70) > cdd_bp (40): the constructor reorders so hdd_bp < cdd_bp.
    coefficients = np.array([70.0, -1.5, 40.0, 2.0, 10.0])

    restored = ModelCoefficients.from_np_arrays(coefficients, ids)

    assert restored.hdd_bp == 40.0
    assert restored.cdd_bp == 70.0


def test_c_hdd_sign_selects_heating_vs_cooling():
    """The c_hdd family resolves to heating for negative beta, cooling for positive."""
    ids = ["c_hdd_bp", "c_hdd_beta", "intercept"]

    heating = ModelCoefficients.from_np_arrays(np.array([50.0, -1.5, 10.0]), ids)
    cooling = ModelCoefficients.from_np_arrays(np.array([70.0, 2.0, 10.0]), ids)

    assert heating.model_type == ModelType.HDD_TIDD
    assert cooling.model_type == ModelType.TIDD_CDD
