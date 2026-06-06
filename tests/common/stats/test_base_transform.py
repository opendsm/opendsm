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

from opendsm.common.stats.distribution_transform import (
    Bisymlog,
    BoxCox,
    Standardize,
    YeoJohnson,
)



# Contract shared by every TransformBase subclass. BoxCox requires strictly
# positive input, so the shared fixtures use positive data for all four.

TRANSFORMS = [Standardize, Bisymlog, YeoJohnson, BoxCox]


@pytest.fixture
def positive_2d():
    """Two positive-valued feature columns with distinct distributions."""
    rng = np.random.default_rng(0)
    a = rng.lognormal(0.0, 0.5, 200)
    b = rng.uniform(1.0, 10.0, 200)
    data = np.column_stack([a, b])

    return data


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_fit_transform_inverse_roundtrip(cls, positive_2d):
    """inverse_transform undoes transform back to the original input."""
    transform = cls()
    transformed = transform.fit_transform(positive_2d)
    recovered = transform.inverse_transform(transformed)

    assert recovered.shape == positive_2d.shape
    assert np.allclose(recovered, positive_2d, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_fit_then_transform_matches_fit_transform(cls, positive_2d):
    """A separate fit + transform equals the fused fit_transform."""
    fused = cls().fit_transform(positive_2d)
    staged = cls().fit(positive_2d).transform(positive_2d)

    assert np.allclose(fused, staged, atol=1e-9)


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_one_dimensional_input_preserved(cls):
    """A 1-D input transforms and inverts back to a 1-D array of the same length."""
    x = np.linspace(1.0, 20.0, 50)

    transform = cls()
    transformed = transform.fit_transform(x)

    assert transformed.ndim == 1
    assert transformed.shape == x.shape
    assert np.allclose(transform.inverse_transform(transformed), x, atol=1e-6)


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_to_dict_from_dict_roundtrip(cls, positive_2d):
    """from_dict(to_dict()) rebuilds a transform that produces identical output."""
    fitted = cls().fit(positive_2d)
    restored = cls.from_dict(fitted.to_dict())

    assert np.allclose(
        fitted.transform(positive_2d), restored.transform(positive_2d), atol=1e-9
    )


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_to_json_from_json_roundtrip(cls, positive_2d):
    """JSON serialisation round-trips to an equivalent transform."""
    fitted = cls().fit(positive_2d)
    restored = cls.from_json(fitted.to_json())

    assert np.allclose(
        fitted.transform(positive_2d), restored.transform(positive_2d), atol=1e-9
    )


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_transform_before_fit_raises(cls, positive_2d):
    """Calling transform on an unfitted instance raises RuntimeError."""
    with pytest.raises(RuntimeError, match="not fitted"):
        cls().transform(positive_2d)


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_wrong_n_features_raises(cls, positive_2d):
    """Transforming with a different feature count than fit raises ValueError."""
    fitted = cls().fit(positive_2d)

    with pytest.raises(ValueError, match="features"):
        fitted.transform(positive_2d[:, :1])


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_three_dimensional_input_raises(cls):
    """A 3-D array is rejected at validation."""
    with pytest.raises(ValueError, match="1-D or 2-D"):
        cls().fit(np.ones((4, 4, 4)))


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_nan_entries_pass_through(cls, positive_2d):
    """NaNs are preserved in place; finite values still round-trip."""
    data = positive_2d.copy()
    data[5, 0] = np.nan

    transform = cls()
    transformed = transform.fit_transform(data)

    assert np.isnan(transformed[5, 0])
    recovered = transform.inverse_transform(transformed)
    finite = np.isfinite(data)
    assert np.allclose(recovered[finite], data[finite], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_too_few_samples_skips_dimension(cls):
    """Fewer finite samples than min_samples leaves the dimension untouched."""
    transform = cls()
    n = transform.min_samples - 1
    data = np.linspace(1.0, 2.0, n).reshape(-1, 1)

    transformed = transform.fit_transform(data)

    assert transform.skip_dims_[0]
    assert np.allclose(transformed, data)


@pytest.mark.parametrize("cls", TRANSFORMS)
def test_zero_variance_column_skipped(cls):
    """A constant (zero-variance) column is skipped and passes through unchanged."""
    data = np.full((50, 1), 3.0)

    transform = cls()
    transformed = transform.fit_transform(data)

    assert transform.skip_dims_[0]
    assert np.allclose(transformed, 3.0)


def test_box_cox_skips_non_positive_column():
    """BoxCox requires positive data; a column with non-positive values is skipped."""
    data = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)

    transform = BoxCox()
    with pytest.warns(RuntimeWarning, match="non-positive"):
        transformed = transform.fit_transform(data)

    assert transform.skip_dims_[0]
    assert np.allclose(transformed, data)
