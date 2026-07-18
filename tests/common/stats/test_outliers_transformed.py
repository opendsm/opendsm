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
"""Tests for the transform-dispatch layer in outliers_transformed.

Guards the alignment between the ``TransformChoice`` enum (the values a
caller supplies via ``OutlierRejectionSettings.transform``) and the
``_TRANSFORMS`` dispatch dict. A drift between the two would silently route
every comparison-group outlier rejection into the unknown-transform error
path.
"""

import numpy as np
import pytest

from opendsm.common.stats.outliers_transformed import remove_outliers, _TRANSFORMS
from opendsm.comparison_groups.savings.settings import TransformChoice


# Strictly positive so the Box-Cox transforms are exercised rather than
# skipped; one extreme value to give the outlier detector something to drop.
_DATA = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0])


def test_every_transformchoice_member_is_dispatchable():
    """Every enum member resolves through the dispatch dict.

    Membership is tested with the enum member itself (not ``.value``)
    because that is what ``model_correction.py`` passes; ``str, Enum``
    members must hash/compare by their string value for this to hold.
    """
    for choice in TransformChoice:
        assert choice in _TRANSFORMS


@pytest.mark.parametrize("choice", list(TransformChoice), ids=lambda c: c.value)
def test_remove_outliers_dispatches_each_transform(choice):
    """Passing each enum member runs without error and returns a valid subset."""
    x_kept, idx_kept = remove_outliers(_DATA, transform=choice)

    assert len(idx_kept) <= len(_DATA)
    assert np.all(np.asarray(idx_kept) < len(_DATA))
    assert len(x_kept) == len(idx_kept)


def test_unknown_transform_raises():
    with pytest.raises(ValueError, match="Unknown transform"):
        remove_outliers(_DATA, transform="not_a_transform")


def test_none_transform_passes_through_to_basic_removal():
    """transform=None still performs basic (untransformed) outlier removal."""
    x_kept, idx_kept = remove_outliers(_DATA, transform=None)

    assert len(idx_kept) <= len(_DATA)
    assert np.all(np.asarray(idx_kept) < len(_DATA))
