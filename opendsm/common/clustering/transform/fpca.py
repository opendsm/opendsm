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

from __future__ import annotations

import numpy as np

from opendsm.common.clustering import settings as _settings

from opendsm.common.clustering.transform.normalize import normalize
from opendsm.common.clustering.transform.parallel_analysis import (
    _fpca_explained_variance,
    _fpca_transform_with_n,
    _parallel_analysis_n_components,
)


# ---------------------------------------------------------------------------
# FPCA transform
# ---------------------------------------------------------------------------

class FpcaError(Exception):
    pass


def _fpca_base(
    x: np.ndarray,
    y: np.ndarray,
    min_var_ratio: float,
) -> np.ndarray:
    """FPCA with automatic n_components via variance-ratio threshold.

    Two-pass: first fit (n_max components) determines n from the cumulative
    explained variance ratio; second fit uses exactly n components so the
    Fourier basis size is appropriate for the retained dimensionality.
    """
    if 0 >= min_var_ratio or min_var_ratio >= 1:
        raise FpcaError("min_var_ratio but be greater than 0 and less than 1")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise FpcaError("provided non finite values for fpca")
    if len(x) == 0 or len(y) == 0:
        raise FpcaError("provided empty values for fpca")

    n_max = max(1, int(np.min(np.array(np.shape(y)) - [1, 5])))

    eig_ratios = _fpca_explained_variance(y, x, n_max)
    var_ratio_arr = np.cumsum(eig_ratios) - min_var_ratio
    n = int(np.argmin(var_ratio_arr < 0.0) + 1)

    return _fpca_transform_with_n(x, y, n)


def fpca_transform(
    data: np.ndarray,
    settings: _settings.ClusteringSettings,
) -> np.ndarray:
    fpca_settings = settings.feature_transform.fpca

    if not np.all(np.isfinite(data)):
        raise FpcaError("provided non finite values for fpca")
    if len(data) == 0:
        raise FpcaError("provided empty values for fpca")

    x = np.arange(data.shape[1])
    seed = settings._seed if settings._seed is not None else 0

    if fpca_settings.use_parallel_analysis:
        n = _parallel_analysis_n_components(
            data, method="fpca", grid_points=x, seed=seed
        )
        result = _fpca_transform_with_n(x, data, n)
    else:
        result = _fpca_base(x, data, fpca_settings.min_var_ratio)

    norm_settings = settings.feature_transform.normalize
    if norm_settings.enabled:
        result = normalize(result, norm_settings, axis=0)

    return result
