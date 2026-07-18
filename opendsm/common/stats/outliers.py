#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2024 OpenEEmeter contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import numba
import numpy as np

from opendsm.common.stats.basic import _weighted_quantile
from opendsm.common.utils import to_np_array


def IQR_outlier(data, weights=None, sigma_threshold=3, quantile=0.25):
    data = to_np_array(data)

    if weights is not None:
        weights = to_np_array(weights)

    return _IQR_outlier(data, weights, sigma_threshold, quantile)


@numba.jit(nopython=True, cache=True)
def _IQR_outlier(data, weights=None, sigma_threshold=3, quantile=0.25):
    # only use finite data
    if weights is None:
        q13 = np.nanquantile(data[np.isfinite(data)], [quantile, 1 - quantile])
    else:  # weighted_quantile could be used always, don't know speed
        q13 = _weighted_quantile(
            data[np.isfinite(data)], np.array([quantile, 1 - quantile]), weights=weights
        )

    q13_scalar = (
        0.7413 * sigma_threshold - 0.5
    )  # this is a pretty good fit to get the scalar for any sigma
    iqr = np.diff(q13)[0] * q13_scalar
    outlier_threshold = np.array([q13[0] - iqr, q13[1] + iqr])

    return outlier_threshold


def remove_outliers(x, weights=None, sigma_threshold=3, quantile=0.25):
    # if all values are the same return back all indices
    if len(np.unique(x)) == 1:
        return x, np.arange(len(x))

    # prevent x_no_outliers from being empty
    for sigma_added in range(10):
        outlier_bnds = _IQR_outlier(x, weights, sigma_threshold + sigma_added, quantile)
        idx_no_outliers = np.argwhere((x >= outlier_bnds[0]) & (x <= outlier_bnds[1])).flatten()

        if idx_no_outliers.size > 0:
            break

    # if idx_no_outliers is empty, keep the closest meter to the outlier bounds.
    # Unreachable for finite input (the q1..q3 interval always retains points);
    # the only path here is all-non-finite input, which raises in _IQR_outlier
    # under NUMBA_DISABLE_JIT, so the block cannot be exercised in coverage.
    if len(idx_no_outliers) == 0:  # pragma: no cover
        # distance between x and outlier bounds
        dist = -np.minimum(x - outlier_bnds[0], outlier_bnds[1] - x)

        # sort by distance
        # idx_no_outliers = np.argsort(dist)

        # select closest
        idx_no_outliers = np.array([np.argmin(dist)])

    x_no_outliers = x[idx_no_outliers]

    return x_no_outliers, idx_no_outliers