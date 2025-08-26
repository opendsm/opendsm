#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations


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

"""
module which contains (for now) both utilities
for clustering transforms and the transform logic itself
"""

import numpy as np
import pandas as pd

from skfda.representation.grid import FDataGrid as _FDataGrid
from skfda.representation.basis import Fourier as _Fourier
from skfda.preprocessing.dim_reduction.feature_extraction import FPCA as _FPCA


_NORMALIZATION_QUANTILE = 0.1


def _normalize_single_loadshape(ls_arr: np.ndarray):
    """
    applies the min and max normalization logic to a dataframe which contains the transformed
    loadshape
    """
    ls_arr = np.array(ls_arr)
    ls_arr_transposed = ls_arr.T
    a, b = [0, 1]  # range to normalize to

    # current range
    ls_min, ls_max = np.quantile(
        ls_arr_transposed,
        [_NORMALIZATION_QUANTILE, 1 - _NORMALIZATION_QUANTILE],
        axis=0,
    )

    if ls_min == ls_max:
        return np.full_like(ls_arr, (a + b) / 2)

    ret_arr = (b - a) * (ls_arr_transposed - ls_min) / (ls_max - ls_min) + a

    return ret_arr.T


def _normalize_df_loadshapes(
    df: pd.DataFrame, normalize_method="min_max"
) -> pd.DataFrame:
    """
    transforms a loadshape dataframe

    It can work either on a dataframe containing all treatment loadshapes
    or a single loadshape.
    """
    # df_list: list[pd.DataFrame] = []
    # # TODO: This is slow. Need to vectorize or use apply?
    # for _id, data in df.iterrows():
    #     transformed_data = _normalize_loadshape(
    #         ls_arr=data.values
    #     )
    #     df_list.append(transformed_data)
    #
    # df_transformed = pd.concat(df_list).to_frame(name="ls")  # type: ignore

    if normalize_method == "min_max":
        df = df.apply(_normalize_single_loadshape, result_type="broadcast", axis=1)

    return df


class FpcaError(Exception):
    pass


def _fpca_base(x: np.ndarray, y: np.ndarray, min_var_ratio: float) -> np.ndarray:
    """
    applies fpca to concatenated transform loadshape dataframe values

    x -> time converted to np array taken from loadshape dataframe
    y -> transformed values

    assumes mixture_components return and fourier basis

    also may return a string as second return value. if it is not None, it implies an error occurred
    """

    if 0 >= min_var_ratio or min_var_ratio >= 1:
        raise FpcaError("min_var_ratio but be greater than 0 and less than 1")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise FpcaError("provided non finite values for fpca")

    if len(x) == 0 or len(y) == 0:
        raise FpcaError("provided empty values for fpca")

    n_min = 1
    # get maximum n components

    # smallest 1  || min(largest = number of samples - 1, # time points)
    n_max = np.min(np.array(np.shape(y)) - [1, 5])
    if n_max < n_min:
        n_max = n_min

    n_max = int(n_max)

    # get maximum principle components
    fd = _FDataGrid(grid_points=x, data_matrix=y)
    basis_fcn = _Fourier

    basis_fd = fd.to_basis(basis_fcn(n_basis=n_max + 4))
    fpca = _FPCA(n_components=n_max, components_basis=basis_fcn(n_basis=n_max + 4))
    fpca.fit(basis_fd)

    var_ratio = np.cumsum(fpca.explained_variance_ratio_) - min_var_ratio
    n = int(np.argmin(var_ratio < 0.0) + 1)

    basis_fd = fd.to_basis(basis_fcn(n_basis=n + 4))
    fpca = _FPCA(n_components=n, components_basis=basis_fcn(n_basis=n + 4))
    fpca.fit(basis_fd)

    mixture_components = fpca.transform(basis_fd)

    return mixture_components


def _get_fpca_from_loadshape(
    df: pd.DataFrame,
    min_var_ratio: float,
) -> tuple[pd.DataFrame, str | None]:
    """
    function which receives concatenated dataframe of normalized loadshape
    dataframes and a min_var_ratio and then performs fpca.

    Also returns a string that if not empty, implies an error message/something went wrong
    """
    ids = df.index
    time = df.columns.to_numpy().astype(float)
    ls = df.values

    try:
        fcpa_mixture_components = _fpca_base(x=time, y=ls, min_var_ratio=min_var_ratio)
    except FpcaError as e:
        return pd.DataFrame(), str(e)

    fpca_components = np.arange(fcpa_mixture_components.shape[1])
    columns = pd.Series(fpca_components, name="fPCA_component")

    df_fpca = pd.DataFrame(fcpa_mixture_components, index=ids, columns=columns)

    return df_fpca, None
