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

import warnings

import numba
import numpy as np
import numpy.ma as ma

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from scipy.interpolate import RBFInterpolator

from copy import deepcopy as copy


def autocorr_fcn(x, lags, exclude_0=True):
    """manualy compute, non partial"""
    x_msk = ma.masked_invalid(x)
    mean = ma.mean(x_msk)
    var = ma.var(x_msk)
    xp = x_msk - mean
    corr = [1.0 if l == 0 else ma.sum(xp[l:] * xp[:-l]) / len(x) / var for l in lags]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Warning: converting a masked element to nan")
        # combine the lags, the correlation values, and mirror to get leads/lags
        res = np.vstack((lags, corr)).T
    if exclude_0:  # remove the 0 lag
        res = res[1:]
        rev_res = copy(res)[::-1]
    else:
        rev_res = copy(res)[::-1][:-1]

    rev_res[:, 0] = -rev_res[:, 0]
    res = np.vstack((rev_res, res))

    return res


# unused
def autocorr_fcn2(x, lags):
    """np.correlate, non partial"""
    x_msk = ma.masked_invalid(x)
    mean = ma.mean(x_msk)
    var = ma.var(x_msk)
    xp = x_msk - mean

    corr = ma.correlate(xp, xp, "full")[len(x) - 1 :] / var / len(x)

    return corr[: len(lags)]


# unused
def autocorr_fcn3(x, lags):
    """fft, pad 0s, non partial"""
    x_msk = ma.masked_invalid(x)
    n = len(x)
    # pad 0s to 2n-1
    ext_size = 2 * n - 1
    # nearest power of 2
    fsize = 2 ** np.ceil(np.log2(ext_size)).astype("int")

    mean = ma.mean(x_msk)
    var = ma.var(x_msk)
    xp = x - mean

    # do fft and ifft
    cf = np.fft.fft(xp, fsize)
    sf = cf.conjugate() * cf
    corr = np.fft.ifft(sf).real
    corr = corr / var / n

    return corr[: len(lags)]


# unused
def multiple_imputation(df, columns=None, **kwargs):
    # get indices of missing values
    missing_idx = df[columns].isna().any(axis=1)

    df_imputed = df[columns].reset_index()
    # convert datetime to hours since earliest datetime
    df_imputed["datetime_elapsed"] = (
        df_imputed["datetime"] - df_imputed["datetime"].min()
    ).dt.total_seconds() / 3600
    df_imputed["hour_of_day"] = df_imputed["datetime"].dt.hour
    df_imputed["day_of_week"] = df_imputed["datetime"].dt.dayofweek
    df_imputed["month"] = df_imputed["datetime"].dt.month
    df_imputed = df_imputed.set_index("datetime")

    settings_dict = {
        "estimator": BayesianRidge(),  # can use SVR, BayesianRidge, etc.
        "max_iter": 10,
        "random_state": None,
    }
    settings_dict.update(kwargs)

    imputer = IterativeImputer(**settings_dict)
    imputer.fit(df_imputed)
    df_imputed[:] = imputer.transform(df_imputed)

    # add df_imputed back to df
    df.loc[missing_idx, columns] = df_imputed.loc[missing_idx, columns]

    # add additional columns to indicate which values were imputed
    for col in columns:
        interp_bool_col = f"interpolated_{col}"

        df[interp_bool_col] = False
        df.loc[missing_idx, interp_bool_col] = True

    return df


# @numba.njit
def shift_array(arr, num, fill_value=np.nan):
    # Courtesy of https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
    # get size of arr
    arr_size = arr.shape[0]

    if arr_size <= 20000:
        if num >= 0:
            return np.concatenate((np.full(num, fill_value), arr[:-num]))
        else:
            return np.concatenate((arr[-num:], np.full(-num, fill_value)))
    else:
        result = np.empty_like(arr)

        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr

        return result


def _interpolate_col(x, lags):
    # check that the column has nans
    if x.isna().sum() == 0:
        return x

    elif x.isna().sum() == len(x):
        return x

    # calculate the number of lags and leads to consider
    if x.name == "observed":
        missing_frac = x.isna().sum() / len(x)
        n_cor_idx_heuristic = (
            np.round((4.012 * np.log(missing_frac) + 24.38) / 2, 0) * 2
        )
        n_cor_idx = int(np.max([6, n_cor_idx_heuristic]))
    else:
        n_cor_idx = 6

    # Calculate the correlation of col with its lags and leads
    # create lags from -lags to lags
    lag_array = np.arange(lags + 1)
    autocorr = autocorr_fcn(x.values, lag_array, exclude_0=True)

    # take the largest n_cor_idx from second column using argpartition
    idx = np.argpartition(autocorr[:, 1], -n_cor_idx)[-n_cor_idx:]
    autocorr = autocorr[idx]

    # sort autocorr by the autocorrelation value
    autocorr = autocorr[np.argsort(autocorr[:, 1])[::-1]]
    autocorr_idx = autocorr[:, 0]

    # interpolate and update the values
    max_iter = 10
    for i, cnt_min in enumerate(np.linspace(n_cor_idx, 1, max_iter).astype(int)):
        num_rows = x.shape[0]
        num_cols = len(autocorr_idx)

        autocorr_helpers = np.empty((num_rows, num_cols))
        for i in range(num_cols):
            shift = int(autocorr_idx[i])
            autocorr_helpers[:, i] = shift_array(x.values, shift)

        # get the indices of the missing values
        nan_series_idx = x.index[x.isna()]
        nan_idx = x.index.get_indexer(nan_series_idx)

        # nan values where helpers are not nan
        valid_idx = np.sum(~np.isnan(autocorr_helpers[nan_idx, :]), axis=1) >= cnt_min
        if valid_idx.sum() == 0:
            continue

        nan_series_idx = nan_series_idx[valid_idx]
        nan_idx = x.index.get_indexer(nan_series_idx)

        # for each row, if the value is missing, calculate the mean of the lags and leads
        # ignore FutureWarning from pandas for now
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            x.loc[nan_series_idx] = np.nanmean(autocorr_helpers[nan_idx, :], axis=1)

        if x.isna().sum() == 0:
            break

    return x


def interpolate(df, columns=None):
    skip_autocorr_interpolation = False
    if len(df) > 6 * 24 * 7:
        lags = 24 * 7 * 2 + 1
    elif (len(df) > 3 * 24 * 7) and (len(df) <= 6 * 24 * 7):
        lags = 24 * 7 + 1
    elif (len(df) > 3 * 24) and (len(df) <= 3 * 24 * 7):
        lags = 24 + 1
    else:
        skip_autocorr_interpolation = True

    interp_cols = columns
    if interp_cols is None:
        interp_cols = ["temperature", "ghi", "observed"]

    # check if the columns are in the dataframe and modify columns appropriately
    for col in interp_cols:
        if col not in df.columns:
            continue

        interp_bool_col = f"interpolated_{col}"
        if interp_bool_col in df.columns:
            continue

        # main interpolation method
        idx_missing = df.loc[df[col].isna()].index
        if not skip_autocorr_interpolation:
            df[col] = _interpolate_col(df[col].copy(), lags)

        # backup interpolation methods
        for method in ["time", "ffill", "bfill"]:
            na_datetime = df.loc[df[col].isna()].index
            if len(na_datetime) == 0:
                break

            if method == "time":
                df[col] = df[col].interpolate(method="time", limit_direction="both")

            elif method == "ffill":
                df[col] = df[col].ffill()

            elif method == "bfill":
                df[col] = df[col].bfill()

        # TODO: we can check if we have similar values multiple times back to back, if yes, raise a warning
        # where na_datetime_original is True and the col is not na, set the interpolation boolean to True
        df[interp_bool_col] = False
        df.loc[df.index.isin(idx_missing) & ~df[col].isna(), interp_bool_col] = True

    return df
