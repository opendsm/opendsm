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
from copy import deepcopy
from typing import List, Optional

import numpy as np
import numpy.ma as ma
import pandas as pd

# Unused imports kept for potential future use
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from scipy.interpolate import RBFInterpolator



# At or below this many lags the direct masked loop is used; above it the FFT
# path is used. Both paths return identical correlation values.
_AUTOCORR_FFT_LAG_THRESHOLD = 16


def _autocorr_fcn(x, lags, exclude_0=True):
    """Compute autocorrelation function for given lags.

    Computes non-partial autocorrelation, handling missing values via masked
    arrays, and returns both positive lags (future) and negative lags (past) by
    mirroring the correlation values. Few lags use a direct masked loop; many
    lags use an FFT with missing values zero-filled after centering, which
    contributes 0 to each lag product and so matches the masked-loop result.

    Args:
        x: Input array (may contain NaN values)
        lags: Array of lag values to compute correlation for
        exclude_0: Whether to exclude zero-lag correlation (always 1.0)

    Returns:
        2D array with shape (n_lags, 2) containing [lag, correlation] pairs
    """
    lags = np.asarray(lags)
    x_masked = ma.masked_invalid(x)
    mean = ma.mean(x_masked)
    var = ma.var(x_masked)

    if not np.isfinite(var) or var == 0:
        var = 1.0

    n = len(x)

    if len(lags) <= _AUTOCORR_FFT_LAG_THRESHOLD:
        x_centered = x_masked - mean
        corr = np.zeros(len(lags))

        for idx, lag in enumerate(lags):
            if lag == 0:
                corr[idx] = 1.0
            else:
                lag_corr = ma.sum(x_centered[lag:] * x_centered[:-lag]) / n / var
                corr[idx] = ma.filled(lag_corr, 0.0)

    else:
        x_filled = ma.filled(x_masked - mean, 0.0)
        spectrum = np.fft.rfft(x_filled, 2 * n)
        acov = np.fft.irfft(spectrum * np.conj(spectrum), 2 * n)[:n] / n / var
        safe_lags = np.clip(lags, 0, n - 1)
        corr = np.where(lags < n, acov[safe_lags], 0.0)
        corr[lags == 0] = 1.0

    result = np.vstack((lags, corr)).T

    # Handle zero-lag exclusion and create mirrored negative lags
    if exclude_0:
        result = result[1:]
        reversed_result = deepcopy(result)[::-1]

    else:
        reversed_result = deepcopy(result)[::-1][:-1]

    reversed_result[:, 0] = -reversed_result[:, 0]
    result = np.vstack((reversed_result, result))

    return result


def _multiple_imputation(df, columns=None, **kwargs):
    """Alternative imputation using sklearn IterativeImputer (unused).

    Uses Bayesian Ridge regression to iteratively impute missing values
    based on temporal features (hour, day, month). Generally slower than
    autocorrelation-based method but can handle complex patterns.
    """
    # Get indices of missing values
    missing_mask = df[columns].isna().any(axis=1)

    df_imputed = df[columns].reset_index()

    # Convert datetime to numerical features
    df_imputed["datetime_elapsed"] = (
        df_imputed["datetime"] - df_imputed["datetime"].min()
    ).dt.total_seconds() / 3600
    df_imputed["hour_of_day"] = df_imputed["datetime"].dt.hour
    df_imputed["day_of_week"] = df_imputed["datetime"].dt.dayofweek
    df_imputed["month"] = df_imputed["datetime"].dt.month
    df_imputed = df_imputed.set_index("datetime")

    # Configure imputer
    imputer_settings = {
        "estimator": BayesianRidge(),
        "max_iter": 10,
        "random_state": None,
    }
    imputer_settings.update(kwargs)

    imputer = IterativeImputer(**imputer_settings)
    imputer.fit(df_imputed)
    df_imputed[:] = imputer.transform(df_imputed)

    # Copy imputed values back to original dataframe
    df.loc[missing_mask, columns] = df_imputed.loc[missing_mask, columns]

    # Add interpolation flags
    for col in columns:
        flag_col = f"interpolated_{col}"
        df[flag_col] = False
        df.loc[missing_mask, flag_col] = True

    return df


def _shift_array(arr, num, fill_value=np.nan):
    """Shift array elements by num positions, filling with fill_value.

    Optimized for large arrays using pre-allocated memory to avoid
    concatenation overhead.

    Args:
        arr: Input array to shift
        num: Number of positions to shift (positive=right, negative=left)
        fill_value: Value to fill shifted positions with

    Returns:
        Shifted array with same shape as input
    """
    if num == 0:
        return arr.copy()

    result = np.empty_like(arr)

    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    else:  # num < 0
        result[num:] = fill_value
        result[:num] = arr[-num:]

    return result


def _interpolate_col(x, lags):
    """Interpolate missing values using autocorrelation-based method.

    Uses temporal patterns (lags/leads) with highest autocorrelation to
    fill missing values. Iteratively relaxes requirements to fill all gaps.

    Args:
        x: Pandas Series with potential missing values
        lags: Maximum lag to consider for autocorrelation

    Returns:
        Series with missing values filled
    """
    # Early exit if no missing values or all missing
    nan_count = x.isna().sum()
    if nan_count == 0 or nan_count == len(x):
        return x

    # Calculate number of autocorrelation indices to use
    if x.name == "observed":
        missing_frac = nan_count / len(x)
        n_cor_idx_heuristic = np.round((4.012 * np.log(missing_frac) + 24.38) / 2, 0) * 2
        n_cor_idx = int(np.max([6, n_cor_idx_heuristic]))
    else:
        n_cor_idx = 6

    # Calculate autocorrelation for lags 1 to lags
    lag_array = np.arange(lags + 1)
    autocorr = _autocorr_fcn(x.values, lag_array, exclude_0=True)

    # Select top n_cor_idx lags by correlation strength
    top_idx = np.argpartition(autocorr[:, 1], -n_cor_idx)[-n_cor_idx:]
    autocorr = autocorr[top_idx]
    autocorr = autocorr[np.argsort(autocorr[:, 1])[::-1]]
    best_lags = autocorr[:, 0].astype(int)

    # Pre-compute shifted arrays for all selected lags (avoid recomputing in loop)
    num_rows = len(x)
    num_lags = len(best_lags)
    shifted_arrays = np.empty((num_rows, num_lags))

    for col_idx, lag in enumerate(best_lags):
        shifted_arrays[:, col_idx] = _shift_array(x.values, lag)

    # Iteratively fill missing values, relaxing minimum valid helpers requirement
    max_iter = 10
    min_valid_counts = np.linspace(n_cor_idx, 1, max_iter).astype(int)

    for min_valid in min_valid_counts:
        # Get current missing value indices
        nan_mask = x.isna().values
        if not nan_mask.any():
            break

        nan_indices = np.where(nan_mask)[0]

        # Count valid (non-NaN) helpers for each missing position
        valid_helper_counts = np.sum(~np.isnan(shifted_arrays[nan_indices, :]), axis=1)
        can_fill_mask = valid_helper_counts >= min_valid

        if not can_fill_mask.any():
            continue

        # Fill positions that have enough valid helpers
        fillable_indices = nan_indices[can_fill_mask]
        fill_values = np.nanmean(shifted_arrays[fillable_indices, :], axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            x.iloc[fillable_indices] = fill_values

        # Update shifted arrays with newly filled values for next iteration
        for col_idx, lag in enumerate(best_lags):
            shifted_arrays[:, col_idx] = _shift_array(x.values, lag)

    return x


def interpolate(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Interpolate missing values in hourly time series data.

    Uses autocorrelation-based method to leverage temporal patterns, with
    fallback to simpler methods (time-based, forward fill, backward fill)
    for any remaining gaps. Adds boolean flags indicating interpolated values.

    The autocorr method is most effective for large gaps (24+ hours) where
    daily/weekly patterns can be leveraged. For very short time series,
    simpler methods are used directly.

    Args:
        df: DataFrame with datetime index and hourly data
        columns: List of column names to interpolate. Defaults to
                ["temperature", "ghi", "observed"]

    Returns:
        DataFrame with missing values filled and "interpolated_{col}" flag
        columns added for each interpolated column
    """
    # Determine lag window based on data length
    n_hours = len(df)
    HOURS_PER_DAY = 24
    HOURS_PER_WEEK = HOURS_PER_DAY * 7

    lags = None
    use_autocorr = True
    if n_hours > 6 * HOURS_PER_WEEK:
        lags = 2 * HOURS_PER_WEEK + 1
    elif n_hours > 3 * HOURS_PER_WEEK:
        lags = HOURS_PER_WEEK + 1
    elif n_hours > 3 * HOURS_PER_DAY:
        lags = HOURS_PER_DAY + 1
    else:
        use_autocorr = False

    # Default columns to interpolate
    if columns is None:
        columns = ["temperature", "ghi", "observed"]

    # Process each column
    for col in columns:
        # Skip if column doesn't exist
        if col not in df.columns:
            continue

        flag_col = f"interpolated_{col}"

        # Skip if interpolation flag already exists
        if flag_col in df.columns:
            continue

        # Store original missing indices
        missing_indices = df[col].isna()
        originally_missing = df.index[missing_indices]

        # Apply autocorr-based interpolation if data is long enough
        if use_autocorr:
            df[col] = _interpolate_col(df[col].copy(), lags)

        # Fallback methods for any remaining missing values
        backup_methods = ["time", "ffill", "bfill"]

        for method in backup_methods:
            if not df[col].isna().any():
                break

            if method == "time":
                df[col] = df[col].interpolate(method="time", limit_direction="both")
            elif method == "ffill":
                df[col] = df[col].ffill()
            elif method == "bfill":
                df[col] = df[col].bfill()
            else:
                raise ValueError(f"Unknown interpolation method: {method!r}")

        # Create interpolation flag: True where originally missing and now filled
        df[flag_col] = False
        was_filled = df.index.isin(originally_missing) & ~df[col].isna()
        df.loc[was_filled, flag_col] = True

    return df
