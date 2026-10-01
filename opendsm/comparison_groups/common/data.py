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

from copy import deepcopy
from typing import Optional

from opendsm.comparison_groups.common.data_settings import Data_Settings
from opendsm.comparison_groups.common import const as _const
import pandas as pd
import numpy as np


def is_datetime(x: pd.Series) -> bool:
    is_dt = [
        pd.api.types.is_datetime64_any_dtype(x),
        # pd.api.types.is_datetime64_ns_dtype(x),
        # pd.api.types.is_datetime64_dtype(x),
        # isinstance(x.dtype, pd.DatetimeTZDtype)
    ]

    return any(is_dt)


# Subtracted from each grouping column so every time digit is 0-based
_time_digit_offset = {"month": 1, "day_of_year": 1}


def fill_missing(
    matrix: np.ndarray,
    layout: Optional[list[tuple[str, int]]],
    method: str = "linear",
) -> np.ndarray:
    """Fill the NaN times of a meters x times loadshape matrix, returning a filled copy.

    Linear interpolation along each row is the only method. When layout contains a
    season column, season is the outermost block, so each row is interpolated
    within each season block and never draws on another season's readings. Without a
    season column, or with layout None, the whole row is one block. Times before the
    first or after the last reading of a block take that reading; a block with no
    readings stays NaN.

    Args:
        matrix: Meters x times loadshape values, NaN where a time is missing.
        layout: The time layout as ``(column, count)`` pairs, outermost first, or None.
        method: The fill method; only ``"linear"`` is supported.

    Returns:
        A filled copy of ``matrix``.

    Raises:
        ValueError: method is not "linear".
    """
    if method != "linear":
        raise ValueError(f"Unknown fill method {method!r}; only 'linear' is supported.")

    filled = np.array(matrix, dtype=float)
    n_blocks = dict(layout or []).get("season", 1)
    block_len = filled.shape[1] // n_blocks
    positions = np.arange(block_len)

    for start in range(0, filled.shape[1], block_len):
        for row in filled[:, start:start + block_len]:
            missing = np.isnan(row)
            if missing.any() and not missing.all():
                row[missing] = np.interp(positions[missing], positions[~missing], row[~missing])

    return filled


def _aggregate_cells(
    key: np.ndarray, values: np.ndarray, n_cells: int, agg_type: _const.AggType
) -> np.ndarray:
    """Aggregate the values sharing each integer cell key in [0, n_cells).

    values must hold no NaN. A cell without values is NaN.

    Raises:
        ValueError: agg_type is neither mean nor median.
    """
    aggregated = np.full(n_cells, np.nan)

    if agg_type == _const.AggType.MEAN:
        counts = np.bincount(key, minlength=n_cells)
        sums = np.bincount(key, weights=values, minlength=n_cells)
        np.divide(sums, counts, out=aggregated, where=counts > 0)

    elif agg_type == _const.AggType.MEDIAN:
        medians = pd.Series(values).groupby(key).median()
        aggregated[medians.index.to_numpy()] = medians.to_numpy()

    else:
        raise ValueError(f"Unknown agg_type {agg_type!r}")

    return aggregated


class Data:
    def __init__(self, 
        loadshape_df: Optional[pd.DataFrame] = None, 
        time_series_df: Optional[pd.DataFrame] = None, 
        features_df: Optional[pd.DataFrame] = None, 
        settings: Optional[Data_Settings] = None
    ):
        if settings is None:
            if loadshape_df is None:
                settings = Data_Settings()
            else: # if loadshape is provided, then apply appropriate settings
                settings = Data_Settings(agg_type=None, loadshape_type=None, time_period=None)

        self._settings = settings

        self._loadshape = None
        self._features = None

        # TODO: let's make id the index for the excluded ids dataframe
        self._excluded_ids = pd.DataFrame(columns=["id", "reason"])

        # basic error checking
        if loadshape_df is None and time_series_df is None and features_df is None:
            raise ValueError(
                "A loadshape, time series, or features dataframe must be provided."
            )

        elif loadshape_df is not None and time_series_df is not None:
            raise ValueError(
                "Both loadshape dataframe and time series dataframe are provided. Please provide only one."
            )

        if self._settings.time_period is not None and (loadshape_df is not None or time_series_df is None):
            # Time period should only be set if a time series dataframe is provided
            raise ValueError(
                "Time period is set, but no time series dataframe is provided. Please provide a time series dataframe."
            )

        # set the data
        self._set_data(loadshape_df, time_series_df, features_df)


    def extend(self, other):
        """
            Extend the current Data instance with the Data instance(s) in other by concatenating the features and loadshape dataframes.
        """
        if not isinstance(other, list):
            other = [other]

        for data_instance in other:
            # TODO : What happens if the same id exists in multiple dataframes? Average them out?
            if isinstance(data_instance, Data):
                if self._settings.time_period != data_instance.settings.time_period:
                    raise ValueError("Time period setting must be the same for all Data instances.")
                if self._features is not None and data_instance.features is not None:
                    self._features = pd.concat([self._features, data_instance.features])
                if self._loadshape is not None and data_instance.loadshape is not None:
                    self._loadshape = pd.concat([self._loadshape, data_instance.loadshape])
            else:
                raise TypeError("All elements in other must be instances of Data")
            

    def _add_index_columns_from_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add hour column
        if "hour" in self._settings.time_period:
            df["hour"] = df['datetime'].dt.hour

        # Add month column
        if "month" in self._settings.time_period:
            df["month"] = df['datetime'].dt.month

        # Add day_of_week column
        if "day_of_week" in self._settings.time_period:
            df["day_of_week"] = df['datetime'].dt.dayofweek

        # Add day_of_year column
        if "day_of_year" in self._settings.time_period:
            df["day_of_year"] = df['datetime'].dt.dayofyear

        # Add weekday_weekend column
        if "weekday_weekend" in self._settings.time_period:
            df["weekday_weekend"] = df['datetime'].dt.dayofweek

            # Setting the ordering to weekday, weekend
            df["weekday_weekend"] = (
                df["weekday_weekend"]
                .map(self._settings.weekday_weekend._num_dict)
                .map(self._settings.weekday_weekend._order)
            )

        # Add season column
        if "season" in self._settings.time_period:
            df["season"] = df['datetime'].dt.month.map(self._settings.season._num_dict).map(
                self._settings.season._order
            )

        return df


    def _time_matrix(self, matrix: pd.DataFrame, n_times: int) -> pd.DataFrame:
        """Reindex an id-indexed matrix to times 1..n_times, exclude and record the meters
        short of times, and fill the rest flat when interpolating."""
        matrix = (
            matrix.reindex(columns=range(1, int(n_times) + 1))
            .rename_axis(None, axis=1)
            .sort_index()
            .astype(float)
        )

        if self._settings.interpolate_missing:
            min_times = n_times * self._settings.min_data_pct_required
        else:
            min_times = n_times

        present_times_per_id = matrix.notna().sum(axis=1)
        invalid_ids = present_times_per_id[present_times_per_id < min_times].index
        excluded_ids = pd.DataFrame(
            {
                "id": invalid_ids,
                "reason": "Unique time counts per id don't have the minimum time counts required",
            }
        )
        self._excluded_ids = pd.concat([self._excluded_ids, excluded_ids], ignore_index=True)
        matrix = matrix.drop(index=invalid_ids)

        if self._settings.interpolate_missing:
            matrix = pd.DataFrame(
                fill_missing(matrix.to_numpy(), layout=None),
                index=matrix.index,
                columns=matrix.columns,
            )

        return matrix


    def _validate_unstacked_loadshape(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("id")
        df.columns = df.columns.astype(int)

        matrix = self._time_matrix(df, df.columns.max())

        return matrix


    def _validate_format_loadshape(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reset index to remove any existing index
        df = df.reset_index()
        df = df.drop(columns="index", errors="ignore")

        # Check columns missing in loadshape_df
        expected_columns = ["id", "time", "loadshape"]
        missing_columns = [c for c in expected_columns if c not in df.columns]

        if missing_columns:
            # TODO : handle the case when index is the id. Then we don't need to check for id in the columns. But how to ensure we don't have wrong index?
            if "loadshape" in missing_columns and "time" in missing_columns and "id" not in missing_columns:
                # Handle loadshapes in unstacked version
                return self._validate_unstacked_loadshape(df)
            
            else:  
                raise ValueError(f"Missing columns in loadshape_df: {missing_columns}")

        # Check if all values are present in the columns as required
        # Else update the values via interpolation if missing, also ignore duplicates if present

        # loadshape df has the "time" column, whereas timeseries df has the "datetime" column
        subset_columns = expected_columns[:-1]

        # To eliminate duplicates, keep the smallest |loadshape| per (id, time), NaN last
        df = df.sort_values(by="loadshape", key=abs, kind="stable").drop_duplicates(
            subset=subset_columns, keep="first"
        )

        # pivot the loadshape_df to have the time as columns, one column per time
        matrix = df.pivot(index="id", columns="time", values="loadshape")
        matrix = self._time_matrix(matrix, df["time"].max())

        return matrix


    def _validate_format_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reset index to remove any existing index
        df = df.reset_index()
        df = df.drop(columns="index", errors="ignore")

        # Check columns missing in features_df
        if "id" not in df.columns:
            raise ValueError(f"Missing columns in features_df: 'id'")

        # get a list of any rows with missing values
        excluded_ids = df[df.isnull().any(axis=1)]["id"].values
        if excluded_ids.size > 0:
            excluded_ids = pd.DataFrame({"id": excluded_ids})
            excluded_ids["reason"] = "null values in features_df"
            self._excluded_ids = pd.concat([self._excluded_ids, excluded_ids])

        # remove any rows with missing values
        df = df.dropna()

        df.drop_duplicates(keep="first" , inplace = True)

        # drop any ids that are in excluded_ids from loadshape (or init)
        df = df[~df["id"].isin(self._excluded_ids["id"])]
        df = (
            df.reset_index()
            .set_index("id")
            .drop(columns="index", errors="ignore")
        )

        # sort by id
        df = df.sort_index()

        return df


    def _convert_timeseries_to_loadshape(
        self, time_series_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Arguments:
            Time series dataframe with columns = [id, datetime, observed, observed_error, modeled, modeled_error

        Returns :
            Loadshape dataframe with columns = [id, time, loadshape]
        """

        base_df = time_series_df.copy()  # don't change the original dataframe

        # Reset index to remove any existing index
        base_df = base_df.reset_index()
        base_df = base_df.drop(columns="index", errors="ignore")

        # Check columns missing in time_series_df
        df_type = self._settings.loadshape_type
        expected_columns = ["id", "datetime"]
        if (df_type == "error") and ("error" in base_df.columns):
            expected_columns.append("error")
        elif (df_type == "error") and ("error" not in base_df.columns):
            expected_columns.extend(["observed", "modeled"])
        else:
            expected_columns.append(df_type)

        missing_columns = [c for c in expected_columns if c not in base_df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in time_series_df: {missing_columns}")

        # Check that the datetime column is actually of type datetime
        if is_datetime(base_df["datetime"]):
            base_df["datetime"] = pd.to_datetime(base_df["datetime"], utc=True) #TODO: should this be utc=True? should this be applied to all datetime types?
        else:
            raise ValueError("The 'datetime' column must be of datetime type")

        if df_type == "error" and ("error" not in base_df.columns):
            base_df["error"] = 1 - base_df["observed"] / base_df["modeled"]

        # Of each duplicated reading keep the smallest |value|, NaN last
        subset_columns = expected_columns[:-1]
        duplicated = base_df.duplicated(subset=subset_columns, keep=False)
        if duplicated.any():
            kept = (
                base_df[duplicated]
                .sort_values(by=df_type, key=abs, kind="stable")
                .drop_duplicates(subset=subset_columns, keep="first")
            )
            base_df = pd.concat([base_df[~duplicated], kept])

        # Order by (id, datetime), which the per-meter interval diff relies on
        meter, ids = pd.factorize(base_df["id"], sort=True)
        timestamp = pd.DatetimeIndex(base_df["datetime"]).asi8
        meter_step = np.diff(meter)
        if np.any((meter_step < 0) | ((meter_step == 0) & (np.diff(timestamp) < 0))):
            order = np.lexsort((timestamp, meter))
            base_df = base_df.iloc[order]
            meter = meter[order]

        base_df = self._add_index_columns_from_datetime(base_df) # Add month / day_of_week / hour / etc columns

        # Check that each id has a minimum granularity lower than requested time period, otherwise we cannot aggregate
        # get minimum time interval per id
        base_df["time_diff"] = base_df.groupby("id")["datetime"].diff()
        min_time_diff_per_id = base_df.groupby("id")["time_diff"].min() / np.timedelta64(1, 'm')

        # Get the ids that have a higher minimum granularity than defined
        if self._settings.time_period != 'month':
            invalid_ids = min_time_diff_per_id[
                min_time_diff_per_id > _const.min_granularity_per_time_period[self._settings.time_period]
            ].index.tolist()

        else:
            # Check that every ID has 12 months available.
            unique_month_counts_per_id = base_df.groupby('id')['month'].nunique()
            invalid_ids = unique_month_counts_per_id[unique_month_counts_per_id < 12].index.tolist()

        # If there are any invalid ids, add them to the excluded_ids dataframe
        if invalid_ids:
            invalid_ids_df = pd.DataFrame(
                {
                    "id": invalid_ids,
                    "reason": "Minimum time interval is more than the specified TimePeriod",
                }
            )
            self._excluded_ids = pd.concat(
                [self._excluded_ids, invalid_ids_df], ignore_index=True
            )

        # Readings that land in a time column: known id and datetime, id not invalid, not the leap day
        has_time = (
            (meter >= 0)
            & base_df["datetime"].notna().to_numpy()
            & ~base_df["id"].isin(invalid_ids).to_numpy()
        )
        if "day_of_year" in base_df.columns:
            has_time &= base_df["day_of_year"].to_numpy() != 366

        # Key each reading to its cell: meter ordinal, then a mixed-radix time index over the
        # grouping columns, season outermost
        key = meter[has_time].astype(np.int64)
        for column, cardinality in self._settings.time_layout:
            digit = base_df[column].to_numpy()[has_time] - _time_digit_offset.get(column, 0)
            key = key * cardinality + digit.astype(np.int64)

        values = base_df[df_type].to_numpy(dtype=float)[has_time]
        observed = ~np.isnan(values)
        n_meters = len(ids)
        n_times = self._settings.n_times
        readings = np.bincount(key, minlength=n_meters * n_times).reshape(n_meters, n_times)
        loadshape = _aggregate_cells(
            key[observed], values[observed], n_meters * n_times, self._settings.agg_type
        ).reshape(n_meters, n_times)

        has_readings = readings.sum(axis=1) > 0
        ids = pd.Index(ids[has_readings], name="id")
        readings = readings[has_readings]
        loadshape = loadshape[has_readings]
        present_times_per_id = pd.Series((~np.isnan(loadshape)).sum(axis=1), index=ids)

        if self._settings.interpolate_missing:
            # Check that the number of missing values is less than the threshold
            # throw out meters with missing values and record them, do not throw error
            invalid_ids = present_times_per_id[
                present_times_per_id < self._settings.min_data_pct_required * n_times
            ].index.tolist()
            excluded_ids = pd.DataFrame(
                {
                    "id": invalid_ids,
                    "reason": "missing minimum number of values in loadshape_df",
                }
            )
            self._excluded_ids = pd.concat(
                [self._excluded_ids, excluded_ids], ignore_index=True
            )

        else:
            # throw out ids with a time whose readings are all null, one record per time
            null_rows, _ = np.nonzero((readings > 0) & np.isnan(loadshape))
            if null_rows.size > 0:
                excluded_ids = pd.DataFrame({"id": ids[null_rows]})
                excluded_ids["reason"] = "null values in features_df"
                self._excluded_ids = pd.concat([self._excluded_ids, excluded_ids])

            # throw out ids missing any time, one record per id
            incomplete_ids = present_times_per_id[present_times_per_id < n_times].index
            incomplete_ids = incomplete_ids[~incomplete_ids.isin(self._excluded_ids["id"])]
            incomplete_ids_df = pd.DataFrame(
                {
                    "id": incomplete_ids,
                    "reason": "Unique time counts per id don't have the minimum time counts required",
                }
            )
            self._excluded_ids = pd.concat(
                [self._excluded_ids, incomplete_ids_df], ignore_index=True
            )

        kept = ~ids.isin(self._excluded_ids["id"])
        ids = ids[kept]
        loadshape = loadshape[kept]
        if self._settings.interpolate_missing:
            loadshape = fill_missing(loadshape, self._settings.time_layout)

            # The fill leaves a season block with no readings NaN, which the minimum-data
            # rule above does not always catch
            empty_block = np.isnan(loadshape).any(axis=1)
            excluded_ids = pd.DataFrame(
                {"id": ids[empty_block], "reason": "a season block has no readings"}
            )
            self._excluded_ids = pd.concat(
                [self._excluded_ids, excluded_ids], ignore_index=True
            )
            ids = ids[~empty_block]
            loadshape = loadshape[~empty_block]

        loadshape_df = pd.DataFrame(loadshape, index=ids, columns=range(1, n_times + 1))

        return loadshape_df


    def _trim_data(self) -> None:
        """
        Trim the loadshape and features dataframes to the maximum size allowed by the settings.
        """

        max_size = self._settings.max_pool_size

        ids = self.ids
        excluded_ids = []
        if len(ids) > max_size:
            # randomly select ids to remove
            rng = np.random.RandomState(self._settings.seed)
            excluded_ids = rng.choice(ids, len(ids) - max_size, replace=False)

            # add excluded ids to excluded_ids dataframe
            excluded_ids_df = pd.DataFrame({"id": excluded_ids})
            excluded_ids_df["reason"] = "randomly selected to reduce pool size"
            self._excluded_ids = pd.concat([self._excluded_ids, excluded_ids_df])

        if (self._loadshape is not None) and (len(excluded_ids) > 0):
            self._loadshape = self._loadshape[
                ~self._loadshape.index.isin(self._excluded_ids["id"])
            ]

        if (self._features is not None) and (len(excluded_ids) > 0):
            self._features = self._features[
                ~self._features.index.isin(self._excluded_ids["id"])
            ]


    def _set_data(
        self, loadshape_df=None, time_series_df=None, features_df=None
    ) -> None:
        """
            Loadshape, timeseries and features dataframes are input. The loadshape and features dataframes are validated and formatted.
            The timeseries dataframe is converted to a loadshape dataframe and then validated and formatted.

            Time period is only set if a timeseries dataframe is provided. If a loadshape dataframe is provided, 
            the aggregation type, loadshape type and time period all must be set to None.

            Either loadshape or timeseries data is allowed, but not both. Atleast one of them must be provided as well.
            Features is independent of the loadshape and timeseries dataframes.

            Loadshape / timeseries only input => Clustering / IMM
            Features only input => Stratified Sampling

            Note the loadshape and features dataframe can only be set once per class.

        Args:
            Loadshape_df: columns = [id, time, loadshape]

            Time_series_df: columns = [id, datetime, observed, observed_error, modeled, modeled_error]

            Features_df: columns = [id, {feature_1}, {feature_2}, ...]

        Output:
            loadshape: index = id, columns = time, values = loadshape

            features: index = id, columns = [{feature_1}, {feature_2}, ...]


        """

        if loadshape_df is not None:
            if self._loadshape is not None :
                raise ValueError("Loadshape Data has already been set.")
            elif self._settings.loadshape_type is not None:
                raise ValueError("Loadshape Type cannot be set for a loadshape dataframe.")

            loadshape_df = self._validate_format_loadshape(loadshape_df)

        elif time_series_df is not None:
            if self._loadshape is not None:
                raise ValueError("Loadshape Data has already been set.")

            loadshape_df = self._convert_timeseries_to_loadshape(time_series_df)

        if features_df is not None:
            if self._features is not None:
                raise ValueError("Features Data has already been set.")
            features_df = self._validate_format_features(features_df)

        if loadshape_df is not None:
            # If loadshape still has id as one of its columns, set it as index
            if 'id' in loadshape_df.columns:
                loadshape_df.set_index('id', inplace=True)

            # drop any ids that are in the excluded_ids list
            loadshape_df = loadshape_df[
                ~loadshape_df.index.isin(self._excluded_ids["id"])
            ]

            if loadshape_df.empty:
                raise self._no_meters_error("loadshape")

        # Empty or absent dataframes become None, not an empty dataframe
        if features_df is not None and not features_df.empty:
            self._features = features_df
        else:
            self._features = None

        if loadshape_df is not None and not loadshape_df.empty:
            self._loadshape = loadshape_df
        else:
            self._loadshape = None

        if self._loadshape is None and self._features is None:
            raise self._no_meters_error("features")

        # filter pool to max size
        self._trim_data()

        return self

    def _no_meters_error(self, frame: str) -> ValueError:
        reasons = "; ".join(self._excluded_ids["reason"].unique())

        return ValueError(f"No meters remain in the {frame}. Exclusion reasons: {reasons}")

    @property
    def settings(self):
        return self._settings.model_copy()
    
    @property
    def loadshape(self):
        if self._loadshape is None:
            return None
        else :
            return self._loadshape.copy()
    
    @property
    def features(self):
        if self._features is None:
            return None
        else :
            return self._features.copy()

    @property
    def ids(self):
        if isinstance(self._loadshape, pd.DataFrame):
            return deepcopy(self._loadshape.index.unique().to_list())
        elif isinstance(self._features, pd.DataFrame):
            return deepcopy(self._features.index.unique().to_list())
        else:
            return None

    @property
    def excluded_ids(self):
        if self._excluded_ids is None:
            return None
        else :
            return self._excluded_ids.copy()