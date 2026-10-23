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



_BELOW_MINIMUM_REASON = "fewer loadshape times present than min_data_pct_required"
_INCOMPLETE_REASON = "missing loadshape times with interpolate_missing off"
_NULL_READINGS_REASON = "only null readings in a loadshape time"


def is_datetime(x: pd.Series) -> bool:
    is_dt = [
        pd.api.types.is_datetime64_any_dtype(x),
        # pd.api.types.is_datetime64_ns_dtype(x),
        # pd.api.types.is_datetime64_dtype(x),
        # isinstance(x.dtype, pd.DatetimeTZDtype)
    ]

    return any(is_dt)


def fill_missing(
    matrix: np.ndarray,
    layout: list[tuple[str, int]],
) -> np.ndarray:
    """Fill the NaN times of a meters x times loadshape matrix, returning a filled copy.

    Each row is interpolated linearly. When layout contains a season column, matrix's
    columns must be layout's times in layout order with season outermost; each row is then
    interpolated within each season block and never draws on another season's readings.
    Without a season column the whole row is one block. Times before the first or after the
    last reading of a block take that reading; a block with no readings stays NaN.

    Args:
        matrix: Meters x times loadshape values, NaN where a time is missing.
        layout: The time layout as ``(column, count)`` pairs, outermost first.

    Returns:
        A filled copy of ``matrix``.
    """
    filled = np.array(matrix, dtype=float)
    n_blocks = dict(layout).get("season", 1)
    block_len = filled.shape[1] // n_blocks
    positions = np.arange(block_len)

    for start in range(0, filled.shape[1], block_len):
        for row in filled[:, start:start + block_len]:
            missing = np.isnan(row)
            if missing.any() and not missing.all():
                row[missing] = np.interp(positions[missing], positions[~missing], row[~missing])

    return filled


def _keep_smallest_abs(df: pd.DataFrame, column: str, subset: list[str]) -> pd.DataFrame:
    """Of each group of rows duplicated on subset, keep the row with the smallest |column|.

    NaN sorts last and ties keep the earliest row. Unduplicated rows keep their order and
    precede the kept duplicates.
    """
    duplicated = df.duplicated(subset=subset, keep=False)
    if not duplicated.any():
        return df

    kept = (
        df[duplicated]
        .sort_values(by=column, key=abs, kind="stable")
        .drop_duplicates(subset=subset, keep="first")
    )
    deduplicated = pd.concat([df[~duplicated], kept])

    return deduplicated


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
            df["month"] = df['datetime'].dt.month - 1

        # Add day_of_week column
        if "day_of_week" in self._settings.time_period:
            df["day_of_week"] = df['datetime'].dt.day_of_week

        # Add weekday_weekend column
        if "weekday_weekend" in self._settings.time_period:
            df["weekday_weekend"] = df['datetime'].dt.day_of_week

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


    def _record_exclusions(self, ids, reason: str) -> None:
        """Append one `{id, reason}` ledger row per distinct id in `ids`, in first-seen order."""
        ids = pd.unique(np.asarray(ids))
        if ids.size == 0:
            return

        rows = pd.DataFrame({"id": ids, "reason": reason})
        self._excluded_ids = pd.concat([self._excluded_ids, rows], ignore_index=True)


    def _record_below_minimum(self, present_times_per_id: pd.Series, n_times: int) -> None:
        """Record the ids with fewer than `min_data_pct_required * n_times` present times."""
        min_times = self._settings.min_data_pct_required * n_times
        below = present_times_per_id.index[present_times_per_id < min_times]
        self._record_exclusions(below, _BELOW_MINIMUM_REASON)


    def _record_incomplete(self, present_times_per_id: pd.Series, n_times: int) -> None:
        """Record the ids missing any of the n_times times that are not already on the ledger."""
        incomplete = present_times_per_id.index[present_times_per_id < n_times]
        incomplete = incomplete[~incomplete.isin(self._excluded_ids["id"])]
        self._record_exclusions(incomplete, _INCOMPLETE_REASON)


    def _time_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """Reindex an id-indexed matrix's integer time columns to a contiguous range, exclude
        and record the meters short of times, and fill the rest with `fill_missing` when
        interpolating.

        Columns must be integer-valued time labels. Times start at 1, or at the smallest column
        when that is below 1 (0-based hour of week is common), so a first time every meter lacks
        is still a missing time; a last time every meter lacks is not detectable here.

        Raises:
            ValueError: a time label is null.
        """
        if matrix.columns.isna().any():
            raise ValueError("Loadshape times must not be null.")

        matrix = matrix.set_axis(matrix.columns.astype(int), axis=1)
        times = range(min(1, int(matrix.columns.min())), int(matrix.columns.max()) + 1)
        matrix = (
            matrix.reindex(columns=times)
            .rename_axis(None, axis=1)
            .sort_index()
            .astype(float)
        )
        n_times = len(times)

        present_times_per_id = matrix.notna().sum(axis=1)
        self._record_below_minimum(present_times_per_id, n_times)
        if not self._settings.interpolate_missing:
            self._record_incomplete(present_times_per_id, n_times)

        matrix = matrix[~matrix.index.isin(self._excluded_ids["id"])]

        if self._settings.interpolate_missing:
            matrix = pd.DataFrame(
                fill_missing(matrix.to_numpy(), self._settings.time_layout),
                index=matrix.index,
                columns=matrix.columns,
            )

        return matrix


    def _validate_unstacked_loadshape(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("id")
        matrix = self._time_matrix(df)

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

        df = _keep_smallest_abs(df, "loadshape", subset_columns)

        matrix = df.pivot(index="id", columns="time", values="loadshape")
        matrix = self._time_matrix(matrix)

        return matrix


    def _validate_format_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reset index to remove any existing index
        df = df.reset_index()
        df = df.drop(columns="index", errors="ignore")

        # Check columns missing in features_df
        if "id" not in df.columns:
            raise ValueError(f"Missing columns in features_df: 'id'")

        # get a list of any rows with missing values
        self._record_exclusions(
            df[df.isnull().any(axis=1)]["id"].values, "null values in features_df"
        )

        # remove any rows with missing values
        df = df.dropna()

        df = df.drop_duplicates(keep="first")

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

        base_df = _keep_smallest_abs(base_df, df_type, ["id", "datetime"])

        # Order by (id, datetime), which the per-meter interval diff relies on
        meter, ids = pd.factorize(base_df["id"], sort=True)
        timestamp = pd.DatetimeIndex(base_df["datetime"]).asi8
        order = np.lexsort((timestamp, meter))
        base_df = base_df.iloc[order]
        meter = meter[order]

        base_df = self._add_index_columns_from_datetime(base_df) # Add month / day_of_week / hour / etc columns

        # Check that each id's finest data interval is within the requested time period's maximum, otherwise we cannot aggregate
        # get minimum time interval per id
        base_df["time_diff"] = base_df.groupby("id")["datetime"].diff()
        min_time_diff_per_id = base_df.groupby("id")["time_diff"].min() / np.timedelta64(1, 'm')

        # Get the ids whose finest data interval exceeds that maximum
        if self._settings.time_period != 'month':
            invalid_ids = min_time_diff_per_id[
                min_time_diff_per_id > _const.max_data_interval_minutes_per_time_period[self._settings.time_period]
            ].index.tolist()

        else:
            # Check that every ID has 12 months available.
            unique_month_counts_per_id = base_df.groupby('id')['month'].nunique()
            invalid_ids = unique_month_counts_per_id[unique_month_counts_per_id < 12].index.tolist()

        self._record_exclusions(
            invalid_ids, "Minimum time interval is more than the specified TimePeriod"
        )

        # Readings that land in a time column: known id and datetime, id not invalid
        has_time = (
            (meter >= 0)
            & base_df["datetime"].notna().to_numpy()
            & ~base_df["id"].isin(invalid_ids).to_numpy()
        )

        # Key each reading to its cell: meter ordinal, then a mixed-radix time index over the
        # grouping columns in time_layout order
        key = meter[has_time].astype(np.int64)
        for column, cardinality in self._settings.time_layout:
            digit = base_df[column].to_numpy()[has_time]
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
        no_readings = ids[~has_readings]
        self._record_exclusions(
            no_readings[~no_readings.isin(self._excluded_ids["id"])],
            "no readings in any loadshape time",
        )
        ids = pd.Index(ids[has_readings], name="id")
        readings = readings[has_readings]
        loadshape = loadshape[has_readings]
        present_times_per_id = pd.Series((~np.isnan(loadshape)).sum(axis=1), index=ids)
        self._record_below_minimum(present_times_per_id, n_times)

        if not self._settings.interpolate_missing:
            # A time whose readings are all null takes precedence over the incomplete reason
            null_rows, _ = np.nonzero((readings > 0) & np.isnan(loadshape))
            null_ids = ids[np.unique(null_rows)]
            self._record_exclusions(
                null_ids[~null_ids.isin(self._excluded_ids["id"])], _NULL_READINGS_REASON
            )
            self._record_incomplete(present_times_per_id, n_times)

        kept = ~ids.isin(self._excluded_ids["id"])
        ids = ids[kept]
        loadshape = loadshape[kept]
        if self._settings.interpolate_missing:
            loadshape = fill_missing(loadshape, self._settings.time_layout)

            # The fill leaves a season block with no readings NaN, which the minimum-data
            # rule above does not always catch
            empty_block = np.isnan(loadshape).any(axis=1)
            self._record_exclusions(ids[empty_block], "a season block has no readings")
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
            self._record_exclusions(excluded_ids, "randomly selected to reduce pool size")

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

        if loadshape_df is not None:
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