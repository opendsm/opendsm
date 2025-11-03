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
            

    def _find_groupby_columns(self) -> list:
        """
        Create the list of columns to be grouped by based on the time_period selected in Settings.

        Time_period : hour => group by (id, hour)
        Time_period : month => group by (id, month)
        Time_period : hourly_day_of_week => group by (id, day_of_week, hour)
        Time_period : weekday_weekend => group by (id, weekday_weekend)
        Time_period : season_day_of_week => group by (id, season, day_of_week)
        Time_period : season_hourly_weekday_weekend => group by (id, season, weekday_weekend, hour)

        """
        cols = ["id"]

        for period in _const.unique_time_periods:
            if period in self._settings.time_period:
                cols.append(period)

        return cols


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


    def _create_values_for_interpolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing values in the dataframe based on the settings.
        
        - create a new dataframe with id's and correct time column
        - join on new df and old
        - interpolate nan values


        """

        if self._settings.interpolate_missing:
            unique_ids = df['id'].unique()
            unique_time_counts = None

            if self._settings.time_period is None: # loadshape type dataframe
                unique_time_counts = df["time"].max()

            else: # timeseries type dataframe
                unique_time_counts = _const.time_period_row_counts[self._settings.time_period]


            time_values = range(1, unique_time_counts + 1)
            # Create the expected dataframe having the correct number of timestamps for each id    
            df_expected = pd.DataFrame({
                'id': np.repeat(unique_ids, unique_time_counts),
                'time': np.tile(time_values, len(unique_ids))
            })

            # Join the expected dataframe with the input dataframe
            df = df_expected.merge(df, how='left', on=['id', 'time'])

        return df


    def _validate_unstacked_loadshape(self, df: pd.DataFrame) -> pd.DataFrame:
        unstacked_cols = df.columns.drop('id')
        unstacked_cols = sorted(map(int, unstacked_cols))

        # TODO : Add the ids that are missing values to the excluded_ids dataframe
        # expected_cols = range(1, max(unstacked_cols) + 1)

        # if unstacked_cols != expected_cols:
        #     if not self._settings.INTERPOLATE_MISSING or unstacked_cols.count() < expected_cols.count() * self._settings.MIN_DATA_PCT_REQUIRED:
        #             raise ValueError(f"Unique time counts per id don't have the minimum time counts required")
            

        # Find the missing columns and add them to df with NaN as the default value
        expected_cols = df.columns.union(range(1, max(unstacked_cols) + 1))
        df.reindex(columns= expected_cols, fill_value=np.nan)

        if self._settings.interpolate_missing:
            # Get non-id columns
            non_id_cols = df.columns[df.columns != 'id']

            # Perform interpolation on non-id columns and update the original DataFrame
            df[non_id_cols] = df[non_id_cols].interpolate(method="linear", limit_direction="both", axis=1)

        return df


    def _validate_format_loadshape(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reset index to remove any existing index
        df = df.reset_index()
        df = df.drop(columns="index", axis=1, errors="ignore")

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

        # To eliminate duplicates, sort the values by loadshape and the keep the first (i.e. the lowest) value
        df = df.sort_values(by='loadshape', key=abs).drop_duplicates(subset=subset_columns, keep="first")

        # Check that the minimum time counts per id is consistent for the input loadshape_df
        unique_time_counts = df["time"].max()
        unique_time_counts_per_id = df.groupby("id")["time"].nunique()

        if self._settings.interpolate_missing:
            if self._settings.time_period is None:
                # for loadshape type dataframe
                # if I input a loadshape, I don't want to have to tell it the time_period I used
                # The time column should directly be pivoted, and the error checking should ensure that the number of values is consistent per meter

                invalid_ids = unique_time_counts_per_id[
                    unique_time_counts_per_id
                    < unique_time_counts * self._settings.min_data_pct_required
                ].index.tolist()
                excluded_ids = pd.DataFrame(
                    {
                        "id": invalid_ids,
                        "reason": "Unique time counts per id don't have the minimum time counts required",
                    }
                )
                self._excluded_ids = pd.concat(
                    [self._excluded_ids, excluded_ids], ignore_index=True
                )

            else:
                # Check that the number of missing values is less than the threshold
                for id, group in df.groupby("id"):
                    if (
                        group.count().min()
                        < self._settings.min_data_pct_required
                        * _const.time_period_row_counts[self._settings.time_period]
                    ):
                        # throw out meters with missing values and record them, do not throw error

                        excluded_ids = pd.DataFrame(
                            {
                                "id": [id],
                                "reason": [
                                    "missing minimum number of values in loadshape_df"
                                ],
                            }
                        )

                        self._excluded_ids = pd.concat(
                            [self._excluded_ids, excluded_ids], ignore_index=True
                        )

            df = self._create_values_for_interpolation(df)

            # Fill NaN values with interpolation
            df['loadshape'] = (
                df.groupby("id")['loadshape']
                .apply(lambda x: x.interpolate(method="linear", limit_direction="both"))
                .reset_index(drop=True)
            )

        else:
            if self._settings.time_period is None:
                # for loadshape type dataframe
                invalid_ids = unique_time_counts_per_id[
                    unique_time_counts_per_id < unique_time_counts
                ].index.tolist()
                invalid_ids_df = pd.DataFrame(
                    {
                        "id": invalid_ids,
                        "reason": "Unique time counts per id don't have the minimum time counts required",
                    }
                )
                self._excluded_ids = pd.concat(
                    [self._excluded_ids, invalid_ids_df], ignore_index=True
                )

            else:
                # throw out id with null values and record them, do not throw error

                # get a list of any rows with missing values
                excluded_ids = df[df.isnull().any(axis=1)]["id"].values
                if excluded_ids.size > 0:
                    excluded_ids = pd.DataFrame({"id": excluded_ids})
                    excluded_ids["reason"] = "null values in features_df"
                    self._excluded_ids = pd.concat([self._excluded_ids, excluded_ids])

        df = df[ ~df["id"].isin(self._excluded_ids["id"])]

        # pivot the loadshape_df to have the time as columns
        df = df.pivot(index="id", columns=["time"], values="loadshape")

        # Convert multi level index to single level
        df = (
            df.rename_axis(None, axis=1)
            .reset_index()
            .set_index("id")
            .drop(columns="index", axis=1, errors="ignore")
        )

        # Convert columns to int
        df.columns = df.columns.astype(int)

        return df


    def _validate_format_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Reset index to remove any existing index
        df = df.reset_index()
        df = df.drop(columns="index", axis=1, errors="ignore")

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
            .drop(columns="index", axis=1, errors="ignore")
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
        base_df = base_df.drop(columns="index", axis=1, errors="ignore")

        # Check columns missing in time_series_df
        df_type = self._settings.LOADSHAPE_TYPE
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

        # Remove duplicates
        subset_columns = expected_columns[:-1]

        # To eliminate duplicates, sort the values by loadshape and the keep the first (i.e. the lowest) value
        base_df = base_df.sort_values(by=df_type, key=abs).drop_duplicates(subset=subset_columns, keep="first")

        base_df = self._add_index_columns_from_datetime(base_df) # Add month / day_of_week / hour / etc columns

        # Check that each id has a minimum granularity lower than requested time period, otherwise we cannot aggregate
        # get minimum time interval per id
        base_df["time_diff"] = base_df.groupby("id")["datetime"].diff()
        min_time_diff_per_id = base_df.groupby("id")["time_diff"].min() / np.timedelta64(1, 'm')

        # Get the ids that have a higher minimum granularity than defined
        if self._settings.TIME_PERIOD != 'month':
            invalid_ids = min_time_diff_per_id[
                min_time_diff_per_id > _const.min_granularity_per_time_period[self._settings.TIME_PERIOD]
            ].index.tolist()

        else:
            # Check that every ID has 12 months available.
            unique_month_counts_per_id = base_df.groupby('id')['month'].nunique()
            invalid_ids = unique_month_counts_per_id[unique_month_counts_per_id < 12].index.tolist()

        # Remove the invalid ids from the base_df
        base_df = base_df[~base_df["id"].isin(invalid_ids)]        

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

        # Set the index to datetime
        base_df = base_df.set_index("datetime")

        # Aggregate the input time_series based on time_period

        group_by_columns = self._find_groupby_columns()

        base_df = base_df.groupby(group_by_columns)[self._settings.loadshape_type]

        base_df = base_df.agg(loadshape=self._settings.agg_type).reset_index()

        # Sort the values so that the ordering is maintained correctly
        base_df = base_df.sort_values(by=group_by_columns)

        # Create the count of the index per ID
        base_df["time"] = base_df.groupby("id").cumcount() + 1

        # Validate that all the values are correct
        loadshape_df = self._validate_format_loadshape(base_df)

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
            excluded_ids = np.random.choice(ids, len(ids) - max_size, replace=False)

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

        # If the dataframes are empty return None, not an empty dataframe
        if features_df is not None:
            self._features = features_df if not features_df.empty else None
        self._loadshape = loadshape_df if not loadshape_df.empty else None

        # filter pool to max size
        self._trim_data()

        return self

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