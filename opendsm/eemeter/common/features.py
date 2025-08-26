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
import pandas as pd
import statsmodels.formula.api as smf

from ..models.hourly_caltrack.segmentation import iterate_segmented_dataset
from .transform import day_counts, overwrite_partial_rows_with_nan
from .warnings import EEMeterWarning

__all__ = (
    "compute_usage_per_day_feature",
    "compute_occupancy_feature",
    "compute_temperature_features",
    "compute_temperature_bin_features",
    "compute_time_features",
    "estimate_hour_of_week_occupancy",
    "fit_temperature_bins",
    "get_missing_hours_of_week_warning",
    "merge_features",
)


def merge_features(features, keep_partial_nan_rows=False):
    """
    Combine dataframes of features which share a datetime index.

    Parameters
    ----------
    features : :any:`list` of :any:`pandas.DataFrame`
        List of dataframes to be concatenated to share an index.
    keep_partial_nan_rows : :any:`bool`, default False
        If True, don't overwrite partial rows with NaN, otherwise any row with a NaN
        value gets changed to all NaN values.

    Returns
    -------
    merged_features : :any:`pandas.DataFrame`
        A single dataframe with the index of the input data and all of the columns
        in the input feature dataframes.
    """

    def _to_frame_if_needed(df_or_series):
        if isinstance(df_or_series, pd.Series):
            return df_or_series.to_frame()
        return df_or_series

    df = pd.concat([_to_frame_if_needed(feature) for feature in features], axis=1)

    if not keep_partial_nan_rows:
        df = overwrite_partial_rows_with_nan(df)
    return df


def compute_usage_per_day_feature(meter_data, series_name="usage_per_day"):
    """Compute average usage per day for billing/daily data.

    Parameters
    ----------
    meter_data : :any:`pandas.DataFrame`
        Meter data for which to compute usage per day.
    series_name : :any:`str`
        Name of the output pandas series

    Returns
    -------
    usage_per_day_feature : :any:`pandas.Series`
        The usage per day feature.
    """
    # CalTrack 3.3.1.1
    # convert to average daily meter values.
    usage_per_day = meter_data.value / day_counts(meter_data.index)
    return pd.Series(usage_per_day, name=series_name)


def get_missing_hours_of_week_warning(hours_of_week):
    """Warn if any hours of week (0-167) are missing.

    Parameters
    ----------
    hours_of_week : :any:`pandas.Series`
        Hour of week feature as given by :any:`eemeter.compute_time_features`.

    Returns
    -------
    warning : :any:`eemeter.EEMeterWarning`
        Warning with qualified name "eemeter.hour_of_week.missing"
    """
    unique = set(hours_of_week.unique())
    total = set(range(168))
    missing = sorted(total - unique)
    if len(missing) == 0:
        return None
    else:
        return EEMeterWarning(
            qualified_name="eemeter.hour_of_week.missing",
            description="Missing some of the (zero-indexed) 168 hours of the week.",
            data={"missing_hours_of_week": missing},
        )


def compute_time_features(index, hour_of_week=True, day_of_week=True, hour_of_day=True):
    """Compute hour of week, day of week, or hour of day features.

    Parameters
    ----------
    index : :any:`pandas.DatetimeIndex`
        Datetime index with hourly frequency.
    hour_of_week : :any:`bool`
        Include the `hour_of_week` feature.
    day_of_week : :any:`bool`
        Include the `day_of_week` feature.
    hour_of_day : :any:`bool`
        Include the `hour_of_day` feature.

    Returns
    -------
    time_features : :any:`pandas.DataFrame`
        A dataframe with the input datetime index and up to three columns

        - hour_of_week : Label for hour of week, 0-167, 0 is 12-1am Monday
        - day_of_week : Label for day of week, 0-6, 0 is Monday.
        - hour_of_day : Label for hour of day, 0-23, 0 is 12-1am.
    """
    if index.freq != "h":
        raise ValueError(
            "index must have hourly frequency (freq='H')."
            " Found: {}".format(index.freq)
        )

    dow_feature = pd.Series(index.dayofweek, index=index, name="day_of_week")
    hod_feature = pd.Series(index.hour, index=index, name="hour_of_day")
    how_feature = (dow_feature * 24 + hod_feature).rename("hour_of_week")

    features = []
    warnings = []

    if day_of_week:
        features.append(dow_feature.astype("category"))
    if hour_of_day:
        features.append(hod_feature.astype("category"))
    if hour_of_week:
        how_feature = how_feature.astype("category")
        features.append(how_feature)
        warning = get_missing_hours_of_week_warning(how_feature)
        if warning is not None:
            warnings.append(warning)

    if len(features) == 0:
        raise ValueError("No features selected.")

    time_features = merge_features(features)
    return time_features


def _matching_groups(index, df, tolerance):
    # convert index to df for use with merge_asof
    index_df = pd.DataFrame({"index_col": index}, index=index)

    # get a dataframe containing mean temperature
    #   1) merge by matching temperature to closest previous meter start date,
    #      up to tolerance limit, using merge_asof.
    #   2) group by meter_index, and take the mean, ignoring all columns except
    #      the temperature column.
    groups = pd.merge_asof(
        left=df, right=index_df, left_index=True, right_index=True, tolerance=tolerance
    ).groupby("index_col")
    return groups


def _degree_day_columns(
    heating_balance_points,
    cooling_balance_points,
    degree_day_method,
    percent_hourly_coverage_per_day,
    percent_hourly_coverage_per_billing_period,
    use_mean_daily_values,
):
    # TODO(philngo): can this be refactored to be a more general without losing
    # on performance?

    # Not used in CalTRACK 2.0
    if degree_day_method == "hourly":

        def _compute_columns(temps):
            n_temps = temps.shape[0]
            n_temps_kept = temps.count()
            count_cols = {
                "n_hours_kept": n_temps_kept,
                "n_hours_dropped": n_temps - n_temps_kept,
            }
            if use_mean_daily_values:
                n_days = 1
            else:
                n_days = n_temps / 24.0
            cdd_cols = {
                "cdd_%s" % bp: np.maximum(temps - bp, 0).mean() * n_days
                for bp in cooling_balance_points
            }
            hdd_cols = {
                "hdd_%s" % bp: np.maximum(bp - temps, 0).mean() * n_days
                for bp in heating_balance_points
            }

            columns = count_cols
            columns.update(cdd_cols)
            columns.update(hdd_cols)
            return columns

    # CalTRACK 2.2.2.3
    n_limit_daily = 24 * percent_hourly_coverage_per_day

    if degree_day_method == "daily":

        def _compute_columns(temps):
            count = temps.shape[0]
            if count > 24:
                day_groups = np.floor(np.arange(count) / 24)
                daily_temps = temps.groupby(day_groups).agg(["mean", "count"])
                n_limit_period = percent_hourly_coverage_per_billing_period * count
                n_days_total = daily_temps.shape[0]

                # CalTrack 2.2.3.2
                if temps.notnull().sum() < n_limit_period:
                    daily_temps = daily_temps["mean"].iloc[0:0]
                else:
                    # CalTRACK 2.2.2.3
                    daily_temps = daily_temps["mean"][
                        daily_temps["count"] > n_limit_daily
                    ]
                n_days_kept = daily_temps.shape[0]
                count_cols = {
                    "n_days_kept": n_days_kept,
                    "n_days_dropped": n_days_total - n_days_kept,
                }

                if use_mean_daily_values:
                    n_days = 1
                else:
                    n_days = n_days_total

                cdd_cols = {
                    "cdd_%s" % bp: np.maximum(daily_temps - bp, 0).mean() * n_days
                    for bp in cooling_balance_points
                }
                hdd_cols = {
                    "hdd_%s" % bp: np.maximum(bp - daily_temps, 0).mean() * n_days
                    for bp in heating_balance_points
                }
            else:  # faster route for daily case, should have same effect.
                if count > n_limit_daily:
                    count_cols = {"n_days_kept": 1, "n_days_dropped": 0}
                    # CalTRACK 2.2.2.3
                    mean_temp = temps.mean()
                else:
                    count_cols = {"n_days_kept": 0, "n_days_dropped": 1}
                    mean_temp = np.nan

                # CalTrack 3.3.4.1.1
                cdd_cols = {
                    "cdd_%s" % bp: np.maximum(mean_temp - bp, 0)
                    for bp in cooling_balance_points
                }

                # CalTrack 3.3.5.1.1
                hdd_cols = {
                    "hdd_%s" % bp: np.maximum(bp - mean_temp, 0)
                    for bp in heating_balance_points
                }

            columns = count_cols
            columns.update(cdd_cols)
            columns.update(hdd_cols)
            return columns

    # TODO(philngo): option to ignore the count columns?

    agg_funcs = [("degree_day_columns", _compute_columns)]
    return agg_funcs


def compute_temperature_features(
    meter_data_index,
    temperature_data,
    heating_balance_points=None,
    cooling_balance_points=None,
    data_quality=False,
    temperature_mean=True,
    degree_day_method="daily",
    percent_hourly_coverage_per_day=0.5,
    percent_hourly_coverage_per_billing_period=0.9,
    use_mean_daily_values=True,
    tolerance=None,
    keep_partial_nan_rows=False,
):
    """Compute temperature features from hourly temperature data using the
    :any:`pandas.DatetimeIndex` meter data..

    Creates a :any:`pandas.DataFrame` with the same index as the meter data.

    .. note::

        For CalTRACK compliance (2.2.2.3), must set
        ``percent_hourly_coverage_per_day=0.5``,
        ``cooling_balance_points=range(30,90,X)``, and
        ``heating_balance_points=range(30,90,X)``, where
        X is either 1, 2, or 3. For natural gas meter use data, must
        set ``fit_cdd=False``.

    .. note::

        For CalTRACK compliance (2.2.3.2), for billing methods, must set
        ``percent_hourly_coverage_per_billing_period=0.9``.

    .. note::

        For CalTRACK compliance (2.3.3), ``meter_data_index`` and ``temperature_data``
        must both be timezone-aware and have matching timezones.

    .. note::

        For CalTRACK compliance (3.3.1.1), for billing methods, must set
        ``use_mean_daily_values=True``.

    .. note::

        For CalTRACK compliance (3.3.1.2), for daily or billing methods,
        must set ``degree_day_method=daily``.

    Parameters
    ----------
    meter_data_index : :any:`pandas.DataFrame`
        A :any:`pandas.DatetimeIndex` corresponding to the index over which
        to compute temperature features.
    temperature_data : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with hourly (``'H'``) frequency
        and a set of temperature values.
    cooling_balance_points : :any:`list` of :any:`int` or :any:`float`, optional
        List of cooling balance points for which to create cooling degree days.
    heating_balance_points : :any:`list` of :any:`int` or :any:`float`, optional
        List of heating balance points for which to create heating degree days.
    data_quality : :any:`bool`, optional
        If True, compute data quality columns for temperature, i.e.,
        ``temperature_not_null`` and ``temperature_null``, containing for
        each meter value
    temperature_mean : :any:`bool`, optional
        If True, compute temperature means for each meter period.
    degree_day_method : :any:`str`, ``'daily'`` or ``'hourly'``
        The method to use in calculating degree days.
    percent_hourly_coverage_per_day : :any:`str`, optional
        Percent hourly temperature coverage per day for heating and cooling
        degree days to not be dropped.
    use_mean_daily_values : :any:`bool`, optional
        If True, meter and degree day values should be mean daily values, not
        totals. If False, totals will be used instead.
    tolerance : :any:`pandas.Timedelta`, optional
        Do not merge more than this amount of temperature data beyond this limit.
    keep_partial_nan_rows: :any:`bool`, optional
        If True, keeps data in resultant :any:`pandas.DataFrame` that has
        missing temperature or meter data. Otherwise, these rows are overwritten
        entirely with ``numpy.nan`` values.

    Returns
    -------
    data : :any:`pandas.DataFrame`
        A dataset with the specified parameters.
    """
    if temperature_data.index.freq != "h":
        raise ValueError(
            "temperature_data.index must have hourly frequency (freq='H')."
            " Found: {}".format(temperature_data.index.freq)
        )

    if not temperature_data.index.tz:
        raise ValueError(
            "temperature_data.index must be timezone-aware. You can set it with"
            " temperature_data.tz_localize(...)."
        )

    if meter_data_index.freq is None and meter_data_index.inferred_freq == "h":
        raise ValueError(
            "If you have hourly data explicitly set the frequency"
            " of the dataframe by setting"
            "``meter_data_index.freq ="
            " pd.tseries.frequencies.to_offset('H')."
        )

    if not meter_data_index.tz:
        raise ValueError(
            "meter_data_index must be timezone-aware. You can set it with"
            " meter_data.tz_localize(...)."
        )

    if meter_data_index.duplicated().any():
        raise ValueError("Duplicates found in input meter trace index.")

    temp_agg_funcs = []
    temp_agg_column_renames = {}

    if heating_balance_points is None:
        heating_balance_points = []
    if cooling_balance_points is None:
        cooling_balance_points = []

    if meter_data_index.freq is not None:
        try:
            freq_timedelta = pd.Timedelta(meter_data_index.freq)
        except ValueError:  # freq cannot be converted to timedelta
            freq_timedelta = None
    else:
        freq_timedelta = None

    if tolerance is None:
        tolerance = freq_timedelta

    if not (heating_balance_points == [] and cooling_balance_points == []):
        if degree_day_method == "hourly":
            pass
        elif degree_day_method == "daily":
            if meter_data_index.freq == "h":
                raise ValueError(
                    "degree_day_method='daily' must be used with"
                    " daily meter data. Found: 'hourly'".format(degree_day_method)
                )
        else:
            raise ValueError("method not supported: {}".format(degree_day_method))

    if freq_timedelta == pd.Timedelta("1h"):
        # special fast route for hourly data.
        df = temperature_data.to_frame("temperature_mean").reindex(meter_data_index)

        if use_mean_daily_values:
            n_days = 1
        else:
            n_days = 1.0 / 24.0

        df = df.assign(
            **{
                "cdd_{}".format(bp): np.maximum(df.temperature_mean - bp, 0) * n_days
                for bp in cooling_balance_points
            }
        )
        df = df.assign(
            **{
                "hdd_{}".format(bp): np.maximum(bp - df.temperature_mean, 0) * n_days
                for bp in heating_balance_points
            }
        )
        df = df.assign(
            n_hours_dropped=df.temperature_mean.isnull().astype(int),
            n_hours_kept=df.temperature_mean.notnull().astype(int),
        )
        # TODO(philngo): bad interface or maybe this is just wrong for some reason?
        if data_quality:
            df = df.assign(
                temperature_null=df.n_hours_dropped,
                temperature_not_null=df.n_hours_kept,
            )
        if not temperature_mean:
            del df["temperature_mean"]
    else:
        # daily/billing route
        # heating/cooling degree day aggregations. Needed for n_days fields as well.
        temp_agg_funcs.extend(
            _degree_day_columns(
                heating_balance_points=heating_balance_points,
                cooling_balance_points=cooling_balance_points,
                degree_day_method=degree_day_method,
                percent_hourly_coverage_per_day=percent_hourly_coverage_per_day,
                percent_hourly_coverage_per_billing_period=percent_hourly_coverage_per_billing_period,
                use_mean_daily_values=use_mean_daily_values,
            )
        )
        temp_agg_column_renames.update(
            {("temp", "degree_day_columns"): "degree_day_columns"}
        )

        if data_quality:
            temp_agg_funcs.extend(
                [("not_null", "count"), ("null", lambda x: x.isnull().sum())]
            )
            temp_agg_column_renames.update(
                {
                    ("temp", "not_null"): "temperature_not_null",
                    ("temp", "null"): "temperature_null",
                }
            )

        if temperature_mean:
            temp_agg_funcs.extend([("mean", "mean")])
            temp_agg_column_renames.update({("temp", "mean"): "temperature_mean"})

        # aggregate temperatures
        temp_df = temperature_data.to_frame("temp")
        temp_groups = _matching_groups(meter_data_index, temp_df, tolerance)
        temp_aggregations = temp_groups.agg({"temp": temp_agg_funcs})

        # expand temp aggregations by faking and deleting the `meter_value` column.
        # I haven't yet figured out a way to avoid this and get the desired
        # structure and behavior. (philngo)
        meter_value = pd.DataFrame({"meter_value": 0}, index=meter_data_index)
        df = pd.concat([meter_value, temp_aggregations], axis=1).rename(
            columns=temp_agg_column_renames
        )
        del df["meter_value"]

        if "degree_day_columns" in df:
            if df["degree_day_columns"].dropna().empty:
                column_defaults = {
                    column: np.full(df["degree_day_columns"].shape, np.nan)
                    for column in ["n_days_dropped", "n_days_kept"]
                }
                df = df.drop(["degree_day_columns"], axis=1).assign(**column_defaults)
            else:
                df = pd.concat(
                    [
                        df.drop(["degree_day_columns"], axis=1),
                        df["degree_day_columns"].dropna().apply(pd.Series),
                    ],
                    axis=1,
                )

    if not keep_partial_nan_rows:
        df = overwrite_partial_rows_with_nan(df)

    if df.dropna(how='all').empty:
        raise ValueError("All rows are NaN.")

    # nan last row
    df = df.iloc[:-1].reindex(df.index)
    return df


def _estimate_hour_of_week_occupancy(model_data, threshold):
    index = pd.CategoricalIndex(range(168))
    if model_data.dropna().empty:
        return pd.Series(np.nan, index=index, name="occupancy")

    usage_model = smf.wls(
        formula="meter_value ~ cdd_65 + hdd_50",
        data=model_data,
        weights=model_data.weight,
    )

    model_data_with_residuals = model_data.merge(
        pd.DataFrame({"residuals": usage_model.fit().resid}),
        left_index=True,
        right_index=True,
    )

    def _is_high_usage(df):
        if df.empty:
            return np.nan
        n_positive_residuals = sum(df.residuals > 0)
        n_residuals = float(len(df.residuals))
        ratio_positive_residuals = n_positive_residuals / n_residuals
        return int(ratio_positive_residuals > threshold)

    return (
        model_data_with_residuals.groupby(["hour_of_week"], observed=False)[
            ["residuals"]
        ]
        .apply(_is_high_usage)
        .rename("occupancy")
        .reindex(index)
        .astype(bool)
    )  # guarantee an index value for all hours


def estimate_hour_of_week_occupancy(data, segmentation=None, threshold=0.65):
    """Estimate occupancy features for each segment.

    Parameters
    ----------
    data : :any:`pandas.DataFrame`
        Input data for the weighted least squares ("meter_value ~ cdd_65 + hdd_50")
        used to estimate occupancy. Must contain meter_value, hour_of_week, cdd_65, and
        hdd_50 columns with an hourly :any:`pandas.DatetimeIndex`.
    segmentation : :any:`pandas.DataFrame`, default None
        A segmentation expressed as a dataframe which shares the timeseries index of
        the data and has named columns of weights, which are of the form returned by
        :any:`eemeter.segment_time_series`.
    threshold : :any:`float`, default 0.65
        To be marked as unoccupied, the ratio of points with negative residuals in the
        weighted least squares in a particular hour of week must exceed this threshold.
        Said another way, in the default case, if more than 35% of values are greater
        than the basic degree day model for any particular hour of the week, that hour
        of week is marked as being occupied.

    Returns
    -------
    occupancy_lookup : :any:`pandas.DataFrame`
        The occupancy lookup has a categorical index with values from 0 to 167 - one
        for each hour of the week, and boolean values indicating an occupied (1, True)
        or unoccupied (0, False) for each of the segments. Each segment has a column
        labeled by its segment name.
    """

    occupancy_lookups = {}
    segmented_datasets = iterate_segmented_dataset(data, segmentation)
    for segment_name, segmented_data in segmented_datasets:
        hour_of_week_occupancy = _estimate_hour_of_week_occupancy(
            segmented_data, threshold
        )
        column = "occupancy" if segment_name is None else segment_name
        occupancy_lookups[column] = hour_of_week_occupancy
    # make sure columns stay in same order
    columns = ["occupancy"] if segmentation is None else segmentation.columns
    return pd.DataFrame(occupancy_lookups, columns=columns)


def _fit_temperature_bins(temperature_data, default_bins, min_temperature_count):
    def _compute_temp_summary(bins):
        bins = [-np.inf] + bins + [np.inf]
        bin_intervals = [
            pd.Interval(bin_left, bin_right, closed="right")
            for bin_left, bin_right in zip(bins, bins[1:])
        ]
        temp_bins = pd.cut(temperature_data, bins=bins).cat.set_categories(
            bin_intervals
        )
        return (
            pd.DataFrame({"temp": temperature_data, "bin": temp_bins})
            .groupby("bin", observed=False)["temp"]
            .count()
            .rename("count")
            .sort_index()
        )

    def _find_endpoints_to_remove(temp_summary):
        if len(temp_summary) == 1:
            return set()

        def _bin_count_invalid(i):
            count = temp_summary.iloc[i]
            return count < min_temperature_count or np.isnan(count)

        # work from outside in assuming less density at distribution edges
        endpoints = set()

        if _bin_count_invalid(0):  # first
            endpoints.add(temp_summary.index[0].right)

        if _bin_count_invalid(-1):  # last
            endpoints.add(temp_summary.index[-1].left)

        if len(endpoints) == 0:
            # try points in middle
            for i in range(1, len(temp_summary) - 1):
                if _bin_count_invalid(i):
                    endpoints.add(temp_summary.index[i].right)

        return endpoints

    test_bins = set(default_bins)

    while True:
        temp_summary = _compute_temp_summary(sorted(test_bins))
        endpoints_to_remove = _find_endpoints_to_remove(temp_summary)

        if len(endpoints_to_remove) == 0:
            break
        for endpoint in endpoints_to_remove:
            test_bins.discard(endpoint)

    return sorted(test_bins)


def fit_temperature_bins(
    data,
    segmentation=None,
    occupancy_lookup=None,
    default_bins=[30, 45, 55, 65, 75, 90],
    min_temperature_count=20,
):
    """Determine appropriate temperature bins for a particular set of temperature
    data given segmentation and occupancy.

    Parameters
    ----------
    data : :any:`pandas.Series`
        Input temperature data with an hourly :any:`pandas.DatetimeIndex`
    segmentation : :any:`pandas.DataFrame`, default None
        A dataframe containing segment weights with one column per segment. If
        left off, segmentation will not be considered.
    occupancy_lookup : :any:`pandas.DataFrame`, default None
        A dataframe of the form returned by :any:`eemeter.estimate_hour_of_week_occupancy`
        containing occupancy for each segment. If None, occupancy will not be
        considered.
    default_bins : :any:`list` of :any:`float` or :any:`int`
        A list of candidate bin endpoints to begin the search with.
    min_temperature_count : :any:`int`
        The minimum number of temperatre values that must be included in any bin. If
        this threshold is not met, bins are dropped from the outside in following the
        algorithm described in the CalTRACK documentation.

    Returns
    -------
    temperature_bins : :any:`pandas.DataFrame` or, if occupancy_lookup is provided a
    two :any:`tuple` of :any:`pandas.DataFrame`
        A dataframe with boolean values indicating whether or not a bin was kept, with a
        categorical index for each candidate bin endpoint and a column for each segment.
    """

    if occupancy_lookup is None:
        segmented_bins = {}
        segmented_datasets = iterate_segmented_dataset(data, segmentation)
        for segment_name, segmented_data in segmented_datasets:
            segmented_bins[segment_name] = _fit_temperature_bins(
                segmented_data.temperature_mean, default_bins, min_temperature_count
            )

        if segmentation is None:
            bins = segmented_bins[None]
            return pd.DataFrame(
                {"keep_bin_endpoint": [endpoint in bins for endpoint in default_bins]},
                index=pd.Series(default_bins, name="bin_endpoints"),
            )

        return pd.DataFrame(
            {
                segment_name: [endpoint in bins for endpoint in default_bins]
                for segment_name, bins in segmented_bins.items()
            },
            columns=segmentation.columns,
            index=pd.Series(default_bins, name="bin_endpoints"),
        )
    else:
        occupied_segmented_bins = {}
        unoccupied_segmented_bins = {}
        segmented_datasets = iterate_segmented_dataset(data, segmentation)
        for segment_name, segmented_data in segmented_datasets:
            hourly_segmented_data = segmented_data.resample("h").mean(numeric_only=True)
            time_features = compute_time_features(
                hourly_segmented_data.index,
                hour_of_week=True,
                day_of_week=False,
                hour_of_day=False,
            )
            if segment_name is None:
                occupancy = occupancy_lookup["occupancy"]
            else:
                occupancy = occupancy_lookup[segment_name]
            occupancy_features = compute_occupancy_feature(
                time_features.hour_of_week, occupancy
            )
            occupied_temperatures = segmented_data.temperature_mean[occupancy_features]
            unoccupied_temperatures = segmented_data.temperature_mean[
                ~occupancy_features
            ]
            occupied_segmented_bins[segment_name] = _fit_temperature_bins(
                occupied_temperatures, default_bins, min_temperature_count
            )
            unoccupied_segmented_bins[segment_name] = _fit_temperature_bins(
                unoccupied_temperatures, default_bins, min_temperature_count
            )

        if segmentation is None:
            occupied_bins = occupied_segmented_bins[None]
            unoccupied_bins = unoccupied_segmented_bins[None]
            return (
                pd.DataFrame(
                    {
                        "keep_bin_endpoint": [
                            endpoint in occupied_bins for endpoint in default_bins
                        ]
                    },
                    index=pd.Series(default_bins, name="bin_endpoints"),
                ),
                pd.DataFrame(
                    {
                        "keep_bin_endpoint": [
                            endpoint in unoccupied_bins for endpoint in default_bins
                        ]
                    },
                    index=pd.Series(default_bins, name="bin_endpoints"),
                ),
            )

        return (
            pd.DataFrame(
                {
                    segment_name: [endpoint in bins for endpoint in default_bins]
                    for segment_name, bins in occupied_segmented_bins.items()
                },
                columns=segmentation.columns,
                index=pd.Series(default_bins, name="bin_endpoints"),
            ),
            pd.DataFrame(
                {
                    segment_name: [endpoint in bins for endpoint in default_bins]
                    for segment_name, bins in unoccupied_segmented_bins.items()
                },
                columns=segmentation.columns,
                index=pd.Series(default_bins, name="bin_endpoints"),
            ),
        )


# TODO(philngo): combine with compute_temperature_features?
def compute_temperature_bin_features(temperatures, bin_endpoints):
    """Compute temperature bin features.

    Parameters
    ----------
    temperatures : :any:`pandas.Series`
        Hourly temperature data.
    bin_endpoints : :any:`list` of :any:`int` or :any:`float`
        List of bin endpoints to use when assigning features.

    Returns
    -------
    temperature_bin_features : :any:`pandas.DataFrame`
        A datafame with the input index and one column per bin. The sum of each
        row (with all of the temperature bins) equals the input temperature. More
        details on this bin feature are available in the CalTRACK documentation.
    """
    bin_endpoints = [-np.inf] + bin_endpoints + [np.inf]

    bins = {}

    for i, (left_bin, right_bin) in enumerate(zip(bin_endpoints, bin_endpoints[1:])):
        bin_name = "bin_{}".format(i)

        in_bin = (temperatures > left_bin) & (temperatures <= right_bin)
        gt_bin = temperatures > right_bin

        not_in_bin_index = temperatures.index[~in_bin]
        gt_bin_index = temperatures.index[gt_bin]

        def _expand_and_fill(partial_temp_series):
            return partial_temp_series.reindex(temperatures.index, fill_value=0)

        def _mask_nans(temp_series):
            return temp_series[temperatures.notnull()].reindex(temperatures.index)

        if i == 0:
            temps_in_bin = _expand_and_fill(temperatures[in_bin])
            temps_out_of_bin = _expand_and_fill(
                pd.Series(right_bin, index=not_in_bin_index)
            )
            bin_values = temps_in_bin + temps_out_of_bin
        else:
            temps_in_bin = _expand_and_fill(temperatures[in_bin] - left_bin)
            temps_gt_bin = _expand_and_fill(
                pd.Series(right_bin - left_bin, index=gt_bin_index)
            )
            bin_values = temps_in_bin + temps_gt_bin
        bins[bin_name] = _mask_nans(bin_values)
    return pd.DataFrame(bins)


def compute_occupancy_feature(hour_of_week, occupancy):
    """Given an hour of week feature, determine the occupancy for that hour of week.

    Parameters
    ----------
    hour_of_week : :any:`pandas.Series`
        Hour of week feature as given by :any:`eemeter.compute_time_features`.
    occupancy : :any:`pandas.Series`
        Boolean occupancy assignents for each hour of week as determined by
        :any:`eemeter.estimate_hour_of_week_occupancy`

    Returns
    -------
    occupancy_feature : :any:`pandas.Series`
        Occupancy labels for the timeseries.
    """
    return pd.merge(
        hour_of_week.dropna().to_frame(),
        occupancy.to_frame("occupancy"),
        how="left",
        left_on="hour_of_week",
        right_index=True,
    ).occupancy.reindex(hour_of_week.index)
