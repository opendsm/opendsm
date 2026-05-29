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

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import re

import numpy as np
import pandas as pd

import sklearn

sklearn.set_config(
    assume_finite=True, skip_parameter_validation=True
)  # Faster, we do checking

from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.kernel_ridge import KernelRidge

import json

from opendsm.eemeter.models.hourly import settings as _settings
from opendsm.eemeter.models.hourly import HourlyBaselineData, HourlyReportingData
from opendsm.eemeter.models.hourly.scalers import (
    SafeStandardScaler,
    SafeRobustScaler,
)
from opendsm.eemeter.models.hourly.regressors import (
    SafeLinearRegression,
    SafeRidge,
    SafeLasso,
    SafeElasticNet,
    AdaptiveElasticNetRegressor,
)
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)
from opendsm.eemeter.common.warnings import EEMeterWarning
from opendsm.common.clustering.cluster import cluster_features
from opendsm.common.metrics import BaselineMetrics, BaselineMetricsFromDict, ReportingMetrics
from opendsm import __version__



def _get_interpolated_mask(df):
    cols = [col for col in df.columns if col.startswith("interpolated_")]
    return df[cols].any(axis=1)


def _eliminate_empty_bins(bin_edges, temp):
    valid_bin_edges = [-np.inf]
    for i in range(len(bin_edges) - 1):
        bin_count = ((temp >= bin_edges[i]) & (temp < bin_edges[i + 1])).sum()
        if bin_count > 0:
            valid_bin_edges.append(bin_edges[i + 1])

    valid_bin_edges[-1] = np.inf

    return np.array(valid_bin_edges)


def _merge_bins(bin_edges, temp, min_bin_count):
    # if less than min_bin_count values from temperature are in a bin,
    # remove bin edge starting from edges and moving inwards
    bin_edges = _eliminate_empty_bins(bin_edges, temp)
    for i in range(int(np.ceil(len(bin_edges) / 2)) - 1):
        if bin_edges[i + 1] == bin_edges[-(i + 2)]:
            continue  # only 1 bin edge left

        # left side
        bin_count = ((temp >= bin_edges[i]) & (temp < bin_edges[i + 1])).sum()
        if bin_count < min_bin_count:
            bin_edges[i + 1] = bin_edges[i]

        # right side
        bin_count = ((temp >= bin_edges[-(i + 2)]) & (temp < bin_edges[-(i + 1)])).sum()
        if bin_count < min_bin_count:
            bin_edges[-(i + 2)] = bin_edges[-(i + 1)]

    return np.unique(bin_edges)


def _fit_exp_growth_decay(x, y, k_only=True, is_x_sorted=False):
    # Courtsey: https://math.stackexchange.com/questions/1337601/fit-exponential-with-constant
    #           https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales
    #           Jean Jacquelin

    # fitting function is actual b*exp(c*x) + a

    # sort x in order
    x = np.array(x)
    y = np.array(y)
    n = len(x)

    if not is_x_sorted:
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

    # cumulative trapezoidal integration
    s = np.concatenate(([0], np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))))

    x_shifted = x - x[0]
    y_shifted = y - y[0]

    x_diff_sq = np.sum(x_shifted ** 2)
    xs_diff = np.sum(s * x_shifted)
    s_sq = np.sum(s**2)
    xy_diff = np.sum(x_shifted * y_shifted)
    ys_diff = np.sum(s * y_shifted)

    A = np.array([[x_diff_sq, xs_diff], [xs_diff, s_sq]])
    b = np.array([xy_diff, ys_diff])

    _, c = np.linalg.solve(A, b)
    with np.errstate(divide='ignore'):
        k = 1 / c # ignore divide by zero, it will be filtered later

    if k_only:
        a, b = None, None
    else:
        theta_i = np.exp(c * x)

        theta = np.sum(theta_i)
        theta_sq = np.sum(theta_i**2)
        y_sum = np.sum(y)
        y_theta = np.sum(y * theta_i)

        A = np.array([[n, theta], [theta, theta_sq]])
        b = np.array([y_sum, y_theta])

        a, b = np.linalg.solve(A, b)

    return a, b, k


def _get_dst_indices(df):
    """
    given a datetime-indexed dataframe,
    return the indices which need to be interpolated and averaged
    in order to ensure exact 24 hour slots
    """
    # TODO test on baselines that begin/end on DST change
    counts = df.groupby(df.index.date).count()
    first_col = counts.columns[0]
    interp = counts[counts[first_col] == 23]
    mean = counts[counts[first_col] == 25]

    interp_idx = []
    for idx in interp.index:
        day_data = df.loc[idx.isoformat()]
        date_idx = counts.index.get_loc(idx)
        missing_hour = set(range(24)) - set(day_data.index.hour)
        if len(missing_hour) != 1:
            raise ValueError("too many missing hours")

        interp_idx.append((date_idx, missing_hour.pop()))

    mean_idx = []
    for idx in mean.index:
        date_idx = counts.index.get_loc(idx)
        day_data = df.loc[idx.isoformat()]
        seen = set()
        for i in day_data.index:
            if i.hour in seen:
                mean_idx.append((date_idx, i.hour))
                break

            seen.add(i.hour)

    return interp_idx, mean_idx


def _transform_dst(prediction, dst_indices):
    interp, mean = dst_indices

    START_END = 0
    REMOVE = 1
    INTERPOLATE = 2

    # get concrete indices
    remove_idx = [(REMOVE, date * 24 + hour) for date, hour in interp]
    interp_idx = [(INTERPOLATE, date * 24 + hour + 1) for date, hour in mean]

    # these values will be inserted for the 25th hour
    interpolated_vals = []
    for _, idx in interp_idx:
        interpolated = (prediction[idx - 1] + prediction[idx]) / 2
        interpolated_vals.append(interpolated)

    interpolation = iter(interpolated_vals)

    # sort "operations" by index (can't assume a strict back-and-forth ordering)
    ops = sorted(remove_idx + interp_idx, key=lambda t: t[1])

    # create fenceposts where slices end
    pairs = list(zip([(START_END, 0)] + ops, ops + [(START_END, None)]))
    slices = []
    for start, end in pairs:
        start_i = start[1]
        end_i = end[1]
        if start[0] == REMOVE:
            start_i += 1

        if start[0] == INTERPOLATE:
            slices.append([next(interpolation)])

        slices.append(prediction[slice(start_i, end_i)])

    return np.concatenate(slices)


class HourlyModel:
    """
    A class to fit a model to the input meter data.

    Attributes:
        settings (dict): A dictionary of settings.
        baseline_metrics (dict): A dictionary of metrics based on input baseline data and model fit.
    """
    
    # thresholds for switching model types
    _alpha_model_threshold = 1E-5
    _l1_ratio_model_threshold = 1E-4
    _model_warning = EEMeterWarning
    _base_settings = _settings

    # set priority columns for sorting
    # this is critical for ensuring predict column order matches fit column order
    _priority_cols = {
        "ts": ["temporal_cluster", "temp_bin", "temperature", "ghi"],
        "cat": ["temporal_cluster", "temp_bin"],
    }

    _temporal_cluster_cols = ["month", "day_of_week"]

    """Note:
        Despite the temporal clusters, we can view all models created as a subset of the same full model.
        The temporal clusters would simply have the same coefficients within the same days/month combinations.
    """

    def __init__(
        self,
        settings: dict | _settings.BaseHourlySettings | None = None,
    ):
        """
        Args:
            settings: HourlySettings to use (generally left default). Will default to solar model if GHI is given to the fit step.
        """

        # TODO move this logic into HourlySettings init
        if isinstance(settings, dict):
            features = settings.get("train_features")
            if features is not None:
                if "ghi" in features:
                    settings = _settings.HourlySolarSettings(**settings)
                else:
                    settings = _settings.HourlyNonSolarSettings(**settings)
            else:
                settings = _settings.BaseHourlySettings(**settings)

        # Initialize settings
        if settings is None:
            self.settings = _settings.BaseHourlySettings()
        else:
            self.settings = settings

        # Initialize model
        self._set_scalers()
        self._model = self._set_model()

        self._T_bin_edges = None
        self._T_edge_bin_coeffs = None
        self._df_temporal_clusters = None
        self._categorical_features = None
        self._ts_feature_norm = None

        self._ts_features = []
        if self.settings.train_features:
            self._ts_features = self.settings.train_features.copy()

        self._is_fit = False
        self.baseline_metrics = None
        self.baseline_hour_metrics = None

        self.warnings = []
        self.disqualification = []

        self.baseline_timezone = None
        self.version = __version__

    
    def _warn_model_mismatch(self, description):
        warning = self._model_warning(
            qualified_name="eemeter.potential_model_mismatch",
            description=description,
            data={},
        )
        warning.warn()
        self.warnings.append(warning)

    def _set_scalers(self):
        # set scalers
        if self.settings.scaling_method == _settings.ScalingChoice.STANDARD_SCALER:
            self._feature_scaler = SafeStandardScaler()
            self._y_scaler = SafeStandardScaler()
        elif self.settings.scaling_method == _settings.ScalingChoice.ROBUST_SCALER:
            self._feature_scaler = SafeRobustScaler(unit_variance=True)
            self._y_scaler = SafeRobustScaler(unit_variance=True)


    def _set_model(self):
        # set base model
        if self.settings.base_model == _settings.BaseModel.ELASTICNET:
            settings = self.settings.elasticnet
            if settings.alpha <= self._alpha_model_threshold:
                model = SafeLinearRegression(
                    fit_intercept=settings.fit_intercept
                )

            else:
                if settings.l1_ratio < self._l1_ratio_model_threshold:
                    base_model = SafeRidge
                elif settings.l1_ratio > (1 - self._l1_ratio_model_threshold):
                    base_model = SafeLasso
                else:
                    base_model = SafeElasticNet

                model = base_model(
                    alpha=settings.alpha,
                    fit_intercept=settings.fit_intercept,
                    max_iter=settings.max_iter,
                    tol=settings.tol,
                    random_state=settings._seed,
                )
                
                if not isinstance(model, Ridge):
                    model.precompute = settings.precompute
                    model.selection = settings.selection
                    model.warm_start = settings.warm_start

                if isinstance(model, ElasticNet):
                    model.l1_ratio = settings.l1_ratio

            if self.settings.adaptive_weights.enabled:
                model = AdaptiveElasticNetRegressor(model, self.settings)
                
        elif self.settings.base_model == _settings.BaseModel.KERNEL_RIDGE:
            settings = self.settings.kernel_ridge
            model = KernelRidge(
                alpha=settings.alpha,
                kernel=settings.kernel,
                gamma=settings.gamma,
            )

        return model

    def _model_prefit_check(self, baseline_data):
        if "ghi" in self._ts_features and "ghi" not in baseline_data.df.columns:
            raise ValueError(
                "Model was explicitly set to use GHI, but baseline data does not contain GHI."
            )

        if "ghi" in baseline_data.df.columns and "ghi" not in self._ts_features:
            self._warn_model_mismatch(
                "Model was explicitly set to ignore GHI, but baseline period contained a GHI column."
            )

        if np.allclose(baseline_data.df["observed"].values, 0):
            model_fit_warning = self._model_warning(
                qualified_name="eemeter.model_fit",
                description="Model cannot be fit: Observed column contains all zeros.",
            )
            model_fit_warning.warn()
            self.disqualification.append(model_fit_warning)
            raise DataSufficiencyError("Cannot fit model: Baseline data contains all zeros in observed values")

        if baseline_data.df["observed"].isnull().all():
            model_fit_warning = self._model_warning(
                qualified_name="eemeter.model_fit",
                description="Model cannot be fit: Observed column contains no finite values.",
            )
            model_fit_warning.warn()
            self.disqualification.append(model_fit_warning)
            raise DataSufficiencyError("Cannot fit model: Baseline data contains no finite observed values")


    def fit(
        self, baseline_data: HourlyBaselineData, ignore_disqualification: bool = False
    ) -> HourlyModel:
        """Fit the model using baseline data.

        Args:
            baseline_data: HourlyBaselineData object.
            ignore_disqualification: Whether to ignore disqualification errors / warnings.

        Returns:
            The fitted model.

        Raises:
            TypeError: If baseline_data is not an HourlyBaselineData object.
            DataSufficiencyError: If the model can't be fit on disqualified baseline data.
        """
        if not isinstance(baseline_data, HourlyBaselineData):
            raise TypeError("baseline_data must be an HourlyBaselineData object")

        baseline_data.log_warnings()

        if baseline_data.disqualification and not ignore_disqualification:
            raise DataSufficiencyError("Can't fit model on disqualified baseline data")

        self.warnings = baseline_data.warnings
        self.disqualification = baseline_data.disqualification

        if not self._ts_features:
            self.settings = self.settings.add_default_features(baseline_data.df.columns)
            self._ts_features = self.settings.train_features.copy()

        self._model_prefit_check(baseline_data)
        
        self._fit(baseline_data)
        self._check_model_fit()

        return self

    def _fit(self, meter_data):
        self._is_fit = False

        # Initialize dataframe
        df_meter = meter_data.df  # used to have a copy here
        self.baseline_timezone = meter_data.tz

        # Prepare feature arrays/matrices
        X, y, fit_mask = self._prepare_features(df_meter)
        X_fit = X[fit_mask, :]
        y_fit = y[fit_mask]

        # fit the model
        self._model.fit(X_fit, y_fit)
        self._is_fit = True

        # get model prediction of baseline
        df_meter = self._predict(meter_data, X=X)

        self._set_baseline_metrics(df_meter, X_fit=X_fit)

        return self

    def predict(
        self,
        reporting_data,
        ignore_disqualification=False,
    ) -> pd.DataFrame:
        """Predicts the energy consumption using the fitted model.

        Args:
            reporting_data (Union[HourlyBaselineData, HourlyReportingData]): The data used for prediction.
            ignore_disqualification (bool, optional): Whether to ignore model disqualification. Defaults to False.

        Returns:
            Dataframe with input data along with predicted energy consumption.

        Raises:
            RuntimeError: If the model is not fitted.
            DisqualifiedModelError: If the model is disqualified and ignore_disqualification is False.
            TypeError: If the reporting data is not of type HourlyBaselineData or HourlyReportingData.
        """
        if not self._is_fit:
            raise RuntimeError("Model must be fit before predictions can be made.")

        if missing_features := (
            set(self._ts_features) - set(reporting_data.df.columns)
        ):
            raise ValueError(
                f"Reporting data is missing the following features: {missing_features}"
            )

        if "ghi" in reporting_data.df.columns and "ghi" not in self._ts_features:
            self._warn_model_mismatch(
                "Reporting data contains GHI, but model was fit without GHI."
            )

        if str(self.baseline_timezone) != str(reporting_data.tz):
            raise ValueError(
                "Reporting data must use the same timezone that the model was initially fit on."
            )

        if self.disqualification and not ignore_disqualification:
            raise DisqualifiedModelError(
                "Attempting to predict using disqualified model without setting ignore_disqualification=True"
            )

        if not isinstance(reporting_data, (HourlyBaselineData, HourlyReportingData)):
            raise TypeError(
                "reporting_data must be a HourlyBaselineData or HourlyReportingData object"
            )

        return self._predict(reporting_data)

    def _predict(self, eval_data, X=None):
        """
        Makes model prediction on given temperature data.

        Parameters:
            df_eval (pandas.DataFrame): The evaluation dataframe.

        Returns:
            pandas.DataFrame: The evaluation dataframe with model predictions added.
        """

        df_eval = eval_data.df  # used to have a copy here
        dst_indices = _get_dst_indices(df_eval)
        datetime_original = df_eval.index
        # get list of columns to keep in output
        columns = df_eval.columns.tolist()
        if "datetime" in columns:
            columns.remove("datetime")  # index in output, not column

        if X is None:
            X, _, _ = self._prepare_features(df_eval)

        y_predict = self._y_scaler.inverse_transform(self._model.predict(X)).flatten()

        y_predict = _transform_dst(y_predict, dst_indices)

        df_eval["predicted"] = y_predict
        df_eval = self._calculate_predicted_uncertianty(df_eval)

        # remove columns not in original columns and predicted
        df_eval = df_eval[[*columns, "predicted", "predicted_unc"]]

        # reindex to original datetime index
        df_eval = df_eval.reindex(datetime_original)

        return df_eval

    def _prepare_features(self, meter_data):
        """Prepare feature matrices from meter data for model fitting or prediction."""
        dst_indices = _get_dst_indices(meter_data)
        meter_data = self._add_categorical_features(meter_data)
        self._add_supplemental_features(meter_data)

        self._ts_features, self._categorical_features = self._sort_features(
            self._ts_features, self._categorical_features
        )

        meter_data = self._daily_fitting_sufficiency(meter_data)
        meter_data = self._normalize_features(meter_data)
        meter_data = self._add_temperature_interactions(meter_data)

        # save actual df used for later inspection
        self._ts_feature_norm, _ = self._sort_features(self._ts_feature_norm)
        selected_features = self._ts_feature_norm + self._categorical_features
        if "observed_norm" in meter_data.columns:
            selected_features += ["observed_norm"]

        self._processed_meter_data_full = meter_data
        self._processed_meter_data = self._processed_meter_data_full[selected_features]

        # get feature matrices
        X, y, fit_mask = self._get_feature_matrices(meter_data, dst_indices)

        # Convert to sparse matrix
        X = csr_matrix(X.astype(float))

        return X, y, fit_mask

    def _add_temperature_bins(self, df):
        # TODO: do we need to do something about empty bins in prediction? I think not but maybe
        settings = self.settings.temperature_bin

        # add temperature bins based on temperature
        if not self._is_fit:
            if settings.method == "equal_sample_count":
                T_bin_edges = pd.qcut(
                    df["temperature"], q=settings.n_bins, labels=False
                )

            elif settings.method == "equal_bin_width":
                T_bin_edges = pd.cut(
                    df["temperature"], bins=settings.n_bins, labels=False
                )

            elif settings.method == "set_bin_width":
                bin_width = settings.bin_width

                min_temp = np.floor(df["temperature"].min())
                max_temp = np.ceil(df["temperature"].max())

                if not settings.include_edge_bins:
                    step_num = (
                        np.round((max_temp - min_temp) / bin_width).astype(int) + 1
                    )

                    # T_bin_edges = np.arange(min_temp, max_temp + bin_width, bin_width)
                    T_bin_edges = np.linspace(min_temp, max_temp, step_num)

                else:
                    set_edge_bin_width = False
                    if set_edge_bin_width:
                        edge_bin_width = bin_width * 1 / 2

                        bin_range = [
                            min_temp + edge_bin_width,
                            max_temp - edge_bin_width,
                        ]

                    else:
                        edge_bin_count = int(len(df) * settings.edge_bin_percent)

                        # get 5th smallest and 5th largest temperatures
                        sorted_temp = np.sort(df["temperature"])
                        min_temp_reg_bin = np.ceil(sorted_temp[edge_bin_count])
                        max_temp_reg_bin = np.floor(sorted_temp[-edge_bin_count])

                        bin_range = [min_temp_reg_bin, max_temp_reg_bin]

                    step_num = (
                        np.round((bin_range[1] - bin_range[0]) / bin_width).astype(int)
                        + 1
                    )

                    # create bins with set width
                    T_bin_edges = np.array(
                        [min_temp, *np.linspace(*bin_range, step_num), max_temp]
                    )

            elif settings.method == "fixed_bins":
                temp = df["temperature"].values

                T_bin_edges = np.array(settings.fixed_bins)
                T_bin_edges = np.array([-np.inf, *T_bin_edges, np.inf])

                if temp.size < settings.min_bin_count:
                    raise ValueError("Not enough data to form temperature bins")
                elif temp.size < settings.min_bin_count*2:
                    T_bin_edges = np.array([-np.inf, np.inf])
                else:
                    T_bin_edges = _merge_bins(T_bin_edges, temp, settings.min_bin_count)

            else:
                raise ValueError("Invalid temperature binning method")

            # set the first and last bin to -inf and inf
            T_bin_edges[0] = -np.inf
            T_bin_edges[-1] = np.inf

            # store bin edges for prediction
            self._T_bin_edges = T_bin_edges

        T_bins = pd.cut(df["temperature"], bins=self._T_bin_edges, labels=False)

        df["temp_bin"] = T_bins

        # Create dummy variables for temperature bins
        bin_dummies = pd.get_dummies(
            pd.Categorical(
                df["temp_bin"], categories=range(len(self._T_bin_edges) - 1)
            ),
            prefix="temp_bin",
        )
        bin_dummies.index = df.index

        col_names = bin_dummies.columns.tolist()
        df = pd.merge(df, bin_dummies, how="left", left_index=True, right_index=True)

        return df, col_names

    def _add_categorical_features(self, df):
        def set_initial_temporal_clusters(df):
            fit_df_grouped = (
                df.groupby(self._temporal_cluster_cols + ["hour_of_day"])["observed"]
                .agg(self.settings.temporal_cluster_aggregation)
                .reset_index()
            )
            # pivot table to get 2D array of observed values
            fit_df_grouped = fit_df_grouped.pivot_table(
                index=self._temporal_cluster_cols,
                columns="hour_of_day",
                values="observed",
            )

            labels = cluster_features(
                fit_df_grouped,
                self.settings.temporal_cluster
            )

            df_temporal_clusters = pd.DataFrame(
                labels,
                columns=["temporal_cluster"],
                index=fit_df_grouped.index,
            )

            return df_temporal_clusters

        def correct_missing_temporal_clusters(df):
            # check and match any missing temporal combinations

            # get all unique combinations of month and day_of_week in df
            df_temporal = df[self._temporal_cluster_cols].drop_duplicates()
            df_temporal = df_temporal.sort_values(self._temporal_cluster_cols)
            df_temporal_index = df_temporal.set_index(self._temporal_cluster_cols).index

            # reindex self.df_temporal_clusters to df_temporal_index
            df_temporal_clusters = self._df_temporal_clusters.reindex(df_temporal_index)

            # get index of any nan values in df_temporal_clusters
            missing_combinations = df_temporal_clusters[
                df_temporal_clusters["temporal_cluster"].isna()
            ].index
            if not missing_combinations.empty:
                if missing_combinations == df_temporal_index:
                    raise ValueError(
                        f"Data does not have known temporal clusters of {self._temporal_cluster_cols}. Can't assign missing temporal clusters"
                    )

                elif "observed" in df.columns and not df["observed"].isnull().all():
                    # precompute temporal index membership for reuse
                    temporal_idx = df.set_index(self._temporal_cluster_cols).index
                    is_missing = temporal_idx.isin(missing_combinations)

                    # filter df to only include missing combinations
                    df_missing = df[is_missing]

                    df_missing_grouped = (
                        df_missing.groupby(
                            self._temporal_cluster_cols + ["hour_of_day"]
                        )["observed"]
                        .agg(self.settings.temporal_cluster_aggregation)
                        .reset_index()
                    )
                    df_missing_grouped = df_missing_grouped.pivot_table(
                        index=self._temporal_cluster_cols,
                        columns="hour_of_day",
                        values="observed",
                    )
                    X = df_missing_grouped.values

                    # calculate average observed for known clusters
                    # join df_temporal_clusters to df on month and day_of_week
                    df = pd.merge(
                        df,
                        df_temporal_clusters,
                        how="left",
                        left_on=self._temporal_cluster_cols,
                        right_index=True,
                    )

                    df_known = df[~is_missing]

                    df_known_mean = (
                        df_known.groupby(self._temporal_cluster_cols + ["hour_of_day"])[
                            "observed"
                        ]
                        .mean()
                        .reset_index()
                    )
                    df_known_mean = df_known_mean.pivot_table(
                        index=self._temporal_cluster_cols,
                        columns="hour_of_day",
                        values="observed",
                    )
                    X_known = df_known_mean.values

                    # get smallest distance between X and X_known
                    dist = cdist(X, X_known, metric="euclidean")
                    min_dist_idx = np.argmin(dist, axis=1)

                    # get temporal clusters df_known
                    temporal_clusters = df_known.groupby(self._temporal_cluster_cols)[
                        "temporal_cluster"
                    ].first()
                    temporal_clusters = temporal_clusters.reindex(df_known_mean.index)

                    # set labels to minimum distance of known clusters
                    labels = temporal_clusters.iloc[min_dist_idx].values
                    df_temporal_clusters.loc[
                        missing_combinations, "temporal_cluster"
                    ] = labels

                    self._df_temporal_clusters = df_temporal_clusters

                else:
                    # TODO: There's better ways of handling this
                    # unstack and fill missing days in each month
                    # assuming months more important than days
                    df_temporal_clusters = df_temporal_clusters.unstack()

                    # fill missing days in each month
                    df_temporal_clusters = df_temporal_clusters.ffill(axis=1)
                    df_temporal_clusters = df_temporal_clusters.bfill(axis=1)

                    # fill missing months if any remaining empty
                    df_temporal_clusters = df_temporal_clusters.ffill(axis=0)
                    df_temporal_clusters = df_temporal_clusters.bfill(axis=0)

                    df_temporal_clusters = df_temporal_clusters.stack()

            return df_temporal_clusters

        # assign basic temporal features
        df["date"] = df.index.date
        df["month"] = df.index.month
        df["day_of_week"] = df.index.dayofweek
        df["hour_of_day"] = df.index.hour

        # assign temporal clusters
        if not self._is_fit:
            self._df_temporal_clusters = set_initial_temporal_clusters(df)
            n_clusters = self._df_temporal_clusters["temporal_cluster"].nunique()

        else:
            self._df_temporal_clusters = correct_missing_temporal_clusters(df)

            # Count unique temporal cluster base columns (e.g. temporal_cluster_0, temporal_cluster_1)
            n_clusters = sum(
                1 for col in self._categorical_features
                if re.match(r'^temporal_cluster_\d+$', col)
            )

        # join df_temporal_clusters to df
        df = pd.merge(
            df,
            self._df_temporal_clusters,
            how="left",
            left_on=self._temporal_cluster_cols,
            right_index=True,
        )

        cluster_dummies = pd.get_dummies(
            pd.Categorical(df["temporal_cluster"], categories=range(n_clusters)),
            prefix="temporal_cluster",
        )
        cluster_dummies.index = df.index

        self._categorical_features = [f"temporal_cluster_{i}" for i in range(n_clusters)]

        df = pd.merge(
            df, cluster_dummies, how="left", left_index=True, right_index=True
        )

        if self.settings.temperature_bin is not None:
            df, temp_bin_cols = self._add_temperature_bins(df)
            self._categorical_features.extend(temp_bin_cols)

        return df

    def _add_supplemental_features(self, df):
        # TODO: should either do upper or lower on all strs
        if self.settings.supplemental_time_series_columns is not None:
            for col in self.settings.supplemental_time_series_columns:
                if col in df.columns and col not in self._ts_features:
                    self._ts_features.append(col)

        if self.settings.supplemental_categorical_columns is not None:
            existing = set(self._ts_features) | set(self._categorical_features)
            for col in self.settings.supplemental_categorical_columns:
                if col in df.columns and col not in existing:
                    self._categorical_features.append(col)
                    existing.add(col)

    def _sort_features(self, ts_features=None, cat_features=None):
        def _sort_by_priority(features, priority_prefixes):
            if features is None:
                return None
            sorted_cols = []
            for prefix in priority_prefixes:
                sorted_cols.extend(sorted(c for c in features if c.startswith(prefix)))

            sorted_set = set(sorted_cols)
            sorted_cols.extend(sorted(c for c in features if c not in sorted_set))
            return sorted_cols

        ts = _sort_by_priority(ts_features, self._priority_cols["ts"])
        cat = _sort_by_priority(cat_features, self._priority_cols["cat"])

        return ts, cat

    # TODO rename to avoid confusion with data sufficiency
    def _daily_fitting_sufficiency(self, df):
        # remove days with insufficient data
        min_hours = self.settings.min_daily_training_hours

        if min_hours > 0:
            # find any rows with interpolated or missing data
            df["interpolated"] = _get_interpolated_mask(df) | df.isnull().any(axis=1)

            # count number of non interpolated hours per day
            daily_hours = 24 - df.groupby("date")["interpolated"].sum()
            sufficient_days = daily_hours[daily_hours >= min_hours].index

            # set "include_day" column to True if day has sufficient hours
            df["include_date"] = df["date"].isin(sufficient_days)

        else:
            df["include_date"] = True

        return df

    def _normalize_features(self, df):
        """ """
        train_features = self._ts_features
        self._ts_feature_norm = [i + "_norm" for i in train_features]

        # need to set scaler if not fit
        if not self._is_fit:
            self._feature_scaler.fit(df[train_features].values)
            self._y_scaler.fit(df["observed"].values.reshape(-1, 1))

        data_transformed = self._feature_scaler.transform(df[train_features].values)
        normalized_df = pd.DataFrame(
            data_transformed, index=df.index, columns=self._ts_feature_norm
        )

        df = pd.concat([df, normalized_df], axis=1)

        if "observed" in df.columns:
            df["observed_norm"] = self._y_scaler.transform(
                df["observed"].values.reshape(-1, 1)
            )

        if "ghi" in self._ts_features and "ghi" in df.columns:
            df["ghi_norm"] *= self.settings.ghi_scalar

        return df
    
    def _add_extreme_temperature_bins(self, df, bin_range):
        settings = self.settings.temperature_bin

        def get_k(int_col, a, b):
            k = []
            for hour in range(24):
                df_hour = df[df["hour_of_day"] == hour]
                df_hour = df_hour.sort_values(by=int_col)

                x_data = a * df_hour[int_col].values + b
                y_data = df_hour["observed"].values

                # Fit the model using robust least squares
                try:
                    params = _fit_exp_growth_decay(
                        x_data, y_data, k_only=True, is_x_sorted=True
                    )
                    # save k for each hour
                    k.append(params[2])
                except Exception:
                    pass

            k = np.abs(np.array(k))
            k_valid = k[k < 5]

            if len(k_valid) > 0:
                k = np.mean(k_valid)
            else:
                k = 1 # if no valid k, set to 1

            # if k is too small, set to minimum
            k_min = 1/np.log(1E6)
            if k < k_min:
                k = k_min

            return k

        if self._T_edge_bin_coeffs is None:
            self._T_edge_bin_coeffs = {}

        cols = bin_range
        # maybe add nonlinear terms to second and second to last columns?
        # cols = [0, 1, last_temp_bin - 1, last_temp_bin]
        # cols = list(set(cols))
        # all columns?
        # cols = range(cols[0], cols[1] + 1)

        # Add all columns using col_dict at end
        col_dict = {}
        for n in cols:
            base_col = f"temp_bin_{n}"
            int_col = f"{base_col}_ts"
            T_col = f"{base_col}_T"

            # get k for exponential growth/decay
            if not self._is_fit:
                # determine temperature conversion for bin
                range_offset = settings.edge_bin_temperature_range_offset
                T_range = [
                    df[int_col].min() - range_offset,
                    df[int_col].max() + range_offset,
                ]
                new_range = [-1, 1]

                T_a = (new_range[1] - new_range[0]) / (T_range[1] - T_range[0])
                T_b = new_range[1] - T_a * T_range[1]

                # The best rate for exponential
                if settings.edge_bin_rate == "heuristic":
                    k = get_k(int_col, T_a, T_b)
                else:
                    k = settings.edge_bin_rate

                # get A for exponential
                A = 1 / (np.exp(1 / k * new_range[1]) - 1)

                self._T_edge_bin_coeffs[n] = {
                    "t_a": float(T_a),
                    "t_b": float(T_b),
                    "k": float(k),
                    "a": float(A),
                }

            T_a = self._T_edge_bin_coeffs[n]["t_a"]
            T_b = self._T_edge_bin_coeffs[n]["t_b"]
            k = self._T_edge_bin_coeffs[n]["k"]
            A = self._T_edge_bin_coeffs[n]["a"]

            col_dict[T_col] = np.where(
                df[base_col].values, T_a * df[int_col].values + T_b, 0
            )

            for label, sign in [("pos", 1), ("neg", -1)]:
                ts_col = f"{base_col}_{label}_exp_ts"

                col_dict[ts_col] = np.where(
                    df[base_col].values, A * np.exp(sign / k * col_dict[T_col]) - A, 0
                )

                self._ts_feature_norm.append(ts_col)

        # create new df with col_dict
        df = pd.concat([df, pd.DataFrame(col_dict, index=df.index)], axis=1)

        return df

    def _add_temperature_interactions(self, df):
        settings = self.settings.temperature_bin

        # TODO: if this permanent then it should not create, erase, make anew
        self._ts_feature_norm.remove("temperature_norm")

        temp_bin_cols = [c for c in df.columns if re.match(r'^temp_bin_\d+$', c)]
        cluster_cols = [c for c in df.columns if re.match(r'^temporal_cluster_\d+$', c)]

        col_dict = {}

        # add global temperature bins
        for col in temp_bin_cols:
            # splits temperature_norm into unique columns if that temp_bin column is True
            ts_col = f"{col}_ts"
            col_dict[ts_col] = df["temperature_norm"] * df[col]

            self._ts_feature_norm.append(ts_col)

        # add temporal cluster interactions
        # multiply each temp_bin by each temporal cluster
        # get all columns that start with temp_bin_ and are a number
        s = self.settings.interaction_scalar
        for temporal_cluster_col in cluster_cols:
            for temp_bin_col in temp_bin_cols:
                # add intercept term
                interaction_col = f"{temporal_cluster_col}_{temp_bin_col}_interact"
                col_dict[interaction_col] = df[temp_bin_col] * df[temporal_cluster_col]

                # add slope term
                interaction_ts_col = f"{interaction_col}_ts"
                # df[interaction_ts_col] = df["temperature_norm"] * df[interaction_col]
                col_dict[interaction_ts_col] = s*df["temperature_norm"] * col_dict[interaction_col]

                # add to feature lists
                self._categorical_features.append(interaction_col)
                self._ts_feature_norm.append(interaction_ts_col)

        # concat df with col_dict
        df = pd.concat([df, pd.DataFrame(col_dict, index=df.index)], axis=1)

        # TODO: Model is better without this, but not sure why
        # remove temporal cluster columns from categorical features
        # cluster_cols = [c for c in df.columns if re.match(r'^temporal_cluster_\d+(?!_)', c)]
        # self._categorical_features = [c for c in self._categorical_features if c not in cluster_cols]

        # add extreme temperature bins to global temperature bins
        if settings.include_edge_bins:
            bin_range = [0, len(temp_bin_cols) - 1]
            df = self._add_extreme_temperature_bins(df, bin_range)

        return df

    def _get_feature_matrices(self, df, dst_indices):
        # get aggregated features with agg function
        agg_dict = {f: list for f in self._ts_feature_norm}

        def correct_dst(agg):
            """interpolate or average hours to account for DST. modifies in place"""
            interp, mean = dst_indices
            for date, hour in interp:
                for feature_idx, feature in enumerate(agg[date]):
                    if hour == 0:
                        # there are a handful of countries that use 0:00 as the DST transition
                        interpolated = (
                            agg[date - 1][feature_idx][-1] + feature[hour]
                        ) / 2

                    else:
                        interpolated = (feature[hour - 1] + feature[hour]) / 2

                    feature.insert(hour, interpolated)

            for date, hour in mean:
                for feature in agg[date]:
                    avg = (feature[hour + 1] + feature.pop(hour)) / 2
                    feature[hour] = avg

        df_grouped = df.groupby("date")
        agg_x = df_grouped.agg(agg_dict).values.tolist()
        correct_dst(agg_x)

        # get the features and target for each day
        ts_feature = np.array(agg_x).reshape(len(agg_x), -1)

        # get the first categorical features for each day for each sample
        unique_dummies = (
            df[["date"] + self._categorical_features].groupby("date").first()
        )

        X = np.concatenate((ts_feature, unique_dummies), axis=1)

        if not self._is_fit:
            agg_y = (
                df_grouped
                .agg({"observed_norm": list})
                .values.tolist()
            )
            correct_dst(agg_y)
            y = np.array(agg_y)
            y = y.reshape(y.shape[0], -1)

            fit_mask = df_grouped["include_date"].first().values

        else:
            y = None
            fit_mask = None

        return X, y, fit_mask

    def _set_baseline_metrics(self, df_meter, X_fit=None):
        # unwrap adaptive model if needed
        adaptive_weights = None
        if self.settings.base_model == _settings.BaseModel.ELASTICNET:
            if self.settings.adaptive_weights.enabled:
                if hasattr(self._model, 'base_model'):
                    adaptive_weights = getattr(self._model.base_model, 'adaptive_weights', None)
                self._model = self._model.base_model

            coef = self._model.coef_
        elif self.settings.base_model == _settings.BaseModel.KERNEL_RIDGE:
            coef = self._model.dual_coef_

        # calculate baseline metrics on non-interpolated data
        interpolated = _get_interpolated_mask(df_meter)
        df_non_interp = df_meter.loc[~interpolated]

        num_parameters = np.count_nonzero(coef)
        self.baseline_metrics = BaselineMetrics(
            df=df_non_interp,
            num_model_params=num_parameters + 1, # + 1 for intercept
        )

        # calculate baseline metrics per hour-of-day with edf-based num_model_params
        self.baseline_hour_metrics = {}
        is_kernel = self.settings.base_model == _settings.BaseModel.KERNEL_RIDGE

        # Compute lambda_2 for edf (only for ElasticNet/Ridge, not KernelRidge)
        lambda_2 = self._compute_lambda_2() if not is_kernel else None

        for hour in range(24):
            hour_coef = coef[:, hour] if is_kernel else coef[hour]
            hour_data = df_non_interp[df_non_interp.index.hour == hour]

            if len(hour_data) < 3:
                self.baseline_hour_metrics[hour] = None
                continue

            # Compute edf via SVD when X_fit is available and model is not KernelRidge
            if X_fit is not None and not is_kernel and lambda_2 is not None:
                edf_h = self._compute_hour_edf(
                    X_fit, hour, hour_coef, lambda_2, adaptive_weights,
                )
                # Clamp so ddof >= 3 → stdtrit(ddof, perc) is finite and reasonable
                n_h = len(hour_data)
                edf_h = max(1, min(edf_h, n_h - 3))
                num_params_h = max(1, round(edf_h))
            else:
                # Fallback: count_nonzero (KernelRidge or missing X_fit)
                num_params_h = np.count_nonzero(hour_coef) + 1

            self.baseline_hour_metrics[hour] = BaselineMetrics(
                df=hour_data,
                num_model_params=num_params_h,
            )

    def _compute_lambda_2(self):
        """Compute the L2 penalty parameter for the edf formula.

        sklearn Ridge and ElasticNet use different loss scaling:
          Ridge:      ||y - Xw||² + α||w||²           → λ₂ = α
          ElasticNet: (1/(2n))||y - Xw||² + ...       → λ₂ = n·α·(1-ρ)
        """
        base = self._model
        if isinstance(base, Ridge) and not isinstance(base, ElasticNet):
            return base.alpha
        else:
            alpha = getattr(base, 'alpha', 0)
            l1_ratio = getattr(base, 'l1_ratio', 0)
            n = self.baseline_metrics.n
            return n * alpha * (1 - l1_ratio)

    def _compute_hour_edf(self, X_fit, hour, hour_coef, lambda_2, adaptive_weights):
        """Compute effective degrees of freedom for hour h via SVD.

        Uses the Zou-Hastie closed form: edf = Σ d²/(d² + λ₂) where d are
        singular values of the (optionally weighted) design matrix restricted
        to active columns at this hour. +1 for intercept.

        X_fit is per-day (shape n_fit_days × n_features), shared across all hours.
        Each hour uses the same X but different active columns from coef[hour].
        """
        active = np.nonzero(hour_coef)[0]
        if len(active) == 0:
            return 1  # intercept only

        X_active = X_fit[:, active]
        if hasattr(X_active, 'toarray'):
            X_active = X_active.toarray()

        # Apply per-hour adaptive weights (also daily-indexed)
        if adaptive_weights is not None:
            w_h = adaptive_weights[:, hour]
            X_active = np.sqrt(w_h)[:, np.newaxis] * X_active

        s = np.linalg.svd(X_active, compute_uv=False)
        edf = float(np.sum(s**2 / (s**2 + lambda_2))) + 1  # +1 for intercept

        return edf

    def _check_model_fit(self):
        cvrmse = self.baseline_metrics.cvrmse_adj
        pnrmse = self.baseline_metrics.pnrmse_adj

        cvrmse_threshold = self.settings.cvrmse_threshold
        pnrmse_threshold = self.settings.pnrmse_threshold

        # sufficient is (0 <= cvrmse <= threshold) or (0 <= pnrmse <= threshold)
        cvrmse_ok = cvrmse is not None and 0 <= cvrmse <= cvrmse_threshold
        pnrmse_ok = pnrmse is not None and 0 <= pnrmse <= pnrmse_threshold

        if not (cvrmse_ok or pnrmse_ok):
            model_fit_warning = self._model_warning(
                qualified_name="eemeter.model_fit_metrics",
                description="Model disqualified due to poor fit.",
                data={
                    "cvrmse_threshold": cvrmse_threshold,
                    "cvrmse_adj": cvrmse,
                    "pnrmse_threshold": pnrmse_threshold,
                    "pnrmse_adj": pnrmse,
                },
            )
            model_fit_warning.warn()
            self.disqualification.append(model_fit_warning)

    def _calculate_predicted_uncertianty(self, df_eval):
        df_eval["predicted_unc"] = np.nan

        if self._has_per_hour_uncertainty():
            return self._calculate_uncertainty_per_hour(df_eval)
        else:
            return self._calculate_uncertainty_global_legacy(df_eval)

    def _has_per_hour_uncertainty(self):
        """True for models trained with per-hour edf-based uncertainty data.
        False for models deserialized from old model_json that lack per-hour metrics.

        DELETE this guard (and _calculate_uncertainty_global_legacy) once all
        deployed model_json artifacts have been retrained."""
        return (
            self.baseline_hour_metrics is not None
            and any(v is not None for v in self.baseline_hour_metrics.values())
        )

    def _calculate_uncertainty_per_hour(self, df_eval):
        """Per-hour heteroscedastic uncertainty (A+B+C+E).

        Each hour uses its own BaselineMetrics (with edf-based num_model_params)
        fed into ReportingMetrics to compute per-point uncertainty via the full
        ASHRAE-14 formula. The uncertainty_scale_factor is applied as a final
        multiplicative correction for pipeline-stage bias not captured by edf.
        """
        interpolated = _get_interpolated_mask(df_eval)
        usf = self.settings.uncertainty_scale_factor
        confidence_level = 1 - self.settings.uncertainty_alpha

        for hour in range(24):
            mask = df_eval.index.hour == hour
            if not mask.any():
                continue

            bm_h = self.baseline_hour_metrics.get(hour)
            if bm_h is None:
                continue

            reporting_df = df_eval.loc[mask & ~interpolated]
            if reporting_df.empty:
                continue

            reporting_h = ReportingMetrics(
                baseline_metrics=bm_h,
                reporting_df=reporting_df,
                data_frequency="hourly",
                confidence_level=confidence_level,
                t_tail=2,
            )

            unc = reporting_h.predicted_data_point_unc
            if unc is not None and np.isfinite(unc):
                df_eval.loc[mask, "predicted_unc"] = usf * unc

        return df_eval

    def _calculate_uncertainty_global_legacy(self, df_eval):
        """Global homoscedastic uncertainty. Legacy path for old model_json
        that lack per-hour baseline metrics.

        DELETE this method (and the _has_per_hour_uncertainty guard) once all
        deployed model_json artifacts have been retrained."""
        interpolated = _get_interpolated_mask(df_eval)

        if self.baseline_metrics is None:
            return df_eval

        reporting_metrics = ReportingMetrics(
            baseline_metrics=self.baseline_metrics,
            reporting_df=df_eval[~interpolated],
            data_frequency="hourly",
            confidence_level=1 - self.settings.uncertainty_alpha,
            t_tail=2,
        )

        df_eval["predicted_unc"] = reporting_metrics.predicted_data_point_unc

        return df_eval

    def to_dict(self) -> dict:
        """Returns a dictionary of model parameters.

        Returns:
            Model parameters.
        """
        loc_attr = "mean_" if self.settings.scaling_method == _settings.ScalingChoice.STANDARD_SCALER else "center_"
        feature_loc = getattr(self._feature_scaler, loc_attr)
        feature_scaler = {
            key: [feature_loc[i], self._feature_scaler.scale_[i]]
            for i, key in enumerate(self._ts_features)
        }
        y_scaler = [
            getattr(self._y_scaler, loc_attr).squeeze(),
            self._y_scaler.scale_.squeeze(),
        ]

        # convert self._df_temporal_clusters to list of lists
        df_temporal_clusters = self._df_temporal_clusters.reset_index().values.tolist()

        # Serialize per-hour baseline metrics (None for old models that lack them)
        baseline_hour_metrics = None
        if self.baseline_hour_metrics:
            baseline_hour_metrics = {
                str(k): v for k, v in self.baseline_hour_metrics.items()
                if v is not None
            }

        params = self._base_settings.SerializeModel(
            settings=self.settings,
            temporal_clusters=df_temporal_clusters,
            temperature_bin_edges=self._T_bin_edges,
            temperature_edge_bin_coefficients=self._T_edge_bin_coeffs,
            ts_features=self._ts_features,
            categorical_features=self._categorical_features,
            coefficients=self._model.coef_.tolist(),
            intercept=self._model.intercept_.tolist(),
            feature_scaler=feature_scaler,
            catagorical_scaler=None,
            y_scaler=y_scaler,
            baseline_metrics=self.baseline_metrics,
            baseline_hour_metrics=baseline_hour_metrics,
            info=self._base_settings.ModelInfo(
                disqualification=self.disqualification,
                warnings=self.warnings,

                baseline_timezone=str(self.baseline_timezone),
                version=self.version,
            ),
        )

        return params.model_dump()

    def to_json(self) -> str:
        """Returns a JSON string of model parameters.

        Returns:
            Model parameters.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data) -> HourlyModel:
        """Create a instance of the class from a dictionary (such as one produced from the to_dict method).

        Args:
            data (dict): The dictionary containing the model data.

        Returns:
            An instance of the class.
        """
        # get settings
        train_features = data.get("settings").get("train_features")

        if "ghi" in train_features:
            settings = _settings.HourlySolarSettings(**data.get("settings"))
        else:
            settings = _settings.HourlyNonSolarSettings(**data.get("settings"))

        # initialize model class
        model_cls = cls(settings=settings)

        df_temporal_clusters = pd.DataFrame(
            data.get("temporal_clusters"),
            columns=model_cls._temporal_cluster_cols + ["temporal_cluster"],
        ).set_index(model_cls._temporal_cluster_cols)

        model_cls._df_temporal_clusters = df_temporal_clusters
        model_cls._T_bin_edges = np.array(data.get("temperature_bin_edges"))
        model_cls._T_edge_bin_coeffs = {
            int(k): v for k, v in data.get("temperature_edge_bin_coefficients").items()
        }

        model_cls._ts_features = data.get("ts_features")
        model_cls._categorical_features = data.get("categorical_features")

        # set scalers
        feature_scaler_values = list(data.get("feature_scaler").values())
        feature_scaler_loc = [i[0] for i in feature_scaler_values]
        feature_scaler_scale = [i[1] for i in feature_scaler_values]

        y_scaler_values = data.get("y_scaler")

        loc_attr = "mean_" if settings.scaling_method == _settings.ScalingChoice.STANDARD_SCALER else "center_"

        setattr(model_cls._feature_scaler, loc_attr, np.array(feature_scaler_loc))
        model_cls._feature_scaler.scale_ = np.array(feature_scaler_scale)
        
        setattr(model_cls._y_scaler, loc_attr, np.array(y_scaler_values[0]))
        model_cls._y_scaler.scale_ = np.array(y_scaler_values[1])

        # set model
        model_cls._model.coef_ = np.array(data.get("coefficients"))
        model_cls._model.intercept_ = np.array(data.get("intercept"))

        model_cls._is_fit = True

        # set baseline metrics
        model_cls.baseline_metrics = BaselineMetricsFromDict(
            data.get("baseline_metrics")
        )

        # Per-hour baseline metrics — absent in old model_json.
        # DELETE this fallback once all model_json have been retrained.
        model_cls.baseline_hour_metrics = None
        raw_hour_metrics = data.get("baseline_hour_metrics")
        if raw_hour_metrics is not None:
            model_cls.baseline_hour_metrics = {
                int(k): BaselineMetricsFromDict(v)
                for k, v in raw_hour_metrics.items()
            }

        info = model_cls._base_settings.ModelInfo(**data.get("info"))
        model_cls.warnings = info.warnings
        model_cls.disqualification = info.disqualification
        model_cls.baseline_timezone = info.baseline_timezone
        model_cls.version = info.version

        return model_cls

    @classmethod
    def from_json(cls, str_data) -> HourlyModel:
        """Create an instance of the class from a JSON string.

        Args:
            str_data: The JSON string representing the object.

        Returns:
            An instance of the class.

        """
        return cls.from_dict(json.loads(str_data))

    def plot(
        self,
        df_eval: HourlyBaselineData | HourlyReportingData,
    ):
        """Plot a model fit with baseline or reporting data.

        Args:
            df_eval: The baseline or reporting data object to plot.
        """
        raise NotImplementedError
