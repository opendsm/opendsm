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

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import sklearn

sklearn.set_config(
    assume_finite=True, skip_parameter_validation=True
)  # Faster, we do checking

from timeit import default_timer as timer

from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, RobustScaler

from opendsm.eemeter.models.hourly.model import HourlyModel, AdaptiveElasticNetRegressor

from opendsm.drmeter.models.hourly import settings as _settings
from opendsm.drmeter.models.hourly import HourlyBaselineData, HourlyReportingData
from opendsm import __version__

from opendsm.eemeter.common.warnings import EEMeterWarning


# TODO: need to make explicit solar/nonsolar versions and set settings requirements/defaults appropriately
class DRHourlyModel(HourlyModel):
    """Note:
    Despite the temporal clusters, we can view all models created as a subset of the same full model.
    The temporal clusters would simply have the same coefficients within the same days combinations.
    """

    _temporal_cluster_cols = ["day_of_week"]

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
            if features := settings.get("train_features"):
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
        self._set_model()

        self._T_bin_edges = None
        self._T_edge_bin_coeffs = None
        self._T_edge_bin_rate = None
        self._df_temporal_clusters = None
        self._categorical_features = None
        self._ts_feature_norm = None

        self._ts_features = []
        if self.settings.train_features:
            self._ts_features = self.settings.train_features.copy()

        self.is_fitted = False
        self.baseline_metrics = None
        self.baseline_hour_metrics = None

        self.warnings: list[EEMeterWarning] = []
        self.disqualification: list[EEMeterWarning] = []

        self.baseline_timezone = None
        self.error = dict()
        self.version = __version__

    
    def _set_scalers(self):
        # set scalers
        if self.settings.scaling_method == _settings.ScalingChoice.STANDARD_SCALER:
            self._feature_scaler = StandardScaler()
            self._y_scaler = StandardScaler()
        elif self.settings.scaling_method == _settings.ScalingChoice.ROBUST_SCALER:
            self._feature_scaler = RobustScaler(unit_variance=True)
            self._y_scaler = RobustScaler(unit_variance=True)


    def _set_model(self):
        # set base model
        if self.settings.base_model == _settings.BaseModel.ELASTICNET:
            settings = self.settings.elasticnet
            if settings.alpha <= self._alpha_model_threshold:
                self._model = LinearRegression(
                    fit_intercept=settings.fit_intercept
                )
            else:
                if settings.l1_ratio <= self._l1_ratio_model_threshold:
                    model = Ridge
                elif settings.l1_ratio >= (1 - self._l1_ratio_model_threshold):
                    model = Lasso
                else:
                    model = ElasticNet

                self._model = model(
                    alpha=settings.alpha,
                    fit_intercept=settings.fit_intercept,
                    precompute=settings.precompute,
                    max_iter=settings.max_iter,
                    tol=settings.tol,
                    selection=settings.selection,
                    warm_start=settings.warm_start,
                    random_state=settings._seed,
                )

                if model == ElasticNet:
                    self._model.l1_ratio = settings.l1_ratio

            if self.settings.adaptive_weights.enabled:
                self._model = AdaptiveElasticNetRegressor(self._model, self.settings)
                
        elif self.settings.base_model == _settings.BaseModel.KERNEL_RIDGE:
            settings = self.settings.kernel_ridge
            self._model = KernelRidge(
                alpha=settings.alpha,
                kernel=settings.kernel,
                gamma=settings.gamma,
            )

    def fit(self, baseline_data, ignore_disqualification=False):
        # if not isinstance(baseline_data, HourlyBaselineData):
        #     raise TypeError("baseline_data must be a DailyBaselineData object")
        # TODO check DQ, log warnings
        self._fit(baseline_data)

        return self

    def predict(
        self,
        reporting_data,
        ignore_disqualification=False,
    ):
        """Perform initial sufficiency and typechecks before passing to private predict"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before predictions can be made.")

        # TODO check DQ, log warnings

        # if not isinstance(reporting_data, (HourlyBaselineData, HourlyReportingData)):
        #     raise TypeError(
        #         "reporting_data must be a DailyBaselineData or DailyReportingData object"
        #     )

        df_eval = self._predict(reporting_data)

        return df_eval

    # def _add_temperature_bins(self, df):
    #     # TODO: do we need to do something about empty bins in prediction? I think not but maybe
    #     settings = self.settings.TEMPERATURE_BIN

    #     # add temperature bins based on temperature
    #     if not self.is_fit:
    #         if settings.METHOD == "equal_sample_count":
    #             T_bin_edges = pd.qcut(
    #                 df["temperature"], q=settings.N_BINS, labels=False
    #             )

    #         elif settings.METHOD == "equal_bin_width":
    #             T_bin_edges = pd.cut(
    #                 df["temperature"], bins=settings.N_BINS, labels=False
    #             )

    #         elif settings.METHOD == "set_bin_width":
    #             bin_width = settings.BIN_WIDTH

    #             min_temp = np.floor(df["temperature"].min())
    #             max_temp = np.ceil(df["temperature"].max())

    #             if not settings.INCLUDE_EDGE_BINS:
    #                 step_num = (
    #                     np.round((max_temp - min_temp) / bin_width).astype(int) + 1
    #                 )

    #                 # T_bin_edges = np.arange(min_temp, max_temp + bin_width, bin_width)
    #                 T_bin_edges = np.linspace(min_temp, max_temp, step_num)

    #             else:
    #                 set_edge_bin_width = False
    #                 if set_edge_bin_width:
    #                     edge_bin_width = bin_width * 1 / 2

    #                     bin_range = [
    #                         min_temp + edge_bin_width,
    #                         max_temp - edge_bin_width,
    #                     ]

    #                 else:
    #                     edge_bin_count = int(len(df) * settings.EDGE_BIN_PERCENT)

    #                     # get 5th smallest and 5th largest temperatures
    #                     sorted_temp = np.sort(df["temperature"])
    #                     min_temp_reg_bin = np.ceil(sorted_temp[edge_bin_count])
    #                     max_temp_reg_bin = np.floor(sorted_temp[-edge_bin_count])

    #                     bin_range = [min_temp_reg_bin, max_temp_reg_bin]

    #                 step_num = (
    #                     np.round((bin_range[1] - bin_range[0]) / bin_width).astype(int)
    #                     + 1
    #                 )

    #                 # create bins with set width
    #                 T_bin_edges = np.array(
    #                     [min_temp, *np.linspace(*bin_range, step_num), max_temp]
    #                 )

    #         else:
    #             raise ValueError("Invalid temperature binning method")

    #         # set the first and last bin to -inf and inf
    #         T_bin_edges[0] = -np.inf
    #         T_bin_edges[-1] = np.inf

    #         # store bin edges for prediction
    #         self._T_bin_edges = T_bin_edges

    #     T_bins = pd.cut(df["temperature"], bins=self._T_bin_edges, labels=False)

    #     df["temp_bin"] = T_bins

    #     # Create dummy variables for temperature bins
    #     bin_dummies = pd.get_dummies(
    #         pd.Categorical(
    #             df["temp_bin"], categories=range(len(self._T_bin_edges) - 1)
    #         ),
    #         prefix="temp_bin",
    #     )
    #     bin_dummies.index = df.index

    #     col_names = bin_dummies.columns.tolist()
    #     df = pd.merge(df, bin_dummies, how="left", left_index=True, right_index=True)

    #     return df, col_names