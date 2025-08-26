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

import json

import numpy as np
import pandas as pd

from opendsm.common.utils import t_stat
from opendsm.eemeter.common.features import (
    estimate_hour_of_week_occupancy,
    fit_temperature_bins,
)
from opendsm.eemeter.models.hourly_caltrack.design_matrices import (
    create_caltrack_hourly_preliminary_design_matrix,
    create_caltrack_hourly_segmented_design_matrices,
)
from opendsm.eemeter.models.hourly_caltrack.model import (
    CalTRACKHourlyModelResults,
    fit_caltrack_hourly_model,
)
from opendsm.eemeter.models.hourly_caltrack.segmentation import segment_time_series

month_dict = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class IntermediateModelVariables:
    preliminary_design_matrix = None
    segmentation = None
    occupancy_lookup = None
    occupied_temperature_bins = None
    unoccupied_temperature_bins = None
    segmented_design_matrices = None


class HourlyModel:
    def __init__(self, settings=None):
        self.segment_type = "three_month_weighted"
        self.alpha = 0.1

    def fit(self, data):
        meter_data = data.df["observed"].to_frame("value")
        temperature_data = data.df["temperature"]

        self.model_process_variables = IntermediateModelVariables()

        # preliminary design matrix
        preliminary_design_matrix = create_caltrack_hourly_preliminary_design_matrix(
            meter_data, temperature_data
        )
        self.model_process_variables.preliminary_design_matrix = (
            preliminary_design_matrix
        )

        # segment time series
        segmentation = segment_time_series(
            preliminary_design_matrix.index, self.segment_type
        )
        self.model_process_variables.segmentation = segmentation

        # estimate occupancy
        occupancy_lookup = estimate_hour_of_week_occupancy(
            preliminary_design_matrix, segmentation=segmentation
        )
        self.model_process_variables.occupancy_lookup = occupancy_lookup

        # fit temperature bins
        (occupied_t_bins, unoccupied_t_bins) = fit_temperature_bins(
            preliminary_design_matrix,
            segmentation=segmentation,
            occupancy_lookup=occupancy_lookup,
        )
        self.model_process_variables.occupied_temperature_bins = occupied_t_bins
        self.model_process_variables.unoccupied_temperature_bins = unoccupied_t_bins

        # create segmented design matrices
        segmented_design_matrices = create_caltrack_hourly_segmented_design_matrices(
            preliminary_design_matrix,
            segmentation,
            occupancy_lookup,
            occupied_t_bins,
            unoccupied_t_bins,
        )
        self.model_process_variables.segmented_design_matrices = (
            segmented_design_matrices
        )

        # fit model
        self.model = fit_caltrack_hourly_model(
            segmented_design_matrices,
            occupancy_lookup,
            occupied_t_bins,
            unoccupied_t_bins,
            self.segment_type,
        )
        self.is_fit = True
        self.model_metrics = self.model.totals_metrics

        # calculate baseline residuals
        prediction = self.model.predict(temperature_data.index, temperature_data)
        meter_data = meter_data.merge(
            prediction.result, left_index=True, right_index=True
        )
        meter_data.dropna(inplace=True)
        meter_data["resid"] = meter_data["value"] - meter_data["predicted_usage"]

        # get uncertainty variables
        self._autocorr_unc_vars = {}
        if list(self.model_metrics.keys()) == ["all"]:
            self._autocorr_unc_vars["all"] = {
                "mean_baseline_usage": np.mean(meter_data["value"]),
                "n": self.model_metrics["all"].observed_length,
                "n_prime": self.model_metrics["all"].n_prime,
                "MSE": np.mean(meter_data["resid"] ** 2),
            }
        else:
            # monthly segment model
            model_month_dict = {
                k.replace("-weighted", "").split("-")[1]: k
                for k in self.model_metrics.keys()
            }
            meter_data["month"] = meter_data.index.month

            for month_abbr, model_key in model_month_dict.items():
                month_n = month_dict[month_abbr]
                month_data = meter_data[meter_data["month"] == month_n]

                self._autocorr_unc_vars[month_n] = {
                    "mean_baseline_usage": np.mean(month_data["value"]),
                    "n": self.model_metrics[model_key].observed_length,
                    "n_prime": self.model_metrics[model_key].n_prime,
                    "MSE": np.mean(month_data["resid"] ** 2),
                }

        return self

    def predict(self, reporting_data):
        if not self.is_fit:
            raise RuntimeError("Model must be fit before predictions can be made.")
        prediction_index = reporting_data.df.index
        temperature_series = reporting_data.df["temperature"]
        model_prediction = self.model.predict(prediction_index, temperature_series)

        df_res = pd.concat([reporting_data.df, model_prediction.result], axis=1)
        df_res = df_res[["temperature", "observed", "predicted_usage"]]
        df_res = df_res.rename(columns={"predicted_usage": "predicted"})
        df_res["predicted_uncertainty"] = np.nan

        # if observed isn't all nan, calculate uncertainty
        if not df_res["observed"].isna().all():
            for month_n, unc_vars in self._autocorr_unc_vars.items():
                if month_n == "all":
                    idx = df_res.index
                else:
                    idx = df_res.index[df_res.index.month == month_n]

                mean_baseline_usage = unc_vars["mean_baseline_usage"]
                n = unc_vars["n"]
                n_prime = unc_vars["n_prime"]
                mse = unc_vars["MSE"]

                reporting_usage = np.sum(df_res.loc[idx]["observed"])
                m = len(idx)
                t = t_stat(self.alpha, m, tail=2)

                # ASHRAE 14
                total_unc = (
                    1.26
                    * t
                    * reporting_usage
                    / (m * mean_baseline_usage)
                    * np.sqrt(mse * n / n_prime * (1 + 2 / n_prime) * m)
                )

                avg_unc = np.sqrt(total_unc**2 / m)
                df_res.loc[idx, "predicted_uncertainty"] = avg_unc

        return df_res

    def to_dict(self):
        model_dict = self.model.json()
        model_dict["model"]["unc_vars"] = self._autocorr_unc_vars
        return model_dict

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data):
        hourly_model = cls()
        hourly_model.model = CalTRACKHourlyModelResults.from_json(data)
        hourly_model._autocorr_unc_vars = data["model"]["unc_vars"]
        hourly_model.is_fit = True
        return hourly_model

    @classmethod
    def from_json(cls, str_data):
        return cls.from_dict(json.loads(str_data))

    @classmethod
    def from_2_0_dict(cls, data):
        """fill default metrics and uncertainty variables to allow deserializing legacy models with new wrapper"""
        monthly_unc_vars = {"mean_baseline_usage": 0, "n": 0, "n_prime": 1, "MSE": 0}
        model_dict = dict(data)
        model_dict["model"]["unc_vars"] = {
            str(month): monthly_unc_vars for month in range(1, 13)
        }
        return cls.from_dict(model_dict)

    @classmethod
    def from_2_0_json(cls, str_data):
        return cls.from_2_0_dict(json.loads(str_data))

    def plot(
        self,
        ax=None,
        title=None,
        figsize=None,
        temp_range=None,
    ):
        raise NotImplementedError
