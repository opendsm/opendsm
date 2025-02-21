#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2025 OpenDSM contributors

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

import itertools
import json
from typing import Union

import numpy as np
import pandas as pd

from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)
from opendsm.eemeter.common.warnings import EEMeterWarning
from opendsm.eemeter.models.daily.base_models.full_model import (
    full_model,
    get_full_model_x,
)
from opendsm.eemeter.models.daily.data import DailyBaselineData, DailyReportingData
from opendsm.eemeter.models.daily.fit_base_models import (
    fit_final_model,
    fit_initial_models_from_full_model,
)
from opendsm.eemeter.models.daily.parameters import (
    DailyModelParameters,
    DailySubmodelParameters,
)
from opendsm.eemeter.models.daily.utilities.base_model import get_smooth_coeffs
from opendsm.eemeter.models.daily.utilities.settings import (
    DailySettings,
    DailyLegacySettings,
    update_daily_settings,
)
from opendsm.eemeter.models.daily.utilities.ellipsoid_test import ellipsoid_split_filter
from opendsm.eemeter.models.daily.utilities.selection_criteria import selection_criteria


class DailyModel:
    """
    A class to fit a model to the input meter data.

    Attributes:
        settings (dict): A dictionary of settings.
        seasonal_options (list): A list of seasonal options (su: Summer, sh: Shoulder, wi: Winter).
            Elements in the list are seasons separated by '_' that represent a model split.
            For example, a list of ['su_sh', 'wi'] represents two splits: summer/shoulder and winter.
        day_options (list): A list of day options.
        combo_dictionary (dict): A dictionary of combinations.
        df_meter (pandas.DataFrame): A dataframe of meter data.
        error (dict): A dictionary of error metrics.
        combinations (list): A list of combinations.
        components (list): A list of components.
        fit_components (list): A list of fit components.
        wRMSE_base (float): The mean bias error for no splits.
        best_combination (list): The best combination of splits.
        model (sklearn.pipeline.Pipeline): The final fitted model.
        id (str): The index of the meter data.
    """

    _baseline_data_type = DailyBaselineData
    _reporting_data_type = DailyReportingData
    _data_df_name = "df"

    def __init__(
        self,
        model: str = "current",
        settings: dict | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            model: The model to use (either 'current' or 'legacy').
            settings: DailySettings to be changed.
            verbose: Whether to print verbose output.
        """

        # Initialize settings
        self._initialize_settings(model, settings)

        # Initialize seasons and weekday/weekend
        self.seasonal_options = [
            ["su_sh_wi"],
            ["su", "sh_wi"],
            ["su_sh", "wi"],
            ["su_wi", "sh"],
            ["su", "sh", "wi"],
        ]
        self.day_options = [["wd", "we"]]

        # make dictionary is_weekday from settings
        day_dict = self.settings.weekday_weekend._num_dict
        n_week = list(range(len(day_dict)))
        self.combo_dictionary = {
            "su": "summer",
            "sh": "shoulder",
            "wi": "winter",
            "fw": [n + 1 for n in n_week],
            "wd": [n + 1 for n in n_week if day_dict[n+1] == "weekday"],
            "we": [n + 1 for n in n_week if day_dict[n+1] == "weekend"],
        }
        self.verbose = verbose

        self.error = {
            "wRMSE": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "CVRMSE": np.nan,
            "PNRMSE": np.nan,
        }

    def _initialize_settings(
        self,
        model: str = "current",
        settings: dict | None = None
    ) -> None:

        # Note: Model designates the base settings, it can be 'current' or 'legacy'
        #       Settings is to be a dictionary of settings to be changed

        if settings is None:
            settings = {}

        if model.replace(" ", "").replace("_", ".").lower() in ["current", "default"]:
            self.settings = DailySettings(**settings)
        elif model.replace(" ", "").replace("_", ".").lower() in ["legacy"]:
            self.settings = DailyLegacySettings(**settings)
        else:
            raise Exception(
                "Invalid 'settings' choice: must be 'current', 'default', or 'legacy'"
            )

    def fit(
        self, 
        baseline_data: DailyBaselineData, 
        ignore_disqualification: bool = False
    ) -> DailyModel:
        """Fit the model using baseline data.

        Args:
            baseline_data: DailyBaselineData object.
            ignore_disqualification: Whether to ignore disqualification errors / warnings.

        Returns:
            The fitted model.

        Raises:
            TypeError: If baseline_data is not a DailyBaselineData object.
            DataSufficiencyError: If the model can't be fit on disqualified baseline data.
        """
        if not isinstance(baseline_data, self._baseline_data_type):
            raise TypeError(f"baseline_data must be a {self._baseline_data_type.__name__} object")
        baseline_data.log_warnings()
        if baseline_data.disqualification and not ignore_disqualification:
            raise DataSufficiencyError("Can't fit model on disqualified baseline data")
        self.baseline_timezone = baseline_data.tz
        self.warnings = baseline_data.warnings
        self.disqualification = baseline_data.disqualification
        df = getattr(baseline_data, self._data_df_name)
        self._fit(df)
        if not self._model_fit_is_acceptable():
            cvrmse_warning = EEMeterWarning(
                qualified_name="eemeter.model_fit_metrics",
                description="Model disqualified due to poor fit.",
                data={
                    "cvrmse_threshold": self.settings.cvrmse_threshold,
                    "cvrmse": self.error["CVRMSE"],
                    "pnrmse_threshold": self.settings.pnrmse_threshold,
                    "pnrmse": self.error["PNRMSE"],
                },
            )
            cvrmse_warning.warn()
            self.disqualification.append(cvrmse_warning)
        return self

    def _fit(self, meter_data):
        # Initialize dataframe
        self.df_meter, _ = self._initialize_data(meter_data)

        # Begin fitting
        self.combinations = self._combinations()
        self.components = self._components()
        self.fit_components = self._fit_components()

        # calculate mean bias error for no splits
        self.wRMSE_base = self._get_error_metrics("fw-su_sh_wi")[0]

        # find best combination
        self.best_combination = self._best_combination(print_out=False)
        self.model = self._final_fit(self.best_combination)

        self.id = meter_data.index.unique()[0]

        wRMSE, RMSE, MAE, CVRMSE, PNRMSE = self._get_error_metrics(
            self.best_combination
        )
        self.error["wRMSE"] = float(wRMSE)
        self.error["RMSE"] = float(RMSE)
        self.error["MAE"] = float(MAE)
        self.error["CVRMSE"] = float(CVRMSE)
        self.error["PNRMSE"] = float(PNRMSE)

        self.params = self._create_params_from_fit_model()
        self.is_fitted = True
        return self

    def predict(
        self,
        reporting_data: DailyBaselineData | DailyReportingData,
        ignore_disqualification=False,
    ) -> pd.DataFrame:
        """Predicts the energy consumption using the fitted model.

        Args:
            reporting_data (Union[DailyBaselineData, DailyReportingData]): The data used for prediction.
            ignore_disqualification (bool, optional): Whether to ignore model disqualification. Defaults to False.

        Returns:
            Dataframe with input data along with predicted energy consumption.

        Raises:
            RuntimeError: If the model is not fitted.
            DisqualifiedModelError: If the model is disqualified and ignore_disqualification is False.
            ValueError: If the reporting data has a different timezone than the model.
            TypeError: If the reporting data is not of type DailyBaselineData or DailyReportingData.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before predictions can be made.")

        if self.disqualification and not ignore_disqualification:
            raise DisqualifiedModelError(
                "Attempting to predict using disqualified model without setting ignore_disqualification=True"
            )

        if str(self.baseline_timezone) != str(reporting_data.tz):
            """would be preferable to directly compare, but
            * using str() helps accomodate mixed tzinfo implementations,
            * the likelihood of sub-hour offset inconsistencies being relevant to the daily model is low
            """
            raise ValueError(
                "Reporting data must use the same timezone that the model was initially fit on."
            )

        if not isinstance(reporting_data, (self._baseline_data_type, self._reporting_data_type)):
            raise TypeError(
                f"reporting_data must be a {self._baseline_data_type.__name__} or {self._reporting_data_type.__name__} object"
            )

        df = getattr(reporting_data, self._data_df_name)
        df_res = self._predict(df)

        return df_res

    def _predict(self, df_eval, mask_observed_with_missing_temperature=True):
        """
        Makes model prediction on given temperature data.

        Parameters:
            df_eval (pandas.DataFrame): The evaluation dataframe.

        Returns:
            pandas.DataFrame: The evaluation dataframe with model predictions added.
        """
        # TODO decide whether to allow temperature series vs requiring "design matrix"
        if isinstance(df_eval, pd.Series):
            df_eval = df_eval.to_frame("temperature")

        # initialize data to input dataframe
        df_eval, dropped_rows = self._initialize_data(df_eval)

        df_all_models = []
        for component_key in self.params.submodels.keys():
            eval_segment = self._meter_segment(component_key, df_eval)
            T = eval_segment["temperature"].values

            # model, unc, hdd_load, cdd_load = self.model[component_key].eval(T)
            model, unc, hdd_load, cdd_load = self._predict_submodel(
                self.params.submodels[component_key], T
            )

            df_model = pd.DataFrame(
                data={
                    "predicted": model,
                    "predicted_unc": unc,
                    "heating_load": hdd_load,
                    "cooling_load": cdd_load,
                },
                index=eval_segment.index,
            )
            df_model["model_split"] = component_key
            df_model["model_type"] = self.params.submodels[
                component_key
            ].model_type.value

            df_all_models.append(df_model)

        df_model_prediction = pd.concat(df_all_models, axis=0)
        df_eval = df_eval.join(df_model_prediction)

        # 3.5.1.1. If a day is missing a temperature value, the corresponding consumption value for that day should be masked.
        if mask_observed_with_missing_temperature:
            dropped_rows[dropped_rows["temperature"].isna()]["observed"] = np.nan

        df_eval = pd.concat([df_eval, dropped_rows])

        return df_eval.sort_index()

    def _model_fit_is_acceptable(self):
        cvrmse = self.error["CVRMSE"]
        pnrmse = self.error["PNRMSE"]

        # sufficient is (0 <= cvrmse <= threshold) or (0 <= pnrmse <= threshold)

        if cvrmse is not None:
            if (0 <= cvrmse) and (cvrmse <= self.settings.cvrmse_threshold):
                return True
            
        if pnrmse is not None:
            # less than 0 is not possible, but just in case
            if (0 <= pnrmse) and (pnrmse <= self.settings.pnrmse_threshold):
                return True

        return False

    def to_dict(self) -> dict:
        """Returns a dictionary of model parameters.

        Returns:
            Model parameters.
        """
        return self.params.model_dump()

    def to_json(self) -> str:
        """Returns a JSON string of model parameters.

        Returns:
            Model parameters.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data) -> DailyModel:
        """Create a instance of the class from a dictionary (such as one produced from the to_dict method).

        Args:
            data (dict): The dictionary containing the model data.

        Returns:
            An instance of the class.

        """
        settings = data.get("settings")
        daily_model = cls(settings=settings)
        info = data.get("info")
        daily_model.params = DailyModelParameters(
            submodels=data.get("submodels"),
            info=info,
            settings=settings,
        )

        def deserialize_warnings(warnings):
            if not warnings:
                return []
            warn_list = []
            for warning in warnings:
                warn_list.append(
                    EEMeterWarning(
                        qualified_name=warning.get("qualified_name"),
                        description=warning.get("description"),
                        data=warning.get("data"),
                    )
                )
            return warn_list

        daily_model.disqualification = deserialize_warnings(
            info.get("disqualification")
        )
        daily_model.warnings = deserialize_warnings(info.get("warnings"))
        daily_model.baseline_timezone = info.get("baseline_timezone")
        daily_model.is_fitted = True
        return daily_model

    @classmethod
    def from_json(cls, str_data: str) -> DailyModel:
        """Create an instance of the class from a JSON string.

        Args:
            str_data: The JSON string representing the object.

        Returns:
            An instance of the class.

        """
        return cls.from_dict(json.loads(str_data))

    @classmethod
    def from_2_0_dict(cls, data) -> DailyModel:
        """Create an instance of the class from a legacy (2.0) model dictionary.

        Args:
            data (dict): A dictionary containing the necessary data (legacy 2.0) to create a DailyModel instance.

        Returns:
            An instance of the class.

        """
        daily_model = cls(model="legacy")
        daily_model.params = DailyModelParameters.from_2_0_params(data)
        daily_model.warnings = []
        daily_model.disqualification = []
        daily_model.baseline_timezone = "UTC"
        daily_model.is_fitted = True
        return daily_model

    @classmethod
    def from_2_0_json(cls, str_data: str) -> DailyModel:
        """Create an instance of the class from a legacy (2.0) JSON string.

        Args:
            str_data: The JSON string.

        Returns:
            An instance of the class.

        """
        return cls.from_2_0_dict(json.loads(str_data))

    def plot(
        self,
        data: DailyBaselineData | DailyReportingData,
    ) -> None:
        """Plot a model fit with baseline or reporting data. Requires matplotlib to use.

        Args:
            df_eval: The baseline or reporting data object to plot.
        """
        try:
            from opendsm.eemeter.models.daily.plot import plot
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib is required for plotting.")

        # TODO: pass more kwargs to plotting function

        plot(self, self._predict(data.df))

    def _create_params_from_fit_model(self):
        submodels = {}
        for key, submodel in self.model.items():
            temperature_constraints = {
                "T_min": submodel.T_min,
                "T_max": submodel.T_max,
                "T_min_seg": submodel.T_min_seg,
                "T_max_seg": submodel.T_max_seg,
            }
            submodels[key] = DailySubmodelParameters(
                coefficients=submodel.named_coeffs,
                temperature_constraints=temperature_constraints,
                f_unc=submodel.f_unc,
            )
        params = DailyModelParameters(
            submodels=submodels,
            settings=self.settings.model_dump(),
            info={
                "error": self.error,
                "baseline_timezone": str(self.baseline_timezone),
                "disqualification": [dq.json() for dq in self.disqualification],
                "warnings": [warning.json() for warning in self.warnings],
            },
        )

        return params

    def _initialize_data(self, meter_data):
        """
        Initializes the meter data by performing the following operations:
        - Renames the 'model' column to 'model_old' if it exists
        - Converts the index to a DatetimeIndex if it is not already
        - Adds a 'season' column based on the month of the index using the settings.season dictionary
        - Adds a 'day_of_week' column based on the day of the week of the index
        - Removes any rows with NaN values in the 'temperature' or 'observed' columns
        - Sorts the data by the index
        - Reorders the columns to have 'season' and 'day_of_week' first, followed by the remaining columns

        Parameters:
        - meter_data: A pandas DataFrame containing the meter data

        Returns:
        - A pandas DataFrame containing the initialized meter data
        - A pandas DataFrame containing rows which were dropped due to NaN in either column
        """

        if "predicted" in meter_data.columns:
            meter_data = meter_data.rename(columns={"predicted": "predicted_old"})

        cols = list(meter_data.columns)

        if "datetime" in cols:
            meter_data.set_index("datetime", inplace=True)
            cols.remove("datetime")

        if not isinstance(meter_data.index, pd.DatetimeIndex):
            try:
                meter_data.index = pd.to_datetime(meter_data.index)
            except:
                raise TypeError("Could not convert 'meter_data.index' to datetime")

        for col in ["season", "day_of_week"]:
            if col in cols:
                meter_data.drop([col], axis=1, inplace=True)
                cols.remove(col)

        meter_data["season"] = meter_data.index.month.map(self.settings.season._num_dict)
        meter_data["day_of_week"] = meter_data.index.dayofweek + 1
        meter_data = meter_data.sort_index()
        meter_data = meter_data[["season", "day_of_week", *cols]]

        dropped_rows = meter_data.copy()
        meter_data = meter_data.dropna()
        if meter_data.empty:
            # return early to avoid np.isfinite exception
            return meter_data, dropped_rows
        meter_data = meter_data[np.isfinite(meter_data["temperature"])]
        if "observed" in cols:
            meter_data = meter_data[np.isfinite(meter_data["observed"])]

        dropped_rows = dropped_rows.loc[~dropped_rows.index.isin(meter_data.index)]
        return meter_data, dropped_rows

    def _combinations(self):
        """
        This method generates all possible combinations of seasonal and day options for the given data.
        It then trims the combinations based on certain conditions such as minimum number of days per season,
        and whether to allow separate splits for summer, shoulder and winter seasons.
        """

        settings = self.settings

        def _get_combinations():
            def add_prefix(list_str, prefix):
                return [f"{prefix}-{s}" for s in list_str]

            def expand_combinations(combos_in):
                """
                Given a list of combinations, expands each combination by adding a new item to it.
                The new item is chosen from the intersection of the items in two specific combinations.
                The new item is then added to a third combination, which is created by combining the remaining items from the two specific combinations.
                The resulting expanded combinations are returned as a list.

                Parameters:
                combos_in (list): A list of combinations, where each combination is a list of items.

                Returns:
                list: A list of expanded combinations, where each expanded combination is a list of items.
                """

                combo_expanded = []
                for combo in combos_in:
                    combo_expanded.append(list(combo))
                    prefixes = [item[0] for item in combo]

                    if "wd" in prefixes and "we" in prefixes:
                        i_wd = prefixes.index("wd")
                        i_we = prefixes.index("we")
                    else:
                        continue

                    if "fw" in prefixes:
                        i_fw = prefixes.index("fw")
                    else:
                        i_fw = None

                    for item in combo[i_wd][1]:
                        if item in combo[i_we][1]:
                            combo_0_trim = [x for x in combo[i_wd][1] if x != item]
                            combo_1_trim = [x for x in combo[i_we][1] if x != item]

                            if i_fw is None:
                                fw_item = ["fw", [item]]
                            else:
                                fw_item = ["fw", [*combo[i_fw][1], item]]

                            if len(combo_0_trim) == 0 and len(combo_1_trim) == 0:
                                combo_new = [fw_item]
                            elif len(combo_0_trim) > 0 and len(combo_1_trim) == 0:
                                combo_new = [fw_item, [combo[i_wd][0], combo_0_trim]]
                            elif len(combo_0_trim) == 0 and len(combo_1_trim) > 0:
                                combo_new = [fw_item, [combo[i_we][0], combo_1_trim]]
                            else:
                                combo_new = [
                                    fw_item,
                                    [combo[i_wd][0], combo_0_trim],
                                    [combo[i_we][0], combo_1_trim],
                                ]

                            combo_expanded.append(combo_new)

                return combo_expanded

            def stringify(combos):
                """
                Converts a list of tuples into a list of strings, where each string is a combination of the tuple values
                separated by '__'. The tuples are expected to have a prefix and a value, and the prefix is used to add context
                to the value.

                Parameters:
                    combos (list): A list of tuples, where each tuple contains a prefix and a value.

                Returns:
                    list: A list of strings, where each string is a combination of the tuple values separated by '__'.
                """

                combos_str = []
                for combo in combos:
                    combo = [add_prefix(item[1], item[0]) for item in combo]
                    combo = [item for sublist in combo for item in sublist]
                    combo = "__".join(combo)

                    combos_str.append(combo)

                combos_str = sorted(list(set(combos_str)), key=lambda x: (len(x), x))

                return combos_str

            for days in self.day_options:
                season_day_combo = []
                for day in days:
                    season_day_combo.append(
                        list(itertools.product([day], self.seasonal_options))
                    )

                combos_expanded = list(itertools.product(*season_day_combo))
                for _ in range(max([len(item) for item in self.seasonal_options])):
                    combos_expanded = expand_combinations(combos_expanded)

                combos_str = stringify(combos_expanded)

            return combos_str

        def _trim_combinations(combo_list, split_min_days=30):
            """
            Trims the list of combinations to be tested based on various conditions.
            - Checks if the ellipsoids created are separated enough to warrant separate seasons and weekday/weekend splits.
            - Checks if there are enough days in each season and weekday/weekend to warrant separate splits.

            Args:
                combo_list (list): List of combinations to be tested.
                split_min_days (int, optional): Minimum number of days required for a split. Defaults to 30.

            Returns:
                list: Trimmed list of combinations to be tested.
            """

            meter = self.df_meter
            allow_sep_summer = settings.split_selection.allow_separate_summer
            allow_sep_shoulder = settings.split_selection.allow_separate_shoulder
            allow_sep_winter = settings.split_selection.allow_separate_winter
            allow_sep_weekday_weekend = settings.split_selection.allow_separate_weekday_weekend

            if settings.split_selection.reduce_splits_by_gaussian:
                allow_split = ellipsoid_split_filter(
                    self.df_meter, n_std=settings.split_selection.reduce_splits_num_std
                )

                if allow_sep_summer and not allow_split["summer"]:
                    allow_sep_summer = False

                if allow_sep_shoulder and not allow_split["shoulder"]:
                    allow_sep_shoulder = False

                if allow_sep_winter and not allow_split["winter"]:
                    allow_sep_winter = False

                if allow_sep_weekday_weekend and not allow_split["weekday_weekend"]:
                    allow_sep_weekday_weekend = False

            we_days = self.combo_dictionary["we"]

            if (meter["season"].values == "summer").sum() < split_min_days:
                allow_sep_summer = False

            if (meter["season"].values == "shoulder").sum() < split_min_days:
                allow_sep_shoulder = False

            if (meter["season"].values == "winter").sum() < split_min_days:
                allow_sep_winter = False

            combo_list_trimmed = []
            for combo in combo_list:
                if "fw-su_sh_wi" == combo:  # always fit the full model with all data
                    combo_list_trimmed.append(combo)
                    continue
                elif "wd" in combo and not allow_sep_weekday_weekend:
                    continue

                banned_season_split = {
                    "su": not allow_sep_summer,
                    "sh": not allow_sep_shoulder,
                    "wi": not allow_sep_winter,
                }

                valid_combo = True
                components = [item[3:] for item in combo.split("__")]
                for component in components:
                    seasons = component.split("_")

                    if (len(seasons) == 1) and banned_season_split[component]:
                        valid_combo = False
                        break

                    we_count = 0
                    for season in seasons:
                        we_count += (
                            (meter["season"].values == self.combo_dictionary[season])
                            & meter["day_of_week"].isin(we_days).values
                        ).sum()

                    if we_count < split_min_days / 3.75:
                        valid_combo = False
                        break

                if valid_combo:
                    combo_list_trimmed.append(combo)

            return combo_list_trimmed

        def _remove_duplicate_permutations(combo_list):
            """
            Removes duplicate permutations from a list of strings.

            Args:
                combo_list (list): A list of strings representing permutations.

            Returns:
                list: A list of unique permutations.
            """

            unique_sorted_combos = []
            unique_combos = []
            for combo in combo_list:
                sorted_combo = "__".join(sorted(combo.split("__")))

                if sorted_combo not in unique_sorted_combos:
                    unique_sorted_combos.append(combo)
                    unique_combos.append(combo)

            return unique_combos

        combo_list = _get_combinations()
        combo_list = _remove_duplicate_permutations(combo_list)
        combo_list = _trim_combinations(combo_list)

        return combo_list

    def _meter_segment(self, component, meter=None):
        """
        Returns a meter segment based on the given component and meter data.

        Parameters:
            component (str): A string representing the component to filter the meter data by.
            meter (pandas.DataFrame, optional): A pandas DataFrame containing the meter data. Defaults to None.

        Returns:
            pandas.DataFrame: A pandas DataFrame containing the meter data filtered by the given component.
        """

        if meter is None:
            meter = self.df_meter

        season_list = component[3:].split("_")
        day_list = component[:2]

        seasons = [self.combo_dictionary[key] for key in season_list]
        days = self.combo_dictionary[day_list]

        meter_segment = meter[
            meter["season"].isin(seasons) & meter["day_of_week"].isin(days)
        ]

        return meter_segment

    def _components(self):
        """
        Returns a sorted list of unique components from the combinations attribute.
        """

        components = list(
            set([i for item in self.combinations for i in item.split("__")])
        )
        components = sorted(components, key=lambda x: (len(x), x))

        return components

    def _fit_components(self):
        """
        Fits initial models for each component using the meter segment data and component settings.

        If the alpha_final_type is "last", the settings are updated to disable the final bounds scalar and set alpha_final_type to None.

        Returns:
            dict: A dictionary containing the fitted components.
        """

        if self.settings.alpha_final_type == "last":
            settings_update = {
                "DEVELOPER_MODE": True,
                "SILENT_DEVELOPER_MODE": True, 
                "ALPHA_FINAL_TYPE": None,
                "FINAL_BOUNDS_SCALAR": None,
            }

            self.component_settings = update_daily_settings(
                self.settings, settings_update
            )
        else:
            self.component_settings = self.settings

        fit_components = {item: None for item in self.components}
        for component in fit_components.keys():
            meter_segment = self._meter_segment(component)

            # Fit new models
            fit_components[component] = fit_initial_models_from_full_model(
                meter_segment, self.component_settings, print_res=False
            )

        return fit_components

    def _combination_selection_criteria(self, combination):
        """
        Calculates the selection criteria for a given combination of components.

        Parameters:
            combination (str): A string representing the combination of components.

        Returns:
            float: The selection criteria for the given combination.
        """

        components = combination.split("__")

        N = np.sum([self.fit_components[X].N for X in components])
        TSS = np.sum([self.fit_components[X].TSS for X in components])
        # starts as added penalties based on # splits
        # num_coeffs = 3*self.df_penalties[combination] # + np.sum([self.fit_components[X].num_coeffs for X in components])
        num_coeffs = len(components)

        if combination == "fw-su_sh_wi":
            wRMSE = self.wRMSE_base
        else:
            wRMSE = self._get_error_metrics(combination)[0]

        loss = wRMSE / self.wRMSE_base

        criteria_type = self.settings.split_selection.criteria.lower()
        penalty_multiplier = self.settings.split_selection.penalty_multiplier
        penalty_power = self.settings.split_selection.penalty_power

        criteria = selection_criteria(
            loss, TSS, N, num_coeffs, criteria_type, penalty_multiplier, penalty_power
        )

        return criteria

    def _best_combination(self, print_out=False):
        """
        Finds the best combination of parameters based on the selection criteria.

        Parameters:
            print_out (bool): Whether to print the combination and selection criteria for each iteration.

        Returns:
            str: The best combination of parameters as a string.
        """

        HoF = {"combination_str": None, "selection_criteria": np.inf}
        for combo in self.combinations:
            selection_criteria = self._combination_selection_criteria(combo)

            if selection_criteria < HoF["selection_criteria"]:
                HoF["combination_str"] = combo
                HoF["selection_criteria"] = selection_criteria

            if print_out:
                print(f"{combo:>40s} {selection_criteria:>8.1f}")

        if print_out:
            print(f"{HoF['combination_str']:>40s} {HoF['selection_criteria']:>8.1f}")

        return HoF["combination_str"]

    def _final_fit(self, combination):
        """
        Fits the final model for a given combination of components.

        Parameters:
            combination (str): A string representing the combination of components.

        Returns:
            dict: A dictionary containing the fitted models for each component in the combination.
        """

        model = {}
        for component in combination.split("__"):
            settings = self.settings
            prior_model = self.fit_components[component]

            if settings.alpha_final_type is None:
                if self.verbose:
                    print(f"{component}__{prior_model.model_name}")

                model[component] = prior_model
                continue

            settings_update = {
                "DEVELOPER_MODE": True, 
                "SILENT_DEVELOPER_MODE": True, 
                "REGULARIZATION_ALPHA": 0.0
            }
            settings = update_daily_settings(self.settings, settings_update)

            # separate meter appropriately
            meter_segment = self._meter_segment(component)

            # Fit new models
            if self.verbose:
                print(f"{component}__{prior_model.model_name}")

            model[component] = fit_final_model(
                meter_segment, prior_model, settings, print_res=self.verbose
            )

            model[component].settings = self.settings  # overwrite to input settings

        return model

    def _get_error_metrics(self, combination):
        """
        Calculates the error metrics for a given combination of components.
        RMSE and MAE are calculated as the mean of the residuals, wRMSE is calculated as the weighted mean of the residuals.

        Parameters:
            combination (str): A string representing the combination of components to calculate error metrics for.
                If None, the best combination will be used.

        Returns:
            tuple: A tuple containing the calculated error metrics (wRMSE, RMSE, MAE).
        """

        if combination is None:
            combination = self.best_combination

        N = 0
        wSSE = 0
        resid = []
        obs = []
        for component in combination.split("__"):
            fit_component = self.fit_components[component]

            wSSE += fit_component.wSSE
            N += fit_component.N
            resid.append(fit_component.resid)
            obs.append(fit_component.obs)

        resid = np.hstack(resid)
        obs = np.hstack(obs)

        wRMSE = np.sqrt(wSSE / N)
        RMSE = np.mean(resid**2) ** 0.5
        MAE = np.mean(np.abs(resid))
        CVRMSE = RMSE / np.mean(obs)
        PNRMSE = RMSE / np.diff(np.quantile(obs, [0.05, 0.95]))[0]

        return wRMSE, RMSE, MAE, CVRMSE, PNRMSE

    def _predict_submodel(self, submodel, T):
        """
        Predicts submodel output for a given set of temperatures.

        Parameters:
            T (numpy.ndarray): Array of temperatures.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                Tuple containing the following arrays:
                - model: Array of model values.
                - f_unc: Array of uncertainties.
                - hdd_load: Array of heating degree day loads.
                - cdd_load: Array of cooling degree day loads.
        """

        T_min = submodel.temperature_constraints["T_min"]
        T_max = submodel.temperature_constraints["T_max"]
        x = get_full_model_x(
            submodel.coefficients.model_key,
            submodel.coefficients.to_np_array(),
            T_min,
            T_max,
            submodel.temperature_constraints["T_min_seg"],
            submodel.temperature_constraints["T_max_seg"],
        )

        if submodel.coefficients.model_key == "hdd_tidd_cdd_smooth":
            [hdd_bp, hdd_beta, pct_hdd_k, cdd_bp, cdd_beta, pct_cdd_k, intercept] = x
            [hdd_bp, hdd_k, cdd_bp, cdd_k] = get_smooth_coeffs(
                hdd_bp, pct_hdd_k, cdd_bp, pct_cdd_k
            )
            x = [hdd_bp, hdd_beta, hdd_k, cdd_bp, cdd_beta, cdd_k, intercept]

        hdd_bp, cdd_bp, intercept = x[0], x[3], x[6]
        T_fit_bnds = np.array([T_min, T_max])

        model = full_model(*x, T_fit_bnds, T.astype(np.float64))
        f_unc = np.ones_like(model) * submodel.f_unc

        load_only = model - intercept

        hdd_load = np.zeros_like(model)
        cdd_load = np.zeros_like(model)

        hdd_idx = np.argwhere(T <= hdd_bp).flatten()
        cdd_idx = np.argwhere(T >= cdd_bp).flatten()

        hdd_load[hdd_idx] = load_only[hdd_idx]
        cdd_load[cdd_idx] = load_only[cdd_idx]

        return model, f_unc, hdd_load, cdd_load
