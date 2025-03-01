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

import pydantic

from enum import Enum
from typing import Optional, Literal, Union

from opendsm.common.base_settings import BaseSettings, CustomField
import opendsm.eemeter.models.daily.utilities.const as _const
from opendsm.eemeter.models.daily.utilities.opt_settings import AlgorithmChoice


# region option definitions
class AlphaFinalType(str, Enum):
    ALL = "all"
    LAST = "last"


class ModelSelectionCriteria(str, Enum):
    RMSE = "rmse"
    RMSE_ADJ = "rmse_adj"
    R_SQUARED = "r_squared"
    R_SQUARED_ADJ = "r_squared_adj"
    AIC = "aic"
    AICC = "aicc"
    CAIC = "caic"
    BIC = "bic"
    SABIC = "sabic"
    FPE = "fpe"

    # Maybe these will be implemented one day
    # DIC = "dic"
    # WAIC = "waic"
    # WBIC = "wbic"


class FullModelSelection(str, Enum):
    HDD_TIDD_CDD = "hdd_tidd_cdd"
    C_HDD_TIDD = "c_hdd_tidd"
    TIDD = "tidd"

# endregion


class Season_Definition(BaseSettings):
    january: str = CustomField(default="winter")
    february: str = CustomField(default="winter")
    march: str = CustomField(default="shoulder")
    april: str = CustomField(default="shoulder")
    may: str = CustomField(default="shoulder")
    june: str = CustomField(default="summer")
    july: str = CustomField(default="summer")
    august: str = CustomField(default="summer")
    september: str = CustomField(default="summer")
    october: str = CustomField(default="shoulder")
    november: str = CustomField(default="winter")
    december: str = CustomField(default="winter")

    options: list[str] = CustomField(default=["summer", "shoulder", "winter"])

    """Set dictionaries of seasons"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> Season_Definition:
        season_dict = {}
        for month, num in _const.season_num.items():
            val = getattr(self, month.lower())
            if val not in self.options:
                raise ValueError(f"SeasonDefinition: {val} is not a valid option. Valid options are {self.options}")
            
            season_dict[num] = val
        
        self._month_index = _const.season_num
        self._num_dict = season_dict
        self._order = {val: i for i, val in enumerate(self.options)}

        return self


class Weekday_Weekend_Definition(BaseSettings):
    monday: str = CustomField(default="weekday")
    tuesday: str = CustomField(default="weekday")
    wednesday: str = CustomField(default="weekday")
    thursday: str = CustomField(default="weekday")
    friday: str = CustomField(default="weekday")
    saturday: str = CustomField(default="weekend")
    sunday: str = CustomField(default="weekend")

    options: list[str] = CustomField(default=["weekday", "weekend"])

    """Set dictionaries of weekday/weekend"""
    @pydantic.model_validator(mode="after")
    def set_numeric_dict(self) -> Weekday_Weekend_Definition:
        weekday_dict = {}
        for day, num in _const.weekday_num.items():
            val = getattr(self, day.lower())
            if val not in self.options:
                raise ValueError(f"WeekdayWeekendDefinition: {val} is not a valid option. Valid options are {self.options}")
            
            weekday_dict[num] = val
        
        self._day_index = _const.weekday_num
        self._num_dict = weekday_dict
        self._order = {val: i for i, val in enumerate(self.options)}

        return self
    

class Split_Selection_Definition(BaseSettings):
    criteria: ModelSelectionCriteria = CustomField(
        default=ModelSelectionCriteria.BIC,
        developer=True,
        description="What selection criteria is used to select data splits of models",
    )

    penalty_multiplier: float = CustomField(
        default=0.24,
        ge=0,
        developer=True,
        description="Penalty multiplier for split selection criteria",
    )

    penalty_power: float = CustomField(
        default=2.061,
        ge=1,
        developer=True,
        description="What power should the penalty of the selection criteria be raised to",
    )

    allow_separate_summer: bool = CustomField(
        default=True,
        developer=True,
        description="Allow summer to be modeled separately",
    )

    allow_separate_shoulder: bool = CustomField(
        default=True,
        developer=True,
        description="Allow shoulder to be modeled separately",
    )

    allow_separate_winter: bool = CustomField(
        default=True,
        developer=True,
        description="Allow winter to be modeled separately",
    )

    allow_separate_weekday_weekend: bool = CustomField(
        default=True,
        developer=True,
        description="Allow weekdays and weekends to be modeled separately",
    )

    reduce_splits_by_gaussian: bool = CustomField(
        default=True,
        developer=True,
        description="Reduces splits by fitting with multivariate Gaussians and testing for overlap",
    )

    reduce_splits_num_std: Optional[list[float]] = CustomField(
        default=[1.4, 0.89],
        developer=True,
        description="Number of standard deviations to use with Gaussians",
    )

    @pydantic.model_validator(mode="after")
    def _check_reduce_splits_num_std(self):
        if self.reduce_splits_num_std is not None:
            if len(self.reduce_splits_num_std) != 2:
                raise ValueError("`REDUCE_SPLITS_NUM_STD` must be a list of length 2")
            
            if self.reduce_splits_num_std[0] <= 0 or self.reduce_splits_num_std[1] <= 0:
                raise ValueError("`REDUCE_SPLITS_NUM_STD` entries must be > 0")
            
        return self


def _check_developer_mode(cls):   
    for k, v in cls.model_fields.items():
        if isinstance(getattr(cls, k), BaseSettings):
            _check_developer_mode(getattr(cls, k))

        elif v.json_schema_extra["developer"] and getattr(cls, k) != v.default:
            raise ValueError(f"Developer mode is not enabled. Cannot change {k} from default value.")

    return cls


class DailySettings(BaseSettings):
    """Settings for creating the daily model.

    These settings should be converted to a dictionary before being passed to the DailyModel class.
    Be advised that any changes to the default settings deviates from OpenEEmeter standard methods and should be used with caution.

    Attributes:
        developer_mode (bool): Allows changing of developer settings
        algorithm_choice (str): Optimization algorithm choice. Developer mode only.
        initial_guess_algorithm_choice (str): Initial guess optimization algorithm choice. Developer mode only.
        full_model (str): The largest model allowed. Developer mode only.
        smoothed_model (bool): Allow smoothed models.
        allow_separate_summer (bool): Allow summer to be modeled separately.
        allow_separate_shoulder (bool): Allow shoulder to be modeled separately.
        allow_separate_winter (bool): Allow winter to be modeled separately.
        allow_separate_weekday_weekend (bool): Allow weekdays and weekends to be modeled separately.
        reduce_splits_by_gaussian (bool): Reduces splits by fitting with multivariate Gaussians and testing for overlap.
        reduce_splits_num_std (list[float]): Number of standard deviations to use with Gaussians.
        alpha_minimum (float): Alpha where adaptive robust loss function is Welsch loss.
        alpha_selection (float): Specified alpha to evaluate which is the best model type.
        alpha_final_type (str): When to use 'alpha_final: 'all': on every model, 'last': on final model, 'None': don't use.
        alpha_final (float | str | None): Specified alpha or 'adaptive' for adaptive loss in model evaluation.
        final_bounds_scalar (float | None): Scalar for calculating bounds of 'alpha_final'.
        regularization_alpha (float): Alpha for elastic net regularization.
        regularization_percent_lasso (float): Percent lasso vs (1 - perc) ridge regularization.
        segment_minimum_count (int): Minimum number of data points for HDD/CDD.
        maximum_slope_OoM_scalar (float): Scaler for initial slope to calculate bounds based on order of magnitude.
        initial_smoothing_parameter (float | None): Initial guess for the smoothing parameter.
        initial_step_percentage (float | None): Initial step-size for relevant algorithms.
        split_selection_criteria (str): What selection criteria is used to select data splits of models.
        split_selection_penalty_multiplier (float): Penalty multiplier for split selection criteria.
        split_selection_penalty_power (float): What power should the penalty of the selection criteria be raised to.
        season (Dict[int, str]): Dictionary of months and their associated season (January is 1).
        is_weekday (Dict[int, bool]): Dictionary of days (1 = Monday) and if that day is a weekday (True/False).
        uncertainty_alpha (float): Significance level used for uncertainty calculations (0 < float < 1).
        cvrmse_threshold (float): Threshold for the CVRMSE to disqualify a model.

    """

    developer_mode: bool = CustomField(
        default=False,
        developer=False,
        description="Developer mode flag",
    )

    silent_developer_mode: bool = CustomField(
        default=False,
        developer=False,
        exclude=True,
        repr=False,
    )

    algorithm_choice: Optional[AlgorithmChoice] = CustomField(
        default=AlgorithmChoice.NLOPT_SBPLX,
        developer=True,
        description="Optimization algorithm choice",
    )

    initial_guess_algorithm_choice: Optional[AlgorithmChoice] = CustomField(
        default=AlgorithmChoice.NLOPT_DIRECT,
        developer=True,
        description="Initial guess optimization algorithm choice",
    )

    full_model: Optional[FullModelSelection] = CustomField(
        default=FullModelSelection.HDD_TIDD_CDD,
        developer=True,
        description="The largest model allowed",
    )

    allow_smooth_model: bool = CustomField(
        default=True,
        developer=True,
        description="Allow smoothed models",
    )

    alpha_minimum: float = CustomField(
        default=-100,
        le=-10,
        developer=True,
        description="Alpha where adaptive robust loss function is Welsch loss",
    )

    alpha_selection: float = CustomField(
        default=2,
        ge=-10,
        le=2,
        developer=True,
        description="Specified alpha to evaluate which is the best model type",
    )

    alpha_final_type: Optional[AlphaFinalType] = CustomField(
        default=AlphaFinalType.LAST,
        developer=True,
        description="When to use 'alpha_final: 'all': on every model, 'last': on final model, 'None': don't use",
    )

    alpha_final: Optional[Union[float, Literal["adaptive"]]] = CustomField(
        default="adaptive",
        developer=True,
        description="Specified alpha or 'adaptive' for adaptive loss in model evaluation",
    )

    final_bounds_scalar: Optional[float] = CustomField(
        default=1,
        developer=True,
        description="Scalar for calculating bounds of 'alpha_final'",
    )

    regularization_alpha: float = CustomField(
        default=0.001,
        ge=0,
        developer=True,
        description="Alpha for elastic net regularization",
    )

    regularization_percent_lasso: float = CustomField(
        default=1,
        ge=0,
        le=1,
        developer=True,
        description="Percent lasso vs (1 - perc) ridge regularization",
    )

    segment_minimum_count: int = CustomField(
        default=6,
        ge=3,
        developer=True,
        description="Minimum number of data points for HDD/CDD",
    )

    maximum_slope_oom_scalar: float = CustomField(
        default=2,
        ge=1,
        developer=True,
        description="Scaler for initial slope to calculate bounds based on order of magnitude",
    )

    initial_step_percentage: Optional[float] = CustomField(
        default=0.1,
        developer=True,
        description="Initial step-size for relevant algorithms",
    )

    split_selection: Split_Selection_Definition = CustomField(
        default_factory=Split_Selection_Definition,
        developer=True,
        description="Settings for split selection",
    )

    season: Season_Definition = CustomField(
        default_factory=Season_Definition,
        developer=False,
        description="Dictionary of months and their associated season (January is 1)",
    )

    weekday_weekend: Weekday_Weekend_Definition = CustomField(
        default_factory=Weekday_Weekend_Definition,
        developer=False,
        description="Dictionary of days (1 = Monday) and if that day is a weekday (True/False)",
    )

    uncertainty_alpha: float = CustomField(
        default=0.1,
        ge=0,
        le=1,
        developer=False,
        description="Significance level used for uncertainty calculations",
    )

    cvrmse_threshold: float = CustomField(
        default=1,
        ge=0,
        developer=True,
        description="Threshold for the CVRMSE to disqualify a model",
    )

    pnrmse_threshold: float = CustomField(
        default=1.6,
        ge=0,
        developer=True,
        description="Threshold for the PNRMSE to disqualify a model",
    )


    @pydantic.model_validator(mode="after")
    def _check_developer_mode(self):
        if self.developer_mode:
            if not self.silent_developer_mode:
                print("Warning: Daily model is nonstandard and should be explicitly stated in any derived work")

            return self
        
        _check_developer_mode(self)

        return self


    @pydantic.model_validator(mode="after")
    def _check_alpha_final(self):
        if self.alpha_final is None:
            if self.alpha_final_type != None:
                raise ValueError("`ALPHA_FINAL` must be set if `ALPHA_FINAL_TYPE` is not None")
            
        elif isinstance(self.alpha_final, float):
            if (self.alpha_minimum > self.alpha_final) or (self.alpha_final > 2.0):
                raise ValueError(
                    f"`ALPHA_FINAL` must be `adaptive` or `ALPHA_MINIMUM` <= float <= 2"
                )

        elif isinstance(self.alpha_final, str):
            if self.alpha_final != "adaptive":
                raise ValueError(
                    f"ALPHA_FINAL must be `adaptive` or `ALPHA_MINIMUM` <= float <= 2"
            )

        return self

    @pydantic.model_validator(mode="after")
    def _check_final_bounds_scalar(self):
        if self.final_bounds_scalar is not None:
            if self.final_bounds_scalar <= 0:
                raise ValueError("`FINAL_BOUNDS_SCALAR` must be > 0")
            
            if self.alpha_final_type is None:
                raise ValueError("`FINAL_BOUNDS_SCALAR` must be None if `ALPHA_FINAL` is None")
            
        else:
            if self.alpha_final_type is not None:
                raise ValueError("`FINAL_BOUNDS_SCALAR` must be > 0 if `ALPHA_FINAL` is not None")

        return self

    
    @pydantic.model_validator(mode="after")
    def _check_initial_step_percentage(self):
        if self.initial_step_percentage is not None:
            if self.initial_step_percentage <= 0 or self.initial_step_percentage > 0.5:
                raise ValueError("`INITIAL_STEP_PERCENTAGE` must be None or 0 < float <= 0.5")
            
        else:
            if self.algorithm_choice[:5] in ["nlopt"]:
                raise ValueError("`INITIAL_STEP_PERCENTAGE` must be specified if `ALGORITHM_CHOICE` is from Nlopt")
            
        return self
            
    
    def __repr__(self):
        text_all = []
        text_all.append(type(self).__name__)

        # get all keys
        keys = list(self.model_fields.keys())

        # print away
        key_max = max([len(k) for k in keys]) + 2
        for key in keys:
            if not self.model_fields[key].repr:
                continue

            val = getattr(self, key)

            if isinstance(val, dict):
                v_max = max([len(str(v)) for v in list(val.values())])
                k_max = max([len(str(k)) for k in list(val.keys())])
                if k_max == 1:
                    k_max = 2

                for n, (k, v) in enumerate(val.items()):
                    if n == 0:
                        text_all.append(f"{key:>{key_max}s}: {{{str(k):>{k_max}s}: {v}")

                    elif n < len(val) - 1:
                        text_all.append(f"{'':>{key_max}s}   {str(k):>{k_max}s}: {v}")

                    else:
                        text_all.append(
                            f"{'':>{key_max}s}   {str(k):>{k_max}s}: {str(v):{v_max}s} }}"
                        )

            else:
                if isinstance(val, str):
                    val = f"'{val}'"

                text_all.append(f"{key:>{key_max}s}: {val}")

        return "\n".join(text_all)
    

class Split_Selection_Legacy_Definition(Split_Selection_Definition):
    allow_separate_summer: bool = CustomField(
        default=False,
        developer=True,
        description="Allow summer to be modeled separately",
    )

    allow_separate_shoulder: bool = CustomField(
        default=False,
        developer=True,
        description="Allow shoulder to be modeled separately",
    )

    allow_separate_winter: bool = CustomField(
        default=False,
        developer=True,
        description="Allow winter to be modeled separately",
    )

    allow_separate_weekday_weekend: bool = CustomField(
        default=False,
        developer=True,
        description="Allow weekdays and weekends to be modeled separately",
    )

    reduce_splits_by_gaussian: bool = CustomField(
        default=False,
        developer=True,
        description="Reduces splits by fitting with multivariate Gaussians and testing for overlap",
    )

    reduce_splits_num_std: Optional[list[float]] = CustomField(
        default=None,
        developer=True,
        description="Number of standard deviations to use with Gaussians",
    )


class DailyLegacySettings(DailySettings):
    allow_smooth_model: bool = CustomField(
        default=False,
        developer=True,
        description="Allow smoothed models",
    )

    alpha_final: Optional[Union[float, Literal["adaptive"]]] = CustomField(
        default=2.0,
        developer=True,
        description="Specified alpha or 'adaptive' for adaptive loss in model evaluation",
    )

    segment_minimum_count: int = CustomField(
        default=10,
        ge=3,
        developer=True,
        description="Minimum number of data points for HDD/CDD",
    )

    split_selection: Split_Selection_Legacy_Definition = CustomField(
        default_factory=Split_Selection_Legacy_Definition,
        developer=True,
        description="Settings for split selection",
    )


def update_daily_settings(settings, update_dict):
    if not isinstance(settings, DailySettings):
        raise TypeError("settings must be an instance of 'Daily_Settings'")

    # update settings with update_dict
    settings_dict = settings.model_dump()
    settings_dict.update(update_dict)

    if isinstance(settings, DailyLegacySettings):
        return DailyLegacySettings(**settings_dict)
     
    return DailySettings(**settings_dict)


# TODO: deprecate
def default_settings(**kwargs) -> DailySettings:
    """
    Returns default settings.
    """
    return DailySettings(**kwargs)

# TODO: deprecate
def caltrack_legacy_settings(**kwargs) -> DailyLegacySettings:
    """
    Returns CalTRACK legacy settings.
    """
    return DailyLegacySettings(**kwargs)