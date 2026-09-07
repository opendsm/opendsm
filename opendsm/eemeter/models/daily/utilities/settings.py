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

import pydantic

from enum import Enum
from typing import Any, ClassVar, Optional, Literal, Union

from opendsm.common.base_settings import BaseSettings, CustomField
from opendsm.eemeter.common.data_settings import DailyDataSettings
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
    january: str = CustomField(default="winter", developer=False)
    february: str = CustomField(default="winter", developer=False)
    march: str = CustomField(default="shoulder", developer=False)
    april: str = CustomField(default="shoulder", developer=False)
    may: str = CustomField(default="shoulder", developer=False)
    june: str = CustomField(default="summer", developer=False)
    july: str = CustomField(default="summer", developer=False)
    august: str = CustomField(default="summer", developer=False)
    september: str = CustomField(default="summer", developer=False)
    october: str = CustomField(default="shoulder", developer=False)
    november: str = CustomField(default="winter", developer=False)
    december: str = CustomField(default="winter", developer=False)

    options: ClassVar[list[str]] = ["summer", "shoulder", "winter"]

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
    monday: str = CustomField(default="weekday", developer=False)
    tuesday: str = CustomField(default="weekday", developer=False)
    wednesday: str = CustomField(default="weekday", developer=False)
    thursday: str = CustomField(default="weekday", developer=False)
    friday: str = CustomField(default="weekday", developer=False)
    saturday: str = CustomField(default="weekend", developer=False)
    sunday: str = CustomField(default="weekend", developer=False)

    options: ClassVar[list[str]] = ["weekday", "weekend"]

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


_LEGACY_PRESET = {
    "allow_smooth_model": False,
    "alpha_final": 2.0,
    "segment_minimum_count": 10,
    "split_selection": {
        "allow_separate_summer": False,
        "allow_separate_shoulder": False,
        "allow_separate_winter": False,
        "allow_separate_weekday_weekend": False,
        "reduce_splits_by_gaussian": False,
        "reduce_splits_num_std": None,
    },
}

_PRESETS = {
    "current": {},
    "legacy": _LEGACY_PRESET,
}


def _lowercase_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k.lower().strip() if isinstance(k, str) else k: _lowercase_keys(v) for k, v in value.items()}

    return value


def _merge_preset(preset: dict, values: dict) -> dict:
    """Preset values for every key the caller did not supply, nested dicts merged key by key"""

    merged = dict(preset)
    for key, value in values.items():
        preset_value = merged.get(key)

        # Only dump nested settings objects into the merge when the preset itself
        # defines that key as a nested dict; otherwise the value carries no
        # preset overrides to merge and must pass through untouched so its type
        # (e.g. a BillingDataSettings instance) is preserved.
        if isinstance(preset_value, dict):
            if isinstance(value, pydantic.BaseModel):
                value = value.model_dump(exclude_unset=True)

            if isinstance(value, dict):
                merged[key] = _merge_preset(preset_value, value)
                continue

        merged[key] = value

    return merged


class DailySettings(BaseSettings):
    """Settings for creating the daily model.

    These settings should be converted to a dictionary before being passed to the DailyModel class.
    Be advised that any changes to the default settings deviates from OpenEEmeter standard methods and should be used with caution.
    """

    preset: Literal["current", "legacy"] = CustomField(
        default="current",
        developer=False,
        description="Named set of default values that all other settings fall back to",
    )

    data: DailyDataSettings = pydantic.Field(default_factory=DailyDataSettings)

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


    @pydantic.model_validator(mode="before")
    def _apply_preset(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        values = _lowercase_keys(values)

        preset = values.get("preset", cls.model_fields["preset"].default)
        if isinstance(preset, str):
            preset = preset.lower().strip()
            values["preset"] = preset

        if preset not in _PRESETS:
            return values

        return _merge_preset(_PRESETS[preset], values)


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
        keys = list(type(self).model_fields.keys())

        # print away
        key_max = max([len(k) for k in keys]) + 2
        for key in keys:
            if not type(self).model_fields[key].repr:
                continue

            val = getattr(self, key)

            if isinstance(val, dict):
                v_max = max([len(str(v)) for v in list(val.values())])
                k_max = max([len(str(k)) for k in list(val.keys())])
                if k_max == 1:
                    k_max = 2

                for n, (k, v) in enumerate(val.items()):
                    if n == 0:
                        text_all.append(f"{key:>{key_max}s}: {str(k):>{k_max}s}: {v}")

                    elif n < len(val) - 1:
                        text_all.append(f"{'':>{key_max}s}   {str(k):>{k_max}s}: {v}")

                    else:
                        text_all.append(
                            f"{'':>{key_max}s}   {str(k):>{k_max}s}: {str(v):{v_max}s}"
                        )

            else:
                if isinstance(val, str):
                    val = f"'{val}'"

                text_all.append(f"{key:>{key_max}s}: {val}")

        return "\n".join(text_all)
    

def update_daily_settings(settings, update_dict):
    if not isinstance(settings, DailySettings):
        raise TypeError("settings must be an instance of 'Daily_Settings'")

    # update settings with update_dict
    settings_dict = settings.model_dump()
    settings_dict.update(update_dict)

    return type(settings)(**settings_dict)
