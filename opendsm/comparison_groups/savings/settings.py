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

from enum import Enum
from typing import Optional

import pydantic

from opendsm.common.base_settings import BaseSettings



class TransformChoice(str, Enum):
    STANDARDIZE = "standardize"
    BISYMLOG = "bisymlog"
    SCIPY_YJ = "scipy_yj"
    ROBUST_SCIPY_YJ = "robust_scipy_yj"
    ROBUST_YJ = "robust_yj"


class OutlierRejectionSettings(BaseSettings):
    """Settings for outlier rejection"""

    enabled: bool = pydantic.Field(
        default=False,
        description="enables outlier rejection"
    )

    transform: Optional[TransformChoice] = pydantic.Field(
        default=None,
        description="transformation to apply prior to outlier removal"
    )

    std_threshold: float = pydantic.Field(
        default = 3.0,
        gt=0.0,
        description="number of standard deviations at which outliers are defined"
    )

    quantile: float = pydantic.Field(
        default=0.25,
        gt=0.0,
        lt=0.5,
        description="quantile to use for iqr outlier detection"
    )


class CorrectionCapChoice(str, Enum):
    GLOBAL = "global"
    SOLAR = "solar"


class CorrectionCapSettings(BaseSettings):
    """Settings for correction cap"""

    enabled: bool = pydantic.Field(
        default=True,
        description="enables correction cap"
    )   

    type: CorrectionCapChoice = pydantic.Field(
        default=CorrectionCapChoice.SOLAR,
        description="what kind of correction cap to apply"
    )

    value: float = pydantic.Field(
        default=3.0,
        description="maximum correction as a percentage of the treatment model value"
    )

    solar_threshold: Optional[float] = pydantic.Field(
        default = 1/3,
        description="threshold below which the cap applies for solar"
    )

    @pydantic.model_validator(mode="after")
    def _check_solar_cap(self):
        if self.enabled and self.type == CorrectionCapChoice.SOLAR:
            if self.solar_threshold is None:
                raise ValueError(
                    "'solar_threshold' must be specified if 'type' is 'solar'."
                )
        elif self.enabled and self.type == CorrectionCapChoice.GLOBAL:
            if self.solar_threshold is not None:
                raise ValueError(
                    "'solar_threshold' should not be specified if 'type' is 'global'."
                )

        return self


class CorrectionAlgorithm(str, Enum):
    ODID = "ordinary_difference_in_differences"
    PCTDID = "percent_difference_in_differences"
    ABSPCTDID = "absolute_percent_difference_in_differences"


class WeightClusterAggChoice(str, Enum):
    MODEL = "model_magnitude"


class CGCorrectionSettings(BaseSettings):
    """Settings for model correction"""
    
    algorithm: Optional[CorrectionAlgorithm] = pydantic.Field(
        default=CorrectionAlgorithm.ABSPCTDID,
        description="algorithm to correct treatment meter using comparison group"
    )

    weight_cluster_aggregation: Optional[WeightClusterAggChoice] = pydantic.Field(
        default = None,
        description="how to weight cluster aggregation"
    )

    outlier_rejection: OutlierRejectionSettings = pydantic.Field(
        default_factory=OutlierRejectionSettings,
        description="outlier rejection settings"
    )

    correction_cap: CorrectionCapSettings = pydantic.Field(
        default_factory=CorrectionCapSettings,
        description="correction cap settings"
    )

    alpha: float = pydantic.Field(
        default=0.10,
        gt=0.0,
        lt=1.0,
        description="significance level for uncertainty calculations"
    )
    

if __name__ == "__main__":
    s = CGCorrectionSettings()

    print(s.model_dump_json())
