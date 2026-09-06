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
    YEO_JOHNSON = "yeo_johnson"
    ROBUST_YEO_JOHNSON = "robust_yeo_johnson"
    BOX_COX = "box_cox"
    ROBUST_BOX_COX = "robust_box_cox"


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
        description="maximum correction as a multiple of the treatment model magnitude (cap = |mTr| * value)"
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
        default = WeightClusterAggChoice.MODEL,
        description=(
            "how to weight cluster aggregation. Model-magnitude weighting can "
            "concentrate weight on a single meter in small or magnitude-skewed "
            "clusters; `weight_cap` bounds this, but when it doesn't (a cap "
            "above 0.5, or a cluster too small to redistribute into) the Kish "
            "effective sample size can still drop below 2. When that happens "
            "the point correction stays weighted, but the cluster's "
            "uncertainty is estimated with uniform weights over its finite "
            "meters, since an effective sample size below 2 cannot support a "
            "weighted interval."
        )
    )

    weight_cap: float = pydantic.Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "upper bound on any single meter's model-magnitude weight within a "
            "cluster. Weights above the cap are clipped to it and the excess is "
            "redistributed proportionally over the uncapped meters (equally if "
            "they all carry zero weight), iterating until no weight exceeds the "
            "cap. A cap of 0.5 or below guarantees a Kish effective sample size "
            "of at least 2 for clusters of 2 or more meters."
        )
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

    min_window_coverage: float = pydantic.Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description=(
            "minimum fraction of the reporting group window a meter must cover "
            "with finite observed and temperature to be corrected. Treatment and "
            "pool meters below it, and meters carrying an eemeter observed "
            "disqualification, are dropped before prediction and recorded on the "
            "correction ledger."
        )
    )
