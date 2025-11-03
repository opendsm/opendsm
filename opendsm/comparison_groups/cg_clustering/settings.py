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
from typing import Optional, Union

import pydantic

from opendsm.common.base_settings import BaseSettings
from opendsm.common.clustering import settings as _settings
from opendsm.common import const as _const

from opendsm.common.stats.adaptive_loss import LOSS_ALPHA_MIN as _LOSS_ALPHA_MIN



class AdaptiveLossChoice(str, Enum):
    SSE = "sse"
    MAE = "mae"
    L2 = "l2"
    L1 = "l1"
    ADAPTIVE = "adaptive"


class TreatmentMatchSettings(BaseSettings):
    """aggregation type for loadshape"""
    agg_type: _settings.AggregateMethod = pydantic.Field(
        default=_settings.AggregateMethod.MEDIAN
    )

    """treatment meter match loss type"""
    adaptive_loss_alpha: Union[AdaptiveLossChoice, float] = pydantic.Field(
        default=AdaptiveLossChoice.MAE
    )

    adaptive_loss_sigma: float = pydantic.Field(
        default=2.698,  # 1.5 IQR
        gt= 0.0,
    )

    adaptive_loss_c_algo: _const.CAlgoChoice = pydantic.Field(
        default=_const.CAlgoChoice.IQR
    )

    percent_cluster_minimum: float = pydantic.Field(
        default=1E-6,
        ge=0.0,
    )

    """Check if valid settings for treatment meter match loss"""
    @pydantic.model_validator(mode="after")
    def _check_treatment_match_loss(self):
        self._adaptive_loss_alpha = self.adaptive_loss_alpha

        if isinstance(self._adaptive_loss_alpha, str):
            if self._adaptive_loss_alpha == "adaptive":
                pass

            elif self._adaptive_loss_alpha in ["sse", "l2"]:
                self._adaptive_loss_alpha = 2.0

            elif self._adaptive_loss_alpha in ["mae", "l1"]:
                self._adaptive_loss_alpha = 1.0
                
            else:
                raise ValueError("`treatment_match_loss` must be either ['SSE', 'MAE', 'L2', 'L1', 'adaptive'] or float")
            
        else:
            if self._adaptive_loss_alpha < _LOSS_ALPHA_MIN:
                raise ValueError(f"`treatment_match_loss` must be greater than {_LOSS_ALPHA_MIN:.0f}")

            if self._adaptive_loss_alpha > 2:
                raise ValueError("`treatment_match_loss` must be less than 2")

        return self


class _CG_Clustering_Settings(_settings.ClusteringSettings):
    treatment_match: TreatmentMatchSettings = pydantic.Field(
        default=TreatmentMatchSettings(),
    )


class ClusteringSettings(BaseSettings):
    pass

def CG_Clustering_Settings(**kwargs) -> _CG_Clustering_Settings:
    default_dict = {
        "normalize": {
            "method": _settings.NormalizeChoice.MIN_MAX_QUANTILE,
            "quantile": 0.1,
            "pre_transform": True,
            "post_transform": False,
            "axis": 1,
        },
        "transform_selection": _settings.TransformChoice.FPCA,
        "fpca_transform": {
            "min_var_ratio": 0.97,
        },
        "algorithm_selection": _settings.ClusterAlgorithms.BISECTING_KMEANS,
        "bisecting_kmeans": {
            "recluster_count": 3,
            "internal_recluster_count": 5,
            "inner_algorithm": _settings.BiKmeansInnerAlgorithms.ELKAN,
            "bisecting_strategy": _settings.BiKmeansBisectingStrategies.LARGEST_CLUSTER,
            "n_cluster": {
                "lower": 8,
                "upper": 1500,
            },
            "scoring": {
                "min_cluster_size": 15,
                "max_non_outlier_cluster_count": 200,
                "score_metric": _settings.ClusterScoringMetric.VARIANCE_RATIO,
                "distance_metric": _settings.DistanceMetric.EUCLIDEAN,
            },
        },
        "sort_clusters": False,
        "cluster_sort_options": {
            "method": _settings.SortMethod.PEAK,
            "aggregation": _settings.AggregateMethod.MEAN,
            "reverse": False,
        },
        "seed": 42,
    }

    # Update default_dict with any provided keyword arguments
    default_dict.update(kwargs)

    return _CG_Clustering_Settings(**default_dict)


if __name__ == "__main__":
    s = CG_Clustering_Settings()

    print(s.model_dump_json())