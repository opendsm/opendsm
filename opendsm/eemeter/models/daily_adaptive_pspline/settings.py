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

from opendsm.common.base_settings import BaseSettings, CustomField
from opendsm.eemeter.models.daily_pspline.settings import DailyPSplineSettings


class DailyAdaptivePSplineSettings(BaseSettings):
    """Settings for the DailyAdaptivePSplineModel.

    Wraps a ``DailyPSplineSettings`` for per-regime fitting and adds
    EM regime-discovery and assignment-classifier hyperparameters.
    """

    developer_mode: bool = CustomField(
        default=False,
        developer=False,
        description="Allows changing of developer settings",
    )

    silent_developer_mode: bool = CustomField(
        default=False,
        developer=False,
        exclude=True,
        repr=False,
    )

    # ------------------------------------------------------------------
    # Nested PSpline fitting settings
    # ------------------------------------------------------------------

    pspline: DailyPSplineSettings = CustomField(
        default_factory=lambda: DailyPSplineSettings(
            developer_mode=True, silent_developer_mode=True,
        ),
        developer=True,
        description="Per-regime PSpline fitting settings",
    )

    # ------------------------------------------------------------------
    # EM regime discovery
    # ------------------------------------------------------------------

    k_max: int = CustomField(
        default=5,
        ge=1,
        le=10,
        developer=True,
        description="Maximum number of regimes to evaluate",
    )

    em_max_iter: int = CustomField(
        default=50,
        ge=1,
        developer=True,
        description="Maximum EM iterations per K",
    )

    em_convergence_tol: float = CustomField(
        default=1e-3,
        ge=0,
        developer=True,
        description="Relative change in total WRMSE below which EM stops",
    )

    min_regime_days: int = CustomField(
        default=30,
        ge=5,
        developer=True,
        description="Minimum days per regime; smaller clusters are merged",
    )

    bic_penalty_multiplier: float = CustomField(
        default=1.0,
        ge=0,
        developer=True,
        description="Regime penalty multiplier for model selection; "
                    "penalty = multiplier * K^power * ln(N)",
    )

    bic_penalty_power: float = CustomField(
        default=2.0,
        ge=1,
        developer=True,
        description="Regime penalty power for model selection; "
                    "higher values penalize additional regimes progressively more",
    )

    # ------------------------------------------------------------------
    # Assignment classifier features
    # ------------------------------------------------------------------

    trailing_avg_window: int = CustomField(
        default=21,
        ge=7,
        le=90,
        developer=True,
        description="Trailing-average temperature window (days) for seasonal context",
    )

    use_trailing_avg_temp: bool = CustomField(
        default=True,
        developer=True,
        description="Include trailing-average temperature as a classifier feature",
    )

    use_doy_harmonics: bool = CustomField(
        default=True,
        developer=True,
        description="Include sin/cos annual harmonics as classifier features",
    )

    use_day_of_week: bool = CustomField(
        default=True,
        developer=True,
        description="Include day-of-week indicators as classifier features",
    )

    use_daily_temp: bool = CustomField(
        default=False,
        developer=True,
        description="Include daily temperature as a classifier feature "
                    "(for dual-fuel; heavily penalized)",
    )

    daily_temp_penalty: float = CustomField(
        default=10.0,
        ge=1.0,
        developer=True,
        description="Relative L2 penalty on the daily-temperature feature "
                    "vs other features in the assignment classifier",
    )

    # ------------------------------------------------------------------
    # Model quality thresholds
    # ------------------------------------------------------------------

    cvrmse_threshold: float = CustomField(
        default=1.0,
        ge=0,
        developer=True,
        description="CV-RMSE threshold above which the model is disqualified",
    )

    pnrmse_threshold: float = CustomField(
        default=1.6,
        ge=0,
        developer=True,
        description="PNRMSE threshold above which the model is disqualified",
    )
