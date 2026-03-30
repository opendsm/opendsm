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

from opendsm.common.base_settings import BaseSettings, CustomField
from opendsm.eemeter.models.daily.utilities.settings import (
    ModelSelectionCriteria,
    Season_Definition,
    Weekday_Weekend_Definition,
    Split_Selection_Definition,
    _check_developer_mode,
)



class BcType(str, Enum):
    NATURAL = "natural"
    CLAMPED = "clamped"


class PSpline_Split_Selection_Definition(Split_Selection_Definition):
    """Split-selection settings with defaults tuned for the P-spline model.

    The P-spline uses ``len(parts)`` (number of structural components) as
    the complexity term in split selection, unlike the DailyModel which
    penalizes total parameter count.  These defaults were optimized via
    grid search on 50 meters × 3 folds to maximize out-of-sample RMSE
    improvement vs the DailyModel baseline.
    """

    penalty_multiplier: float = CustomField(
        default=0.40,
        ge=0,
        developer=True,
        description="Penalty multiplier for season/weekday-weekend split selection",
    )

    penalty_power: float = CustomField(
        default=1.50,
        ge=1,
        developer=True,
        description="Penalty power for season/weekday-weekend split selection",
    )


class Zone_Settings(BaseSettings):
    """Zone-specific configuration for DailyPSpline fitting."""

    n_min: int = CustomField(
        default=5,
        ge=1,
        developer=True,
        description="Minimum number of data points required per zone (HDD / CDD)",
    )

    knot_count_max: Optional[int] = CustomField(
        default=None,
        ge=0,
        developer=True,
        description="Upper bound on internal knots per zone (HDD / CDD); None disables the cap",
    )

    allow_heating_zone: bool = CustomField(
        default=True,
        developer=True,
        description="Whether to allow fitting a monotonically decreasing HDD zone",
    )

    allow_cooling_zone: bool = CustomField(
        default=True,
        developer=True,
        description="Whether to allow fitting a monotonically increasing CDD zone",
    )

    criteria: ModelSelectionCriteria = CustomField(
        default=ModelSelectionCriteria.AIC,
        developer=True,
        description="What selection criteria is used to select data splits of models",
    )

    penalty_multiplier: float = CustomField(
        default=0.31,
        gt=0,
        developer=True,
        description="Penalty multiplier for zone knot-count selection criteria",
    )

    penalty_power: float = CustomField(
        default=1.75,
        gt=0,
        developer=True,
        description="What power should the penalty of the zone knot-count selection criteria be raised to",
    )


# Per-degree default overrides.  Applied automatically based on bspline_degree
# unless the user explicitly sets the field.  Optimized via grid search on
# 50 meters × 3 folds against DailyModel baseline.
_DEGREE_DEFAULTS: dict[int, dict] = {
    0: {
        "regularization_alpha": 0.01,
        "split_selection": {"penalty_multiplier": 0.24, "penalty_power": 2.061},
        "zone": {"penalty_multiplier": 0.24, "penalty_power": 2.061},
    },
    1: {
        "regularization_alpha": 0.01,
        "split_selection": {"penalty_multiplier": 0.24, "penalty_power": 2.061},
        "zone": {"penalty_multiplier": 0.24, "penalty_power": 2.061},
    },
    # Degree 2 inherits from the class-level defaults (degree 3 values).
    3: {
        "regularization_alpha": 0.0,
        "split_selection": {"penalty_multiplier": 0.40, "penalty_power": 1.50},
        "zone": {"penalty_multiplier": 0.31, "penalty_power": 1.75},
    },
}


class DailyPSplineSettings(BaseSettings):
    """Settings for the DailyPSplineModel.

    Per-degree defaults are applied automatically based on ``bspline_degree``.
    Any field explicitly set by the user takes precedence over the per-degree
    default.  Degrees not listed in ``_DEGREE_DEFAULTS`` use the class-level
    defaults (which match degree 3).

    Attributes:
        developer_mode: Unlocks developer-only settings.
        bspline_degree: Degree of the B-spline basis (1–5). Developer only.
        bc_type: Boundary condition type for the spline ('natural' or 'clamped'). Developer only.
        lambda_smoothing: Third-derivative smoothing penalty weight. Developer only.
        kappa_penalty: Monotonicity constraint penalty weight. Developer only.
        maxiter: Maximum iterations for the monotonicity constraint loop. Developer only.
        adaptive_iterations: Maximum outer adaptive-reweighting iterations. Developer only.
        zone: Zone-specific fitting settings (knot counts, zone toggles, selection criteria). Developer only.
        regularization_alpha: Strength of breakpoint regularization. Developer only.
        regularization_percent_lasso: Lasso fraction of breakpoint regularization (0–1). Developer only.
        split_selection: Season/weekday-weekend split-selection sub-settings. Developer only.
        season: Month → season label mapping.
        weekday_weekend: Day-of-week → weekday/weekend label mapping.
        cvrmse_threshold: CV-RMSE above which the fitted model is disqualified.
        pnrmse_threshold: PNRMSE above which the fitted model is disqualified.
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
    # P-spline hyperparameters
    # ------------------------------------------------------------------

    bspline_degree: int = CustomField(
        default=3,
        ge=0,
        le=5,
        developer=True,
        description="Degree of the B-spline basis functions",
    )

    bc_type: Optional[BcType] = CustomField(
        default=BcType.NATURAL,
        developer=True,
        description="Boundary condition type: 'natural' (zero second derivative) "
                    "or 'clamped' (zero first derivative) at fit bounds",
    )

    lambda_smoothing: float = CustomField(
        default=0.0,
        ge=0,
        developer=True,
        description="Third-derivative smoothing penalty weight; 0 disables smoothing",
    )

    kappa_penalty: float = CustomField(
        default=1e9,
        gt=0,
        developer=True,
        description="Monotonicity constraint penalty weight; higher enforces stricter monotonicity",
    )

    maxiter: int = CustomField(
        default=100,
        ge=1,
        developer=True,
        description="Maximum iterations for the monotonicity constraint loop",
    )

    adaptive_iterations: int = CustomField(
        default=10,
        ge=1,
        developer=True,
        description="Maximum outer adaptive-reweighting iterations",
    )

    freeze_bp_on_convergence: bool = CustomField(
        default=False,
        developer=True,
        description=(
            "Skip re-optimizing breakpoints in later adaptive iterations "
            "when bp has already converged (saves ~27% per skipped iteration)"
        ),
    )

    regularization_alpha: float = CustomField(
        default=0.0,
        ge=0,
        developer=True,
        description="Strength of breakpoint regularization; 0 disables regularization",
    )

    regularization_percent_lasso: float = CustomField(
        default=1.0,
        ge=0,
        le=1,
        developer=True,
        description="Lasso fraction of breakpoint regularization (remainder is ridge)",
    )

    # ------------------------------------------------------------------
    # Zone settings
    # ------------------------------------------------------------------

    zone: Zone_Settings = CustomField(
        default_factory=Zone_Settings,
        developer=True,
        description="Zone-specific fitting settings (knot counts, zone toggles, selection criteria)",
    )

    # ------------------------------------------------------------------
    # Split-selection settings  (shared with DailyModel)
    # ------------------------------------------------------------------

    split_selection: PSpline_Split_Selection_Definition = CustomField(
        default_factory=PSpline_Split_Selection_Definition,
        developer=True,
        description="Season / weekday-weekend split-selection sub-settings",
    )

    # ------------------------------------------------------------------
    # Season and weekday/weekend definitions  (user-facing)
    # ------------------------------------------------------------------

    season: Season_Definition = CustomField(
        default_factory=Season_Definition,
        developer=False,
        description="Month → season label mapping",
    )

    weekday_weekend: Weekday_Weekend_Definition = CustomField(
        default_factory=Weekday_Weekend_Definition,
        developer=False,
        description="Day-of-week → weekday/weekend label mapping",
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

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @pydantic.model_validator(mode="after")
    def _apply_degree_defaults(self) -> "DailyPSplineSettings":
        """Apply per-degree default overrides for fields the user didn't set.

        Checks ``_DEGREE_DEFAULTS`` for the current ``bspline_degree``.
        For each override, applies it only if the user did not explicitly
        provide that field (determined via ``model_fields_set``).
        """
        degree_overrides = _DEGREE_DEFAULTS.get(self.bspline_degree)
        if degree_overrides is None:
            return self

        user_set = self.model_fields_set

        for field_name, value in degree_overrides.items():
            if isinstance(value, dict):
                # Nested model (zone, split_selection): apply sub-field overrides
                # only for sub-fields the user didn't explicitly set.
                nested_obj = getattr(self, field_name)
                nested_set = nested_obj.model_fields_set if field_name in user_set else set()
                for sub_field, sub_value in value.items():
                    if sub_field not in nested_set:
                        object.__setattr__(nested_obj, sub_field, sub_value)
            else:
                if field_name not in user_set:
                    object.__setattr__(self, field_name, value)

        return self

    @pydantic.model_validator(mode="after")
    def _check_developer_mode(self) -> "DailyPSplineSettings":
        if self.developer_mode:
            if not self.silent_developer_mode:
                print(
                    "Warning: DailyPSplineModel is nonstandard and should be "
                    "explicitly stated in any derived work"
                )
            return self

        _check_developer_mode(self)
        return self

    # @pydantic.model_validator(mode="after")
    # def _check_knot_degree_compatibility(self) -> "DailyPSplineSettings":
    #     if self.zone_knot_count - 1 < self.bspline_degree:
    #         raise ValueError(
    #             f"zone_knot_count ({self.zone_knot_count}) must be >= "
    #             f"bspline_degree ({self.bspline_degree}); "
    #             f"a degree-{self.bspline_degree} spline requires at least "
    #             f"{self.bspline_degree} knots per zone"
    #         )
    #     return self