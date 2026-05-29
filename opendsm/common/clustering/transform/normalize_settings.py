from __future__ import annotations

import numpy as np

import pydantic

from enum import Enum

from opendsm.common.base_settings import BaseSettings


class NormalizeChoice(str, Enum):
    MIN_MAX_QUANTILE = "min_max_quantile"
    STANDARDIZE = "standardize"
    MED_MAD = "med_mad"


class NormalizeScope(str, Enum):
    GLOBAL = "global"
    SAMPLE = "sample"


class NormalizeSettings(BaseSettings):
    """Normalization configuration. If enabled=False, no normalization is applied."""

    enabled: bool = pydantic.Field(
        default=True,
        description="Enable normalization (applies both before and after transform)",
    )

    method: NormalizeChoice | None = pydantic.Field(
        default=NormalizeChoice.MED_MAD,
    )

    quantile: float | None = pydantic.Field(
        default=0.1,
        gt=0.0,
        lt=0.5,
    )

    winsorize_threshold: float | None = pydantic.Field(
        default=10.0,
        gt=0.0,
    )

    scope: NormalizeScope = pydantic.Field(
        default=NormalizeScope.SAMPLE,
    )

    _axis: int | None = pydantic.PrivateAttr(
        default=None,
    )

    @pydantic.model_validator(mode="after")
    def _set_axis_from_scope(self):
        self._axis = 1 if self.scope == NormalizeScope.SAMPLE else None
        return self

    @pydantic.model_validator(mode="after")
    def _check_quantile(self):
        if self.method == NormalizeChoice.MIN_MAX_QUANTILE and self.quantile is None:
            raise ValueError(
                f"'quantile' must be specified when 'method' is '{self.method.value}'"
            )
        return self

    @pydantic.model_validator(mode="after")
    def _check_enable(self):
        if self.enabled and self.method is None:
            raise ValueError(
                "'method' must be specified if 'enabled' is True"
            )
        return self
