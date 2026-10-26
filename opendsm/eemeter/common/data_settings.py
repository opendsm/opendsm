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

import numpy as np

import pydantic

from typing import Any, ClassVar, Optional

from opendsm.common.base_settings import BaseSettings, DeveloperField, UserField, lowercase_keys
from opendsm.eemeter.common.sufficiency_registry import ALL_PHASES, RULE_SPECS



class RuleToggle(BaseSettings):
    """Whether a sufficiency rule runs on baseline data, on reporting data, or neither."""

    baseline: bool = DeveloperField(
        default=False,
        description="Run the rule when judging baseline data.",
    )

    reporting: bool = DeveloperField(
        default=False,
        description="Run the rule when judging reporting data.",
    )


class AdvisoryRuleToggle(RuleToggle):
    """Toggle for a rule whose verdict is only advisory."""

    baseline: bool = UserField(
        default=False,
        description="Run the rule when judging baseline data.",
    )

    reporting: bool = UserField(
        default=False,
        description="Run the rule when judging reporting data.",
    )


def _toggle(toggle: type[RuleToggle], baseline: bool = False, reporting: bool = False) -> Any:
    """Declare a toggle field switched on for the given phases by default.

    The field itself stays developer-tier so that deviation reports recurse into it; the
    user/developer tier of a rule lives on the bools of its toggle class.
    """
    field = DeveloperField(
        default_factory=lambda: toggle(baseline=baseline, reporting=reporting),
    )

    return field


class RuleToggleSettings(BaseSettings):
    """A settings block whose toggle fields are named for the rules they switch on."""

    model_config = pydantic.ConfigDict(extra="forbid")

    # toggle field name -> the threshold field the rule cannot run without
    rule_thresholds: ClassVar[dict[str, str]] = {}

    @pydantic.model_validator(mode="after")
    def check_toggled_rules_can_run(self):
        for name in type(self).model_fields:
            toggle = getattr(self, name)

            if not isinstance(toggle, RuleToggle):
                continue

            spec = RULE_SPECS[name]
            threshold = self.rule_thresholds.get(name)

            for phase in ALL_PHASES:
                if not getattr(toggle, phase):
                    continue

                if phase not in spec.phases:
                    raise ValueError(
                        f"sufficiency rule '{name}' never runs in the {phase} phase and cannot be enabled there"
                    )

                if threshold is not None and getattr(self, threshold) is None:
                    raise ValueError(
                        f"sufficiency rule '{name}' is enabled for the {phase} phase but {threshold} is None"
                    )

        return self


class WholeFrameSufficiencySettings(RuleToggleSettings):
    """Toggles for the sufficiency rules that judge the frame rather than one column."""

    baseline_day_length: RuleToggle = _toggle(
        RuleToggle, baseline=True,
    )

    negative_observed_values: RuleToggle = _toggle(
        RuleToggle, baseline=True, reporting=True,
    )

    extreme_values: AdvisoryRuleToggle = _toggle(
        AdvisoryRuleToggle, baseline=True,
    )


class HourlyWholeFrameSufficiencySettings(WholeFrameSufficiencySettings):
    consecutive_missing_temperature_hours: RuleToggle = _toggle(
        RuleToggle,
    )


class DailyWholeFrameSufficiencySettings(WholeFrameSufficiencySettings):
    high_frequency_meter_coverage: AdvisoryRuleToggle = _toggle(
        AdvisoryRuleToggle, baseline=True, reporting=True,
    )


class BillingWholeFrameSufficiencySettings(WholeFrameSufficiencySettings):
    offcycle_reads: RuleToggle = _toggle(
        RuleToggle, baseline=True, reporting=True,
    )

    estimated_reads: AdvisoryRuleToggle = _toggle(
        AdvisoryRuleToggle, baseline=True, reporting=True,
    )


class ColumnSufficiencySettings(RuleToggleSettings):
    rule_thresholds: ClassVar[dict[str, str]] = {
        "unique_values": "min_pct_unique_values",
        "daily_coverage": "min_pct_daily_coverage",
        "monthly_coverage": "min_pct_monthly_coverage",
    }

    min_pct_daily_coverage: Optional[float] = pydantic.Field(
        default=None,
        gt=0,
        le=1,
        description="Minimum percentage of daily coverage.",
    )

    min_pct_monthly_coverage: Optional[float] = pydantic.Field(
        default=None,
        gt=0,
        le=1,
        description="Minimum percentage of monthly coverage.",
    )

    min_pct_unique_values: Optional[float] = DeveloperField(
        default=None,
        gt=0,
        le=1,
        description="Minimum percentage of unique values compared to total values.",
    )


class TemperatureSufficiencySettings(ColumnSufficiencySettings):
    min_pct_period_coverage: Optional[float] = DeveloperField(
        default=None,
        gt=0,
        le=1,
        description="Minimum percentage of period coverage.",
    )

    unique_values: RuleToggle = _toggle(RuleToggle)

    daily_coverage: RuleToggle = _toggle(
        RuleToggle, baseline=True, reporting=True,
    )

    monthly_coverage: RuleToggle = _toggle(
        RuleToggle, baseline=True, reporting=True,
    )

    def __init__(self, **kwargs):
        settings = {
            'min_pct_daily_coverage': 0.9,
            'min_pct_monthly_coverage': 0.9,
            'min_pct_period_coverage': 0.9,
            'min_pct_unique_values': None,
        }
        settings.update(lowercase_keys(kwargs))
        super().__init__(**settings)


class GhiSufficiencySettings(ColumnSufficiencySettings):
    unique_values: RuleToggle = _toggle(RuleToggle)

    monthly_coverage: RuleToggle = _toggle(
        RuleToggle, baseline=True, reporting=True,
    )

    def __init__(self, **kwargs):
        # Define settings with defaults for GHI-specific fields
        settings = {
            'min_pct_daily_coverage': None,
            'min_pct_monthly_coverage': 0.9,
            'min_pct_unique_values': None,
        }
        settings.update(lowercase_keys(kwargs))
        super().__init__(**settings)


class ObservedSufficiencySettings(ColumnSufficiencySettings):
    min_pct_hourly_coverage: Optional[float] = DeveloperField(
        default=None,
        gt=0,
        le=1,
        description="Minimum percentage of hourly coverage.",
    )

    unique_values: RuleToggle = _toggle(
        RuleToggle, baseline=True, reporting=True,
    )

    daily_coverage: RuleToggle = _toggle(
        RuleToggle, baseline=True, reporting=True,
    )

    monthly_coverage: RuleToggle = _toggle(
        RuleToggle, baseline=True,
    )

    def __init__(self, **kwargs):
        # Define settings with defaults for Observed-specific fields
        settings = {
            'min_pct_hourly_coverage': 0.5,
            'min_pct_daily_coverage': 0.9,
            'min_pct_monthly_coverage': 0.9,
            'min_pct_unique_values': 0.10,
        }
        settings.update(lowercase_keys(kwargs))
        super().__init__(**settings)


class JointSufficiencySettings(ColumnSufficiencySettings):
    daily_coverage: RuleToggle = _toggle(
        RuleToggle, baseline=True, reporting=True,
    )

    def __init__(self, **kwargs):
        # Define settings with defaults for Joint-specific fields
        settings = {
            'min_pct_daily_coverage': 0.9,
            'min_pct_monthly_coverage': None,
            'min_pct_unique_values': None,
        }
        settings.update(lowercase_keys(kwargs))
        super().__init__(**settings)


class BaseSufficiencySettings(BaseSettings):
    model_config = pydantic.ConfigDict(extra="forbid")

    min_baseline_length: int = pydantic.Field(
        default=int(np.ceil(0.9 * 365)),
        ge=1,
        description="Minimum number of days in the baseline.",
    )

    max_baseline_length: int = pydantic.Field(
        default=366, # 366 for leap year
        ge=2,
        description="Maximum number of days in the baseline.",
    )

    rules: WholeFrameSufficiencySettings = DeveloperField(
        default_factory=WholeFrameSufficiencySettings,
    )

    temperature: TemperatureSufficiencySettings = pydantic.Field(
        default_factory=TemperatureSufficiencySettings,
    )

    ghi: GhiSufficiencySettings = pydantic.Field(
        default_factory=GhiSufficiencySettings,
    )

    observed: ObservedSufficiencySettings = pydantic.Field(
        default_factory=ObservedSufficiencySettings,
    )

    joint: JointSufficiencySettings = pydantic.Field(
        default_factory=JointSufficiencySettings,
    )

    @pydantic.field_validator("min_baseline_length", "max_baseline_length", mode="before")
    @classmethod
    def convert_float_to_int(cls, v):
        if isinstance(v, float) and v.is_integer():
            v = int(v)

        return v

    @pydantic.model_validator(mode="after")
    def check_baseline_lengths(self):
        max_baseline_length = self.max_baseline_length
        min_baseline_length = self.min_baseline_length
        if max_baseline_length <= min_baseline_length:
            raise ValueError(
                f"max_baseline_length ({max_baseline_length}) must be greater than min_baseline_length ({min_baseline_length})"
            )

        return self


class DailyTemperatureSufficiencySettings(TemperatureSufficiencySettings):
    unique_values: RuleToggle = _toggle(
        RuleToggle, baseline=True,
    )

    def __init__(self, **kwargs):
        # a constant temperature series leaves the balance points unidentifiable
        settings = {
            'min_pct_unique_values': 0.10,
        }
        settings.update(lowercase_keys(kwargs))
        super().__init__(**settings)


class DailyObservedSufficiencySettings(ObservedSufficiencySettings):
    monthly_coverage: RuleToggle = _toggle(RuleToggle)


class BillingObservedSufficiencySettings(ObservedSufficiencySettings):
    unique_values: RuleToggle = _toggle(RuleToggle)

    monthly_coverage: RuleToggle = _toggle(RuleToggle)


class DailyDataSufficiencySettings(BaseSufficiencySettings):
    ghi: None = None

    rules: DailyWholeFrameSufficiencySettings = DeveloperField(
        default_factory=DailyWholeFrameSufficiencySettings,
    )

    temperature: DailyTemperatureSufficiencySettings = pydantic.Field(
        default_factory=DailyTemperatureSufficiencySettings,
    )

    observed: DailyObservedSufficiencySettings = DeveloperField(
        default_factory=DailyObservedSufficiencySettings,
    )


class BillingDataSufficiencySettings(BaseSufficiencySettings):
    ghi: None = None

    rules: BillingWholeFrameSufficiencySettings = DeveloperField(
        default_factory=BillingWholeFrameSufficiencySettings,
    )

    temperature: DailyTemperatureSufficiencySettings = pydantic.Field(
        default_factory=DailyTemperatureSufficiencySettings,
    )

    observed: BillingObservedSufficiencySettings = DeveloperField(
        default_factory=BillingObservedSufficiencySettings,
    )

    min_days_in_period: int = pydantic.Field(
        default=25,
        ge=1,
        description="Minimum number of days in a billing period.",
    )

    max_days_in_monthly_period: int = pydantic.Field(
        default=35,
        ge=1,
        description="Maximum number of days in a monthly billing period.",
    )

    max_days_in_bimonthly_period: int = pydantic.Field(
        default=70,
        ge=1,
        description="Maximum number of days in a bimonthly billing period.",
    )

    @pydantic.field_validator("min_days_in_period", "max_days_in_monthly_period", "max_days_in_bimonthly_period", mode="before")
    @classmethod
    def convert_float_to_int(cls, v):
        if isinstance(v, float) and v.is_integer():
            v = int(v)

        return v


class HourlyTemperatureSufficiencySettings(TemperatureSufficiencySettings):
    max_consecutive_hours_missing: int = pydantic.Field(
        default=6,
        ge=0,
        description=(
            "Longest run of missing temperature hours allowed; a longer run disqualifies the "
            "meter when the consecutive_missing_temperature_hours rule is on."
        ),
    )

    @pydantic.field_validator("max_consecutive_hours_missing", mode="before")
    @classmethod
    def convert_float_to_int(cls, v):
        if isinstance(v, float) and v.is_integer():
            v = int(v)

        return v

class HourlyDataSufficiencySettings(BaseSufficiencySettings):
    rules: HourlyWholeFrameSufficiencySettings = DeveloperField(
        default_factory=HourlyWholeFrameSufficiencySettings,
    )

    temperature: HourlyTemperatureSufficiencySettings = pydantic.Field(
        default_factory=HourlyTemperatureSufficiencySettings,
    )


_REMOVED_TOP_LEVEL_SUFFICIENCY_KEYS: set[str] = {"requested_start", "requested_end"}

_REMOVED_SUFFICIENCY_KEYS: dict[str, set[str]] = {
    "temperature": {"min_pct_hourly_coverage"},
    "observed": {"min_pct_period_coverage"},
    "ghi": {"min_pct_hourly_coverage", "min_pct_period_coverage"},
    "joint": {"min_pct_hourly_coverage", "min_pct_period_coverage"},
}


def strip_removed_sufficiency_keys(settings_payload: dict) -> dict:
    """Drop the keys in `_REMOVED_TOP_LEVEL_SUFFICIENCY_KEYS` and `_REMOVED_SUFFICIENCY_KEYS`
    from a settings payload's sufficiency blocks without mutating the payload.

    A payload carrying none of those keys is returned as is.
    """
    data = settings_payload.get("data")
    if not isinstance(data, dict):
        return settings_payload

    sufficiency = data.get("sufficiency")
    if not isinstance(sufficiency, dict):
        return settings_payload

    new_sufficiency = {
        k: v for k, v in sufficiency.items() if k not in _REMOVED_TOP_LEVEL_SUFFICIENCY_KEYS
    }
    for block_name, removed_keys in _REMOVED_SUFFICIENCY_KEYS.items():
        block = new_sufficiency.get(block_name)
        if isinstance(block, dict):
            new_sufficiency[block_name] = {
                k: v for k, v in block.items() if k not in removed_keys
            }

    if new_sufficiency == sufficiency:
        return settings_payload

    new_data = {**data, "sufficiency": new_sufficiency}
    stripped = {**settings_payload, "data": new_data}

    return stripped


class DailyDataSettings(BaseSettings):
    sufficiency: DailyDataSufficiencySettings = pydantic.Field(
        default_factory=DailyDataSufficiencySettings,
    )

class BillingDataSettings(BaseSettings):
    sufficiency: BillingDataSufficiencySettings = pydantic.Field(
        default_factory=BillingDataSufficiencySettings,
    )

class HourlyDataSettings(BaseSettings):
    sufficiency: HourlyDataSufficiencySettings = pydantic.Field(
        default_factory=HourlyDataSufficiencySettings,
    )
