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

import pydantic
import pytest

from opendsm.eemeter.common.data_settings import (
    BaseSufficiencySettings,
    BillingDataSufficiencySettings,
    DailyDataSufficiencySettings,
    TemperatureSufficiencySettings,
)



def test_default_baseline_lengths():
    """Defaults: minimum is ceil(0.9 * 365) = 329, maximum 366."""
    settings = BaseSufficiencySettings()

    assert settings.min_baseline_length == 329
    assert settings.max_baseline_length == 366


def test_max_baseline_must_exceed_min():
    """max_baseline_length <= min_baseline_length is rejected."""
    with pytest.raises(pydantic.ValidationError, match="must be greater than"):
        BaseSufficiencySettings(min_baseline_length=300, max_baseline_length=300)


def test_baseline_length_float_coerced_to_int():
    """An integer-valued float baseline length is coerced to int."""
    settings = BaseSufficiencySettings(min_baseline_length=330.0)

    assert settings.min_baseline_length == 330
    assert isinstance(settings.min_baseline_length, int)


@pytest.mark.parametrize("bad_pct", [0.0, 1.5, -0.1])
def test_column_coverage_percentage_out_of_range_rejected(bad_pct):
    """Coverage fractions must lie in (0, 1]."""
    with pytest.raises(pydantic.ValidationError):
        TemperatureSufficiencySettings(min_pct_daily_coverage=bad_pct)


def test_temperature_defaults():
    """Temperature coverage thresholds default to the documented values."""
    settings = TemperatureSufficiencySettings()

    assert settings.min_pct_hourly_coverage == 0.5
    assert settings.min_pct_daily_coverage == 0.9
    assert settings.min_pct_period_coverage == 0.9


def test_billing_period_bounds_default_and_validate():
    """Billing period bounds default sensibly and reject non-positive day counts."""
    settings = BillingDataSufficiencySettings()

    assert settings.min_days_in_period == 25
    assert settings.max_days_in_monthly_period == 70

    with pytest.raises(pydantic.ValidationError):
        BillingDataSufficiencySettings(min_days_in_period=0)


def test_daily_settings_have_no_ghi():
    """Daily/billing settings disable the GHI column entirely."""
    assert DailyDataSufficiencySettings().ghi is None
    assert BillingDataSufficiencySettings().ghi is None
