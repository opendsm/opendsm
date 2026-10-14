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

from opendsm.eemeter.common.sufficiency_registry import RULE_SPECS, registered_rules
import opendsm.eemeter.common.sufficiency_rules  # noqa: F401  binds the rule functions
from opendsm.eemeter.common.data_settings import (
    BaseSufficiencySettings,
    BillingDataSufficiencySettings,
    DailyDataSufficiencySettings,
    GhiSufficiencySettings,
    HourlyDataSufficiencySettings,
    HourlyWholeFrameSufficiencySettings,
    RuleToggle,
    TemperatureSufficiencySettings,
    WholeFrameSufficiencySettings,
    strip_removed_sufficiency_keys,
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

    assert settings.min_pct_daily_coverage == 0.9
    assert settings.min_pct_period_coverage == 0.9


def test_billing_period_bounds_default_and_validate():
    """Billing period bounds default sensibly and reject non-positive day counts."""
    settings = BillingDataSufficiencySettings()

    assert settings.min_days_in_period == 25
    assert settings.max_days_in_monthly_period == 35
    assert settings.max_days_in_bimonthly_period == 70

    with pytest.raises(pydantic.ValidationError):
        BillingDataSufficiencySettings(min_days_in_period=0)


def test_daily_settings_have_no_ghi():
    """Daily/billing settings disable the GHI column entirely."""
    assert DailyDataSufficiencySettings().ghi is None
    assert BillingDataSufficiencySettings().ghi is None


SETTINGS_BY_FAMILY = {
    "hourly": HourlyDataSufficiencySettings,
    "daily": DailyDataSufficiencySettings,
    "billing": BillingDataSufficiencySettings,
}

COLUMN_BLOCKS = ("temperature", "ghi", "observed", "joint")

# (baseline, reporting) each rule toggle must default to, per data family
DEFAULT_TOGGLES = {
    "hourly": {
        "rules.baseline_day_length": (True, False),
        "rules.negative_observed_values": (True, True),
        "rules.extreme_values": (True, False),
        "rules.boundary_gap": (False, False),
        "rules.consecutive_missing_temperature_hours": (False, False),
        "rules.high_frequency_observed_coverage": (False, False),
        "temperature.unique_values": (False, False),
        "temperature.daily_coverage": (True, True),
        "temperature.monthly_coverage": (True, True),
        "ghi.unique_values": (False, False),
        "ghi.daily_coverage": (False, False),
        "ghi.monthly_coverage": (True, True),
        "observed.unique_values": (True, True),
        "observed.daily_coverage": (True, True),
        "observed.monthly_coverage": (True, False),
        "joint.unique_values": (False, False),
        "joint.daily_coverage": (True, True),
        "joint.monthly_coverage": (False, False),
    },
    "daily": {
        "rules.baseline_day_length": (True, False),
        "rules.negative_observed_values": (True, True),
        "rules.extreme_values": (True, False),
        "rules.boundary_gap": (False, False),
        "rules.high_frequency_meter_coverage": (True, True),
        "temperature.unique_values": (True, False),
        "temperature.daily_coverage": (True, True),
        "temperature.monthly_coverage": (True, True),
        "observed.unique_values": (True, True),
        "observed.daily_coverage": (True, True),
        "observed.monthly_coverage": (False, False),
        "joint.unique_values": (False, False),
        "joint.daily_coverage": (True, True),
        "joint.monthly_coverage": (False, False),
    },
    "billing": {
        "rules.baseline_day_length": (True, False),
        "rules.negative_observed_values": (True, True),
        "rules.extreme_values": (True, False),
        "rules.boundary_gap": (False, False),
        "rules.high_frequency_meter_coverage": (True, True),
        "rules.offcycle_reads": (True, True),
        "rules.estimated_reads": (True, True),
        "temperature.unique_values": (True, False),
        "temperature.daily_coverage": (True, True),
        "temperature.monthly_coverage": (True, True),
        "observed.unique_values": (False, False),
        "observed.daily_coverage": (True, True),
        "observed.monthly_coverage": (False, False),
        "joint.unique_values": (False, False),
        "joint.daily_coverage": (True, True),
        "joint.monthly_coverage": (False, False),
    },
}


def toggle_paths(settings):
    """Return the dotted paths of every rule toggle a settings object carries."""
    paths = []
    for block_name in ("rules", *COLUMN_BLOCKS):
        block = getattr(settings, block_name)

        if block is None:
            continue

        for field_name in type(block).model_fields:
            if isinstance(getattr(block, field_name), RuleToggle):
                paths.append(f"{block_name}.{field_name}")

    return paths


@pytest.mark.parametrize(
    "family, path, expected",
    [
        (family, path, expected)
        for family, toggles in DEFAULT_TOGGLES.items()
        for path, expected in toggles.items()
    ],
    ids=[
        f"{family}-{path}"
        for family, toggles in DEFAULT_TOGGLES.items()
        for path in toggles
    ],
)
def test_default_rule_toggle_matrix(family, path, expected):
    """Every rule toggle defaults to the phases in ``DEFAULT_TOGGLES``."""
    settings = SETTINGS_BY_FAMILY[family]()
    block_name, field_name = path.split(".")

    toggle = getattr(getattr(settings, block_name), field_name)

    assert (toggle.baseline, toggle.reporting) == expected, (
        f"{family} {path}: expected (baseline, reporting) {expected}, "
        f"got {(toggle.baseline, toggle.reporting)}"
    )


@pytest.mark.parametrize("family", sorted(DEFAULT_TOGGLES))
def test_family_declares_exactly_its_rule_toggles(family):
    """A family carries a toggle for each rule it can run and for no other rule."""
    settings = SETTINGS_BY_FAMILY[family]()

    assert set(toggle_paths(settings)) == set(DEFAULT_TOGGLES[family])


def test_enabling_a_phase_the_rule_never_runs_in_is_rejected():
    """extreme_values is baseline-only, so enabling its reporting phase fails."""
    with pytest.raises(pydantic.ValidationError, match="never runs in the reporting phase"):
        WholeFrameSufficiencySettings(extreme_values={"baseline": True, "reporting": True})


def test_enabling_a_rule_without_its_threshold_is_rejected():
    """A column rule cannot be switched on while its threshold is None."""
    with pytest.raises(pydantic.ValidationError, match="min_pct_daily_coverage is None"):
        GhiSufficiencySettings(daily_coverage={"baseline": True})


def test_enabling_a_rule_with_its_threshold_supplied_is_accepted():
    """Supplying the threshold alongside the toggle switches the rule on."""
    settings = GhiSufficiencySettings(
        daily_coverage={"baseline": True},
        min_pct_daily_coverage=0.9,
    )

    assert settings.daily_coverage.baseline is True


def test_misspelled_toggle_field_is_rejected():
    """A typo in a toggle name fails at construction rather than silently doing nothing."""
    with pytest.raises(pydantic.ValidationError, match="bounary_gap"):
        HourlyWholeFrameSufficiencySettings(bounary_gap={"baseline": True})


def test_misspelled_threshold_field_is_rejected():
    """A typo in a threshold name fails at construction."""
    with pytest.raises(pydantic.ValidationError, match="min_pct_dayly_coverage"):
        TemperatureSufficiencySettings(min_pct_dayly_coverage=0.9)


def test_ghi_coverage_and_unique_value_rules_default_off():
    """GHI is judged on monthly coverage alone."""
    settings = GhiSufficiencySettings()

    assert (settings.daily_coverage.baseline, settings.daily_coverage.reporting) == (False, False)
    assert (settings.unique_values.baseline, settings.unique_values.reporting) == (False, False)


def test_dump_and_rebuild_preserves_a_disabled_rule():
    """A rule switched off survives a settings round trip."""
    settings = DailyDataSufficiencySettings(
        rules={"negative_observed_values": {"baseline": False, "reporting": False}},
    )

    rebuilt = DailyDataSufficiencySettings(**settings.model_dump())

    assert rebuilt.rules.negative_observed_values.baseline is False
    assert rebuilt.rules.negative_observed_values.reporting is False
    assert rebuilt.rules.baseline_day_length.baseline is True


PRIOR_FORMAT_HOURLY_PAYLOAD = {
    "data": {
        "sufficiency": {
            "temperature": {
                "min_pct_hourly_coverage": 0.5,
                "min_pct_daily_coverage": 0.9,
                "min_pct_monthly_coverage": 0.9,
                "min_pct_period_coverage": 0.9,
                "min_pct_unique_values": None,
            },
            "ghi": {
                "min_pct_hourly_coverage": None,
                "min_pct_daily_coverage": None,
                "min_pct_monthly_coverage": 0.9,
                "min_pct_period_coverage": None,
                "min_pct_unique_values": None,
            },
            "observed": {
                "min_pct_hourly_coverage": 0.5,
                "min_pct_daily_coverage": 0.9,
                "min_pct_monthly_coverage": 0.9,
                "min_pct_period_coverage": None,
                "min_pct_unique_values": 0.10,
            },
            "joint": {
                "min_pct_hourly_coverage": None,
                "min_pct_daily_coverage": 0.9,
                "min_pct_monthly_coverage": None,
                "min_pct_period_coverage": None,
                "min_pct_unique_values": None,
            },
        },
    },
}


def test_strip_drops_every_key_the_schema_does_not_declare():
    """Only the thresholds declared on each block survive the strip."""
    stripped = strip_removed_sufficiency_keys(PRIOR_FORMAT_HOURLY_PAYLOAD)

    sufficiency = stripped["data"]["sufficiency"]
    surviving = {block: sorted(keys) for block, keys in sufficiency.items()}

    assert surviving == {
        "temperature": [
            "min_pct_daily_coverage",
            "min_pct_monthly_coverage",
            "min_pct_period_coverage",
            "min_pct_unique_values",
        ],
        "ghi": [
            "min_pct_daily_coverage",
            "min_pct_monthly_coverage",
            "min_pct_unique_values",
        ],
        "observed": [
            "min_pct_daily_coverage",
            "min_pct_hourly_coverage",
            "min_pct_monthly_coverage",
            "min_pct_unique_values",
        ],
        "joint": [
            "min_pct_daily_coverage",
            "min_pct_monthly_coverage",
            "min_pct_unique_values",
        ],
    }


def test_stripped_prior_format_payload_rebuilds_hourly_sufficiency():
    """A prior-format payload carrying undeclared threshold keys loads once stripped."""
    stripped = strip_removed_sufficiency_keys(PRIOR_FORMAT_HOURLY_PAYLOAD)

    settings = HourlyDataSufficiencySettings(**stripped["data"]["sufficiency"])

    assert settings.observed.min_pct_hourly_coverage == 0.5


# ---------------------------------------------------------------------------
# settings schema against the rule registry
# ---------------------------------------------------------------------------

def _expected_toggle_paths(family):
    """Dotted toggle paths the registry says this family must expose."""
    paths = set()
    for name, spec in RULE_SPECS.items():
        if not spec.toggleable or family not in spec.families:
            continue

        if spec.columns:
            for column in spec.columns:
                paths.add(f"{column}.{name}")
        else:
            paths.add(f"rules.{name}")

    return paths


@pytest.mark.parametrize("family", sorted(SETTINGS_BY_FAMILY))
def test_settings_expose_a_toggle_for_every_registered_rule(family):
    """Each family declares a toggle for every registered rule it can run, read from
    RULE_SPECS. A column the family does not carry (ghi outside hourly) is excluded, since
    the settings object has no block to hang the toggle on.
    """
    settings = SETTINGS_BY_FAMILY[family]()
    carried_blocks = {
        block for block in ("rules", *COLUMN_BLOCKS) if getattr(settings, block) is not None
    }
    expected = {
        path for path in _expected_toggle_paths(family) if path.split(".")[0] in carried_blocks
    }

    assert set(toggle_paths(settings)) == expected


def test_every_registered_spec_is_bound_to_a_rule_function():
    """No spec is declared without an implementation behind it."""
    bound = set(registered_rules())

    assert bound == set(RULE_SPECS)
