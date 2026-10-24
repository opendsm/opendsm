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

from opendsm.eemeter.common.sufficiency_registry import RULE_SPECS
from opendsm.eemeter.common.sufficiency_rules import RULE_FUNCTIONS, _check_rule_binding
from opendsm.eemeter.common.data_settings import (
    AdvisoryRuleToggle,
    BaseSufficiencySettings,
    BillingDataSufficiencySettings,
    DailyDataSufficiencySettings,
    GhiSufficiencySettings,
    HourlyDataSufficiencySettings,
    HourlyWholeFrameSufficiencySettings,
    JointSufficiencySettings,
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


def test_sufficiency_settings_accept_uppercase_kwargs_over_defaults():
    """An uppercase keyword overrides the lowercase default of the same setting."""
    settings = TemperatureSufficiencySettings(MIN_PCT_DAILY_COVERAGE=0.5)

    assert settings.min_pct_daily_coverage == 0.5


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


def test_enabling_a_phase_the_rule_never_runs_in_is_rejected():
    """extreme_values is baseline-only, so enabling its reporting phase fails."""
    with pytest.raises(pydantic.ValidationError, match="never runs in the reporting phase"):
        WholeFrameSufficiencySettings(extreme_values={"baseline": True, "reporting": True})


def test_enabling_a_rule_without_its_threshold_is_rejected():
    """A column rule cannot be switched on while its threshold is None."""
    with pytest.raises(pydantic.ValidationError, match="min_pct_unique_values is None"):
        GhiSufficiencySettings(unique_values={"baseline": True})


def test_enabling_a_rule_with_its_threshold_supplied_is_accepted():
    """Supplying the threshold alongside the toggle switches the rule on."""
    settings = GhiSufficiencySettings(
        unique_values={"baseline": True},
        min_pct_unique_values=0.1,
    )

    assert settings.unique_values.baseline is True


@pytest.mark.parametrize(
    "block, rule",
    [
        (GhiSufficiencySettings, "daily_coverage"),
        (JointSufficiencySettings, "unique_values"),
        (JointSufficiencySettings, "monthly_coverage"),
    ],
)
def test_a_toggle_for_a_rule_the_column_has_no_check_for_is_rejected(block, rule):
    """A column block carries toggles only for the rules that judge that column."""
    with pytest.raises(pydantic.ValidationError, match=rule):
        block(**{rule: {"baseline": True}})


def test_misspelled_toggle_field_is_rejected():
    """A typo in a toggle name fails at construction rather than silently doing nothing."""
    with pytest.raises(pydantic.ValidationError, match="bounary_gap"):
        HourlyWholeFrameSufficiencySettings(bounary_gap={"baseline": True})


def test_misspelled_threshold_field_is_rejected():
    """A typo in a threshold name fails at construction."""
    with pytest.raises(pydantic.ValidationError, match="min_pct_dayly_coverage"):
        TemperatureSufficiencySettings(min_pct_dayly_coverage=0.9)


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
    """A payload carrying undeclared threshold keys loads once stripped."""
    stripped = strip_removed_sufficiency_keys(PRIOR_FORMAT_HOURLY_PAYLOAD)

    settings = HourlyDataSufficiencySettings(**stripped["data"]["sufficiency"])

    assert settings.observed.min_pct_hourly_coverage == 0.5


@pytest.mark.parametrize("family", list(SETTINGS_BY_FAMILY))
def test_stripped_payload_with_a_requested_window_builds_sufficiency(family):
    """requested_start and requested_end at the top of the sufficiency block are stripped
    with an undeclared column-block threshold, and what remains builds the family's
    settings."""
    payload = {
        "data": {
            "sufficiency": {
                "requested_start": "2020-01-01T00:00:00+00:00",
                "requested_end": "2020-12-31T00:00:00+00:00",
                "min_baseline_length": 300,
                "observed": {"min_pct_period_coverage": None, "min_pct_daily_coverage": 0.8},
            },
        },
    }

    stripped = strip_removed_sufficiency_keys(payload)
    settings = SETTINGS_BY_FAMILY[family](**stripped["data"]["sufficiency"])

    assert settings.min_baseline_length == 300
    assert settings.observed.min_pct_daily_coverage == 0.8


@pytest.mark.parametrize("key", ["requested_start", "requested_end"])
def test_sufficiency_settings_reject_a_requested_window_bound(key):
    """The sufficiency settings declare no requested window, so either bound is rejected."""
    with pytest.raises(pydantic.ValidationError, match=key):
        DailyDataSufficiencySettings(**{key: "2020-01-01"})


# Rule toggles


def _expected_toggle_paths(family):
    """Dotted toggle paths RULE_SPECS says this family must expose."""
    paths = set()
    for name, spec in RULE_SPECS.items():
        if spec.severity == "infeasible" or family not in spec.families:
            continue

        if spec.columns:
            for column in spec.columns:
                paths.add(f"{column}.{name}")
        else:
            paths.add(f"rules.{name}")

    return paths


@pytest.mark.parametrize("family", sorted(SETTINGS_BY_FAMILY))
def test_settings_expose_a_toggle_for_every_registered_rule(family):
    """Each family declares a toggle for every rule in RULE_SPECS it can run, and for no
    other rule. A column the family does not carry (ghi outside hourly) is excluded, since
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


@pytest.mark.parametrize("family", sorted(SETTINGS_BY_FAMILY))
def test_each_toggle_class_follows_its_rule_severity(family):
    """An advisory rule's toggle is the user-tier AdvisoryRuleToggle; every other rule's toggle
    is the developer-tier RuleToggle."""
    settings = SETTINGS_BY_FAMILY[family]()

    mismatched = {}
    for path in toggle_paths(settings):
        block_name, name = path.split(".")
        toggle_class = type(getattr(getattr(settings, block_name), name))
        if RULE_SPECS[name].severity == "advisory":
            expected_class = AdvisoryRuleToggle
        else:
            expected_class = RuleToggle

        if toggle_class is not expected_class:
            mismatched[path] = (toggle_class.__name__, expected_class.__name__)

    assert mismatched == {}, f"(declared, expected) toggle class by path: {mismatched}"


def test_rule_binding_check_accepts_the_declared_tables():
    """The shipped spec and function tables name the same rules."""
    _check_rule_binding(RULE_SPECS, RULE_FUNCTIONS)


def test_rule_binding_check_raises_on_a_spec_without_a_function():
    """A declared rule with no function behind it is named in the error."""
    functions = {name: function for name, function in RULE_FUNCTIONS.items() if name != "no_data"}

    with pytest.raises(ValueError, match=r"specs without a function \['no_data'\]"):
        _check_rule_binding(RULE_SPECS, functions)


def test_rule_binding_check_raises_on_a_function_without_a_spec():
    """A function bound under an undeclared name is named in the error."""
    functions = {**RULE_FUNCTIONS, "not_a_rule": RULE_FUNCTIONS["no_data"]}

    with pytest.raises(ValueError, match=r"functions without a spec \['not_a_rule'\]"):
        _check_rule_binding(RULE_SPECS, functions)
