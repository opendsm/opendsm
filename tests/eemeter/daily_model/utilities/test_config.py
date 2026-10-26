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

from opendsm.eemeter.models.daily.utilities.settings import (
    DailySettings,
    Season_Definition,
    Split_Selection_Definition,
    Weekday_Weekend_Definition,
    update_daily_settings,
)


def test_default_settings():
    settings = DailySettings()
    assert settings.preset == "current"
    assert settings.algorithm_choice.lower() == "nlopt_sbplx"
    assert settings.initial_guess_algorithm_choice.lower() == "nlopt_direct"
    assert settings.alpha_selection == 2.0
    assert settings.alpha_final == "adaptive"
    assert settings.alpha_final_type == "last"
    assert settings.regularization_alpha == 0.001
    assert settings.regularization_percent_lasso == 1.0
    assert settings.allow_smooth_model is True
    assert settings.split_selection.allow_separate_summer is True
    assert settings.split_selection.allow_separate_shoulder is True
    assert settings.split_selection.allow_separate_winter is True
    assert settings.split_selection.allow_separate_weekday_weekend is True
    assert settings.split_selection.reduce_splits_by_gaussian is True
    assert settings.segment_minimum_count == 6


def test_custom_settings():
    settings_dict = {
        "algorithm_choice": "scipy_SLSQP",
        "initial_guess_algorithm_choice": "nlopt_DIRECT_L",
        "alpha_selection": 1.5,
        "alpha_final": 1.5,
        "alpha_final_type": "last",
        "regularization_alpha": 0.01,
        "regularization_percent_lasso": 0.5,
        "allow_smooth_model": True,
        "split_selection": {
            "allow_separate_summer": True,
            "allow_separate_shoulder": True,
            "allow_separate_winter": True,
            "allow_separate_weekday_weekend": True,
            "reduce_splits_by_gaussian": True,
        },
        "segment_minimum_count": 20,
    }
    settings = DailySettings(**settings_dict)

    assert settings.algorithm_choice.lower() == "scipy_slsqp"
    assert settings.initial_guess_algorithm_choice.lower() == "nlopt_direct_l"
    assert settings.alpha_selection == 1.5
    assert settings.alpha_final == 1.5
    assert settings.alpha_final_type == "last"
    assert settings.regularization_alpha == 0.01
    assert settings.regularization_percent_lasso == 0.5
    assert settings.allow_smooth_model is True
    assert settings.split_selection.allow_separate_summer is True
    assert settings.split_selection.allow_separate_shoulder is True
    assert settings.split_selection.allow_separate_winter is True
    assert settings.split_selection.allow_separate_weekday_weekend is True
    assert settings.split_selection.reduce_splits_by_gaussian is True
    assert settings.segment_minimum_count == 20


def test_invalid_settings():
    with pytest.raises(ValueError):
        DailySettings(algorithm_choice="invalid_algorithm")
    with pytest.raises(ValueError):
        DailySettings(alpha_selection=-11)
    with pytest.raises(ValueError):
        DailySettings(alpha_selection=3)
    with pytest.raises(ValueError):
        DailySettings(alpha_final_type="invalid_type")


def test_legacy_preset_sets_legacy_defaults():
    settings = DailySettings(preset="legacy")

    assert settings.allow_smooth_model is False
    assert settings.alpha_final == 2.0
    assert settings.segment_minimum_count == 10
    assert settings.split_selection.allow_separate_summer is False
    assert settings.split_selection.allow_separate_shoulder is False
    assert settings.split_selection.allow_separate_winter is False
    assert settings.split_selection.allow_separate_weekday_weekend is False
    assert settings.split_selection.reduce_splits_by_gaussian is False
    assert settings.split_selection.reduce_splits_num_std is None


def test_explicit_value_overrides_preset_value():
    settings = DailySettings(preset="legacy", segment_minimum_count=7)

    assert settings.segment_minimum_count == 7


def test_nested_override_keeps_untouched_preset_values():
    settings = DailySettings(preset="legacy", split_selection={"penalty_multiplier": 0.5})

    assert settings.split_selection.penalty_multiplier == 0.5
    assert settings.split_selection.allow_separate_summer is False


def test_nested_override_as_instance_keeps_untouched_preset_values():
    settings = DailySettings(
        preset="legacy",
        split_selection=Split_Selection_Definition(penalty_multiplier=0.5),
    )

    assert settings.split_selection.penalty_multiplier == 0.5
    assert settings.split_selection.allow_separate_summer is False
    assert settings.split_selection.reduce_splits_by_gaussian is False
    assert settings.split_selection.reduce_splits_num_std is None


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        DailySettings(preset="not_a_preset")


def test_non_string_preset_raises_validation_error_naming_preset():
    with pytest.raises(pydantic.ValidationError, match="preset"):
        DailySettings(preset=["legacy"])


def test_uppercase_keys_apply_the_preset_and_the_nested_value():
    settings = DailySettings(
        **{"PRESET": "legacy", "SPLIT_SELECTION": {"PENALTY_MULTIPLIER": 0.5}}
    )

    assert settings.preset == "legacy"
    assert settings.segment_minimum_count == 10
    assert settings.split_selection.penalty_multiplier == 0.5
    assert settings.split_selection.allow_separate_summer is False


def test_two_spellings_of_one_key_raise_naming_the_key():
    with pytest.raises(ValueError, match="segment_minimum_count"):
        DailySettings(**{"segment_minimum_count": 7, "SEGMENT_MINIMUM_COUNT": 9})


def test_update_with_uppercase_keys_replaces_the_lowercase_fields():
    settings = DailySettings()

    updated = update_daily_settings(
        settings, {"ALPHA_FINAL_TYPE": None, "FINAL_BOUNDS_SCALAR": None}
    )

    assert updated.alpha_final_type is None
    assert updated.final_bounds_scalar is None


def test_unknown_key_is_dropped():
    settings = DailySettings(**{"not_a_setting": 1})

    assert "not_a_setting" not in settings.model_dump()
    assert not hasattr(settings, "not_a_setting")


def test_season_and_weekday_options_are_constants_not_settings():
    settings = DailySettings()
    settings_dict = settings.model_dump()

    assert "options" not in settings_dict["season"]
    assert "options" not in settings_dict["weekday_weekend"]
    assert Season_Definition.options == ["summer", "shoulder", "winter"]
    assert Weekday_Weekend_Definition.options == ["weekday", "weekend"]
