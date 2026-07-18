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

"""Validator tests for individual-meter-matching settings."""

import pytest

from pydantic import ValidationError

from opendsm.comparison_groups.individual_meter_matching.settings import Settings
from opendsm.comparison_groups.individual_meter_matching.highs_settings import HiGHS_Settings



class TestIMMSettings:
    """IMM Settings field bounds and cross-field validation."""

    def test_default_constructs(self):
        """Defaults construct without error."""
        settings = Settings()
        assert settings.n_matches_per_treatment == 4

    def test_duplicates_require_meter_distance(self):
        """allow_duplicate_matches=True is only valid with meter-distance selection."""
        with pytest.raises(ValidationError, match="minimize_meter_distance"):
            Settings(
                allow_duplicate_matches=True,
                selection_method="minimize_loadshape_distance",
            )

    def test_duplicates_with_meter_distance_ok(self):
        """The same flag is accepted with the required selection method."""
        settings = Settings(
            allow_duplicate_matches=True,
            selection_method="minimize_meter_distance",
        )
        assert settings.allow_duplicate_matches is True

    @pytest.mark.parametrize("field,value", [
        ("n_matches_per_treatment", 0),
        ("n_pool_meters_per_chunk", 0),
        ("candidate_multiplier", 1),
    ])
    def test_below_minimum_raises(self, field, value):
        """Count fields enforce their lower bounds."""
        with pytest.raises(ValidationError):
            Settings(**{field: value})

    def test_bad_selection_method_enum_raises(self):
        """An unknown selection method is rejected."""
        with pytest.raises(ValidationError):
            Settings(selection_method="nearest_vibes")


class TestHiGHSSettings:
    """HiGHS solver settings bounds and literal enums."""

    def test_default_constructs(self):
        """Defaults construct without error."""
        settings = HiGHS_Settings()
        assert settings.time_limit == float("inf")

    def test_negative_time_limit_raises(self):
        """time_limit must be >= 0."""
        with pytest.raises(ValidationError):
            HiGHS_Settings(time_limit=-1.0)

    def test_infinite_cost_below_floor_raises(self):
        """infinite_cost has a 1e15 floor."""
        with pytest.raises(ValidationError):
            HiGHS_Settings(infinite_cost=1e10)

    def test_small_matrix_value_below_floor_raises(self):
        """small_matrix_value has a 1e-12 floor."""
        with pytest.raises(ValidationError):
            HiGHS_Settings(small_matrix_value=1e-15)

    @pytest.mark.parametrize("field", ["presolve", "parallel", "run_crossover"])
    def test_bad_literal_option_raises(self, field):
        """on/off/choose literals reject arbitrary strings."""
        with pytest.raises(ValidationError):
            HiGHS_Settings(**{field: "maybe"})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
