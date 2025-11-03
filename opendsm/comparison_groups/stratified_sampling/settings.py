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

import opendsm.comparison_groups.stratified_sampling.const as _const
from opendsm.common.base_settings import BaseSettings

from typing import Optional, Literal, Union


class StratificationColumnSettings(BaseSettings):
    """column name to use for stratification"""
    column_name: str = pydantic.Field()

    """fixed number of bins to use for stratification"""
    n_bins: Optional[int] = pydantic.Field(
        default=8, 
        ge=2, 
        validate_default=True,
    )

    """minimum treatment value used to construct bins (used to remove outliers)"""
    min_value_allowed: int = pydantic.Field(
        default=3000, 
        ge=0, 
        validate_default=True,
    )

    """maximum treatment value used to construct bins (used to remove outliers)"""
    max_value_allowed: int = pydantic.Field(
        default=6000, 
        ge=0, 
        validate_default=True,
    )

    """whether to use fixed width bins or fixed proportion bins"""
    is_fixed_width: bool = pydantic.Field(
        default=False, 
    )

    """column requires equivalence when auto-binning"""
    auto_bin_equivalence: Literal[False] = False


class DSS_StratificationColumnSettings(StratificationColumnSettings):
    """fixed number of bins to use for stratification"""
    n_bins: Literal[None] = None

    """column requires equivalence when auto-binning"""
    auto_bin_equivalence: Literal[True] = True


class Settings(BaseSettings):
    """
    min_n_sampled_to_n_treatment_ratio: int
        TODO: FILL THIS OUT
    seed: int
        Seed for random number generator
    """

    min_n_treatment_per_bin: int = pydantic.Field(
        default=0, 
        ge=0, 
        validate_default=True,
    )

    seed: int = pydantic.Field(
        default=42, 
        ge=0, 
        validate_default=True,
    )


class StratifiedSamplingSettings(Settings):
    """
    n_samples_approx: int
        approximate number of total samples from df_pool. It is approximate because
        there may be some slight discrepancies around the total count to ensure
        that each bin has the correct percentage of the total.
    min_n_treatment_per_bin: int
        minimum number of treatment samples that must exist in a given bin for 
        it to be considered a non-outlier bin (only applicable if there are 
        cols with fixed_width=True)
    min_n_sampled_to_n_treatment_ratio: int
    relax_n_samples_approx_constraint: bool
        If True, treats n_samples_approx as an upper bound, but gets as many comparison group
        meters as available up to n_samples_approx. if false, it raises an exception
        if there are not enough comparison pool meters to reach n_samples_approx.
    """

    n_samples_approx: Optional[int] = pydantic.Field(
        default=None, 
        ge=1, 
        validate_default=True,
    )

    relax_n_samples_approx_constraint: bool = pydantic.Field(
        default=False, 
    )

    equivalence_method: Literal[None] = None

    equivalence_quantile: Literal[None] = None

    min_n_bins: Literal[None] = None

    max_n_bins: Literal[None] = None

    min_n_sampled_to_n_treatment_ratio: float = pydantic.Field(
        default=4, 
        ge=0, 
        validate_default=True,
    )

    stratification_column: Union[list[StratificationColumnSettings], list[dict]] = pydantic.Field(
        default=[
            StratificationColumnSettings(column_name="summer_usage"),
            StratificationColumnSettings(column_name="winter_usage"),
        ],
    )

    """set stratification column classes with given dictionaries"""
    @pydantic.model_validator(mode="after")
    def _set_nested_classes(self):
        if len(self.stratification_column) > 3:
            raise ValueError("a maximum of 3 stratification_column's are allowed")

        strat_settings = []
        has_dict = False
        for strat_item in self.stratification_column:
            if isinstance(strat_item, dict):
                has_dict = True
                strat_class = StratificationColumnSettings(**strat_item)

            else:
                strat_class = strat_item

            strat_settings.append(strat_class)

        if has_dict:
            self.stratification_column = strat_settings

        return self


# subclass Settings to change default values
class DistanceStratifiedSamplingSettings(Settings):
    """
    n_samples_approx: int
        approximate number of total samples from df_pool. It is approximate because
        there may be some slight discrepancies around the total count to ensure
        that each bin has the correct percentage of the total.
    min_n_treatment_per_bin: int
        Minimum number of treatment samples that must exist in a given bin for 
        it to be considered a non-outlier bin (only applicable if there are 
        cols with fixed_width=True)
    min_n_sampled_to_n_treatment_ratio: int
    relax_n_samples_approx_constraint: bool
        If True, treats n_samples_approx as an upper bound, but gets as many comparison group
        meters as available up to n_samples_approx. If False, it raises an exception
        if there are not enough comparison pool meters to reach n_samples_approx.
    """
    
    n_samples_approx: Optional[int] = pydantic.Field(
        default=5000, 
        ge=1, 
        validate_default=True,
    )

    relax_n_samples_approx_constraint: bool = pydantic.Field(
        default=True, 
    )

    equivalence_method: _const.DistanceMetric = pydantic.Field(
        default=_const.DistanceMetric.CHISQUARE,
        validate_default=True,
    )

    equivalence_quantile: int = pydantic.Field(
        default=25,
        validate_default=True,
    )

    min_n_bins: int = pydantic.Field(
        default=1, 
        ge=1, 
        validate_default=True,
    )

    max_n_bins: int = pydantic.Field(
        default=8, 
        ge=2, 
        validate_default=True,
    )

    min_n_sampled_to_n_treatment_ratio: float = pydantic.Field(
        default=0.25, 
        ge=0, 
        validate_default=True,
    )

    stratification_column: Union[list[DSS_StratificationColumnSettings], list[dict]] = pydantic.Field(
        default=[
            DSS_StratificationColumnSettings(column_name="summer_usage"),
            DSS_StratificationColumnSettings(column_name="winter_usage"),
        ],
    )

    """set stratification column classes with given dictionaries"""
    @pydantic.model_validator(mode="after")
    def _set_nested_classes(self):
        if len(self.stratification_column) > 3:
            raise ValueError("A maximum of 3 stratification_column's are allowed")

        strat_settings = []
        has_dict = False
        for strat_item in self.stratification_column:
            if isinstance(strat_item, dict):
                has_dict = True
                strat_class = DSS_StratificationColumnSettings(**strat_item)

            else:
                strat_class = strat_item

            strat_settings.append(strat_class)

        if has_dict:
            self.stratification_column = strat_settings

        return self


if __name__ == "__main__":
    s = StratifiedSamplingSettings()
    # s = DistanceStratifiedSamplingSettings()

    print(s.model_dump_json())