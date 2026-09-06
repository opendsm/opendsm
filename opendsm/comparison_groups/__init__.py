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

"""Comparison groups: selection algorithms and the treatment-savings analysis
pipeline (populations, selection, correction, savings)."""

from opendsm.comparison_groups.cg_clustering import (
    CG_Clustering,
    CG_Clustering_Settings,
)
from opendsm.comparison_groups.individual_meter_matching import (
    IMM,
    IMM_Settings,
)
from opendsm.comparison_groups.stratified_sampling import (
    Stratified_Sampling,
    SS_Settings,
    DSS_Settings,
)
from opendsm.comparison_groups.random_sampling import (
    Random_Sampling,
    RS_Settings,
)
from opendsm.comparison_groups.common import (
    Data,
    Data_Settings,
)
from opendsm.comparison_groups.population import (
    ComparisonPool,
    MeterPopulation,
    TreatmentGroup,
)
from opendsm.comparison_groups.selection import (
    ComparisonGroupSelection,
    select_comparison_group,
)
from opendsm.comparison_groups.savings import (
    CGCorrectionSettings,
    CorrectionResult,
    SavingsAggregation,
    SavingsResult,
    compute_savings,
    correct_reporting,
    model_correction_matrix,
)
from opendsm.comparison_groups.analysis import MeterAnalysis



__all__ = [
    "CGCorrectionSettings",
    "CG_Clustering",
    "CG_Clustering_Settings",
    "MeterAnalysis",
    "ComparisonGroupSelection",
    "ComparisonPool",
    "CorrectionResult",
    "DSS_Settings",
    "Data",
    "Data_Settings",
    "IMM",
    "IMM_Settings",
    "MeterPopulation",
    "Random_Sampling",
    "RS_Settings",
    "SS_Settings",
    "SavingsAggregation",
    "SavingsResult",
    "Stratified_Sampling",
    "TreatmentGroup",
    "compute_savings",
    "correct_reporting",
    "model_correction_matrix",
    "select_comparison_group",
]
