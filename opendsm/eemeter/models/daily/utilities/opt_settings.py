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

from enum import Enum
from typing import Optional

from opendsm.common.base_settings import BaseSettings, CustomField


class AlgorithmChoice(str, Enum):
    # SciPy scalar optimization algorithms
    SCIPY_BRENT = "scipy_brent"
    SCIPY_BOUNDED = "scipy_bounded"
    SCIPY_GOLDEN = "scipy_golden"

    # SciPy local optimization algorithms
    SCIPY_NELDERMEAD = "scipy_nelder-mead"
    SCIPY_L_BFGS_B = "scipy_l-bfgs-b"
    SCIPY_TNC = "scipy_tnc"
    SCIPY_COBYLA = "scipy_cobyla"
    SCIPY_COBYQA = "scipy_cobyqa"
    SCIPY_SLSQP = "scipy_slsqp"
    SCIPY_POWELL = "scipy_powell"
    SCIPY_TRUST_CONSTR = "scipy_trust-constr"

    # SciPy global optimization algorithms
    SCIPY_DIRECT = "scipy_direct"

    # nlopt-based algorithms
    NLOPT_DIRECT = "nlopt_direct"
    NLOPT_DIRECT_NOSCAL = "nlopt_direct_noscal"
    NLOPT_DIRECT_L = "nlopt_direct_l"
    NLOPT_DIRECT_L_RAND = "nlopt_direct_l_rand"
    NLOPT_DIRECT_L_NOSCAL = "nlopt_direct_l_noscal"
    NLOPT_DIRECT_L_RAND_NOSCAL = "nlopt_direct_l_rand_noscal"
    NLOPT_ORIG_DIRECT = "nlopt_orig_direct"
    NLOPT_ORIG_DIRECT_L = "nlopt_orig_direct_l"
    NLOPT_CRS2_LM = "nlopt_crs2_lm"
    NLOPT_MLSL_LDS = "nlopt_mlsl_lds"
    NLOPT_MLSL = "nlopt_mlsl"
    NLOPT_STOGO = "nlopt_stogo"
    NLOPT_STOGO_RAND = "nlopt_stogo_rand"
    NLOPT_AGS = "nlopt_ags"
    NLOPT_ISRES = "nlopt_isres"
    NLOPT_ESCH = "nlopt_esch"
    NLOPT_COBYLA = "nlopt_cobyla"
    NLOPT_BOBYQA = "nlopt_bobyqa"
    NLOPT_NEWUOA = "nlopt_newuoa"
    NLOPT_NEWUOA_BOUND = "nlopt_newuoa_bound"
    NLOPT_PRAXIS = "nlopt_praxis"
    NLOPT_NELDERMEAD = "nlopt_neldermead"
    NLOPT_SBPLX = "nlopt_sbplx"
    NLOPT_MMA = "nlopt_mma"
    NLOPT_CCSAQ = "nlopt_ccsaq"
    NLOPT_SLSQP = "nlopt_slsqp"
    NLOPT_L_BFGS = "nlopt_lbfgs"
    NLOPT_TNEWTON = "nlopt_tnewton"
    NLOPT_TNEWTON_PRECOND = "nlopt_tnewton_precond"
    NLOPT_TNEWTON_RESTART = "nlopt_tnewton_restart"
    NLOPT_TNEWTON_PRECOND_RESTART = "nlopt_tnewton_precond_restart"
    NLOPT_VAR1 = "nlopt_var1"
    NLOPT_VAR2 = "nlopt_var2"


class StopCriteriaChoice(str, Enum):
    ITERATION_MAXIMUM = "iteration maximum"
    MAXIMUM_TIME = "maximum time [min]"


class OptimizationSettings(BaseSettings):
    algorithm: AlgorithmChoice = CustomField(
        default=AlgorithmChoice.NLOPT_SBPLX,
        description="Optimization algorithm choice",
    )

    stop_criteria_type: StopCriteriaChoice = CustomField(
        default=StopCriteriaChoice.ITERATION_MAXIMUM,
        description="Stopping criteria",
    )

    stop_criteria_value: float = CustomField(
        default=2000,
        gt=0,
        description="Stopping criteria value for the optimization algorithm",
    )
    
    initial_step: Optional[float] = CustomField(
        default=0.1,
        description="Initial step size for the optimization algorithm",
    )

    x_tol_rel: float = CustomField(
        default=1e-5,
        gt=0,
        description="Relative cutoff X tolerance for the optimization algorithm",
    )

    f_tol_rel: float = CustomField(
        default=1e-5,
        gt=0,
        description="Relative cutoff function tolerance for the optimization algorithm",
    )

    initial_population_multiplier: Optional[float] = CustomField(
        default=None,
        description="Initial population multiplier for the optimization algorithm",
    )


    @pydantic.model_validator(mode="after")
    def _check_population_multiplier(self):
        if self.initial_population_multiplier is None:
            if self.algorithm == AlgorithmChoice.NLOPT_ISRES:
                raise ValueError("INITIAL_POPULATION_MULTIPLIER must be > 1 for nlopt_ISRES")
        else:
            if self.algorithm == AlgorithmChoice.NLOPT_ISRES:
                if self.initial_population_multiplier <= 1:
                    raise ValueError("INITIAL_POPULATION_MULTIPLIER must be > 1 for nlopt_ISRES")
            else:
                raise ValueError("INITIAL_POPULATION_MULTIPLIER must be None for all algorithms except nlopt_ISRES")

        return self