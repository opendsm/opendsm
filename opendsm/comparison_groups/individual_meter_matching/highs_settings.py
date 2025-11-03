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

from opendsm.common.base_settings import BaseSettings

from typing import Optional, Literal


# system maximum float
MIN_FLOAT = np.finfo(np.float64).tiny
MAX_FLOAT = np.finfo(np.float64).max
    

class HiGHS_Settings(BaseSettings):
    """Settings for HiGHS optimization solver"""

    """Presolve option"""
    presolve: Literal["off", "choose", "on"] = pydantic.Field(
        default="choose",
        validate_default=True,
    )

    """If 'simplex'/'ipm'/'pdlp' is chosen then, for a MIP (QP) the integrality constraint (quadratic term) will be ignored"""
    # solver: Literal["simplex", "choose", "ipm", "pdlp"] = pydantic.Field(
    #     default="choose",
    #     validate_default=True,
    # )

    """Parallel option"""
    parallel: Literal["off", "choose", "on"] = pydantic.Field(
        default="off", # was "choose"
        validate_default=True,
    )

    """Run IPM crossover"""
    run_crossover: Literal["off", "choose", "on"] = pydantic.Field(
        default="on",
        validate_default=True,
    )

    """Time limit (seconds)"""
    time_limit: float = pydantic.Field(
        default=float('inf'),
        ge=0,
        le=float('inf'),
        validate_default=True,
    )

    """Compute cost, bound, RHS and basic solution ranging"""
    ranging: Literal["off", "on"] = pydantic.Field(
        default="off",
        validate_default=True,
    )

    """Limit on |cost coefficient|: values greater than or equal to this will be treated as infinite"""
    infinite_cost: float = pydantic.Field(
        default=1e+20,
        ge=1e+15,
        le=float('inf'),
        validate_default=True,
    )

    """Limit on |constraint bound|: values greater than or equal to this will be treated as infinite"""
    infinite_bound: float = pydantic.Field(
        default=1e+20,
        ge=1e+15,
        le=float('inf'),
        validate_default=True,
    )

    """Lower limit on |matrix entries|: values less than or equal to this will be treated as zero"""
    small_matrix_value: float = pydantic.Field(
        default=1e-09,
        ge=1e-12,
        le=float('inf'),
        validate_default=True,
    )

    """Upper limit on |matrix entries|: values greater than or equal to this will be treated as infinite"""
    large_matrix_value: float = pydantic.Field(
        default=1e+15,
        ge=1,
        le=float('inf'),
        validate_default=True,
    )

    """Primal feasibility tolerance"""
    primal_feasibility_tolerance: float = pydantic.Field(
        default=1e-07,
        ge=1e-10,
        le=float('inf'),
        validate_default=True,
    )

    """Dual feasibility tolerance"""
    dual_feasibility_tolerance: float = pydantic.Field(
        default=1e-07,
        ge=1e-10,
        le=float('inf'),
        validate_default=True,
    )

    """IPM optimality tolerance"""
    ipm_optimality_tolerance: float = pydantic.Field(
        default=1e-08,
        ge=1e-12,
        le=float('inf'),
        validate_default=True,
    )

    """Objective bound for termination of the dual simplex solver"""
    objective_bound: float = pydantic.Field(
        default=float('inf'),
        ge=float('-inf'),
        le=float('inf'),
        validate_default=True,
    )

    """Objective target for termination of the MIP solver"""
    objective_target: float = pydantic.Field(
        default=float('-inf'),
        ge=float('-inf'),
        le=float('inf'),
        validate_default=True,
    )

    """Random seed used in HiGHS"""
    random_seed: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Number of threads used by HiGHS (0: automatic)"""
    threads: int = pydantic.Field(
        default=0,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Exponent of power-of-two bound scaling for model"""
    user_bound_scale: int = pydantic.Field(
        default=0,
        ge=-2147483647,
        le=2147483647,
        validate_default=True,
    )

    """Exponent of power-of-two cost scaling for model"""
    user_cost_scale: int = pydantic.Field(
        default=0,
        ge=-2147483647,
        le=2147483647,
        validate_default=True,
    )

    """Strategy for simplex solver [0: Choose; 1: Dual (serial); 2: Dual (PAMI); 3: Dual (SIP); 4: Primal]"""
    simplex_strategy: int = pydantic.Field(
        default=1,
        ge=0,
        le=4,
        validate_default=True,
    )

    """Simplex scaling strategy: [0: off; 1: choose; 2: equilibration; 3: forced equilibration; 4: max value 0; 5: max value 1]"""
    simplex_scale_strategy: int = pydantic.Field(
        default=1,
        ge=0,
        le=5,
        validate_default=True,
    )

    """Strategy for simplex dual edge weights: [-1: Choose; 0: Dantzig; 1: Devex; 2: Steepest Edge]"""
    simplex_dual_edge_weight_strategy: int = pydantic.Field(
        default=-1,
        ge=-1,
        le=2,
        validate_default=True,
    )

    """Strategy for simplex primal edge weights: [-1: Choose; 0: Dantzig; 1: Devex; 2: Steepest Edge]"""
    simplex_primal_edge_weight_strategy: int = pydantic.Field(
        default=-1,
        ge=-1,
        le=2,
        validate_default=True,
    )

    """Iteration limit for simplex solver when solving LPs, but not subproblems in the MIP solver"""
    simplex_iteration_limit: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Limit on the number of simplex UPDATE operations"""
    simplex_update_limit: int = pydantic.Field(
        default=5000,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Maximum level of concurrency in parallel simplex"""
    simplex_max_concurrency: int = pydantic.Field(
        default=8,
        ge=1,
        le=8,
        validate_default=True,
    )

    """Enables or disables solver output"""
    # output_file: bool = pydantic.Field(
    #     default=True,
    #     validate_default=True,
    # )

    """Enables or disables console logging"""
    # log_to_console: bool = pydantic.Field(
    #     default=True,
    #     validate_default=True,
    # )

    """Solution file"""
    solution_file: str = pydantic.Field(
        default="",
        validate_default=True,
    )

    """Log file"""
    log_file: str = pydantic.Field(
        default="",
        validate_default=True,
    )

    """Write the primal and dual solution to a file"""
    write_solution_to_file: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """Style of solution file: [0: HiGHS raw; 1: HiGHS pretty; 2: Glpsol raw; 3: Glpsol pretty; 4: HiGHS sparse raw] (raw = computer-readable, pretty = human-readable)"""
    write_solution_style: int = pydantic.Field(
        default=0,
        ge=0,
        le=4,
        validate_default=True,
    )

    """Location of cost row for Glpsol file: -2 => Last; -1 => None; 0 => None if empty, otherwise data file location; 1 <= n <= num_row => Location n; n > num_row => Last"""
    glpsol_cost_row_location: int = pydantic.Field(
        default=0,
        ge=-2,
        le=2147483647,
        validate_default=True,
    )

    """Write model file"""
    write_model_file: str = pydantic.Field(
        default="",
        validate_default=True,
    )

    """Write the model to a file"""
    write_model_to_file: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """Whether MIP symmetry should be detected"""
    mip_detect_symmetry: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    """Whether MIP restart is permitted"""
    mip_allow_restart: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    """MIP solver max number of nodes"""
    mip_max_nodes: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """MIP solver max number of nodes where estimate is above cutoff bound"""
    mip_max_stall_nodes: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Whether improving MIP solutions should be saved"""
    mip_improving_solution_save: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """Whether improving MIP solutions should be reported in sparse format"""
    mip_improving_solution_report_sparse: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """File for reporting improving MIP solutions: not reported for an empty string ''"""
    mip_improving_solution_file: str = pydantic.Field(
        default="",
        validate_default=True,
    )

    """MIP solver max number of leave nodes"""
    mip_max_leaves: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Limit on the number of improving solutions found to stop the MIP solver prematurely"""
    mip_max_improving_sols: int = pydantic.Field(
        default=2147483647,
        ge=1,
        le=2147483647,
        validate_default=True,
    )

    """Maximal age of dynamic LP rows before they are removed from the LP relaxation in the MIP solver"""
    mip_lp_age_limit: int = pydantic.Field(
        default=10,
        ge=0,
        le=32767,
        validate_default=True,
    )

    """Maximal age of rows in the MIP solver cutpool before they are deleted"""
    mip_pool_age_limit: int = pydantic.Field(
        default=30,
        ge=0,
        le=1000,
        validate_default=True,
    )

    """Soft limit on the number of rows in the MIP solver cutpool for dynamic age adjustment"""
    mip_pool_soft_limit: int = pydantic.Field(
        default=10000,
        ge=1,
        le=2147483647,
        validate_default=True,
    )

    """Minimal number of observations before MIP solver pseudo costs are considered reliable"""
    mip_pscost_minreliable: int = pydantic.Field(
        default=8,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Minimal number of entries in the MIP solver cliquetable before neighbourhood queries of the conflict graph use parallel processing"""
    mip_min_cliquetable_entries_for_parallelism: int = pydantic.Field(
        default=100000,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """MIP feasibility tolerance"""
    mip_feasibility_tolerance: float = pydantic.Field(
        default=1e-06,
        ge=1e-10,
        le=float('inf'),
        validate_default=True,
    )

    """Effort spent for MIP heuristics"""
    mip_heuristic_effort: float = pydantic.Field(
        default=0.05,
        ge=0,
        le=1,
        validate_default=True,
    )

    """Tolerance on relative gap, |ub-lb|/|ub|, to determine whether optimality has been reached for a MIP instance"""
    mip_rel_gap: float = pydantic.Field(
        default=0.0001,
        ge=0,
        le=float('inf'),
        validate_default=True,
    )

    """Tolerance on absolute gap of MIP, |ub-lb|, to determine whether optimality has been reached for a MIP instance"""
    mip_abs_gap: float = pydantic.Field(
        default=1e-06,
        ge=0,
        le=float('inf'),
        validate_default=True,
    )

    """MIP minimum logging interval"""
    mip_min_logging_interval: float = pydantic.Field(
        default=5,
        ge=0,
        le=float('inf'),
        validate_default=True,
    )

    """Iteration limit for IPM solver"""
    ipm_iteration_limit: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Use native termination for PDLP solver: Default = false"""
    pdlp_native_termination: bool = pydantic.Field(
        default=False,
        validate_default=True,
    )

    """Scaling option for PDLP solver: Default = true"""
    pdlp_scaling: bool = pydantic.Field(
        default=True,
        validate_default=True,
    )

    """Iteration limit for PDLP solver"""
    pdlp_iteration_limit: int = pydantic.Field(
        default=2147483647,
        ge=0,
        le=2147483647,
        validate_default=True,
    )

    """Restart mode for PDLP solver: 0 => none; 1 => GPU (default); 2 => CPU"""
    pdlp_e_restart_method: int = pydantic.Field(
        default=1,
        ge=0,
        le=2,
        validate_default=True,
    )

    """Duality gap tolerance for PDLP solver: Default = 1e-4"""
    pdlp_d_gap_tol: float = pydantic.Field(
        default=0.0001,
        ge=1e-12,
        le=float('inf'),
        validate_default=True,
    )


    """Make seed random if None"""
    @pydantic.model_validator(mode="after")
    def _random_seed(self):
        self.model_config["frozen"] = False

        if self.random_seed is None:
            try:
                min_int = self.model_fields["random_seed"].metadata[0].ge
                max_int = self.model_fields["random_seed"].metadata[1].le
            except:
                min_int = 0
                max_int = 2147483647

            self.random_seed = np.random.randint(min_int, max_int)

        self.model_config["frozen"] = True

        return self

if __name__ == "__main__":
    s = HiGHS_Settings()

    print(s.model_dump_json())
