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

from timeit import default_timer as timer

import nlopt
import numpy as np
from scipy.optimize import (
    direct as scipy_direct,
    minimize as scipy_minimize,
    minimize_scalar as scipy_minimize_scalar,
)
from opendsm.eemeter.models.daily.optimize_results import OptimizedResult

nlopt_algorithms = {
    "nlopt_direct": nlopt.GN_DIRECT,
    "nlopt_direct_noscal": nlopt.GN_DIRECT_NOSCAL,
    "nlopt_direct_l": nlopt.GN_DIRECT_L,
    "nlopt_direct_l_rand": nlopt.GN_DIRECT_L_RAND,
    "nlopt_direct_l_noscal": nlopt.GN_DIRECT_L_NOSCAL,
    "nlopt_direct_l_rand_noscal": nlopt.GN_DIRECT_L_RAND_NOSCAL,
    "nlopt_orig_direct": nlopt.GN_ORIG_DIRECT,
    "nlopt_orig_direct_l": nlopt.GN_ORIG_DIRECT_L,
    "nlopt_crs2_lm": nlopt.GN_CRS2_LM,
    "nlopt_mlsl_lds": nlopt.G_MLSL_LDS,
    "nlopt_mlsl": nlopt.G_MLSL,
    "nlopt_stogo": nlopt.GD_STOGO,
    "nlopt_stogo_rand": nlopt.GD_STOGO_RAND,
    "nlopt_ags": nlopt.GN_AGS,
    "nlopt_isres": nlopt.GN_ISRES,
    "nlopt_esch": nlopt.GN_ESCH,
    "nlopt_cobyla": nlopt.LN_COBYLA,
    "nlopt_bobyqa": nlopt.LN_BOBYQA,
    "nlopt_newuoa": nlopt.LN_NEWUOA,
    "nlopt_newuoa_bound": nlopt.LN_NEWUOA_BOUND,
    "nlopt_praxis": nlopt.LN_PRAXIS,
    "nlopt_neldermead": nlopt.LN_NELDERMEAD,
    "nlopt_sbplx": nlopt.LN_SBPLX,
    "nlopt_mma": nlopt.LD_MMA,
    "nlopt_ccsaq": nlopt.LD_CCSAQ,
    "nlopt_slsqp": nlopt.LD_SLSQP,
    "nlopt_lbfgs": nlopt.LD_LBFGS,
    "nlopt_tnewton": nlopt.LD_TNEWTON,
    "nlopt_tnewton_precond": nlopt.LD_TNEWTON_PRECOND,
    "nlopt_tnewton_restart": nlopt.LD_TNEWTON_RESTART,
    "nlopt_tnewton_precond_restart": nlopt.LD_TNEWTON_PRECOND_RESTART,
    "nlopt_var1": nlopt.LD_VAR1,
    "nlopt_var2": nlopt.LD_VAR2,
}

nlopt_algorithms = {k.lower(): v for k, v in nlopt_algorithms.items()}

pos_msg = [
    "Optimization terminated successfully.",
    "Optimization terminated: Stop Value was reached.",
    "Optimization terminated: Function tolerance was reached.",
    "Optimization terminated: X tolerance was reached.",
    "Optimization terminated: Max number of evaluations was reached.",
    "Optimization terminated: Max time was reached.",
]
neg_msg = [
    "Optimization failed",
    "Optimization failed: Invalid arguments given",
    "Optimization failed: Out of memory",
    "Optimization failed: Roundoff errors limited progress",
    "Optimization failed: Forced termination",
]


def obj_fcn_dec(obj_fcn, x0, bnds):
    """
    Returns a function that evaluates the objective function with the given bounds.

    Args:
    - obj_fcn: the objective function to be evaluated
    - x0: the initial guess for the optimization
    - bnds: the bounds for the optimization

    Returns:
    - obj_fcn_eval: a function that evaluates the objective function with the given bounds
    - idx_opt: the indices of the variables with non-equal bounds
    """

    idx_opt = [n for n in range(np.shape(bnds)[0]) if (bnds[n, 0] < bnds[n, 1])]

    def obj_fcn_eval(
        x, *args, **kwargs
    ):  # only modify x0 where it has bounds which are not the same
        x0[idx_opt] = x

        return obj_fcn(x0, *args, **kwargs)

    return obj_fcn_eval, idx_opt


class BaseOptimizedResult:
    x = None
    success = None
    status = None
    message = None
    fun = None
    jac = None
    hess = None
    hess_inv = None
    nfev = None
    njev = None
    nhev = None
    nit = None
    maxcv = None
    time_elapsed = None


class BaseOptimizer:
    def __init__(self, obj_fcn, x0, bnds, settings):
        """
        The constructor for the Optimizer class.

        Parameters:
            obj_fcn (function): The objective function to be optimized.
            x0 (np.array): The initial guess for the optimization.
            bnds (list): The bounds for the optimization.
            coef_id (str): The identifier for the coefficient.
            settings (dict): The settings for the optimization.
            opt_settings (Opt_Settings): The settings for the optimization.
        """
        self.bnds = np.array(bnds)
        self.x0 = np.clip(
            x0, bnds[:, 0], bnds[:, 1]
        )  # clip x0 to the bnds, just in case

        self.obj_fcn, self.idx_opt = obj_fcn_dec(obj_fcn, x0, bnds)

        self.settings = settings


class SciPyOptimizer(BaseOptimizer):
    def run(self):
        """
        Optimize the objective function using the SciPy library. Different optimization options are available,
        such as scipy_COBYLA, scipy_SLSQP, scipy_L_BFGS_B, scipy_TNC, scipy_BFGS, scipy_Powell, scipy_Nelder-Mead.
        options argument needs to have the algorithm specified.

        Args:
            x0 (list): Initial guess for the optimization.
            bnds (tuple): Bounds for the optimization.
            settings (Opt_Settings): The settings for the optimization.

        Returns:
            res_out (OptimizedResult): An object containing the results of the optimization.
        """

        settings = self.settings
        x0 = self.x0
        bnds = self.bnds

        timer_start = timer()

        algorithm = settings.algorithm[6:]

        if algorithm.lower() in ["brent", "golden", "bounded"]:
            scipy_obj_fcn = lambda x: self.obj_fcn([x])

            if algorithm.lower() in ["brent", "golden"]:
                res = scipy_minimize_scalar(
                    scipy_obj_fcn, bracket=bnds, method=algorithm.lower()
                )

            elif algorithm.lower() == "bounded":
                res = scipy_minimize_scalar(
                    scipy_obj_fcn, bounds=bnds[0], method="bounded"
                )

            res.x = [res.x]

        else:
            x0_opt = x0[self.idx_opt]
            bnds_opt = bnds[self.idx_opt, :]
            bnds_opt = tuple(map(tuple, bnds_opt))

            scipy_obj_fcn = lambda x: self.obj_fcn(x)

            if algorithm.lower() == "direct":
                res = scipy_direct(
                    scipy_obj_fcn, 
                    bnds_opt,
                    maxiter=int(settings.stop_criteria_value),
                    f_min_rtol=settings.f_tol_rel,
                )
            else:
                res = scipy_minimize(
                    scipy_obj_fcn, x0_opt, method=algorithm, bounds=bnds_opt
                )

        res.time_elapsed = timer() - timer_start

        return res
    

class NLoptOptimizer(BaseOptimizer):
    def run(self):
        """
        Optimize the objective function using the NLopt library.

        Args:
            x0 (ndarray): Initial guess for the optimization.
            bnds (ndarray): Bounds on the variables.
            options (dict): Dictionary of options for the optimization.

        Returns:
            res_out (OptimizedResult): Object containing the results of the optimization.
        """
        settings = self.settings
        x0 = self.x0
        bnds = self.bnds

        timer_start = timer()

        obj_fcn = self.obj_fcn
        idx_opt = self.idx_opt

        x0_opt = x0[idx_opt]
        bnds_opt = bnds[idx_opt, :].T
        
        algorithm = nlopt_algorithms[settings.algorithm]

        opt = nlopt.opt(algorithm, np.size(x0_opt))
        opt.set_min_objective(obj_fcn)
        if settings.stop_criteria_type == "iteration maximum":
            opt.set_maxeval(int(settings.stop_criteria_value) - 1)
        elif settings.stop_criteria_type == "maximum time [min]":
            opt.set_maxtime(settings.stop_criteria_value * 60)

        opt.set_xtol_rel(settings.x_tol_rel)
        opt.set_ftol_rel(settings.f_tol_rel)
        opt.set_lower_bounds(bnds_opt[0])
        opt.set_upper_bounds(bnds_opt[1])

        # initial_step
        max_initial_step = np.max(np.abs(bnds_opt - x0_opt), axis=0)

        initial_step = (bnds_opt[1] - bnds_opt[0]) * settings.initial_step

        # TODO: bring this back in at some point?
        # coef_id_opt = [id for n, id in enumerate(self.coef_id) if n in idx_opt]
        # for n, coef_name in enumerate(coef_id_opt):
        #     if "dd_bp" in coef_name:
        #         initial_step[n] *= 2

        #     if coef_name == "hdd_bp":
        #         initial_step[n] *= -1

        initial_step = np.clip(initial_step, -max_initial_step, max_initial_step)

        x1 = x0_opt + initial_step
        np.putmask(
            initial_step, (x1 < bnds_opt[0]) | (x1 > bnds_opt[1]), -initial_step
        )  # first step in direction of more variable space

        opt.set_initial_step(initial_step)

        # alter default size of population in relevant algorithms
        if settings.algorithm == "nlopt_crs2_lm":
            default_pop_size = 10 * (len(x0_opt) + 1)
        elif settings.algorithm in ["nlopt_mlsl_lds", "nlopt_mlsl"]:
            default_pop_size = 4
        elif settings.algorithm == "nlopt_isres":
            default_pop_size = 20 * (len(x0_opt) + 1)

            opt.set_population(
                int(np.rint(default_pop_size * settings["initial_pop_multiplier"]))
            )

        # if using multistart algorithm as global, set subopt
        if (settings.algorithm == "nlopt_mlsl_lds"):  
            raise NotImplementedError("nlopt_mlsl_lds not implemented")
            local_algorithm = nlopt_algorithms[self.opt_settings.algorithm]
            sub_opt = nlopt.opt(local_algorithm, np.size(x0_opt))
            sub_opt.set_initial_step(initial_step)
            sub_opt.set_xtol_rel(settings.x_tol_rel)
            sub_opt.set_ftol_rel(settings.f_tol_rel)
            opt.set_local_optimizer(sub_opt)

        x_opt = opt.optimize(x0_opt)  # optimize!

        if nlopt.SUCCESS > 0:
            success = True
            msg = pos_msg[nlopt.SUCCESS - 1]
        else:
            success = False
            msg = neg_msg[nlopt.SUCCESS - 1]

        res = BaseOptimizedResult()
        res.x = x_opt
        res.success = success
        res.message = msg
        res.fun = opt.last_optimum_value()
        res.nfev = opt.get_numevals()
        res.time_elapsed = timer() - timer_start

        return res


class InitialGuessOptimizer:
    def __init__(self, obj_fcn, x0, bnds, settings):
        """
        The constructor for the Optimizer class.

        Parameters:
            obj_fcn (function): The objective function to be optimized.
            x0 (np.array): The initial guess for the optimization.
            bnds (list): The bounds for the optimization.
            opt_settings (Opt_Settings): The settings for the optimization.
        """
        self.x0 = np.array(x0)
        self.bnds = np.array(bnds)
        
        self.obj_fcn = obj_fcn

        self.settings = settings

    def run(self):
        """
        This method runs the optimization process.

        Returns:
            OptimizedResult: An object containing the results of the optimization.
        """
        bnds = self.bnds

        res_all = []
        for settings in [self.settings]:
            if len(res_all) == 0:
                x0 = self.x0
            else:
                x0 = res_all[list(res_all.keys())[-1]].x

            if settings.algorithm[:5] == "scipy":
                res = SciPyOptimizer(self.obj_fcn, x0, bnds, settings).run()
            elif settings.algorithm[:5] == "nlopt":
                res = NLoptOptimizer(self.obj_fcn, x0, bnds, settings).run()

            res_all.append(res)

            if (
                settings.algorithm == "nlopt_MLSL_LDS"
            ):  # if using multistart algorithm, break upon finishing loop
                break

        return res_all[-1]


class Optimizer:
    """
    This class is used to perform optimization on a given objective function using either the SciPy or NLopt library.
    The optimization can be performed globally or locally based on the options provided.

    Attributes:
        bnds (np.array): The bounds for the optimization.
        x0 (np.array): The initial guess for the optimization.
        obj_fcn (function): The objective function to be optimized.
        idx_opt (int): The index of the optimal solution.
        coef_id (str): The identifier for the coefficient.
        settings (dict): The settings for the optimization.
        opt_options (dict): The options for the optimization.
    """

    def __init__(self, obj_fcn, x0, bnds, coef_id, settings, opt_settings):
        """
        The constructor for the Optimizer class.

        Parameters:
            obj_fcn (function): The objective function to be optimized.
            x0 (np.array): The initial guess for the optimization.
            bnds (list): The bounds for the optimization.
            coef_id (str): The identifier for the coefficient.
            settings (dict): The settings for the optimization.
            opt_settings (Opt_Settings): The settings for the optimization.
        """
        self.coef_id = coef_id
        self.x0 = np.array(x0)
        self.bnds = np.array(bnds)
        
        self.obj_fcn = obj_fcn

        self.settings = settings
        self.opt_settings = opt_settings

    def run(self):
        """
        This method runs the optimization process.

        Returns:
            OptimizedResult: An object containing the results of the optimization.
        """
        bnds = self.bnds

        res_all = []
        for settings in [self.opt_settings]:
            if len(res_all) == 0:
                x0 = self.x0
            else:
                x0 = res_all[list(res_all.keys())[-1]].x

            if settings.algorithm[:5] == "scipy":
                optimizer_class = SciPyOptimizer
            elif settings.algorithm[:5] == "nlopt":
                optimizer_class = NLoptOptimizer
                
            optimizer = optimizer_class(self.obj_fcn, x0, bnds, settings)
            res = optimizer.run()

            x, mean_loss, TSS, T, model, weight, resid, jac, alpha, C = optimizer.obj_fcn(
                res.x, optimize_flag=False
            )

            res = OptimizedResult(
                x,
                bnds,
                self.coef_id,
                alpha,
                C,
                T,
                model,
                weight,
                resid,
                jac,
                mean_loss,
                TSS,
                res.success,
                res.message,
                res.nfev,
                res.time_elapsed,
                self.settings,
            )

            res_all.append(res)

            if (
                settings.algorithm == "nlopt_MLSL_LDS"
            ):  # if using multistart algorithm, break upon finishing loop
                break

        return res_all[-1]