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

from opendsm.comparison_groups.individual_meter_matching import highs_settings as _highs_settings
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy import sparse
from qpsolvers import solve_ls

from opendsm.comparison_groups.individual_meter_matching.settings import Settings

__all__ = ("DistanceMatching",)



def cp_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _distances(ls_t, ls_cp, weights=None, dist_metric="euclidean", n_meters_per_chunk=10000):
    if weights is not None:
        ls_t = ls_t * weights

    # calculate distances in chunks
    n_chunk = len(ls_cp)
    if n_meters_per_chunk < n_chunk:
        n_chunk = n_meters_per_chunk

    dist = []
    for ls_cp_chunk in cp_chunks(ls_cp, n_meters_per_chunk):
        if weights is not None:
            ls_cp_chunk = ls_cp_chunk * weights

        # perform weighted distance calculation
        chunked_dist = cdist(ls_t, ls_cp_chunk, metric=dist_metric)

        dist.append(chunked_dist)

    dist = np.hstack(dist)

    return dist


def highs_fit_comparison_group_loadshape(t_ls, cp_ls, coef_sum=1, solver="highs", settings=None, verbose=False):
    if settings is None:
        if coef_sum == 1:
            settings = _highs_settings.HiGHS_Settings(
                primal_feasibility_tolerance=1E-4, 
                dual_feasibility_tolerance=1E-4, 
            )
        else:
            settings = _highs_settings.HiGHS_Settings(
                primal_feasibility_tolerance=1, 
                dual_feasibility_tolerance=1, 
            )
        settings = {k.lower(): v for k, v in dict(settings).items()}

    if coef_sum == 1:
        _MIN_X = 1E-6
    else:
        _MIN_X = 5E-3
    
    num_pool_meters = cp_ls.shape[0]

    R = sparse.csc_matrix(cp_ls.T)

    h = np.zeros(num_pool_meters)
    eye = sparse.eye(num_pool_meters, format="csc")
    A = sparse.csc_matrix(np.ones(num_pool_meters))
    b = np.array([coef_sum])

    lb = np.zeros(num_pool_meters)
    ub = np.ones(num_pool_meters)

    x_opt = solve_ls(R, t_ls, G=-eye, h=h, A=A, b=b, lb=lb, ub=ub, solver=solver, verbose=verbose, **settings)

    x_opt[x_opt < 0] = 0
    x_opt[x_opt > 1] = 1
    x_opt[np.abs(x_opt) < _MIN_X] = 0
    x_opt *= coef_sum/x_opt.sum()

    return x_opt


class DistanceMatchingError(Exception):
    pass


class DistanceMatching:
    """
    Parameters
    ----------
    treatment_group: pd.DataFrame
        A dataframe representing treatment group meters, indexed by id, with each column being a data point in a usage pattern.
    comparison_pool: pd.DataFrame
        A dataframe representing comparison pool meters, indexed by id, with each column being a data point in a usage pattern.
    weights: list | 1D np.array
        A list of floats (must be of length of the treatment group columns) to scale the usage patterns in order to ensure that certain components of usage have higher weights towards matching than others.
    n_treatments_per_chunk: int
        Due to local memory limitations, treatment meters can be chunked so that the cdist calculation can happen in memory. 10,000 meters appear to be sufficient for most memory constraints.
    """

    def __init__(
        self,
        settings=None,
    ):
        if settings is None:
            self.settings = Settings()
        elif isinstance(settings, Settings):
            self.settings = settings
        else:
            raise Exception(
                "invalid settings provided to 'individual_metering_matching'"
            )

        self.dist_metric = settings.distance_metric
        if self.dist_metric == "manhattan":
            self.dist_metric = "cityblock"

    def _closest_idx_duplicates_allowed(self, distances, n_match=None):
        if n_match is None:
            n_match = self.settings.n_matches_per_treatment

        if n_match > distances.shape[1]:
            n_match = distances.shape[1]

        # sort distances by row and get the indices of the sorted distances
        # Note: pypi bottleneck is faster than numpy for this
        cg_idx = np.argpartition(distances, n_match, axis=1)[:, :n_match]

        return cg_idx

    def _closest_idx_duplicates_not_allowed(self, ls_t, ls_cp, distances):
        n_match = self.settings.n_matches_per_treatment
        selection_method = self.settings.selection_method

        n_treatment = ls_t.shape[0]
        n_pool = ls_cp.shape[0]

        if n_match*n_treatment > n_pool:
            n_match = int(n_pool / n_treatment)

        if n_match == 0:
            raise DistanceMatchingError(f"Not enough treatment pool meters {n_pool} to match with {n_treatment} treatment meters without duplicates")
        
        if selection_method == "minimize_meter_distance":
            # normalize distances by min distance of each row
            # min_dist = np.take_along_axis(distances, self._closest_idx_duplicates_allowed(distances, n_match=1), axis=1)
            # distances = distances / min_dist

            # duplicate rows n_match times
            distances = np.repeat(distances, n_match, axis=0)
            t_idx = np.repeat(np.arange(distances.shape[0]), n_match)

            row_idx, col_idx = linear_sum_assignment(distances)

            cg_idx = [[] for _ in range(distances.shape[0])]
            for i, cp_idx in zip(row_idx, col_idx):
                cg_idx[t_idx[i]].append(cp_idx)

        elif selection_method == "minimize_loadshape_distance":
            coef_sum = n_match*len(ls_t)
            ls_t_mean = np.mean(ls_t.values, axis=0)*coef_sum

            x_opt = highs_fit_comparison_group_loadshape(
                ls_t_mean, ls_cp.values, coef_sum=coef_sum, solver="highs", settings=None, verbose=False
            )

            # argsort x_opt
            x_opt_idx = np.argsort(x_opt)[::-1][:coef_sum]

            # reshape distances to be ls_t.shape[0] x n_match
            cg_idx = np.reshape(x_opt_idx, (ls_t.shape[0], n_match))

        else:
            raise DistanceMatchingError(f"Invalid selection method: {selection_method}")

        return cg_idx
    
    
    def get_comparison_group(
        self,
        treatment_group,
        comparison_pool,
        weights=None,
    ):
        ls_t = treatment_group
        ls_cp = comparison_pool

        n_match = self.settings.n_matches_per_treatment
        max_distance_threshold = self.settings.max_distance_threshold
        n_meters_per_chunk = self.settings.n_treatments_per_chunk

        # TODO: if matching loadshapes, this isn't necessary
        distances = _distances(ls_t, ls_cp, weights, self.dist_metric, n_meters_per_chunk)

        if self.settings.allow_duplicate_matches:
            cg_idx = self._closest_idx_duplicates_allowed(distances, n_match=n_match)
        else:
            cg_idx = self._closest_idx_duplicates_not_allowed(ls_t, ls_cp, distances)

        data = []
        for t_idx in range(ls_t.shape[0]):
            t_id = ls_t.index[t_idx]
            for cp_idx in cg_idx[t_idx]:
                cg_id = ls_cp.index[cp_idx]

                data.append([cg_id, t_id, distances[t_idx, cp_idx]])

        df = pd.DataFrame(data, columns=["id", "treatment", "distance"])

        # check that the distance is less than the threshold
        if max_distance_threshold is not None:
            df = df[df["distance"] <= max_distance_threshold]

        # add column if id is duplicated
        df["duplicated"] = df.duplicated(subset="id", keep=False)
        
        return df


if __name__ == "__main__":
    d = DistanceMatching()
    print(d.settings)
