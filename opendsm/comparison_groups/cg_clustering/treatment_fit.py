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

"""
functions for dealing with fitting to clusters
"""

import scipy.optimize
import scipy.spatial.distance

import numpy as np
import pandas as pd

from opendsm.common.clustering import transform as _transform
from opendsm.common.stats.adaptive_loss import adaptive_weights
from opendsm.comparison_groups.cg_clustering import settings as _settings



def _get_cluster_ls(df_cp_ls: pd.DataFrame, cluster_df: pd.DataFrame, agg_type: str):
    """
    original cp loadshape and cluster df
    settings for agg_type
    """

    cluster_df = cluster_df.join(df_cp_ls, on="id")
    cluster_df = cluster_df.reset_index().set_index(["id", "cluster"])  # type: ignore

    # calculate cp_df
    df_cluster_ls = cluster_df.groupby("cluster").agg(agg_type)  # type: ignore
    df_cluster_ls = df_cluster_ls[df_cluster_ls.index.get_level_values(0) > -1]  # don't match to outlier cluster

    return df_cluster_ls


def fit_to_clusters(
    t_ls, 
    cp_ls, 
    x0, 
    settings_dict,
):
    # instantiate settings from settings_dict
    settings = _settings.CG_Clustering_Settings(**settings_dict)
    match_settings = settings.treatment_match

    _min_pct_cluster = match_settings.percent_cluster_minimum

    def _remove_small_x(x: np.ndarray):
        # remove small values and normalize to 1
        x[x < _min_pct_cluster] = 0
        x /= np.sum(x)

        return x

    def obj_fcn_dec(t_ls, cp_ls, idx=None):
        if idx is not None:
            cp_ls = cp_ls[idx, :]

        def obj_fcn(x):
            x = _remove_small_x(x)
            resid = (t_ls - (cp_ls * x[:, None]).sum(axis=0)).flatten()

            if match_settings._adaptive_loss_alpha == 2:
                wSSE = np.sum(resid**2)

            else:
                weight, _, _ = adaptive_weights(
                    resid, 
                    alpha=match_settings._adaptive_loss_alpha, 
                    sigma=match_settings.adaptive_loss_sigma, 
                    quantile=0.25,
                    min_weight=0.0,
                    C_algo=match_settings.adaptive_loss_c_algo,
                ) # type: ignore

                wSSE = np.sum(weight * resid**2)

            return wSSE

        return obj_fcn

    def sum_to_one(x):
        zero = np.sum(x) - 1
        return zero

    x0 = np.array(x0).flatten()

    # only optimize if >= _MIN_PCT_CLUSTER
    idx = np.argwhere(x0 >= _min_pct_cluster).flatten()
    if len(idx) == 0:
        idx = np.arange(0, len(x0))

    x0_n = x0[idx]

    bnds = np.repeat(np.array([0, 1])[:, None], x0_n.shape[0], axis=1).T
    const = [{"type": "eq", "fun": sum_to_one}]

    res = scipy.optimize.minimize(
        obj_fcn_dec(t_ls, cp_ls, idx),
        x0_n,
        bounds=bnds,
        constraints=const,
        method="SLSQP",
    )  # trust-constr, SLSQP
    # res = minimize(obj_fcn, x0, bounds=bnds, method='SLSQP') # trust-constr, SLSQP, L-BFGS-B
    # res = differential_evolution(obj_fcn, bnds, maxiter=100)
    # res = basinhopping(obj_fcn, x0, niter=10, minimizer_kwargs={'bounds': bnds, 'method': 'Powell'})

    x = np.zeros_like(x0)
    x[idx] = _remove_small_x(res.x)

    return x


class ClusterTreatmentMatchError(Exception):
    pass


def _match_treatment_to_cluster(
    df_ls_t: pd.DataFrame, 
    df_ls_cluster: pd.Series, 
    settings: _settings.Settings
):
    # Create null dataframe
    coeffs = np.empty((df_ls_t.shape[0], df_ls_cluster.shape[0]))
    t_ids = df_ls_t.index
    columns = [f"pct_cluster_{int(n)}" for n in df_ls_cluster.index]
    df_t_coeffs = pd.DataFrame(coeffs, index=t_ids, columns=columns)

    # error checking going into cdist
    if df_ls_t.shape[0] == 0:
        raise ClusterTreatmentMatchError("No valid treatment loadshapes")
    
    if df_ls_cluster.shape[0] == 0:
        raise ClusterTreatmentMatchError("No valid cluster loadshapes")
    
    if df_ls_t.shape[1] != df_ls_cluster.shape[1]:
        shape_str = f"Treatment[{df_ls_t.shape[1]}] != Cluster[{df_ls_cluster.shape[1]}]"
        raise ClusterTreatmentMatchError(f"Treatment and cluster loadshapes have different lengths: {shape_str}")

    # identify invalid rows
    idx_invalid = df_ls_t.isnull().any(axis=1) | ~np.isfinite(df_ls_t).any(axis=1)
    idx_valid = ~idx_invalid

    # convert to numpy
    t_ls = df_ls_t.to_numpy()
    cp_ls = df_ls_cluster.to_numpy()

    # filter to valid rows
    t_ls = t_ls[idx_valid, :]

    # Get percent from each cluster
    distances = scipy.spatial.distance.cdist(t_ls, cp_ls, metric="euclidean")  # type: ignore
    distances_norm = (np.min(distances, axis=1) / distances.T).T
    # change this number (20) to alter weights, larger centralizes the weight, smaller spreads them out
    distances_norm = (distances_norm**20)  
    distances_norm = (distances_norm.T / np.sum(distances_norm, axis=1)).T

    coeffs = []
    for n, t_id in enumerate(df_ls_t.index):
        t_id_ls = t_ls[n, :]
        x0 = distances_norm[n, :]

        coeffs_n = fit_to_clusters(t_id_ls, cp_ls, x0, settings.model_dump())

        coeffs.append(coeffs_n)

    coeffs = np.vstack(coeffs)

    # only update rows
    df_t_coeffs.loc[idx_invalid, :] = np.nan
    df_t_coeffs.loc[idx_valid, :] = coeffs

    return df_t_coeffs


def match_treatment_to_clusters(
    df_ls_t: pd.DataFrame,
    df_ls_cluster: pd.DataFrame,
    df_cluster: pd.DataFrame,
    settings: _settings._CG_Clustering_Settings,
):
    """
    performs the matching logic to a provided treatment_loadshape dataframe

    TODO: Handle call when no valid scores were found?

    """

    # get cluster loadshape and normalize
    df_ls_cluster_agg = _get_cluster_ls(
        df_cp_ls=df_ls_cluster,
        cluster_df=df_cluster,
        agg_type=settings.treatment_match.agg_type,
    )

    df_ls_cluster_agg[:] = _transform.normalize(
        data=df_ls_cluster_agg.to_numpy(),
        settings=settings.normalize,
    )

    # normalize treatment loadshape
    df_ls_t_norm = df_ls_t.copy()
    df_ls_t_norm[:] = _transform.normalize(
        data=df_ls_t_norm.to_numpy(),
        settings=settings.normalize,
    )

    # fit treatment to clusters
    df_t_coeffs = _match_treatment_to_cluster(
        df_ls_t=df_ls_t_norm,
        df_ls_cluster=df_ls_cluster_agg,
        settings=settings,
    )

    return df_t_coeffs