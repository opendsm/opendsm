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

import numpy as np
import pandas as pd
import pytest

import opendsm.comparison_groups as cg
from opendsm.comparison_groups.cg_clustering import treatment_fit
from opendsm.comparison_groups.cg_clustering.treatment_fit import (
    _initial_cluster_weights,
    _match_treatment_to_cluster,
)
from opendsm.comparison_groups.cg_clustering.settings import CG_Clustering_Settings



def test_initial_cluster_weights_exact_match_gets_full_weight():
    """Regression: a treatment equal to a cluster centroid must put full weight
    on that cluster. The previous in-place divide collapsed the row to uniform
    weights instead."""
    cp_ls = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    t_ls = np.array([[1.0, 2.0, 3.0]])  # exactly equals cluster 1

    weights = _initial_cluster_weights(t_ls, cp_ls)

    assert weights.shape == (1, 3)
    np.testing.assert_allclose(weights[0], [0.0, 1.0, 0.0], atol=1e-6)


def test_initial_cluster_weights_peak_on_nearest_and_normalized():
    """Non-exact treatment: weight peaks on the nearest cluster and sums to 1."""
    cp_ls = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [9.0, 9.0, 9.0]])
    t_ls = np.array([[1.1, 2.0, 3.0]])  # closest to cluster 1

    weights = _initial_cluster_weights(t_ls, cp_ls)

    assert weights[0].argmax() == 1
    np.testing.assert_allclose(weights[0].sum(), 1.0, atol=1e-6)
    assert (weights[0] >= 0).all()



N_POOL = 60
N_TREATMENT = 5
N_FEATURES = 24
N_CLUSTERS = 3


def _build_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)

    cluster_centers = rng.normal(loc=0.0, scale=1.0, size=(N_CLUSTERS, N_FEATURES))
    pool_labels = np.repeat(np.arange(N_CLUSTERS), N_POOL // N_CLUSTERS)
    pool_data = cluster_centers[pool_labels] + 0.1 * rng.normal(size=(N_POOL, N_FEATURES))

    df_ls_cluster = pd.DataFrame(
        pool_data,
        index=pd.Index([f"pool_{i}" for i in range(N_POOL)], name="id"),
        columns=[f"h_{h}" for h in range(N_FEATURES)],
    )
    df_cluster = pd.DataFrame(
        {"cluster": pool_labels},
        index=df_ls_cluster.index,
    )

    treatment_labels = rng.integers(0, N_CLUSTERS, size=N_TREATMENT)
    treatment_data = cluster_centers[treatment_labels] + 0.1 * rng.normal(size=(N_TREATMENT, N_FEATURES))
    df_ls_t = pd.DataFrame(
        treatment_data,
        index=pd.Index([f"treat_{i}" for i in range(N_TREATMENT)], name="id"),
        columns=df_ls_cluster.columns,
    )

    return df_ls_t, df_ls_cluster, df_cluster, treatment_labels


@pytest.fixture
def loadshape_inputs():
    return _build_inputs(seed=0)


def test_match_treatment_to_clusters_default_settings(loadshape_inputs):
    """match_treatment_to_clusters must work with the public factory's default
    settings, which nest normalize under feature_transform. Previously it read
    settings.normalize (a stale top-level path removed during the clustering
    refactor) and raised AttributeError."""
    df_ls_t, df_ls_cluster, df_cluster, _ = loadshape_inputs

    df_coeffs = treatment_fit.match_treatment_to_clusters(
        df_ls_t,
        df_ls_cluster,
        df_cluster,
        settings=cg.CG_Clustering_Settings(),
    )

    assert df_coeffs.shape == (df_ls_t.shape[0], N_CLUSTERS)
    assert list(df_coeffs.index) == list(df_ls_t.index)
    assert df_coeffs.columns.tolist() == [f"pct_cluster_{c}" for c in range(N_CLUSTERS)]
    assert np.all(np.isfinite(df_coeffs.to_numpy()))
    np.testing.assert_allclose(df_coeffs.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert (df_coeffs.to_numpy() >= 0).all()


def test_match_treatment_to_clusters_recovers_source_cluster(loadshape_inputs):
    """Each treatment is generated as cluster_center + small noise, so the
    recovered weight vector should peak on the source cluster."""
    df_ls_t, df_ls_cluster, df_cluster, expected_labels = loadshape_inputs

    df_coeffs = treatment_fit.match_treatment_to_clusters(
        df_ls_t,
        df_ls_cluster,
        df_cluster,
        settings=cg.CG_Clustering_Settings(),
    )

    predicted = df_coeffs.to_numpy().argmax(axis=1)

    assert (predicted == expected_labels).mean() >= 0.6


def test_single_cluster_assigns_full_weight():
    """With one cluster the sum-to-one constraint forces all weight onto it."""
    rng = np.random.default_rng(0)
    t_ls = pd.DataFrame(rng.normal(0, 1, (3, 24)), index=["t0", "t1", "t2"])
    cluster_ls = pd.DataFrame(rng.normal(0, 1, (1, 24)), index=[0])

    coeffs = _match_treatment_to_cluster(t_ls, cluster_ls, CG_Clustering_Settings())

    np.testing.assert_allclose(coeffs.to_numpy(), 1.0, atol=1e-6)


def test_invalid_treatment_rows_yield_nan():
    """A NaN treatment loadshape row produces NaN weights; valid rows still sum to 1."""
    rng = np.random.default_rng(0)
    t_ls = pd.DataFrame(rng.normal(0, 1, (3, 24)), index=["t0", "t1", "t2"])
    t_ls.iloc[1, :] = np.nan
    cluster_ls = pd.DataFrame(rng.normal(0, 1, (2, 24)), index=[0, 1])

    coeffs = _match_treatment_to_cluster(t_ls, cluster_ls, CG_Clustering_Settings())

    assert coeffs.loc["t1"].isnull().all()
    assert not coeffs.loc["t0"].isnull().any()
    np.testing.assert_allclose(coeffs.loc["t0"].sum(), 1.0, atol=1e-6)
