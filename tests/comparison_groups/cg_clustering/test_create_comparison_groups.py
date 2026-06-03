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
import pytest

from opendsm.comparison_groups.cg_clustering.create_comparison_groups import CG_Clustering
from opendsm.comparison_groups.cg_clustering.settings import CG_Clustering_Settings


# FPCA (the default feature transform) builds a Fourier basis via skfda, which
# emits a third-party DeprecationWarning the strict filter would otherwise fail on.
pytestmark = pytest.mark.filterwarnings("ignore:The Fourier class is deprecated:DeprecationWarning")


def test_get_labels_assigns_every_pool_meter(cg_clustering_data):
    _, comparison_pool_data = cg_clustering_data
    clustering = CG_Clustering(CG_Clustering_Settings())

    clusters = clustering.get_labels(comparison_pool_data)

    assert sorted(clusters.index) == sorted(comparison_pool_data.ids)
    assert clusters["cluster"].notna().all()
    # at least one real (non-outlier) cluster is found
    assert (clusters["cluster"] >= 0).any()


def test_match_treatment_to_clusters_rows_are_simplex(cg_clustering_data):
    treatment_data, comparison_pool_data = cg_clustering_data
    clustering = CG_Clustering(CG_Clustering_Settings())
    clustering.get_labels(comparison_pool_data)

    coeffs = clustering.match_treatment_to_clusters(treatment_data)

    assert coeffs.shape[0] == len(treatment_data.ids)
    assert (coeffs.to_numpy() >= 0).all()
    np.testing.assert_allclose(coeffs.sum(axis=1).to_numpy(), 1.0, atol=1e-6)


def test_match_before_labels_raises(cg_clustering_data):
    treatment_data, _ = cg_clustering_data
    clustering = CG_Clustering(CG_Clustering_Settings())

    with pytest.raises(ValueError):
        clustering.match_treatment_to_clusters(treatment_data)


def test_get_comparison_group_is_deterministic(cg_clustering_data):
    treatment_data, comparison_pool_data = cg_clustering_data

    labels_1, coeffs_1 = CG_Clustering(CG_Clustering_Settings()).get_comparison_group(
        treatment_data, comparison_pool_data
    )
    labels_2, coeffs_2 = CG_Clustering(CG_Clustering_Settings()).get_comparison_group(
        treatment_data, comparison_pool_data
    )

    assert list(labels_1["cluster"]) == list(labels_2["cluster"])
    np.testing.assert_allclose(coeffs_1.to_numpy(), coeffs_2.to_numpy())


def test_clustering_output_snapshot(cg_clustering_data, snapshot):
    """Pin permutation-invariant clustering outputs: number of non-outlier
    clusters, sorted cluster sizes, and each treatment's dominant weight."""
    treatment_data, comparison_pool_data = cg_clustering_data
    labels, coeffs = CG_Clustering(CG_Clustering_Settings()).get_comparison_group(
        treatment_data, comparison_pool_data
    )

    sizes = labels[labels["cluster"] >= 0]["cluster"].value_counts().sort_values()
    summary = {
        "n_clusters": int((labels["cluster"] >= 0).sum() and labels.loc[labels["cluster"] >= 0, "cluster"].nunique()),
        "sorted_cluster_sizes": sorted(sizes.tolist()),
        "dominant_weight_per_treatment": sorted(np.round(coeffs.to_numpy().max(axis=1), 4).tolist()),
    }

    assert summary == snapshot
