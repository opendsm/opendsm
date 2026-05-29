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

import warnings

import numpy as np

from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh as _sparse_eigsh
from scipy.linalg import eigh as _scipy_eigh

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster._spectral import discretize as _discretize

from opendsm.common.clustering.algorithms.settings import AffinityMatrixOptions
from opendsm.common.clustering.metrics.labels import ClusteringResult

from opendsm.common.clustering.algorithms.spectral._affinity import (
    _SELF_TUNING_SPARSE_THRESHOLD,
    _self_tuning_affinity_dense,
    _self_tuning_affinity_sparse,
    _affinity_matrix,
)


def suggest_k_from_eigengap(A, topK=5):
    """Compute eigengap heuristic k-candidates from an affinity matrix.

    Returns (nb_clusters, eigenvalues, eigenvectors) where nb_clusters is
    sorted by descending eigengap (largest gap = most likely k).
    """
    L = csgraph.laplacian(A, normed=True)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1]
    nb_clusters = index_largest_gap + 1
    return nb_clusters, eigenvalues, eigenvectors


def eigendecomp_cluster_count(
    X,
    algo_settings: SpectralSettings,
):
    """Filter eigengap k-candidates to the configured cluster range."""
    min_clusters = algo_settings.n_cluster.lower
    max_clusters = algo_settings.n_cluster.upper

    nb_clusters, _, _ = suggest_k_from_eigengap(X)
    nb_clusters = nb_clusters[nb_clusters >= min_clusters]
    nb_clusters = nb_clusters[nb_clusters <= max_clusters]
    return nb_clusters


def _eigengap_scores(eigenvalues, n_cluster_lower, n_cluster_upper):
    """Return eigengap-based scores for each k candidate (lower = better)."""
    gaps = np.diff(eigenvalues)
    n_clusters_range = np.arange(n_cluster_lower, n_cluster_upper + 1)
    scores = np.array([
        -gaps[k - 1] if k <= len(gaps) else np.inf
        for k in n_clusters_range
    ], dtype=np.float64)
    return scores


def _assign_labels(vectors, n_clusters, method, seed):
    """Assign cluster labels from spectral embedding vectors."""
    if method == "cluster_qr":
        try:
            from sklearn.cluster._spectral import cluster_qr as _cluster_qr
            return _cluster_qr(vectors[:, :n_clusters])
        except ImportError:
            method = "kmeans"

    if method == "discretize":
        return _discretize(vectors[:, :n_clusters], random_state=seed)

    v = vectors[:, :n_clusters]
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / np.maximum(norms, 1e-10)
    return KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=10
    ).fit_predict(v)



def _single_spectral_clustering(data, settings, seed):
    """Single-pass spectral clustering over the configured k range."""
    algo_settings = getattr(settings, settings.algorithm_selection.value)

    n_cluster_lower = algo_settings.n_cluster.lower
    n_cluster_upper = algo_settings.n_cluster.upper

    lbl = ClusteringResult(
        data=data,
        score_settings=algo_settings.scoring,
        seed=seed,
        n_cluster_lower=n_cluster_lower,
        min_cluster_size=settings.min_cluster_size,
        small_cluster_mode=settings.small_cluster_mode,
    )
    eigenvalues = None
    n_clusters_range = np.arange(n_cluster_lower, n_cluster_upper + 1)
    affinity = algo_settings.affinity

    if affinity == AffinityMatrixOptions.NEAREST_NEIGHBORS:
        algo = SpectralClustering(
            n_clusters=n_cluster_lower,
            eigen_solver=algo_settings.eigen_solver,
            n_components=algo_settings.n_components,
            affinity=affinity,
            n_neighbors=algo_settings.nearest_neighbors,
            gamma=algo_settings.gamma,
            eigen_tol=algo_settings.eigen_tol,
            assign_labels=algo_settings.assign_labels,
            random_state=seed,
        )
        for n_clusters in n_clusters_range:
            if n_clusters > n_cluster_lower:
                algo.n_clusters = n_clusters
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                labels = algo.fit_predict(data)
            lbl.add(int(n_clusters), labels)
        del algo

    elif (
        affinity == AffinityMatrixOptions.SELF_TUNING
        and data.shape[0] > _SELF_TUNING_SPARSE_THRESHOLD
    ):
        k_st = algo_settings.local_scale_neighbors
        k_connect = 2 * (k_st + n_cluster_upper)
        X_sparse = _self_tuning_affinity_sparse(data, k_st, k_connect)
        L_sparse = csgraph.laplacian(X_sparse, normed=True)
        del X_sparse

        n_eigvecs = algo_settings.n_components or n_cluster_upper
        try:
            eigenvalues, eigenvectors = _sparse_eigsh(
                L_sparse, k=n_eigvecs, which="SM", maxiter=1000, tol=1e-6
            )
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        except Exception:
            raise
        del L_sparse

        embedding = eigenvectors[:, :n_eigvecs]
        assign_method = algo_settings.assign_labels
        for n_clusters in n_clusters_range:
            vectors = embedding.copy() if algo_settings.n_components is not None \
                else embedding[:, :n_clusters].copy()
            labels = _assign_labels(vectors, int(n_clusters), assign_method, seed)
            lbl.add(int(n_clusters), labels)

    else:
        if affinity == AffinityMatrixOptions.SELF_TUNING:
            X = _self_tuning_affinity_dense(data, algo_settings.local_scale_neighbors)
        else:
            algo = SpectralClustering(
                n_clusters=n_cluster_lower,
                eigen_solver=algo_settings.eigen_solver,
                n_components=algo_settings.n_components,
                affinity=affinity,
                n_neighbors=algo_settings.nearest_neighbors,
                gamma=algo_settings.gamma,
                eigen_tol=algo_settings.eigen_tol,
                assign_labels=algo_settings.assign_labels,
                random_state=seed,
            )
            X = _affinity_matrix(data, algo)
            del algo

        L = csgraph.laplacian(X, normed=True)
        del X
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        n_eigvecs = algo_settings.n_components or n_cluster_upper
        embedding = eigenvectors[:, :n_eigvecs]
        assign_method = algo_settings.assign_labels

        for n_clusters in n_clusters_range:
            vectors = embedding.copy() if algo_settings.n_components is not None \
                else embedding[:, :n_clusters].copy()
            labels = _assign_labels(vectors, int(n_clusters), assign_method, seed)
            lbl.add(int(n_clusters), labels)

    eigengap_weight = algo_settings.eigengap_weight
    if eigenvalues is not None and eigengap_weight > 0:
        n_lower = algo_settings.n_cluster.lower
        n_upper = algo_settings.n_cluster.upper
        scores = _eigengap_scores(eigenvalues, n_lower, n_upper)
        if scores is not None:
            lbl._eigengap_scores = {
                k: float(scores[i])
                for i, k in enumerate(range(n_lower, n_upper + 1))
                if i < len(scores)
            }
            lbl._eigengap_weight = eigengap_weight

    return lbl


def spectral(data, settings):
    """Spectral clustering over a k range with optional multi-seed reclustering."""
    algo_settings = getattr(settings, settings.algorithm_selection.value)
    seed = settings._seed
    recluster_count = algo_settings.recluster_count

    if recluster_count == 0:
        return _single_spectral_clustering(data, settings, seed)

    lbl = ClusteringResult(
        data=data,
        score_settings=algo_settings.scoring,
        seed=seed,
        n_cluster_lower=algo_settings.n_cluster.lower,
        min_cluster_size=settings.min_cluster_size,
        small_cluster_mode=settings.small_cluster_mode,
    )

    for n in range(recluster_count + 1):
        lbl_inner = _single_spectral_clustering(data, settings, seed + n)

        if lbl_inner._labels_store:
            winner_merged = lbl_inner.labels
            k_actual = int(len(np.unique(winner_merged[winner_merged != -1])))
            lbl.add(k_actual, winner_merged)

    return lbl
