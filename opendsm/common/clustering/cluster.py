#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

   Copyright 2014-2025 OpenDSM contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from pydantic import BaseModel, ConfigDict

import pywt

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN, HDBSCAN, Birch, SpectralClustering



from opendsm.common.clustering import (
    bisect_k_means as _bisect_k_means,
    scoring as _scoring,
    settings as _settings,
)



class _LabelResult(BaseModel):
    """
    contains metrics about a cluster label returned from sklearn
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    labels: np.ndarray
    score: float
    score_unable_to_be_calculated: bool
    n_clusters: int


def _bisecting_kmeans_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using Bisecting K-Means algorithm
    """
    
    recluster_count = settings.bisecting_kmeans.recluster_count
    n_cluster_lower = settings.bisecting_kmeans.n_cluster.lower
    n_cluster_upper = settings.bisecting_kmeans.n_cluster.upper
    n_init = settings.bisecting_kmeans.internal_recluster_count
    inner_algorithm = settings.bisecting_kmeans.inner_algorithm
    bisecting_strategy = settings.bisecting_kmeans.bisecting_strategy

    score_choice = settings.bisecting_kmeans.scoring.score_metric
    dist_metric = settings.bisecting_kmeans.scoring.distance_metric
    min_cluster_size = settings.bisecting_kmeans.scoring.min_cluster_size
    max_non_outlier_cluster_count = 200

    seed = settings._seed

    results = []
    for i in range(recluster_count):
        algo = _bisect_k_means.BisectingKMeans(
            n_clusters=n_cluster_upper,
            init="k-means++",  # does not benefit from k-means++ like other k-means
            n_init=n_init,
            random_state=seed + i,
            algorithm=inner_algorithm,
            bisecting_strategy=bisecting_strategy,
        )
        algo.fit(data)
        labels_dict = algo.labels_full

        # if specifying clusters, only score the specified clusters
        if n_cluster_lower == n_cluster_upper:
            labels_dict = {n_cluster_lower: labels_dict[n_cluster_lower]}

        for n_cluster, labels in labels_dict.items():
            score, score_unable_to_be_calculated = _scoring.score_clusters(
                data,
                labels,
                n_cluster_lower,
                score_choice,
                dist_metric,
                min_cluster_size,
                max_non_outlier_cluster_count,
            )

            label_res = _LabelResult(
                labels=labels,
                score=score,
                score_unable_to_be_calculated=score_unable_to_be_calculated,
                n_clusters=n_cluster,
            )
            results.append(label_res)

    # get the results index with the smallest score
    HoF = None
    for result in results:
        if result.score_unable_to_be_calculated:
            continue

        if HoF is None or result.score < HoF.score:
            HoF = result

    return HoF.labels


def _birch_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    Clusters features using Birch algorithm
    """

    n_cluster_lower = settings.birch.n_cluster.lower
    n_cluster_upper = settings.birch.n_cluster.upper
    threshold = settings.birch.threshold
    branching_factor = settings.birch.branching_factor

    score_choice = settings.birch.scoring.score_metric
    dist_metric = settings.birch.scoring.distance_metric
    min_cluster_size = settings.birch.scoring.min_cluster_size
    max_non_outlier_cluster_count = 200

    results = []
    for n_clusters in range(n_cluster_lower, n_cluster_upper + 1):
        algo = Birch(
            n_clusters=n_clusters,
            threshold=threshold,
            branching_factor=branching_factor,
        )
        labels = algo.fit_predict(data)
        
        # Calculate score for the clusters
        score, score_unable_to_be_calculated = _scoring.score_clusters(
                data,
                labels,
                n_cluster_lower,
                score_choice,
                dist_metric,
                min_cluster_size,
                max_non_outlier_cluster_count,
            )
        
        label_res = _LabelResult(
                labels=labels,
                score=score,
                score_unable_to_be_calculated=score_unable_to_be_calculated,
                n_clusters=n_clusters,
            )
        
        results.append(label_res)

    # get the results index with the smallest score
    HoF = None
    for result in results:
        if result.score_unable_to_be_calculated:
            continue

        if HoF is None or result.score < HoF.score:
            HoF = result

    return HoF.labels


def _dbscan_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using DBSCAN algorithm
    """
    algo = DBSCAN(
        eps=settings.dbscan.epsilon, 
        min_samples=settings.dbscan.min_samples, 
        metric=settings.dbscan.distance_metric.value,
        algorithm=settings.dbscan.nearest_neighbors_algorithm,
        leaf_size=settings.dbscan.leaf_size,
        p=settings.dbscan.minkowski_p,
    )
    labels = algo.fit_predict(data)

    return labels


def _hdbscan_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using HDBSCAN algorithm
    """
    min_samples = settings.hdbscan.min_samples
    if settings.hdbscan.min_samples == 1:
        min_samples = 2

    algo = HDBSCAN(
        min_samples=settings.hdbscan.scoring_sample_count, 
        min_cluster_size=min_samples,
        allow_single_cluster=settings.hdbscan.allow_single_cluster,
        max_cluster_size=settings.hdbscan.max_cluster_size,
        metric=settings.hdbscan.distance_metric,
        cluster_selection_epsilon=settings.hdbscan.cluster_selection_epsilon,
        alpha=settings.hdbscan.robust_single_linkage_scaling,
        algorithm=settings.hdbscan.nearest_neighbors_algorithm,
        leaf_size=settings.hdbscan.leaf_size,
        cluster_selection_method=settings.hdbscan.cluster_selection_method,
    )
    labels = algo.fit_predict(data)

    if settings.hdbscan.min_samples == 1:
        # get count of -1 labels
        outlier_count = np.sum(labels == -1)

        if outlier_count == 0:
            return labels

        # add to all labels to make room for outliers
        labels[labels != -1] += outlier_count

        # make labels with -1 defined as arange(max_label+1, n_samples)
        labels[labels == -1] = np.arange(0, outlier_count)

    return labels


def _spectral_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using Spectral Clustering algorithm
    """
    n_cluster_lower = settings.spectral.n_cluster.lower
    n_cluster_upper = settings.spectral.n_cluster.upper
    
    score_choice = settings.spectral.scoring.score_metric
    dist_metric = settings.spectral.scoring.distance_metric
    min_cluster_size = settings.spectral.scoring.min_cluster_size
    max_non_outlier_cluster_count = 200

    # transform data as spectral clustering doesn't like negative values
    # data = np.exp(-data / np.std(data))

    X = data
    results = []
    for n_clusters in range(n_cluster_lower, n_cluster_upper + 1):
        if n_clusters == n_cluster_lower:
            affinity = settings.spectral.affinity
        else:
            affinity = "precomputed"

        algo = SpectralClustering(
            n_clusters=n_clusters,
            eigen_solver=settings.spectral.eigen_solver,
            n_components=settings.spectral.n_components,
            affinity=affinity,
            n_neighbors=settings.spectral.nearest_neighbors,
            gamma=settings.spectral.gamma,
            eigen_tol=settings.spectral.eigen_tol,
            assign_labels=settings.spectral.assign_labels,
        )

        # hide UserWarning from sklearn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            labels = algo.fit_predict(X)

        X = algo.affinity_matrix_

        # Calculate a score for the clustering
        score, score_unable_to_be_calculated = _scoring.score_clusters(
                data,
                labels,
                n_cluster_lower,
                score_choice,
                dist_metric,
                min_cluster_size,
                max_non_outlier_cluster_count,
            )
        
        label_res = _LabelResult(
                labels=labels,
                score=score,
                score_unable_to_be_calculated=score_unable_to_be_calculated,
                n_clusters=n_clusters,
            )
        
        results.append(label_res)

    # get the results index with the smallest score
    HoF = None
    for result in results:
        if result.score_unable_to_be_calculated:
            continue

        if HoF is None or result.score < HoF.score:
            HoF = result

    return HoF.labels


def _transform_data(
    data: pd.DataFrame,
    settings: _settings.ClusteringSettings
):
    """
    Transforms the data using the wavelet transform settings
    """
    settings = settings.transform_settings

    def _dwt_coeffs(data, wavelet="db1", wavelet_mode="periodization", n_levels=4):
        all_features = []
        # iterate through rows of numpy array
        for row in range(len(data)):
            decomp_coeffs = pywt.wavedec(
                data[row], wavelet=wavelet, mode=wavelet_mode, level=n_levels
            )
            # remove last level
            # if n_levels > 4:
            # decomp_coeffs = decomp_coeffs[:-1]

            decomp_coeffs = np.hstack(decomp_coeffs)

            all_features.append(decomp_coeffs)

        return np.vstack(all_features)

    def _pca_coeffs(features, method, min_var_ratio_explained=0.95, n_components=None):
        if min_var_ratio_explained is not None:
            n_components = min_var_ratio_explained

        # kernel pca is not fully developed
        if method == "kernel_pca":
            pca = KernelPCA(n_components=None, kernel="rbf")
            pca_features = pca.fit_transform(features)

            if min_var_ratio_explained is not None:
                explained_variance_ratio = pca.eigenvalues_ / np.sum(pca.eigenvalues_)

                # get the cumulative explained variance ratio
                cumulative_explained_variance = np.cumsum(explained_variance_ratio)

                # find number of components that explain pct% of the variance
                n_components = np.argmax(cumulative_explained_variance > n_components).astype(int)

            if not isinstance(n_components, (int, np.integer)):
                raise ValueError("n_components must be an integer for kernel PCA")

            # pca = PCA(n_components=n_components)
            pca = KernelPCA(n_components=n_components, kernel="rbf")
            pca_features = pca.fit_transform(features)

        else:
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(features)

        return pca_features

    # calculate wavelet coefficients
    with warnings.catch_warnings():
        # TODO wavelet level 5 was chosen during hyperparam optimization, but
        # worth investigating this further
        warnings.filterwarnings("ignore", module="pywt._multilevel")
        features = _dwt_coeffs(
            data, 
            settings.wavelet_name, 
            settings.wavelet_mode, 
            settings.wavelet_n_levels
        )

    pca_features = _pca_coeffs(
        features,
        settings.pca_method,
        settings.pca_min_variance_ratio_explained,
        settings.pca_n_components,
    )

    # normalize pca features
    if settings._standardize:
        pca_features = (pca_features - pca_features.mean()) / pca_features.std()

    if settings.pca_include_median:
        pca_features = np.hstack([pca_features, np.median(data, axis=1)[:, None]])

    return pca_features



def cluster_features(
    data: np.ndarray,
    settings: _settings.ClusteringSettings,
):
    
    # bypass clustering if cluster count is >= data
    if settings.algorithm_selection not in ["dbscan", "hdbscan"]:
        algo = getattr(settings, f"{settings.algorithm_selection.value}")
        if algo.n_cluster.lower >= len(data):
            return np.arange(len(data))

    # standardize the data
    if settings.standardize:
        data = (data - data.mean()) / data.std()

    # transform the data
    if settings.transform_settings is not None:
        data = _transform_data(data, settings)

    # cluster the pca features
    if settings.algorithm_selection == "bisecting_kmeans":
        cluster_fcn = _bisecting_kmeans_clustering
    elif settings.algorithm_selection == "birch":
        cluster_fcn = _birch_clustering
    elif settings.algorithm_selection == "dbscan":
        cluster_fcn = _dbscan_clustering
    elif settings.algorithm_selection == "hdbscan":
        cluster_fcn = _hdbscan_clustering
    elif settings.algorithm_selection == "spectral":
        cluster_fcn = _spectral_clustering
    else:
        raise ValueError(f"Unknown clustering algorithm: {settings.algorithm_selection}")
    
    cluster_labels = cluster_fcn(data, settings)

    return cluster_labels