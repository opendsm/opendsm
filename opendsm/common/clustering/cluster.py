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
import pandas as pd

from scipy.signal import find_peaks
from scipy.spatial.distance import cdist

from opendsm.common.clustering import (
    settings as _settings,
    transform as _transform,
)

from opendsm.common.clustering.algorithms import (
    _bisecting_kmeans_clustering,
    _birch_clustering,
    _dbscan_clustering,
    _hdbscan_clustering,
    _spectral_clustering,
)



def _cluster_merge(
    cluster_labels: np.ndarray,
    data: np.ndarray,
    settings: _settings.ClusteringSettings,
    W: float = 0.5,
):
    
    # get unique labels
    unique_labels = np.unique(cluster_labels)

    # get the distance between all rows in data
    distances = cdist(data, data)

    intra_cluster_similarity = np.zeros(len(unique_labels))
    inter_cluster_similarity = np.zeros((len(unique_labels), len(unique_labels)))
    for i in range(len(unique_labels)):
        idx_i = np.where(cluster_labels == unique_labels[i])[0]
        for j in range(len(unique_labels)):
            idx_j = np.where(cluster_labels == unique_labels[j])[0]

            if i == j:
                intra_cluster_similarity[i] = np.mean(distances[idx_i, :][:, idx_i])
                inter_cluster_similarity[i, j] = 0
                continue
            elif i < j:
                continue

            inter_cluster_similarity[i, j] = np.sum(distances[idx_i, :][:, idx_j])
            inter_cluster_similarity[j, i] = np.nan

    # if there are only two clusters, merge them if the similarity is less than W
    if unique_labels.shape[0] == 2:
        cluster_similarity = inter_cluster_similarity[0, 1]
        mean_similarity = np.mean(distances[distances != 0])

        ratio = cluster_similarity / mean_similarity

        if ratio < W:
            return np.zeros(len(cluster_labels))
        
        return cluster_labels

    # if there are more than two clusters, merge them if the similarity is less than W
    mean_similarity = np.mean(inter_cluster_similarity)

    for i in reversed(range(len(unique_labels))):
        for j in reversed(range(len(unique_labels))):
            if i == j:
                continue

            ratio = inter_cluster_similarity[i, j] / mean_similarity

            if ratio < W:
                cluster_labels[cluster_labels == unique_labels[j]] = unique_labels[i]

    return cluster_labels


def cluster_reorder(
    data: pd.DataFrame, 
    cluster_labels: np.ndarray,
    settings: _settings.ClusteringSettings,
):
    sort_method = settings.cluster_sort.method
    agg_type = settings.cluster_sort.aggregation
    reverse = settings.cluster_sort.reverse

    # assign labels to data
    df = data.copy()
    df["label"] = cluster_labels
     # exclude label -1 (outliers) from reordering
    df = df[df['label'] >= 0]

    # calculate n_clusters after filtering out outliers
    uniq_labels = df['label'].unique()
    n_clusters = len(uniq_labels)

    if sort_method == "size":
        # sort clusters by count
        cluster_size = df['label'].value_counts()
        cluster_size = cluster_size.sort_values()

        features = cluster_size

    elif sort_method == "peak":
        # TODO: This is a work in progress

        # group by cluster and aggregate
        df_cluster = df.groupby('label').agg(agg_type)

        # subtract each cluster's median from the cluster's median
        df_cluster_norm = df_cluster.sub(df_cluster.agg(agg_type, axis=1), axis=0)
        cluster_max = df_cluster_norm.abs().max().max()
        df_cluster_norm = df_cluster_norm/cluster_max

        # define threshold for peak and valley
        threshold = np.quantile(abs(df_cluster.values), 0.75)

        # find peaks and valleys
        peak = {}
        valley = {}
        norm = {}
        for i in range(n_clusters):
            cluster_normal = df_cluster.iloc[i]
            norm[i] = cluster_normal.agg(agg_type)
            df_cluster_norm = cluster_normal - norm[i]
            thresh = threshold - norm[i]

            peak[i] = find_peaks(df_cluster_norm.values, height=thresh, width=1)[0]
            valley[i] = find_peaks(-df_cluster_norm.values, height=thresh, width=1)[0]

            if len(peak[i]) == 0:
                peak[i] = None
            else:
                peak[i] = peak[i][0]

            if len(valley[i]) == 0:
                valley[i] = None
            else:
                valley[i] = valley[i][0]
        
        # create df with peak and valley
        features = pd.DataFrame({'peak': peak, 'valley': valley, "norm": norm})

        features = features.sort_values(by=["peak", "valley", "norm"], na_position='first')

    # create dictionary to remap cluster numbers to features order
    cluster_map = {i: i for i in cluster_labels}

    if not reverse:
        cluster_map.update({features.index[i]: i for i in range(n_clusters)})
    else:
        cluster_map.update({features.index[i]: i for i in range(n_clusters)[::-1]})

    return cluster_map


def _cluster_features(
    data: np.ndarray,
    settings: _settings.ClusteringSettings,
) -> np.ndarray:

    # adjust upper cluster count if necessary
    if settings.algorithm_selection not in ["dbscan", "hdbscan"]:
        algo = f"{settings.algorithm_selection.value}"
        algo_settings = getattr(settings, algo)

        data_count = len(data)
        cluster_count = algo_settings.n_cluster.upper
        min_cluster_size = algo_settings.scoring.min_cluster_size
        min_required_data = min_cluster_size * cluster_count

        if data_count < min_required_data:
            settings_dict = settings.model_dump()
            settings_dict[algo]["n_cluster"]["upper"] = data_count // min_cluster_size
            settings = _settings.ClusteringSettings(**settings_dict)
    
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


def cluster_features(
    df: pd.DataFrame,
    settings: _settings.ClusteringSettings,
):
    # convert data to numpy array
    data = df.to_numpy()
    
    # bypass clustering if cluster count is >= data
    if settings.algorithm_selection not in ["dbscan", "hdbscan"]:
        algo = f"{settings.algorithm_selection.value}"
        algo_settings = getattr(settings, algo)
        if algo_settings.n_cluster.lower >= len(data):
            return np.arange(len(data))

    data = _transform.transform_features(data, settings)

    cluster_labels = _cluster_features(data, settings)

    skip_merge = True
    if not skip_merge and np.unique(cluster_labels).shape[0] == 2:
        cluster_labels = _cluster_merge(cluster_labels, data, settings)

    if settings.cluster_sort.enable:
        cluster_remap_dict = cluster_reorder(df, cluster_labels, settings)

        # remap cluster labels using cluster_remap_dict
        cluster_labels = np.vectorize(cluster_remap_dict.get)(cluster_labels)

    return cluster_labels