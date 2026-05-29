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

from opendsm.common.clustering import (
    settings as _settings,
    transform as _transform,
)

from opendsm.common.clustering.algorithms.bisect_k_means import bisect_k_means as _bisecting_kmeans_clustering
from opendsm.common.clustering.algorithms.bisect_k_medians import bisect_k_medians as _bisecting_kmedians_clustering
from opendsm.common.clustering.algorithms.k_medians import kmedians as _kmedians_clustering
from opendsm.common.clustering.algorithms.birch import birch as _birch_clustering
from opendsm.common.clustering.algorithms.dbscan import dbscan as _dbscan_clustering
from opendsm.common.clustering.algorithms.hdbscan import hdbscan as _hdbscan_clustering
from opendsm.common.clustering.algorithms.spectral import spectral as _spectral_clustering
from opendsm.common.clustering.algorithms.spectral import spectral_divisive as _spectral_divisive_clustering
from opendsm.common.clustering.metrics.labels import ClusteringResult
from opendsm.common.clustering.metrics.label_ops import remove_outliers_mad
from opendsm.common.clustering.metrics.settings import SmallClusterMode


def _build_label_remap(
    cluster_labels: np.ndarray,
    settings: _settings.ClusteringSettings,
) -> dict:
    sort_method = settings.cluster_sort.method
    reverse = settings.cluster_sort.reverse

    non_outlier_mask = cluster_labels >= 0
    non_outlier_labels = cluster_labels[non_outlier_mask]
    uniq_labels, counts = np.unique(non_outlier_labels, return_counts=True)
    n_clusters = len(uniq_labels)

    if sort_method == "size":
        order = np.argsort(counts)
        features_index = uniq_labels[order]

    elif sort_method == "peak":
        raise NotImplementedError("'peak' sort method is not yet implemented")

    else:
        raise ValueError(f"Unsupported sort method: {sort_method!r}")

    cluster_map = {-1: -1}
    if not reverse:
        cluster_map.update({features_index[i]: i for i in range(n_clusters)})
    else:
        cluster_map.update({features_index[i]: n_clusters - 1 - i for i in range(n_clusters)})

    return cluster_map


from opendsm.common.clustering.algorithms.protocol import ClusterAlgorithm
_ALGORITHM_DISPATCH: dict[_settings.ClusterAlgorithms, ClusterAlgorithm] = {
    _settings.ClusterAlgorithms.KMEDIANS: _kmedians_clustering,
    _settings.ClusterAlgorithms.BISECTING_KMEDIANS: _bisecting_kmedians_clustering,
    _settings.ClusterAlgorithms.BISECTING_KMEANS: _bisecting_kmeans_clustering,
    _settings.ClusterAlgorithms.BIRCH: _birch_clustering,
    _settings.ClusterAlgorithms.DBSCAN: _dbscan_clustering,
    _settings.ClusterAlgorithms.HDBSCAN: _hdbscan_clustering,
    _settings.ClusterAlgorithms.SPECTRAL: _spectral_clustering,
    _settings.ClusterAlgorithms.SPECTRAL_DIVISIVE: _spectral_divisive_clustering,
}


def _k_range_algo_settings(settings: _settings.ClusteringSettings):
    """Return the active algorithm's settings object, or None for density-based algorithms."""
    if settings.algorithm_selection in (_settings.ClusterAlgorithms.DBSCAN, _settings.ClusterAlgorithms.HDBSCAN):
        return None
    return getattr(settings, settings.algorithm_selection.value)


def _cap_cluster_range(
    settings: _settings.ClusteringSettings,
    data: np.ndarray,
) -> _settings.ClusteringSettings:
    """Cap the upper cluster count when the dataset is too small to support it."""
    algo_settings = _k_range_algo_settings(settings)
    if algo_settings is None:
        return settings

    algo = settings.algorithm_selection.value
    data_count = len(data)
    min_cluster_size = settings.min_cluster_size
    new_upper = data_count // min_cluster_size

    if new_upper >= algo_settings.n_cluster.upper:
        return settings

    new_n_cluster = algo_settings.n_cluster.model_copy(update={"upper": new_upper})
    new_algo_settings = algo_settings.model_copy(update={"n_cluster": new_n_cluster})
    return settings.model_copy(update={algo: new_algo_settings})


def _cluster_features(
    data: np.ndarray,
    settings: _settings.ClusteringSettings,
) -> ClusteringResult:
    """Extract algorithm-specific settings and call the selected algorithm."""
    settings = _cap_cluster_range(settings, data)
    cluster_fcn = _ALGORITHM_DISPATCH[settings.algorithm_selection]
    return cluster_fcn(data, settings)


def _run_pipeline(
    df: pd.DataFrame,
    settings: _settings.ClusteringSettings,
) -> tuple[ClusteringResult | None, np.ndarray]:
    """Shared preprocessing pipeline. Returns (result, cluster_labels)."""
    data = df.to_numpy(dtype=np.float32, na_value=np.nan)

    algo_settings = _k_range_algo_settings(settings)
    if algo_settings is not None:
        if algo_settings.n_cluster.lower >= len(data):
            return None, np.arange(len(data))

    transform_result = _transform.transform_features(data, settings)
    data = transform_result.data
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    result = _cluster_features(data, settings)
    # Null tests use original (unnormalized) data — no normalization
    # artifacts, no variance cap distortion. Detects whether the raw
    # data has genuine cluster structure.
    result.__dict__["_null_test_data"] = transform_result.null_test_data
    cluster_labels = result.labels

    # Post-council MAD-based outlier removal
    if settings._outlier_mad_threshold is not None and result.k > 0:
        cluster_labels = remove_outliers_mad(
            data, cluster_labels, settings._outlier_mad_threshold,
            small_cluster_mode=settings.small_cluster_mode,
        )

    if settings.cluster_sort.enable:
        cluster_remap_dict = _build_label_remap(cluster_labels, settings)
        cluster_labels = np.vectorize(cluster_remap_dict.__getitem__)(cluster_labels)

    return result, cluster_labels


def cluster_features(
    df: pd.DataFrame,
    settings: _settings.ClusteringSettings,
) -> np.ndarray:
    """Cluster features and return the best cluster label array."""
    _, cluster_labels = _run_pipeline(df, settings)
    return cluster_labels


def cluster_result(
    df: pd.DataFrame,
    settings: _settings.ClusteringSettings,
) -> ClusteringResult | None:
    """Cluster features and return the full ClusteringResult with metrics."""
    result, _ = _run_pipeline(df, settings)
    return result
