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

from typing import Any

from opendsm.common.base_settings import BaseSettings
from opendsm.common.stats.basic import MAD_k

from .transform.normalize_settings import (
    NormalizeChoice,
    NormalizeScope,
    NormalizeSettings,
)
from .transform.settings import (
    fPCATransformSettings,
    MagnitudeFeature,
    WaveletSelection,
    WaveletTransformSettings,
    FeatureTransformSettings,
)
from .metrics.settings import (
    DistanceMetric,
    SmallClusterMode,
    _DEFAULT_SCORE_WEIGHTS,
    ClusterRangeSettings,
    ScoreSettings,
)
from .algorithms.settings import (
    BiKmeansInnerAlgorithms,
    BiKmeansBisectingStrategies,
    BisectingKMeansSettings,
    BisectingKMediansSettings,
    KMediansSettings,
    BirchSettings,
    DbscanDistanceAlgorithm,
    DBSCANSettings,
    HdbscanClusterSelectionMethod,
    HDBSCANSettings,
    SpectralEigenSolver,
    AffinityMatrixOptions,
    SpectralAssignLabels,
    SpectralSettings,
    SortMethod,
    AggregateMethod,
    ClusterSortSettings,
    ClusterAlgorithms,
)


class ClusteringSettings(BaseSettings):
    distance_metric: DistanceMetric = pydantic.Field(
        default=DistanceMetric.EUCLIDEAN,
        description="Distance metric for all clustering algorithms and scoring indices. "
                    "Algorithms use sqeuclidean internally for argmin when metric is euclidean "
                    "(monotonic optimization). Non-Euclidean metrics (manhattan, cosine) use "
                    "the exact metric throughout.",
    )

    min_cluster_size: int = pydantic.Field(
        default=1,
        ge=1,
        description="Minimum number of points a cluster must have to be kept.",
    )

    small_cluster_mode: SmallClusterMode = pydantic.Field(
        default=SmallClusterMode.KEEP,
        description="How to handle clusters smaller than min_cluster_size. "
                    "OUTLIER: relabel as -1 and exclude from scoring — suitable for "
                    "centroid-based and spectral algorithms. "
                    "KEEP: skip the merge step entirely and preserve pre-existing -1 noise labels — "
                    "recommended for density-based algorithms (HDBSCAN, DBSCAN) where -1 has "
                    "genuine noise semantics and algorithm-produced singletons are valid clusters. "
                    "ABSORB: reassign small-cluster points to the nearest large-cluster centroid. "
                    "See SmallClusterMode for full semantics.",
    )

    feature_transform: FeatureTransformSettings = pydantic.Field(
        default_factory=FeatureTransformSettings,
        description="Feature preprocessing and transformation pipeline",
    )

    algorithm_selection: ClusterAlgorithms = pydantic.Field(
        default=ClusterAlgorithms.KMEDIANS,
        description="clustering choice",
    )

    kmedians: KMediansSettings | None = pydantic.Field(
        default_factory=KMediansSettings,
        description="Direct KMedians settings (default algorithm)",
    )

    bisecting_kmedians: BisectingKMediansSettings | None = pydantic.Field(
        default_factory=BisectingKMediansSettings,
        description="Bisecting KMedians settings (legacy)",
    )

    bisecting_kmeans: BisectingKMeansSettings | None = pydantic.Field(
        default_factory=BisectingKMeansSettings,
        description="BisectingKMeans settings",
    )

    birch: BirchSettings | None = pydantic.Field(
        default_factory=BirchSettings,
        description="Birch settings",
    )

    dbscan: DBSCANSettings | None = pydantic.Field(
        default_factory=DBSCANSettings,
        description="DBSCAN settings",
    )

    hdbscan: HDBSCANSettings | None = pydantic.Field(
        default_factory=HDBSCANSettings,
        description="HDBSCAN settings",
    )

    spectral: SpectralSettings | None = pydantic.Field(
        default_factory=SpectralSettings,
        description="Spectral settings",
    )

    spectral_divisive: SpectralSettings | None = pydantic.Field(
        default_factory=SpectralSettings,
        description="Spectral divisive (recursive Fiedler bisection) settings",
    )

    outlier_removal_sigma: float | None = pydantic.Field(
        default=3.0,
        gt=0,
        description=(
            "Post-council outlier removal threshold in Gaussian sigma units. "
            "After the scoring council selects the best k, points whose "
            "deviation from their cluster median exceeds this many sigma "
            "(estimated robustly via MAD × 1.4826) along any principal "
            "component are flagged as outliers and handled according to "
            "the algorithm's small_cluster_mode: KEEP preserves them in a "
            "new cluster (renumbered), OUTLIER relabels as -1, ABSORB "
            "reassigns to the nearest non-outlier cluster centroid. "
            "Default 3.0σ (0.27% of a Gaussian tail). "
            "5.0σ is more conservative (0.00006%). "
            "None disables outlier removal. "
            "Only applied to clusters with >= 5 members."
        ),
    )

    cluster_sort: ClusterSortSettings = pydantic.Field(
        default_factory=ClusterSortSettings,
        description="sort clusters",
    )

    seed: int | None = pydantic.Field(
        default=None,
        ge=0,
        description="seed for random state assignment",
    )

    _seed: int | None = pydantic.PrivateAttr(
        default=None
    )

    _outlier_mad_threshold: float | None = pydantic.PrivateAttr(
        default=None
    )

    @property
    def _argmin_metric(self) -> str:
        """Fast metric for argmin operations (assignment step).

        For Euclidean, sqeuclidean gives the same argmin but avoids sqrt.
        For all other metrics, use the exact metric.
        """
        if self.distance_metric in (
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.SQUARED_EUCLIDEAN,
        ):
            return "sqeuclidean"
        return self.distance_metric.value

    @property
    def _metric_value(self) -> str:
        """The distance metric as a plain string for scipy/sklearn."""
        return self.distance_metric.value

    @pydantic.model_validator(mode="before")
    @classmethod
    def _null_unselected(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        selected_algo = data.get("algorithm_selection", ClusterAlgorithms.KMEDIANS)
        if isinstance(selected_algo, str):
            try:
                selected_algo = ClusterAlgorithms(selected_algo)
            except ValueError:
                pass
        for algo in ClusterAlgorithms:
            if algo != selected_algo:
                data[algo.value] = None
        return data

    @pydantic.model_validator(mode="after")
    def _init_seed(self):
        if self.seed is None and self._seed is None:
            self._seed = np.random.randint(0, 2**32 - 1, dtype=np.int64)
        else:
            self._seed = self.seed

        for transform in [self.feature_transform.wavelet, self.feature_transform.fpca]:
            if transform is not None:
                transform._seed = self._seed

        return self

    @pydantic.model_validator(mode="after")
    def _init_outlier_threshold(self):
        if self.outlier_removal_sigma is not None:
            self._outlier_mad_threshold = self.outlier_removal_sigma / MAD_k
        else:
            self._outlier_mad_threshold = None
        return self

    @pydantic.model_validator(mode="after")
    def _check_cluster_size_mode(self):
        # min_cluster_size and small_cluster_mode are logically coupled:
        #   min_cluster_size=1  ↔  KEEP  (nothing is "small", nothing to merge)
        #   min_cluster_size≥2  ↔  OUTLIER or ABSORB  (merge small clusters)
        if self.min_cluster_size < 2 and self.small_cluster_mode != SmallClusterMode.KEEP:
            raise ValueError(
                f"min_cluster_size={self.min_cluster_size} requires "
                f"small_cluster_mode='keep'. With '{self.small_cluster_mode.value}' "
                f"mode, there are no clusters below the threshold to act on."
            )
        if self.min_cluster_size >= 2 and self.small_cluster_mode == SmallClusterMode.KEEP:
            raise ValueError(
                f"small_cluster_mode='keep' requires min_cluster_size=1. "
                f"KEEP preserves all clusters regardless of size, making "
                f"min_cluster_size={self.min_cluster_size} contradictory."
            )
        return self


if __name__ == "__main__":
    settings = ClusteringSettings()
    print(settings)
