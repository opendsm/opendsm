from __future__ import annotations

import pydantic

from enum import Enum
from typing import TypeAlias

from opendsm.common.base_settings import BaseSettings

MetricWeights: TypeAlias = dict[str, float]


class DistanceMetric(str, Enum):
    """
    what distance method to use
    """
    EUCLIDEAN = "euclidean"
    SEUCLIDEAN = "seuclidean"
    SQUARED_EUCLIDEAN = "sqeuclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class SmallClusterMode(str, Enum):
    """Strategy for handling clusters smaller than ``min_cluster_size``.

    OUTLIER (default)
        Small clusters are relabelled as -1 and excluded from scoring.
        Coverage = n_non_outlier / n_total; if ``outlier_fraction_penalty``
        is True, scores are divided by coverage so solutions that discard
        many samples rank worse.

    ABSORB
        Points in sub-threshold clusters are reassigned to the nearest
        large-cluster centroid.  Pre-existing -1 outliers (e.g. from
        HDBSCAN) are left untouched.  All non-outlier points are scored,
        so coverage is always 1.0 and the penalty never fires.

    KEEP
        No merge step — all clusters are renumbered and kept regardless
        of size.  Pre-existing -1 values (semantic noise from density-based
        algorithms such as HDBSCAN/DBSCAN) are preserved as-is.  This is
        the natural choice for density-based algorithms where (a) outlier
        (-1) carries genuine noise semantics and should not be disturbed,
        and (b) singleton or small clusters identified by the algorithm are
        real clusters that should be scored.  Coverage can still be < 1 if
        the algorithm emits noise points; the outlier-fraction penalty then
        fires normally, penalising labelings with many noise points.

    Recommended pairings
        - k-means / k-medoids / spectral  →  OUTLIER (default)
        - HDBSCAN / DBSCAN                →  KEEP (preserve noise semantics
                                               and algorithm-produced singletons)
        - Either                           →  ABSORB (no outlier budget at all)
    """
    ABSORB  = "absorb"
    OUTLIER = "outlier"
    KEEP    = "keep"

# Complete registry of valid index names.  This is the single source of
# truth — SingleKMetrics.available_indices() and CrossKMetrics.available_indices()
# should return subsets of these.  Avoids circular import between settings
# and the metric classes.
SINGLE_K_INDEX_NAMES: frozenset[str] = frozenset({
    "ball_hall_index",
    "banfeld_raftery_index",
    "c_index",
    "calinski_harabasz_index",
    "cop_index",
    "davies_bouldin_index",
    "density_based_clustering_validation_index",
    "det_ratio_index",
    "duda_hart_index",
    "dunn_index",
    "gamma_index",
    "generalized_dunn_index",
    "i_index",
    "ksq_detw_index",
    "log_det_ratio_index",
    "log_ss_ratio_index",
    "mcclain_rao_index",
    "mean_squared_error_index",
    "negentropy_index",
    "point_biserial_index",
    "s_dbw_index",
    "scott_symons_index",
    "sd_validity_index",
    "silhouette_index",
    "silhouette_median_index",
    "simplified_silhouette_index",
    "sum_of_squared_errors_index",
    "trace_w_index",
    "trace_wb_index",
    "wb_index",
    "xie_beni_index",
})

CROSS_K_INDEX_NAMES: frozenset[str] = frozenset({
    "distortion_jump_index",
    "hartigan_index",
    "krzanowski_lai_index",
    "log_wcss_acceleration_index",
    "xu_index",
})

ALL_INDEX_NAMES: frozenset[str] = SINGLE_K_INDEX_NAMES | CROSS_K_INDEX_NAMES


_DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    # Active by default — 4-member council.  Each measures a different
    # aspect of cluster quality.  Tested on 23 meters (7 regression +
    # 16 random/problem) across k=1..24.
    'point_biserial_index': 1.0,             # distance-label correlation
    'silhouette_median_index': 1.0,          # per-point cluster fit (robust median)
    'xie_beni_index': 1.0,                   # density (min inter-cluster distance)
    'wb_index': 1.0,                         # within/between scatter ratio
    # wb_index has a moderate-k preference (median=4, std=5.6) that
    # counterbalances the low-k bias of the other three members.
    # Recovers higher k on undersplit meters (e.g. k=2→5 on reg_1501,
    # k=2→3 on solar meter 3875377852).

    # Tested and rejected:
    # - davies_bouldin_index: separation/compactness ratio, median=3 but
    #   std=5.7 — occasionally drifts to k=14-23, causing oversplitting.

    # Viable indices (inactive by default)
    'calinski_harabasz_index': 0.0,          # variance ratio (F-statistic)
    'cop_index': 0.0,                        # compactness (similar to silhouette)
    'davies_bouldin_index': 0.0,             # separation / compactness ratio
    'dunn_index': 0.0,                       # min inter / max intra distance
    'generalized_dunn_index': 0.0,           # min-max separation ratio
    'i_index': 0.0,                          # scatter, separation, and built-in 1/k penalty
    'negentropy_index': 0.0,                 # information-theoretic
    'sd_validity_index': 0.0,                # scatter + separation
    'silhouette_index': 0.0,                 # per-point cluster fit (mean)
    'simplified_silhouette_index': 0.0,      # per-point fit vs centroid
}


class KPenaltySettings(BaseSettings):
    """Low-k complexity penalty for Schulze voting.

    Penalizes low k values (especially k=2) where every dataset trivially
    clusters well.  Applied to normalized Schulze scores after MAD-clipping:
    the penalty shifts each candidate's score toward worst by
    ``strength / k^rate * (1 - normalized_score)``.

    Higher k values are less affected because 1/k^rate decays asymptotically.
    """
    enabled: bool = pydantic.Field(
        default=True,
        description="Enable the low-k complexity penalty.",
    )

    strength: float = pydantic.Field(
        default=0.0,
        ge=0,
        le=2,
        description=(
            "Magnitude of the penalty.  0 (default) = no effect.  "
            "Tested at 0.3 with rate=1.0: fixes some undersplit meters "
            "(k=2→5) but oversplits others (k=3→16) when the council has "
            "low confidence.  Left disabled until a confidence-gated "
            "variant is developed.  "
            "0.3 = mild parsimony (15% at k=2, 6% at k=5, 3% at k=10).  "
            "1.0 = full penalty.  >1 = more aggressive."
        ),
    )

    rate: float = pydantic.Field(
        default=1.0,
        gt=0,
        le=3,
        description=(
            "Decay rate.  Controls how quickly the penalty diminishes as k "
            "increases.  rate=1.0 (default): 1/k decay.  rate=0.5: slower "
            "decay (penalizes mid-range k more).  rate=2.0: faster decay "
            "(only k=2-3 significantly penalized)."
        ),
    )


class ScoreSettings(BaseSettings):
    """maximum number of non-outlier clusters"""
    max_non_outlier_cluster_count: int = pydantic.Field(
        default=200,
        ge=1,
    )

    max_scoring_samples: int | None = pydantic.Field(
        default=10_000,
        ge=4,
        description="Maximum samples for scoring index evaluation.  When the dataset "
                    "exceeds this size, a fixed random subsample is drawn once and "
                    "shared across all k values so index comparisons remain consistent. "
                    "This bounds the distance matrix to max_scoring_samples² entries, "
                    "preventing OOM at large n.  None means no subsampling.",
    )

    """scoring metric weights — keys must be valid *_index properties on SingleKMetrics"""
    weights: MetricWeights = pydantic.Field(
        default_factory=lambda: dict(_DEFAULT_SCORE_WEIGHTS),
    )

    window_size: float = pydantic.Field(
        default=0.0,
        ge=0,
    )

    """distance metric for clustering"""
    distance_metric: DistanceMetric = pydantic.Field(
        default=DistanceMetric.EUCLIDEAN,
    )

    null_test_alpha: float = pydantic.Field(
        default=0.10,
        gt=0,
        lt=1,
        description="Significance threshold for the cluster structure null tests. "
                    "Lower values require stronger evidence of structure before "
                    "clustering (fewer false positives, more false negatives). "
                    "Higher values are more permissive, catching weaker cluster "
                    "separation at the cost of occasionally clustering "
                    "structureless data. Only applies when k=1 is a candidate.",
    )

    outlier_fraction_penalty: bool = pydantic.Field(
        default=True,
        description="Penalise labelings that discard a large fraction of samples as outliers. "
                    "Divides each finite score by coverage = n_non_outlier / n_total so "
                    "high-outlier solutions rank worse. No effect when coverage = 1.",
    )

    k_penalty: KPenaltySettings = pydantic.Field(
        default_factory=KPenaltySettings,
        description="Low-k complexity penalty settings for Schulze voting.",
    )

    @pydantic.model_validator(mode="after")
    def _check_weights(self):
        invalid = set(self.weights.keys()) - ALL_INDEX_NAMES
        if invalid:
            raise ValueError(
                f"Unknown metric(s): {invalid}. "
                f"Valid metrics: {sorted(ALL_INDEX_NAMES)}"
            )

        if not any(w > 0 for w in self.weights.values()):
            raise ValueError("At least one scoring weight must be greater than 0")

        for name, w in self.weights.items():
            if w < 0:
                raise ValueError(f"Weight for '{name}' must be >= 0, got {w}")

        return self


class ClusterRangeSettings(BaseSettings):
    """lower bound for number of clusters"""
    lower: int = pydantic.Field(
        default=1,
        ge=1,
    )

    """upper bound for number of clusters"""
    upper: int = pydantic.Field(
        default=24,
        ge=1,
    )

    @pydantic.model_validator(mode="after")
    def _check_n_cluster_range(self):
        if self.lower > self.upper:
            raise ValueError(
                "'n_cluster_lower' must be <= 'n_cluster_upper'"
            )

        return self
