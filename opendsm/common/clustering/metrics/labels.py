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

from functools import cached_property

import numpy as np
import pydantic
from scipy.spatial.distance import pdist, squareform

from opendsm.common.pydantic_utils import ArbitraryPydanticModel
from opendsm.common.clustering.metrics.single_k_metrics import SingleKMetrics
from opendsm.common.clustering.metrics.label_ops import prepare_labels
from opendsm.common.clustering.metrics import cross_k_metrics as _cross_k
from opendsm.common.clustering.metrics import selection as _selection
from opendsm.common.clustering.metrics.settings import ScoreSettings, SmallClusterMode


class DistanceProvider:
    """Lazily computes and caches a float32 pairwise distance matrix.

    Shared across all SingleKMetrics instances for the same dataset so that
    the O(n²p) pdist is computed at most once — and only when an index actually
    needs it.  Indices that don't use the distance matrix (e.g. davies_bouldin,
    calinski_harabasz) never trigger computation at all.
    """

    def __init__(self, data: np.ndarray) -> None:
        self._data = data
        self._dist: np.ndarray | None = None

    def get(self) -> np.ndarray:
        if self._dist is None:
            self._dist = squareform(pdist(self._data.astype(np.float32, copy=False))).astype(np.float32)
        return self._dist

    def get_submatrix(self, kept: np.ndarray) -> np.ndarray:
        return self.get()[np.ix_(kept, kept)]


class LabelStore(ArbitraryPydanticModel):
    """Storage layer: accumulates per-k labelings and their SingleKMetrics.

    Algorithms construct a LabelStore, call add() for each labeling, and
    return it.  ClusteringResult extends this with selection logic.
    """

    data: np.ndarray = pydantic.Field(exclude=True, repr=False)

    score_settings: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings,
    )

    min_cluster_size: int = pydantic.Field(default=1, ge=1)

    small_cluster_mode: SmallClusterMode = pydantic.Field(
        default=SmallClusterMode.KEEP,
    )

    seed: int = pydantic.Field(default=0)

    n_cluster_lower: int | None = pydantic.Field(default=None)

    _labels_store: dict[int, list[SingleKMetrics]] = pydantic.PrivateAttr(
        default_factory=dict,
    )
    _labels_k_best_override: dict[int, int] = pydantic.PrivateAttr(
        default_factory=dict,
    )
    _eigengap_scores: dict[int, float] | None = pydantic.PrivateAttr(default=None)
    _eigengap_weight: float = pydantic.PrivateAttr(default=0.0)
    _insertion_order: list[SingleKMetrics | None] = pydantic.PrivateAttr(
        default_factory=list,
    )
    _submatrix_cache: dict = pydantic.PrivateAttr(default_factory=dict)
    _scoring_idx: np.ndarray | None = pydantic.PrivateAttr(default=None)
    _scoring_data: np.ndarray | None = pydantic.PrivateAttr(default=None)

    @pydantic.model_validator(mode="after")
    def _init_scoring_subsample(self):
        """Draw a fixed random subsample for scoring when n exceeds the budget.

        The same row indices are used for every k, ensuring consistent
        cross-k comparisons in the Schulze vote.  The subsample is drawn
        once at construction and never changes.
        """
        max_n = self.score_settings.max_scoring_samples
        n = self.data.shape[0]
        if max_n is not None and n > max_n:
            rng = np.random.default_rng(self.seed)
            self._scoring_idx = np.sort(rng.choice(n, max_n, replace=False))
            self._scoring_data = self.data[self._scoring_idx].astype(np.float32, copy=False)
        else:
            self._scoring_idx = None
            self._scoring_data = None
        return self

    # ------------------------------------------------------------------
    # Shared distance provider — computes the O(n²p) distance matrix at most
    # once and only when an index actually reads it.  All SingleKMetrics for
    # the same dataset share the same DistanceProvider instance.
    # When scoring is subsampled, the provider wraps the subsampled data.
    # ------------------------------------------------------------------

    @cached_property
    def _distance_provider(self) -> DistanceProvider:
        if self._scoring_data is not None:
            return DistanceProvider(self._scoring_data)
        return DistanceProvider(self.data)

    def add(self, k: int, labels: np.ndarray) -> SingleKMetrics | None:
        """Add labels for a given k.

        Applies small-cluster policy, optionally subsamples for scoring,
        and stores the result.  Invalidates all caches.
        """
        # Always compute the full-data merged labels (for output).
        full_merged, _, _, full_coverage = prepare_labels(
            labels, self.data, self.score_settings, self.n_cluster_lower,
            min_cluster_size=self.min_cluster_size,
            small_cluster_mode=self.small_cluster_mode,
        )

        # Apply scoring subsample: use the fixed row subset drawn at init.
        # This keeps all k evaluations on identical rows.
        if self._scoring_idx is not None:
            scoring_labels = labels[self._scoring_idx]
            scoring_data = self._scoring_data
        else:
            scoring_labels = labels
            scoring_data = self.data

        merged, data_clean, labels_clean, coverage = prepare_labels(
            scoring_labels, scoring_data, self.score_settings, self.n_cluster_lower,
            min_cluster_size=self.min_cluster_size,
            small_cluster_mode=self.small_cluster_mode,
        )

        if data_clean is None:
            self._insertion_order.append(None)
            return None

        lm = SingleKMetrics(
            data=data_clean.astype(np.float32, copy=False),
            labels=labels_clean,
            distance_metric=self.score_settings.distance_metric,
            seed=self.seed,
        )

        if data_clean.shape[0] == scoring_data.shape[0]:
            lm._dist_provider = self._distance_provider
        else:
            # Outlier-trimmed submatrix from the (possibly subsampled) data.
            kept = np.where(merged != -1)[0]
            key = kept.tobytes()
            if key not in self._submatrix_cache:
                self._submatrix_cache[key] = DistanceProvider(data_clean.astype(np.float32, copy=False))
                self._submatrix_cache[key]._dist = self._distance_provider.get_submatrix(kept)
            lm._dist_provider = self._submatrix_cache[key]
        # Store the full-data merged labels for output (not the subsampled ones).
        lm._merged_full = full_merged
        eff_coverage = full_coverage if self.score_settings.outlier_fraction_penalty else 1.0
        lm._coverage = eff_coverage
        self._labels_store.setdefault(k, []).append(lm)
        self._insertion_order.append(lm)
        self._invalidate_caches()
        return lm

    def _add_scored(self, k: int, lm: SingleKMetrics) -> None:
        """Add a pre-scored SingleKMetrics directly, bypassing prepare_labels."""
        self._labels_store.setdefault(k, []).append(lm)
        self._insertion_order.append(lm)
        self._invalidate_caches()

    @classmethod
    def from_labels(
        cls,
        data: np.ndarray,
        k_to_labels: dict[int, np.ndarray | list[np.ndarray]],
        score_settings: ScoreSettings | None = None,
        seed: int = 0,
        n_cluster_lower: int | None = None,
    ):
        """Batch construction from a dict of k → label array(s)."""
        if score_settings is None:
            score_settings = ScoreSettings()
        obj = cls(data=data, score_settings=score_settings, seed=seed, n_cluster_lower=n_cluster_lower)
        for k, labels in k_to_labels.items():
            label_list = labels if isinstance(labels, list) else [labels]
            for lbl in label_list:
                obj.add(k, lbl)
        return obj

    def _invalidate_caches(self):
        for key in list(self.__dict__):
            if key.startswith("_cache_"):
                del self.__dict__[key]

    def _set_labels_k_best(self, k: int, index: int):
        """Override within-k selection for a given k."""
        if k not in self._labels_store:
            raise IndexError(f"No labels at k={k}, index={index}")
        if not (0 <= index < len(self._labels_store[k])):
            raise IndexError(f"No labels at k={k}, index={index}")
        self._labels_k_best_override[k] = index
        self._invalidate_caches()

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def k_values(self) -> list[int]:
        cached = self.__dict__.get("_cache_k_values")
        if cached is not None:
            return cached
        result = sorted(self._labels_store)
        self.__dict__["_cache_k_values"] = result
        return result

    def __getitem__(self, k: int) -> list[SingleKMetrics]:
        return self._labels_store[k]


class ClusteringResult(LabelStore):
    """Clustering result with selection and public API.

    Public API
    ----------
    labels : np.ndarray
        Best merged cluster labels (includes -1 for outliers).
    k : int
        Number of non-outlier clusters in the best labeling.
    metrics : SingleKMetrics
        Per-k metrics for the best labeling.
    cross_k_metrics : CrossKMetrics
        Cross-k indices (Krzanowski-Lai, Hartigan, etc.).

    Construction
    ------------
    >>> result = ClusteringResult(data=X, score_settings=score_settings, seed=42)
    >>> result.add(3, my_labels)
    >>> result.labels  # best merged labels
    """

    @property
    def labels(self) -> np.ndarray:
        """Best merged labels (with -1 outliers)."""
        return self._labels_best_merged

    @property
    def k(self) -> int:
        """Number of non-outlier clusters in the best labeling."""
        return self._labels_best.valid_label_count

    @property
    def metrics(self) -> SingleKMetrics:
        """Per-k metrics for the best labeling."""
        return self._labels_best

    @property
    def selection_confidence(self) -> float:
        """Confidence in k-selection, in [0, 1].

        Derived from Schulze voting: the winner's share of total preference
        strength.  0 = all candidates tied; 1 = winner dominates.  Accessing
        this property triggers k-selection if not already computed.
        """
        _ = self._labels_best  # ensure selection has run
        return self.__dict__.get("_cache_selection_confidence", 0.0)

    @property
    def has_cluster_structure(self) -> bool:
        """Whether the data has detectable cluster structure.

        Combines null test p-values from ``cross_k_metrics`` via
        grouped agreement — see :func:`cross_k_metrics.has_cluster_structure`.
        """
        return _cross_k.has_cluster_structure(
            self.cross_k_metrics, alpha=self.score_settings.null_test_alpha,
        )

    @property
    def cross_k_metrics(self) -> _cross_k.CrossKMetrics:
        """Cross-k indices (require WCSS across multiple k values)."""
        cached = self.__dict__.get("_cache_cross_k_metrics")
        if cached is not None:
            return cached
        k_best = self._labels_k_best
        wcss = {k: lm._WCSS for k, lm in k_best.items()}
        n_scored = {k: lm.n_total for k, lm in k_best.items()}
        scoring_data = self._scoring_data if self._scoring_data is not None else self.data
        # n_samples must match the scoring data, not the full dataset,
        # because WCSS values and indices operate on the subsample.
        n_scoring = scoring_data.shape[0]
        # Null tests use original (untransformed) data when available,
        # so they detect genuine cluster structure rather than transform
        # artifacts.  WCSS-based cross-k indices don't use raw_data at all.
        null_test_data = self.__dict__.get("_null_test_data", scoring_data)
        result = _cross_k.CrossKMetrics(
            wcss_by_k=wcss,
            n_scored_by_k=n_scored,
            k_values=self.k_values,
            n_features=self.n_features,
            n_samples=n_scoring,
            raw_data=null_test_data,
            seed=self.seed,
        )
        self.__dict__["_cache_cross_k_metrics"] = result
        return result

    @property
    def _labels_k_best(self) -> dict[int, SingleKMetrics]:
        """Best SingleKMetrics per k (cached)."""
        cached = self.__dict__.get("_cache_labels_k_best")
        if cached is not None:
            return cached

        council = self.score_settings.weights
        result = {}
        for k, lms in self._labels_store.items():
            if k in self._labels_k_best_override:
                result[k] = lms[self._labels_k_best_override[k]]
            elif len(lms) == 1:
                result[k] = lms[0]
            else:
                result[k] = _selection.select_best_within_k(lms, council)

        self.__dict__["_cache_labels_k_best"] = result
        return result

    @property
    def _labels_best(self) -> SingleKMetrics:
        """Single best SingleKMetrics via flat Schulze voting (cached)."""
        cached = self.__dict__.get("_cache_labels_best")
        if cached is not None:
            return cached

        result = self._compute_labels_best()
        self.__dict__["_cache_labels_best"] = result
        return result

    def _compute_labels_best(self) -> SingleKMetrics:
        candidates: list[SingleKMetrics | None]
        if self._insertion_order:
            candidates = list(self._insertion_order)
        else:
            candidates = []
            for k in sorted(self._labels_store):
                candidates.extend(self._labels_store[k])

        all_valid = [c for c in candidates if c is not None]
        if not all_valid:
            if not candidates:
                raise ValueError("No labels have been added")

            lower = self.n_cluster_lower
            raise ValueError(
                f"No clustering met the cluster-count constraints "
                f"(n_cluster_lower={lower}): every candidate labeling was "
                f"rejected — each collapsed below the lower bound or exceeded "
                f"the maximum cluster count. The data may be degenerate."
            )

        if len(all_valid) == 1:
            self.__dict__["_cache_selection_confidence"] = 1.0

            return all_valid[0]

        # A single specified k needs no cross-k selection — return the
        # within-k best directly and skip the gate and council.
        if len(self.k_values) == 1:
            only_k = self.k_values[0]
            self.__dict__["_cache_selection_confidence"] = 1.0

            return self._labels_k_best[only_k]

        # Gap statistic gate: check if any k>1 has genuine cluster
        # structure exceeding a uniform null.  If not, return k=1
        # (the trivial single-cluster labeling) with confidence 0.
        k_best = self._labels_k_best
        if 1 in k_best and not self.has_cluster_structure:
            self.__dict__["_cache_selection_confidence"] = 0.0
            return k_best[1]

        # Exclude k=1 from the council — all its indices are NaN (abstain),
        # so it can't meaningfully compete.  The gate already decided
        # whether structure exists; the council only ranks k >= 2.
        lm_to_k: dict[int, int] = {}
        for k, lms in self._labels_store.items():
            for lm in lms:
                lm_to_k[id(lm)] = k
        council_candidates = [
            c if c is None or lm_to_k.get(id(c), 0) >= 2 else None
            for c in candidates
        ]

        council = dict(self.score_settings.weights)
        extra_scores_list = self._build_cross_k_extra_scores(council_candidates, council)

        # Inject eigengap voter if available (spectral_divisive path)
        if extra_scores_list is None:
            extra_scores_list = [None] * len(council_candidates)
        if self._eigengap_scores is not None and self._eigengap_weight > 0:
            eigengap_name = "eigengap"
            council[eigengap_name] = self._eigengap_weight
            for i, c in enumerate(council_candidates):
                if c is None:
                    continue
                k = lm_to_k.get(id(c), 0)
                score = self._eigengap_scores.get(k, float('nan'))
                if extra_scores_list[i] is None:
                    extra_scores_list[i] = {eigengap_name: score}
                else:
                    extra_scores_list[i][eigengap_name] = score

        candidate_k_values = [
            lm_to_k.get(id(c), 1) if c is not None else 1
            for c in council_candidates
        ]

        k_pen = self.score_settings.k_penalty
        winner_idx, confidence = _selection.select_best_across_k(
            council_candidates, council, self.score_settings.window_size,
            extra_scores_list=extra_scores_list,
            candidate_k_values=candidate_k_values,
            k_penalty_strength=k_pen.strength if k_pen.enabled else 0.0,
            k_penalty_rate=k_pen.rate,
        )

        self.__dict__["_cache_selection_confidence"] = confidence

        winner = council_candidates[winner_idx]
        if winner is not None:
            return winner

        # Council could not score the candidates; fall back to the labeling
        # with the fewest clusters.  Every valid candidate cleared
        # prepare_labels, so its cluster count is >= n_cluster_lower — the
        # fallback can never drop below the user's lower bound.
        smallest_k_winner = min(all_valid, key=lambda c: c.valid_label_count)

        return smallest_k_winner

    def _build_cross_k_extra_scores(
        self,
        candidates: list[SingleKMetrics | None],
        council: dict[str, float],
    ) -> list[dict[str, float]] | None:
        cross_k_names = set(_cross_k.CrossKMetrics.available_indices())
        active_cross_k = {m for m, w in council.items() if m in cross_k_names and w > 0}

        if not active_cross_k or len(self.k_values) < 2:
            return None

        ckm = self.cross_k_metrics

        k_to_extra: dict[int, dict[str, float]] = {}
        for k in self.k_values:
            scores: dict[str, float] = {}
            for metric in active_cross_k:
                idx_dict: dict[int, float] = getattr(ckm, metric, {})
                scores[metric] = idx_dict.get(k, float('nan'))
            k_to_extra[k] = scores

        lm_to_k: dict[int, int] = {}
        for k, lms in self._labels_store.items():
            for lm in lms:
                lm_to_k[id(lm)] = k

        extra_scores_list = []
        for c in candidates:
            if c is None:
                extra_scores_list.append({})
            else:
                intended_k = lm_to_k.get(id(c), c.label_count)
                extra_scores_list.append(k_to_extra.get(intended_k, {}))

        return extra_scores_list

    @property
    def _labels_best_merged(self) -> np.ndarray:
        """Full merged labels (including -1 outliers) for the best labeling (cached)."""
        cached = self.__dict__.get("_cache_labels_best_merged")
        if cached is not None:
            return cached

        result = self._labels_best._merged_full
        self.__dict__["_cache_labels_best_merged"] = result

        return result
