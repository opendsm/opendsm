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

"""Tests for clustering metrics: SingleKMetrics, CrossKMetrics, Labels, selection."""

from __future__ import annotations

import numpy as np
import pytest

from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, davies_bouldin_score

from opendsm.common.clustering.metrics.single_k_metrics import SingleKMetrics
from opendsm.common.clustering.metrics.cross_k_metrics import CrossKMetrics
from opendsm.common.clustering.metrics.dbcv import dbcv, dbcv_prevalidated
from opendsm.common.clustering.metrics.labels import ClusteringResult
from opendsm.common.clustering.metrics import selection
from opendsm.common.clustering.metrics.label_ops import prepare_labels
from opendsm.common.clustering.metrics.settings import (
    ClusterRangeSettings,
    ScoreSettings,
    SmallClusterMode,
    SINGLE_K_INDEX_NAMES,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def well_separated_lm():
    """Three tight, well-separated clusters — all active indices should be finite."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]])
    X = np.vstack([rng.normal(c, 0.4, (20, 2)) for c in centers])
    labels = np.repeat([0, 1, 2], 20)
    return SingleKMetrics(data=X, labels=labels)


@pytest.fixture(scope='module')
def all_singletons_lm():
    """Every cluster has exactly one point — no within-cluster distances."""
    X = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    return SingleKMetrics(data=X, labels=np.array([0, 1, 2]))


@pytest.fixture(scope='module')
def coincident_lm():
    """All points at the origin — within and between distances both zero."""
    X = np.zeros((10, 2))
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    return SingleKMetrics(data=X, labels=labels)


@pytest.fixture(scope='module')
def ckm_simple():
    """CrossKMetrics with hand-checkable WCSS values (p=2, n=30)."""
    return CrossKMetrics(
        wcss_by_k={2: 100.0, 3: 60.0, 4: 50.0},
        k_values=[2, 3, 4],
        n_features=2,
        n_samples=30,
    )


@pytest.fixture(scope='module')
def well_separated_data():
    """Three tight clusters — raw (X, labels) for use in Labels tests."""
    rng = np.random.default_rng(1)
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]])
    X = np.vstack([rng.normal(c, 0.4, (15, 2)) for c in centers])
    labels = np.repeat([0, 1, 2], 15)
    return X, labels


# ── SingleKMetrics ────────────────────────────────────────────────────────────

class TestSingleKMetricsAvailableIndices:
    @pytest.mark.parametrize('cls', [SingleKMetrics, CrossKMetrics])
    def test_returns_list_of_index_names(self, cls):
        indices = cls.available_indices()
        assert isinstance(indices, list)
        assert all(name.endswith('_index') for name in indices)

    def test_contains_default_council_members(self):
        indices = set(SingleKMetrics.available_indices())
        for name in [
            'calinski_harabasz_index', 'c_index', 'gamma_index',
            'point_biserial_index', 'silhouette_median_index', 'davies_bouldin_index',
        ]:
            assert name in indices

    @pytest.mark.parametrize('name', [
        'krzanowski_lai_index', 'hartigan_index', 'distortion_jump_index',
        'log_wcss_acceleration_index', 'xu_index',
    ])
    def test_contains_cross_k_indices(self, name):
        assert name in set(CrossKMetrics.available_indices())


class TestSingleKMetricsNanSemantics:
    """Structurally undefined cases return NaN (abstain), not 0 or inf."""

    @pytest.mark.parametrize('index', [
        'c_index',
        'gamma_index',
        'point_biserial_index',
    ])
    def test_singleton_nan(self, all_singletons_lm, index):
        """No within-cluster pairs/distances -> ratio/correlation undefined -> NaN."""
        assert np.isnan(getattr(all_singletons_lm, index))

    @pytest.mark.parametrize('index', [
        'point_biserial_index',
        'duda_hart_index',
        'mcclain_rao_index',
    ])
    def test_coincident_nan(self, coincident_lm, index):
        """All pairwise distances zero -> undefined -> NaN."""
        assert np.isnan(getattr(coincident_lm, index))

    def test_gamma_index_nan_for_all_tied_pairs(self):
        """All within/between distances equal -> s_plus + s_minus = 0 -> NaN."""
        lm = SingleKMetrics(data=np.zeros((6, 2)), labels=np.array([0, 0, 0, 1, 1, 1]))
        assert np.isnan(lm.gamma_index)


# ── CrossKMetrics ─────────────────────────────────────────────────────────────

class TestCrossKMetricsBoundaryNaN:
    """Indices that need k+/-1 return NaN at boundary k values."""

    def test_hartigan_nan_at_k_max(self, ckm_simple):
        """Hartigan needs k+1; k_max=4 has no k=5 -> NaN."""
        assert np.isnan(ckm_simple.hartigan_index[4])

    @pytest.mark.parametrize('index', [
        'krzanowski_lai_index',
        'log_wcss_acceleration_index',
    ])
    def test_boundary_nan_at_both_ends(self, ckm_simple, index):
        """KL and log-WCSS-accel need both k-1 and k+1; boundaries -> NaN."""
        vals = getattr(ckm_simple, index)
        assert np.isnan(vals[2])
        assert np.isnan(vals[4])

    def test_distortion_jump_nan_at_k_min(self, ckm_simple):
        """Distortion jump needs k-1; k_min=2 has no k=1 -> NaN."""
        assert np.isnan(ckm_simple.distortion_jump_index[2])

    @pytest.mark.parametrize('k', [2, 3, 4])
    def test_xu_finite_for_all_k(self, ckm_simple, k):
        """Xu index needs only WCSS at k itself -> always finite."""
        assert np.isfinite(ckm_simple.xu_index[k])


class TestCrossKMetricsExactValues:
    """Verify computed values for the hand-checked wcss={2:100,3:60,4:50} fixture."""

    def test_hartigan_exact(self, ckm_simple):
        # H(k) = (W(k)/W(k+1) - 1) * (n - k - 1)
        # H(2) = (100/60 - 1) * 27 = (2/3) * 27 = 18.0
        # H(3) = (60/50 - 1) * 26 = 0.2 * 26 = 5.2
        assert ckm_simple.hartigan_index[2] == pytest.approx(18.0, rel=1e-6)
        assert ckm_simple.hartigan_index[3] == pytest.approx(5.2, rel=1e-6)

    def test_krzanowski_lai_exact(self, ckm_simple):
        # diff[3] = 2*100 - 3*60 = 20; diff[4] = 3*60 - 4*50 = -20
        # KL(3) = -|20/-20| = -1.0
        assert ckm_simple.krzanowski_lai_index[3] == pytest.approx(-1.0, rel=1e-6)

    def test_distortion_jump_sign(self, ckm_simple):
        # J(k) is negated (maximize->minimize); jump at k=3,4 is positive raw -> negative stored
        assert ckm_simple.distortion_jump_index[3] < 0
        assert ckm_simple.distortion_jump_index[4] < 0

    def test_log_wcss_acceleration_sign(self, ckm_simple):
        # Second difference of log-WCSS; negated for minimization
        assert ckm_simple.log_wcss_acceleration_index[3] < 0

    def test_xu_monotone_for_this_wcss(self, ckm_simple):
        # For these specific WCSS values Xu(3) < Xu(2)
        xu = ckm_simple.xu_index
        assert xu[3] < xu[2]


# ── Labels container ──────────────────────────────────────────────────────────

class TestLabelsContainer:
    def test_add_and_labels_property(self, well_separated_data):
        X, labels_k3 = well_separated_data
        lbl = ClusteringResult(data=X, score_settings=ScoreSettings())
        lbl.add(3, labels_k3)
        out = lbl.labels
        assert isinstance(out, np.ndarray)
        assert out.shape[0] == X.shape[0]

    def test_k_property_matches_cluster_count(self, well_separated_data):
        X, labels = well_separated_data
        lbl = ClusteringResult.from_labels(X, {3: labels})
        assert lbl.k == 3

    def test_from_labels_batch_init(self, well_separated_data):
        X, labels_k3 = well_separated_data
        labels_k2 = np.repeat([0, 1], [22, 23])
        lbl = ClusteringResult.from_labels(X, {2: labels_k2, 3: labels_k3})
        assert set(lbl.k_values) == {2, 3}

    def test_best_labeling_selects_well_separated_k(self, well_separated_data):
        """For very tight clusters, the true k=3 should win over k=2."""
        X, labels_k3 = well_separated_data
        labels_k2 = np.repeat([0, 1], [22, 23])
        lbl = ClusteringResult.from_labels(X, {2: labels_k2, 3: labels_k3})
        assert lbl.k == 3

    def test_cross_k_metrics_boundary_nan(self, well_separated_data):
        """Hartigan at k_max yields NaN; interior k values are finite."""
        X, labels_k3 = well_separated_data
        labels_k2 = np.repeat([0, 1], [22, 23])
        labels_k4 = np.concatenate([np.repeat([0, 1, 2], 12), np.full(9, 3)])
        lbl = ClusteringResult.from_labels(X, {2: labels_k2, 3: labels_k3, 4: labels_k4})

        ckm = lbl.cross_k_metrics
        assert np.isnan(ckm.hartigan_index[4])   # k_max has no k+1
        assert np.isfinite(ckm.hartigan_index[2])

    def test_cache_invalidated_on_add(self, well_separated_data):
        """Adding a new labeling clears the cross_k_metrics cache."""
        X, labels_k3 = well_separated_data
        lbl = ClusteringResult.from_labels(X, {3: labels_k3})
        _ = lbl.cross_k_metrics  # populate cache

        labels_k2 = np.repeat([0, 1], [22, 23])
        lbl.add(2, labels_k2)
        assert 2 in lbl.k_values

    def test_single_cluster_labeling_accepted(self, well_separated_data):
        """k=1 (all-same labels) is a valid labeling."""
        X, _ = well_separated_data
        lbl = ClusteringResult(data=X, score_settings=ScoreSettings(), n_cluster_lower=1)
        result = lbl.add(1, np.zeros(X.shape[0], dtype=int))
        assert result is not None, "k=1 should be accepted as a valid labeling"
        assert 1 in lbl.k_values


# ── selection helpers ─────────────────────────────────────────────────────────

class TestBuildScoreProxy:
    def test_none_lm_gives_inf_for_single_k_metrics(self):
        """Invalid lm -> all-inf per-k scores (active worst vote)."""
        council = {'calinski_harabasz_index': 1.0}
        proxy = selection._compute_composite_score(None, council)
        assert proxy.score['calinski_harabasz_index'] == np.inf

    def test_extra_scores_override_inf_for_none_lm(self):
        """Cross-k extra scores injected even when lm is None."""
        council = {'calinski_harabasz_index': 1.0, 'hartigan_index': 1.0}
        proxy = selection._compute_composite_score(None, council, extra_scores={'hartigan_index': 5.0})
        assert proxy.score['calinski_harabasz_index'] == np.inf  # default for None lm
        assert proxy.score['hartigan_index'] == pytest.approx(5.0)

    def test_uncomputable_metric_gives_nan(self, well_separated_data):
        """Metric that raises AttributeError -> NaN (abstain), not inf."""
        X, labels = well_separated_data
        lm = SingleKMetrics(data=X, labels=labels)
        council = {'nonexistent_metric': 1.0}
        proxy = selection._compute_composite_score(lm, council)
        assert np.isnan(proxy.score['nonexistent_metric'])

    def test_extra_scores_not_overridden_by_lm_attribute(self, well_separated_data):
        """Extra scores take precedence over the lm's own attribute value."""
        X, labels = well_separated_data
        lm = SingleKMetrics(data=X, labels=labels)
        real_val = lm.calinski_harabasz_index
        injected_val = -999.0
        council = {'calinski_harabasz_index': 1.0}
        proxy = selection._compute_composite_score(
            lm, council, extra_scores={'calinski_harabasz_index': injected_val}
        )
        assert proxy.score['calinski_harabasz_index'] == pytest.approx(injected_val)
        assert proxy.score['calinski_harabasz_index'] != pytest.approx(real_val)


class TestSelectBestWithinK:
    def test_single_lm_returns_it(self, well_separated_data):
        X, labels = well_separated_data
        lm = SingleKMetrics(data=X, labels=labels)
        council = {'calinski_harabasz_index': 1.0}
        result = selection.select_best_within_k([lm], council)
        assert result is lm

    def test_clear_winner_selected(self, well_separated_data):
        """lm with lower (better) scores wins."""
        X, _ = well_separated_data
        rng = np.random.default_rng(42)
        labels_good = np.repeat([0, 1, 2], 15)
        labels_bad = rng.integers(0, 3, size=45)
        labels_bad[0] = 0; labels_bad[1] = 1; labels_bad[2] = 2
        lm_good = SingleKMetrics(data=X, labels=labels_good)
        lm_bad = SingleKMetrics(data=X, labels=labels_bad)
        council = {'calinski_harabasz_index': 1.0, 'silhouette_median_index': 1.0}
        result = selection.select_best_within_k([lm_good, lm_bad], council)
        assert result is lm_good

    def test_all_nan_metric_uses_wcss_fallback(self, well_separated_data):
        """When all voters abstain (all NaN), falls back to WCSS min."""
        X, labels = well_separated_data
        lm = SingleKMetrics(data=X, labels=labels)
        council = {'nonexistent_a': 1.0, 'nonexistent_b': 1.0}
        result = selection.select_best_within_k([lm], council)
        assert result is lm


class TestSelectBestAcrossK:
    def test_returns_valid_index(self, well_separated_data):
        X, labels_k3 = well_separated_data
        labels_k2 = np.repeat([0, 1], [22, 23])
        lm2 = SingleKMetrics(data=X, labels=labels_k2)
        lm3 = SingleKMetrics(data=X, labels=labels_k3)
        council = {'calinski_harabasz_index': 1.0}
        idx, conf = selection.select_best_across_k([lm2, lm3], council, window_size=0)
        assert idx in [0, 1]
        assert 0.0 <= conf <= 1.0

    def test_extra_scores_influence_result(self, well_separated_data):
        """Cross-k extra scores participate in the Schulze vote."""
        X, labels_k3 = well_separated_data
        labels_k2 = np.repeat([0, 1], [22, 23])
        lm2 = SingleKMetrics(data=X, labels=labels_k2)
        lm3 = SingleKMetrics(data=X, labels=labels_k3)

        council = {'hartigan_index': 1.0}
        extra = [
            {'hartigan_index': 0.001},   # k=2: excellent
            {'hartigan_index': 1000.0},  # k=3: terrible
        ]
        idx, _ = selection.select_best_across_k(
            [lm2, lm3], council, window_size=0, extra_scores_list=extra
        )
        assert idx == 0, "Sole voter (Hartigan) favouring k=2 should win"

    def test_none_candidates_treated_as_worst(self, well_separated_data):
        """None entries (invalid labelings) don't win over valid candidates."""
        X, labels = well_separated_data
        lm = SingleKMetrics(data=X, labels=labels)
        council = {'calinski_harabasz_index': 1.0}
        idx, _ = selection.select_best_across_k(
            [None, lm, None], council, window_size=0
        )
        assert idx == 1  # the only valid candidate


# ── ScoreSettings validation ──────────────────────────────────────────────────

class TestScoreSettingsValidation:
    def test_unknown_metric_raises(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            ScoreSettings(weights={'nonexistent_index': 1.0})

    def test_cross_k_metrics_accepted(self):
        """Cross-k metric names pass validation."""
        s = ScoreSettings(weights={'calinski_harabasz_index': 1.0, 'hartigan_index': 1.0})
        assert s.weights['hartigan_index'] == 1.0

    def test_all_zero_weights_raises(self):
        from opendsm.common.clustering.metrics.settings import _DEFAULT_SCORE_WEIGHTS
        all_zero = {k: 0.0 for k in _DEFAULT_SCORE_WEIGHTS}
        with pytest.raises(ValueError, match="At least one scoring weight"):
            ScoreSettings(weights=all_zero)


class TestClusterRangeSettingsValidation:
    def test_lower_greater_than_upper_raises(self):
        with pytest.raises(ValueError, match="n_cluster_lower.*<=.*n_cluster_upper"):
            ClusterRangeSettings(lower=10, upper=5)

    def test_lower_equal_upper_accepted(self):
        s = ClusterRangeSettings(lower=6, upper=6)
        assert s.lower == 6
        assert s.upper == 6

    def test_defaults(self):
        s = ClusterRangeSettings()
        assert s.lower == 1
        assert s.upper == 24


class TestSmallClusterModeEnum:
    def test_outlier_value(self):
        assert SmallClusterMode.OUTLIER == "outlier"

    def test_absorb_value(self):
        assert SmallClusterMode.ABSORB == "absorb"

    def test_keep_value(self):
        assert SmallClusterMode.KEEP == "keep"


class TestScoreSettingsMaxScoringSamples:
    def test_none_accepted(self):
        s = ScoreSettings(max_scoring_samples=None)
        assert s.max_scoring_samples is None

    def test_below_minimum_rejected(self):
        with pytest.raises(ValueError):
            ScoreSettings(max_scoring_samples=3)


# ── Exact-value helpers (float64 naive references) ────────────────────────────

def _D64(data: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import squareform, pdist
    return squareform(pdist(data.astype(np.float64)))


def _means64(data: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return (centroids[k], label_indices) in float64."""
    unique = np.unique(labels)
    idx = {lbl: np.where(labels == lbl)[0] for lbl in unique}
    centroids = np.array([data[idx[lbl]].astype(np.float64).mean(axis=0) for lbl in unique])
    return centroids, idx


def _ref_mean_intra(data: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    D = _D64(data)
    result = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) == 1:
            result[lbl] = np.array([0.0])
        else:
            sub = D[np.ix_(idx, idx)].copy()
            np.fill_diagonal(sub, np.nan)
            result[lbl] = np.nanmean(sub, axis=1)
    return result


def _ref_median_intra(data: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    D = _D64(data)
    result = {}
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) == 1:
            result[lbl] = np.array([0.0])
        else:
            sub = D[np.ix_(idx, idx)].copy()
            np.fill_diagonal(sub, np.nan)
            result[lbl] = np.nanmedian(sub, axis=1)
    return result


def _ref_mean_nearest(data: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    D = _D64(data)
    unique = np.unique(labels)
    result = {}
    for lbl_i in unique:
        idx_i = np.where(labels == lbl_i)[0]
        nearest = np.full(len(idx_i), np.inf)
        for lbl_j in unique:
            if lbl_j == lbl_i:
                continue
            idx_j = np.where(labels == lbl_j)[0]
            nearest = np.minimum(nearest, D[np.ix_(idx_i, idx_j)].mean(axis=1))
        result[lbl_i] = nearest
    return result


def _ref_median_nearest(data: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    D = _D64(data)
    unique = np.unique(labels)
    result = {}
    for lbl_i in unique:
        idx_i = np.where(labels == lbl_i)[0]
        nearest = np.full(len(idx_i), np.inf)
        for lbl_j in unique:
            if lbl_j == lbl_i:
                continue
            idx_j = np.where(labels == lbl_j)[0]
            nearest = np.minimum(nearest, np.median(D[np.ix_(idx_i, idx_j)], axis=1))
        result[lbl_i] = nearest
    return result


def _ref_sil_coeffs(intra, nearest, labels, agg):
    coeffs = []
    for lbl in np.unique(labels):
        a, b = intra[lbl], nearest[lbl]
        denom = np.maximum(a, b)
        coeffs.extend(np.where(denom > 0, (b - a) / denom, 0.0).tolist())
    return np.array(coeffs)


def _ref_wcss(data: np.ndarray, labels: np.ndarray) -> float:
    d64 = data.astype(np.float64)
    total = 0.0
    for lbl in np.unique(labels):
        X = d64[labels == lbl]
        mu = X.mean(axis=0)
        total += float(np.sum((X - mu) ** 2))
    return total


def _ref_davies_bouldin(data: np.ndarray, labels: np.ndarray) -> float:
    from scipy.spatial.distance import squareform, pdist, cdist
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    k = len(unique)
    centroids, idx = _means64(d64, labels)
    s = np.array([cdist(d64[idx[lbl]], centroids[[i]]).mean() for i, lbl in enumerate(unique)])
    inter = squareform(pdist(centroids))
    with np.errstate(divide='ignore', invalid='ignore'):
        sim = (s[:, None] + s[None, :]) / inter
    np.fill_diagonal(sim, 0)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0)
    return float(np.sum(np.max(sim, axis=1)) / k)


def _ref_calinski_harabasz(data: np.ndarray, labels: np.ndarray) -> float:
    d64 = data.astype(np.float64)
    n, unique = len(d64), np.unique(labels)
    k = len(unique)
    WCSS = _ref_wcss(data, labels)
    grand = d64.mean(axis=0)
    BCSS = sum(len(d64[labels == lbl]) * float(np.sum((d64[labels == lbl].mean(axis=0) - grand) ** 2))
               for lbl in unique)
    if WCSS < 1e-10:
        return -np.inf
    return -((BCSS / WCSS) * ((n - k) / (k - 1.0)))


def _ref_c_index(data: np.ndarray, labels: np.ndarray) -> float:
    D = _D64(data)
    n = len(data)
    unique = np.unique(labels)
    within = []
    for lbl in unique:
        idx = np.where(labels == lbl)[0]
        if len(idx) > 1:
            sub = D[np.ix_(idx, idx)]
            within.extend(sub[np.triu_indices_from(sub, k=1)].tolist())
    within = np.array(within)
    n_w = len(within)
    if n_w == 0:
        return np.nan
    S_w = within.sum()
    all_d = np.sort(D[np.triu_indices(n, k=1)])
    S_min, S_max = all_d[:n_w].sum(), all_d[-n_w:].sum()
    denom = S_max - S_min
    return float((S_w - S_min) / denom) if denom > 1e-10 else 0.0


def _ref_gamma(data: np.ndarray, labels: np.ndarray) -> float:
    D = _D64(data)
    n = len(data)
    unique = np.unique(labels)
    within, between = [], []
    for i, li in enumerate(unique):
        ii = np.where(labels == li)[0]
        if len(ii) > 1:
            sub = D[np.ix_(ii, ii)]
            within.extend(sub[np.triu_indices_from(sub, k=1)].tolist())
        for lj in unique[i + 1:]:
            ij = np.where(labels == lj)[0]
            between.extend(D[np.ix_(ii, ij)].ravel().tolist())
    within, between = np.array(within), np.array(between)
    if len(within) == 0 or len(between) == 0:
        return np.nan
    nb = len(between)
    bs = np.sort(between)
    r = np.searchsorted(bs, within, side='right')
    l = np.searchsorted(bs, within, side='left')
    s_plus = int(np.sum(nb - r))
    s_minus = int(np.sum(l))
    denom = s_plus + s_minus
    if denom == 0:
        return np.nan
    return -(float(s_plus - s_minus) / float(denom))


def _ref_point_biserial(data: np.ndarray, labels: np.ndarray) -> float:
    D = _D64(data)
    unique = np.unique(labels)
    within, between = [], []
    for i, li in enumerate(unique):
        ii = np.where(labels == li)[0]
        if len(ii) > 1:
            sub = D[np.ix_(ii, ii)]
            within.extend(sub[np.triu_indices_from(sub, k=1)].tolist())
        for lj in unique[i + 1:]:
            ij = np.where(labels == lj)[0]
            between.extend(D[np.ix_(ii, ij)].ravel().tolist())
    within, between = np.array(within), np.array(between)
    if len(within) == 0 or len(between) == 0:
        return np.nan
    n_w, n_b = len(within), len(between)
    n_t = n_w + n_b
    m_w, m_b = within.mean(), between.mean()
    std = np.std(np.concatenate([within, between]))
    if std < 1e-10:
        return np.nan
    return -(float((m_b - m_w) / std) * np.sqrt(n_w * n_b / (n_t ** 2)))


def _ref_mcclain_rao(data: np.ndarray, labels: np.ndarray) -> float:
    D = _D64(data)
    unique = np.unique(labels)
    within, between = [], []
    for i, li in enumerate(unique):
        ii = np.where(labels == li)[0]
        if len(ii) > 1:
            sub = D[np.ix_(ii, ii)]
            within.extend(sub[np.triu_indices_from(sub, k=1)].tolist())
        for lj in unique[i + 1:]:
            ij = np.where(labels == lj)[0]
            between.extend(D[np.ix_(ii, ij)].ravel().tolist())
    within, between = np.array(within), np.array(between)
    if len(within) == 0 or len(between) == 0:
        return np.nan
    m_b = between.mean()
    if m_b < 1e-10:
        return np.nan if within.mean() < 1e-10 else np.inf
    return float(within.mean() / m_b)


def _ref_banfeld_raftery(data: np.ndarray, labels: np.ndarray) -> float:
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    res = 0.0
    for lbl in unique:
        X = d64[labels == lbl]
        n_k = len(X)
        mu = X.mean(axis=0)
        W_k = (X - mu).T @ (X - mu)
        tr = float(np.trace(W_k))
        tr_safe = max(tr, 1e-10)
        res += n_k * np.log(tr_safe / n_k)
    return float(res)


def _ref_scott_symons(data: np.ndarray, labels: np.ndarray) -> float:
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    res = 0.0
    for lbl in unique:
        X = d64[labels == lbl]
        n_k = len(X)
        mu = X.mean(axis=0)
        W_k = (X - mu).T @ (X - mu)
        try:
            sign, logdet = np.linalg.slogdet(W_k / n_k)
            res += n_k * (logdet if sign > 0 else np.log(1e-10))
        except np.linalg.LinAlgError:
            res += n_k * np.log(1e-10)
    return float(res)


def _ref_xie_beni(data: np.ndarray, labels: np.ndarray) -> float:
    from scipy.spatial.distance import pdist
    d64 = data.astype(np.float64)
    WCSS = _ref_wcss(data, labels)
    n = len(d64)
    centroids, _ = _means64(d64, labels)
    if len(centroids) < 2:
        return np.inf
    d_sq = pdist(centroids, metric='sqeuclidean')
    d_min_sq = float(np.min(d_sq))
    if d_min_sq < 1e-10:
        return np.inf
    return float(WCSS / (n * d_min_sq))


def _ref_duda_hart(data: np.ndarray, labels: np.ndarray) -> float:
    D = _D64(data)
    from scipy.spatial.distance import cdist
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    centroids, idx = _means64(d64, labels)
    dtm = cdist(d64, np.vstack([d64.mean(axis=0), centroids]))
    intra = sum(float(np.mean(dtm[idx[lbl], i + 1])) for i, lbl in enumerate(unique))
    inter_total = 0.0
    for i, lbl_i in enumerate(unique):
        others = [D[np.ix_(idx[lbl_i], idx[lbl_j])] for lbl_j in unique if lbl_j != lbl_i]
        inter_total += float(np.mean(np.hstack(others)))
    if inter_total < 1e-10:
        return np.nan
    return float(intra / inter_total)


def _ref_simplified_silhouette(data: np.ndarray, labels: np.ndarray) -> float:
    from scipy.spatial.distance import cdist
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    centroids, idx = _means64(d64, labels)
    dtm = cdist(d64, centroids)
    label_to_col = {lbl: i for i, lbl in enumerate(unique)}
    own_col = np.array([label_to_col[lbl] for lbl in labels])
    a = dtm[np.arange(len(d64)), own_col]
    dtm_m = dtm.copy()
    dtm_m[np.arange(len(d64)), own_col] = np.inf
    b = np.min(dtm_m, axis=1)
    denom = np.maximum(a, b)
    s = np.where(denom > 1e-10, (b - a) / denom, 0.0)
    return -float(np.mean(s))


def _ref_cop(data: np.ndarray, labels: np.ndarray) -> float:
    from scipy.spatial.distance import cdist
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    centroids, idx = _means64(d64, labels)
    dtm = cdist(d64, centroids)
    label_to_col = {lbl: i for i, lbl in enumerate(unique)}
    own_col = np.array([label_to_col[lbl] for lbl in labels])
    a = dtm[np.arange(len(d64)), own_col]
    dtm_m = dtm.copy()
    dtm_m[np.arange(len(d64)), own_col] = np.inf
    b = np.min(dtm_m, axis=1)
    ratio = np.where(b > 1e-10, a / b, np.where(a < 1e-10, 0.0, np.inf))
    return float(np.mean(ratio))


def _ref_generalized_dunn(data: np.ndarray, labels: np.ndarray) -> float:
    from scipy.spatial.distance import cdist
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    centroids, idx = _means64(d64, labels)
    min_inter = np.inf
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            min_inter = min(min_inter, float(np.linalg.norm(centroids[i] - centroids[j])))
    dtm = cdist(d64, centroids)
    max_intra = max(float(np.mean(dtm[idx[lbl], i])) for i, lbl in enumerate(unique))
    if max_intra < 1e-10:
        return np.inf
    return -(min_inter / max_intra)


def _ref_i_index(data: np.ndarray, labels: np.ndarray) -> float:
    from scipy.spatial.distance import cdist, pdist
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    centroids, idx = _means64(d64, labels)
    grand = np.array([d64.mean(axis=0)])
    dtm_grand = cdist(d64, grand)
    dtm_cluster = cdist(d64, centroids)
    label_to_col = {lbl: i for i, lbl in enumerate(unique)}
    own_col = np.array([label_to_col[lbl] for lbl in labels])
    E_1 = float(dtm_grand.sum())
    E_K = float(sum(dtm_cluster[idx[lbl], i].sum() for i, lbl in enumerate(unique)))
    D_K = float(np.max(pdist(centroids))) if len(centroids) > 1 else np.nan
    if np.isnan(D_K):
        return np.nan
    if E_K < 1e-10:
        return np.inf
    k = len(unique)
    return -(((1.0 / k) * (E_1 / E_K) * D_K) ** 2)


def _ref_wb(data: np.ndarray, labels: np.ndarray) -> float:
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    k = len(unique)
    WCSS = _ref_wcss(data, labels)
    grand = d64.mean(axis=0)
    BCSS = sum(len(d64[labels == lbl]) * float(np.sum((d64[labels == lbl].mean(axis=0) - grand) ** 2))
               for lbl in unique)
    if BCSS < 1e-10:
        return np.inf
    return float(k * WCSS / BCSS)


def _ref_det_ratio(data: np.ndarray, labels: np.ndarray) -> float:
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    grand = d64.mean(axis=0)
    centered = d64 - grand
    T = centered.T @ centered
    W = sum(((d64[labels == lbl] - d64[labels == lbl].mean(axis=0)).T @
             (d64[labels == lbl] - d64[labels == lbl].mean(axis=0)))
            for lbl in unique)
    det_T = np.linalg.det(T)
    det_W = np.linalg.det(W)
    if abs(det_W) < 1e-10:
        return np.inf
    return float(det_T / det_W)


def _ref_log_ss_ratio(data: np.ndarray, labels: np.ndarray) -> float:
    d64 = data.astype(np.float64)
    unique = np.unique(labels)
    grand = d64.mean(axis=0)
    WCSS = _ref_wcss(data, labels)
    BCSS = sum(len(d64[labels == lbl]) * float(np.sum((d64[labels == lbl].mean(axis=0) - grand) ** 2))
               for lbl in unique)
    if WCSS < 1e-10:
        return np.inf
    if BCSS < 1e-10:
        return -np.inf
    return -(np.log(BCSS) - np.log(WCSS))


# Loose tolerance: float32 internal data/distance vs float64 reference
_AV_ATOL = 5e-3
_AV_RTOL = 5e-3


@pytest.fixture(scope='module')
def exact_lm():
    """(data_f32, labels, lm) with k=4 tight clusters for exact-value tests."""
    rng = np.random.default_rng(99)
    k, n_per = 4, 25
    centers = rng.standard_normal((k, 6)) * 8.0
    data_parts, label_parts = [], []
    for i, c in enumerate(centers):
        data_parts.append(c + rng.standard_normal((n_per, 6)) * 0.3)
        label_parts.append(np.full(n_per, i))
    data = np.vstack(data_parts).astype(np.float32)
    labels = np.concatenate(label_parts)
    return data, labels, SingleKMetrics(data=data, labels=labels)


class TestSingleKMetricsExactValues:
    """Each *_index is compared against a naive float64 reference or sklearn,
    verifying that optimised paths and float32 conversion preserve correctness."""

    # ── intermediate optimised-path arrays ───────────────────────────────────

    @pytest.mark.parametrize('attr,ref_fn', [
        ('_labeled_mean_distance_intra_cluster', _ref_mean_intra),
        ('_labeled_median_distance_intra_cluster', _ref_median_intra),
        ('_labeled_mean_distance_to_nearest_cluster', _ref_mean_nearest),
        ('_labeled_median_distance_to_nearest_cluster', _ref_median_nearest),
    ])
    def test_intermediate_arrays_match_naive(self, exact_lm, attr, ref_fn):
        data, labels, lm = exact_lm
        ref = ref_fn(data, labels)
        for lbl in np.unique(labels):
            np.testing.assert_allclose(
                getattr(lm, attr)[lbl], ref[lbl],
                atol=_AV_ATOL, rtol=_AV_RTOL)

    # ── silhouette variants ───────────────────────────────────────────────────

    def test_silhouette_index(self, exact_lm):
        data, labels, lm = exact_lm
        ref_i = _ref_mean_intra(data, labels)
        ref_n = _ref_mean_nearest(data, labels)
        ref = -np.mean(_ref_sil_coeffs(ref_i, ref_n, labels, 'mean'))
        np.testing.assert_allclose(lm.silhouette_index, ref, atol=_AV_ATOL, rtol=_AV_RTOL)

    def test_silhouette_median_index(self, exact_lm):
        data, labels, lm = exact_lm
        ref_i = _ref_median_intra(data, labels)
        ref_n = _ref_median_nearest(data, labels)
        ref = -np.median(_ref_sil_coeffs(ref_i, ref_n, labels, 'median'))
        np.testing.assert_allclose(lm.silhouette_median_index, ref, atol=_AV_ATOL, rtol=_AV_RTOL)

    def test_simplified_silhouette_index(self, exact_lm):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            lm.simplified_silhouette_index, _ref_simplified_silhouette(data, labels),
            atol=_AV_ATOL, rtol=_AV_RTOL)

    # ── davies-bouldin ────────────────────────────────────────────────────────

    def test_davies_bouldin_index(self, exact_lm):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            lm.davies_bouldin_index, _ref_davies_bouldin(data, labels),
            atol=_AV_ATOL, rtol=_AV_RTOL)

    # ── WCSS-derived indices ──────────────────────────────────────────────────

    def test_sum_of_squared_errors_index(self, exact_lm):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            lm.sum_of_squared_errors_index, _ref_wcss(data, labels),
            atol=1.0, rtol=_AV_RTOL)  # larger atol: float32 accum on n=100

    def test_mean_squared_error_index(self, exact_lm):
        data, labels, lm = exact_lm
        ref = _ref_wcss(data, labels) / len(data)
        np.testing.assert_allclose(lm.mean_squared_error_index, ref, atol=0.05, rtol=_AV_RTOL)

    def test_ball_hall_index(self, exact_lm):
        data, labels, lm = exact_lm
        ref = _ref_wcss(data, labels) / len(np.unique(labels))
        np.testing.assert_allclose(lm.ball_hall_index, ref, atol=0.5, rtol=_AV_RTOL)

    def test_trace_w_equals_sse(self, exact_lm):
        _, _, lm = exact_lm
        np.testing.assert_allclose(
            lm.trace_w_index, lm.sum_of_squared_errors_index, atol=1e-4, rtol=1e-4)

    def test_calinski_harabasz_index(self, exact_lm):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            lm.calinski_harabasz_index, _ref_calinski_harabasz(data, labels),
            atol=1.0, rtol=_AV_RTOL)

    def test_variance_ratio_criterion_equals_ch(self, exact_lm):
        _, _, lm = exact_lm
        assert lm.variance_ratio_criterion == lm.calinski_harabasz_index

    def test_log_ss_ratio_index(self, exact_lm):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            lm.log_ss_ratio_index, _ref_log_ss_ratio(data, labels),
            atol=_AV_ATOL, rtol=_AV_RTOL)

    def test_wb_index(self, exact_lm):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            lm.wb_index, _ref_wb(data, labels), atol=0.01, rtol=_AV_RTOL)

    def test_banfeld_raftery_index(self, exact_lm):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            lm.banfeld_raftery_index, _ref_banfeld_raftery(data, labels),
            atol=0.1, rtol=_AV_RTOL)

    def test_scott_symons_index(self, exact_lm):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            lm.scott_symons_index, _ref_scott_symons(data, labels),
            atol=0.5, rtol=_AV_RTOL)

    # ── pairwise-distance indices ─────────────────────────────────────────────

    @pytest.mark.parametrize('index,ref_fn', [
        ('c_index', _ref_c_index),
        ('gamma_index', _ref_gamma),
        ('point_biserial_index', _ref_point_biserial),
        ('mcclain_rao_index', _ref_mcclain_rao),
        ('xie_beni_index', _ref_xie_beni),
        ('duda_hart_index', _ref_duda_hart),
        ('cop_index', _ref_cop),
        ('generalized_dunn_index', _ref_generalized_dunn),
    ])
    def test_pairwise_index(self, exact_lm, index, ref_fn):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            getattr(lm, index), ref_fn(data, labels), atol=_AV_ATOL, rtol=_AV_RTOL)

    def test_i_index(self, exact_lm):
        data, labels, lm = exact_lm
        np.testing.assert_allclose(
            lm.i_index, _ref_i_index(data, labels), atol=0.1, rtol=_AV_RTOL)

    # ── matrix/determinant indices ────────────────────────────────────────────

    def test_det_ratio_index(self, exact_lm):
        data, labels, lm = exact_lm
        ref_raw = _ref_det_ratio(data, labels)
        np.testing.assert_allclose(
            lm.det_ratio_index, -ref_raw, atol=max(1.0, abs(ref_raw) * 0.02), rtol=0.05)

    def test_log_det_ratio_index(self, exact_lm):
        data, labels, lm = exact_lm
        ref_raw = _ref_det_ratio(data, labels)
        ref = -(len(data) * np.log(abs(ref_raw)))
        np.testing.assert_allclose(lm.log_det_ratio_index, ref, atol=1.0, rtol=0.05)

    # ── indices with finiteness/sign checks only (no naive reference) ─────────

    @pytest.mark.parametrize('index,sign_upper', [
        ('trace_wb_index', 0),                              # maximize -> negated -> negative
        ('dunn_index', 0),                                   # maximize -> negated -> negative
        ('density_based_clustering_validation_index', 0),    # maximize -> negated -> non-positive
    ])
    def test_finite_and_negative(self, exact_lm, index, sign_upper):
        _, _, lm = exact_lm
        val = getattr(lm, index)
        assert np.isfinite(val), f"{index} should be finite, got {val}"
        assert val <= sign_upper, f"{index} should be <= {sign_upper}, got {val}"

    @pytest.mark.parametrize('index', [
        's_dbw_index',
        'sd_validity_index',
        'negentropy_index',
    ])
    def test_finite_and_nonnegative(self, exact_lm, index):
        """Minimize-direction indices: finite and non-negative for valid clusters."""
        _, _, lm = exact_lm
        val = getattr(lm, index)
        assert np.isfinite(val), f"{index} should be finite, got {val}"
        assert val >= 0, f"{index} should be >= 0, got {val}"

    def test_ksq_detw_index_finite(self, exact_lm):
        _, _, lm = exact_lm
        assert np.isfinite(lm.ksq_detw_index)


class TestDefaultSettings:
    """Verify default settings match expected values (regression guard)."""

    def test_default_council_members(self):
        """Default scoring weights activate exactly pb+smed+wb+xb (4-member council)."""
        settings = ScoreSettings()
        active = sorted(k for k, v in settings.weights.items() if v > 0)
        expected = sorted([
            'point_biserial_index',
            'silhouette_median_index',
            'wb_index',
            'xie_beni_index',
        ])
        assert active == expected, f"Default council changed: {active}"

    def test_default_normalization_method(self):
        """Default normalization is med_mad."""
        from opendsm.common.clustering.transform.normalize_settings import (
            NormalizeSettings, NormalizeChoice,
        )
        settings = NormalizeSettings()
        assert settings.method == NormalizeChoice.MED_MAD

    def test_default_max_scoring_samples(self):
        """Default max_scoring_samples is 10000."""
        settings = ScoreSettings()
        assert settings.max_scoring_samples == 10_000


class TestScoringSubsample:
    """Tests for LabelStore scoring subsample (max_scoring_samples)."""

    def _make_data(self, n, d=10, k=3, seed=42):
        rng = np.random.default_rng(seed)
        centers = rng.random((k, d)) * 10
        labels = np.repeat(np.arange(k), n // k + 1)[:n]
        data = centers[labels] + rng.normal(0, 0.5, (n, d))
        return data.astype(np.float32), labels

    def test_no_subsampling_below_threshold(self):
        """When n < max_scoring_samples, no subsampling occurs."""
        data, labels = self._make_data(100)
        settings = ScoreSettings(max_scoring_samples=200)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        assert result._scoring_idx is None
        assert result._scoring_data is None

    def test_subsampling_above_threshold(self):
        """When n > max_scoring_samples, subsample is drawn."""
        data, labels = self._make_data(200)
        settings = ScoreSettings(max_scoring_samples=50)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        assert result._scoring_idx is not None
        assert len(result._scoring_idx) == 50
        assert result._scoring_data.shape == (50, data.shape[1])

    def test_subsample_is_deterministic(self):
        """Same seed produces same subsample."""
        data, _ = self._make_data(200)
        settings = ScoreSettings(max_scoring_samples=50)
        r1 = ClusteringResult(data=data, score_settings=settings, seed=42)
        r2 = ClusteringResult(data=data, score_settings=settings, seed=42)
        np.testing.assert_array_equal(r1._scoring_idx, r2._scoring_idx)

    def test_different_seed_different_subsample(self):
        """Different seed produces different subsample."""
        data, _ = self._make_data(200)
        settings = ScoreSettings(max_scoring_samples=50)
        r1 = ClusteringResult(data=data, score_settings=settings, seed=0)
        r2 = ClusteringResult(data=data, score_settings=settings, seed=1)
        assert not np.array_equal(r1._scoring_idx, r2._scoring_idx)

    def test_output_labels_are_full_length(self):
        """Labels from ClusteringResult.labels have length n, not subsample size."""
        data, labels = self._make_data(200)
        settings = ScoreSettings(max_scoring_samples=50)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        result.add(3, labels)
        assert len(result.labels) == 200

    def test_subsample_none_disables(self):
        """max_scoring_samples=None disables subsampling."""
        data, _ = self._make_data(200)
        settings = ScoreSettings(max_scoring_samples=None)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        assert result._scoring_idx is None

    def test_consistent_k_selection_across_adds(self):
        """All add() calls share the same subsample rows."""
        data, _ = self._make_data(200)
        settings = ScoreSettings(max_scoring_samples=50)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        idx = result._scoring_idx.copy()
        labels_k2 = np.where(np.arange(200) < 100, 0, 1)
        labels_k3 = np.where(np.arange(200) < 67, 0, np.where(np.arange(200) < 134, 1, 2))
        result.add(2, labels_k2)
        result.add(3, labels_k3)
        np.testing.assert_array_equal(result._scoring_idx, idx)

    def test_selection_confidence_works_with_subsample(self):
        """selection_confidence is accessible when subsampling is active."""
        data, labels = self._make_data(200)
        settings = ScoreSettings(max_scoring_samples=50)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        result.add(2, np.where(np.arange(200) < 100, 0, 1))
        result.add(3, np.where(np.arange(200) < 67, 0, np.where(np.arange(200) < 134, 1, 2)))
        conf = result.selection_confidence
        assert 0.0 <= conf <= 1.0

    def test_subsample_indices_are_sorted_and_unique(self):
        """Subsample indices are sorted with no duplicates."""
        data, _ = self._make_data(500)
        settings = ScoreSettings(max_scoring_samples=100)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        idx = result._scoring_idx
        assert len(idx) == len(np.unique(idx)), "Subsample has duplicate indices"
        np.testing.assert_array_equal(idx, np.sort(idx), err_msg="Indices not sorted")

    def test_subsample_indices_within_bounds(self):
        """All subsample indices are valid row indices."""
        data, _ = self._make_data(500)
        settings = ScoreSettings(max_scoring_samples=100)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        assert result._scoring_idx.min() >= 0
        assert result._scoring_idx.max() < 500

    def test_subsample_data_matches_indexed_data(self):
        """Subsampled data equals data[scoring_idx]."""
        data, _ = self._make_data(500)
        settings = ScoreSettings(max_scoring_samples=100)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        expected = data[result._scoring_idx].astype(np.float32)
        np.testing.assert_array_equal(result._scoring_data, expected)

    def test_add_with_all_outlier_labels(self):
        """Adding labels where subsampled labels produce all-outlier is handled."""
        data, _ = self._make_data(200)
        settings = ScoreSettings(max_scoring_samples=50)
        result = ClusteringResult(
            data=data, score_settings=settings, seed=0,
            min_cluster_size=100, small_cluster_mode="outlier",
        )
        labels = np.repeat(np.arange(10), 20)
        lm = result.add(10, labels)
        assert lm is None

    def test_single_k_with_subsample(self):
        """Single k value added with subsampling produces valid labels."""
        data, _ = self._make_data(200)
        settings = ScoreSettings(max_scoring_samples=50)
        result = ClusteringResult(data=data, score_settings=settings, seed=0)
        labels = np.where(np.arange(200) < 100, 0, 1)
        result.add(2, labels)
        assert len(result.labels) == 200
        assert set(result.labels[result.labels >= 0]).issubset({0, 1})


class TestSelectionEdgeCases:
    """Edge-case tests for selection.py code paths."""

    # ── 1. select_best_within_k: all NaN scores → fallback to min _WCSS ─────

    def test_select_best_within_k_all_nan_falls_back_to_min_wcss(self):
        """When every metric raises for every candidate, wins stays zero
        and the function falls back to the candidate with the lowest _WCSS."""

        class _BadMetric:
            """Mock whose metric attributes always raise."""
            def __init__(self, wcss):
                self._WCSS = wcss

            def __getattr__(self, name):
                # Any metric attribute access raises → NaN via except branch
                raise AttributeError(f"{name} not available")

        lm_a = _BadMetric(wcss=100.0)
        lm_b = _BadMetric(wcss=50.0)   # lowest WCSS → should be selected
        lm_c = _BadMetric(wcss=200.0)

        council = {
            'silhouette_index': 1.0,
            'davies_bouldin_index': 1.0,
            'xie_beni_index': 1.0,
        }

        result = selection.select_best_within_k([lm_a, lm_b, lm_c], council)
        assert result is lm_b, (
            f"Expected candidate with _WCSS=50, got _WCSS={result._WCSS}"
        )

    # ── 2. _compute_composite_score: coverage < 1.0 penalty ─────────────────

    def test_composite_score_coverage_penalty(self):
        """When _coverage < 1.0 on lm, finite scores are divided by coverage."""
        import types as _types

        council = {'metric_a': 1.0, 'metric_b': 1.0, 'metric_c': 1.0}

        # Mock lm with known metric values and a coverage < 1.0
        lm = _types.SimpleNamespace(
            metric_a=2.0,       # finite → should be divided by coverage
            metric_b=np.inf,    # inf → should NOT be divided
            metric_c=-np.inf,   # -inf → should NOT be divided
            _coverage=0.5,
        )

        ns = selection._compute_composite_score(lm, council)

        # finite score 2.0 / 0.5 = 4.0
        assert ns.score['metric_a'] == pytest.approx(4.0), (
            f"Expected 2.0/0.5=4.0, got {ns.score['metric_a']}"
        )
        # inf values should pass through unchanged
        assert ns.score['metric_b'] == np.inf
        assert ns.score['metric_c'] == -np.inf

    def test_composite_score_no_penalty_at_full_coverage(self):
        """When _coverage is 1.0 (or absent), finite scores are unchanged."""
        import types as _types

        council = {'metric_a': 1.0}
        lm = _types.SimpleNamespace(metric_a=3.0, _coverage=1.0)

        ns = selection._compute_composite_score(lm, council)
        assert ns.score['metric_a'] == pytest.approx(3.0)

    def test_composite_score_coverage_absent_means_no_penalty(self):
        """When _coverage attribute is missing, getattr defaults to 1.0."""
        import types as _types

        council = {'metric_a': 1.0}
        lm = _types.SimpleNamespace(metric_a=7.0)
        # no _coverage attribute → getattr(lm, '_coverage', 1.0) returns 1.0

        ns = selection._compute_composite_score(lm, council)
        assert ns.score['metric_a'] == pytest.approx(7.0)

    # ── 3. _compute_composite_score: getattr raises → NaN (abstain) ─────────

    def test_composite_score_getattr_exception_yields_nan(self):
        """If getattr(lm, metric) raises, the score for that metric is NaN."""

        class _PartialMetric:
            """Mock where one metric works and another raises."""
            def __init__(self):
                self._coverage = 1.0

            @property
            def good_metric(self):
                return 5.0

            @property
            def bad_metric(self):
                raise RuntimeError("computation failed")

        council = {'good_metric': 1.0, 'bad_metric': 1.0}
        lm = _PartialMetric()

        ns = selection._compute_composite_score(lm, council)

        assert ns.score['good_metric'] == pytest.approx(5.0)
        assert np.isnan(ns.score['bad_metric']), (
            f"Expected NaN for failing metric, got {ns.score['bad_metric']}"
        )

    def test_composite_score_all_metrics_raise_yields_all_nan(self):
        """If every metric raises, all scores are NaN."""

        class _AllBad:
            def __init__(self):
                self._coverage = 1.0

            def __getattr__(self, name):
                if name.startswith('_'):
                    raise AttributeError(name)
                raise ValueError(f"{name} is broken")

        council = {'alpha': 1.0, 'beta': 1.0}
        lm = _AllBad()

        ns = selection._compute_composite_score(lm, council)

        for metric in council:
            assert np.isnan(ns.score[metric]), (
                f"Expected NaN for {metric}, got {ns.score[metric]}"
            )


# ── Singleton index behavior ─────────────────────────────────────────────────

class TestSingletonIndexBehavior:
    """Verify that singleton clusters are handled correctly by all indices."""

    @pytest.fixture(scope="class")
    def singleton_lm(self):
        """k=3 labeling with 5+1+1: clusters 1 and 2 are singletons."""
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal([0.0, 0.0], 0.4, (5, 2)),
            np.array([[20.0, 20.0]]),
            np.array([[30.0, 30.0]]),
        ])
        labels = np.array([0, 0, 0, 0, 0, 1, 2])
        return SingleKMetrics(data=X, labels=labels)

    def test_silhouette_zero_for_singleton_points(self, singleton_lm):
        """Points in singleton clusters should have s_i = 0 (neutral)."""
        coeffs = singleton_lm._silhouette_coefficients()
        np.testing.assert_array_equal(
            coeffs[-2:], [0.0, 0.0],
            err_msg="Singleton cluster points should have silhouette coefficient 0",
        )
        # Non-singleton points should have positive silhouette (well-separated)
        assert np.all(coeffs[:5] > 0), (
            "Non-singleton points should have positive silhouette coefficients"
        )

    def test_davies_bouldin_nan_with_fewer_than_2_non_singleton(self, singleton_lm):
        """DB excludes singletons; with only 1 non-singleton cluster, returns NaN."""
        assert np.isnan(singleton_lm.davies_bouldin_index), (
            "DB should return NaN when fewer than 2 non-singleton clusters exist"
        )

    def test_xie_beni_inf_with_1_non_singleton_centroid(self, singleton_lm):
        """XB excludes singletons; with only 1 non-singleton centroid, returns inf."""
        assert singleton_lm.xie_beni_index == np.inf, (
            "XB should return inf when only 1 non-singleton centroid exists "
            "(no inter-centroid distance)"
        )

    def test_dbcv_nan_when_any_singleton(self, singleton_lm):
        """DBCV requires >=2 points per cluster for core distance; returns NaN."""
        assert np.isnan(singleton_lm.density_based_clustering_validation_index), (
            "DBCV should return NaN when any cluster is a singleton"
        )

    def test_determinant_indices_nan_when_rank_deficient(self, singleton_lm):
        """Determinant-based indices must return NaN when any cluster has n_k < d."""
        assert singleton_lm._has_rank_deficient_cluster, (
            "Singleton clusters (n_k=1 < d=2) should trigger rank deficiency"
        )
        assert np.isnan(singleton_lm.scott_symons_index), (
            "scott_symons_index should return NaN for rank-deficient clusters"
        )
        assert np.isnan(singleton_lm.banfeld_raftery_index), (
            "banfeld_raftery_index should return NaN for rank-deficient clusters"
        )

    def test_active_council_finite_on_no_singletons(self):
        """Regression: all active council indices are finite on clean 3-cluster data."""
        rng = np.random.default_rng(42)
        centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]])
        X = np.vstack([rng.normal(c, 0.4, (20, 2)) for c in centers])
        labels = np.repeat([0, 1, 2], 20)
        lm = SingleKMetrics(data=X, labels=labels)

        expected = {
            'davies_bouldin_index': 0.07936187995148962,
            'point_biserial_index': -0.9959886843291079,
            'silhouette_median_index': -0.9490914046764374,
            'xie_beni_index': 0.0018903512758990508,
        }
        for name, pinned in expected.items():
            val = getattr(lm, name)
            assert np.isfinite(val), f"{name} should be finite, got {val}"
            assert val == pytest.approx(pinned, rel=1e-6), (
                f"Regression: {name} = {val}, expected {pinned}"
            )


# ── Settings coupling ────────────────────────────────────────────────────────

class TestSettingsCoupling:
    """Validate the min_cluster_size / small_cluster_mode coupling rules.

    These settings are on ClusteringSettings (elevated from ScoreSettings).
    """

    def test_min_cluster_size_1_keep_accepted(self):
        """min_cluster_size=1 + KEEP is the valid pairing for density-based algos."""
        from opendsm.common.clustering.settings import ClusteringSettings
        cs = ClusteringSettings(min_cluster_size=1, small_cluster_mode=SmallClusterMode.KEEP)
        assert cs.min_cluster_size == 1
        assert cs.small_cluster_mode == SmallClusterMode.KEEP

    def test_min_cluster_size_1_outlier_rejected(self):
        """min_cluster_size=1 + OUTLIER is contradictory (nothing to relabel)."""
        from opendsm.common.clustering.settings import ClusteringSettings
        with pytest.raises(ValueError, match="min_cluster_size=1"):
            ClusteringSettings(min_cluster_size=1, small_cluster_mode=SmallClusterMode.OUTLIER)

    def test_min_cluster_size_1_absorb_rejected(self):
        """min_cluster_size=1 + ABSORB is contradictory (nothing to absorb)."""
        from opendsm.common.clustering.settings import ClusteringSettings
        with pytest.raises(ValueError, match="min_cluster_size=1"):
            ClusteringSettings(min_cluster_size=1, small_cluster_mode=SmallClusterMode.ABSORB)

    def test_min_cluster_size_2_keep_rejected(self):
        """min_cluster_size=2 + KEEP is contradictory (KEEP preserves all sizes)."""
        from opendsm.common.clustering.settings import ClusteringSettings
        with pytest.raises(ValueError, match="small_cluster_mode='keep'"):
            ClusteringSettings(min_cluster_size=2, small_cluster_mode=SmallClusterMode.KEEP)

    @pytest.mark.parametrize("mode", [SmallClusterMode.OUTLIER, SmallClusterMode.ABSORB],
                             ids=["outlier", "absorb"])
    def test_min_cluster_size_2_non_keep_accepted(self, mode):
        """min_cluster_size>=2 with OUTLIER or ABSORB is valid."""
        from opendsm.common.clustering.settings import ClusteringSettings
        cs = ClusteringSettings(min_cluster_size=2, small_cluster_mode=mode)
        assert cs.min_cluster_size == 2
        assert cs.small_cluster_mode == mode

    def test_defaults(self):
        """Default settings: min_cluster_size=1, small_cluster_mode=KEEP."""
        from opendsm.common.clustering.settings import ClusteringSettings
        cs = ClusteringSettings()
        assert cs.min_cluster_size == 1, (
            f"Default min_cluster_size should be 1, got {cs.min_cluster_size}"
        )
        assert cs.small_cluster_mode == SmallClusterMode.KEEP, (
            f"Default small_cluster_mode should be KEEP, got {cs.small_cluster_mode}"
        )


# ── Validity-index correctness: analytic values & monotonicity ───────────────

def _blob_lm(sep, k=3, n=40, d=4, seed=0):
    """SingleKMetrics on k isotropic blobs whose centres are `sep` apart."""
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c * sep, 1.0, (n, d)) for c in range(k)]).astype(np.float32)
    labels = np.repeat(np.arange(k), n)
    lm = SingleKMetrics(data=X, labels=labels, seed=42)

    return lm


# ksq_detw measures within-cluster compactness only (k^2 * det(W)); pure
# separation moves centres without changing within-cluster scatter, so it is
# separation-invariant rather than separation-monotone.
_SEPARATION_MONOTONE_INDICES = sorted(SINGLE_K_INDEX_NAMES - {"ksq_detw_index"})


class TestValidityIndexAnalyticValues:
    """Pin validity indices to reference implementations on fixed geometry."""

    def test_silhouette_matches_hand_computed_three_point(self):
        """Silhouette on a 3-point, 2-cluster geometry equals the hand value.

        Points (0,0),(0,1) in cluster 0 and (0,5) alone in cluster 1:
        s=0.8 for (0,0), s=0.75 for (0,1), s=0 for the singleton, mean
        0.51667.  The index negates (minimize convention), so it is the
        negative of that mean.
        """
        X = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 5.0]], dtype=np.float32)
        labels = np.array([0, 0, 1])
        lm = SingleKMetrics(data=X, labels=labels, seed=42)
        assert float(lm.silhouette_index) == pytest.approx(-0.516667, abs=1e-5)

    def test_silhouette_index_is_negated_sklearn(self):
        """silhouette_index equals the negative of sklearn's silhouette_score."""
        lm = _blob_lm(sep=8.0)
        X, labels = np.asarray(lm.data), np.asarray(lm.labels)
        assert float(lm.silhouette_index) == pytest.approx(-silhouette_score(X, labels), abs=1e-5)

    def test_davies_bouldin_matches_sklearn(self):
        """davies_bouldin_index equals sklearn's davies_bouldin_score."""
        lm = _blob_lm(sep=8.0)
        X, labels = np.asarray(lm.data), np.asarray(lm.labels)
        assert float(lm.davies_bouldin_index) == pytest.approx(davies_bouldin_score(X, labels), abs=1e-5)


class TestValidityIndexMonotonicity:
    """Every quality index improves as cluster separation grows.

    All indices are normalized to minimize (maximize-natural ones are
    negated), so a wider gap must produce a strictly smaller value.  A single
    failing index pinpoints a sign/formula inversion.
    """

    @pytest.fixture(scope="class")
    def low_sep(self):
        return _blob_lm(sep=2.0)

    @pytest.fixture(scope="class")
    def high_sep(self):
        return _blob_lm(sep=30.0)

    @pytest.mark.parametrize("index", _SEPARATION_MONOTONE_INDICES)
    def test_improves_with_separation(self, index, low_sep, high_sep):
        """index(well-separated) < index(overlapping)."""
        v_low = float(getattr(low_sep, index))
        v_high = float(getattr(high_sep, index))
        assert np.isfinite(v_low) and np.isfinite(v_high), f"{index} not finite"
        assert v_high < v_low, f"{index} did not improve with separation ({v_low} -> {v_high})"

    def test_ksq_detw_is_separation_invariant(self):
        """ksq_detw (within-cluster only) is unchanged by pure separation."""
        v_low = float(_blob_lm(sep=2.0).ksq_detw_index)
        v_high = float(_blob_lm(sep=30.0).ksq_detw_index)
        assert v_high == pytest.approx(v_low, rel=1e-6)


class TestDBCV:
    """Density-Based Clustering Validation index behaviour."""

    def test_increases_with_separation(self):
        """DBCV is higher (better) for well-separated than overlapping blobs."""
        def blobs(sep, seed=0):
            rng = np.random.default_rng(seed)
            X = np.vstack([rng.normal(c * sep, 1.0, (40, 4)) for c in range(3)])
            y = np.repeat(np.arange(3), 40)

            return X, y

        x_lo, y_lo = blobs(3.0)
        x_hi, y_hi = blobs(30.0)
        assert dbcv(x_hi, y_hi) > dbcv(x_lo, y_lo)

    def test_low_for_random_labels_on_uniform(self):
        """Random labels over uniform data score poorly (well below structure)."""
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 10, (120, 4))
        y = rng.integers(0, 3, 120)
        assert dbcv(X, y) < 0.1

    def test_all_noise_returns_zero(self):
        """All-noise labeling has no clusters to validate -> 0.0."""
        X = np.random.default_rng(0).normal(0, 1, (40, 4))
        y = np.full(40, -1)
        assert dbcv(X, y) == 0.0

    def test_duplicates_raise_when_checked(self):
        """Duplicate rows are rejected when check_duplicates is on."""
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(c * 30, 1.0, (20, 4)) for c in range(2)])
        X[1] = X[0]
        with pytest.raises(ValueError, match="Duplicated samples"):
            dbcv(X, np.repeat([0, 1], 20), check_duplicates=True)

    def test_duplicates_tolerated_when_unchecked(self):
        """With the check off, duplicates are tolerated and a score returns."""
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(c * 30, 1.0, (20, 4)) for c in range(2)])
        X[1] = X[0]
        result = dbcv(X, np.repeat([0, 1], 20), check_duplicates=False)
        assert np.isfinite(result)

    def test_prevalidated_matches_full_dbcv(self):
        """dbcv_prevalidated (fast path on noise-free inputs) equals dbcv()."""
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(c * 30, 1.0, (15, 4)) for c in range(3)])
        y = np.repeat(np.arange(3), 15)

        full = dbcv(X, y)

        distances = cdist(X, X, metric="sqeuclidean")
        members = [np.where(y == c)[0] for c in range(3)]
        sizes = np.array([len(m) for m in members])
        prevalidated = dbcv_prevalidated(len(X), X.shape[1], sizes, members, distances)

        assert prevalidated == pytest.approx(full, rel=1e-9)


# ── Voter discriminability weighting ─────────────────────────────────────────

class TestDiscriminabilityWeights:
    """Council weights scale by each voter's coefficient of variation."""

    def test_flat_voter_downweighted_sharp_kept(self):
        """A constant-score voter loses almost all weight; a varied one keeps it."""
        score_matrix = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 10.0]])
        adjusted = selection._discriminability_weights(
            score_matrix, ["flat", "sharp"], {"flat": 1.0, "sharp": 1.0},
        )
        assert adjusted["flat"] == pytest.approx(0.0, abs=1e-6)
        assert adjusted["sharp"] > 0.5

    def test_zero_weight_voter_skipped(self):
        """A voter with non-positive council weight is left untouched."""
        score_matrix = np.array([[1.0], [2.0], [3.0]])
        adjusted = selection._discriminability_weights(
            score_matrix, ["v"], {"v": 0.0},
        )
        assert adjusted["v"] == 0.0


# ── prepare_labels small-cluster strategies ──────────────────────────────────

class TestPrepareLabels:
    """The small-cluster strategy branch of label preparation."""

    @pytest.fixture
    def labels_with_singleton(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, (10, 3))
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 2])  # cluster 2 is a singleton

        return data, labels

    def test_keep_preserves_all_clusters(self, labels_with_singleton):
        """KEEP retains every cluster, full coverage, no outliers introduced."""
        data, labels = labels_with_singleton
        merged, _, labels_clean, coverage = prepare_labels(
            labels, data, ScoreSettings(), None,
            min_cluster_size=1, small_cluster_mode=SmallClusterMode.KEEP,
        )
        assert len(np.unique(labels_clean)) == 3
        assert coverage == 1.0
        assert -1 not in merged

    def test_outlier_relabels_small_cluster(self, labels_with_singleton):
        """OUTLIER demotes the sub-threshold cluster to -1 and lowers coverage."""
        data, labels = labels_with_singleton
        merged, _, labels_clean, coverage = prepare_labels(
            labels, data, ScoreSettings(), None,
            min_cluster_size=3, small_cluster_mode=SmallClusterMode.OUTLIER,
        )
        assert -1 in merged
        assert coverage == pytest.approx(0.9)
        assert len(np.unique(labels_clean)) == 2

    def test_below_lower_bound_returns_none(self, labels_with_singleton):
        """Fewer clusters than n_cluster_lower invalidates the labeling."""
        data, labels = labels_with_singleton
        _, data_clean, labels_clean, _ = prepare_labels(
            labels, data, ScoreSettings(), 5,
            min_cluster_size=1, small_cluster_mode=SmallClusterMode.KEEP,
        )
        assert data_clean is None
        assert labels_clean is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
