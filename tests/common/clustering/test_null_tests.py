"""Tests for cluster structure null tests (CrossKMetrics null tests + gate)."""

from __future__ import annotations

import warnings

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.cluster import KMeans

from opendsm.common.clustering.metrics.cross_k_metrics import (
    CrossKMetrics,
    has_cluster_structure,
)
from opendsm.common.clustering.metrics import selection as _selection
from opendsm.common.clustering.metrics.labels import ClusteringResult
from opendsm.common.clustering.metrics.settings import ScoreSettings, SmallClusterMode
from opendsm.common.clustering.metrics.single_k_metrics import SingleKMetrics


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_ckm(X: np.ndarray, max_k: int = 6, seed: int = 42) -> CrossKMetrics:
    """Build CrossKMetrics from raw data with KMeans WCSS."""
    n = X.shape[0]
    wcss = {}
    for k in range(1, min(max_k + 1, n)):
        wcss[k] = KMeans(
            n_clusters=k, n_init=3, random_state=seed
        ).fit(X).inertia_
    return CrossKMetrics(
        wcss_by_k=wcss,
        k_values=sorted(wcss),
        n_features=X.shape[1],
        n_samples=n,
        raw_data=X,
        seed=seed,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def well_separated_2cl():
    rng = np.random.default_rng(42)
    return np.vstack([
        rng.normal(0, 0.5, (30, 5)),
        rng.normal(5, 0.5, (30, 5)),
    ])


@pytest.fixture(scope="module")
def single_gaussian():
    return np.random.default_rng(42).normal(0, 1, (60, 5))


@pytest.fixture(scope="module")
def uniform_random():
    return np.random.default_rng(42).uniform(0, 10, (60, 5))


_ALL_NULL_TESTS = ["gap_statistic", "hopkins_test", "sigclust_test", "spectral_gap_test"]


# ── Cross-cutting: NaN when data is None, determinism ────────────────────────

class TestNullTestsNanWhenNoData:
    @pytest.mark.parametrize("test_name", _ALL_NULL_TESTS)
    def test_returns_nan_when_data_is_none(self, test_name):
        ckm = CrossKMetrics(
            wcss_by_k={1: 100, 2: 50}, k_values=[1, 2],
            n_features=5, n_samples=60, raw_data=None,
        )
        assert np.isnan(getattr(ckm, test_name)), (
            f"{test_name} should return NaN when data is None"
        )


class TestNullTestsDeterminism:
    @pytest.mark.parametrize("test_name", _ALL_NULL_TESTS)
    @pytest.mark.slow
    def test_deterministic_with_same_seed(self, well_separated_2cl, test_name):
        r1 = getattr(_make_ckm(well_separated_2cl, seed=77), test_name)
        r2 = getattr(_make_ckm(well_separated_2cl, seed=77), test_name)
        assert np.isclose(r1, r2, rtol=1e-10, equal_nan=True), (
            f"{test_name} not deterministic: {r1} vs {r2}"
        )


# ── Regression: exact p-values for canonical datasets ────────────────────────

class TestNullTestRegression:
    """Pin exact p-values so silent behavioral changes are caught."""

    def test_well_separated_regression(self, well_separated_2cl):
        ckm = _make_ckm(well_separated_2cl)
        assert ckm.gap_statistic == pytest.approx(0.0, abs=1e-20)
        assert ckm.sigclust_test == pytest.approx(8.252e-29, rel=0.05)
        assert ckm.spectral_gap_test < 0.01  # strongly significant with 5 refs

    def test_gaussian_regression(self, single_gaussian):
        ckm = _make_ckm(single_gaussian)
        assert ckm.gap_statistic == pytest.approx(1.165e-41, rel=0.01)
        assert ckm.sigclust_test == pytest.approx(0.5, abs=0.01)
        assert ckm.spectral_gap_test == pytest.approx(1.0, abs=0.01)

    def test_uniform_regression(self, uniform_random):
        ckm = _make_ckm(uniform_random)
        assert ckm.gap_statistic > 0.05  # not significant (uniform data)
        assert ckm.sigclust_test == pytest.approx(0.5, abs=0.01)


# ── Gap statistic ────────────────────────────────────────────────────────────

class TestGapStatistic:
    def test_detects_well_separated_clusters(self, well_separated_2cl):
        assert _make_ckm(well_separated_2cl).gap_statistic < 0.05

    def test_returns_1_when_no_k_ge_2(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (10, 3))
        ckm = CrossKMetrics(
            wcss_by_k={1: 100.0}, k_values=[1], n_features=3,
            n_samples=10, raw_data=X, seed=42,
        )
        assert ckm.gap_statistic == 1.0

    def test_n_scored_changes_result(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (60, 5))
        wcss = {1: 300.0, 2: 150.0, 3: 100.0}
        ckm_full = CrossKMetrics(
            wcss_by_k=wcss, k_values=[1, 2, 3], n_features=5,
            n_samples=60, raw_data=X, seed=42,
        )
        ckm_scored = CrossKMetrics(
            wcss_by_k=wcss, n_scored_by_k={1: 60, 2: 54, 3: 54},
            k_values=[1, 2, 3], n_features=5, n_samples=60,
            raw_data=X, seed=42,
        )
        assert ckm_scored.gap_statistic != ckm_full.gap_statistic


# ── Hopkins test ─────────────────────────────────────────────────────────────

class TestHopkinsTest:
    def test_detects_well_separated_clusters(self, well_separated_2cl):
        assert _make_ckm(well_separated_2cl).hopkins_test < 0.05

    def test_high_pvalue_for_uniform(self, uniform_random):
        assert _make_ckm(uniform_random).hopkins_test > 0.05

    @pytest.mark.parametrize("d, expect_finite", [(10, True), (11, True), (24, True)])
    def test_finite_for_high_d(self, d, expect_finite):
        """Hopkins uses PCA reduction for d > 10, so it should remain finite."""
        rng = np.random.default_rng(42)
        result = _make_ckm(rng.normal(0, 1, (60, d))).hopkins_test
        assert np.isfinite(result) == expect_finite

    def test_nan_when_n_too_small(self):
        ckm = CrossKMetrics(
            wcss_by_k={1: 1.0}, k_values=[1], n_features=2,
            n_samples=2, raw_data=np.array([[0.0, 1.0], [1.0, 0.0]]), seed=42,
        )
        assert np.isnan(ckm.hopkins_test)


# ── SigClust test ────────────────────────────────────────────────────────────

class TestSigclustTest:
    def test_detects_well_separated_clusters(self, well_separated_2cl):
        assert _make_ckm(well_separated_2cl).sigclust_test < 0.05

    def test_high_pvalue_for_single_gaussian(self, single_gaussian):
        assert _make_ckm(single_gaussian).sigclust_test > 0.10

    def test_nan_when_n_lt_4(self):
        ckm = CrossKMetrics(
            wcss_by_k={1: 1.0, 2: 0.5}, k_values=[1, 2],
            n_features=2, n_samples=3,
            raw_data=np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]), seed=42,
        )
        assert np.isnan(ckm.sigclust_test)

    def test_computes_wcss1_analytically_when_missing(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (30, 5)), rng.normal(5, 0.5, (30, 5))])
        wcss = {2: KMeans(n_clusters=2, n_init=3, random_state=42).fit(X).inertia_}
        ckm = CrossKMetrics(
            wcss_by_k=wcss, k_values=[2], n_features=5,
            n_samples=60, raw_data=X, seed=42,
        )
        assert not np.isnan(ckm.sigclust_test)

    def test_effect_size_gate_on_gaussian(self):
        rng = np.random.default_rng(42)
        assert _make_ckm(rng.normal(0, 1, (200, 5))).sigclust_test >= 0.5

    def test_multi_k_detects_5_cluster_structure(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(c, 0.5, (20, 5)) for c in np.eye(5) * 3])
        wcss = {k: KMeans(n_clusters=k, n_init=3, random_state=42).fit(X).inertia_
                for k in [1, 2, 3, 4, 5]}
        ckm = CrossKMetrics(
            wcss_by_k=wcss, k_values=[1, 2, 3, 4, 5], n_features=5,
            n_samples=100, raw_data=X, seed=42,
        )
        assert ckm.sigclust_test < 0.05


# ── Spectral gap test ────────────────────────────────────────────────────────

class TestSpectralGapTest:
    def test_lower_pvalue_for_clusters_than_gaussian(self):
        rng = np.random.default_rng(42)
        p_cl = _make_ckm(np.vstack([rng.normal(0, 0.5, (50, 5)), rng.normal(5, 0.5, (50, 5))])).spectral_gap_test
        p_ga = _make_ckm(rng.normal(0, 1, (100, 5))).spectral_gap_test
        # Both should produce small p-values (strong structure detected)
        assert p_cl < 0.05
        assert isinstance(p_ga, float)

    def test_nan_when_n_lt_4(self):
        ckm = CrossKMetrics(
            wcss_by_k={1: 1.0}, k_values=[1], n_features=2,
            n_samples=3, raw_data=np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]),
            seed=42,
        )
        assert np.isnan(ckm.spectral_gap_test)

    def test_effect_size_gate_at_large_n(self):
        rng = np.random.default_rng(42)
        assert _make_ckm(rng.normal(0, 1, (1000, 5))).spectral_gap_test >= 0.5

    def test_axis_aligned_clusters_detected(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal([-5, 0, 0, 0, 0], 1, (40, 5)),
                        rng.normal([5, 0, 0, 0, 0], 1, (40, 5))])
        ckm = _make_ckm(X)
        assert ckm.sigclust_test < 0.05 or ckm.spectral_gap_test < 0.05


# ── _max_eigengap helper ────────────────────────────────────────────────────

class TestMaxEigengap:
    @pytest.mark.parametrize("n, k_nn", [(20, 2), (50, 5)],
                             ids=["dense_path", "sparse_path"])
    def test_returns_non_negative_float(self, n, k_nn):
        X = np.random.default_rng(42).normal(0, 1, (n, 3))
        gap = CrossKMetrics._max_eigengap(X, k_nn=k_nn, n_eig=5, seed=42)
        assert isinstance(gap, float) and gap >= 0

    def test_deterministic_with_seed(self):
        X = np.random.default_rng(42).normal(0, 1, (50, 3))
        assert CrossKMetrics._max_eigengap(X, 5, 5, seed=99) == \
               CrossKMetrics._max_eigengap(X, 5, 5, seed=99)

    def test_larger_for_clustered_than_gaussian(self):
        rng = np.random.default_rng(42)
        X_cl = np.vstack([rng.normal(0, 0.5, (25, 3)), rng.normal(5, 0.5, (25, 3))])
        X_ga = rng.normal(0, 1, (50, 3))
        assert CrossKMetrics._max_eigengap(X_cl, 5, 5, 42) > \
               CrossKMetrics._max_eigengap(X_ga, 5, 5, 42)


# ── has_cluster_structure gate ───────────────────────────────────────────────

class TestHasClusterStructureGate:
    @pytest.mark.parametrize("fixture_name, expected", [
        ("well_separated_2cl", True),
        ("single_gaussian", False),
        ("uniform_random", False),
    ])
    @pytest.mark.slow
    def test_gate_decision(self, fixture_name, expected, request):
        result = has_cluster_structure(_make_ckm(request.getfixturevalue(fixture_name)))
        assert result is expected, f"Expected {expected} for {fixture_name}"

    def test_permissive_when_no_data(self):
        ckm = CrossKMetrics(
            wcss_by_k={2: 100}, k_values=[2], n_features=5,
            n_samples=60, raw_data=None,
        )
        assert has_cluster_structure(ckm) is True

    @pytest.mark.slow
    def test_requires_both_distance_and_model_groups(self, single_gaussian):
        assert has_cluster_structure(_make_ckm(single_gaussian)) is False

    @pytest.mark.slow
    def test_hopkins_works_for_high_d_clusters(self):
        """Hopkins now uses PCA for d > 10, so it detects structure in high-d data."""
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (30, 20)), rng.normal(3, 0.5, (30, 20))])
        ckm = _make_ckm(X)
        assert np.isfinite(ckm.hopkins_test)
        assert ckm.hopkins_test < 0.05  # detects the two clusters
        assert has_cluster_structure(ckm) is True


class TestGateGroupedAgreementLogic:
    """Branch logic of ``has_cluster_structure`` with injected p-values.

    The gate reads only the four null-test p-values; injecting them directly
    exercises every grouped-agreement branch deterministically, independent
    of any dataset.  Distance group = (gap, hopkins); model group =
    (sigclust, spectral_gap).  alpha defaults to 0.05.
    """

    @staticmethod
    def _stub(gap, hopkins, sigclust, spectral_gap):
        """A minimal object exposing the four p-value attributes the gate reads."""
        ns = SimpleNamespace(
            gap_statistic=gap,
            hopkins_test=hopkins,
            sigclust_test=sigclust,
            spectral_gap_test=spectral_gap,
        )

        return ns

    @pytest.mark.parametrize("pvals, expected, why", [
        ((0.01, 0.50, 0.01, 0.50), True, "both groups have a significant test"),
        ((0.01, 0.50, 0.50, 0.50), False, "only the distance group is significant"),
        ((0.50, 0.50, 0.01, 0.50), False, "only the model group is significant"),
        ((0.50, 0.50, 0.50, 0.50), False, "no group is significant"),
        ((0.01, 0.01, np.nan, np.nan), True, "model group absent, 2-of-N majority met"),
        ((0.01, 0.50, np.nan, np.nan), False, "model group absent, majority not met"),
        ((np.nan, np.nan, 0.01, 0.01), True, "distance group absent, majority met"),
        ((0.01, np.nan, np.nan, np.nan), True, "fewer than 2 computable -> permissive"),
        ((np.nan, np.nan, np.nan, np.nan), True, "nothing computable -> permissive"),
    ])
    def test_branches(self, pvals, expected, why):
        """Each grouped-agreement branch resolves as documented."""
        result = has_cluster_structure(self._stub(*pvals))
        assert result is expected, why

    def test_alpha_threshold_is_strict(self):
        """A p-value exactly at alpha is not significant (strict less-than)."""
        at_alpha = self._stub(0.05, 0.50, 0.05, 0.50)
        assert has_cluster_structure(at_alpha, alpha=0.05) is False

        below = self._stub(0.049, 0.50, 0.049, 0.50)
        assert has_cluster_structure(below, alpha=0.05) is True


# ── k=1 index abstention ────────────────────────────────────────────────────

class TestK1IndexAbstention:
    ALL_INDICES = [
        'silhouette_index', 'silhouette_median_index', 'davies_bouldin_index',
        'calinski_harabasz_index', 'dunn_index', 'xie_beni_index', 'i_index',
        'point_biserial_index', 'sd_validity_index', 'simplified_silhouette_index',
        'cop_index', 'negentropy_index', 'wb_index', 'generalized_dunn_index',
    ]

    @pytest.fixture(scope="class")
    def k1_metrics(self):
        rng = np.random.default_rng(42)
        return SingleKMetrics(
            data=rng.normal(0, 1, (30, 5)).astype(np.float32),
            labels=np.zeros(30, dtype=int), seed=42,
        )

    @pytest.mark.parametrize("index", ALL_INDICES)
    def test_returns_nan_at_k1(self, k1_metrics, index):
        assert np.isnan(getattr(k1_metrics, index)), (
            f"{index} should return NaN at k=1"
        )


# ── ClusteringResult integration ────────────────────────────────────────────

class TestClusteringResultGateIntegration:
    def test_returns_k1_with_confidence_0_when_no_structure(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (60, 5))
        cr = ClusteringResult(data=X, seed=42)
        cr.add(1, np.zeros(60, dtype=int))
        cr.add(2, np.repeat([0, 1], 30))
        assert cr.k == 1
        assert cr.selection_confidence == 0.0
        assert np.all(cr.labels == 0)

    def test_selects_k_gt_1_when_structure_exists(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (30, 5)), rng.normal(10, 0.5, (30, 5))])
        cr = ClusteringResult(data=X, seed=42)
        cr.add(1, np.zeros(60, dtype=int))
        cr.add(2, KMeans(n_clusters=2, n_init=3, random_state=42).fit_predict(X))
        assert cr.has_cluster_structure is True
        assert cr.k >= 2

    @pytest.mark.slow
    def test_council_never_picks_k1(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(c, 0.5, (20, 3)) for c in [0, 5, 10]])
        cr = ClusteringResult(data=X, seed=42)
        cr.add(1, np.zeros(60, dtype=int))
        for k in [2, 3]:
            cr.add(k, KMeans(n_clusters=k, n_init=3, random_state=42).fit_predict(X))
        if cr.has_cluster_structure:
            assert cr.k >= 2


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    @pytest.mark.slow
    def test_d_equals_1(self):
        ckm = _make_ckm(np.random.default_rng(42).normal(0, 1, (30, 1)))
        for name in ["gap_statistic", "sigclust_test", "spectral_gap_test"]:
            assert isinstance(getattr(ckm, name), float)

    def test_n_equals_4_minimum(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.1, (2, 3)), rng.normal(5, 0.1, (2, 3))])
        ckm = _make_ckm(X, max_k=2)
        assert isinstance(ckm.sigclust_test, float)
        assert isinstance(ckm.spectral_gap_test, float)

    def test_identical_data_does_not_crash(self):
        ckm = CrossKMetrics(
            wcss_by_k={1: 0.0, 2: 0.0}, k_values=[1, 2],
            n_features=3, n_samples=20, raw_data=np.ones((20, 3)), seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name in _ALL_NULL_TESTS:
                assert isinstance(getattr(ckm, name), float)


# ── Settings ─────────────────────────────────────────────────────────────────

class TestNullTestAlphaSetting:
    def test_default_is_010(self):
        assert ScoreSettings().null_test_alpha == 0.10

    def test_accepts_valid_range(self):
        assert ScoreSettings(null_test_alpha=0.10).null_test_alpha == 0.10

    @pytest.mark.parametrize("bad_value", [0.0, 1.0, -0.1])
    def test_rejects_invalid(self, bad_value):
        with pytest.raises(Exception):
            ScoreSettings(null_test_alpha=bad_value)


# ── Labels integration edge cases ────────────────────────────────────────────

class TestLabelsIntegrationEdgeCases:
    """Test untested code paths in labels.py."""

    @pytest.mark.slow
    def test_council_k1_exclusion_fallback_returns_first_valid_k2(self):
        """When the gate passes but the Schulze winner is None (because the
        only k>=2 candidate was invalidated), _compute_labels_best should
        fall back to the first valid k>=2 candidate.

        We construct a scenario with k=1 (valid) and two k=2 entries where
        one is valid and one is None.  We force has_cluster_structure=True
        by using well-separated data.  The council should exclude k=1;
        even if Schulze picks the None slot, the fallback loop returns the
        valid k=2 entry.
        """
        rng = np.random.default_rng(99)
        X = np.vstack([rng.normal(0, 0.3, (30, 3)), rng.normal(8, 0.3, (30, 3))])
        n = X.shape[0]

        cr = ClusteringResult(data=X, seed=42)
        # k=1: valid single-cluster labeling
        cr.add(1, np.zeros(n, dtype=int))
        # k=2: good labeling
        good_labels = np.array([0] * 30 + [1] * 30)
        cr.add(2, good_labels)
        # k=2: degenerate labeling — all points in one tiny cluster + rest
        # outlier.  Make a labeling where min_cluster_size policy will
        # collapse it to a single cluster (which then becomes k=1 effective),
        # but it's stored under k=2.  Use labels where one "cluster" has
        # only 1 member so prepare_labels merges it → effective k=1 → returns None.
        bad_labels = np.zeros(n, dtype=int)
        bad_labels[0] = 1  # cluster 1 has 1 member → below min_cluster_size=2
        cr.add(2, bad_labels)

        # The result must be k>=2 (the good labeling), not k=1
        assert cr.k >= 2
        assert cr.has_cluster_structure is True

    @pytest.mark.parametrize("lower, k1_allowed", [(1, True), (2, False)])
    def test_degenerate_labeling_respects_k1_allowance(self, lower, k1_allowed):
        """A labeling that collapses to one cluster is returned as a single
        cluster when the lower bound allows k=1, and raises a clear error
        (naming n_cluster_lower) when it does not.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (40, 4))

        # OUTLIER trimming sends the lone 2-point cluster to -1, collapsing
        # the labeling to a single cluster.
        cr = ClusteringResult(
            data=X, seed=42, n_cluster_lower=lower, min_cluster_size=3,
            small_cluster_mode=SmallClusterMode.OUTLIER,
        )
        labels = np.zeros(40, dtype=int)
        labels[:2] = 1
        cr.add(2, labels)

        if k1_allowed:
            assert cr.k == 1
            assert cr.selection_confidence == 1.0
        else:
            assert not cr._labels_store
            for accessor in ("k", "metrics", "labels"):
                with pytest.raises(ValueError, match=f"n_cluster_lower={lower}"):
                    getattr(cr, accessor)

    def test_valid_label_count_excludes_outliers(self):
        """valid_label_count / unique_valid_labels drop the -1 outlier label,
        while label_count / unique_labels keep it as a literal unique label.
        """
        X = np.random.default_rng(0).normal(0, 1, (10, 3)).astype(np.float32)
        labels = np.array([0, 0, 0, 1, 1, 1, -1, -1, 2, 2])
        lm = SingleKMetrics(data=X, labels=labels, distance_metric="euclidean", seed=0)

        assert lm.label_count == 4
        assert -1 in lm.unique_labels
        assert lm.valid_label_count == 3
        assert -1 not in lm.unique_valid_labels

    def test_single_specified_k_bypasses_cross_k_selection(self, monkeypatch):
        """With a single specified k, the within-k best is returned with
        full confidence and the cross-k council never runs.
        """
        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(c, 0.3, (20, 4)) for c in (0, 5, 10)])

        cr = ClusteringResult(data=X, seed=42, n_cluster_lower=3)
        cr.add(3, KMeans(n_clusters=3, n_init=3, random_state=0).fit_predict(X))
        cr.add(3, KMeans(n_clusters=3, n_init=3, random_state=1).fit_predict(X))

        def _must_not_run(*args, **kwargs):
            raise AssertionError("cross-k council should be bypassed for one k")

        monkeypatch.setattr(_selection, "select_best_across_k", _must_not_run)

        assert cr.k_values == [3]
        assert cr.k == 3
        assert cr.selection_confidence == 1.0

    def test_metrics_raise_when_no_labels_added(self):
        """Accessing metrics on a result with no labels at all raises a
        distinct message from the all-rejected case.
        """
        rng = np.random.default_rng(2)
        X = rng.normal(0, 1, (30, 4))

        cr = ClusteringResult(data=X, seed=42)

        with pytest.raises(ValueError, match="No labels have been added"):
            _ = cr.metrics

    def test_unscored_council_returns_smallest_specified_k(self, monkeypatch):
        """When valid candidates exist but the council yields no scored
        winner (None-slot), the result falls back to the candidate at the
        smallest user-specified k (n_cluster_lower), not insertion order.
        """
        rng = np.random.default_rng(7)
        X = np.vstack([rng.normal(c, 0.4, (25, 4)) for c in (0, 6, 12)])

        cr = ClusteringResult(
            data=X, seed=42, n_cluster_lower=2, min_cluster_size=3,
            small_cluster_mode=SmallClusterMode.OUTLIER,
        )
        cr.add(2, KMeans(n_clusters=2, n_init=3, random_state=0).fit_predict(X))
        cr.add(3, KMeans(n_clusters=3, n_init=3, random_state=0).fit_predict(X))
        cr.add(4, KMeans(n_clusters=4, n_init=3, random_state=0).fit_predict(X))
        # A rejected (collapsing) candidate: its 2-point cluster is outliered,
        # dropping below the lower bound, so it becomes a None slot the
        # council winner can land on, forcing the unscored fallback.
        collapse = np.zeros(75, dtype=int)
        collapse[:2] = 1
        cr.add(2, collapse)

        none_slot = cr._insertion_order.index(None)
        monkeypatch.setattr(
            _selection, "select_best_across_k",
            lambda *a, **k: (none_slot, 0.0),
        )

        assert cr.k == 2

    def test_build_cross_k_extra_scores_injects_into_candidates(self):
        """_build_cross_k_extra_scores should return a list parallel to
        candidates, with cross-k metric scores keyed by metric name.
        Verify that scores are non-empty dicts for valid candidates and
        empty dicts for None candidates.
        """
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(c, 0.5, (25, 3)) for c in [0, 5, 10]])
        n = X.shape[0]

        # Enable a cross-k metric in the council
        weights = {
            'davies_bouldin_index': 1.0,
            'krzanowski_lai_index': 1.0,
        }
        ss = ScoreSettings(weights=weights)
        cr = ClusteringResult(data=X, score_settings=ss, seed=42)
        cr.add(1, np.zeros(n, dtype=int))
        cr.add(2, KMeans(n_clusters=2, n_init=3, random_state=42).fit_predict(X))
        cr.add(3, KMeans(n_clusters=3, n_init=3, random_state=42).fit_predict(X))

        # Build candidates the same way _compute_labels_best does
        candidates = list(cr._insertion_order)
        # Exclude k=1
        lm_to_k = {}
        for k, lms in cr._labels_store.items():
            for lm in lms:
                lm_to_k[id(lm)] = k
        council_candidates = [
            c if c is None or lm_to_k.get(id(c), 0) >= 2 else None
            for c in candidates
        ]

        extra = cr._build_cross_k_extra_scores(council_candidates, ss.weights)

        assert extra is not None
        assert len(extra) == len(council_candidates)

        for i, c in enumerate(council_candidates):
            if c is None:
                assert extra[i] == {}
            else:
                # Should have the cross-k metric injected
                assert 'krzanowski_lai_index' in extra[i]

    def test_has_cluster_structure_no_k1_skips_gate(self):
        """When k=1 is NOT in k_best, has_cluster_structure is still
        evaluated but the gate in _compute_labels_best (line 372:
        ``if 1 in k_best and not self.has_cluster_structure``) does
        not fire.  The council selects among k>=2 candidates directly.
        """
        rng = np.random.default_rng(42)
        # Use uniform data that would fail the structure gate
        X = rng.uniform(0, 10, (60, 5))

        cr = ClusteringResult(data=X, seed=42)
        # Only add k=2 and k=3, no k=1
        cr.add(2, KMeans(n_clusters=2, n_init=3, random_state=42).fit_predict(X))
        cr.add(3, KMeans(n_clusters=3, n_init=3, random_state=42).fit_predict(X))

        # Gate should NOT fire (no k=1 in k_best), so we get a k>=2 result
        # even though the data has no real structure
        assert 1 not in cr._labels_k_best
        assert cr.k >= 2
        # Confidence should be > 0 since the gate didn't zero it out
        assert cr.selection_confidence > 0.0

    def test_set_labels_k_best_overrides_within_k_selection(self):
        """_set_labels_k_best forces a specific index within a k's
        labeling list to be the within-k winner, overriding the
        default select_best_within_k vote.
        """
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (30, 3)), rng.normal(5, 0.5, (30, 3))])
        n = X.shape[0]

        cr = ClusteringResult(data=X, seed=42)
        cr.add(1, np.zeros(n, dtype=int))

        # Two different k=2 labelings
        labels_a = np.array([0] * 30 + [1] * 30)
        labels_b = np.array([0] * 20 + [1] * 40)  # shifted boundary
        cr.add(2, labels_a)
        cr.add(2, labels_b)

        # Default within-k selection picks one
        default_best = cr._labels_k_best[2]

        # Override to pick index 1 (labels_b)
        cr._set_labels_k_best(2, 1)
        overridden = cr._labels_k_best[2]
        assert overridden is cr._labels_store[2][1]

        # Override to pick index 0 (labels_a)
        cr._set_labels_k_best(2, 0)
        overridden2 = cr._labels_k_best[2]
        assert overridden2 is cr._labels_store[2][0]

    def test_set_labels_k_best_raises_on_invalid_k(self):
        """_set_labels_k_best raises IndexError for non-existent k."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 3))
        cr = ClusteringResult(data=X, seed=42)
        cr.add(2, np.repeat([0, 1], 10))

        with pytest.raises(IndexError):
            cr._set_labels_k_best(5, 0)

    def test_set_labels_k_best_raises_on_invalid_index(self):
        """_set_labels_k_best raises IndexError for out-of-range index."""
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 3))
        cr = ClusteringResult(data=X, seed=42)
        cr.add(2, np.repeat([0, 1], 10))

        with pytest.raises(IndexError):
            cr._set_labels_k_best(2, 5)


# ── Council k=1 exclusion ────────────────────────────────────────────────────

class TestCouncilK1Exclusion:
    """Verify that k=1 is never selected by the council when data has structure."""

    @pytest.mark.slow
    def test_council_selects_k_ge_2_with_clustered_data(self):
        """With well-separated clusters and a k=1 candidate, council picks k>=2."""
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal([0, 0, 0], 0.3, (30, 3)),
            rng.normal([10, 10, 10], 0.3, (30, 3)),
            rng.normal([20, 20, 20], 0.3, (30, 3)),
        ])
        n = X.shape[0]

        cr = ClusteringResult(data=X, seed=42)
        cr.add(1, np.zeros(n, dtype=int))
        cr.add(2, KMeans(n_clusters=2, n_init=3, random_state=42).fit_predict(X))
        cr.add(3, KMeans(n_clusters=3, n_init=3, random_state=42).fit_predict(X))

        assert cr.has_cluster_structure is True, (
            "Well-separated data should pass the structure gate"
        )
        assert cr.k >= 2, (
            f"Council should select k>=2 for clustered data, got k={cr.k}"
        )

    def test_all_singletons_labeling_not_selected(self):
        """All-singletons labeling (k=n) does NOT win: all indices NaN -> excluded."""
        rng = np.random.default_rng(42)
        X = np.vstack([
            rng.normal([0, 0, 0], 0.3, (20, 3)),
            rng.normal([10, 10, 10], 0.3, (20, 3)),
        ])
        n = X.shape[0]

        cr = ClusteringResult(data=X, seed=42)
        cr.add(1, np.zeros(n, dtype=int))
        # Good k=2 labeling
        cr.add(2, KMeans(n_clusters=2, n_init=3, random_state=42).fit_predict(X))
        # All-singletons labeling: k=n
        cr.add(n, np.arange(n))

        assert cr.has_cluster_structure is True
        # The all-singletons labeling should NOT win — most indices return NaN
        # because every cluster is a singleton.
        assert cr.k != n, (
            f"All-singletons labeling (k={n}) should not be selected by the council"
        )
        assert cr.k >= 2, (
            f"Council should select k>=2, got k={cr.k}"
        )
