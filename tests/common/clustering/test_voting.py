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

import types

import numpy as np
import pandas as pd
import pytest

from opendsm.common.clustering.metrics.voting import (
    schulze_voting,
    build_rank_matrix,
    _schulze_pairwise_preference,
    _schulze_path_strength,
    _schulze_rank_strength,
    _build_rank_arrays,
)


def _proxy(scores: dict) -> types.SimpleNamespace:
    """Build a score proxy (same shape as production code uses)."""
    return types.SimpleNamespace(score=scores)


def _rank_arrays(proxies, voter_weights=None):
    """Helper: build rank arrays from proxies via build_rank_matrix."""
    score_matrix, voter_names, _ = build_rank_matrix(proxies)
    return _build_rank_arrays(score_matrix, voter_names, voter_weights)


class TestBuildRankMatrix:
    """build_rank_matrix returns (score_matrix, voter_names, abstain_mask)."""

    def test_nan_in_abstain_mask_inf_not(self):
        """NaN score -> abstain_mask entry; inf score -> active worst, not in mask."""
        results = [
            _proxy({'m': float('nan')}),  # candidate 0: abstain
            _proxy({'m': float('inf')}),  # candidate 1: active worst
            _proxy({'m': 1.0}),           # candidate 2: normal
        ]
        score_matrix, voter_names, mask = build_rank_matrix(results)
        assert isinstance(score_matrix, np.ndarray)
        assert isinstance(voter_names, list)
        assert isinstance(mask, dict)
        assert 0 in mask['m']     # NaN -> abstain
        assert 1 not in mask['m'] # inf -> active worst, not abstain

    def test_all_inf_column_present_in_raw_matrix(self):
        """All-inf columns are included in the raw score matrix (dropped later in voting)."""
        results = [
            _proxy({'m': 1.0, 'n': float('inf')}),
            _proxy({'m': 2.0, 'n': float('inf')}),
        ]
        score_matrix, voter_names, mask = build_rank_matrix(results)
        assert 'm' in voter_names
        assert 'n' in voter_names  # still present in raw output


class TestAbstainSemantics:
    """NaN = abstain (skip comparisons); inf = active worst vote (participates)."""

    def test_abstaining_voter_skips_pair_involving_abstained_candidate(self):
        """Voter that abstains on candidate 0 casts no vote for pairs involving it."""
        n_cand, m = 3, 2
        rank_of_candidate = np.array([[0, 0], [1, 1], [2, 2]])  # (n_cand, m)
        weights = np.array([1.0, 1.0])
        abstain_bool = np.array([[False, True], [False, False], [False, False]])

        P = _schulze_pairwise_preference(rank_of_candidate, weights, abstain_bool)

        assert P[0, 1] == pytest.approx(1.0)
        assert P[1, 0] == pytest.approx(0.0)
        assert P[1, 2] == pytest.approx(2.0)

    def test_inf_score_participates_but_ranks_last(self):
        """inf score is an active worst vote: candidate participates, ranks last."""
        results = [
            _proxy({'m': float('inf')}),  # candidate 0: active worst
            _proxy({'m': 1.0}),           # candidate 1: best
        ]
        score_matrix, voter_names, mask = build_rank_matrix(results)
        assert 0 not in mask.get('m', set())
        assert score_matrix[0, 0] > score_matrix[1, 0]

    def test_full_pipeline_nan_abstain_does_not_penalise_winner(self):
        """End-to-end: a voter with NaN on a candidate doesn't penalise that candidate."""
        results = [
            _proxy({'m': 1.0, 'n': float('nan')}),  # candidate 0
            _proxy({'m': 2.0, 'n': 0.5}),            # candidate 1
        ]
        score_matrix, voter_names, _ = build_rank_matrix(results)
        council = {'m': 1.0, 'n': 1.0}
        winner, conf = schulze_voting(score_matrix, voter_names, voter_weights=council)
        assert winner == 0
        assert conf == pytest.approx(1.0)


class TestSchulzePairwisePreference:
    """Pairwise preference matrix construction."""

    @pytest.mark.parametrize("i", [0, 1, 2])
    def test_diagonal_is_zero(self, i):
        proxies = [_proxy({'v1': 0.0, 'v2': 1.0}),
                   _proxy({'v1': 1.0, 'v2': 0.0}),
                   _proxy({'v1': 2.0, 'v2': 2.0})]
        score_matrix, voter_names, _ = build_rank_matrix(proxies)
        rank, weights, _, abstain, norm = _build_rank_arrays(score_matrix, voter_names)
        P = _schulze_pairwise_preference(rank, weights, abstain, normalized_scores=norm)
        assert P[i, i] == pytest.approx(0.0)

    def test_weighted_voters(self):
        """Higher-weight voter dominates pairwise outcome."""
        proxies = [_proxy({'v1': 0.0, 'v2': 1.0}),
                   _proxy({'v1': 1.0, 'v2': 0.0})]
        score_matrix, voter_names, _ = build_rank_matrix(proxies)
        weights_dict = {'v1': 10.0, 'v2': 1.0}
        rank, weights, _, abstain, norm = _build_rank_arrays(score_matrix, voter_names, weights_dict)
        P = _schulze_pairwise_preference(rank, weights, abstain, normalized_scores=norm)
        assert P[0, 1] > P[1, 0]

    def test_tie_gives_zero_preference(self):
        """Identical scores -> zero margin -> no preference for either candidate."""
        proxies = [_proxy({'v1': 1.0}), _proxy({'v1': 1.0})]  # tie
        score_matrix, voter_names, _ = build_rank_matrix(proxies)
        rank, weights, _, abstain, norm = _build_rank_arrays(score_matrix, voter_names)
        P = _schulze_pairwise_preference(rank, weights, abstain, normalized_scores=norm)
        assert P[0, 1] == pytest.approx(0.0)
        assert P[1, 0] == pytest.approx(0.0)


class TestSchulzePathStrength:
    """Floyd-Warshall path strength propagation."""

    def test_path_strength_transitive_dominance(self):
        """Candidate 0 beats 1 and 2; 1 beats 2. Path strengths reflect dominance."""
        n = 3
        P = np.zeros((n, n))
        P[0, 1] = 2; P[1, 0] = 1
        P[1, 2] = 2; P[2, 1] = 1
        P[0, 2] = 3; P[2, 0] = 0

        P_r = _schulze_path_strength(P.copy())

        np.testing.assert_allclose(
            P_r,
            [[1.0, 2.0, 3.0],
             [1.0, 1.0, 2.0],
             [1.0, 1.0, 1.0]],
            err_msg="Path strength regression values changed",
        )


class TestSchulzeRankStrength:
    """Margin-weighted win accumulation."""

    def test_clear_winner_accumulates_positive_margin(self):
        n = 3
        P = np.zeros((n, n))
        P[0, 1] = 10; P[1, 0] = 5
        P[0, 2] = 10; P[2, 0] = 5
        P[1, 2] = 6;  P[2, 1] = 6

        wins = _schulze_rank_strength(P)

        assert wins[0] == pytest.approx(10)  # (10-5) + (10-5)
        assert wins[1] == pytest.approx(0)
        assert wins[2] == pytest.approx(0)


class TestSchulzeVoting:
    """Full Schulze voting pipeline."""

    def _make(self, scores_list: list[dict]):
        """Build (score_matrix, voter_names) from a list of score dicts."""
        proxies = [_proxy(s) for s in scores_list]
        sm, vn, _ = build_rank_matrix(proxies)
        return sm, vn

    def test_clear_majority_winner(self):
        """Candidate preferred by most voters wins."""
        sm, vn = self._make([
            {'v1': 2.0, 'v2': 2.0, 'v3': 1.0},   # candidate 0
            {'v1': 1.0, 'v2': 1.0, 'v3': 2.0},   # candidate 1
            {'v1': 0.0, 'v2': 0.0, 'v3': 0.0},   # candidate 2 -- lowest score = best
            {'v1': 3.0, 'v2': 3.0, 'v3': 3.0},   # candidate 3
        ])
        assert schulze_voting(sm, vn)[0] == 2

    def test_condorcet_winner(self):
        """Candidate that beats all others head-to-head wins."""
        sm, vn = self._make([
            {'v1': 1.0, 'v2': 1.0, 'v3': 2.0},  # candidate 0
            {'v1': 0.0, 'v2': 2.0, 'v3': 1.0},  # candidate 1
            {'v1': 2.0, 'v2': 0.0, 'v3': 0.0},  # candidate 2 -- condorcet winner
        ])
        assert schulze_voting(sm, vn)[0] == 2

    @pytest.mark.parametrize("weights,expected_winner", [
        ({'v1': 10.0, 'v2': 1.0, 'v3': 1.0}, 0),
        ({'v1': 1e10, 'v2': 1.0, 'v3': 1.0}, 0),
    ])
    def test_weighted_voting(self, weights, expected_winner):
        """Voter with much higher weight dominates."""
        sm, vn = self._make([
            {'v1': 0.0, 'v2': 1.0, 'v3': 2.0},
            {'v1': 1.0, 'v2': 0.0, 'v3': 1.0},
            {'v1': 2.0, 'v2': 2.0, 'v3': 0.0},
        ])
        assert schulze_voting(sm, vn, voter_weights=weights)[0] == expected_winner

    def test_single_candidate(self):
        sm, vn = self._make([{'v1': 0.0, 'v2': 0.0}])
        winner, conf = schulze_voting(sm, vn)
        assert winner == 0
        assert conf == pytest.approx(1.0)

    def test_return_preference_df(self):
        sm, vn = self._make([
            {'v1': 0.0, 'v2': 1.0},
            {'v1': 1.0, 'v2': 0.0},
            {'v1': 2.0, 'v2': 2.0},
        ])
        winner, conf, pref_df = schulze_voting(sm, vn, return_preference_df=True)
        assert isinstance(pref_df, pd.DataFrame)
        assert 'wins' in pref_df.columns
        assert winner == int(np.argmax(pref_df['wins'].to_numpy()))
        assert isinstance(conf, float)
        assert conf == pytest.approx(0.0)

    def test_window_smoothing_changes_result(self):
        """Gaussian smoothing over candidate scores can shift the winner."""
        sm, vn = self._make([
            {'v1': 2.0, 'v2': 1.0, 'v3': 2.0},
            {'v1': 1.0, 'v2': 0.0, 'v3': 1.0},
            {'v1': 0.0, 'v2': 2.0, 'v3': 0.0},
            {'v1': 3.0, 'v2': 3.0, 'v3': 3.0},
            {'v1': 4.0, 'v2': 4.0, 'v3': 4.0},
        ])
        w_no, c_no         = schulze_voting(sm, vn, window_size=0)
        w_smooth, c_smooth  = schulze_voting(sm, vn, window_size=2)
        assert w_no == 2
        assert c_no == pytest.approx(0.02564, abs=1e-3)
        assert w_smooth == 0
        assert c_smooth == pytest.approx(0.0, abs=0.03)

    def test_none_voter_weights_equals_equal_weights(self):
        sm, vn = self._make([
            {'v1': 0.0, 'v2': 1.0, 'v3': 2.0},
            {'v1': 1.0, 'v2': 0.0, 'v3': 1.0},
            {'v1': 2.0, 'v2': 2.0, 'v3': 0.0},
        ])
        w_none, _  = schulze_voting(sm, vn, voter_weights=None)
        w_equal, _ = schulze_voting(sm, vn, voter_weights={'v1': 1.0, 'v2': 1.0, 'v3': 1.0})
        assert w_none == w_equal

    def test_weight_scale_invariant(self):
        """Scaling all weights by a constant doesn't change the winner."""
        sm, vn = self._make([
            {'v1': 0.0, 'v2': 1.0},
            {'v1': 1.0, 'v2': 0.0},
            {'v1': 2.0, 'v2': 2.0},
        ])
        w1, _ = schulze_voting(sm, vn, voter_weights={'v1': 0.3, 'v2': 0.7})
        w2, _ = schulze_voting(sm, vn, voter_weights={'v1': 300.0, 'v2': 700.0})
        assert w1 == w2

    def test_cyclic_preferences_resolves(self):
        """Condorcet paradox still produces a winner with zero confidence."""
        sm, vn = self._make([
            {'v1': 0.0, 'v2': 1.0, 'v3': 2.0},
            {'v1': 1.0, 'v2': 2.0, 'v3': 0.0},
            {'v1': 2.0, 'v2': 0.0, 'v3': 1.0},
        ])
        winner, conf = schulze_voting(sm, vn)
        assert winner == 0
        assert conf == pytest.approx(0.0)

    def test_missing_voter_in_weights_defaults_to_zero(self):
        """Voter absent from weights dict gets weight 0 (silenced)."""
        sm, vn = self._make([
            {'v1': 0.0, 'v2': 1.0, 'v3': 2.0},
            {'v1': 1.0, 'v2': 0.0, 'v3': 1.0},
            {'v1': 2.0, 'v2': 2.0, 'v3': 0.0},
        ])
        w_partial, conf = schulze_voting(sm, vn, voter_weights={'v1': 2.0, 'v2': 1.0})
        assert w_partial == 0
        assert conf == pytest.approx(0.2, abs=1e-6)

    def test_many_candidates(self):
        n = 50
        rng = np.random.default_rng(0)
        scores_list = [{'v1': float(i), 'v2': float(n - 1 - i), 'v3': float(rng.integers(n))}
                       for i in range(n)]
        sm, vn = self._make(scores_list)
        winner, conf = schulze_voting(sm, vn)
        assert winner == 23
        assert conf == pytest.approx(0.02658, abs=1e-3)

    def test_empty_raises(self):
        with pytest.raises((IndexError, ValueError, KeyError)):
            schulze_voting(np.empty((0, 0)), [])


class TestScoreMagnitudeNormalization:
    """Tests for MAD-clipped min-max normalization in _build_rank_arrays."""

    def test_normalization_regression_values(self):
        """Regression: MAD-clipped normalization matches known output for various inputs."""
        sm = np.array([[0.1], [0.3], [0.5], [100.0]], dtype=np.float64)
        _, _, _, _, norm = _build_rank_arrays(sm, ["v1"])
        np.testing.assert_allclose(
            norm[:, 0], [0.25, 0.4167, 0.5833, 1.0], atol=1e-3,
            err_msg="MAD-clipped normalization regression values changed",
        )
        # Also verify outlier doesn't compress the range
        gap_01 = norm[1, 0] - norm[0, 0]
        assert gap_01 > 0.05, f"Gap {gap_01} too small -- outlier compressed the range"

    def test_multi_voter_normalization(self):
        """Multi-voter normalization produces expected values in [0, 1]."""
        sm = np.array([[0.1, 0.5], [0.3, 0.2], [0.9, 0.8]], dtype=np.float64)
        _, _, _, _, norm = _build_rank_arrays(sm, ["v1", "v2"])
        np.testing.assert_allclose(
            norm,
            [[1 / 3, 0.5], [0.5, 1 / 3], [1.0, 2 / 3]],
            atol=1e-4,
            err_msg="Multi-voter normalization regression values changed",
        )

    @pytest.mark.parametrize("value,expected", [
        (np.inf, 1.0),
        (-np.inf, 0.0),
    ])
    def test_inf_maps_to_boundary(self, value, expected):
        """Positive inf maps to 1.0 (worst); negative inf maps to 0.0 (best)."""
        sm = np.array([[value], [0.1], [0.5]], dtype=np.float64)
        _, _, _, _, norm = _build_rank_arrays(sm, ["v1"])
        assert norm[0, 0] == pytest.approx(expected)

    def test_identical_scores_normalize_to_same_value(self):
        """All-equal finite scores produce identical normalized values (0.5)."""
        sm = np.array([[5.0], [5.0], [5.0]], dtype=np.float64)
        _, _, _, _, norm = _build_rank_arrays(sm, ["v1"])
        np.testing.assert_allclose(norm[:, 0], [0.5, 0.5, 0.5])

    def test_score_magnitude_affects_pairwise_preference(self):
        """Large score gaps produce stronger pairwise preferences than small gaps."""
        sm = np.array([[0.0, 0.4], [0.9, 0.5]], dtype=np.float64)
        rank, weights, _, abstain, norm = _build_rank_arrays(sm, ["v1", "v2"])
        P = _schulze_pairwise_preference(rank, weights, abstain, normalized_scores=norm)
        assert P[0, 1] > P[1, 0], "Score-magnitude voting should favor the large-gap voter"

    def test_two_candidates_degrades_gracefully(self):
        """With only 2 candidates, normalization still produces valid output."""
        sm = np.array([[0.1], [0.5]], dtype=np.float64)
        _, _, _, _, norm = _build_rank_arrays(sm, ["v1"])
        assert norm.shape == (2, 1)
        np.testing.assert_allclose(norm[:, 0], [1 / 3, 2 / 3], atol=1e-4)

    def test_single_finite_with_infs(self):
        """One finite value among infs normalizes to 0.5 (middle)."""
        sm = np.array([[-np.inf], [0.5], [np.inf]], dtype=np.float64)
        _, _, _, _, norm = _build_rank_arrays(sm, ["v1"])
        np.testing.assert_allclose(norm[:, 0], [0.0, 0.5, 1.0])

    def test_all_nan_voter_is_dropped(self):
        """A voter where all candidates are NaN is filtered out."""
        sm = np.array([[0.1, np.nan], [0.5, np.nan]], dtype=np.float64)
        rank, _, kept_names, _, _ = _build_rank_arrays(sm, ["v1", "v2"])
        assert "v2" not in kept_names, "All-NaN voter should be dropped"
        assert rank.shape[1] == 1, "Only v1 should remain"


class TestSchulzeConfidence:
    """Tests for selection_confidence on schulze_voting return."""

    def _make(self, scores_list):
        proxies = [types.SimpleNamespace(score=s) for s in scores_list]
        return build_rank_matrix(proxies)

    def test_unanimous_agreement_gives_high_confidence(self):
        """All voters agree -> confidence = 0.5 (score-magnitude pairwise)."""
        sm, vn, _ = self._make([
            {"v1": 0.0, "v2": 0.0}, {"v1": 1.0, "v2": 1.0}, {"v1": 2.0, "v2": 2.0},
        ])
        _, conf = schulze_voting(sm, vn)
        assert conf == pytest.approx(0.5, abs=0.01)

    def test_perfect_cycle_gives_zero_confidence(self):
        """Condorcet cycle -> confidence = 0.0."""
        sm, vn, _ = self._make([
            {"v1": 0.0, "v2": 1.0, "v3": 2.0},
            {"v1": 1.0, "v2": 2.0, "v3": 0.0},
            {"v1": 2.0, "v2": 0.0, "v3": 1.0},
        ])
        _, conf = schulze_voting(sm, vn)
        assert conf == pytest.approx(0.0)


class TestKPenalty:
    """The low-k complexity penalty shifts ties toward higher k."""

    def test_penalty_flips_tie_toward_higher_k(self):
        """With equal scores, a positive penalty makes the higher-k candidate win."""
        score_matrix = np.array([[0.5], [0.5]])
        voter_names = ["v"]

        no_penalty, _ = schulze_voting(
            score_matrix, voter_names, candidate_k_values=[2, 5], k_penalty_strength=0.0,
        )
        with_penalty, _ = schulze_voting(
            score_matrix, voter_names, candidate_k_values=[2, 5], k_penalty_strength=3.0,
        )
        assert no_penalty == 0
        assert with_penalty == 1

    def test_no_penalty_when_strength_zero(self):
        """Strength 0 leaves the winner unchanged from the unpenalised vote."""
        score_matrix = np.array([[0.2], [0.8]])
        voter_names = ["v"]
        winner, _ = schulze_voting(
            score_matrix, voter_names, candidate_k_values=[2, 5], k_penalty_strength=0.0,
        )
        assert winner == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
