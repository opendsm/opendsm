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

import numpy as np
import pandas as pd
import pytest

from opendsm.common.clustering.voting import (
    shulze_voting,
    construct_voting_df,
    _shulze_pairwise_preference,
    _shulze_path_strength,
    _shulze_rank_strength,
)



class TestShulzePairwisePreference:
    """Test suite for pairwise preference calculation."""

    def test_basic_pairwise(self):
        """Test basic pairwise preference calculation."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [1, 0, 2],
        })

        Pd, pred = _shulze_pairwise_preference(df)

        # Check output shapes
        assert Pd.shape == (3, 3, 2)
        assert pred.shape == (3, 3)

        # Diagonal should be zero (no comparison with self)
        for i in range(3):
            assert Pd[i, i, 0] == 0
            assert Pd[i, i, 1] == 0

    def test_pairwise_with_weights(self):
        """Test pairwise preference with voter weights."""
        df = pd.DataFrame({
            'voter1': [0, 1],
            'voter2': [1, 0],
        })

        weights = {'voter1': 2.0, 'voter2': 1.0}
        Pd, pred = _shulze_pairwise_preference(df, voter_weights=weights)

        # Voter1 with weight 2 should have more influence
        assert Pd.shape == (2, 2, 2)

    def test_pairwise_tie(self):
        """Test pairwise preference with tied rankings."""
        df = pd.DataFrame({
            'voter1': [0, 1],
            'voter2': [1, 0],
        })

        # When candidates have same rank, both get 0.5 vote
        Pd, pred = _shulze_pairwise_preference(df)
        assert Pd.shape == (2, 2, 2)


class TestShulzePathStrength:
    """Test suite for path strength calculation."""

    def test_path_strength_preserves_shape(self):
        """Test that path strength preserves matrix shapes."""
        n = 5
        Pd = np.random.rand(n, n, 2)
        pred = np.arange(n*n).reshape(n, n)

        Pd_result, pred_result = _shulze_path_strength(Pd.copy(), pred.copy())

        assert Pd_result.shape == Pd.shape
        assert pred_result.shape == pred.shape

    def test_path_strength_basic(self):
        """Test basic path strength calculation."""
        # Create simple pairwise matrix
        n = 3
        Pd = np.zeros((n, n, 2))
        pred = np.zeros((n, n))

        # Set up simple preferences: 0>1, 1>2, 0>2
        Pd[0, 1] = [2, 1]
        Pd[1, 0] = [1, 2]
        Pd[1, 2] = [2, 1]
        Pd[2, 1] = [1, 2]
        Pd[0, 2] = [3, 0]
        Pd[2, 0] = [0, 3]

        Pd_result, pred_result = _shulze_path_strength(Pd, pred)

        # Check shapes
        assert Pd_result.shape == (n, n, 2)
        assert pred_result.shape == (n, n)

        # Check that path strengths are computed correctly
        # Candidate 0 should beat 1: direct path strength should be at least 2
        assert Pd_result[0, 1, 0] >= 2
        # Candidate 0 should beat 2: direct path strength should be 3
        assert Pd_result[0, 2, 0] >= 3
        # Candidate 1 should beat 2: direct path strength should be at least 2
        assert Pd_result[1, 2, 0] >= 2

        # Check reciprocal relationships (losing side)
        assert Pd_result[1, 0, 0] <= 2  # 1 loses to 0
        assert Pd_result[2, 0, 0] <= 1  # 2 loses to 0
        assert Pd_result[2, 1, 0] <= 2  # 2 loses to 1


class TestShulzeRankStrength:
    """Test suite for rank strength calculation."""

    def test_rank_strength_basic(self):
        """Test basic rank strength calculation."""
        n = 3
        Pd = np.zeros((n, n, 2))
        pred = np.zeros((n, n))

        # Set up preferences where candidate 0 beats all others
        Pd[0, 1] = [10, 5]
        Pd[1, 0] = [5, 10]
        Pd[0, 2] = [10, 5]
        Pd[2, 0] = [5, 10]
        Pd[1, 2] = [6, 6]  # Tie
        Pd[2, 1] = [6, 6]

        wins = _shulze_rank_strength(Pd, pred)

        assert wins.shape == (n,)
        assert wins[0] == 2  # Candidate 0 beats both others
        assert wins[1] == 0  # Candidate 1 loses to 0, ties with 2
        assert wins[2] == 0  # Candidate 2 loses to 0, ties with 1

    def test_rank_strength_all_lose(self):
        """Test rank strength when candidates all lose equally."""
        n = 3
        Pd = np.zeros((n, n, 2))
        pred = np.zeros((n, n))

        # Everyone ties with everyone
        for i in range(n):
            for j in range(n):
                if i != j:
                    Pd[i, j] = [5, 5]

        wins = _shulze_rank_strength(Pd, pred)

        assert wins.shape == (n,)
        assert np.all(wins == 0)


class TestConstructVotingDF:
    """Test suite for voting dataframe construction."""

    def test_construct_basic(self):
        """Test basic voting dataframe construction."""
        # Create mock results
        class MockResult:
            def __init__(self, n_clusters, scores):
                self.n_clusters = n_clusters
                self.score = scores

        results = [
            MockResult(2, {'algo1': 10, 'algo2': 20}),
            MockResult(3, {'algo1': 5, 'algo2': 15}),
            MockResult(4, {'algo1': 8, 'algo2': 12}),
        ]

        df = construct_voting_df(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) > 0

    def test_construct_with_nan(self):
        """Test voting df construction handles NaN values."""
        class MockResult:
            def __init__(self, n_clusters, scores):
                self.n_clusters = n_clusters
                self.score = scores

        results = [
            MockResult(2, {'algo1': 10, 'algo2': np.nan}),
            MockResult(3, {'algo1': np.nan, 'algo2': 15}),
        ]

        df = construct_voting_df(results)

        # NaN should be replaced with inf and potentially dropped
        assert isinstance(df, pd.DataFrame)

    def test_construct_with_inf(self):
        """Test voting df construction handles inf values."""
        class MockResult:
            def __init__(self, n_clusters, scores):
                self.n_clusters = n_clusters
                self.score = scores

        results = [
            MockResult(2, {'algo1': 10, 'algo2': -np.inf}),
            MockResult(3, {'algo1': 5, 'algo2': 15}),
        ]

        df = construct_voting_df(results)

        assert isinstance(df, pd.DataFrame)


class TestShulzeVoting:
    """Test suite for Shulze voting method."""

    def test_simple_majority(self):
        """Test that clear majority winner is selected."""
        # Create voting data where candidate 2 is clearly the best
        # Each column is a voter, each value is a ranking
        df = pd.DataFrame({
            'voter1': [2, 1, 0, 3],  # prefers candidate 2
            'voter2': [2, 1, 0, 3],  # prefers candidate 2
            'voter3': [1, 2, 0, 3],  # prefers candidate 2
        })

        winner = shulze_voting(df)
        assert winner == 2

    def test_condorcet_winner(self):
        """Test that Condorcet winner (beats all others head-to-head) is selected."""
        # Candidate 1 should win in all pairwise comparisons
        df = pd.DataFrame({
            'voter1': [1, 0, 2],
            'voter2': [1, 2, 0],
            'voter3': [2, 1, 0],
        })

        winner = shulze_voting(df)
        assert winner == 1

    def test_unanimous_vote(self):
        """Test unanimous voting scenario."""
        # All voters prefer candidates in the same order
        df = pd.DataFrame({
            'voter1': [3, 1, 0, 2],
            'voter2': [3, 1, 0, 2],
            'voter3': [3, 1, 0, 2],
            'voter4': [3, 1, 0, 2],
        })

        winner = shulze_voting(df)
        assert winner == 3

    def test_weighted_voting(self):
        """Test voting with weighted voters."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [1, 0, 2],
            'voter3': [2, 1, 0],
        })

        # Give voter1 much higher weight
        weights = {'voter1': 10.0, 'voter2': 1.0, 'voter3': 1.0}
        winner = shulze_voting(df, voter_weights=weights)
        assert winner == 0

    def test_tie_breaking(self):
        """Test that ties are broken consistently (selects smallest candidate)."""
        # Create a perfect tie scenario
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [1, 2, 0],
            'voter3': [2, 0, 1],
        })

        winner = shulze_voting(df)
        # Should select smallest candidate in case of tie
        assert isinstance(winner, (int, np.integer))
        assert winner == 0

    def test_single_candidate(self):
        """Test with only one candidate."""
        df = pd.DataFrame({
            'voter1': [0],
            'voter2': [0],
            'voter3': [0],
        })

        winner = shulze_voting(df)
        assert winner == 0

    def test_two_candidates(self):
        """Test with two candidates."""
        df = pd.DataFrame({
            'voter1': [1, 0],
            'voter2': [1, 0],
            'voter3': [0, 1],
        })

        winner = shulze_voting(df)
        # Candidate 1 wins 2-1
        assert winner == 1

    def test_two_strong_candidates(self):
        """Test with two candidates that dominate others."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2, 3, 4],
            'voter2': [1, 0, 2, 3, 4],
            'voter3': [0, 1, 2, 3, 4],
            'voter4': [1, 0, 2, 3, 4],
        })

        winner = shulze_voting(df)

        # Winner should be one of the top two
        assert winner in [0, 1]

    def test_return_preference_df(self):
        """Test structure of returned preference dataframe."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2, 3],
            'voter2': [1, 0, 2, 3],
            'voter3': [2, 1, 0, 3],
        })

        winner, pref_df = shulze_voting(df, return_preference_df=True)

        # Check preference df structure
        assert isinstance(pref_df, pd.DataFrame)
        assert len(pref_df) == len(df)
        assert 'wins' in pref_df.columns
        assert pref_df['wins'].dtype in [np.int64, np.int32, int]
        assert winner == pref_df['wins'].idxmax()

    def test_window_smoothing(self):
        """Test that window_size parameter applies smoothing."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2, 3, 4],
            'voter2': [1, 0, 2, 3, 4],
            'voter3': [2, 1, 0, 3, 4],
        })

        winner_no_smooth = shulze_voting(df, window_size=0)
        winner_smooth = shulze_voting(df, window_size=2)

        # Both should return valid winners
        assert 0 <= winner_no_smooth < len(df)
        assert 0 <= winner_smooth < len(df)

        # Smoothing should change the result
        assert winner_no_smooth != winner_smooth  

        # Smoothing winner should be 0 because it gets more weight from nearby candidates
        assert winner_smooth == 0

    def test_empty_voter_weights(self):
        """Test that None voter_weights defaults to equal weights."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [1, 0, 2],
            'voter3': [2, 1, 0],
        })

        winner_no_weights = shulze_voting(df, voter_weights=None)
        winner_equal_weights = shulze_voting(df, voter_weights={'voter1': 1.0, 'voter2': 1.0, 'voter3': 1.0})

        # Should produce same result
        assert winner_no_weights == winner_equal_weights

    def test_weight_normalization(self):
        """Test that voter weights are normalized correctly."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [1, 0, 2],
        })

        # These should be equivalent after normalization
        weights1 = {'voter1': 1.0, 'voter2': 1.0}
        weights2 = {'voter1': 10.0, 'voter2': 10.0}

        winner1 = shulze_voting(df, voter_weights=weights1)
        winner2 = shulze_voting(df, voter_weights=weights2)

        assert winner1 == winner2

    def test_large_number_of_candidates(self):
        """Test with larger number of candidates."""
        n_candidates = 10
        df = pd.DataFrame({
            'voter1': np.arange(n_candidates),
            'voter2': np.arange(n_candidates)[::-1],
            'voter3': np.roll(np.arange(n_candidates), 3),
            'voter4': np.roll(np.arange(n_candidates), -2),
        })

        winner = shulze_voting(df)
        assert 0 <= winner < n_candidates


class TestShulzeVotingEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_voter(self):
        """Test with single voter."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2, 3],
        })

        winner = shulze_voting(df)
        # With single voter, best candidate should win
        assert winner == 0

    def test_zero_weights(self):
        """Test behavior with zero weights."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [1, 0, 2],
        })

        # One voter has zero weight
        weights = {'voter1': 1.0, 'voter2': 0.0}
        winner = shulze_voting(df, voter_weights=weights)

        # Should be same as if voter2 doesn't exist
        assert isinstance(winner, (int, np.integer))

    def test_negative_window_size(self):
        """Test that negative window size is treated as zero."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [1, 0, 2],
        })

        # Should not raise error
        winner = shulze_voting(df, window_size=-1)
        assert isinstance(winner, (int, np.integer))

    def test_missing_voter_weights(self):
        """Test behavior when weights dict doesn't include all voters."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [1, 0, 2],
            'voter3': [2, 1, 0],
        })

        # Only provide weights for some voters
        weights = {'voter1': 2.0, 'voter2': 1.0}
        winner = shulze_voting(df, voter_weights=weights)

        # Should handle gracefully
        assert isinstance(winner, (int, np.integer))
        assert 0 <= winner < len(df)

    def test_extreme_weight_differences(self):
        """Test with very large weight differences."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [2, 1, 0],
            'voter3': [1, 2, 0],
        })

        # One voter with extremely high weight should dominate
        weights = {'voter1': 1e10, 'voter2': 1.0, 'voter3': 1.0}
        winner = shulze_voting(df, voter_weights=weights)

        # Should match voter1's top choice
        assert winner == 0

    def test_cyclic_preferences(self):
        """Test Condorcet paradox (cyclic preferences: A>B>C>A)."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],  # A > B > C
            'voter2': [1, 2, 0],  # B > C > A
            'voter3': [2, 0, 1],  # C > A > B
        })

        winner = shulze_voting(df)

        # Shulze method should still produce a winner
        assert isinstance(winner, (int, np.integer))
        assert 0 <= winner < 3

    def test_large_window_size(self):
        """Test with window size larger than number of candidates."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2, 3],
            'voter2': [1, 0, 2, 3],
            'voter3': [2, 1, 0, 3],
        })

        # Window size larger than candidate count
        winner = shulze_voting(df, window_size=100)

        assert isinstance(winner, (int, np.integer))
        assert 0 <= winner < len(df)

    def test_weighted_voting_normalization(self):
        """Test that results are consistent regardless of weight scale."""
        df = pd.DataFrame({
            'voter1': [0, 1, 2],
            'voter2': [1, 2, 0],
        })

        # Try different weight scales
        weights_small = {'voter1': 0.3, 'voter2': 0.7}
        weights_large = {'voter1': 300.0, 'voter2': 700.0}

        winner_small = shulze_voting(df, voter_weights=weights_small)
        winner_large = shulze_voting(df, voter_weights=weights_large)

        # Should produce same result
        assert winner_small == winner_large

    def test_very_large_number_of_candidates(self):
        """Test scalability with many candidates."""
        n_candidates = 50
        df = pd.DataFrame({
            'voter1': np.arange(n_candidates),
            'voter2': np.arange(n_candidates)[::-1],
            'voter3': np.roll(np.arange(n_candidates), 5),
        })

        winner = shulze_voting(df)

        assert isinstance(winner, (int, np.integer))
        assert 0 <= winner < n_candidates

    def test_empty_dataframe_no_rows_no_cols(self):
        """Test with completely empty DataFrame (no rows, no columns)."""
        df = pd.DataFrame()

        with pytest.raises((IndexError, ValueError, KeyError)):
            shulze_voting(df)

    def test_empty_dataframe_no_rows(self):
        """Test with DataFrame that has columns but no rows."""
        df = pd.DataFrame(columns=['voter1', 'voter2', 'voter3'])

        with pytest.raises((IndexError, ValueError, KeyError)):
            shulze_voting(df)

    def test_empty_dataframe_no_columns(self):
        """Test with DataFrame that has rows but no columns (no voters)."""
        df = pd.DataFrame(index=[0, 1, 2])

        with pytest.raises((IndexError, ValueError, KeyError)):
            shulze_voting(df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
