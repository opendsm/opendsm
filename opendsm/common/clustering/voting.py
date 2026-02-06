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

from scipy.ndimage import gaussian_filter1d


def construct_voting_df(results):
    """
    Construct a voting DataFrame from the results and score council.
    """
    # Create a dataframe of all score algorithms and their scores for each number of clusters
    score_dict = {}
    for n, label_res in enumerate(results):
        # n_clusters = label_res.n_clusters
        score_dict[n] = label_res.score

    res_df = pd.DataFrame.from_dict(score_dict, orient='index')

    # Drop columns where all values are np.inf
    res_df = res_df.loc[:, ~(res_df == np.inf).all()]

    # replace non-finite values with inf
    res_df = res_df.replace([np.nan, -np.inf], np.inf)

    # convert from index value to order of cluster number
    res_df = res_df.apply(lambda col: col.sort_values().index.to_numpy(), axis=0)

    # reset index and delete old index
    res_df = res_df.reset_index(drop=True)
    
    return res_df


def _shulze_pairwise_preference(df, voter_weights=None):
    """
    Perform pairwise comparison to select the best candidate (row) from a DataFrame.
    Each column is a 'voter' (score algorithm), and each row index is a candidate (n_clusters).
    Each column contains a ranking of candidates (row indices), with lower values being better.
    
    """

    if voter_weights is None:
        voter_weights = {voter: 1.0 for voter in df.columns}

    candidates = np.unique(df.iloc[:, 0])

    Pd = np.zeros((len(candidates), len(candidates), 2))
    pred = np.zeros((len(candidates), len(candidates)))
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if a == b:
                continue

            votes = {
                "a": 0,
                "b": 0,
            }
            for voter in df.columns:
                rank = {
                    "a": pd.Index(df[voter]).get_loc(a),
                    "b": pd.Index(df[voter]).get_loc(b),
                }

                if rank["a"] < rank["b"]:
                    votes["a"] += 1.0*voter_weights[voter]
                elif rank["a"] > rank["b"]:
                    votes["b"] += 1.0*voter_weights[voter]
                else:
                    votes["a"] += 0.5*voter_weights[voter]
                    votes["b"] += 0.5*voter_weights[voter]
            
            Pd[i, j] = [votes["a"], votes["b"]]
            pred[i, j] = i
    
    return Pd, pred


def _shulze_path_strength(Pd, pred):
    """
    Compute the path strength for each candidate.
    """
    n_candidates = Pd.shape[0]

    for i in range(n_candidates):
        for j in range(n_candidates):
            if i == j:
                continue

            for k in range(n_candidates):
                if k == i or k == j:
                    continue

                current_strength = Pd[j, k][0]
                idx_alt = np.argmin([Pd[j, i][0], Pd[i, k][0]]).flatten()[0]
                if idx_alt == 0:
                    idx_alt = [j, i]
                    potential_strength = Pd[j, i][0]
                else:
                    idx_alt = [i, k]
                    potential_strength = Pd[i, k][0]

                if current_strength < potential_strength:
                    Pd[j, k] = Pd[*idx_alt, :]

                    if pred[j, k] != pred[i, k]:
                        pred[j, k] = pred[i, k]

    return Pd, pred


def _shulze_rank_strength(Pd, pred):
    """
    Compute the rank strength for each candidate.
    """
    n_candidates = Pd.shape[0]
    candidate_wins = np.zeros(n_candidates)

    for i in range(n_candidates):
        for j in range(n_candidates):
            if i == j:
                continue

            if Pd[i, j][0] > Pd[j, i][0]:
                candidate_wins[i] += 1
    
    return candidate_wins


def shulze_voting(df, voter_weights=None, window_size=0, return_preference_df=False):
    """
    Perform Shulze voting to select the best candidate (row) from a DataFrame.
    Each column is a 'voter' (score algorithm), and each row index is a candidate (n_clusters).
    Each column contains a ranking of candidates (row indices), with lower values being better.
    
    Based on: A New Monotonic, Clone-Independent, Reversal Symmetric, and Condorcet-Consistent 
              Single-Winner Election Method by Markus Schulze
              http://www.9mail.de/m-schulze/schulze1.pdf
    """

    if voter_weights is None:
        voter_weights = {voter: 1.0 for voter in df.columns}

    # Normalize voter_weights to sum to the total number of voters
    n_voters = len(df.columns)
    total_weight = sum(voter_weights.values())
    if total_weight != n_voters and total_weight > 0:
        scale = n_voters / total_weight
        voter_weights = {k: v * scale for k, v in voter_weights.items()}

    candidates = df.index.to_numpy()

    Pd, pred = _shulze_pairwise_preference(df, voter_weights=voter_weights)
    Pd, pred = _shulze_path_strength(Pd, pred)
    candidate_wins = _shulze_rank_strength(Pd, pred)

    df_wins = pd.DataFrame({
        "candidate": candidates,
        "wins": candidate_wins
    })
    if window_size > 0:
        df_wins["wins"] = gaussian_filter1d(
            df_wins["wins"], 
            sigma=window_size,
            mode="nearest", # constant or nearest?
            cval=0.0 # for constant mode
        )

    # this should select the smallest candidate if there is a tie
    # there is a procedure for this in the paper if we want to improve this later
    winner_idx = int(np.argmax(df_wins["wins"]))

    if not return_preference_df:
        return winner_idx

    # Change each voter column in df to preference starting at zero
    df_pref = df.stack().reset_index()
    df_pref.columns = ["preference", "score_algo", "n_clusters"]
    df_pref = df_pref.pivot(index="n_clusters", columns="score_algo", values="preference")

    # invert preferences so that higher is better
    # df_pref = np.max(df_pref) - df_pref

    # Join df_pref and df_wins on the index (n_clusters/candidate)
    df_pref = df_pref.merge(df_wins, how="left", left_index=True, right_on="candidate")
    df_pref = df_pref.set_index("candidate")

    return winner_idx, df_pref


def stv_voting(df, voter_weights=None, window_size=0):
    """
    Perform cluster voting to select the best candidate (row) from a DataFrame.
    Each column is a 'voter' (score algorithm), and each row index is a candidate (n_clusters).
    Each column contains a ranking of candidates (row indices), with lower values being better.

    Problems: 
      If a candidate is second best for every voter, it will be eliminated.
      Indeterminate if all voters select different candidates.
      Voter weight not implemented.
    
    """

    total_voters = len(df.columns)
    vote_threshold = total_voters / 2
    candidates = set(range(np.min(df), np.max(df) + 1))
    eliminated = set()

    voter_power = 1.0

    has_winner = False
    for _ in range(len(candidates)):
        vote_counts = {candidate: 0 for candidate in candidates}
        for voter in df.columns:
            for preference in df[voter]:
                if preference not in eliminated:
                    vote_counts[preference] += 1.0

                    if window_size > 0:
                        for window in range(1, window_size + 1):
                            vote_power = window_size / 2
                            if preference - window in candidates:
                                vote_counts[preference - window] += 0.5

                            if preference + window in candidates:
                                vote_counts[preference + window] += 0.5

                    break

        vote_counts = {candidate: count * voter_power for candidate, count in vote_counts.items()}

        # if any candidate in vote_counts has greater or equal to vote_threshold, break
        for candidate, count in vote_counts.items():
            if count >= vote_threshold:
                has_winner = True
                break

        # print(vote_counts)
        if has_winner:
            break

        # if all vote counts are the same, break
        if len(set(vote_counts.values())) == 1:
            break
            # what happens if voters all select different candidates?

        min_votes = min(vote_counts.values())
        to_eliminate = {cand for cand, count in vote_counts.items() if count == min_votes}

        candidates -= to_eliminate
        eliminated.update(to_eliminate)

    if not has_winner:
        # select candidate with most votes from vote_counts
        max_votes = max(vote_counts.values())

        top_candidates = [cand for cand, count in vote_counts.items() if count == max_votes]
        candidate = min(top_candidates)

    return candidate