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

    # replace non-finite values with inf
    res_df = res_df.replace([np.nan, -np.inf], np.inf)

    # Drop columns where all values are np.inf
    res_df = res_df.loc[:, ~(res_df == np.inf).all()]

    # convert from index value to order of cluster number
    res_df = res_df.apply(lambda col: col.sort_values().index.to_numpy(), axis=0)

    # reset index and delete old index
    res_df = res_df.reset_index(drop=True)

    # If res_df is a Series, convert it to a DataFrame
    if isinstance(res_df, pd.Series):
        res_df = res_df.to_frame()
    
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

    # Pre-build rank lookup per voter to avoid repeated pd.Index construction
    voter_ranks = {}
    for voter in df.columns:
        idx = pd.Index(df[voter])
        voter_ranks[voter] = {candidate: idx.get_loc(candidate) for candidate in candidates}

    Pd = np.zeros((len(candidates), len(candidates), 2))
    pred = np.zeros((len(candidates), len(candidates)))
    for i, a in enumerate(candidates):
        for j, b in enumerate(candidates):
            if a == b:
                continue

            votes_a = 0.0
            votes_b = 0.0
            for voter in df.columns:
                w = voter_weights[voter]
                rank_a = voter_ranks[voter][a]
                rank_b = voter_ranks[voter][b]

                if rank_a < rank_b:
                    votes_a += w
                elif rank_a > rank_b:
                    votes_b += w
                else:
                    votes_a += 0.5 * w
                    votes_b += 0.5 * w

            Pd[i, j] = [votes_a, votes_b]
            pred[i, j] = i

    return Pd, pred


def _shulze_path_strength(Pd, pred):
    """
    Compute strongest path strengths using Floyd-Warshall.
    Updates Pd so that Pd[j, k][0] holds the strength of the
    strongest path from candidate j to candidate k.
    """
    n_candidates = Pd.shape[0]

    for i in range(n_candidates):
        for j in range(n_candidates):
            if i == j:
                continue

            for k in range(n_candidates):
                if k == i or k == j:
                    continue

                # Strength of path j→i→k is the bottleneck (min) of two edges
                strength_ji = Pd[j, i][0]
                strength_ik = Pd[i, k][0]

                if strength_ji <= strength_ik:
                    bottleneck = (j, i)
                    potential_strength = strength_ji
                else:
                    bottleneck = (i, k)
                    potential_strength = strength_ik

                if Pd[j, k][0] < potential_strength:
                    Pd[j, k] = Pd[bottleneck[0], bottleneck[1], :]
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

    if df.shape[0] == 0:
        raise ValueError("Input DataFrame has no rows.")

    if voter_weights is None:
        voter_weights = {voter: 1.0 for voter in df.columns}
    else:
        # If voter_weights exists but doesn't include all voters, add missing voters with weight 1.0
        for voter in df.columns:
            if voter not in voter_weights:
                voter_weights[voter] = 1.0

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

    # If df_wins is empty, return 0
    if df_wins.empty:
        if not return_preference_df:
            return 0
        else:
            df_pref = df.stack().reset_index()
            df_pref.columns = ["preference", "score_algo", "n_clusters"]
            df_pref = df_pref.pivot(index="n_clusters", columns="score_algo", values="preference")
            
            return 0, df_pref

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

    df_pref["wins"] = df_pref["wins"].astype(int)

    return winner_idx, df_pref