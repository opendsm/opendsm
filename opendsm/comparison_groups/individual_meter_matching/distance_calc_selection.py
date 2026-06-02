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

import logging

import numpy as np
import pandas as pd

from scipy import sparse
from scipy.spatial.distance import cdist

from qpsolvers import solve_ls

from opendsm.comparison_groups.individual_meter_matching import highs_settings as _highs_settings
from opendsm.comparison_groups.individual_meter_matching.settings import Settings

__all__ = ("DistanceMatching",)


logger = logging.getLogger(__name__)


def _iter_chunks(lst, n):
    """Yield successive non-overlapping chunks of size *n* from *lst*."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _distances(ls_t, ls_cp, weights=None, dist_metric="euclidean", n_meters_per_chunk=10000):
    """Compute pairwise distances between treatment and comparison pool load shapes.

    Returns float32 to halve memory usage — loadshape distances do not need
    float64 precision.

    Parameters
    ----------
    ls_t : np.ndarray, shape (n_treatment, n_features)
        Treatment load shapes.
    ls_cp : np.ndarray, shape (n_pool, n_features)
        Comparison pool load shapes.
    weights : array-like, optional
        Per-feature weights applied before distance calculation.
    dist_metric : str
        Distance metric accepted by ``scipy.spatial.distance.cdist``.
    n_meters_per_chunk : int
        Maximum number of comparison pool meters per chunk to limit peak
        memory usage.

    Returns
    -------
    np.ndarray, shape (n_treatment, n_pool), dtype float32
        Pairwise distance matrix.
    """
    if weights is not None:
        ls_t = ls_t * weights

    dist = [
        cdist(ls_t, chunk * weights if weights is not None else chunk, metric=dist_metric).astype(np.float32)
        for chunk in _iter_chunks(ls_cp, n_meters_per_chunk)
    ]

    return np.hstack(dist)


def _prefilter_pool(ls_t, ls_cp, per_treatment_k, n_meters_per_chunk, dist_metric, weights=None):
    """Union of each treatment's *per_treatment_k* nearest pool meters.

    Each treatment contributes its own nearest pool meters; their union forms
    the candidate set. Unlike a single global centroid, this preserves the true
    neighbours of treatments lying in different regions of load-shape space.
    The pool is processed in chunks with a running per-treatment top-k so peak
    memory stays bounded by one chunk's treatment x chunk distance matrix.

    Parameters
    ----------
    ls_t : pd.DataFrame, shape (n_treatment, n_features)
    ls_cp : pd.DataFrame, shape (n_pool, n_features)
    per_treatment_k : int
        Number of nearest pool meters kept per treatment before the union.
    n_meters_per_chunk : int
        Pool meters per chunk, bounding peak memory.
    dist_metric : str
        Distance metric accepted by ``scipy.spatial.distance.cdist``.
    weights : array-like, optional
        Per-feature weights applied before distance calculation.

    Returns
    -------
    np.ndarray
        Sorted indices into *ls_cp* for the kept candidates.
    """
    t = ls_t.values
    if weights is not None:
        t = t * weights

    n_treatment = t.shape[0]
    n_pool = ls_cp.shape[0]
    k = min(per_treatment_k, n_pool)

    best_dist = np.full((n_treatment, k), np.inf, dtype=np.float32)
    best_idx = np.full((n_treatment, k), -1, dtype=np.int64)

    start = 0
    for chunk in _iter_chunks(ls_cp.values, n_meters_per_chunk):
        if weights is not None:
            chunk = chunk * weights

        chunk_dist = cdist(t, chunk, metric=dist_metric).astype(np.float32)
        chunk_idx = np.broadcast_to(np.arange(start, start + len(chunk)), chunk_dist.shape)

        # merge running top-k with this chunk, keep the k nearest per treatment
        merged_dist = np.concatenate([best_dist, chunk_dist], axis=1)
        merged_idx = np.concatenate([best_idx, chunk_idx], axis=1)
        keep = np.argpartition(merged_dist, k - 1, axis=1)[:, :k]
        best_dist = np.take_along_axis(merged_dist, keep, axis=1)
        best_idx = np.take_along_axis(merged_idx, keep, axis=1)
        start += len(chunk)

    candidate_idx = np.unique(best_idx)
    candidate_idx = candidate_idx[candidate_idx >= 0]

    return candidate_idx


def highs_fit_comparison_group_loadshape(t_ls, cp_ls, coef_sum=1, solver="highs", settings=None, verbose=False):
    """Fit comparison pool load shapes to a target using constrained least squares.

    Solves a least-squares problem over the comparison pool to find a convex
    (or scaled) combination of pool meters that best approximates the mean
    treatment load shape.

    Parameters
    ----------
    t_ls : np.ndarray, shape (n_features,)
        Target (scaled mean treatment) load shape vector.
    cp_ls : np.ndarray, shape (n_pool, n_features)
        Comparison pool load shape matrix.
    coef_sum : float
        Required sum of the returned coefficients. Use ``1`` for a convex
        combination, or ``n_matches`` for a scaled variant.
    solver : str
        qpsolvers solver identifier (default ``"highs"``).
    settings : dict | None
        Solver settings dict (already lowercased). Defaults to tolerances
        tuned for the value of *coef_sum*.
    verbose : bool
        Whether to print solver output.

    Returns
    -------
    np.ndarray, shape (n_pool,)
        Optimal coefficient vector summing to *coef_sum*, clipped to [0, 1].
    """
    if settings is None:
        if coef_sum == 1:
            settings = _highs_settings.HiGHS_Settings(
                primal_feasibility_tolerance=1E-4,
                dual_feasibility_tolerance=1E-4,
            )
        else:
            settings = _highs_settings.HiGHS_Settings(
                primal_feasibility_tolerance=1,
                dual_feasibility_tolerance=1,
            )
        settings = {k.lower(): v for k, v in dict(settings).items()}

    _MIN_X = 1E-6 if coef_sum == 1 else 5E-3

    num_pool_meters = cp_ls.shape[0]

    R = sparse.csc_matrix(cp_ls.T)

    h = np.zeros(num_pool_meters)
    eye = sparse.eye(num_pool_meters, format="csc")
    A = sparse.csc_matrix(np.ones(num_pool_meters))
    b = np.array([coef_sum])

    lb = np.zeros(num_pool_meters)
    ub = np.ones(num_pool_meters)

    x_opt = solve_ls(R, t_ls, G=-eye, h=h, A=A, b=b, lb=lb, ub=ub, solver=solver, verbose=verbose, **settings)

    x_opt[x_opt < 0] = 0
    x_opt[x_opt > 1] = 1
    x_opt[np.abs(x_opt) < _MIN_X] = 0
    x_opt *= coef_sum / x_opt.sum()

    return x_opt


class DistanceMatchingError(Exception):
    """Raised when distance-based comparison group matching cannot be completed."""


class DistanceMatching:
    """
    Parameters
    ----------
    treatment_group: pd.DataFrame
        A dataframe representing treatment group meters, indexed by id, with each column being a data point in a usage pattern.
    comparison_pool: pd.DataFrame
        A dataframe representing comparison pool meters, indexed by id, with each column being a data point in a usage pattern.
    weights: list | 1D np.array
        A list of floats (must be of length of the treatment group columns) to scale the usage patterns in order to ensure that certain components of usage have higher weights towards matching than others.
    n_pool_meters_per_chunk: int
        Comparison pool meters are chunked so the cdist calculation fits in memory. 10,000 meters appear to be sufficient for most memory constraints.
    """

    def __init__(
        self,
        settings=None,
    ):
        if settings is None:
            self.settings = Settings()
        elif isinstance(settings, Settings):
            self.settings = settings
        else:
            raise DistanceMatchingError(
                "invalid settings provided to 'individual_metering_matching'"
            )

        self.dist_metric = self.settings.distance_metric
        if self.dist_metric == "manhattan":
            self.dist_metric = "cityblock"

    def _closest_idx_duplicates_allowed(self, distances, n_match=None):
        """Return indices of the *n_match* closest pool meters for each treatment.

        Duplicate assignments are permitted (one pool meter may appear in
        multiple treatment groups).

        Parameters
        ----------
        distances : np.ndarray, shape (n_treatment, n_pool)
        n_match : int | None
            Number of matches per treatment. Defaults to ``settings.n_matches_per_treatment``.

        Returns
        -------
        np.ndarray, shape (n_treatment, n_match)
            Column indices into *distances* for the closest meters.
        """
        if n_match is None:
            n_match = self.settings.n_matches_per_treatment

        n_match = min(n_match, distances.shape[1])

        # When all pool meters are requested, skip partitioning entirely.
        if n_match == distances.shape[1]:
            return np.tile(np.arange(n_match), (distances.shape[0], 1))

        # argpartition is faster than argsort for finding top-k (not fully sorted).
        # kth=n_match is valid here because n_match < distances.shape[1].
        return np.argpartition(distances, n_match, axis=1)[:, :n_match]

    def _closest_idx_no_duplicates_greedy(self, distances, n_match):
        """Greedy nearest-first no-duplicate matching.

        Treatments are processed closest-first (by their minimum distance) and
        each is assigned its n_match nearest pool meters not already taken. A
        cheap top-candidate block is searched first; if it is exhausted by
        earlier assignments the search falls back to the full pool, nearest
        first, so every treatment is filled while the pool lasts.

        Parameters
        ----------
        distances : np.ndarray, shape (n_treatment, n_pool), dtype float32
        n_match : int

        Returns
        -------
        list[list[int]]
            For each treatment, up to n_match pool indices (fewer only if the
            pool is exhausted across all treatments).
        """
        n_treatment, n_pool = distances.shape
        # cheap candidate block per treatment to avoid a full row sort in the common case
        candidate_k = min(n_match * 10, n_pool)

        if candidate_k < n_pool:
            top_k_idx = np.argpartition(distances, candidate_k, axis=1)[:, :candidate_k]
        else:
            top_k_idx = np.tile(np.arange(n_pool), (n_treatment, 1))

        assigned = set()
        cg_idx = [[] for _ in range(n_treatment)]

        # closest-first priority improves overall match quality
        treatment_order = np.argsort(distances.min(axis=1))

        for t_idx in treatment_order:
            row = distances[t_idx]

            # nearest available within the cheap candidate block
            block = top_k_idx[t_idx]
            block_sorted = block[np.argsort(row[block])]
            selected = [int(c) for c in block_sorted if c not in assigned][:n_match]

            # block exhausted by prior assignments: fall back to the full pool
            if len(selected) < n_match:
                taken = assigned.union(selected)
                for cp in np.argsort(row):
                    cp = int(cp)
                    if cp in taken:
                        continue

                    selected.append(cp)
                    if len(selected) == n_match:
                        break

            assigned.update(selected)
            cg_idx[t_idx] = selected

        return cg_idx

    def _closest_idx_duplicates_not_allowed(self, ls_t, ls_cp, distances):
        """Return comparison group indices such that no pool meter is assigned twice.

        Parameters
        ----------
        ls_t : pd.DataFrame, shape (n_treatment, n_features)
        ls_cp : pd.DataFrame, shape (n_pool, n_features)
        distances : np.ndarray, shape (n_treatment, n_pool)

        Returns
        -------
        list[list[int]] | np.ndarray
            For each treatment meter, a list (or row) of pool meter indices.
        """
        n_match = self.settings.n_matches_per_treatment
        selection_method = self.settings.selection_method

        n_treatment = ls_t.shape[0]
        n_pool = ls_cp.shape[0]

        requested_n_match = n_match
        n_match = min(n_match, n_pool // n_treatment)

        if n_match == 0:
            raise DistanceMatchingError(
                f"Not enough treatment pool meters {n_pool} to match with "
                f"{n_treatment} treatment meters without duplicates"
            )

        if n_match < requested_n_match:
            logger.warning(
                "Reduced matches per treatment from %d to %d: a pool of %d cannot supply "
                "%d unique matches for %d treatments.",
                requested_n_match, n_match, n_pool, requested_n_match, n_treatment,
            )

        if selection_method == "minimize_meter_distance":
            cg_idx = self._closest_idx_no_duplicates_greedy(distances, n_match)

        elif selection_method == "minimize_loadshape_distance":
            coef_sum = n_match * len(ls_t)
            ls_t_mean = np.mean(ls_t.values, axis=0) * coef_sum

            x_opt = highs_fit_comparison_group_loadshape(
                ls_t_mean, ls_cp.values, coef_sum=coef_sum, solver="highs", settings=None, verbose=False
            )

            x_opt_idx = np.argsort(x_opt)[::-1][:coef_sum]
            cg_idx = np.reshape(x_opt_idx, (ls_t.shape[0], n_match))

        else:
            raise DistanceMatchingError(f"Invalid selection method: {selection_method}")

        return cg_idx

    def get_comparison_group(
        self,
        treatment_group,
        comparison_pool,
        weights=None,
    ):
        ls_t = treatment_group
        ls_cp = comparison_pool

        n_match = self.settings.n_matches_per_treatment
        max_distance_threshold = self.settings.max_distance_threshold
        n_meters_per_chunk = self.settings.n_pool_meters_per_chunk
        candidate_multiplier = self.settings.candidate_multiplier

        # pre-filter the pool to the union of each treatment's nearest candidates
        n_treatment = ls_t.shape[0]
        n_pool = ls_cp.shape[0]

        if candidate_multiplier is not None:
            per_treatment_k = n_match * candidate_multiplier
            if n_treatment * per_treatment_k < n_pool:
                candidate_idx = _prefilter_pool(
                    ls_t, ls_cp, per_treatment_k, n_meters_per_chunk, self.dist_metric, weights
                )
                ls_cp = ls_cp.iloc[candidate_idx]

        # --- Optimization #1: chunk treatments, process end-to-end -----------
        if self.settings.allow_duplicate_matches:
            # Duplicates-allowed path: chunk treatments, compute distances per
            # chunk, select top-K, and discard each chunk's distance matrix.
            all_data = []
            for t_start in range(0, n_treatment, n_meters_per_chunk):
                t_end = min(t_start + n_meters_per_chunk, n_treatment)
                ls_t_chunk = ls_t.iloc[t_start:t_end]

                dist_chunk = _distances(
                    ls_t_chunk, ls_cp, weights, self.dist_metric, n_meters_per_chunk
                )

                cg_idx_chunk = self._closest_idx_duplicates_allowed(dist_chunk, n_match=n_match)

                # --- Optimization #6: vectorized assembly --------------------
                chunk_size = ls_t_chunk.shape[0]
                t_indices = np.repeat(np.arange(chunk_size), cg_idx_chunk.shape[1])
                cp_indices = cg_idx_chunk.ravel()

                ids = ls_cp.index[cp_indices]
                treatments = ls_t_chunk.index[t_indices]
                dists = dist_chunk[t_indices, cp_indices]

                chunk_df = pd.DataFrame({
                    "id": ids,
                    "treatment": treatments,
                    "distance": dists,
                })
                all_data.append(chunk_df)
                del dist_chunk

            df = pd.concat(all_data, ignore_index=True)
        else:
            # No-duplicates path: must compute full distance matrix for global
            # assignment (block-Hungarian or QP solver need cross-treatment view).
            # Still benefits from float32 + pre-filter.
            distances = _distances(ls_t, ls_cp, weights, self.dist_metric, n_meters_per_chunk)
            cg_idx = self._closest_idx_duplicates_not_allowed(ls_t, ls_cp, distances)

            # --- Optimization #6: vectorized assembly ------------------------
            all_data = []
            for t_idx in range(n_treatment):
                matches = cg_idx[t_idx]
                n_actual = len(matches)
                if n_actual == 0:
                    continue
                cp_arr = np.array(matches)
                all_data.append(pd.DataFrame({
                    "id": ls_cp.index[cp_arr],
                    "treatment": ls_t.index[t_idx],
                    "distance": distances[t_idx, cp_arr],
                }))

            del distances
            df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame(columns=["id", "treatment", "distance"])

        # check that the distance is less than the threshold
        if max_distance_threshold is not None:
            df = df[df["distance"] <= max_distance_threshold]

        # add column if id is duplicated
        df["duplicated"] = df.duplicated(subset="id", keep=False)

        return df


if __name__ == "__main__":
    d = DistanceMatching()
    print(d.settings)
