#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Adapted from https://github.com/FelSiq/DBCV
#  MIT License, Copyright (c) 2024 Felipe Alves Siqueira
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

#  -----------------------------------------------------------------------------

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

import itertools
from collections.abc import Sequence
from typing import Protocol

import numba
import numpy as np
import scipy.spatial.distance
import scipy.sparse.csgraph
import sklearn.neighbors



_EPS = 1e-12
_CORE_CLIP = 1e12


class _DistanceSource(Protocol):
    """Protocol for providing within- and between-cluster distance submatrices."""

    def within_cluster(self, cid: int) -> np.ndarray: ...

    def between_clusters(
        self, ci: int, cj: int,
        local_i: np.ndarray,
        local_j: np.ndarray,
    ) -> np.ndarray: ...


class _FullMatrixSource:
    """Extracts submatrices from the full pairwise distance matrix."""

    __slots__ = ("_dists", "_members")

    def __init__(
        self,
        dists: np.ndarray,
        members: Sequence[np.ndarray],
    ) -> None:
        self._dists = dists
        self._members = members

    def within_cluster(self, cid: int) -> np.ndarray:
        return self._dists[np.ix_(self._members[cid], self._members[cid])]

    def between_clusters(
        self, ci: int, cj: int,
        local_i: np.ndarray,
        local_j: np.ndarray,
    ) -> np.ndarray:
        return self._dists[np.ix_(self._members[ci][local_i], self._members[cj][local_j])]



@numba.jit(nopython=True, cache=True)
def _core_distances(dists, n_features):
    """Inverse d-distance power mean core distance.

    Fuses floor, power, diagonal exclusion, sum, clip, and inversion
    into a single pass per row. Reads *dists* without mutation.
    """
    EPS = 1e-12
    CLIP = 1e12
    n = dists.shape[0]
    core = np.empty((n, 1), dtype=dists.dtype)
    inv_d = -1.0 / n_features
    for i in range(n):
        total = 0.0
        for j in range(n):
            if i != j:
                d = dists[i, j]
                if d < EPS:
                    d = EPS
                total += d ** (-n_features)
        s = total / (n - 1)
        if s < 0.0:
            s = 0.0
        elif s > CLIP:
            s = CLIP
        core[i, 0] = s ** inv_d
    return core


def _mst_internal_nodes(
    mutual_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build MST and return internal node indices and their connecting edge weights.

    Internal nodes have degree > 1 in the MST.
    """
    mst_upper = scipy.sparse.csgraph.minimum_spanning_tree(mutual_reach)
    coo = mst_upper.tocoo()

    n_nodes = mutual_reach.shape[0]
    degrees = np.bincount(coo.row, minlength=n_nodes) + np.bincount(coo.col, minlength=n_nodes)
    internal = np.flatnonzero(degrees > 1)

    if internal.size > 0:
        is_internal = np.zeros(n_nodes, dtype=bool)
        is_internal[internal] = True
        mask = is_internal[coo.row] & is_internal[coo.col]
        if mask.any():
            return internal, coo.data[mask]

    # Fallback: no internal nodes, or no edges between them
    mst = mst_upper.toarray()
    mst += mst.T
    if internal.size == 0:
        return np.arange(n_nodes), mst
    
    return internal, mst


def _cluster_sparseness(
    dists: np.ndarray,
    n_features: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Density Sparseness (DSC): max internal MST edge weight for one cluster.

    Reads *dists* without mutation.
    """
    core = _core_distances(dists, n_features)

    # Mutual reachability: max(dists_ij, core_i, core_j) — own allocation
    mutual = np.maximum(dists, core)
    np.maximum(mutual, core.T, out=mutual)

    internal_local, edge_weights = _mst_internal_nodes(mutual)

    dsc = float(edge_weights.max())

    return dsc, core[internal_local], internal_local


def _dbcv_core(
    n_total: int,
    n_features: int,
    n_clusters: int,
    cluster_sizes: np.ndarray,
    dist: _DistanceSource,
) -> float:
    """Core DBCV computation on pre-validated, noise-free inputs."""
    dscs = np.zeros(n_clusters, dtype=float)
    internal_local: list[np.ndarray] = [None] * n_clusters      # type: ignore[list-item]
    internal_core: list[np.ndarray] = [None] * n_clusters    # type: ignore[list-item]

    for cid in range(n_clusters):
        dsc, core_i, local_i = _cluster_sparseness(
            dist.within_cluster(cid), n_features,
        )
        dscs[cid] = dsc
        internal_local[cid] = local_i
        internal_core[cid] = core_i

    min_dspcs = np.full(n_clusters, np.inf)

    for ci, cj in itertools.combinations(range(n_clusters), 2):
        pair = dist.between_clusters(ci, cj, internal_local[ci], internal_local[cj])
        np.maximum(pair, internal_core[ci], out=pair)
        np.maximum(pair, internal_core[cj].T, out=pair)
        dspc = float(pair.min())
        min_dspcs[ci] = min(min_dspcs[ci], dspc)
        min_dspcs[cj] = min(min_dspcs[cj], dspc)

    np.nan_to_num(min_dspcs, copy=False, posinf=_CORE_CLIP)
    vcs = (min_dspcs - dscs) / (_EPS + np.maximum(min_dspcs, dscs))
    np.nan_to_num(vcs, copy=False, nan=0.0)
    return float(np.sum(vcs * cluster_sizes)) / n_total


def dbcv_prevalidated(
    n_total: int,
    n_features: int,
    cluster_sizes: np.ndarray,
    cluster_members: Sequence[np.ndarray],
    precomputed_distances: np.ndarray,
) -> float:
    """Fast path for callers that have already removed noise/singletons.

    Skips validation, noise filtering, and label parsing.
    Distance matrix is read-only (no copy, no mutation).
    """
    dist = _FullMatrixSource(precomputed_distances, cluster_members)
    return _dbcv_core(n_total, n_features, len(cluster_members), cluster_sizes, dist)


def dbcv(
    X: np.ndarray,
    y: np.ndarray,
    precomputed_distances: np.ndarray | None = None,
    metric: str = "sqeuclidean",
    noise_id: int = -1,
    check_duplicates: bool = True,
) -> float:
    """Density-Based Clustering Validation index.

    Moulavi et al., "Density-Based Clustering Validation", SDM 2014.
    https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf

    Adapted from https://github.com/FelSiq/DBCV
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = np.asarray(y, dtype=int)
    n, n_features = X.shape

    if n != y.size:
        raise ValueError(f"Mismatch in {X.shape[0]=} and {y.size=} dimensions.")
    if y.size == 0:
        return 0.0

    # Filter noise and singletons
    ids, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    keep = (ids != noise_id) & (counts > 1)
    non_noise_inds = np.flatnonzero(keep[inverse])

    if non_noise_inds.size == 0:
        return 0.0

    _, y_dense, cluster_sizes = np.unique(
        inverse[non_noise_inds], return_inverse=True, return_counts=True,
    )
    n_clusters = cluster_sizes.size

    # Build cluster member lists
    order = np.argsort(y_dense, kind='stable')
    splits = np.searchsorted(y_dense[order], np.arange(1, n_clusters))
    cluster_members = np.split(order, splits)

    # Build distance source
    if precomputed_distances is None:
        X = X[non_noise_inds, :]
        if check_duplicates and X.shape[0] > 1:
            nn = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
            nn.fit(X)
            nn_dists, _ = nn.kneighbors(return_distance=True)
            if np.any(nn_dists < 1e-9):
                raise ValueError("Duplicated samples have been found in X.")
            
        pairwise = scipy.spatial.distance.cdist(X, X, metric=metric)
        np.maximum(pairwise, _EPS, out=pairwise)
        np.fill_diagonal(pairwise, np.inf)
        dist = _FullMatrixSource(pairwise, cluster_members)

    else:
        if non_noise_inds.size == precomputed_distances.shape[0]:
            ref = precomputed_distances
        else:
            ref = precomputed_distances[np.ix_(non_noise_inds, non_noise_inds)]
        dist = _FullMatrixSource(ref, cluster_members)

    return _dbcv_core(n, n_features, n_clusters, cluster_sizes, dist)
