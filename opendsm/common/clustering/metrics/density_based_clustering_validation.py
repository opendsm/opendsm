"""
From https://github.com/FelSiq/DBCV

MIT License

Copyright (c) 2024 Felipe Alves Siqueira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import multiprocessing
import typing as t
import itertools
import functools

import numpy as np
import numpy.typing as npt
import sklearn.neighbors
import scipy.spatial.distance
import scipy.sparse.csgraph
import scipy.stats
import mpmath



_MP = mpmath.mp.clone()


def prim_mst(
    graph: npt.NDArray[np.float32], ind_root: int = 0
) -> npt.NDArray[np.float32]:
    """Python translation of the original implementation of Prim's MST in MATLAB.

    Reference source: https://github.com/pajaskowiak/dbcv/blob/main/src/MST_Edges.m
    """

    n = len(graph)
    intree = np.full(n, fill_value=False)
    d = np.full(n, fill_value=np.inf)

    d[ind_root] = 0
    v = ind_root
    counter = 0

    G = {
        "MST_edges": {
            "node_inds": np.zeros((n - 1, 2), dtype=int),
            "weights": np.zeros(n - 1, dtype=float),
        },
        "MST_degrees": np.zeros(n, dtype=int),
        "MST_parent": np.arange(n),
    }

    while counter < n - 1:
        intree[v] = True
        dist = np.inf

        for w in np.arange(n):
            if w != v and not intree[w]:
                weight = graph[v, w]

                if d[w] > weight:
                    d[w] = weight
                    G["MST_parent"][w] = v

                if dist > d[w]:
                    dist = d[w]
                    next_v = w

        counter += 1
        G["MST_edges"]["node_inds"][counter - 1, :] = (G["MST_parent"][next_v], next_v)
        G["MST_edges"]["weights"][counter - 1] = graph[G["MST_parent"][next_v], next_v]
        G["MST_degrees"][G["MST_parent"][next_v]] += 1
        G["MST_degrees"][next_v] += 1
        v = next_v

    (inds_a, inds_b) = G["MST_edges"]["node_inds"].T
    weights = G["MST_edges"]["weights"]

    mst = np.zeros_like(graph)
    mst[inds_a, inds_b] = weights
    mst[inds_b, inds_a] = weights

    return mst


def compute_pair_to_pair_dists(
    X: npt.NDArray[np.float64], metric: str
) -> npt.NDArray[np.float64]:
    dists = scipy.spatial.distance.cdist(X, X, metric=metric)
    np.maximum(dists, 1e-12, out=dists)
    # NOTE: set self-distance to +inf to prevent points being self-neighbors.
    np.fill_diagonal(dists, val=np.inf)
    return dists


def get_subarray(
    arr: npt.NDArray[np.float64],
    /,
    inds_a: t.Optional[npt.NDArray[np.int32]] = None,
    inds_b: t.Optional[npt.NDArray[np.int32]] = None,
) -> npt.NDArray[np.float64]:
    if inds_a is None:
        return arr
    if inds_b is None:
        inds_b = inds_a
    inds_a_mesh, inds_b_mesh = np.meshgrid(inds_a, inds_b)
    return arr[inds_a_mesh, inds_b_mesh].T


def get_internal_objects(
    mutual_reach_dists: npt.NDArray[np.float64], use_original_mst_implementation: bool
) -> npt.NDArray[np.float64]:
    if use_original_mst_implementation:
        mutual_reach_dists = np.copy(mutual_reach_dists)
        np.fill_diagonal(mutual_reach_dists, 0.0)
        mst = prim_mst(mutual_reach_dists)

    else:
        mst = scipy.sparse.csgraph.minimum_spanning_tree(mutual_reach_dists)
        mst = mst.toarray()
        mst += mst.T

    is_mst_edges = (mst > 0.0).astype(int, copy=False)

    internal_node_inds = is_mst_edges.sum(axis=0) > 1
    internal_node_inds = np.flatnonzero(internal_node_inds)

    internal_edge_weights = get_subarray(mst, inds_a=internal_node_inds)

    graph_has_internal_nodes = bool(internal_node_inds.size > 0)
    graph_has_at_least_two_internal_nodes = bool(internal_edge_weights.size > 1)

    return (
        (
            internal_node_inds
            if graph_has_internal_nodes
            else np.arange(mutual_reach_dists.shape[0])
        ),
        internal_edge_weights if graph_has_at_least_two_internal_nodes else mst,
    )


def compute_cluster_core_distance(
    dists: npt.NDArray[np.float64], d: int, enable_dynamic_precision: bool
) -> npt.NDArray[np.float64]:
    n, _ = dists.shape
    orig_dists_dtype = dists.dtype

    if enable_dynamic_precision:
        dists = np.asarray(_MP.matrix(dists), dtype=object).reshape(*dists.shape)

    core_dists = np.power(dists, -d).sum(axis=-1, keepdims=True) / (n - 1)

    if not enable_dynamic_precision:
        np.clip(core_dists, a_min=0.0, a_max=1e12, out=core_dists)

    np.power(core_dists, -1.0 / d, out=core_dists)

    if enable_dynamic_precision:
        core_dists = np.asarray(core_dists, dtype=orig_dists_dtype)

    return core_dists


def compute_mutual_reach_dists(
    dists: npt.NDArray[np.float64],
    d: float,
    enable_dynamic_precision: bool,
) -> npt.NDArray[np.float64]:
    core_dists = compute_cluster_core_distance(
        d=d, dists=dists, enable_dynamic_precision=enable_dynamic_precision
    )
    mutual_reach_dists = dists.copy()
    np.maximum(mutual_reach_dists, core_dists, out=mutual_reach_dists)
    np.maximum(mutual_reach_dists, core_dists.T, out=mutual_reach_dists)
    return (core_dists, mutual_reach_dists)


def fn_density_sparseness(
    cls_inds: npt.NDArray[np.int32],
    dists: npt.NDArray[np.float64],
    d: int,
    enable_dynamic_precision: bool,
    use_original_mst_implementation: bool,
) -> t.Tuple[float, npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    (core_dists, mutual_reach_dists) = compute_mutual_reach_dists(
        dists=dists, d=d, enable_dynamic_precision=enable_dynamic_precision
    )
    (internal_node_inds, internal_edge_weights) = get_internal_objects(
        mutual_reach_dists,
        use_original_mst_implementation=use_original_mst_implementation,
    )
    dsc = float(internal_edge_weights.max())
    internal_core_dists = core_dists[internal_node_inds]
    internal_node_inds = cls_inds[internal_node_inds]
    return (dsc, internal_core_dists, internal_node_inds)


def fn_density_separation(
    cls_i: int,
    cls_j: int,
    dists: npt.NDArray[np.float64],
    internal_core_dists_i: npt.NDArray[np.float64],
    internal_core_dists_j: npt.NDArray[np.float64],
) -> t.Tuple[int, int, float]:
    sep = dists.copy()
    np.maximum(sep, internal_core_dists_i, out=sep)
    np.maximum(sep, internal_core_dists_j.T, out=sep)
    dspc_ij = float(sep.min()) if sep.size else np.inf
    return (cls_i, cls_j, dspc_ij)


def _check_duplicated_samples(X: npt.NDArray[np.float64], threshold: float = 1e-9):
    if X.shape[0] <= 1:
        return

    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
    nn.fit(X)
    dists, _ = nn.kneighbors(return_distance=True)

    if np.any(dists < threshold):
        raise ValueError("Duplicated samples have been found in X.")


def _convert_singleton_clusters_to_noise(
    y: npt.NDArray[np.int32], noise_id: int
) -> npt.NDArray[np.int32]:
    """Cast clusters containing a single instance as noise."""
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)
    singleton_clusters = cluster_ids[cluster_sizes == 1]

    if singleton_clusters.size == 0:
        return y

    return np.where(np.isin(y, singleton_clusters), noise_id, y)


def dbcv(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int32],
    precomputed_distances: t.Optional[npt.NDArray[np.float64]] = None,
    metric: str = "sqeuclidean",
    noise_id: int = -1,
    check_duplicates: bool = True,
    n_processes: t.Union[int, str] = "auto",
    enable_dynamic_precision: bool = False,
    bits_of_precision: int = 512,
    use_original_mst_implementation: bool = False,
) -> float:
    """Compute DBCV metric.

    Density-Based Clustering Validation (DBCV) is an intrinsic (= unsupervised/unlabeled)
    relative metric. See reference [1] for the original reference.

    Parameters
    ----------
    X : npt.NDArray[np.float64] of shape (N, D)
        Sample embeddings.

    y : npt.NDArray[np.int32] of shape (N,)
        Cluster IDs assigned for each sample in X.

    metric : str, default="sqeuclidean"
        This parameter specifies the metric function to compute dissimilarities between observations.
        The DBCV metric estimation may vary depending on the distance metric used.
        This argument is passed to `scipy.spatial.distance.cdist`.
        The default value is the squared Euclidean distance, which is also employed in the original
        MATLAB implementation (see reference [2]).

    noise_id : int, default=-1
        The noise "cluster" ID refers to instances where `y[i] = noise_id`, which are considered noise.
        Additionally, singleton clusters, meaning clusters containing only a single instance, are automatically
        classified as noise.

    check_duplicates : bool, default=True
        If set to True, check for duplicated samples before execution.
        Instances with Euclidean distance to their nearest neighbor below 1e-9 are considered
        duplicates.

    n_processes : int or "auto", default="auto"
        Maximum number of parallel processes for processing clusters and cluster pairs.
        If `n_processes="auto"`, the number of parallel processes will be set to 1 for
        datasets with 500 or fewer instances, and 4 for datasets with more than 500 instances.

    enable_dynamic_precision : bool, default=False
        If set to True, this activates a dynamic quantity of bits of precision for floating point during
        density calculation, as defined by the `bits_of_precision` argument below. Enabling this argument
        ensures proper density calculation for very high-dimensional data, although it significantly slows
        down the process compared to standard calculations.

    bits_of_precision : int, default=512
        Bits of precision for density calculation. High values are necessary for high
        dimensions to avoid underflow/overflow.

    use_original_mst_implementation : bool, default=False
        If set to False, the function will use Scipy's MST implementation (Kruskal's implementation).
        If set to True, the function will use an exact replica of the original MATLAB implementation.
        This version is a variant of Prim's MST algorithm.
        The original implementation is slower than Scipy's implementation and tends to create hub nodes
        much more often.
        Since these implementations are not equivalent, the DBCV metric estimation tends to vary depending
        on the MST algorithm used.

    Returns
    -------
    DBCV : float
        DBCV metric estimation.

    Source
    ------
    .. [1] "Density-Based Clustering Validation". Davoud Moulavi, Pablo A. Jaskowiak,
           Ricardo J. G. B. Campello, Arthur Zimek, Jörg Sander.
           https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf
    .. [2] https://github.com/pajaskowiak/dbcv/
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y = np.asarray(y, dtype=int)

    n, d = X.shape  # NOTE: 'n' must be calculated before removing noise.

    if n != y.size:
        raise ValueError(f"Mismatch in {X.shape[0]=} and {y.size=} dimensions.")

    if y.size == 0:
        return 0.0
    
    if precomputed_distances is None:
        y = _convert_singleton_clusters_to_noise(y, noise_id=noise_id)

        non_noise_inds = y != noise_id
        X = X[non_noise_inds, :]
        y = y[non_noise_inds]

        if y.size == 0:
            return 0.0

        if check_duplicates:
            _check_duplicated_samples(X)

        dists = compute_pair_to_pair_dists(X=X, metric=metric)

    else:
        dists = precomputed_distances.copy()
        np.maximum(dists, 1e-12, out=dists)
        # NOTE: set self-distance to +inf to prevent points being self-neighbors.
        np.fill_diagonal(dists, val=np.inf)

    y = scipy.stats.rankdata(y, method="dense") - 1
    cluster_ids, cluster_sizes = np.unique(y, return_counts=True)

    # DSC: 'Density Sparseness of a Cluster'
    dscs = np.zeros(cluster_ids.size, dtype=float)

    # DSPC: 'Density Separation of a Pair of Clusters'
    min_dspcs = np.full(cluster_ids.size, fill_value=np.inf)

    # Internal objects = Internal nodes = nodes such that degree(node) > 1 in MST.
    internal_objects_per_cls: t.Dict[int, npt.NDArray[np.int32]] = {}

    # internal core distances = core distances of internal nodes
    internal_core_dists_per_cls: t.Dict[int, npt.NDArray[np.float32]] = {}

    cls_inds = [np.flatnonzero(y == cls_id) for cls_id in cluster_ids]

    if n_processes == "auto":
        n_processes = 4 if y.size > 500 else 1

    with _MP.workprec(bits_of_precision), multiprocessing.Pool(
        processes=min(n_processes, cluster_ids.size)
    ) as ppool:
        fn_density_sparseness_ = functools.partial(
            fn_density_sparseness,
            d=d,
            enable_dynamic_precision=enable_dynamic_precision,
            use_original_mst_implementation=use_original_mst_implementation,
        )

        args = [(cls_ind, get_subarray(dists, inds_a=cls_ind)) for cls_ind in cls_inds]

        for cls_id, (dsc, internal_core_dists, internal_node_inds) in enumerate(
            ppool.starmap(fn_density_sparseness_, args)
        ):
            internal_objects_per_cls[cls_id] = internal_node_inds
            internal_core_dists_per_cls[cls_id] = internal_core_dists
            dscs[cls_id] = dsc

    n_cls_pairs = (cluster_ids.size * (cluster_ids.size - 1)) // 2

    if n_cls_pairs > 0:
        with _MP.workprec(bits_of_precision), multiprocessing.Pool(
            processes=min(n_processes, n_cls_pairs)
        ) as ppool:
            args = [
                (
                    cls_i,
                    cls_j,
                    get_subarray(
                        dists,
                        inds_a=internal_objects_per_cls[cls_i],
                        inds_b=internal_objects_per_cls[cls_j],
                    ),
                    internal_core_dists_per_cls[cls_i],
                    internal_core_dists_per_cls[cls_j],
                )
                for cls_i, cls_j in itertools.combinations(cluster_ids, 2)
            ]

            for cls_i, cls_j, dspc_ij in ppool.starmap(fn_density_separation, args):
                min_dspcs[cls_i] = min(min_dspcs[cls_i], dspc_ij)
                min_dspcs[cls_j] = min(min_dspcs[cls_j], dspc_ij)

    np.nan_to_num(min_dspcs, copy=False, posinf=1e12)
    vcs = (min_dspcs - dscs) / (1e-12 + np.maximum(min_dspcs, dscs))
    np.nan_to_num(vcs, copy=False, nan=0.0)
    dbcv = float(np.sum(vcs * cluster_sizes)) / n

    return dbcv