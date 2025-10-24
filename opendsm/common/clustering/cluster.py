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

import warnings

import numpy as np
import pandas as pd

import pywt

from scipy.signal import find_peaks
from scipy.spatial.distance import cdist, pdist, squareform


from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN, HDBSCAN, Birch, SpectralClustering
from sklearn.metrics.pairwise import pairwise_kernels

from opendsm.common.clustering import (
    bisect_k_means as _bisect_k_means,
    scoring as _scoring,
    settings as _settings,
    voting as _voting,
)
from opendsm.common.utils import median_absolute_deviation as MAD


def score_council(settings: _settings.ClusteringSettings):
    """
    Set the score council for the given settings.
    """
    score_council = {
        'calinski_harabasz_index': settings.spectral.scoring.calinski_harabasz_weight,
        'davies_bouldin_index': settings.spectral.scoring.davies_bouldin_weight,
        'density_based_clustering_validation_index': settings.spectral.scoring.density_based_clustering_validation_weight,
        'dunn_index': settings.spectral.scoring.dunn_weight,
        'silhouette_index': settings.spectral.scoring.silhouette_weight,
        'silhouette_median_index': settings.spectral.scoring.silhouette_median_weight,
        'xie_beni_index': settings.spectral.scoring.xie_beni_weight,
    }

    return score_council


def score_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    Score clusters of the given data with the selected choices.
    """
    n_cluster_lower = settings.spectral.n_cluster.lower

    dist_metric = settings.spectral.scoring.distance_metric
    min_cluster_size = settings.spectral.scoring.min_cluster_size
    max_non_outlier_cluster_count = 200
    
    label_res = _scoring.score_clusters(
        data,
        labels,
        n_cluster_lower,
        score_council(settings),
        dist_metric,
        min_cluster_size,
        max_non_outlier_cluster_count,
    )

    return label_res


def _bisecting_kmeans_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using Bisecting K-Means algorithm
    """
    
    recluster_count = settings.bisecting_kmeans.recluster_count
    n_cluster_lower = settings.bisecting_kmeans.n_cluster.lower
    n_cluster_upper = settings.bisecting_kmeans.n_cluster.upper
    n_init = settings.bisecting_kmeans.internal_recluster_count
    inner_algorithm = settings.bisecting_kmeans.inner_algorithm
    bisecting_strategy = settings.bisecting_kmeans.bisecting_strategy

    window_size = settings.bisecting_kmeans.scoring.window_size

    seed = settings._seed

    results = []
    for i in range(recluster_count):
        algo = _bisect_k_means.BisectingKMeans(
            n_clusters=n_cluster_upper,
            init="k-means++",  # does not benefit from k-means++ like other k-means
            n_init=n_init,
            random_state=seed + i,
            algorithm=inner_algorithm,
            bisecting_strategy=bisecting_strategy,
        )
        algo.fit(data)
        labels_dict = algo.labels_full

        # if specifying clusters, only score the specified clusters
        if n_cluster_lower == n_cluster_upper:
            labels_dict = {n_cluster_lower: labels_dict[n_cluster_lower]}

        for n_cluster, labels in labels_dict.items():
            label_res = score_clusters(data, labels, settings)
            results.append(label_res)

    df_votes = _voting.construct_voting_df(results)
    winner, df_votes = _voting.shulze_voting(df_votes, score_council(settings), window_size)

    # get labels of winner from results
    winner_labels = None
    for res in results:
        if res.n_clusters == winner:
            winner_labels = res.labels
            break

    return winner_labels


def _birch_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    Clusters features using Birch algorithm
    """

    n_cluster_lower = settings.birch.n_cluster.lower
    n_cluster_upper = settings.birch.n_cluster.upper
    threshold = settings.birch.threshold
    branching_factor = settings.birch.branching_factor

    window_size = settings.birch.scoring.window_size

    results = []
    for n_clusters in range(n_cluster_lower, n_cluster_upper + 1):
        algo = Birch(
            n_clusters=n_clusters,
            threshold=threshold,
            branching_factor=branching_factor,
        )
        labels = algo.fit_predict(data)
        
        # Calculate score for the clusters
        label_res = score_clusters(data, labels, settings)
        
        results.append(label_res)

    df_votes = _voting.construct_voting_df(results)
    winner, df_votes = _voting.shulze_voting(df_votes, score_council(settings), window_size)

    # get labels of winner from results
    winner_labels = None
    for res in results:
        if res.n_clusters == winner:
            winner_labels = res.labels
            break

    return winner_labels


def _dbscan_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using DBSCAN algorithm
    """
    algo = DBSCAN(
        eps=settings.dbscan.epsilon, 
        min_samples=settings.dbscan.min_samples, 
        metric=settings.dbscan.distance_metric.value,
        algorithm=settings.dbscan.nearest_neighbors_algorithm,
        leaf_size=settings.dbscan.leaf_size,
        p=settings.dbscan.minkowski_p,
    )
    labels = algo.fit_predict(data)

    return labels


def _hdbscan_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using HDBSCAN algorithm
    """
    min_samples = settings.hdbscan.min_samples
    if settings.hdbscan.min_samples == 1:
        min_samples = 2

    algo = HDBSCAN(
        min_samples=settings.hdbscan.scoring_sample_count, 
        min_cluster_size=min_samples,
        allow_single_cluster=settings.hdbscan.allow_single_cluster,
        max_cluster_size=settings.hdbscan.max_cluster_size,
        metric=settings.hdbscan.distance_metric,
        cluster_selection_epsilon=settings.hdbscan.cluster_selection_epsilon,
        alpha=settings.hdbscan.robust_single_linkage_scaling,
        algorithm=settings.hdbscan.nearest_neighbors_algorithm,
        leaf_size=settings.hdbscan.leaf_size,
        cluster_selection_method=settings.hdbscan.cluster_selection_method,
    )
    labels = algo.fit_predict(data)

    if settings.hdbscan.min_samples == 1:
        # get count of -1 labels
        outlier_count = np.sum(labels == -1)

        if outlier_count == 0:
            return labels

        # add to all labels to make room for outliers
        labels[labels != -1] += outlier_count

        # make labels with -1 defined as arange(max_label+1, n_samples)
        labels[labels == -1] = np.arange(0, outlier_count)

    return labels


from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

def eigenDecomposition(A, topK = 5):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors
    
    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic
    
    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    
    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
    # the euclidean norm of complex numbers.
#     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = np.linalg.eig(L)
        
    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1]
    nb_clusters = index_largest_gap + 1
        
    return nb_clusters, eigenvalues, eigenvectors

def eigendecomp_cluster_count(
    X,
    settings: _settings.ClusteringSettings
):
    """
    Votes on the optimal number of clusters using the eigen decomposition
    """
    min_clusters = settings.spectral.n_cluster.lower
    max_clusters = settings.spectral.n_cluster.upper

    nb_clusters, _, _ = eigenDecomposition(X)

    # only include clusters in the range of min_clusters to max_clusters
    nb_clusters = nb_clusters[nb_clusters >= min_clusters]
    nb_clusters = nb_clusters[nb_clusters <= max_clusters]

    return nb_clusters


def _affinity_matrix(
    data: np.ndarray,
    algo: SpectralClustering,
):
    """
    Computes the affinity matrix for the given data
    """
    params = algo.kernel_params
    if params is None:
        params = {}
    if not callable(algo.affinity):
        params["gamma"] = algo.gamma
        params["degree"] = algo.degree
        params["coef0"] = algo.coef0
    X = pairwise_kernels(
        data, metric=algo.affinity, filter_params=True, **params
    )

    return X

def _single_spectral_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using Spectral Clustering algorithm
    """

    n_cluster_lower = settings.spectral.n_cluster.lower
    n_cluster_upper = settings.spectral.n_cluster.upper

    window_size = settings.spectral.scoring.window_size

    algo = SpectralClustering(
        n_clusters=n_cluster_lower,
        eigen_solver=settings.spectral.eigen_solver,
        n_components=settings.spectral.n_components,
        affinity=settings.spectral.affinity,
        n_neighbors=settings.spectral.nearest_neighbors,
        gamma=settings.spectral.gamma,
        eigen_tol=settings.spectral.eigen_tol,
        assign_labels=settings.spectral.assign_labels,
        random_state=settings._seed
    )

    # transform data as spectral clustering doesn't like negative values
    # data = np.exp(-data / np.std(data))

    # X = _local_affinity_matrix(data)
    X = _affinity_matrix(data, algo)
    algo.affinity = "precomputed"

    results = []
    n_clusters_range = np.arange(n_cluster_lower, n_cluster_upper + 1)
    for n_clusters in n_clusters_range:
        if n_clusters > n_cluster_lower:
            algo.n_clusters = n_clusters

        np_state = np.random.get_state()
        np.random.seed(settings._seed)
            
        # hide UserWarning from sklearn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            labels = algo.fit_predict(X)

        # Calculate a score for the clustering
        label_res = score_clusters(data, labels, settings)

        np.random.set_state(np_state)
        
        results.append(label_res)

    df_votes = _voting.construct_voting_df(results)
    # df_votes.index = n_clusters_range

    # # drop single cluster from df_votes
    # df_votes = df_votes.drop(index=1, errors='ignore')

    winner_idx, df_votes = _voting.shulze_voting(df_votes, score_council(settings), window_size)
    df_votes.index = n_clusters_range

    # get labels of winner from results
    label_res = results[winner_idx]

    return label_res, df_votes
    

def _local_affinity_matrix(X):
    dim = X.shape[0]
    dist_ = pdist(X)
    pd = np.zeros([dim, dim])
    dist = iter(dist_)
    for i in range(dim):
        for j in range(i+1, dim):  
            d = next(dist)
            pd[i,j] = d
            pd[j,i] = d
            
    #calculate local sigma
    sigmas = np.zeros(dim)
    for i in range(len(pd)):
        sigmas[i] = sorted(pd[i])[7]
    
    A = np.zeros([dim, dim])
    dist = iter(dist_)
    for i in range(dim):
        for j in range(i+1, dim):  
            d = np.exp(-1*next(dist)**2/(sigmas[i]*sigmas[j]))

            A[i,j] = d
            A[j,i] = d

    return A

def _spectral_clustering(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    clusters features using Spectral Clustering algorithm
    """
    recluster_count = settings.spectral.recluster_count
    
    results = []
    df_votes_recluster = []
    for n in range(recluster_count + 1):
        if n > 0:
            settings_dict = settings.model_dump()
            settings_dict["seed"] = settings_dict["seed"] + 1
            settings = _settings.ClusteringSettings(**settings_dict)
        
        label_res, df_votes = _single_spectral_clustering(data, settings)
        
        results.append(label_res)
        df_votes_recluster.append(df_votes)

    winner_idx = 0
    if recluster_count > 0:
        df_votes = _voting.construct_voting_df(results)
        winner_idx, df_votes = _voting.shulze_voting(df_votes, score_council(settings), window_size=0)

    # get labels of winner from results
    winner_labels = results[winner_idx].labels
    df_votes_recluster = df_votes_recluster[winner_idx]

    return winner_labels, df_votes_recluster

class RobustSpectralClustering:
    """
    Implementation of the method proposed in the paper:
    'Robust Spectral Clustering for Noisy Data: Modeling Sparse Corruptions Improves Latent Embeddings'

    If you publish material based on algorithms or evaluation measures obtained from this code,
    then please note this in your acknowledgments and please cite the following paper:
        Aleksandar Bojchevski, Yves Matkovic, and Stephan Günnemann.
        2017. Robust Spectral Clustering for Noisy Data.
        In Proceedings of KDD’17, August 13–17, 2017, Halifax, NS, Canada.

    Copyright (C) 2017
    Aleksandar Bojchevski
    Yves Matkovic
    Stephan Günnemann
    Technical University of Munich, Germany
    """

    def __init__(self, k, nn=15, theta=20, m=0.5, laplacian=1, n_iter=50, normalize=False, affinity="local", verbose=False):
        """
        :param k: number of clusters
        :param nn: number of neighbours to consider for constructing the KNN graph (excluding the node itself)
        :param theta: number of corrupted edges to remove
        :param m: minimum percentage of neighbours to keep per node (omega_i constraints)
        :param n_iter: number of iterations of the alternating optimization procedure
        :param laplacian: which graph Laplacian to use: 0: L, 1: L_rw, 2: L_sym
        :param normalize: whether to row normalize the eigen vectors before performing k-means
        :param verbose: verbosity
        """

        self.k = k
        self.nn = nn
        self.theta = theta
        self.m = m
        self.n_iter = n_iter
        self.normalize = normalize
        self.verbose = verbose
        self.laplacian = laplacian
        self.affinity = affinity

        if laplacian == 0:
            if self.verbose:
                print('Using unnormalized Laplacian L')
        elif laplacian == 1:
            if self.verbose:
                print('Using random walk based normalized Laplacian L_rw')
        elif laplacian == 2:
            raise NotImplementedError('The symmetric normalized Laplacian L_sym is not implemented yet.')
        else:
            raise ValueError('Choice of graph Laplacian not valid. Please use 0, 1 or 2.')

    def __affinity_matrix(self, X):
        # compute the KNN graph
        A = kneighbors_graph(X=X, n_neighbors=self.nn, metric='euclidean', include_self=False, mode='connectivity')
        A = A.maximum(A.T)  # make the graph undirected

        return A

    def __local_affinity_matrix(self, X):
        dim = X.shape[0]
        dist_ = pdist(X)
        pd = np.zeros([dim, dim])
        dist = iter(dist_)
        for i in range(dim):
            for j in range(i+1, dim):  
                d = next(dist)
                pd[i,j] = d
                pd[j,i] = d
                
        #calculate local sigma
        sigmas = np.zeros(dim)
        for i in range(len(pd)):
            sigmas[i] = sorted(pd[i])[7]
        
        A = np.zeros([dim, dim])
        dist = iter(dist_)
        for i in range(dim):
            for j in range(i+1, dim):  
                d = np.exp(-1*next(dist)**2/(sigmas[i]*sigmas[j]))

                A[i,j] = d
                A[j,i] = d

        return A

    def __latent_decomposition(self, X):
        # compute the KNN graph
        if self.affinity != "local":
            A = self.__affinity_matrix(X)
        else:   
            A = self.__local_affinity_matrix(X)
            
        N = A.shape[0]  # number of nodes
        deg = A.sum(0).A1  # node degrees

        prev_trace = np.inf  # keep track of the trace for convergence
        Ag = A.copy()

        for it in range(self.n_iter):

            # form the unnormalized Laplacian
            D = sp.diags(Ag.sum(0).A1).tocsc()
            L = D - Ag

            # solve the normal eigenvalue problem
            if self.laplacian == 0:
                h, H = eigsh(L, self.k, which='SM')
            # solve the generalized eigenvalue problem
            elif self.laplacian == 1:
                h, H = eigsh(L, self.k, D, which='SM')

            trace = h.sum()

            if self.verbose:
                print('Iter: {} Trace: {:.4f}'.format(it, trace))

            if self.theta == 0:
                # no edges are removed
                Ac = sp.coo_matrix((N, N), [np.int])
                break

            if prev_trace - trace < 1e-10:
                # we have converged
                break

            allowed_to_remove_per_node = (deg * self.m).astype(np.int)
            prev_trace = trace

            # consider only the edges on the lower triangular part since we are symmetric
            edges = sp.tril(A).nonzero()
            removed_edges = []

            if self.laplacian == 1:
                # fix for potential numerical instability of the eigenvalues computation
                h[np.isclose(h, 0)] = 0

                # equation (5) in the paper
                p = np.linalg.norm(H[edges[0]] - H[edges[1]], axis=1) ** 2 \
                    - np.linalg.norm(H[edges[0]] * np.sqrt(h), axis=1) ** 2 \
                    - np.linalg.norm(H[edges[1]] * np.sqrt(h), axis=1) ** 2
            else:
                # equation (4) in the paper
                p = np.linalg.norm(H[edges[0]] - H[edges[1]], axis=1) ** 2

            # greedly remove the worst edges
            for ind in p.argsort()[::-1]:
                e_i, e_j, p_e = edges[0][ind], edges[1][ind], p[ind]

                # remove the edge if it satisfies the constraints
                if allowed_to_remove_per_node[e_i] > 0 and allowed_to_remove_per_node[e_j] > 0 and p_e > 0:
                    allowed_to_remove_per_node[e_i] -= 1
                    allowed_to_remove_per_node[e_j] -= 1
                    removed_edges.append((e_i, e_j))
                    if len(removed_edges) == self.theta:
                        break

            removed_edges = np.array(removed_edges)
            Ac = sp.coo_matrix((np.ones(len(removed_edges)), (removed_edges[:, 0], removed_edges[:, 1])), shape=(N, N))
            Ac = Ac.maximum(Ac.T)
            Ag = A - Ac

        return Ag, Ac, H

    def fit_predict(self, X):
        """
        :param X: array-like or sparse matrix, shape (n_samples, n_features)
        :return: cluster labels ndarray, shape (n_samples,)
        """

        Ag, Ac, H = self.__latent_decomposition(X)
        self.Ag = Ag
        self.Ac = Ac

        if self.normalize:
            self.H = H / np.linalg.norm(H, axis=1)[:, None]
        else:
            self.H = H

        centroids, labels, *_ = k_means(X=self.H, n_clusters=self.k)

        self.centroids = centroids
        self.labels = labels

        return labels


def _transform_data(
    data: np.ndarray,
    settings: _settings.ClusteringSettings
):
    """
    Transforms the data using the wavelet transform settings
    """
    settings = settings.transform_settings

    def _dwt_coeffs(data, wavelet="db1", wavelet_mode="periodization", n_levels=4):
        dwt_max_level = np.inf
        for row in range(len(data)):
            # get max level of decomposition
            dwt_max_level_i = pywt.dwt_max_level(data[row].shape[0], wavelet)
            if dwt_max_level_i < dwt_max_level:
                dwt_max_level = dwt_max_level_i

        if n_levels > dwt_max_level:
            n_levels = dwt_max_level
        
        all_features = []
        # iterate through rows of numpy array
        for row in range(len(data)):            
            decomp_coeffs = pywt.wavedec(
                data[row], wavelet=wavelet, mode=wavelet_mode, level=n_levels
            )

            # decomp_coeffs = np.hstack(decomp_coeffs[:-2])
            decomp_coeffs = decomp_coeffs[0]

            all_features.append(decomp_coeffs)

        return np.vstack(all_features)

    def _pca_coeffs(features, method, min_var_ratio_explained=0.95, n_components=None):
        if min_var_ratio_explained is not None:
            n_components = min_var_ratio_explained

        # kernel pca is not fully developed
        if method == "kernel_pca":
            if n_components ==  "mle":
                pca = PCA(n_components=n_components)
                pca_features = pca.fit_transform(features)

            pca = KernelPCA(n_components=None, kernel="rbf")
            pca_features = pca.fit_transform(features)

            if min_var_ratio_explained is not None:
                explained_variance_ratio = pca.eigenvalues_ / np.sum(pca.eigenvalues_)

                # get the cumulative explained variance ratio
                cumulative_explained_variance = np.cumsum(explained_variance_ratio)

                # find number of components that explain pct% of the variance
                n_components = np.argmax(cumulative_explained_variance > n_components).astype(int)

            if not isinstance(n_components, (int, np.integer)):
                raise ValueError("n_components must be an integer for kernel PCA")

            # pca = PCA(n_components=n_components)
            pca = KernelPCA(n_components=n_components, kernel="rbf")
            pca_features = pca.fit_transform(features)

        else:
            pca = PCA(
                n_components=n_components,
                random_state=settings._seed,
            )
            pca_features = pca.fit_transform(features)

        return pca_features

    # calculate wavelet coefficients
    with warnings.catch_warnings():
        features = _dwt_coeffs(
            data, 
            settings.wavelet_name, 
            settings.wavelet_mode, 
            settings.wavelet_n_levels
        )

    features = _pca_coeffs(
        features,
        settings.pca_method,
        settings.pca_min_variance_ratio_explained,
        settings.pca_n_components,
    )

    if settings.include_scale_feature:
        agg = "median"

        if agg == "mean":
            data_scale = np.mean(data, axis=1)[:, None]

            mean = np.mean(data_scale)
            std = np.std(data_scale)
            data_scale = (data_scale - mean) / std

        elif agg == "median":
            data_scale = np.median(data, axis=1)[:, None]

            # normalize data_median to pca_features percentiles
            percentile = 5

            data_min = np.percentile(data_scale, percentile)
            data_max = np.percentile(data_scale, 100 - percentile)
            data_range = data_max - data_min

            pca_min = np.percentile(features, percentile)
            pca_max = np.percentile(features, 100 - percentile)
            pca_range = pca_max - pca_min
            
            data_scale = pca_range*(data_scale - data_min) / data_range + pca_min

        features = np.hstack([features, data_scale])

    # rescale = settings._rescale
    rescale = False
    if rescale:
        features = (features - features.mean()) / features.std()
        # mean = np.mean(features, axis=0)
        # std = np.std(features, axis=0)
        # features = (features - mean) / std

        # normalize using median and mad
        # median = np.median(features, axis=0)
        # mad = MAD(features, median, axis=0)
        # features = (features - median) / mad

        # median = np.median(features)
        # mad = MAD(features, median)
        # features = (features - median) / mad

        # Rescale features
        # min_new = -1
        # max_new = 1
        # new_range = max_new - min_new
        # min_base = np.min(features)
        # max_base = np.max(features)
        # base_range = max_base - min_base
        # features = new_range * (features - min_base) / base_range + min_new

    return features


def _cluster_merge(
    cluster_labels: np.ndarray,
    data: np.ndarray,
    settings: _settings.ClusteringSettings,
    W: float = 0.5,
):
    
    # get unique labels
    unique_labels = np.unique(cluster_labels)

    # get the distance between all rows in data
    distances = cdist(data, data)

    intra_cluster_similarity = np.zeros(len(unique_labels))
    inter_cluster_similarity = np.zeros((len(unique_labels), len(unique_labels)))
    for i in range(len(unique_labels)):
        idx_i = np.where(cluster_labels == unique_labels[i])[0]
        for j in range(len(unique_labels)):
            idx_j = np.where(cluster_labels == unique_labels[j])[0]

            if i == j:
                intra_cluster_similarity[i] = np.mean(distances[idx_i, :][:, idx_i])
                inter_cluster_similarity[i, j] = 0
                continue
            elif i < j:
                continue

            inter_cluster_similarity[i, j] = np.sum(distances[idx_i, :][:, idx_j])
            inter_cluster_similarity[j, i] = np.nan

    # if there are only two clusters, merge them if the similarity is less than W
    if unique_labels.shape[0] == 2:
        cluster_similarity = inter_cluster_similarity[0, 1]
        mean_similarity = np.mean(distances[distances != 0])

        ratio = cluster_similarity / mean_similarity

        if ratio < W:
            return np.zeros(len(cluster_labels))
        
        return cluster_labels

    # if there are more than two clusters, merge them if the similarity is less than W
    mean_similarity = np.mean(inter_cluster_similarity)

    for i in reversed(range(len(unique_labels))):
        for j in reversed(range(len(unique_labels))):
            if i == j:
                continue

            ratio = inter_cluster_similarity[i, j] / mean_similarity

            if ratio < W:
                cluster_labels[cluster_labels == unique_labels[j]] = unique_labels[i]

    return cluster_labels


def cluster_reorder(
    data: pd.DataFrame, 
    cluster_labels: np.ndarray,
    settings: _settings.ClusteringSettings,
):
    sort_method = settings.cluster_sort.method
    agg_type = settings.cluster_sort.aggregation
    reverse = settings.cluster_sort.reverse

    # assign labels to data
    df = data.copy()
    df["label"] = cluster_labels
    n_clusters = len(np.unique(cluster_labels))

    # exclude label -1 (outlier) from reordering
    if -1 in df['label'].unique():
        df = df[df['label'] != -1]
        n_clusters = n_clusters - 1

    if sort_method == "size":
        # sort clusters by count
        cluster_size = df['label'].value_counts()
        cluster_size = cluster_size.sort_values()

        features = cluster_size

    elif sort_method == "peak":
        # TODO: This is a work in progress

        # group by cluster and aggregate
        df_cluster = df.groupby('label').agg(agg_type)

        # subtract each cluster's median from the cluster's median
        df_cluster_norm = df_cluster.sub(df_cluster.agg(agg_type, axis=1), axis=0)
        cluster_max = df_cluster_norm.abs().max().max()
        df_cluster_norm = df_cluster_norm/cluster_max

        # define threshold for peak and valley
        threshold = np.quantile(abs(df_cluster.values), 0.75)

        # find peaks and valleys
        peak = {}
        valley = {}
        norm = {}
        for i in range(n_clusters):
            cluster_normal = df_cluster.iloc[i]
            norm[i] = cluster_normal.agg(agg_type)
            df_cluster_norm = cluster_normal - norm[i]
            thresh = threshold - norm[i]

            peak[i] = find_peaks(df_cluster_norm.values, height=thresh, width=1)[0]
            valley[i] = find_peaks(-df_cluster_norm.values, height=thresh, width=1)[0]

            if len(peak[i]) == 0:
                peak[i] = None
            else:
                peak[i] = peak[i][0]

            if len(valley[i]) == 0:
                valley[i] = None
            else:
                valley[i] = valley[i][0]
        
        # create df with peak and valley
        features = pd.DataFrame({'peak': peak, 'valley': valley, "norm": norm})

        features = features.sort_values(by=["peak", "valley", "norm"], na_position='first')

    # create dictionary to remap cluster numbers to features order
    cluster_map = {i: i for i in cluster_labels}

    if not reverse:
        cluster_map.update({features.index[i]: i for i in range(n_clusters)})
    else:
        cluster_map.update({features.index[i]: i for i in range(n_clusters)[::-1]})

    return cluster_map


def cluster_features(
    df: pd.DataFrame,
    settings: _settings.ClusteringSettings,
) -> np.ndarray:

    # adjust upper cluster count if necessary
    if settings.algorithm_selection not in ["dbscan", "hdbscan"]:
        algo = getattr(settings, f"{settings.algorithm_selection.value}")
        if algo.n_cluster.lower >= len(data):
            return np.arange(len(data))

    # rescale the data by standardizing
    if settings.rescale:
        data = (data - data.mean()) / data.std()

        if data_count < min_required_data:
            settings_dict = settings.model_dump()
            settings_dict[algo]["n_cluster"]["upper"] = data_count // min_cluster_size
            settings = _settings.ClusteringSettings(**settings_dict)
    
    # cluster the pca features
    if settings.algorithm_selection == "bisecting_kmeans":
        cluster_fcn = _bisecting_kmeans_clustering
    elif settings.algorithm_selection == "birch":
        cluster_fcn = _birch_clustering
    elif settings.algorithm_selection == "dbscan":
        cluster_fcn = _dbscan_clustering
    elif settings.algorithm_selection == "hdbscan":
        cluster_fcn = _hdbscan_clustering
    elif settings.algorithm_selection == "spectral":
        cluster_fcn = _spectral_clustering
    else:
        raise ValueError(f"Unknown clustering algorithm: {settings.algorithm_selection}")
    
    cluster_labels, df_votes = cluster_fcn(data, settings) # TODO: remove df_votes from return in future

    if np.unique(cluster_labels).shape[0] == 2:
        cluster_labels = _cluster_merge(cluster_labels, data, settings)

    if settings.cluster_sort.enable:
        cluster_remap_dict = cluster_reorder(df, cluster_labels, settings)

        # remap cluster labels using cluster_remap_dict
        cluster_labels = np.vectorize(cluster_remap_dict.get)(cluster_labels)

    return cluster_labels, df_votes