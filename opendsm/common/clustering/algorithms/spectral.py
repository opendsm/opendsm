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

from scipy.spatial.distance import pdist
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import pairwise_kernels

from opendsm.common.clustering import (
    scoring as _scoring,
    settings as _settings,
    voting as _voting,
)



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
        label_res = _scoring.score_clusters(data, labels, settings)

        np.random.set_state(np_state)
        
        results.append(label_res)

    df_votes = _voting.construct_voting_df(results)
    # df_votes.index = n_clusters_range

    # # drop single cluster from df_votes
    # df_votes = df_votes.drop(index=1, errors='ignore')

    winner_idx, df_votes = _voting.shulze_voting(
        df_votes, 
        _scoring.score_council(settings), 
        window_size,
        return_preference_df=True
        )
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


# defunct/experimental
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


def spectral(
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
        winner_idx = _voting.shulze_voting(
            df_votes, 
            _scoring.score_council(settings), 
            window_size=0
        )

    # get labels of winner from results
    winner_labels = results[winner_idx].labels
    # df_votes_recluster = df_votes_recluster[winner_idx]

    return winner_labels