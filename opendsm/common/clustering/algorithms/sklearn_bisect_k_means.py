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

from copy import deepcopy as copy

import numpy as np
import scipy.sparse as sp

from sklearn.cluster import BisectingKMeans as _sklearn_BisectingKMeans
from sklearn.cluster import _bisect_k_means
from sklearn.cluster._kmeans import (
    _kmeans_single_elkan,
    _kmeans_single_lloyd,
) # type: ignore
from sklearn.cluster._k_means_common import (
    _inertia_dense,
    _inertia_sparse,
) # type: ignore
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import (
    _check_sample_weight,
    check_random_state,
) # type: ignore
try:
    from sklearn.utils.validation import validate_data # type: ignore
except ImportError:
    validate_data = None  # type: ignore
# from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BisectingKMeans(_sklearn_BisectingKMeans):
    """
    Override of sklearn class which simply saves the labels
    of all intermediate cluster steps.

    Only overrides fit

    Should always take the upper bound of number of clusters to try.
    Contains a new property named labels_full which is a dictionary where the key is the number of clusters
    and the value is the labels using that number.

    This should be used to score all the labels and determine the best number/labels to use/


    """

    def fit(self, X, y=None, sample_weight=None):
        """Compute bisecting k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)

            Training instances to cluster.

            .. note:: The data will be converted to C ordering,
                which will cause a memory copy
                if the given data is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self
            Fitted estimator.
        """
        # return self._fit_test(X, y, sample_weight)

        self._validate_params()  # type: ignore

        if validate_data is not None:
            X = validate_data(  # type: ignore
                self,
                X,
                accept_sparse="csr",
                dtype=[np.float64, np.float32],
                order="C",
                copy=self.copy_x,  # type: ignore
                accept_large_sparse=False,
            )
        else:
            X = self._validate_data(  # type: ignore
                X,
                accept_sparse="csr",
                dtype=[np.float64, np.float32],
                order="C",
                copy=self.copy_x,  # type: ignore
                accept_large_sparse=False,
            )

        self._check_params_vs_input(X)  # type: ignore

        self._random_state = check_random_state(self.random_state)  # type: ignore
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        # self._n_threads = _openmp_effective_n_threads()
        self._n_threads = 1  # OVERRIDE OF ABOVE SO THAT RESULTS ARE DETERMINISTIC

        if self.algorithm == "lloyd" or self.n_clusters == 1:  # type: ignore
            self._kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])  # type: ignore
        else:
            self._kmeans_single = _kmeans_single_elkan

        # Subtract of mean of X for more accurate distance computations
        if not sp.issparse(X):
            self._X_mean = X.mean(axis=0)
            X -= self._X_mean

        # Initialize the hierarchical clusters tree
        self._bisecting_tree = _bisect_k_means._BisectingTree(
            indices=np.arange(X.shape[0]),
            center=X.mean(axis=0),
            score=0,
        )

        x_squared_norms = row_norms(X, squared=True)
        self.labels_full = {}
        for i in range(self.n_clusters - 1):  # type: ignore
            # Chose cluster to bisect
            try:
                cluster_to_bisect = self._bisecting_tree.get_cluster_to_bisect()
            except RecursionError:
                logger.warn(
                    f"encountered Recursion error during bisection for cluster size {i + 2}. Returning early"
                )
                return self

            # Split this cluster into 2 subclusters
            try:
                self._bisect(X, x_squared_norms, sample_weight, cluster_to_bisect)  # type: ignore
            except IndexError:
                logger.warn(
                    f"encountered IndexError during bisection for cluster size {i + 2}"
                )
                return self  # return early so that calculated labels can be returned until an error arose

            # Aggregate final labels and centers from the bisecting tree
            labels = np.full(X.shape[0], -1, dtype=np.int32)

            for j, cluster_node in enumerate(self._bisecting_tree.iter_leaves()):
                labels[cluster_node.indices] = j  # type: ignore

            self.labels_full[i + 2] = copy(labels)

        # Aggregate final labels and centers from the bisecting tree
        self.labels_ = np.full(X.shape[0], -1, dtype=np.int32)
        self.cluster_centers_ = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)  # type: ignore

        for i, cluster_node in enumerate(self._bisecting_tree.iter_leaves()):
            self.labels_[cluster_node.indices] = i  # type: ignore
            self.cluster_centers_[i] = cluster_node.center  # type: ignore
            cluster_node.label = i  # type: ignore
            cluster_node.indices = None  # type: ignore

        # Restore original data
        if not sp.issparse(X):
            X += self._X_mean
            self.cluster_centers_ += self._X_mean

        _inertia = _inertia_sparse if sp.issparse(X) else _inertia_dense
        self.inertia_ = _inertia(
            X, sample_weight, self.cluster_centers_, self.labels_, self._n_threads
        )

        self._n_features_out = self.cluster_centers_.shape[0]

        return self
