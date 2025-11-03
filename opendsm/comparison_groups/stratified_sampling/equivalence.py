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

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import chisquare


def ids_to_index(subset_ids, all_ids):
    """Convert an array of ids to an array of indexes relative to a superset of ids."""
    
    df_1 = pd.DataFrame({'a': subset_ids}).reset_index()
    df_2 = pd.DataFrame({'a': all_ids}).reset_index().rename(columns={'index': 'x'})

    df_out = df_1.merge(df_2)
    diff = len(df_1) - len(df_out)
    if diff > 0:
        raise ValueError(f"{diff} IDs present in subset are missing in pool")

    return df_out.x.values


class Equivalence:
    """ Computes equivalence between two sets of features, by cutting into 
    quantiles, computing distance between each quantile, and summing distances.

    Parameters:
    ------------

    ix_x: List or array
        Array of indices which map to the first row-set of features in features_matrix.
    ix_y: List or array
        Array of indices which map to the second row-set of features in features_matrix.
    features_matrix: pd.DataFrame or numpy.ndarray
        Dataframe or array of features, one row per item, one column per feature.
    n_quantiles: int
        Number of quantiles to cut eachset into.
    how: str
        Distance metric, either 'euclidean' or 'chisquare'


    """
    def __init__(self, ix_x, ix_y, features_matrix, n_quantiles=1, how='euclidean'):
        self.ix_x = ix_x
        self.ix_y = ix_y
        self.n_quantiles = n_quantiles
        self.how = how

        if type(features_matrix) == pd.DataFrame:
            features_matrix = features_matrix.to_numpy() # pragma: no cover
        elif type(features_matrix) == np.ndarray:
            pass
        else:
            raise ValueError("features_matrix must be a pandas DataFrame or numpy ndarray.") # pragma: no cover

        self.features_matrix = features_matrix
        self.X = self.features_matrix[ix_x].transpose()
        self.Y = self.features_matrix[ix_y].transpose()


    def compute(self):
        means_x, means_y, quantiles_x, quantiles_y = quantile_means_population(self.X, self.Y, self.n_quantiles)
        distance = sum_column_distance(means_x, means_y, how=self.how)
        equiv_x = reshape_outputs(means_x, quantiles_x)
        equiv_y = reshape_outputs(means_y, quantiles_y)
        return equiv_x, equiv_y, distance



def reshape_outputs(means, quantiles):
    out = []
    for feature in range(len(means)):
        for q in range(len(quantiles[0])-1):
            bin_label = f"[{quantiles[feature][q]}, {quantiles[feature][q+1]}]" 
            mean = means[feature][q]
            out.append({'_bin_label': bin_label, 'value': mean, 'feature_index': feature})
    return pd.DataFrame(out)



def get_quantile_indexes(n_quantiles):
    return np.linspace(0, 1, n_quantiles + 1)

def get_quantiles(col, n_quantiles):
    return np.quantile(col, get_quantile_indexes(n_quantiles))

def cut_column(col, q_this, q_next):
    # return slice of an array between q_this and q_next inclusive
    # inclusive means we include 0th and 100th percentiles, at 
    # the expense of possible duplication of middle quantiles 
    return col[(col >= q_this) & (col <= q_next)]

def quantile_means_array(col, n_quantiles):
    # slice an array into n quantiles and compute the mean value for each 
    if type(col) != np.ndarray:
        col = np.array(col)
    quantiles = get_quantiles(col, n_quantiles)
    means = np.ndarray(n_quantiles)
    for i in range(len(quantiles) - 1):
        means[i] = np.mean(cut_column(col, quantiles[i], quantiles[i+1]))
    return means, quantiles


def quantile_means_population(X, Y, n_quantiles):
    # compute means per quantile, for each column in X and Y
    n_cols = len(X)
    if not len(X) == len(Y):
        raise ValueError("Matrices must have the same number of columns.") # pragma: no cover

    means_x = np.ndarray((n_cols, n_quantiles))
    means_y = np.ndarray((n_cols, n_quantiles))
    quantiles_x = np.ndarray((n_cols, n_quantiles + 1))
    quantiles_y = np.ndarray((n_cols, n_quantiles + 1))

    for i in range(n_cols):
        col_x = X[i]
        col_y = Y[i]
        means_x[i], quantiles_x[i] = quantile_means_array(col_x, n_quantiles)
        means_y[i], quantiles_y[i] = quantile_means_array(col_y, n_quantiles)

    return means_x, means_y, quantiles_x, quantiles_y

def chisquare_dist(X,Y):
    distance = 0
    for i in range(len(X)):
        distance = distance + ((X[i] - Y[i])**2 / (X[i] + Y[i]))
    return distance 

def get_distance_func(how="euclidean"):
    if how == "euclidean":
        return lambda x, y: pdist([x,y])[0]
    elif how == "chisquare":
        return chisquare_dist 
    else:
        raise ValueError(f"Unsupported distance metric: {how}") # pragma: no cover


def sum_column_distance(means_x, means_y, how="euclidean"):
    column_distances = np.ndarray(len(means_x))
    distance_func = get_distance_func(how)
    for i in range(len(means_x)):
        column_distances[i] = distance_func(means_x[i], means_y[i])
    return np.sum(column_distances)



