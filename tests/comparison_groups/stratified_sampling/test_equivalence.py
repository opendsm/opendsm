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

import pytest

import numpy as np
import pandas as pd

from opendsm.comparison_groups.stratified_sampling.equivalence import *



@pytest.fixture
def equiv_X():
    # 3 columns, 6 rows, column first 
    return np.array(
            [[1,1,1,10,10,10],
            [1,1,1,10,10,10],
            [1,1,1,10,10,10]])


@pytest.fixture
def equiv_Y():
    # 3 columns, 6 rows, column first 
    return np.array(
            [[0,1,2,3,4,5],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            ])


@pytest.fixture
def feature_matrix(equiv_X, equiv_Y):
    x = equiv_X
    y = equiv_Y
    return np.concatenate([x.transpose(), y.transpose()])



def test_reshape_outputs():
    means = [[11,12], [21,22]]

    quantiles = [[11, 12, 13], 
                 [21, 22, 23]]

    df = pd.DataFrame({'_bin_label': ['[11, 12]', '[12, 13]', '[21, 22]', '[22, 23]'],
                  'value': [11, 12, 21, 22], 'feature_index': [0,0,1,1]})
    assert df.equals(reshape_outputs(means, quantiles))


def test_equivalene_distance(feature_matrix):

    eq = Equivalence(ix_x = [0,1,2,3,4,5], ix_y = [6,7,8,9,10,11], 
        features_matrix = feature_matrix, n_quantiles=2, how="euclidean")
    equiv_x, equiv_y, distance = eq.compute()
    assert round(distance,2) == round(6 + 2*101**0.5,2)


    eq = Equivalence(ix_x = [0,1,2,3,4,5], ix_y = [6,7,8,9,10,11], 
        features_matrix = feature_matrix, n_quantiles=2, how="chisquare")
    equiv_x, equiv_y, distance = eq.compute()
    assert round(distance,2) == round(36/14 + 2*11,2)




def test_get_quantiles():
    assert (get_quantile_indexes(1) == [0, 1]).all()
    assert (get_quantile_indexes(2) == [0, 0.5, 1]).all()
    assert (get_quantile_indexes(3) == [0, 1/3, 2/3, 1]).all()




def test_quantile_means_array(equiv_X, equiv_Y):
    x = [0,1,2,3,4,5]
    means, quantiles = quantile_means_array(x, n_quantiles=1)
    assert (means == np.mean(x)).all()
    means, quantiles = quantile_means_array(x, n_quantiles=len(x))
    assert (means == x).all()

    x = [1,1,1,10,10,10]
    means, quantiles = quantile_means_array(x, n_quantiles=2)
    assert (means == [1, 10]).all()

    means_x, means_y, quantiles_x, quantiles_y = quantile_means_population(equiv_X, equiv_Y, 2)
    assert (means_x == np.array([[1,10], [1,10], [1,10]])).all()
    assert (means_y == np.array([[1,4], [0,0], [0,0]])).all()


def test_quantile_distance(equiv_X, equiv_Y):
    means_x, means_y, quantiles_x, quantiles_y = quantile_means_population(equiv_X, equiv_Y, 2)
    assert round(sum_column_distance(means_x, means_y, "euclidean"),2) == round(6 + 2*101**0.5,2)
    assert round(sum_column_distance(means_x, means_y, "chisquare"),2) == round(36/14 + 2*11,2)


def test_equivalence_inputs(feature_matrix):

    eq = Equivalence(ix_x = [1,2], ix_y = [3, 10, 11], features_matrix = feature_matrix)    
    # X should be [1,1,1] and [1,1,1]
    # eq.X should be sliced by column so [1,1], [1,1], [1,1]
    # Y should be [10, 10, 10], [4, 0, 0], [5, 0, 0]
    # eq.Y should be [10, 4, 5], [10, 0, 0], [10, 0 0]
    assert (eq.X == np.array([[1,1], [1,1], [1,1]])).all()
    assert (eq.Y == np.array([[10,4,5], [10,0,0], [10,0,0]])).all()





def test_index_to_ids():
    t = np.array([11,17,15])
    a = np.array([11,12,13,15,16,17])
    assert (np.array(ids_to_index(t,a) == np.array([0,5,3]))).all()

    t1 = np.array(["11","17","15"])
    a1 = np.array(["11","12","13","15","16","17"])
    assert (np.array(ids_to_index(t1,a1) == np.array([0,5,3]))).all()

    with pytest.raises(ValueError):
        t2 = np.array([98123])
        ids_to_index(t2,a)




