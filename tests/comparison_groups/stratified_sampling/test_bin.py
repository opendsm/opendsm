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

from opendsm.comparison_groups.stratified_sampling.bins import Bin, MultiBin, BinnedData, Binning



def test_bin_filtering():
    this_bin = Bin("col", min=5, max=100, index=0)
    filter_expr = this_bin.filter_expr()

    df = pd.DataFrame({"col": [1, 5, 6, 100, 101]})
    df = df[filter_expr(df)]

    assert set(df["col"]) == set([5, 6, 100])


def test_binned_data_bin_label_label_leading_zeroes():
    col_name = 'c1'
    b1 = Bin(col_name, min=1, max=2, index=0)

    multi_bin = MultiBin(bins=[b1])
    df = pd.DataFrame({col_name: [1.5]})

    binning = Binning()
    binning.multibins = [multi_bin]

    binned_data = BinnedData(df, binning)
    mapped_bins = binned_data._map_bins(df)
    assert set(mapped_bins['_bin_label'].values) == set(['c1_000'])



'''

def test_multi_bin_filtering():
    b1 = Bin("c1", min=5, max=100, index=0)
    b2 = Bin("c2", min=50, max=500, index=1)

    mb = MultiBin(bins=[b1, b2])

    df = pd.DataFrame(
        [
            {"c1": 1, "c2": 1, "in": False},
            {"c1": 1, "c2": 100, "in": False},
            {"c1": 10, "c2": 1, "in": False},
            {"c1": 10, "c2": 100, "in": True},
            {"c1": 10, "c2": 1000, "in": False},
        ]
    )
    filter_expr = mb.filter_expr()
    df = df[filter_expr(df)]
    assert len(df) == 1
    assert df["in"].iloc[0] == True



def test_remove_bins_too_small():

    bins = [
    Bin("c1", min=0, max=10, index=0),
    Bin("c1", min=10, max=20, index=1),
    Bin("c1", min=20, max=30, index=2),
    Bin("c2", min=100, max=110, index=0),
    Bin("c2", min=110, max=120, index=1),
    Bin("c2", min=120, max=130, index=2),
    Bin("c2", min=130, max=140, index=3),
    ]

    mb = MultiBin(bins=bins)
'''
