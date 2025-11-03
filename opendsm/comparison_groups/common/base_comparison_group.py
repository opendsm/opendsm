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

import matplotlib as mpl
import matplotlib.pyplot as plt


class Comparison_Group_Algorithm:
    settings = None
    _loadshape_aggregation = "mean"

    treatment_ids = None
    treatment_loadshape = None
    treatment_match_loadshape = None
    comparison_pool_loadshape = None

    clusters = None
    treatment_weights = None
    

    def _get_treatment_loadshape(self, id):
        ls = self.treatment_loadshape.loc[id]

        agg_ls = ls.agg(self._loadshape_aggregation).to_frame().T
        
        if len(id) == 1:
            agg_ls.index = ['Treatment Meter']
        else:
            agg_ls.index = ['Treatment Group']

        return agg_ls

    def _set_treatment_match_loadshape(self):
        pool_ls = self.comparison_pool_loadshape
        cluster_ls = self.clusters[["cluster"]].join(pool_ls)
        cluster_ls = cluster_ls.groupby("cluster").agg(self._loadshape_aggregation)

        agg_ls = np.einsum("ij,ik->jk", cluster_ls.loc[0:], self.treatment_weights.T).T
        agg_ls = pd.DataFrame(agg_ls, columns=pool_ls.columns, index=self.treatment_weights.index)

        return agg_ls
    
    def _get_treatment_match_loadshape(self, id):
        if self.treatment_match_loadshape is None:
            self.treatment_match_loadshape = self._set_treatment_match_loadshape()

        ls = self.treatment_match_loadshape.loc[id]

        agg_ls = ls.agg(self._loadshape_aggregation).to_frame().T
        agg_ls.index = ['Comparison Group']

        return agg_ls

    def get_comparison_pool_loadshape(self):
        ls = self.comparison_pool_loadshape

        agg_ls = ls.agg(self._loadshape_aggregation).to_frame().T
        agg_ls.index = ['Comparison Pool']

        return agg_ls
    
    def get_loadshapes(self, id=None):
        if id is None:
            id = self.treatment_data.ids
        if not isinstance(id, (list, np.ndarray, pd.Series)):
            id = [id]

        treatment_ls = self._get_treatment_loadshape(id)
        treatment_match_ls = self._get_treatment_match_loadshape(id)
        comparison_pool_ls = self.get_comparison_pool_loadshape()

        # concat ls
        ls = pd.concat([treatment_ls, treatment_match_ls, comparison_pool_ls])
        ls.columns = [int(col) - 1 for col in ls.columns]

        return ls
    
    def _validate_ls_weights(self, weights):
        if weights is None:
            return
        
        # if weights are all the same then return
        if len(set(weights)) == 1:
            return

        if len(weights) != len(self.treatment_loadshape.iloc[0].values):
            raise ValueError("weights must be the same length as the number of columns in the treatment group and comparison pool")
        
        # normalize weights to 1
        weights = np.array(weights) / np.sum(weights)

        return weights

    def plot_loadshapes(self, id=None):
        ls = self.get_loadshapes(id=id)

        t_min = ls.T.index[0]
        t_max = ls.T.index[-1]

        # plot ls
        fig = plt.figure(figsize=(14, 4), dpi=300)
        ax = fig.subplots()

        for col in ls.T.columns:
            ax.plot(ls.T.index, ls.T[col], label=col)

        if (t_max - t_min) % 24 and (t_max - t_min) > 24:
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
            ax.set_xticks(np.arange(t_min, t_max, 24))

        ax.set_xlim([t_min, t_max])
        ax.set_xlabel('Time')
        ax.set_ylabel('Loadshape')
        ax.legend()

        plt.close(fig) # prevent displaying immediately

        return fig