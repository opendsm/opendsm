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

import random

from functools import cached_property

from opendsm import eemeter as em
from opendsm import comparison_groups as cg

from opendsm.common.utils import unc_factor
from opendsm.common.utils import sigmoid


def get_t_cg_df(data, num_treatment=None, num_control=None, seed=21):
    def get_subpopulation(df, ids):
        df = df.reset_index()
        df = df[df["dpsm_id"].isin(ids)]
        df = df.rename(columns={"dpsm_id": "id", "start_local": "datetime", "model": "modeled"})

        period = ["baseline", "reporting"]
        df = df[df["period"].isin(period)]

        # remove datetimes after 1 year from first reporting period date
        first_reporting_date = df[df["period"] == "reporting"]["datetime"].min()
        df = df[df["datetime"] < first_reporting_date + pd.Timedelta(days=365)]
        
        return df
    
    # get list of ids
    id_list = list(data.df["meter"].index.unique())

    random.seed(seed)
    if num_treatment is None:
        # assign treatment ids in same proportion as original cluster research (1000 cp, 100 t)
        num_treatment = int(round(100*(len(id_list)/1100)))

    treatment_ids = random.sample(id_list, num_treatment)
    pool_ids = [x for x in id_list if x not in treatment_ids]

    if num_control is not None:
        num_control = int(num_control*len(treatment_ids))
        if num_control <= len(pool_ids):         
            pool_ids = random.sample(pool_ids, num_control)

    # get treatment and pool dataframes
    df_t = get_subpopulation(data.df["meter"], treatment_ids)
    df_cp = get_subpopulation(data.df["meter"], pool_ids)

    return df_t, df_cp


def get_comparison_groups(df_t, df_cp, agg, cg_type="cluster", multiprocessing=True):
    # set data classes
    data_settings = gm.Data_Settings(AGG_TYPE=agg, LOADSHAPE_TYPE="modeled")

    data_cls = {
        "t":  gm.Data(time_series_df=df_t[df_t["period"] == "baseline"], settings=data_settings), 
        "cp": gm.Data(time_series_df=df_cp[df_cp["period"] == "baseline"], settings=data_settings),
    }

    if "cluster" in cg_type.lower():
        # get clustered comparison groups
        clustering_settings = gm.Clustering_Settings(USE_MULTIPROCESSING=multiprocessing)
        clustering = gm.Clustering(clustering_settings)
        return clustering.get_comparison_group(data_cls["t"], data_cls["cp"])

    elif "imm" in cg_type.lower():
        # get IMM comparison groups
        imm_settings = gm.IMM_Settings(USE_MULTIPROCESSING=multiprocessing)
        imm = gm.IMM(imm_settings)
        return imm.get_comparison_group(data_cls["t"], data_cls["cp"])
    
    else:
        raise ValueError("cg_type must be either 'cluster' or 'imm'")


def add_datetime_loadshape_mapping_col(df, data_settings):
    # get mapping between datetime and loadshape
    if data_settings.TIME_PERIOD != 'seasonal_hourly_day_of_week':
        raise ValueError("This only works for seasonal_hourly_day_of_week")

    df_key = pd.DataFrame({"datetime": df["datetime"].unique()})
    df_key["month"] = df_key["datetime"].dt.month

    # map month to season using _NUM_DICT
    df_key["season"] = df_key["month"].map(data_settings.SEASON._NUM_DICT)
    df_key["season_num"] = df_key["season"].map(data_settings.SEASON._ORDER)

    df_key["hour_of_week"] = df_key["datetime"].dt.dayofweek*24 + df_key["datetime"].dt.hour

    df_key["ls_key"] = df_key["season_num"]*24*7 + df_key["hour_of_week"] + 1

    df_key = df_key.set_index("datetime")
    df_key = df_key["ls_key"]

    # merge df_t_cg and df_dt_ls_key on datetime
    df = df.merge(df_key, left_on="datetime", right_index=True)

    return df

# TODO: need to cap IMM size if doing this for memory reasons
class Savings:
    def __init__(self, df_t, df_cp, df_cluster_id, df_t_coeffs, agg_type="mean", reject_outliers=False, scale_diff=True):
        self.df_t = df_t
        self.df_cp = df_cp
        self.df_cluster_id = df_cluster_id
        self.df_t_coeffs = df_t_coeffs

        self.agg_type = agg_type
        self.reject_outliers = reject_outliers
        self.scale_diff = scale_diff

        self.data_settings = gm.Data_Settings(AGG_TYPE=agg_type, LOADSHAPE_TYPE="modeled")

        # calculate diffs for df_t and df_cp
        self.df_t = self._initialize_df(self.df_t, is_treatment=True)
        self.df_cp = self._initialize_df(self.df_cp, is_treatment=False)

        self.df_cluster = self._agg_cluster_data()
        self._df_t_cg = self._get_treatment_cg_data()

    
    def _initialize_df(self, df, is_treatment=False):
        data_settings = self.data_settings
        
        df["ratio"] = df["observed"]/df["modeled"]
        df["diff"] = df["modeled"] - df["observed"]

        df = add_datetime_loadshape_mapping_col(df, data_settings)

        if not is_treatment and self.scale_diff:
            period = "baseline"
            groupby_keys = ["id", "ls_key"]
            # groupby_keys = ["id"]

            df_p = df[df["period"] == period]
            
            df_p_grouped = df_p.groupby(groupby_keys)
            for col in ["diff"]:
                # calculate IQR
                scale = df_p_grouped[col].quantile(0.75) - df_p_grouped[col].quantile(0.25)
                scale = scale.rename(f"{col}_scale")

                # add to df
                df = df.merge(scale, left_on=groupby_keys, right_index=True)

                df[col] /= df[f"{col}_scale"]

        # get columns to aggregate on etc
        # TODO: remove unnecessary columns such as temperature, observed, modeled?
        cols_drop = ["id", "datetime", "period"]
        # cols_drop.extend([col for col in df.columns if col.endswith("_scale")])
        self.df_cols = [col for col in df.columns if col not in cols_drop]

        return df


    def _agg_cluster_data(self):
        df_cp = self.df_cp
        df_cluster_id = self.df_cluster_id
        agg_type = self.agg_type

        # get cluster data
        # merge df_cp_period with df_cg to get cluster number
        df_cp = df_cp.merge(df_cluster_id[["cluster"]], left_on="id", right_index=True)

        df_cols = [col for col in self.df_cols if col not in ["temperature", "ls_key"]]
        df_cp_groupby = df_cp[["cluster", "datetime", *df_cols]].groupby(["cluster", "datetime"])

        if self.reject_outliers:
            label_dict = {0.25: "Q1", 0.75: "Q3"}
            df_cp_iqr = df_cp_groupby.quantile([0.25, 0.75]).unstack()
            df_cp_iqr.columns = [f"{col}_{label_dict[q]}" for col, q in df_cp_iqr.columns]

            # join iqr data with original data
            df_cp = df_cp.merge(df_cp_iqr, on=["cluster", "datetime"])

            # get cluster data
            df_cluster = pd.concat(
                [df_cp[["cluster", "datetime", "ls_key"]].groupby(["cluster", "datetime"]).first(),
                 df_cp[["cluster", "datetime", "temperature"]].groupby(["cluster", "datetime"]).median()], 
                axis=1)

            for col in df_cols:
                Q1 = df_cp[f"{col}_Q1"]
                Q3 = df_cp[f"{col}_Q3"]
                IQR = Q3 - Q1

                temp = df_cp[["cluster", "datetime", col]]
                temp = temp[(temp[col] >= Q1 - 1.5*IQR) & (temp[col] <= Q3 + 1.5*IQR)]
                temp = temp[["cluster", "datetime", col]].groupby(["cluster", "datetime"]).median()

                df_cluster = pd.concat([df_cluster, temp], axis=1)

            df_cluster = df_cluster.reset_index()
            
        else:
            agg_dict = {col: agg_type for col in self.df_cols}

            df_cluster = df_cp.groupby(["cluster", "datetime"]).agg(agg_dict).reset_index()

        # get columns that end in _scale
        cols_scaled = [col.replace("_scale", "") for col in df_cluster.columns if col.endswith("_scale")]
        for col in cols_scaled:
            df_cluster[col] *= df_cluster[f"{col}_scale"]
        
        return df_cluster


    def _get_treatment_cg_data(self):
        df_t = self.df_t
        df_cluster = self.df_cluster
        df_t_coeffs = self.df_t_coeffs

        # rescale 
        # get comparison group data for each id
        df_cluster = df_cluster[df_cluster["cluster"] != -1]
        g = df_cluster.groupby('cluster', sort=False).cumcount()
        
        cluster_data = np.array(df_cluster.set_index(['cluster', g])[self.df_cols]
            .unstack(fill_value=1E30)    # replace any empty values with one
            .stack().groupby(level=0)
            .apply(lambda x: x.values.tolist())
            .tolist())

        t_coeffs = df_t_coeffs.values

        # multiplies each cluster by the percentage for each treatment meter and sums them per hour
        cg = {}
        for n, col in enumerate(self.df_cols):
            cg[col] = np.einsum("ij,ik->jk", cluster_data[:,:,n], t_coeffs.T).T

        t_datetime_contiguous = np.sort(df_t["datetime"].unique())
        cg_datetime_contiguous = np.sort(df_cluster["datetime"].unique())

        if np.all(t_datetime_contiguous != cg_datetime_contiguous):
            raise ValueError("Treatment and Comparison Group datetime arrays do not match")

        # repeat datetime array for each treatment meter
        cg_datetime = np.tile(cg_datetime_contiguous, cg["temperature"].shape[0])
        cg_ids = np.repeat(df_t_coeffs.index, cg["temperature"].shape[1])

        df_cg_dict = {"id": cg_ids, "datetime": cg_datetime}
        df_cg_dict.update({col: cg[col].flatten() for col in self.df_cols})
        
        df_cg = pd.DataFrame(df_cg_dict)

        df_cg["datetime"] = pd.to_datetime(df_cg["datetime"])

        # join df_t_period and df_cg_period on id and datetime
        df_t_cg = pd.merge(df_t, df_cg, on=["id", "datetime"], suffixes=["_t", "_cg"])

        return df_t_cg


    def add_pct_did(self, simplified_eqn=False):
        df = self._df_t_cg

        if simplified_eqn:
            cg_factor = df["ratio_cg"]
            res = cg_factor*df["modeled_t"] - df["observed_t"]

        else:
            res = df["diff_t"] - df["diff_cg"]*df["modeled_t"]/df["modeled_cg"]

        self._df_t_cg["%did"] = res


    def add_abs_pct_did(self, simplified_eqn=False):
        df = self._df_t_cg

        if simplified_eqn:
            res = np.empty(len(df))

            # get sign matching indices
            match = np.sign(df["modeled_t"]) == np.sign(df["modeled_cg"])          
            res[match] = df["ratio_cg"][match]*df["modeled_t"][match] - df["observed_t"][match]
            res[~match] = (2 - df["ratio_cg"][~match])*df["modeled_t"][~match] - df["observed_t"][~match]

        else:
            res = df["diff_t"] - df["diff_cg"]*(df["modeled_t"]/df["modeled_cg"]).abs()

        self._df_t_cg["abs_%did"] = res


    def add_sig_pct_did(self, k=0.01, m_0=0.1):
        df = self._df_t_cg

        if "abs_%did" not in df.columns:
            self.add_abs_pct_did(simplified_eqn=True)

        # df["scale"] = (df["abs_%did"] + df["observed_t"])/df["modeled_t"]

        scale = (
            ((df["modeled_t"] - df["observed_t"])*sigmoid(np.abs(df["modeled_t"]), m_0, k) + df["observed_t"]) / 
            ((df["modeled_cg"] - df["observed_cg"])*sigmoid(np.abs(df["modeled_cg"]), m_0, k) + df["observed_cg"])
        )

        # scale = (
        #     (df["diff_t"]*sigmoid(np.abs(df["modeled_t"]), m_0, k) + df["observed_t"]) / 
        #     (df["diff_cg"]*sigmoid(np.abs(df["modeled_cg"]), m_0, k) + df["observed_cg"])
        # )

        res = df["diff_t"] - df["diff_cg"]*np.abs(scale)

        self._df_t_cg["sig_%did"] = res
    

    def add_scaled_ordinary_did(self):
        # calculate scaled ordinary difference in differences

        df = self._df_t_cg
        cols = df.columns
        data_settings = self.data_settings

        comparison_col = "diff" # modeled or diff?

        comp_t = f"{comparison_col}_t"
        df_t_baseline = df[df["period"] == "baseline"][["id", "datetime", comp_t]]
        df_t_baseline = df_t_baseline.rename(columns={comp_t: "modeled"})
        data_t = gm.Data(time_series_df=df_t_baseline, settings=data_settings)

        comp_cg = f"{comparison_col}_cg"
        df_cg_baseline = df[df["period"] == "baseline"][["id", "datetime", comp_cg]]
        df_cg_baseline = df_cg_baseline.rename(columns={comp_cg: "modeled"})
        data_cg = gm.Data(time_series_df=df_cg_baseline, settings=data_settings)

        # scale based on loadshape in baseline period
        df_cg_scale = data_t.loadshape/data_cg.loadshape

        df_cg_scale = df_cg_scale.unstack().reset_index().rename(columns={"level_0": "ls_key", 0: "scale"})

        # merge df_cg_scale with df_t_cg on ls_key and id
        df = add_datetime_loadshape_mapping_col(df, data_settings)
        df = df.merge(df_cg_scale, on=["id", "ls_key"])

        df["sodid"] = df["diff_t"] - df["diff_cg"]*df["scale"]
        df = df.rename(columns={"scale": "sodid_scale"})

        # remove all columns except input and sodid columns
        self._df_t_cg = df[[*cols, "sodid", "sodid_scale"]]


    def add_modeled_scaled_ordinary_did(self):
        df_t_cg = self._df_t_cg

        df_t_cg["diff_ratio"] = df_t_cg["diff_t"]/df_t_cg["diff_cg"]

        df_ratio = df_t_cg[["id", "datetime", "period", "temperature_cg", "diff_ratio"]]
        df_ratio = df_ratio.rename(columns={"temperature_cg": "temperature", "diff_ratio": "observed"})

        ratio_modeled = []
        for id in df_ratio["id"].unique():
            df_ratio_id = df_ratio[df_ratio["id"] == id]
            df_ratio_id_baseline =  df_ratio_id[df_ratio_id["period"] == "baseline"][["datetime", "temperature", "observed"]]

            settings = em.HourlySettings()
            model = em.HourlyModel(settings)
            model.fit(df_ratio_id_baseline)

            df_predict = model.predict(df_ratio_id[["datetime", "temperature", "observed"]])
            df_predict = df_predict.reset_index()
            df_predict.insert(0, "id", id)

            ratio_modeled.append(df_predict)

        df_scale = pd.concat(ratio_modeled, ignore_index=True)

        # merge df_t_cg and df_scale on id and datetime
        df_t_cg["scale_predicted"] = df_scale["predicted"]

        # calculate model scale did
        res = df_t_cg["diff_t"] - df_t_cg["diff_cg"]*df_t_cg["scale_predicted"]

        self._df_t_cg["modeled_sodid"] = res


    def _get_did_cols(self):
        all_did_cols = ["%did", "abs_%did", "sig_%did", "sodid", "modeled_sodid"]
        did_cols = [col for col in all_did_cols if col in self._df_t_cg.columns]

        return did_cols

    @cached_property
    def df(self):
        # get which columns exist in %did, abs_%did sodid, modeled_sodid
        did_cols = self._get_did_cols()

        # if observed_t, observed_cg, modeled_t, or modeled_cg are nan, then did cols are nan
        df_t_cg = self._df_t_cg
        measured_cols = ["observed_t", "observed_cg", "modeled_t", "modeled_cg"]
        df_t_cg[did_cols] = df_t_cg[did_cols].where(
            ~df_t_cg[measured_cols].isna().any(axis=1),
            np.nan
        )

        # remove diff_t and diff_cg columns
        df_t_cg = df_t_cg.drop(columns=["diff_t", "diff_cg"])

        return df_t_cg
    

    def df_agg(self, period="reporting"):
        # TODO: This is only for testing new did methods
        self.add_pct_did(simplified_eqn=True)
        self.add_abs_pct_did(simplified_eqn=True)
        # self.add_sig_pct_did()
        self.add_scaled_ordinary_did()

        did_cols = self._get_did_cols()

        df_t_cg = self.df[self.df["period"] == period]

        # groupby id and aggregate observed, modeled and did_cols
        agg_dict = {
            "observed_t": "sum",
            "modeled_t": "sum",
            "observed_cg": "sum",
            "modeled_cg": "sum",
        }
        agg_dict.update({col: "sum" for col in did_cols})

        return df_t_cg.groupby("id").agg(agg_dict)


    def df_stats(self, period="reporting"):
        df_res = self.df_agg(period)

        # count number of unique ids
        id_count = len(df_res)

        # calculate mean and uncertainty of each did_column
        stats = {}
        for col in self._get_did_cols():
            mean = df_res[col].mean()
            unc = df_res[col].std()*unc_factor(id_count, alpha=0.05, interval="CI")

            stats.update({f"{col}": [mean], f"{col}_unc": [unc]})

        return pd.DataFrame(stats)