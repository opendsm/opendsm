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

"""Generate real model_correction snapshot inputs from ComStock meters.

Run once (JIT-on) via the guarded ``test_generate_correction_fixtures`` test to
produce committed ``.npz`` fixtures consumed by ``test_model_correction.py``;
NOT part of the normal test run.  ``build_fixtures`` is driven through pytest so
it shares the suite's ComStock data loading.  The comparison-group clustering is
derived once from hourly load shapes (a property of the meters' profiles) and
reused across granularities; only the observed/modelled reporting-period totals
differ per model.
"""

import pathlib

import numpy as np

from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const
from opendsm.comparison_groups.cg_clustering.create_comparison_groups import CG_Clustering
from opendsm.comparison_groups.cg_clustering.settings import CG_Clustering_Settings
from opendsm.eemeter import (
    DailyModel, DailyBaselineData, DailyReportingData,
    HourlyModel, HourlyBaselineData, HourlyReportingData,
    BillingModel, BillingBaselineData, BillingReportingData,
)



FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"

_MODELS = {
    "hourly": (HourlyModel, HourlyBaselineData, HourlyReportingData),
    "daily": (DailyModel, DailyBaselineData, DailyReportingData),
    "billing": (BillingModel, BillingBaselineData, BillingReportingData),
}


def _loadshape_data(df_baseline, meter_ids):
    """24-hour observed load-shape Data for the given meters (for clustering)."""
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )
    time_series = df_baseline.loc[meter_ids].reset_index()[["id", "datetime", "observed"]]
    data = Data(time_series_df=time_series, settings=settings)

    return data


def _cluster_pool(df_hourly_baseline, treatment_id, pool_ids, min_cluster_size):
    """Cluster the comparison pool and fit treatment weights on hourly shapes.

    A smaller min_cluster_size yields more, smaller clusters (and lets small
    pools cluster at all).  Returns (labels_by_id: dict, weight_by_cluster:
    dict) where weight_by_cluster maps each non-negative cluster label to its
    treatment weight.
    """
    treatment_data = _loadshape_data(df_hourly_baseline, [treatment_id])
    pool_data = _loadshape_data(df_hourly_baseline, pool_ids)

    clustering = CG_Clustering(CG_Clustering_Settings(min_cluster_size=min_cluster_size))
    df_cg, df_weights = clustering.get_comparison_group(treatment_data, pool_data)

    labels_by_id = {str(i): int(c) for i, c in df_cg["cluster"].items()}
    weights = np.asarray(df_weights).flatten()
    sorted_clusters = sorted(c for c in set(labels_by_id.values()) if c >= 0)
    weight_by_cluster = {c: float(weights[i]) for i, c in enumerate(sorted_clusters)}

    return labels_by_id, weight_by_cluster


def _model_reporting_totals(model_cls, baseline_cls, reporting_cls, df_b, df_r, meter_id):
    """Fit a model on one meter's baseline, predict reporting, return totals.

    Returns (observed_sum, modelled_sum, modelled_unc) or None if the fit or
    prediction fails or produces non-finite totals.
    """
    try:
        b = df_b.xs(meter_id, level="id").reset_index()
        r = df_r.xs(meter_id, level="id").reset_index()
        model = model_cls().fit(baseline_cls(b, is_electricity_data=True), ignore_disqualification=True)
        pred = model.predict(reporting_cls(r, is_electricity_data=True), ignore_disqualification=True)
    except Exception:
        return None

    observed = float(pred["observed"].sum())
    modelled = float(pred["predicted"].sum())
    unc = float(np.sqrt(np.nansum(pred["predicted_unc"].to_numpy() ** 2)))
    if not (np.isfinite(observed) and np.isfinite(modelled)):
        return None

    return observed, modelled, unc


def build_fixtures(hourly_data, daily_data, monthly_data, n_pool=99, min_cluster_size=5):
    """Fit models on real ComStock meters and save model_correction inputs.

    Each ``*_data`` argument is a ``(df_baseline, df_reporting)`` tuple as
    returned by the ``_comstock_*_all`` session fixtures.
    """
    FIXTURE_DIR.mkdir(exist_ok=True)

    granularity_data = {"hourly": hourly_data, "daily": daily_data, "billing": monthly_data}

    df_hb, _ = hourly_data
    all_ids = sorted(df_hb.index.get_level_values("id").unique())
    treatment_id = all_ids[0]
    pool_ids = all_ids[1:1 + n_pool]
    print(f"treatment={treatment_id}, pool={len(pool_ids)} meters")

    labels_by_id, weight_by_cluster = _cluster_pool(df_hb, treatment_id, pool_ids, min_cluster_size)
    n_clusters = len(weight_by_cluster)
    print(f"clusters found: {n_clusters}  weights={weight_by_cluster}")

    for gran, (model_cls, baseline_cls, reporting_cls) in _MODELS.items():
        df_b, df_r = granularity_data[gran]

        treatment = _model_reporting_totals(model_cls, baseline_cls, reporting_cls, df_b, df_r, treatment_id)
        if treatment is None:
            raise RuntimeError(f"{gran}: treatment meter failed to model")
        oTr, mTr, mTr_unc = treatment

        oCGr, mCGr, mCGr_unc, labels = [], [], [], []
        for pid in pool_ids:
            totals = _model_reporting_totals(model_cls, baseline_cls, reporting_cls, df_b, df_r, pid)
            label = labels_by_id[str(pid)]
            if totals is None or label < 0:
                continue
            o, m, u = totals
            oCGr.append(o)
            mCGr.append(m)
            mCGr_unc.append(u)
            labels.append(label)

        labels = np.array(labels)
        oCGr, mCGr, mCGr_unc = np.array(oCGr), np.array(mCGr), np.array(mCGr_unc)

        # Keep only clusters that retain >= 5 members after modelling, so each
        # cluster comfortably clears the per-cluster minimum in model_correction.
        counts = {c: int((labels == c).sum()) for c in set(labels.tolist())}
        kept_clusters = sorted(c for c, n in counts.items() if n >= 5)
        keep = np.isin(labels, kept_clusters)
        labels, oCGr, mCGr, mCGr_unc = labels[keep], oCGr[keep], mCGr[keep], mCGr_unc[keep]

        t_weight = np.array([weight_by_cluster[c] for c in kept_clusters])
        t_weight = t_weight / t_weight.sum()

        out = FIXTURE_DIR / f"model_correction_{gran}.npz"
        np.savez(
            out,
            oTr=np.float64(oTr), mTr=np.float64(mTr), mTr_unc=np.float64(mTr_unc),
            oCGr=oCGr, mCGr=mCGr, mCGr_unc=mCGr_unc,
            CG_label=labels.astype(float), T_weight=t_weight,
        )
        print(f"{gran}: saved {out.name}  n_cg={len(oCGr)} clusters={len(kept_clusters)} "
              f"oTr={oTr:.1f} mTr={mTr:.1f}")
