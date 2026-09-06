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

from typing import Optional

import numpy as np
import pandas as pd

from opendsm.comparison_groups.common.data_settings import Data_Settings
from opendsm.common.stats.outliers_transformed import remove_outliers
from opendsm.common.stats.basic import fast_std, unc_factor

import opendsm.comparison_groups.savings.settings as _settings



def _unit_correction_unc(
    oTr, 
    mTr,
    oCGr, 
    mCGr,
    scale,
    CG_diff,
    correction,
    oTr_unc,
    mTr_unc,
    oCGr_unc,
    mCGr_unc,
    CGr_corr, # only needed if oCGr_unc != 0
    method=None
):
    """Calculates correction uncertainty for each comparison group meter of a single treatment meter for a single hour
    
    Args:
        oTr_unc: treatment meter observed uncertainty from reporting period
        mTr_unc: treatment meter model uncertainty from reporting period
        oCGr_unc: comparison group observed uncertainty from reporting period
        mCGr_unc: comparison group model uncertainty from reporting period
        CGr_corr: correlation between oCGr and mCGr over entire reporting period for each meter
        scale: scale factor used in correction calculation
        scale_var: variance of scale factor used in correction calculation
    """
    # The generalized function: m_cT = m_T - s_CG∙(m_CG - o_CG)
    # Correction = s_CG∙(m_CG - o_CG)

    mTr_var = mTr_unc**2
    mCGr_var = mCGr_unc**2

    if method == "ordinary_difference_in_differences":
        # scale = 1, so it carries no variance
        scale_var = 0

    elif method in ("percent_difference_in_differences", "absolute_percent_difference_in_differences"):
        # scale = mTr/mCGr (abs for the latter; |.| has unit-magnitude derivative).
        # Absolute form of Var(mTr/mCGr), neglecting covariance between mTr and mCGr;
        # avoids dividing by mTr (singular when mTr == 0).  A zero mCGr (guarded to
        # scale 0 in _unit_correction) likewise contributes no scale variance.
        denom = np.asarray(mCGr, dtype=float)
        inv_sq = np.divide(1.0, denom ** 2, out=np.zeros_like(denom), where=denom != 0)
        scale_var = mTr_var * inv_sq + (mTr ** 2) * mCGr_var * inv_sq ** 2

    else:
        raise ValueError(f"unknown correction method: {method}")

    if np.all(oCGr_unc == 0):
        CG_diff_var = mCGr_var
    else: # if observed has uncertainty, it and it's covariance with model should be considered
        cov = mCGr_unc*oCGr_unc*CGr_corr
        CG_diff_var = mCGr_var + oCGr_unc**2 - 2*cov

    # correction = scale * CG_diff. Propagate in absolute form (neglecting covariance
    # between scale and CG_diff) so CG_diff == 0 or scale == 0 do not divide by zero.
    correction_var = scale**2*CG_diff_var + CG_diff**2*scale_var
    correction_unc = np.sqrt(correction_var)

    return correction_unc


def _unit_correction(
    oTr, 
    mTr,
    oCGr, 
    mCGr,
    oTr_unc,
    mTr_unc,
    oCGr_unc,
    mCGr_unc,
    CGr_corr, # only needed if oCGr_unc != 0
    calculate_unc,
    method=None
):
    """Calculates corrections for each comparison group meter of a single treatment meter for a single hour
       for a single cluster
    
    Args:
        oTr: treatment meter observed from reporting period
        mTr: treatment meter model from reporting period
        oCGr: comparison group observed from reporting period
        mCGr: comparison group model from reporting period
        oTr_unc: treatment meter observed uncertainty from reporting period
        mTr_unc: treatment meter model uncertainty from reporting period
        oCGr_unc: comparison group observed uncertainty from reporting period
        mCGr_unc: comparison group model uncertainty from reporting period
        CGr_corr: correlation between oCGr and mCGr over entire reporting period for each meter
    """
    # The generalized function: m_cT = m_T - s_CG∙(m_CG - o_CG)
    # Correction = s_CG∙(m_CG - o_CG)

    if method is None:
        # scale = 0
        # scale_unc = 0
        correction = np.zeros_like(mTr)
        correction_unc = np.zeros_like(mTr)

        return correction, correction_unc

    if method == "ordinary_difference_in_differences":
        scale = 1

    elif method in (
        "percent_difference_in_differences",
        "absolute_percent_difference_in_differences",
    ):
        # scale = mTr / mCGr (abs for the latter). A zero comparison-group model
        # magnitude makes the percent scale undefined; guard it to 0 — that meter
        # contributes no correction — rather than dividing to inf.
        denom = np.asarray(mCGr, dtype=float)
        scale = np.divide(mTr, denom, out=np.zeros_like(denom), where=denom != 0)
        if method == "absolute_percent_difference_in_differences":
            scale = np.abs(scale)

    CG_diff = mCGr - oCGr

    # correction
    correction = scale*CG_diff

    if calculate_unc:
        correction_unc = _unit_correction_unc(
            oTr, 
            mTr,
            oCGr, 
            mCGr,
            scale,
            CG_diff,
            correction,
            oTr_unc,
            mTr_unc,
            oCGr_unc,
            mCGr_unc,
            CGr_corr, # only needed if oCGr_unc != 0
            method=method
        )
    else:
        correction_unc = np.full_like(correction, np.nan)

    return correction, correction_unc


def _update_mask(global_mask, mask=None, idx_valid=None, idx_invalid=None):
    if sum(arg is not None for arg in [mask, idx_valid, idx_invalid]) > 1:
        raise ValueError("Only one of `mask`, `idx_valid`, or `idx_invalid` can be provided.")
    
    if mask is not None:
        pass
    
    elif idx_valid is not None:
        mask = np.full_like(global_mask, False, dtype=bool)
        mask[idx_valid] = True

    elif idx_invalid is not None:
        mask = np.full_like(global_mask, True, dtype=bool)
        mask[idx_invalid] = False

    return global_mask & mask


def _apply_mask(mask, *arrays):
    res = []
    for arr in arrays:
        arr_updated = None
        if arr is not None:
            arr_updated = arr[mask]

            if len(arr_updated) < 3:
                raise ValueError("After applying mask, array has insufficient length.")

        res.append(arr_updated)

    if len(res) == 1:
        return res[0]
    
    return tuple(res)


def _effective_sample_size(weight):
    # Kish's effective sample size, weights normalized https://doi.org/10.1002/bimj.19680100122
    n = 1 / np.sum(np.power(weight, 2))

    return n


def _water_fill_weights(weights, valid, cap):
    """Cap weights above `cap`, redistributing the excess over uncapped
    meters (proportionally to their current weight, or equally if every
    uncapped meter carries zero weight), iterating until no weight exceeds
    the cap. Operates along the last axis; terminates in at most
    `weights.shape[-1]` passes. `weights` must already sum to 1 along the
    last axis over `valid` entries; invalid entries are ignored and left 0.
    When the cap is infeasible (`cap * n_valid < 1`, so no capped distribution
    sums to 1) the row falls back to uniform weights over its valid meters.
    """
    w = np.where(valid, weights, 0.0)
    capped = np.zeros_like(valid, dtype=bool)
    n_valid = valid.sum(axis=-1, keepdims=True)
    # A lone valid meter has nowhere to send its excess, so it keeps full weight
    # regardless of the cap.
    capable = n_valid > 1
    # An infeasible cap (cap * n_valid < 1) cannot hold every weight at or below
    # the cap while still summing to 1, so water-filling would return weights
    # summing below 1; fall back to uniform weights over the valid meters (the
    # least-concentrated valid distribution).
    infeasible = capable & (cap * n_valid < 1.0)

    for _ in range(w.shape[-1]):
        over = valid & ~capped & (w > cap) & capable

        if not over.any():
            break

        excess = np.sum(np.where(over, w - cap, 0.0), axis=-1, keepdims=True)
        w = np.where(over, cap, w)
        capped = capped | over

        uncapped = valid & ~capped
        uncapped_sum = np.sum(np.where(uncapped, w, 0.0), axis=-1, keepdims=True)
        n_uncapped = np.sum(uncapped, axis=-1, keepdims=True)

        with np.errstate(invalid="ignore", divide="ignore"):
            proportional = np.divide(w, uncapped_sum, out=np.zeros_like(w), where=uncapped_sum > 0)
            equal = np.divide(
                np.ones_like(w), n_uncapped, out=np.zeros_like(w), where=n_uncapped > 0
            )

        share = np.where(uncapped_sum > 0, proportional, equal) * excess
        w = w + np.where(uncapped, share, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        uniform = np.divide(
            valid.astype(np.float64), n_valid, out=np.zeros_like(w), where=n_valid > 0
        )
    w = np.where(infeasible, uniform, w)

    return w


def _model_magnitude_weights(mCGr, weight_cap):
    # Normalized |model| weights; None when the magnitudes sum to zero (uniform fallback).
    abs_mCGr = np.abs(mCGr)
    total = np.sum(abs_mCGr)

    if total == 0:
        return None

    weights = abs_mCGr / total
    valid = np.ones_like(weights, dtype=bool)
    weights = _water_fill_weights(weights[np.newaxis, :], valid[np.newaxis, :], weight_cap)[0]

    return weights


def _cluster_correction(
    oTr: float, 
    mTr: float,
    oCGr: np.ndarray, 
    mCGr: np.ndarray,
    oTr_unc: Optional[float],
    mTr_unc: Optional[float],
    oCGr_unc: Optional[np.ndarray],
    mCGr_unc: Optional[np.ndarray],
    CGr_corr: Optional[np.ndarray], # only needed if oCGr_unc != 0
    calculate_unc: bool,
    settings: _settings.CGCorrectionSettings,
):
    # Operates on a single cluster's data for a single hour
    
    mask = np.full_like(mCGr, True, dtype=bool)

    # get correction and correction uncertainty
    correct, correct_unc = _unit_correction(
        oTr, 
        mTr,
        oCGr, 
        mCGr,
        oTr_unc,
        mTr_unc,
        oCGr_unc,
        mCGr_unc,
        CGr_corr, # only needed if oCGr_unc != 0
        calculate_unc,
        method=settings.algorithm
    )

    # set initial weights
    if settings.weight_cluster_aggregation is None:
        cluster_weight = None
    elif settings.weight_cluster_aggregation == _settings.WeightClusterAggChoice.MODEL:
        cluster_weight = _model_magnitude_weights(mCGr, settings.weight_cap)

    # remove outliers
    if settings.outlier_rejection.enabled:
        # remove outliers
        _, idx_no_outliers = remove_outliers(
            correct, # if normalized (correct / mTr), small denominator issue introduced
            weights=cluster_weight, 
            sigma_threshold=settings.outlier_rejection.std_threshold, 
            quantile=settings.outlier_rejection.quantile, 
            transform=settings.outlier_rejection.transform
        )

        # update global mask and cluster mask
        mask = _update_mask(mask, idx_valid=idx_no_outliers)

        # remove outliers from data
        correct, correct_unc = _apply_mask(mask, correct, correct_unc)
        mCGr = _apply_mask(mask, mCGr)

        # renormalize weights
        if cluster_weight is not None:
            cluster_weight = _model_magnitude_weights(mCGr, settings.weight_cap)

    # apply caps
    # decision: should capped values have their uncertainty considered or excluded?
    if settings.correction_cap.enabled:
        cap = np.abs(mTr)*settings.correction_cap.value
        if settings.correction_cap.type == _settings.CorrectionCapChoice.GLOBAL:
            correct = np.clip(correct, -cap, cap)
            
        elif settings.correction_cap.type == _settings.CorrectionCapChoice.SOLAR:
            solar_threshold = settings.correction_cap.solar_threshold
            solar_mask = np.abs(mCGr) < solar_threshold
            
            correct[solar_mask] = np.clip(correct[solar_mask], -cap, cap)

    # compute mean and unc
    cluster_mean = np.average(correct, weights=cluster_weight)

    # select the sample size and weighting for the uncertainty estimate. The
    # point correction stays weighted, but a Kish effective sample size below 2
    # cannot support a weighted interval, so when the weights concentrate the
    # spread is estimated with uniform weights over the finite meters.
    unc_weights = cluster_weight
    if calculate_unc:
        n_finite = len(correct)

        if cluster_weight is None:
            n = n_finite
        else:
            n_eff = _effective_sample_size(cluster_weight)

            if n_eff < 2:
                n = n_finite
                unc_weights = None
            else:
                n = n_eff

        if n < 2:
            calculate_unc = False

    # uncertainty calculation
    cluster_unc = np.nan
    if calculate_unc:
        # aggregation uncertainty
        correct_std = fast_std(
            correct,
            mean = cluster_mean,
            weights = unc_weights
        )
        # uncertain if this should be a confidence interval or prediction interval, CI for now
        _unc_factor = unc_factor(n, interval="CI", alpha=settings.alpha)
        correct_agg_unc = correct_std * _unc_factor

        # model uncertainty
        model_var = np.average(correct_unc**2, weights=unc_weights)

        cluster_unc = np.sqrt(correct_agg_unc**2 + model_var)

    return cluster_mean, cluster_unc, mask


def model_correction(
    oTr: float,         # observed treatment meter value during reporting period
    mTr: float,         # model treatment meter value during reporting period
    oCGr: np.ndarray, 
    mCGr: np.ndarray,
    oTr_unc: Optional[float],
    mTr_unc: Optional[float],
    oCGr_unc: Optional[np.ndarray],
    mCGr_unc: Optional[np.ndarray],
    CGr_corr: Optional[np.ndarray], # only needed if oCGr_unc != 0
    CG_label: np.ndarray,
    T_weight: np.ndarray,
    settings: _settings.CGCorrectionSettings,
):
    # if no did, return
    if settings.algorithm is None:
        # no difference-in-differences correction applied
        mTrc = float(mTr)

        if mTr_unc is not None:
            mTrc_unc = float(mTr_unc)
        else:
            mTrc_unc = np.nan

        # no comparison-group meters are used; mask spans the CG meters
        mask = np.zeros(np.shape(oCGr), dtype=bool)

        return mTrc, mTrc_unc, mask
    
    # input validation
    if mTr is None or not np.isfinite(mTr):
        raise ValueError("`mTr` must be a finite number")

    if len(oCGr) < 5:
        raise ValueError("`oCGr` cannot have a length less than 5")
    
    if not (len(oCGr) == len(mCGr) == len(CG_label)):
        raise ValueError("`oCGr`, `mCGr`, and `CG_label` must have the same length")
    
    if len(T_weight) != np.sum(np.unique(CG_label) >= 0):
        raise ValueError("`T_weight` must have the same number of elements as the unique number of labels in `CG_label`")

    if oCGr_unc is None:
        oCGr_unc = np.zeros_like(oCGr)

    if not (len(oCGr) == len(oCGr_unc)):
        raise ValueError("`oCGr` and `oCGr_unc` must have the same length")

    if mCGr_unc is not None:
        if not (len(mCGr) == len(mCGr_unc)):
            raise ValueError("`mCGr` and `mCGr_unc` must have the same length")

    if CGr_corr is None:
        CGr_corr = np.zeros_like(oCGr)

    if not (len(oCGr_unc) == len(CGr_corr)):
        raise ValueError("`oCGr_unc` and `CGr_corr` must have the same length")
    
    # check length of CG inputs and set global_mask to exclude non-finite values
    global_mask = np.isfinite(oCGr) & np.isfinite(mCGr) & np.isfinite(CG_label)
    global_mask = global_mask & (oCGr is not None) & (mCGr is not None)
    global_mask = global_mask & (CG_label is not None)

    calculate_unc = False
    if mTr_unc is not None and mCGr_unc is not None:
        calculate_unc = True
        global_mask = global_mask & np.isfinite(mCGr_unc) & (mCGr_unc is not None)

    if calculate_unc and oCGr_unc is not None and CGr_corr is not None:
        global_mask = global_mask & np.isfinite(oCGr_unc) & (oCGr_unc is not None)
        global_mask = global_mask & np.isfinite(CGr_corr) & (CGr_corr is not None)

    unique_labels = np.unique(CG_label)
    unique_labels = unique_labels[np.isfinite(unique_labels)]
    unique_labels = unique_labels[unique_labels >= 0] # exclude outlier label(s)

    # T_weight is positionally aligned with the sorted non-negative labels,
    # so index it by enumeration position, not by label value (which may be
    # non-contiguous or float).
    T_weight = np.asarray(T_weight).flatten()

    cluster_correct = np.full(unique_labels.shape, np.nan)
    cluster_correct_unc = np.full(unique_labels.shape, np.nan)
    for i, label in enumerate(unique_labels):
        # get label mask
        label_mask = CG_label == label
        mask = global_mask & label_mask

        if T_weight[i] == 0:
            # zero-weight cluster: drop from the global mask, leave correction nan
            global_mask[label_mask] = False

        else:
            _correct, _correct_unc, _mask = _cluster_correction(
                oTr,
                mTr,
                _apply_mask(mask, oCGr),
                _apply_mask(mask, mCGr),
                oTr_unc,
                mTr_unc,
                _apply_mask(mask, oCGr_unc),
                _apply_mask(mask, mCGr_unc),
                _apply_mask(mask, CGr_corr), # only needed if oCGr_unc != 0
                calculate_unc,
                settings,
            )

            if not np.isfinite(_correct_unc):
                calculate_unc = False

            # update global mask
            global_mask[mask] = _update_mask(global_mask[mask], mask=_mask)

            cluster_correct[i] = _correct
            cluster_correct_unc[i] = _correct_unc

    # combine clusters with weights to get corrected model
    idx_valid = (T_weight > 0).flatten()
    correction = np.average(cluster_correct[idx_valid], weights=T_weight[idx_valid])
    mTrc = float(mTr - correction)

    mTrc_unc = np.nan
    if calculate_unc:
        correction_var = np.sum((T_weight[idx_valid]**2)*(cluster_correct_unc[idx_valid]**2))
        mTrc_unc = float(np.sqrt(mTr_unc**2 + correction_var))

    return mTrc, mTrc_unc, global_mask


def _cluster_reduce(correct_c, correct_unc_c, valid_c, mag_c, calculate_unc, settings):
    """Vectorized single-cluster aggregation over timesteps.

    Reproduces the per-hour aggregation of `_cluster_correction` (default path,
    no outlier rejection) for one cluster across all T timesteps at once.

    Args:
        correct_c: per-meter corrections for the cluster, shape (T, m)
        correct_unc_c: per-meter correction uncertainties, shape (T, m) or None
        valid_c: finite/usable mask for the cluster's meters, shape (T, m)
        mag_c: comparison-group model magnitudes for the cluster, shape (T, m)
        calculate_unc: whether uncertainty is propagated

    Returns:
        cluster_mean: weighted cluster correction per timestep, shape (T,)
        cluster_unc: cluster uncertainty per timestep (NaN where undefined),
            shape (T,)
    """
    T = correct_c.shape[0]
    v = valid_c.astype(np.float64)
    nc = valid_c.sum(axis=1)

    x = np.where(valid_c, correct_c, 0.0)
    mag = np.where(valid_c, np.abs(mag_c), 0.0)

    if settings.weight_cluster_aggregation is None:
        wraw = v
        zero_total = np.zeros(T, dtype=bool)
    elif settings.weight_cluster_aggregation == _settings.WeightClusterAggChoice.MODEL:
        zero_total = mag.sum(axis=1) == 0.0
        wraw = np.where(zero_total[:, None], v, mag)
    else:
        raise ValueError(
            f"unknown weight_cluster_aggregation: {settings.weight_cluster_aggregation}"
        )

    wsum = wraw.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        wn = np.divide(wraw, wsum[:, None], out=np.zeros_like(wraw), where=wsum[:, None] > 0)

    if settings.weight_cluster_aggregation == _settings.WeightClusterAggChoice.MODEL:
        cap_rows = ~zero_total & (wsum > 0)
        wn_capped = _water_fill_weights(wn, valid_c, settings.weight_cap)
        wn = np.where(cap_rows[:, None], wn_capped, wn)

    mu = np.where(wsum > 0, (wn * x).sum(axis=1), np.nan)

    dev2 = np.where(valid_c, (x - mu[:, None]) ** 2, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        var_unw = np.divide((v * dev2).sum(axis=1), nc, out=np.full(T, np.nan), where=nc > 0)
        denom_w = 1.0 - np.divide(1.0, nc, out=np.full(T, np.nan), where=nc > 0)
        var_w = np.divide((wn * dev2).sum(axis=1), denom_w, out=np.full(T, np.nan), where=denom_w > 0)

    std_unw = np.sqrt(np.clip(var_unw, 0.0, None))
    std_w = np.sqrt(np.clip(var_w, 0.0, None))

    n_unc = nc.astype(np.float64)
    weighted_path = np.zeros(T, dtype=bool)
    if settings.weight_cluster_aggregation is not None:
        sum_wn2 = (wn ** 2).sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            n_eff_model = np.divide(1.0, sum_wn2, out=np.full(T, np.nan), where=sum_wn2 > 0)

        # fast_std treats uniform weights as unweighted (population std); mirror
        # its np.allclose(w - w[0], 0) test (default atol 1e-8) per timestep.
        idx0 = valid_c.argmax(axis=1)
        wn0 = wn[np.arange(T), idx0]
        uniform = np.where(valid_c, np.abs(wn - wn0[:, None]), 0.0).max(axis=1) <= 1e-8

        # a Kish effective sample size below 2 cannot support a weighted interval,
        # so weighted spread is used only where the sample supports it; uniform,
        # zero-magnitude, or concentrated timesteps fall back to the unweighted
        # spread over the finite meters while the point correction stays weighted.
        weighted_path = ~zero_total & ~uniform & np.isfinite(n_eff_model) & (n_eff_model >= 2)
        n_unc = np.where(weighted_path, n_eff_model, nc.astype(np.float64))

    std = np.where(weighted_path, std_w, std_unw)

    cluster_unc = np.full(T, np.nan)
    if calculate_unc:
        xu = np.where(valid_c, correct_unc_c, 0.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            model_var_w = np.where(wsum > 0, (wn * xu ** 2).sum(axis=1), np.nan)
            model_var_unw = np.divide((v * xu ** 2).sum(axis=1), nc, out=np.full(T, np.nan), where=nc > 0)

        model_var = np.where(weighted_path, model_var_w, model_var_unw)

        ok = np.isfinite(n_unc) & (n_unc >= 2)
        ufac = np.full(T, np.nan)
        if ok.any():
            ufac[ok] = unc_factor(n_unc[ok], interval="CI", alpha=settings.alpha)

        with np.errstate(invalid="ignore"):
            agg_var = (std * ufac) ** 2 + model_var
            cluster_unc = np.where(ok, np.sqrt(np.clip(agg_var, 0.0, None)), np.nan)

    return mu, cluster_unc


def model_correction_matrix(
    oTr: np.ndarray,        # observed treatment values, shape (T,)
    mTr: np.ndarray,        # model treatment values, shape (T,)
    oCGr: np.ndarray,       # comparison-group observed, shape (T, M)
    mCGr: np.ndarray,       # comparison-group model, shape (T, M)
    oTr_unc: Optional[np.ndarray],   # (T,) or None
    mTr_unc: Optional[np.ndarray],   # (T,) or None
    oCGr_unc: Optional[np.ndarray],  # (T, M) or None
    mCGr_unc: Optional[np.ndarray],  # (T, M) or None
    CGr_corr: Optional[np.ndarray],  # per-meter correlation, shape (M,) or None
    CG_label: np.ndarray,   # per-meter cluster label, shape (M,)
    T_weight: np.ndarray,   # per-cluster weight, shape (K,)
    settings: _settings.CGCorrectionSettings,
):
    """Vectorized difference-in-differences correction over a whole window.

    Applies the same per-timestep correction math as the scalar
    `model_correction` to a stack of T timesteps at once. All math is done in
    float64 regardless of the input dtype (float32 inputs are cast up). `oTr`
    is accepted for parity with the scalar kernel's signature and does not
    enter the math.

    This kernel degrades per cluster, not per timestep: at each timestep a
    nonzero-weight cluster with fewer than 3 finite comparison-group meters is
    dropped and the surviving clusters are averaged with their weights
    renormalized over the survivors, so a sparse cluster does not abort the whole
    timestep. `mTrc[t]` is NaN only when no cluster survives (all degenerate) or
    `mTr[t]` is non-finite. This is a deliberate divergence from the scalar
    `model_correction`, which raises when any cluster has too few finite meters.
    Uncertainty is a quadrature over the surviving clusters that carry a finite
    uncertainty, using the same survivor-renormalized weights; a surviving
    cluster with a non-finite uncertainty stays in the point but is omitted from
    the band, understating it. The quadrature treats the comparison-group
    meters as independent, so the band is a heuristic rather than a calibrated
    interval. `mTrc_unc[t]` is NaN
    only when `mTr_unc[t]` is non-finite or no surviving cluster contributes a
    finite uncertainty. The global `M < 5` check is a hard error.

    The per-timestep model uncertainty this kernel combines is
    granularity-dependent: at hourly cadence it already reconstructs the
    ASHRAE hourly aggregate band; at daily/billing cadence it is a t-scaled
    prediction-interval band treated as sigma-like for the combination. The
    `*_unc` outputs are therefore heuristic combined bands, not calibrated
    (1 - alpha) intervals, at any cadence.

    Args:
        oTr, mTr: treatment observed/model, shape (T,)
        oCGr, mCGr: comparison-group observed/model, shape (T, M)
        oTr_unc, mTr_unc: treatment uncertainties, shape (T,) or None
        oCGr_unc, mCGr_unc: comparison-group uncertainties, shape (T, M) or None
        CGr_corr: per-meter observed-vs-model correlation, shape (M,) or None
        CG_label: per-meter cluster label, shape (M,)
        T_weight: per-cluster weight aligned with sorted non-negative labels,
            shape (K,)

    Returns:
        mTrc: corrected treatment model, shape (T,)
        mTrc_unc: corrected uncertainty (heuristic band), shape (T,)
        mask: per-timestep per-meter usage mask, shape (T, M): a meter is
            marked where it is finite, in a nonzero-weight cluster, and that
            cluster survived the timestep; a NaN row is all False, and with
            `algorithm` None no comparison meter enters the point, so the mask
            is all False everywhere
    """
    mTr = np.asarray(mTr, dtype=np.float64)
    oCGr = np.asarray(oCGr, dtype=np.float64)
    mCGr = np.asarray(mCGr, dtype=np.float64)
    CG_label = np.asarray(CG_label, dtype=np.float64)
    T_weight = np.asarray(T_weight, dtype=np.float64).flatten()

    T = mTr.shape[0]
    M = oCGr.shape[1]
    method = settings.algorithm

    if method is None:
        # no difference-in-differences correction applied
        mTrc = mTr.copy()

        if mTr_unc is not None:
            mTrc_unc = np.asarray(mTr_unc, dtype=np.float64).copy()
        else:
            mTrc_unc = np.full(T, np.nan)

        mask = np.zeros((T, M), dtype=bool)

        return mTrc, mTrc_unc, mask

    if M < 5:
        raise ValueError("`oCGr` cannot have fewer than 5 comparison-group meters")

    if mCGr.shape != (T, M) or oCGr.shape != (T, M):
        raise ValueError("`oCGr` and `mCGr` must both have shape (T, M)")

    if CG_label.shape != (M,):
        raise ValueError("`CG_label` must have shape (M,)")

    n_clusters = int(np.sum(np.unique(CG_label) >= 0))
    if T_weight.shape[0] != n_clusters:
        raise ValueError(
            "`T_weight` must have the same number of elements as the unique number of "
            "non-negative labels in `CG_label`"
        )

    calculate_unc = mTr_unc is not None and mCGr_unc is not None
    if calculate_unc:
        mTr_unc = np.asarray(mTr_unc, dtype=np.float64)
        mCGr_unc = np.asarray(mCGr_unc, dtype=np.float64)

        if mTr_unc.shape != (T,):
            raise ValueError("`mTr_unc` must have shape (T,)")

        if mCGr_unc.shape != (T, M):
            raise ValueError("`mCGr_unc` must have shape (T, M)")

        if oCGr_unc is None:
            oCGr_unc = np.zeros((T, M))
        else:
            oCGr_unc = np.asarray(oCGr_unc, dtype=np.float64)

        if CGr_corr is None:
            CGr_corr = np.zeros(M)
        else:
            CGr_corr = np.asarray(CGr_corr, dtype=np.float64)

        if oCGr_unc.shape != (T, M):
            raise ValueError("`oCGr_unc` must have shape (T, M)")

        if CGr_corr.shape != (M,):
            raise ValueError("`CGr_corr` must have shape (M,)")

    finite = np.isfinite(oCGr) & np.isfinite(mCGr) & np.isfinite(CG_label)[None, :]
    if calculate_unc:
        finite = finite & np.isfinite(mCGr_unc) & np.isfinite(oCGr_unc) & np.isfinite(CGr_corr)[None, :]

    unique_labels = np.unique(CG_label)
    unique_labels = unique_labels[np.isfinite(unique_labels)]
    unique_labels = unique_labels[unique_labels >= 0]

    # meters in zero-weight clusters are never usable, mirroring the scalar kernel
    mask_base = finite.copy()
    for i, label in enumerate(unique_labels):
        if T_weight[i] == 0:
            mask_base[:, CG_label == label] = False

    if settings.outlier_rejection.enabled:
        mTrc, mTrc_unc, mask = _matrix_correction_loop(
            oTr, mTr, oCGr, mCGr,
            oTr_unc, mTr_unc, oCGr_unc, mCGr_unc, CGr_corr,
            CG_label, T_weight, settings, mask_base,
        )

        return mTrc, mTrc_unc, mask

    # per-meter unit correction, fully vectorized
    CG_diff = mCGr - oCGr
    if method == _settings.CorrectionAlgorithm.ODID:
        scale = np.ones((T, M))
    elif method in (
        _settings.CorrectionAlgorithm.PCTDID,
        _settings.CorrectionAlgorithm.ABSPCTDID,
    ):
        scale = np.divide(mTr[:, None], mCGr, out=np.zeros((T, M)), where=mCGr != 0)
        if method == _settings.CorrectionAlgorithm.ABSPCTDID:
            scale = np.abs(scale)
    else:
        raise ValueError(f"unknown correction method: {method}")

    correct = scale * CG_diff

    correct_unc = None
    if calculate_unc:
        mTr_var = (mTr_unc ** 2)[:, None]
        mCGr_var = mCGr_unc ** 2

        if method == _settings.CorrectionAlgorithm.ODID:
            scale_var = 0.0
        else:
            inv_sq = np.divide(1.0, mCGr ** 2, out=np.zeros((T, M)), where=mCGr != 0)
            scale_var = mTr_var * inv_sq + (mTr[:, None] ** 2) * mCGr_var * inv_sq ** 2

        cov = mCGr_unc * oCGr_unc * CGr_corr[None, :]
        CG_diff_var = mCGr_var + oCGr_unc ** 2 - 2 * cov
        correction_var = scale ** 2 * CG_diff_var + CG_diff ** 2 * scale_var

        with np.errstate(invalid="ignore"):
            correct_unc = np.sqrt(correction_var)

    # elementwise correction caps
    if settings.correction_cap.enabled:
        cap = np.abs(mTr)[:, None] * settings.correction_cap.value

        if settings.correction_cap.type == _settings.CorrectionCapChoice.GLOBAL:
            correct = np.clip(correct, -cap, cap)
        elif settings.correction_cap.type == _settings.CorrectionCapChoice.SOLAR:
            solar_mask = np.abs(mCGr) < settings.correction_cap.solar_threshold
            correct = np.where(solar_mask, np.clip(correct, -cap, cap), correct)

    cluster_means = []
    cluster_uncs = []
    cluster_deg = []
    comb_weights = []
    comb_cols = []
    for i, label in enumerate(unique_labels):
        if T_weight[i] == 0:
            continue

        cols = CG_label == label
        valid_c = finite[:, cols]
        # a cluster with fewer than 3 finite meters is degenerate at that
        # timestep and drops from the combination there
        deg_c = valid_c.sum(axis=1) < 3

        if calculate_unc:
            correct_unc_c = correct_unc[:, cols]
        else:
            correct_unc_c = None

        mean_c, unc_c = _cluster_reduce(
            correct[:, cols], correct_unc_c, valid_c, mCGr[:, cols], calculate_unc, settings
        )

        cluster_means.append(np.where(deg_c, np.nan, mean_c))
        cluster_uncs.append(np.where(deg_c, np.nan, unc_c))
        cluster_deg.append(deg_c)
        comb_weights.append(T_weight[i])
        comb_cols.append(cols)

    if not comb_weights:
        raise ValueError("all clusters have zero weight; no correction can be formed")

    cm = np.stack(cluster_means, axis=1)
    cu = np.stack(cluster_uncs, axis=1)
    cw = np.asarray(comb_weights)

    # per timestep, average the surviving clusters (>= 3 finite meters) with
    # their weights renormalized over the survivors; the row is NaN only when no
    # cluster survives or the treatment model is non-finite. A survivor whose
    # point is itself non-finite propagates NaN into the row, as in the scalar
    # kernel.
    surv = ~np.stack(cluster_deg, axis=1)
    w = cw[None, :] * surv
    wsum = w.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        wn = np.divide(w, wsum[:, None], out=np.zeros_like(w), where=wsum[:, None] > 0)
        correction = np.where(surv, wn * cm, 0.0).sum(axis=1)
        mTrc = mTr - correction

    no_survivor = (wsum <= 0) | ~np.isfinite(mTr)
    mTrc = np.where(no_survivor, np.nan, mTrc)

    # the usage mask reports the meters that entered the point: a dropped
    # cluster's members leave it at that timestep, and a NaN row uses none
    mask = mask_base.copy()
    for j, cols in enumerate(comb_cols):
        mask[:, cols] &= surv[:, j][:, None]
    mask[no_survivor] = False

    if calculate_unc:
        # quadrature over survivors that also carry a finite uncertainty, using
        # the survivor-renormalized point weights; a surviving cluster with a
        # non-finite uncertainty stays in the point but drops here, spending its
        # weight without contributing — a documented understatement of the band.
        unc_contrib = surv & np.isfinite(cu)

        with np.errstate(invalid="ignore"):
            correction_var = np.where(unc_contrib, (wn ** 2) * (cu ** 2), 0.0).sum(axis=1)
            mTrc_unc = np.sqrt(mTr_unc ** 2 + correction_var)

        unc_bad = ~np.isfinite(mTr_unc) | ~unc_contrib.any(axis=1) | ~np.isfinite(mTrc)
        mTrc_unc = np.where(unc_bad, np.nan, mTrc_unc)
    else:
        mTrc_unc = np.full(T, np.nan)

    return mTrc, mTrc_unc, mask


def _matrix_correction_loop(
    oTr, mTr, oCGr, mCGr,
    oTr_unc, mTr_unc, oCGr_unc, mCGr_unc, CGr_corr,
    CG_label, T_weight, settings, mask_base,
):
    """Per-timestep fallback for the outlier-rejection path.

    Outlier rejection depends on the joint distribution of a cluster's meters at
    each timestep, so the vectorized aggregation does not apply. This runs the
    per-cluster reduction (`_cluster_correction`, which rejects outliers) for
    each cluster at each timestep and applies the same per-cluster degradation
    as the vectorized path: a cluster with fewer than 3 finite meters (or that
    outlier rejection drives below 3) is dropped, the surviving clusters are
    averaged with their weights renormalized over the survivors, and the row is
    NaN only when no cluster survives. Uncertainty is a quadrature over the
    survivors that carry a finite uncertainty, using those renormalized weights;
    a survivor with a non-finite uncertainty stays in the point but is omitted
    from the band, understating it. A dropped cluster's meters are cleared from
    the usage mask at that timestep, and a NaN row uses none.
    """
    T = oCGr.shape[0]
    calculate_unc = mTr_unc is not None and mCGr_unc is not None

    unique_labels = np.unique(CG_label)
    unique_labels = unique_labels[np.isfinite(unique_labels)]
    unique_labels = unique_labels[unique_labels >= 0]

    mTrc = np.full(T, np.nan)
    mTrc_unc = np.full(T, np.nan)
    mask = mask_base.copy()
    for t in range(T):
        if not np.isfinite(mTr[t]):
            mask[t] = False
            continue

        oTr_unc_t = None
        if oTr_unc is not None:
            oTr_unc_t = float(oTr_unc[t])

        mTr_unc_t = None
        if mTr_unc is not None:
            mTr_unc_t = float(mTr_unc[t])

        global_mask_t = mask_base[t].copy()

        surv_weight = []
        surv_correct = []
        surv_correct_unc = []
        for i, label in enumerate(unique_labels):
            if T_weight[i] == 0:
                continue

            cluster_mask = global_mask_t & (CG_label == label)

            if cluster_mask.sum() < 3:
                # degenerate cluster: drop from this timestep
                global_mask_t[cluster_mask] = False
                continue

            oCGr_unc_c = None
            if oCGr_unc is not None:
                oCGr_unc_c = oCGr_unc[t][cluster_mask]

            mCGr_unc_c = None
            if mCGr_unc is not None:
                mCGr_unc_c = mCGr_unc[t][cluster_mask]

            CGr_corr_c = None
            if CGr_corr is not None:
                CGr_corr_c = CGr_corr[cluster_mask]

            try:
                _correct, _correct_unc, _mask = _cluster_correction(
                    float(oTr[t]), float(mTr[t]),
                    oCGr[t][cluster_mask], mCGr[t][cluster_mask],
                    oTr_unc_t, mTr_unc_t, oCGr_unc_c, mCGr_unc_c, CGr_corr_c,
                    calculate_unc, settings,
                )
            except ValueError as e:
                if "insufficient length" in str(e):
                    # outlier rejection drove this cluster below 3 meters: drop it
                    global_mask_t[cluster_mask] = False
                    continue

                raise

            global_mask_t[cluster_mask] = _update_mask(global_mask_t[cluster_mask], mask=_mask)

            surv_weight.append(float(T_weight[i]))
            surv_correct.append(_correct)
            surv_correct_unc.append(_correct_unc)

        mask[t] = global_mask_t

        if not surv_weight:
            # every cluster degenerate at this timestep: row stays NaN
            continue

        w = np.asarray(surv_weight, dtype=np.float64)
        wn = w / w.sum()
        c = np.asarray(surv_correct, dtype=np.float64)
        mTrc[t] = float(mTr[t] - np.sum(wn * c))

        if calculate_unc and np.isfinite(mTr_unc_t):
            cu = np.asarray(surv_correct_unc, dtype=np.float64)
            unc_ok = np.isfinite(cu)

            if unc_ok.any():
                correction_var = float(np.sum((wn[unc_ok] ** 2) * (cu[unc_ok] ** 2)))
                mTrc_unc[t] = float(np.sqrt(mTr_unc_t ** 2 + correction_var))

    return mTrc, mTrc_unc, mask