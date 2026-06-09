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


def _model_magnitude_weights(mCGr):
    # Normalized |model| weights; None when the magnitudes sum to zero (uniform fallback).
    abs_mCGr = np.abs(mCGr)
    total = np.sum(abs_mCGr)

    if total == 0:
        return None

    weights = abs_mCGr / total

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
        cluster_weight = _model_magnitude_weights(mCGr)

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
            cluster_weight = _model_magnitude_weights(mCGr)

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

    # check n to see if unc can be calculated
    if calculate_unc:
        if cluster_weight is None:
            n = len(correct)
        else:
            n = _effective_sample_size(cluster_weight)

        if n < 2:
            calculate_unc = False

    # uncertainty calculation
    cluster_unc = np.nan
    if calculate_unc:
        # aggregation uncertainty
        correct_std = fast_std(
            correct,
            mean = cluster_mean,
            weights = cluster_weight
        )
        # uncertain if this should be a confidence interval or prediction interval, CI for now
        _unc_factor = unc_factor(n, interval="CI", alpha=settings.alpha)
        correct_agg_unc = correct_std * _unc_factor

        # model uncertainty
        model_var = np.average(correct_unc**2, weights=cluster_weight)

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