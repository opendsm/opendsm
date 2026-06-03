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

import numpy as np
import pytest

from opendsm.comparison_groups.savings.model_correction import (
    model_correction,
    _model_magnitude_weights,
)
from opendsm.comparison_groups.savings.settings import (
    CGCorrectionSettings,
    CorrectionAlgorithm,
    WeightClusterAggChoice,
)



# A single-hour correction scenario: one treatment meter, six comparison-group
# meters split across two clusters. No real savings data exists, so these are
# constructed inputs exercising one hour of model_correction.
OTR = 100.0
MTR = 110.0
OCGR = np.array([95.0, 98.0, 102.0, 90.0, 105.0, 100.0])
MCGR = np.array([100.0, 100.0, 100.0, 95.0, 110.0, 100.0])
T_WEIGHT = np.array([0.5, 0.5])


def _settings(**overrides):
    base = {
        "algorithm": CorrectionAlgorithm.ABSPCTDID,
        "correction_cap": {"enabled": False},
    }
    base.update(overrides)
    settings = CGCorrectionSettings(**base)

    return settings


def test_model_magnitude_weights_zero_sum_returns_none():
    """All-zero model magnitudes would divide by zero when normalized; the
    helper must fall back to None (uniform) instead of producing NaN weights."""
    assert _model_magnitude_weights(np.zeros(4)) is None

    weights = _model_magnitude_weights(np.array([1.0, 1.0, 2.0]))
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_model_correction_contiguous_labels_runs():
    """Baseline single-hour correction with contiguous integer labels."""
    cg_label = np.array([0, 0, 0, 1, 1, 1])

    mTrc, mTrc_unc, mask = model_correction(
        OTR, MTR, OCGR, MCGR,
        None, None, None, None, None,
        cg_label, T_WEIGHT, _settings(),
    )

    assert np.isfinite(mTrc)
    assert mask.shape == OCGR.shape
    assert mask.dtype == bool


@pytest.mark.parametrize(
    "cg_label",
    [
        np.array([0, 0, 0, 2, 2, 2]),       # non-contiguous integer labels
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),  # float-valued labels
    ],
    ids=["noncontiguous_int", "float_labels"],
)
def test_model_correction_label_indexed_by_position(cg_label):
    """Regression: cluster outputs and T_weight must be indexed by enumeration
    position, not label value. Non-contiguous or float labels previously raised
    IndexError (T_weight[2]) or could not index at all (float)."""
    mTrc, _, _ = model_correction(
        OTR, MTR, OCGR, MCGR,
        None, None, None, None, None,
        cg_label, T_WEIGHT, _settings(),
    )

    assert np.isfinite(mTrc)


def test_model_correction_algorithm_none_returns_vector_mask():
    """Regression: the algorithm-None early exit must return a per-CG-meter
    boolean mask, not a 0-d scalar built from the scalar mTr."""
    cg_label = np.array([0, 0, 0, 1, 1, 1])

    mTrc, mTrc_unc, mask = model_correction(
        OTR, MTR, OCGR, MCGR,
        None, None, None, None, None,
        cg_label, T_WEIGHT, _settings(algorithm=None),
    )

    assert mTrc == MTR
    assert mask.shape == OCGR.shape
    assert mask.dtype == bool
    assert not mask.any()


def test_model_correction_uncertainty_finite_when_model_equals_observed():
    """Regression: when CG model == observed (CG_diff == 0) the uncertainty
    propagation must stay finite. The relative-form variance divided by
    CG_diff**2 and produced NaN; the absolute form does not."""
    cg_label = np.array([0, 0, 0, 1, 1, 1])
    mCGr = OCGR.copy()  # CG_diff == 0 for every comparison meter
    mCGr_unc = np.full(OCGR.shape, 2.0)

    mTrc, mTrc_unc, _ = model_correction(
        OTR, MTR, OCGR, mCGr,
        None, 5.0, None, mCGr_unc, None,
        cg_label, T_WEIGHT, _settings(),
    )

    assert np.isfinite(mTrc)
    assert np.isfinite(mTrc_unc)


def test_uncertainty_with_observed_uncertainty_and_correlation():
    """Exercises the covariance branch (oCGr_unc != 0) of the uncertainty math."""
    cg_label = np.array([0, 0, 0, 1, 1, 1])
    oCGr_unc = np.full(OCGR.shape, 1.5)
    mCGr_unc = np.full(OCGR.shape, 2.0)
    CGr_corr = np.full(OCGR.shape, 0.5)

    mTrc, mTrc_unc, _ = model_correction(
        OTR, MTR, OCGR, MCGR,
        None, 5.0, oCGr_unc, mCGr_unc, CGr_corr,
        cg_label, T_WEIGHT, _settings(),
    )

    assert np.isfinite(mTrc)
    assert np.isfinite(mTrc_unc)


def test_outlier_rejection_path_runs():
    """A cluster with an outlier comparison meter runs the rejection path."""
    cg_label = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    oCGr = np.array([95.0, 98.0, 102.0, 90.0, 500.0, 105.0, 100.0, 103.0, 99.0, 101.0])
    mCGr = np.full(10, 100.0)
    t_weight = np.array([0.5, 0.5])

    mTrc, _, _ = model_correction(
        OTR, MTR, oCGr, mCGr,
        None, None, None, None, None,
        cg_label, t_weight, _settings(outlier_rejection={"enabled": True}),
    )

    assert np.isfinite(mTrc)


def test_global_cap_bounds_correction():
    """A tight global cap bounds the correction to |mTr| * value."""
    cg_label = np.array([0, 0, 0, 1, 1, 1])
    cap_value = 0.01
    settings = _settings(
        correction_cap={"enabled": True, "type": "global", "value": cap_value, "solar_threshold": None}
    )

    mTrc, _, _ = model_correction(
        OTR, MTR, OCGR, MCGR,
        None, None, None, None, None,
        cg_label, T_WEIGHT, settings,
    )

    assert abs(mTrc - MTR) <= abs(MTR) * cap_value + 1e-9


def test_solar_cap_bounds_low_model_meters():
    """Solar cap clips corrections for sub-threshold (low-model) meters."""
    cg_label = np.array([0, 0, 0, 1, 1, 1])
    oCGr = np.full(6, 0.05)
    mCGr = np.full(6, 0.1)  # below the default solar threshold of 1/3
    cap_value = 0.01
    settings = _settings(
        correction_cap={"enabled": True, "type": "solar", "value": cap_value}
    )

    mTrc, _, _ = model_correction(
        OTR, MTR, oCGr, mCGr,
        None, None, None, None, None,
        cg_label, T_WEIGHT, settings,
    )

    assert abs(mTrc - MTR) <= abs(MTR) * cap_value + 1e-9


def test_model_correction_zero_model_cluster_weights_finite():
    """Regression: a cluster whose model magnitudes are all zero yields a
    zero-sum weight normalization. With MODEL weighting it must fall back to a
    uniform mean (finite), not NaN. Uses ODID so scale stays 1 (no mTr/mCGr)."""
    cg_label = np.array([0, 0, 0, 1, 1, 1])
    mCGr = np.array([0.0, 0.0, 0.0, 95.0, 110.0, 100.0])

    mTrc, _, _ = model_correction(
        OTR, MTR, OCGR, mCGr,
        None, None, None, None, None,
        cg_label, T_WEIGHT,
        _settings(
            algorithm=CorrectionAlgorithm.ODID,
            weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
        ),
    )

    assert np.isfinite(mTrc)
