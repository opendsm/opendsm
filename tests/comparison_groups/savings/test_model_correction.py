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

import os

import numpy as np
import pytest

from opendsm.common.stats.basic import fast_std, unc_factor
from opendsm.comparison_groups.savings.model_correction import (
    model_correction,
    _cluster_correction,
    _model_magnitude_weights,
    _water_fill_weights,
)
from .generate_correction_fixtures import build_fixtures
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
    """Zero total model magnitude cannot be normalized into weights, so the
    helper returns None, signaling callers to use uniform weighting."""
    assert _model_magnitude_weights(np.zeros(4), weight_cap=0.5) is None

    weights = _model_magnitude_weights(np.array([1.0, 1.0, 2.0]), weight_cap=0.5)
    np.testing.assert_allclose(weights.sum(), 1.0)


# ── Water-filling weight cap ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "magnitudes, expected",
    [
        ([0.8, 0.1, 0.1], [0.5, 0.25, 0.25]),
        ([0.9, 0.1], [0.5, 0.5]),
        ([1.0, 0.0, 0.0], [0.5, 0.25, 0.25]),
    ],
)
def test_model_magnitude_weights_water_fills_excess_over_cap(magnitudes, expected):
    """Weight above `weight_cap` is clipped to the cap and its excess is
    redistributed over the uncapped meters (proportionally, or equally when
    they all carry zero weight)."""
    weights = _model_magnitude_weights(np.array(magnitudes), weight_cap=0.5)

    np.testing.assert_allclose(weights, expected)


@pytest.mark.parametrize("n_meters", [2, 3, 5, 10])
def test_model_magnitude_weights_default_cap_keeps_kish_ess_at_least_two(n_meters):
    """A weight_cap of 0.5 caps any single meter's share at half the cluster's
    weight, which the R5 design guarantees keeps Kish's effective sample size
    at or above 2 for any cluster of 2 or more meters, however skewed the
    magnitudes."""
    rng = np.random.default_rng(0)
    magnitudes = rng.exponential(scale=1.0, size=n_meters)
    magnitudes[0] *= 1000.0  # force one meter to dominate

    weights = _model_magnitude_weights(magnitudes, weight_cap=0.5)
    ess = 1.0 / np.sum(weights**2)

    assert ess >= 2.0 - 1e-9


def test_model_magnitude_weights_infeasible_cap_falls_back_to_uniform():
    """A cap below 1/M cannot hold every weight at or below it while summing to
    1 (M*cap < 1); rather than returning weights that sum below 1, the helper
    falls back to uniform weights over the meters. M=3, cap=0.3 -> [1/3, 1/3,
    1/3] summing to exactly 1."""
    weights = _model_magnitude_weights(np.array([5.0, 3.0, 2.0]), weight_cap=0.3)

    np.testing.assert_allclose(weights, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert weights.sum() == pytest.approx(1.0, abs=1e-15)


def test_water_fill_infeasible_cap_uniform_only_over_valid_meters():
    """The infeasible-cap uniform fallback spreads over the valid meters only:
    with one invalid meter of three, cap 0.3 is infeasible for the two valid
    meters (2*0.3 < 1) so they take 0.5 each and the invalid meter stays 0."""
    weights = np.array([[0.5, 0.5, 0.0]])
    valid = np.array([[True, True, False]])

    filled = _water_fill_weights(weights, valid, cap=0.3)

    np.testing.assert_allclose(filled, [[0.5, 0.5, 0.0]])
    assert filled.sum() == pytest.approx(1.0, abs=1e-15)


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
    """Regression: cluster outputs and T_weight are indexed by enumeration
    position, not label value, so non-contiguous integer labels and
    float-valued labels both index correctly."""
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


@pytest.mark.parametrize(
    "algorithm",
    [CorrectionAlgorithm.ODID, CorrectionAlgorithm.PCTDID, CorrectionAlgorithm.ABSPCTDID],
)
def test_uncertainty_finite_for_each_algorithm(algorithm):
    """Each scale_var branch (ODID constant, PCT/ABSPCT ratio) stays finite."""
    cg_label = np.array([0, 0, 0, 1, 1, 1])
    mCGr_unc = np.full(OCGR.shape, 2.0)

    _, mTrc_unc, _ = model_correction(
        OTR, MTR, OCGR, MCGR,
        None, 5.0, None, mCGr_unc, None,
        cg_label, T_WEIGHT, _settings(algorithm=algorithm),
    )

    assert np.isfinite(mTrc_unc)


@pytest.mark.parametrize("algorithm", [CorrectionAlgorithm.PCTDID, CorrectionAlgorithm.ABSPCTDID])
def test_zero_cg_model_magnitude_stays_finite(algorithm):
    """A comparison meter with zero model magnitude has an undefined percent
    scale mTr/mCGr; the guard makes it contribute no correction, so the
    correction and its uncertainty stay finite. Outlier rejection is disabled
    so the guard, not the rejection, is what is exercised."""
    cg_label = np.array([0, 0, 0, 1, 1, 1])
    mCGr = np.array([0.0, 100.0, 100.0, 95.0, 110.0, 100.0])  # first model magnitude == 0
    mCGr_unc = np.full(6, 2.0)

    mTrc, mTrc_unc, _ = model_correction(
        OTR, MTR, OCGR, mCGr,
        None, 5.0, None, mCGr_unc, None,
        cg_label, T_WEIGHT,
        _settings(algorithm=algorithm, outlier_rejection={"enabled": False}),
    )

    assert np.isfinite(mTrc)
    assert np.isfinite(mTrc_unc)


def test_rejects_non_finite_mtr():
    cg_label = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        model_correction(
            OTR, np.nan, OCGR, MCGR, None, None, None, None, None,
            cg_label, T_WEIGHT, _settings(),
        )


def test_rejects_comparison_group_shorter_than_five():
    short = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        model_correction(
            OTR, MTR, short, short, None, None, None, None, None,
            np.array([0, 0, 0]), np.array([1.0]), _settings(),
        )


def test_rejects_mismatched_cg_lengths():
    cg_label = np.array([0, 0, 0, 1, 1, 1])
    with pytest.raises(ValueError):
        model_correction(
            OTR, MTR, OCGR, MCGR[:5], None, None, None, None, None,
            cg_label, T_WEIGHT, _settings(),
        )


def test_rejects_t_weight_length_mismatch():
    cg_label = np.array([0, 0, 0, 1, 1, 1])  # two clusters
    with pytest.raises(ValueError):
        model_correction(
            OTR, MTR, OCGR, MCGR, None, None, None, None, None,
            cg_label, np.array([1.0]), _settings(),  # only one weight
        )


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


# ── Effective-sample-size uncertainty fallback (scalar kernel) ───────────────

def test_cluster_unc_falls_back_to_uniform_weights_below_ess_two():
    """When model-magnitude weights concentrate so the Kish effective sample size
    drops below 2, the cluster's point correction stays weighted but its
    uncertainty is estimated with uniform weights over the finite meters. The
    resulting uncertainty must equal the hand-computed uniform-weight value.

    weight_cap is set to 1.0 (disabling the water-filling cap) so this scenario
    still concentrates enough to drop the effective sample size below 2; at the
    default cap of 0.5 the same magnitudes resolve to the weighted path."""
    mCGr = np.array([1000.0, 10.0, 10.0, 10.0])  # one meter carries ~97% of magnitude
    oCGr = np.array([990.0, 8.0, 12.0, 9.0])
    mCGr_unc = np.array([4.0, 3.0, 2.0, 1.0])
    zeros = np.zeros_like(mCGr)
    alpha = 0.10

    settings = CGCorrectionSettings(
        algorithm=CorrectionAlgorithm.ODID,
        correction_cap={"enabled": False},
        outlier_rejection={"enabled": False},
        weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
        weight_cap=1.0,
        alpha=alpha,
    )

    cluster_mean, cluster_unc, _ = _cluster_correction(
        100.0, 110.0, oCGr, mCGr, None, 5.0, zeros, mCGr_unc, zeros, True, settings,
    )

    # ODID scale is 1, so each meter's correction is mCGr - oCGr and its model
    # uncertainty is mCGr_unc; the point correction is the magnitude-weighted mean.
    correct = mCGr - oCGr
    weights = _model_magnitude_weights(mCGr, weight_cap=1.0)
    weighted_mean = np.average(correct, weights=weights)
    # uniform-weight fallback: population std about the weighted mean, unc_factor
    # on the finite meter count, unweighted mean of the per-meter model variance.
    std = fast_std(correct, mean=weighted_mean, weights=None)
    agg_unc = std * unc_factor(len(correct), interval="CI", alpha=alpha)
    model_var = np.mean(mCGr_unc**2)
    expected_unc = np.sqrt(agg_unc**2 + model_var)

    assert cluster_mean == pytest.approx(weighted_mean, rel=1e-12)
    assert np.isfinite(cluster_unc) and cluster_unc > 0
    assert cluster_unc == pytest.approx(expected_unc, rel=1e-12)
    assert cluster_unc == pytest.approx(10.099162001366, rel=1e-9)


def test_cluster_unc_is_nan_with_single_finite_meter():
    """A cluster with only one finite meter has nothing to estimate spread from,
    so its uncertainty stays NaN even under the effective-sample-size fallback."""
    settings = CGCorrectionSettings(
        algorithm=CorrectionAlgorithm.ODID,
        correction_cap={"enabled": False},
        outlier_rejection={"enabled": False},
        weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
    )

    _, cluster_unc, _ = _cluster_correction(
        100.0, 110.0,
        np.array([990.0]), np.array([1000.0]),
        None, 5.0, np.zeros(1), np.array([4.0]), np.zeros(1), True, settings,
    )

    assert np.isnan(cluster_unc)


# ── Fixture generation (run once, manually, JIT-on) ──────────────────────────

@pytest.mark.skipif(
    not os.environ.get("GENERATE_FIXTURES"),
    reason="regenerates committed model_correction fixtures; run manually",
)
def test_generate_correction_fixtures(
    _comstock_hourly_all, _comstock_daily_all, _comstock_monthly_all
):
    """Regenerate the real-data .npz fixtures from ComStock meters.

    GENERATE_FIXTURES=1 [GENERATE_FIXTURES_N=99] pytest -k generate_correction_fixtures
    """
    n_pool = int(os.environ.get("GENERATE_FIXTURES_N", "99"))
    min_cluster_size = int(os.environ.get("GENERATE_FIXTURES_MIN", "5"))
    build_fixtures(
        _comstock_hourly_all, _comstock_daily_all, _comstock_monthly_all,
        n_pool=n_pool, min_cluster_size=min_cluster_size,
    )


# ── Real-data snapshot across granularities x correction algorithms ──────────


# Pinned corrected reporting-period usage (mTrc) and its uncertainty, produced
# by model_correction on the committed real-ComStock fixtures. Regenerate the
# fixtures with generate_correction_fixtures (GENERATE_FIXTURES=1) if the
# upstream models change; update these alongside.
_EXPECTED = {
    ("hourly", CorrectionAlgorithm.ODID): (725413.0, 701720.3),
    ("hourly", CorrectionAlgorithm.PCTDID): (775788.8, 95833.3),
    ("daily", CorrectionAlgorithm.ODID): (723842.4, 700256.7),
    ("daily", CorrectionAlgorithm.PCTDID): (774513.8, 98120.7),
    ("billing", CorrectionAlgorithm.ODID): (723322.1, 702686.8),
    ("billing", CorrectionAlgorithm.PCTDID): (774280.6, 108783.8),
}


def _run_correction(data, algorithm, weight_cluster_aggregation=None, weight_cap=0.5):
    """Run model_correction on fixture arrays for one algorithm."""
    settings = CGCorrectionSettings(
        algorithm=algorithm,
        correction_cap={"enabled": False},
        weight_cluster_aggregation=weight_cluster_aggregation,
        weight_cap=weight_cap,
    )
    mTrc, mTrc_unc, mask = model_correction(
        float(data["oTr"]), float(data["mTr"]), data["oCGr"], data["mCGr"],
        None, float(data["mTr_unc"]), None, data["mCGr_unc"], None,
        data["CG_label"], data["T_weight"], settings,
    )

    return mTrc, mTrc_unc, mask


@pytest.mark.regression
class TestModelCorrectionRealData:
    """Snapshot model_correction on real ComStock-derived inputs.

    Inputs are frozen in committed .npz fixtures, so the correction is
    deterministic; the pinned outputs catch drift in the correction math.
    """

    GRANULARITIES = ["hourly", "daily", "billing"]

    @pytest.mark.parametrize("granularity", GRANULARITIES)
    @pytest.mark.parametrize("algorithm", [CorrectionAlgorithm.ODID, CorrectionAlgorithm.PCTDID])
    def test_corrected_value_matches_snapshot(self, granularity, algorithm, model_correction_inputs):
        """Corrected usage and uncertainty match the pinned real-data snapshot."""
        data = model_correction_inputs[granularity]
        mTrc, mTrc_unc, _ = _run_correction(data, algorithm)

        expected_mTrc, expected_unc = _EXPECTED[(granularity, algorithm)]
        assert mTrc == pytest.approx(expected_mTrc, rel=1e-4)
        assert mTrc_unc == pytest.approx(expected_unc, rel=1e-4)

    @pytest.mark.parametrize("granularity", GRANULARITIES)
    def test_correction_pulls_inflated_estimate_toward_observed(self, granularity, model_correction_inputs):
        """The DID correction moves the (over-predicting) model estimate toward
        observed: oTr < mTrc < mTr when the comparison group shares the gap."""
        data = model_correction_inputs[granularity]
        oTr, mTr = float(data["oTr"]), float(data["mTr"])
        mTrc, _, _ = _run_correction(data, CorrectionAlgorithm.PCTDID)

        assert mTr > mTrc > oTr

    @pytest.mark.parametrize("granularity", GRANULARITIES)
    def test_abspct_equals_pct_for_positive_magnitudes(self, granularity, model_correction_inputs):
        """With positive model magnitudes the absolute-percent scale equals the
        percent scale (the |.| is a no-op)."""
        data = model_correction_inputs[granularity]
        pct, pct_unc, _ = _run_correction(data, CorrectionAlgorithm.PCTDID)
        abspct, abspct_unc, _ = _run_correction(data, CorrectionAlgorithm.ABSPCTDID)

        assert abspct == pytest.approx(pct, rel=1e-9)
        assert abspct_unc == pytest.approx(pct_unc, rel=1e-9)


# Pinned (mTrc, mTrc_unc) for MODEL-weighted correction with weight_cap=1.0
# (the cap disabled), produced by model_correction on the same committed
# real-ComStock fixtures. This is the v1-equivalence anchor: at weight_cap=1.0
# the water-filling cap never triggers, so these reproduce the un-capped
# weighting exactly. Each cluster's model-magnitude weighting concentrates on
# its largest meter, dropping the Kish effective sample size below 2, so every
# cluster's uncertainty is estimated with uniform weights over its finite
# meters. The point correction stays weighted.
_EXPECTED_MODEL_WEIGHTED = {
    ("hourly", CorrectionAlgorithm.ODID): (-454686.4, 1275358.6),
    ("hourly", CorrectionAlgorithm.PCTDID): (771013.3, 110099.4),
    ("daily", CorrectionAlgorithm.ODID): (-452123.3, 1271456.5),
    ("daily", CorrectionAlgorithm.PCTDID): (770111.2, 111793.7),
    ("billing", CorrectionAlgorithm.ODID): (-451915.0, 1272270.6),
    ("billing", CorrectionAlgorithm.PCTDID): (769590.4, 121178.9),
}


@pytest.mark.regression
class TestModelCorrectionRealDataModelWeightedUncapped:
    """Snapshot model_correction on real ComStock-derived inputs with
    MODEL-weighted aggregation and weight_cap=1.0 (v1-equivalence anchor)."""

    GRANULARITIES = ["hourly", "daily", "billing"]

    @pytest.mark.parametrize("granularity", GRANULARITIES)
    @pytest.mark.parametrize("algorithm", [CorrectionAlgorithm.ODID, CorrectionAlgorithm.PCTDID])
    def test_corrected_value_matches_snapshot(self, granularity, algorithm, model_correction_inputs):
        """Corrected usage and uncertainty match the pinned real-data snapshot.
        The model-magnitude weighting concentrates on one meter per cluster,
        dropping the effective sample size below 2, so the uncertainty is the
        uniform-weight fallback — finite and positive rather than NaN."""
        data = model_correction_inputs[granularity]
        mTrc, mTrc_unc, _ = _run_correction(
            data,
            algorithm,
            weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
            weight_cap=1.0,
        )

        expected_mTrc, expected_unc = _EXPECTED_MODEL_WEIGHTED[(granularity, algorithm)]
        assert mTrc == pytest.approx(expected_mTrc, rel=1e-4)
        assert mTrc_unc == pytest.approx(expected_unc, rel=1e-4)
        assert np.isfinite(mTrc_unc) and mTrc_unc > 0


# Pinned (mTrc, mTrc_unc) for the MODEL-weighted default
# (weight_cluster_aggregation=MODEL, weight_cap=0.5), produced by
# model_correction on the same committed real-ComStock fixtures. The cap keeps
# every cluster's Kish effective sample size at or above 2, so the point
# correction and its uncertainty both stay on the weighted path.
_EXPECTED_MODEL_WEIGHTED_DEFAULT_CAP = {
    ("hourly", CorrectionAlgorithm.ODID): (79685.4, 1861504.0),
    ("hourly", CorrectionAlgorithm.PCTDID): (780930.1, 131401.2),
    ("daily", CorrectionAlgorithm.ODID): (80079.3, 1855869.4),
    ("daily", CorrectionAlgorithm.PCTDID): (779806.8, 132522.8),
    ("billing", CorrectionAlgorithm.ODID): (80135.7, 1856947.9),
    ("billing", CorrectionAlgorithm.PCTDID): (779402.1, 140519.2),
}


@pytest.mark.regression
class TestModelCorrectionRealDataModelWeighted:
    """Snapshot model_correction on real ComStock-derived inputs with the
    MODEL-weighted default (weight_cluster_aggregation=MODEL, weight_cap=0.5)."""

    GRANULARITIES = ["hourly", "daily", "billing"]

    @pytest.mark.parametrize("granularity", GRANULARITIES)
    @pytest.mark.parametrize("algorithm", [CorrectionAlgorithm.ODID, CorrectionAlgorithm.PCTDID])
    def test_corrected_value_matches_snapshot(self, granularity, algorithm, model_correction_inputs):
        """Corrected usage and uncertainty match the pinned real-data snapshot.
        The weight cap keeps every cluster's effective sample size at or above
        2, so the point correction and uncertainty both stay on the weighted
        path (finite and positive)."""
        data = model_correction_inputs[granularity]
        mTrc, mTrc_unc, _ = _run_correction(
            data, algorithm, weight_cluster_aggregation=WeightClusterAggChoice.MODEL
        )

        expected_mTrc, expected_unc = _EXPECTED_MODEL_WEIGHTED_DEFAULT_CAP[(granularity, algorithm)]
        assert mTrc == pytest.approx(expected_mTrc, rel=1e-4)
        assert mTrc_unc == pytest.approx(expected_unc, rel=1e-4)
        assert np.isfinite(mTrc_unc) and mTrc_unc > 0
