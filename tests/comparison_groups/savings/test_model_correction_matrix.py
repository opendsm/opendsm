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


import importlib

import numpy as np
import pytest

from opendsm.comparison_groups.savings.model_correction import (
    MIN_CG_METERS,
    model_correction,
    model_correction_matrix,
)
from opendsm.comparison_groups.savings.settings import (
    CGCorrectionSettings,
    CorrectionAlgorithm,
    WeightClusterAggChoice,
)



# the package re-exports a function under the module's name, so attribute lookup
# would not reach the module
_MODEL_CORRECTION_MODULE = importlib.import_module(
    "opendsm.comparison_groups.savings.model_correction"
)


def _settings(**overrides):
    base = {
        "algorithm": CorrectionAlgorithm.ABSPCTDID,
        "weight_cluster_aggregation": None,
        "correction_cap": {"enabled": False},
    }
    base.update(overrides)
    settings = CGCorrectionSettings(**base)

    return settings


def _tile_fixture(data, n_timesteps, perturb=0.0, seed=0):
    """Build a (T, M) stack from a single-timestep (M,) fixture.

    Row 0 is the untouched fixture (so scalar-vs-matrix equivalence includes the
    committed values exactly); the remaining rows are multiplicatively perturbed
    with seeded noise so cluster aggregation sees non-identical timesteps.
    """
    rng = np.random.default_rng(seed)

    M = data["oCGr"].shape[0]

    oTr = np.full(n_timesteps, float(data["oTr"]))
    mTr = np.full(n_timesteps, float(data["mTr"]))
    mTr_unc = np.full(n_timesteps, float(data["mTr_unc"]))
    oCGr = np.tile(data["oCGr"], (n_timesteps, 1))
    mCGr = np.tile(data["mCGr"], (n_timesteps, 1))
    mCGr_unc = np.tile(data["mCGr_unc"], (n_timesteps, 1))

    if perturb > 0.0:
        mTr[1:] *= 1.0 + perturb * rng.standard_normal(n_timesteps - 1)
        mTr_unc[1:] *= 1.0 + perturb * rng.standard_normal(n_timesteps - 1)
        oCGr[1:] *= 1.0 + perturb * rng.standard_normal((n_timesteps - 1, M))
        mCGr[1:] *= 1.0 + perturb * rng.standard_normal((n_timesteps - 1, M))
        mCGr_unc[1:] *= 1.0 + perturb * rng.standard_normal((n_timesteps - 1, M))

    stack = {
        "oTr": oTr,
        "mTr": mTr,
        "mTr_unc": mTr_unc,
        "oCGr": oCGr,
        "mCGr": mCGr,
        "mCGr_unc": mCGr_unc,
        "CG_label": data["CG_label"],
        "T_weight": data["T_weight"],
    }

    return stack


def _scalar_oracle(stack, settings, with_unc):
    """Run the scalar kernel per timestep as the equivalence oracle.

    Returns (mTrc[T], mTrc_unc[T], mask[T, M]) with NaN rows where the scalar
    kernel raises because a cluster has too few finite comparison meters; those
    rows are the matrix kernel's documented degradation and are excluded from
    equivalence checks. Any other scalar error propagates.
    """
    T = stack["mTr"].shape[0]
    M = stack["oCGr"].shape[1]

    mTrc = np.full(T, np.nan)
    mTrc_unc = np.full(T, np.nan)
    mask = np.zeros((T, M), dtype=bool)
    for t in range(T):
        mTr_unc_t = None
        mCGr_unc_t = None
        oCGr_unc_t = None
        CGr_corr = None
        if with_unc:
            mTr_unc_t = float(stack["mTr_unc"][t])
            mCGr_unc_t = stack["mCGr_unc"][t]

            if stack.get("oCGr_unc") is not None:
                oCGr_unc_t = stack["oCGr_unc"][t]

            CGr_corr = stack.get("CGr_corr")

        try:
            _mTrc, _unc, _mask = model_correction(
                float(stack["oTr"][t]), float(stack["mTr"][t]),
                stack["oCGr"][t], stack["mCGr"][t],
                None, mTr_unc_t, oCGr_unc_t, mCGr_unc_t, CGr_corr,
                stack["CG_label"], stack["T_weight"], settings,
            )
        except ValueError as e:
            if "insufficient length" not in str(e):
                raise

            continue

        mTrc[t] = _mTrc
        mTrc_unc[t] = _unc
        mask[t] = _mask

    return mTrc, mTrc_unc, mask


def _matrix(stack, settings, with_unc):
    """Run the matrix kernel on a (T, M) stack."""
    mTr_unc = None
    mCGr_unc = None
    oCGr_unc = None
    CGr_corr = None
    if with_unc:
        mTr_unc = stack["mTr_unc"]
        mCGr_unc = stack["mCGr_unc"]
        oCGr_unc = stack.get("oCGr_unc")
        CGr_corr = stack.get("CGr_corr")

    result = model_correction_matrix(
        stack["mTr"], stack["oCGr"], stack["mCGr"],
        mTr_unc, oCGr_unc, mCGr_unc, CGr_corr,
        stack["CG_label"], stack["T_weight"], settings,
    )

    return result


def _assert_matches_oracle(stack, settings, with_unc, atol=1e-9):
    """Assert the matrix kernel reproduces the scalar oracle on every row the
    oracle populates: point, uncertainty finiteness and value, and usage mask.
    Returns the matrix and oracle results."""
    matrix = _matrix(stack, settings, with_unc)
    oracle = _scalar_oracle(stack, settings, with_unc)
    m_mTrc, m_unc, m_mask = matrix
    s_mTrc, s_unc, s_mask = oracle

    populated = np.isfinite(s_mTrc)
    assert populated.any(), "the stack produced no timestep the scalar kernel accepts"

    np.testing.assert_allclose(
        m_mTrc[populated], s_mTrc[populated], rtol=1e-9, atol=atol,
        err_msg="matrix point correction diverged from the scalar oracle",
    )

    for t in np.flatnonzero(populated):
        assert np.isnan(s_unc[t]) == np.isnan(m_unc[t]), (
            f"unc finiteness mismatch at t={t}: scalar={s_unc[t]} matrix={m_unc[t]}"
        )
        assert (m_mask[t] == s_mask[t]).all(), (
            f"mask mismatch at t={t}: scalar={s_mask[t]} matrix={m_mask[t]}"
        )

    unc_finite = populated & np.isfinite(s_unc)
    np.testing.assert_allclose(
        m_unc[unc_finite], s_unc[unc_finite], rtol=1e-9, atol=atol,
        err_msg="matrix uncertainty diverged from the scalar oracle",
    )

    return matrix, oracle


# ── scalar-vs-matrix equivalence ──

@pytest.mark.parametrize("model_frequency", ["hourly", "daily", "billing"])
@pytest.mark.parametrize(
    "algorithm",
    [CorrectionAlgorithm.ODID, CorrectionAlgorithm.PCTDID, CorrectionAlgorithm.ABSPCTDID],
)
@pytest.mark.parametrize(
    "weight", [None, WeightClusterAggChoice.MODEL], ids=["unweighted", "model"]
)
@pytest.mark.parametrize("with_unc", [False, True], ids=["nounc", "unc"])
@pytest.mark.parametrize("capped", [False, True], ids=["nocap", "cap"])
def test_matrix_matches_scalar_on_populated_timesteps(
    model_frequency, algorithm, weight, with_unc, capped, model_correction_inputs
):
    """On timesteps where every nonzero-weight cluster has >=3 finite meters the
    matrix kernel reproduces the scalar kernel exactly (the scalar is the oracle
    there). Covers every algorithm x weighting x unc x cap combination."""
    if capped:
        cap = {"enabled": True, "type": "global", "value": 3.0, "solar_threshold": None}
    else:
        cap = {"enabled": False}

    settings = _settings(algorithm=algorithm, weight_cluster_aggregation=weight, correction_cap=cap)

    stack = _tile_fixture(model_correction_inputs[model_frequency], n_timesteps=6, perturb=0.15, seed=1)

    _assert_matches_oracle(stack, settings, with_unc, atol=1e-6)


def test_matrix_matches_scalar_dropped_absent_cluster_renormalized():
    """After a cluster absent from the pool is dropped its weight is
    redistributed so the surviving cluster weights sum to 1 (the kernel
    contract). On fully-populated timesteps the matrix kernel reproduces the
    scalar oracle for this renormalized-Sigma, three-cluster case, for both the
    point and the uncertainty."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ABSPCTDID,
        weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
    )
    stack = _constructed_stack(n_timesteps=4, seed=22)
    # three equally weighted clusters, as if a fourth was dropped and the
    # remaining weights renormalized to sum 1
    stack["CG_label"] = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    stack["T_weight"] = np.full(3, 1.0 / 3.0)
    rng = np.random.default_rng(23)
    stack["mCGr"] = 100.0 + 5.0 * rng.standard_normal((4, 9))
    stack["oCGr"] = stack["mCGr"] + 3.0 * rng.standard_normal((4, 9))
    stack["mCGr_unc"] = np.full((4, 9), 2.0)

    (_, m_unc, _), _ = _assert_matches_oracle(stack, settings, with_unc=True)

    assert np.isfinite(m_unc).any(), "no finite uncertainty to compare"


def test_weighted_std_branch_matches_scalar():
    """Model-magnitude weighting with a moderate (non-uniform, non-concentrated)
    magnitude spread keeps the effective sample size >= 2, so the weighted-std
    branch produces a finite uncertainty that must match the scalar kernel
    exactly. The fixture clusters concentrate (n_eff < 2), so this uses a
    constructed spread that actually surfaces the weighted-std value."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
    )
    stack = _constructed_stack(n_timesteps=5, seed=1)

    (_, m_unc, _), _ = _assert_matches_oracle(stack, settings, with_unc=True)

    assert np.isfinite(m_unc).all(), f"weighted-std path left a non-finite band: {m_unc}"


@pytest.mark.parametrize(
    "corr_sign", [1.0, -1.0], ids=["positive_corr", "negative_corr"]
)
def test_matrix_matches_scalar_with_observed_uncertainty_covariance(corr_sign):
    """Nonzero comparison-group observed uncertainty and its correlation with the
    model engage the covariance term ``CG_diff_var = mCGr_var + oCGr_unc**2 -
    2*cov``. The matrix kernel must reproduce the scalar oracle's per-timestep
    uncertainty for both positive and negative correlations, so a sign flip in
    that term is caught."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ABSPCTDID,
        weight_cluster_aggregation=None,
    )
    stack = _constructed_stack(n_timesteps=5, seed=20)
    M = stack["oCGr"].shape[1]
    stack["oCGr_unc"] = np.tile(np.linspace(1.0, 3.0, M), (5, 1))
    stack["CGr_corr"] = np.full(M, corr_sign * 0.6)

    (_, m_unc, _), _ = _assert_matches_oracle(stack, settings, with_unc=True)

    assert np.isfinite(m_unc).any(), "covariance test produced no finite uncertainties to compare"


# ── constructed scenarios ──

# Two clusters of four meters each, constructed so each test can place NaNs and
# magnitudes in a chosen cluster and timestep.
_LABEL = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
_TWEIGHT = np.array([0.5, 0.5])


def _constructed_stack(n_timesteps=4, seed=0):
    rng = np.random.default_rng(seed)

    M = _LABEL.shape[0]

    oTr = 100.0 + rng.standard_normal(n_timesteps)
    mTr = 110.0 + rng.standard_normal(n_timesteps)
    mTr_unc = np.full(n_timesteps, 5.0)
    mCGr = 100.0 + 5.0 * rng.standard_normal((n_timesteps, M))
    oCGr = mCGr + 3.0 * rng.standard_normal((n_timesteps, M))
    mCGr_unc = np.full((n_timesteps, M), 2.0)

    stack = {
        "oTr": oTr, "mTr": mTr, "mTr_unc": mTr_unc,
        "oCGr": oCGr, "mCGr": mCGr, "mCGr_unc": mCGr_unc,
        "CG_label": _LABEL, "T_weight": _TWEIGHT,
    }

    return stack


# ── degradation semantics ──

def test_sparse_cluster_drops_and_survivor_renormalizes():
    """A cluster with <3 finite meters drops from that timestep and the
    surviving cluster is averaged with its weight renormalized to 1. With two
    identical clusters the renormalized survivor reproduces the fully-populated
    point exactly; its correction variance carries weight 1 for the lone
    survivor, doubling the two-cluster (0.5^2 + 0.5^2) value."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)  # unweighted mean
    stack = _constructed_stack(n_timesteps=3, seed=3)
    # make the two clusters identical so the survivor equals the full-run point
    stack["oCGr"][:, 4:8] = stack["oCGr"][:, 0:4]
    stack["mCGr"][:, 4:8] = stack["mCGr"][:, 0:4]
    stack["mCGr_unc"][:, 4:8] = stack["mCGr_unc"][:, 0:4]

    m_full, u_full, _ = _matrix(stack, settings, with_unc=True)

    # drop cluster 0 to 2 finite meters at t=1; cluster 1 (identical) survives
    stack["oCGr"][1, 0] = np.nan
    stack["mCGr"][1, 1] = np.nan
    m_deg, u_deg, _ = _matrix(stack, settings, with_unc=True)

    assert np.isfinite(m_deg[1])
    assert m_deg[1] == pytest.approx(m_full[1], rel=1e-12)

    mtu2 = float(stack["mTr_unc"][1]) ** 2
    assert (u_deg[1] ** 2 - mtu2) == pytest.approx(2.0 * (u_full[1] ** 2 - mtu2), rel=1e-9)

    # untouched timesteps are unchanged
    assert m_deg[0] == pytest.approx(m_full[0], rel=1e-12)
    assert m_deg[2] == pytest.approx(m_full[2], rel=1e-12)


def test_all_clusters_degenerate_yields_nan():
    """When every cluster drops below 3 finite meters no cluster survives, so
    both the point and the uncertainty are NaN; other timesteps stay finite."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)
    stack = _constructed_stack(n_timesteps=2, seed=18)

    # t=0: both clusters fall to 2 finite meters
    stack["oCGr"][0, 0] = np.nan
    stack["oCGr"][0, 1] = np.nan
    stack["oCGr"][0, 4] = np.nan
    stack["oCGr"][0, 5] = np.nan

    m_mTrc, m_unc, _ = _matrix(stack, settings, with_unc=True)

    assert np.isnan(m_mTrc[0]) and np.isnan(m_unc[0])
    assert np.isfinite(m_mTrc[1]) and np.isfinite(m_unc[1])


def test_degraded_cluster_meters_leave_the_usage_mask():
    """A cluster that drops below three finite meters at a timestep has every
    column cleared from the usage mask there, its finite members included, while
    the surviving cluster keeps its finite columns."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)
    stack = _constructed_stack(n_timesteps=1, seed=7)
    # cluster 0 (columns 0-3) falls to 2 finite meters; cluster 1 survives
    stack["oCGr"][0, 0:2] = np.nan

    m_mTrc, _, mask = _matrix(stack, settings, with_unc=True)

    assert np.isfinite(m_mTrc[0])
    assert not mask[0, 0:4].any(), f"dropped cluster still marked used: {mask[0, 0:4]}"
    assert mask[0, 4:8].all(), f"surviving cluster not marked used: {mask[0, 4:8]}"


def test_row_with_no_surviving_cluster_uses_no_meters():
    """A timestep where every cluster drops below three finite meters is NaN and
    marks no comparison meter as used."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)
    stack = _constructed_stack(n_timesteps=1, seed=7)
    stack["oCGr"][0, 0:2] = np.nan
    stack["oCGr"][0, 4:6] = np.nan

    m_mTrc, _, mask = _matrix(stack, settings, with_unc=True)

    assert np.isnan(m_mTrc[0])
    assert not mask[0].any(), f"no-survivor row marks meters used: {mask[0]}"


def test_fully_finite_row_uses_every_meter():
    """With every meter finite and every cluster weighted, the usage mask marks
    every comparison meter."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)
    stack = _constructed_stack(n_timesteps=2, seed=7)

    _, _, mask = _matrix(stack, settings, with_unc=True)

    assert mask.all(), f"fully finite stack left meters unused: {mask}"


def test_ess_fallback_finite_unc_matches_scalar():
    """Model-magnitude weighting that concentrates one cluster onto a single
    meter drives its effective sample size below 2. The point correction stays
    weighted while the cluster's uncertainty falls back to a uniform-weight
    estimate over its finite meters, so the row uncertainty stays finite and the
    matrix kernel must reproduce the scalar oracle exactly."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
        weight_cap=1.0,
    )
    stack = _constructed_stack(n_timesteps=3, seed=4)

    # t=1: one 1e6-magnitude meter carries ~99% of cluster 1's weight; weight_cap=1.0
    # keeps the default 0.5 cap's water-fill from lifting the ESS back to 2
    stack["mCGr"][1, 4] = 1.0e6

    # the concentration must actually drive the uncapped Kish ESS below 2, or the
    # weighted path is taken and this test exercises nothing
    cluster1_mag = np.abs(stack["mCGr"][1, 4:8])
    cluster1_w = cluster1_mag / cluster1_mag.sum()
    assert 1.0 / np.sum(cluster1_w**2) < 2.0

    (m_mTrc, m_unc, _), _ = _assert_matches_oracle(stack, settings, with_unc=True)

    assert np.isfinite(m_mTrc[1])
    assert np.isfinite(m_unc[1]) and m_unc[1] > 0, f"fallback band at t=1: {m_unc[1]}"


def test_infeasible_weight_cap_uniform_fallback_matches_scalar():
    """An infeasible weight cap (`cap * M < 1`, here cap=0.2 over 4-meter
    clusters) cannot hold every model-magnitude weight at or below the cap while
    summing to 1, so both kernels fall back to uniform weights over the cluster's
    meters. The matrix kernel must reproduce the scalar oracle exactly for both
    the point and the uncertainty on this shared fallback."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ABSPCTDID,
        weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
        weight_cap=0.2,
    )
    stack = _constructed_stack(n_timesteps=5, seed=24)

    (_, m_unc, _), _ = _assert_matches_oracle(stack, settings, with_unc=True)

    assert np.isfinite(m_unc).any(), "no finite uncertainty to compare"


def test_weight_cap_magnitude_floor_matches_scalar():
    """A dominant meter beside meters at or below `weight_cap_min_magnitude` has
    no recipient for the cap's excess, so it keeps its weight. The matrix kernel
    must apply the same floor as the scalar oracle for both the point and the
    uncertainty."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
    )
    stack = _constructed_stack(n_timesteps=3, seed=6)

    # t=1: cluster 1 (meters 4-7) is one real meter beside three near-zero ones
    stack["mCGr"][1, 4:8] = [150.0, 0.02, 0.01, 0.03]
    stack["oCGr"][1, 4:8] = [140.0, 0.05, 0.0, 0.01]

    (m_mTrc, _, _), _ = _assert_matches_oracle(stack, settings, with_unc=True)

    # the floor must bind here, or this test exercises nothing: without it the
    # cap hands the near-zero meters half the cluster and the point moves
    no_floor = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
        weight_cap_min_magnitude=0.0,
    )
    m_no_floor, _, _ = _matrix(stack, no_floor, with_unc=True)

    assert m_no_floor[1] != pytest.approx(m_mTrc[1], rel=1e-3)


def test_degraded_cluster_does_not_nan_row_when_survivor_exists():
    """When one cluster degrades the row is carried by the surviving cluster
    (finite point and uncertainty), not forced to NaN. The point equals mTr
    minus the survivor's mean correction at renormalized weight 1."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)  # unweighted mean
    stack = _constructed_stack(n_timesteps=2, seed=5)

    # cluster 0 collapses to 2 finite meters at t=0; cluster 1 untouched
    stack["oCGr"][0, 0] = np.nan
    stack["mCGr"][0, 1] = np.nan

    m_mTrc, m_unc, _ = _matrix(stack, settings, with_unc=True)

    assert np.isfinite(m_mTrc[0])
    assert np.isfinite(m_unc[0])

    cg_diff1 = stack["mCGr"][0, 4:8] - stack["oCGr"][0, 4:8]
    assert m_mTrc[0] == pytest.approx(stack["mTr"][0] - cg_diff1.mean())


def test_cluster_with_nonfinite_unc_omitted_from_quadrature():
    """A cluster with a valid point but a non-finite uncertainty stays in the
    point and is dropped from the uncertainty quadrature, so the row stays finite
    while its band is understated. With two identical clusters, omitting one
    cluster's uncertainty halves the correction variance and leaves the point
    unchanged."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)  # unweighted mean
    stack = _constructed_stack(n_timesteps=2, seed=21)
    # identical clusters: equal per-cluster point and uncertainty
    stack["oCGr"][:, 4:8] = stack["oCGr"][:, 0:4]
    stack["mCGr"][:, 4:8] = stack["mCGr"][:, 0:4]
    stack["mCGr_unc"][:, 4:8] = stack["mCGr_unc"][:, 0:4]
    M = stack["oCGr"].shape[1]
    stack["oCGr_unc"] = np.full((2, M), 1.0)
    stack["CGr_corr"] = np.zeros(M)

    m_full, u_full, _ = _matrix(stack, settings, with_unc=True)

    # force cluster 0's per-meter correction variance negative (a correlation
    # above 1 is unphysical but the cleanest trigger), making its cluster
    # uncertainty non-finite while leaving its point untouched
    stack["CGr_corr"] = stack["CGr_corr"].copy()
    stack["CGr_corr"][0:4] = 2.0

    m_omit, u_omit, _ = _matrix(stack, settings, with_unc=True)

    assert np.isfinite(m_omit).all()
    assert np.isfinite(u_omit).all()
    np.testing.assert_allclose(m_omit, m_full, rtol=1e-12)

    mtu2 = stack["mTr_unc"] ** 2
    np.testing.assert_allclose(u_omit ** 2 - mtu2, 0.5 * (u_full ** 2 - mtu2), rtol=1e-9)


# ── edge cases ──

def test_zero_weight_cluster_excluded_and_masked_out():
    """A zero-weight cluster contributes nothing to the correction and its meters
    are False in the returned mask; the result equals the scalar kernel."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)

    label = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    t_weight = np.array([0.0, 0.5, 0.5])  # cluster 0 dropped

    stack = _constructed_stack(n_timesteps=3, seed=6)
    stack["CG_label"] = label
    stack["T_weight"] = t_weight
    # widen the constructed arrays to 9 meters
    rng = np.random.default_rng(7)
    stack["mCGr"] = 100.0 + 5.0 * rng.standard_normal((3, 9))
    stack["oCGr"] = stack["mCGr"] + 3.0 * rng.standard_normal((3, 9))
    stack["mCGr_unc"] = np.full((3, 9), 2.0)

    (_, _, m_mask), _ = _assert_matches_oracle(stack, settings, with_unc=True)

    assert not m_mask[:, label == 0.0].any(), "zero-weight cluster meters must be masked out"


def test_noncontiguous_labels_indexed_by_position():
    """Non-contiguous integer labels (0 and 2, skipping 1) are indexed by
    enumeration position against T_weight, matching the scalar kernel."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)

    stack = _constructed_stack(n_timesteps=3, seed=8)
    stack["CG_label"] = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])

    _assert_matches_oracle(stack, settings, with_unc=False)


def test_zero_model_magnitude_meter_stays_finite():
    """A comparison meter with zero model magnitude has an undefined percent
    scale; the guard makes it contribute no correction, so the row stays finite
    and matches the scalar kernel."""
    settings = _settings(algorithm=CorrectionAlgorithm.PCTDID)

    stack = _constructed_stack(n_timesteps=3, seed=9)
    stack["mCGr"][0, 0] = 0.0  # zero model magnitude at t=0

    (m_mTrc, _, _), _ = _assert_matches_oracle(stack, settings, with_unc=True)

    assert np.isfinite(m_mTrc[0])


def test_solar_cap_matches_scalar():
    """The solar correction cap (per-meter, sub-threshold clipping) reproduces
    the scalar kernel across the whole stack."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        correction_cap={"enabled": True, "type": "solar", "value": 0.5},
    )

    stack = _constructed_stack(n_timesteps=4, seed=10)
    # push some comparison meters below the solar threshold so the cap engages
    stack["mCGr"][:, :2] = 0.1

    _assert_matches_oracle(stack, settings, with_unc=False)


def test_algorithm_none_returns_passthrough():
    """With algorithm=None no correction is applied: mTrc == mTr per timestep,
    mTrc_unc == mTr_unc, and the mask is all False."""
    settings = _settings(algorithm=None)
    stack = _constructed_stack(n_timesteps=4, seed=11)

    m_mTrc, m_unc, m_mask = _matrix(stack, settings, with_unc=True)

    np.testing.assert_array_equal(m_mTrc, stack["mTr"])
    np.testing.assert_array_equal(m_unc, stack["mTr_unc"])
    assert m_mask.shape == stack["oCGr"].shape
    assert m_mask.dtype == bool
    assert not m_mask.any()


def test_algorithm_none_uncertainty_nan_without_input():
    """Algorithm=None with no treatment model uncertainty yields NaN unc."""
    settings = _settings(algorithm=None)
    stack = _constructed_stack(n_timesteps=3, seed=12)

    _, m_unc, _ = _matrix(stack, settings, with_unc=False)

    assert np.isnan(m_unc).all()


def test_outlier_rejection_fallback_matches_scalar(model_correction_inputs):
    """With outlier rejection enabled the per-timestep fallback path reproduces
    the scalar kernel on well-populated timesteps."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        outlier_rejection={"enabled": True},
    )
    stack = _tile_fixture(model_correction_inputs["daily"], n_timesteps=5, perturb=0.15, seed=13)

    _assert_matches_oracle(stack, settings, with_unc=True, atol=1e-6)


def test_outlier_rejection_fallback_drops_sparse_cluster():
    """The fallback path drops a cluster that falls below 3 finite meters, clears
    its meters from the usage mask, and lets the surviving cluster carry the
    timestep with a finite point and band."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        outlier_rejection={"enabled": True},
    )
    stack = _constructed_stack(n_timesteps=1, seed=14)
    # cluster 0 falls to 2 finite meters; cluster 1 survives
    stack["oCGr"][0, 0] = np.nan
    stack["mCGr"][0, 1] = np.nan

    m_mTrc, m_unc, mask = _matrix(stack, settings, with_unc=True)

    assert np.isfinite(m_mTrc[0]) and np.isfinite(m_unc[0])
    assert not mask[0, 0:4].any(), f"dropped cluster still marked used: {mask[0, 0:4]}"
    assert mask[0, 4:8].sum() >= 3, f"survivor marks too few meters: {mask[0, 4:8]}"


def test_outlier_rejection_fallback_drops_cluster_rejection_leaves_sparse(monkeypatch):
    """A cluster with 4 finite meters that outlier rejection cuts to 2 is dropped
    from the timestep and cleared from the usage mask; the surviving cluster alone
    carries the row, matching the row where that cluster is sparse before rejection."""

    def reject_large(x, weights=None, sigma_threshold=3, quantile=0.25, transform=None):
        keep = np.abs(x) <= 1e3

        return x[keep], np.flatnonzero(keep)

    monkeypatch.setattr(_MODEL_CORRECTION_MODULE, "remove_outliers", reject_large)
    settings = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        outlier_rejection={"enabled": True},
    )
    stack = _constructed_stack(n_timesteps=1, seed=14)
    reference = _constructed_stack(n_timesteps=1, seed=14)
    stack["oCGr"][0, 0:2] += 1e6
    reference["oCGr"][0, 0:2] = np.nan

    m_mTrc, m_unc, mask = _matrix(stack, settings, with_unc=True)
    r_mTrc, r_unc, _ = _matrix(reference, settings, with_unc=True)

    assert not mask[0, 0:4].any(), f"rejected cluster still marked used: {mask[0, 0:4]}"
    assert mask[0, 4:8].all(), f"survivor meters not all marked used: {mask[0, 4:8]}"
    np.testing.assert_allclose([m_mTrc[0], m_unc[0]], [r_mTrc[0], r_unc[0]], rtol=1e-12)


def test_outlier_rejection_fallback_row_with_no_survivor_is_nan():
    """On the fallback path a timestep where every cluster falls below 3 finite
    meters is NaN in point and band and marks no comparison meter as used."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        outlier_rejection={"enabled": True},
    )
    stack = _constructed_stack(n_timesteps=1, seed=14)
    stack["oCGr"][0, 0:2] = np.nan
    stack["oCGr"][0, 4:6] = np.nan

    m_mTrc, m_unc, mask = _matrix(stack, settings, with_unc=True)

    assert np.isnan(m_mTrc[0]) and np.isnan(m_unc[0])
    assert not mask[0].any(), f"no-survivor row marks meters used: {mask[0]}"


def test_outlier_rejection_fallback_nan_treatment_row_uses_no_meters():
    """On the per-timestep fallback path a non-finite treatment model leaves the
    row NaN and marks no comparison meter as used."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.ODID,
        outlier_rejection={"enabled": True},
    )
    stack = _constructed_stack(n_timesteps=2, seed=21)
    stack["mTr"][1] = np.nan

    m_mTrc, _, mask = _matrix(stack, settings, with_unc=True)

    assert np.isfinite(m_mTrc[0])
    assert np.isnan(m_mTrc[1])
    assert not mask[1].any()


# ── float32 input ──

def test_float32_input_promoted_to_float64_math(model_correction_inputs):
    """float32 inputs are cast to float64 for the math; the result matches the
    float64-input result within the float32 input-rounding tolerance, and the
    outputs are float64."""
    settings = _settings(
        algorithm=CorrectionAlgorithm.PCTDID,
        weight_cluster_aggregation=WeightClusterAggChoice.MODEL,
    )
    stack = _tile_fixture(model_correction_inputs["daily"], n_timesteps=6, perturb=0.15, seed=15)

    m64, u64, _ = model_correction_matrix(
        stack["mTr"], stack["oCGr"], stack["mCGr"],
        stack["mTr_unc"], None, stack["mCGr_unc"], None,
        stack["CG_label"], stack["T_weight"], settings,
    )
    m32, u32, _ = model_correction_matrix(
        stack["mTr"].astype(np.float32),
        stack["oCGr"].astype(np.float32), stack["mCGr"].astype(np.float32),
        stack["mTr_unc"].astype(np.float32), None,
        stack["mCGr_unc"].astype(np.float32), None,
        stack["CG_label"], stack["T_weight"], settings,
    )

    assert m64.dtype == np.float64
    assert m32.dtype == np.float64

    populated = np.isfinite(m64)
    np.testing.assert_allclose(m32[populated], m64[populated], rtol=1e-5, atol=1e-2)


# ── validation ──

def test_fewer_than_min_cg_meters_is_hard_error():
    """Fewer than `MIN_CG_METERS` comparison meters is a hard error even in the
    degrade-friendly matrix kernel."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)
    M = MIN_CG_METERS - 1
    oCGr = np.ones((3, M))
    with pytest.raises(ValueError, match=f"fewer than {MIN_CG_METERS}"):
        model_correction_matrix(
            np.ones(3), oCGr, oCGr,
            None, None, None, None,
            np.zeros(M), np.array([1.0]), settings,
        )


def test_min_cg_meters_is_accepted():
    """Exactly `MIN_CG_METERS` comparison meters in one cluster form a finite
    correction."""
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)
    M = MIN_CG_METERS
    oCGr = np.ones((3, M))

    mTrc, _, _ = model_correction_matrix(
        np.full(3, 2.0), oCGr, oCGr + 0.5,
        None, None, None, None,
        np.zeros(M), np.array([1.0]), settings,
    )

    np.testing.assert_allclose(mTrc, np.full(3, 1.5), rtol=0, atol=1e-12)


def test_t_weight_length_must_match_cluster_count():
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)
    stack = _constructed_stack(n_timesteps=3, seed=16)
    with pytest.raises(ValueError, match="T_weight"):
        model_correction_matrix(
            stack["mTr"], stack["oCGr"], stack["mCGr"],
            None, None, None, None,
            stack["CG_label"], np.array([1.0]), settings,
        )


def test_all_zero_weight_clusters_is_error():
    settings = _settings(algorithm=CorrectionAlgorithm.ODID)
    stack = _constructed_stack(n_timesteps=3, seed=17)
    with pytest.raises(ValueError, match="zero weight"):
        model_correction_matrix(
            stack["mTr"], stack["oCGr"], stack["mCGr"],
            None, None, None, None,
            stack["CG_label"], np.array([0.0, 0.0]), settings,
        )
