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
import pandas as pd
import pytest

from opendsm.common.metrics import compute_baseline_profile
from opendsm.common.stats.period_kernel import cell_index
from opendsm.comparison_groups.population import (
    PooledLaws,
    pooled_laws_from_profiles,
)
from opendsm.comparison_groups.savings.ndid import ndid_correction_matrix
from opendsm.comparison_groups.savings.settings import NDIDSettings




def _laws(n_clusters=1, n_cells=2, v=0.04, spread=0.1, lam=1.0, acf=(0.5, 0.2),
          gamma_used=0.0, S_ref=1.0):
    laws = PooledLaws(
        v=np.broadcast_to(np.asarray(v, dtype=float), (n_clusters, n_cells)).copy(),
        v_pop=np.full(n_cells, 0.04),
        spread=np.broadcast_to(np.asarray(spread, dtype=float), (n_clusters, n_cells)).copy(),
        lam=np.full(n_clusters, lam),
        acf=np.tile(np.asarray(acf, dtype=float), (n_clusters, 1)),
        gamma_fit=np.nan,
        gamma_used=gamma_used,
        S_ref=S_ref,
        cluster_sizes=np.ones(n_clusters, dtype=int),
        profile_settings_seen=[],
    )

    return laws


def _pool(T=6, M=4, seed=0):
    rng = np.random.default_rng(seed)
    case = {
        "mTr": rng.uniform(5.0, 10.0, T),
        "mTr_unc": rng.uniform(0.5, 1.0, T),
        "s_Tr": rng.uniform(4.0, 8.0, T),
        "S_Tr": 1.5,
        "mCGr": rng.uniform(1.0, 3.0, (T, M)),
        "s": rng.uniform(1.0, 3.0, (T, M)),
        "S": rng.uniform(0.5, 3.0, M),
        "q": np.arange(T) % 2,
        "cg_label": np.zeros(M, dtype=int),
        "t_weight": np.array([1.0]),
    }
    case["oCGr"] = case["mCGr"] * rng.uniform(0.8, 1.2, (T, M))

    return case


def _run(case, laws, settings, **kwargs):
    result = ndid_correction_matrix(
        case["mTr"], case["mTr_unc"], case["s_Tr"], case["S_Tr"], case["oCGr"],
        case["mCGr"], case["s"], case["S"], case["q"], case["cg_label"], case["t_weight"],
        laws, settings, **kwargs,
    )

    return result


def _columns(case, cols):
    sub = dict(case)
    for key in ("oCGr", "mCGr", "s"):
        sub[key] = case[key][:, cols]

    for key in ("S", "cg_label"):
        sub[key] = case[key][cols]

    return sub


GAMMA0 = NDIDSettings(gamma=0.0)


def test_plain_mean_identity():
    case = _pool(T=24, M=6, seed=1)
    case["cg_label"] = np.array([0, 0, 0, 1, 1, 1])
    case["t_weight"] = np.array([0.3, 0.7])
    case["oCGr"][3, 1] = np.nan
    case["mCGr"][5, 4] = np.nan
    for key in ("oCGr", "mCGr", "s"):
        case[key] = case[key].astype(np.float32)

    settings = NDIDSettings(gamma=0.0, state_bandwidth_factor=1e6)
    mTrc, _, _, _, _, mask = _run(case, _laws(n_clusters=2), settings, trust_constant=np.inf)

    m = case["mCGr"].astype(np.float64)
    e = (m - case["oCGr"].astype(np.float64)) / case["s"].astype(np.float64)
    mean_0 = np.nanmean(e[:, :3], axis=1)
    mean_1 = np.nanmean(e[:, 3:], axis=1)
    expected = 0.3 * mean_0 + 0.7 * mean_1

    np.testing.assert_allclose((case["mTr"] - mTrc) / case["s_Tr"], expected, rtol=0, atol=1e-9)
    assert not mask[3, 1] and not mask[5, 4]
    assert mask.sum() == mask.size - 2


def test_single_meter_cluster_returns_its_residual_and_pooled_variance():
    case = _pool(T=5, M=1, seed=2)
    laws = _laws(v=[[0.01, 0.04]])
    mTrc, corrected_unc, cg_var, n_eff, _, _ = _run(case, laws, GAMMA0)

    e = (case["mCGr"][:, 0] - case["oCGr"][:, 0]) / case["s"][:, 0]
    np.testing.assert_allclose(mTrc, case["mTr"] - case["s_Tr"] * e, rtol=1e-12)
    np.testing.assert_allclose(cg_var / case["s_Tr"] ** 2, laws.v[0, case["q"]], rtol=1e-12)
    np.testing.assert_allclose(corrected_unc, np.sqrt(case["mTr_unc"] ** 2 + cg_var), rtol=1e-12)
    np.testing.assert_array_equal(n_eff, 1.0)


def test_two_identical_meters_give_kish_two_and_cross_sectional_variance():
    case = _pool(T=4, M=2, seed=3)
    case["mCGr"][:, 1] = case["mCGr"][:, 0]
    case["s"][:, 1] = case["s"][:, 0]
    case["S"][1] = case["S"][0]
    _, _, cg_var, n_eff, _, _, trust = _run(case, _laws(), GAMMA0, return_trust=True)

    e = (case["mCGr"] - case["oCGr"]) / case["s"]
    se2 = (e[:, 0] - e[:, 1]) ** 2 / 4

    np.testing.assert_array_equal(trust, 1.0)
    np.testing.assert_array_equal(n_eff, 2.0)
    np.testing.assert_allclose(cg_var / case["s_Tr"] ** 2, se2, rtol=1e-12)


def test_hand_computed_weighted_median_and_trust():
    # Equal states, so w0 is equal across meters; lam = 1 makes the transform the identity.
    # e = [0, 1, 10]: med 1, mad 1.4826, z = [-0.6745, 0, 6.0704], trust_3 = 3 * 1.4826 / 9.
    case = {
        "mTr": np.array([1.0]),
        "mTr_unc": np.array([0.0]),
        "s_Tr": np.array([2.0]),
        "S_Tr": 1.0,
        "mCGr": np.ones((1, 3)),
        "oCGr": np.array([[1.0, 0.0, -9.0]]),
        "s": np.ones((1, 3)),
        "S": np.ones(3),
        "q": np.array([0]),
        "cg_label": np.zeros(3, dtype=int),
        "t_weight": np.array([1.0]),
    }
    mTrc, corrected_unc, cg_var, n_eff, _, mask, trust = _run(
        case, _laws(), GAMMA0, return_trust=True
    )

    np.testing.assert_allclose(trust, [[1.0, 1.0, 0.4942]], rtol=1e-12)
    np.testing.assert_allclose(mTrc, 1.0 - 2.0 * 2.382326998636837, rtol=1e-12)
    np.testing.assert_allclose(n_eff, 2.7720080160637823, rtol=1e-12)
    np.testing.assert_allclose(cg_var, 4.0 * 5.4714669027315885, rtol=1e-12)
    np.testing.assert_allclose(corrected_unc, np.sqrt(cg_var), rtol=1e-12)
    assert mask.all()


def test_size_law_and_size_relevance_weights():
    case = {
        "mTr": np.array([1.0]),
        "mTr_unc": np.array([0.0]),
        "s_Tr": np.array([1.0]),
        "S_Tr": 1.0,
        "mCGr": np.ones((1, 2)),
        "oCGr": np.array([[1.0, 0.0]]),
        "s": np.ones((1, 2)),
        "S": np.array([1.0, 4.0]),
        "q": np.array([0]),
        "cg_label": np.zeros(2, dtype=int),
        "t_weight": np.array([1.0]),
    }
    laws = _laws(gamma_used=0.5, S_ref=2.0)

    mTrc = _run(case, laws, GAMMA0)[0]
    np.testing.assert_allclose(1.0 - mTrc, 2.0 / 2.5, rtol=1e-12)

    sized = NDIDSettings(gamma=0.0, size_bandwidth=np.log(4.0))
    mTrc = _run(case, laws, sized)[0]
    np.testing.assert_allclose(1.0 - mTrc, 1.0 / 1.5, rtol=1e-12)


def test_zero_spread_matches_only_equal_states():
    case = _pool(T=2, M=2, seed=4)
    case["s"][:] = 1.0
    case["s_Tr"][:] = 1.0
    case["mCGr"][0] = [case["mTr"][0], case["mTr"][0] + 1.0]
    case["mCGr"][1] = case["mTr"][1] + 1.0
    mTrc, _, _, n_eff, _, mask = _run(case, _laws(spread=0.0), GAMMA0)

    e = case["mCGr"][0, 0] - case["oCGr"][0, 0]
    np.testing.assert_allclose(mTrc[0], case["mTr"][0] - e, rtol=1e-12)
    np.testing.assert_array_equal(mask, [[True, False], [False, False]])
    assert np.isnan(mTrc[1]) and n_eff[1] == 0


def test_row_rules():
    case = _pool(T=6, M=4, seed=5)
    case["mTr"][0] = np.nan
    case["s_Tr"][1] = np.inf
    case["s_Tr"][2] = 0.0
    case["oCGr"][3] = np.nan
    case["mTr_unc"][4] = np.nan
    mTrc, corrected_unc, cg_var, n_eff, _, mask, trust = _run(
        case, _laws(), GAMMA0, return_trust=True
    )

    for t in (0, 1, 3):
        assert np.isnan(mTrc[t]) and np.isnan(corrected_unc[t]) and np.isnan(cg_var[t])

    assert mTrc[2] == case["mTr"][2]
    assert cg_var[2] == 0.0
    assert corrected_unc[2] == case["mTr_unc"][2]

    assert np.isfinite(mTrc[4]) and np.isfinite(cg_var[4]) and np.isnan(corrected_unc[4])
    assert np.isfinite([mTrc[5], cg_var[5], corrected_unc[5]]).all()

    np.testing.assert_array_equal(n_eff[:4], 0.0)
    assert not mask[:4].any() and not trust[:4].any()
    assert mask[4:].all()


def test_zero_t_weight_cluster_is_skipped():
    case = _pool(T=6, M=4, seed=6)
    case["cg_label"] = np.array([0, 0, 1, 1])
    case["t_weight"] = np.array([1.0, 0.0])
    full = _run(case, _laws(n_clusters=2), GAMMA0, return_trust=True)

    sub = _columns(case, [0, 1])
    sub["t_weight"] = np.array([1.0])
    alone = _run(sub, _laws(), GAMMA0)

    for got, want in zip(full[:4], alone[:4]):
        np.testing.assert_allclose(got, want, rtol=1e-12)

    assert not full[5][:, 2:].any() and not full[6][:, 2:].any()


def test_relevance_zero_removes_meter():
    case = _pool(T=6, M=4, seed=7)
    relevance = np.array([1.0, 1.0, 0.0, 1.0])
    full = _run(case, _laws(), GAMMA0, relevance=relevance)
    alone = _run(_columns(case, [0, 1, 3]), _laws(), GAMMA0)

    for got, want in zip(full[:4], alone[:4]):
        np.testing.assert_allclose(got, want, rtol=1e-12)

    assert not full[5][:, 2].any()


def test_negative_label_excluded():
    case = _pool(T=6, M=4, seed=8)
    case["cg_label"] = np.array([0, -1, 0, 0])
    full = _run(case, _laws(), GAMMA0, return_trust=True)
    alone = _run(_columns(case, [0, 2, 3]), _laws(), GAMMA0)

    for got, want in zip(full[:4], alone[:4]):
        np.testing.assert_allclose(got, want, rtol=1e-12)

    assert not full[5][:, 1].any() and not full[6][:, 1].any()


def test_single_timestep():
    case = _pool(T=1, M=3, seed=9)
    mTrc, corrected_unc, cg_var, n_eff, cg_acf, mask = _run(case, _laws(), GAMMA0)

    assert mTrc.shape == corrected_unc.shape == cg_var.shape == n_eff.shape == (1,)
    assert mask.shape == (1, 3)
    assert np.isfinite([mTrc[0], corrected_unc[0], cg_var[0]]).all()
    np.testing.assert_allclose(cg_acf, [0.5, 0.2])


def test_cg_acf_is_t_weighted_mean():
    case = _pool(T=3, M=4, seed=10)
    case["cg_label"] = np.array([0, 0, 1, 1])
    case["t_weight"] = np.array([0.25, 0.75])
    laws = _laws(n_clusters=2)
    laws.acf = np.array([[0.4, 0.2, 0.0], [0.8, 0.4, 0.1]])
    cg_acf = _run(case, laws, GAMMA0)[4]

    np.testing.assert_allclose(cg_acf, [0.7, 0.35, 0.075], rtol=1e-12)


def test_return_trust_bounds():
    case = _pool(T=8, M=12, seed=11)
    case["oCGr"][:, 5] = case["mCGr"][:, 5] - 50.0 * case["s"][:, 5]
    case["oCGr"][2, 3] = np.nan
    settings = NDIDSettings(gamma=0.0, state_bandwidth_factor=1e6)
    *_, mask, trust = _run(case, _laws(lam=0.7), settings, return_trust=True)

    assert ((trust >= 0) & (trust <= 1)).all()
    assert trust[2, 3] == 0.0 and not mask[2, 3]
    assert (trust[:, 5] < 1).all()
    assert (trust[mask] > 0).all()


def test_near_zero_pool_prediction_gives_bounded_correction():
    case = _pool(T=6, M=5, seed=12)
    case["mCGr"][:, 0] = 1e-9
    mTrc, _, cg_var, _, _, _ = _run(case, _laws(), GAMMA0)

    e = (case["mCGr"] - case["oCGr"]) / case["s"]
    correction = (case["mTr"] - mTrc) / case["s_Tr"]
    assert np.isfinite(mTrc).all() and np.isfinite(cg_var).all()
    assert (np.abs(correction) <= np.abs(e).max(axis=1) + 1e-12).all()


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("t_weight", np.array([0.5, 0.5]), "t_weight"),
        ("s", np.ones((6, 3)), "'s'"),
        ("S", np.ones(3), "'S'"),
        ("q", np.arange(6) % 3, "cells"),
        ("q", np.zeros(6), "integer"),
        ("cg_label", np.array([0, 0, 1, 1]), "clusters"),
    ],
)
def test_invalid_inputs_raise(key, value, match):
    case = _pool(T=6, M=4, seed=13)
    if key == "cg_label":
        case["t_weight"] = np.array([0.5, 0.5])

    case[key] = value
    with pytest.raises(ValueError, match=match):
        _run(case, _laws(), GAMMA0)


def test_profile_laws_and_kernel_run_end_to_end_on_a_synthetic_pool():
    """Profiles from four months of hourly baseline fits, laws pooled from them and the
    kernel on the following week give finite, bounded corrections with a positive band."""
    rng = np.random.default_rng(0)
    index = pd.date_range(
        "2024-01-01", "2024-05-08", freq="h", tz="America/Los_Angeles", inclusive="left"
    )
    baseline = np.asarray(index < pd.Timestamp("2024-05-01", tz="America/Los_Angeles"))
    window = ~baseline
    n_pool = 8
    shape = 1.0 + 0.5 * np.cos((index.hour.to_numpy() - 15) / 24 * 2 * np.pi)
    size = rng.uniform(5.0, 50.0, n_pool + 1)
    predicted = size[None, :] * shape[:, None]
    observed = predicted * (1 + 0.1 * rng.standard_normal(predicted.shape))

    profiles = [
        compute_baseline_profile(
            pd.DataFrame(
                {"observed": observed[baseline, j], "predicted": predicted[baseline, j]},
                index=index[baseline],
            ),
            "hourly",
        )
        for j in range(n_pool + 1)
    ]
    pool, treatment = profiles[:n_pool], profiles[n_pool]
    settings = NDIDSettings(sector="commercial")
    q = cell_index(index[window], "hourly")
    S = np.array([profile.annual_scale for profile in pool])
    cg_label = np.zeros(n_pool, dtype=int)
    laws = pooled_laws_from_profiles(pool, cg_label, S, settings)
    s = np.column_stack([profile.typical_load[q] for profile in pool])
    s_Tr = treatment.typical_load[q]
    mTr = predicted[window, n_pool]

    mTrc, corrected_unc, cg_var, cg_effective_n, cg_acf, mask = ndid_correction_matrix(
        mTr, np.zeros_like(mTr), s_Tr, treatment.annual_scale, observed[window, :n_pool],
        predicted[window, :n_pool], s, S, q, cg_label, np.array([1.0]), laws, settings,
    )

    assert np.isfinite(mTrc).all()
    assert (cg_var > 0).all()
    assert np.array_equal(corrected_unc, np.sqrt(cg_var))
    assert (np.abs(mTr - mTrc) / s_Tr < 1.0).all()
    assert (cg_effective_n > 1).all()
    assert cg_acf.shape == (168,)
    assert mask.shape == (window.sum(), n_pool)
