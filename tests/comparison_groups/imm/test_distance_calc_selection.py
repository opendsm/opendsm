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

import logging
import os
import random

import numpy as np
import pandas as pd
import pytest


def _total_memory_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (AttributeError, ValueError):
        return float("inf")

from opendsm.comparison_groups.individual_meter_matching.settings import Settings
from opendsm.comparison_groups.individual_meter_matching.distance_calc_selection import (
    DistanceMatching,
    _distances,
    _iter_chunks,
)


def generate_group(n_entries, make_random=True, non_random_value=5, id_prefix="t"):
    return pd.DataFrame(
        [
            {
                "id": f"{id_prefix}_{i}",
                "month_1": random.random() if make_random else non_random_value,
                "month_2": random.random() if make_random else non_random_value,
                "month_3": random.random() if make_random else non_random_value,
            }
            for i in range(1, n_entries + 1)
        ]
    ).set_index("id")


def test_distance_match():
    random.seed(1)
    n_treatment = 10
    n_pool = 100
    n_matches_per_treatment = 4
    allow_duplicate_matches = False

    comparison_pool = pd.DataFrame(
        [
            {
                "id": f"c_{i}",
                "month_1": random.random(),
                "month_2": random.random(),
                "month_3": random.random(),
            }
            for i in range(1, n_pool + 1)
        ]
    ).set_index("id")
    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=True, id_prefix="c")
    for selection_method in ["minimize_meter_distance", "minimize_loadshape_distance"]:
        settings = Settings(
            selection_method=selection_method,
            n_matches_per_treatment=n_matches_per_treatment,
            allow_duplicate_matches=allow_duplicate_matches,
        )
        IMM = DistanceMatching(
            settings=settings
        )
        
        comparison_group = IMM.get_comparison_group(
            treatment_group=treatment_group,
            comparison_pool=comparison_pool
        )
        assert not comparison_group.empty


def test_distance_match_duplicates_allowed():
    random.seed(1)
    n_treatment = 10
    n_pool = 5
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = True
    n_matches_per_treatment = 1

    # this will run out of comparison pool meters and therefore still have duplicates
    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=False, id_prefix="c")
    settings = Settings(
        selection_method=selection_method,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    assert comparison_group["duplicated"].any()


def test_distance_match_duplicates_forbidden():
    random.seed(1)
    n_treatment = 8
    n_pool = 10
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = False
    n_matches_per_treatment = 1

    # this will run through the 'duplicates' loop several times before finding unique values
    # however since here are more 'max runs allowed' than treatment meters, it will be
    # able to iterate enough times to find unique matches

    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=False)
    settings = Settings(
        selection_method=selection_method,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    assert not comparison_group["duplicated"].any()


@pytest.mark.skipif(
    _total_memory_gb() < 12,
    reason="Needs >12 GB RAM; large allocations OOM-kill the process on smaller machines",
)
def test_distance_match_large_treatments():
    random.seed(1)

    n_treatment = 10000
    n_pool = 20000
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = False
    n_matches_per_treatment = 1
    n_pool_meters_per_chunk = 5000

    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=True, id_prefix="c")
    settings = Settings(
        selection_method=selection_method,
        n_pool_meters_per_chunk=n_pool_meters_per_chunk,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    assert not comparison_group.empty


def test_distance_duplicate_best_match():
    n_treatment = 2
    n_pool = 2
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = False
    n_matches_per_treatment = 1

    t_ids = ["far", "close"]
    t_vals = [10, 1]
    c_ids = ["match_1", "match_2"]
    c_vals = [2, 100]
    treatment_group = pd.DataFrame({"id": t_ids, "month_1": t_vals}).set_index("id")
    comparison_pool = pd.DataFrame({"id": c_ids, "month_1": c_vals}).set_index("id")

    settings = Settings(
        selection_method=selection_method,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    comparison_group.set_index("id", inplace=True)

    assert comparison_group.loc["match_1", "treatment"] == "close"
    assert comparison_group.loc["match_2", "treatment"] == "far"


def test_multiple_meter_matches():
    random.seed(1)

    n_treatment = 8
    n_pool = 2000
    selection_method = "minimize_meter_distance"
    allow_duplicate_matches = False
    n_matches_per_treatment = 5

    # this will run through the 'duplicates' loop several times before finding unique values
    # however since here are more 'max runs allowed' than treatment meters, it will be
    # able to iterate enough times to find unique matches

    treatment_group = generate_group(n_treatment, make_random=True)
    comparison_pool = generate_group(n_pool, make_random=True)
    settings = Settings(
        selection_method=selection_method,
        n_matches_per_treatment=n_matches_per_treatment,
        allow_duplicate_matches=allow_duplicate_matches,
    )
    IMM = DistanceMatching(
        settings=settings
    )
    comparison_group = IMM.get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool
    )
    assert not comparison_group["duplicated"].any()
    assert len(comparison_group) == 40
    assert comparison_group.index.nunique() == 40
    assert comparison_group.treatment.value_counts().nunique() == 1


# ---------------------------------------------------------------------------
# Regression tests: verify exact outputs so refactoring cannot silently change
# results. All inputs are fully deterministic (no random state).
# ---------------------------------------------------------------------------

def test_iter_chunks_exact():
    """_iter_chunks yields non-overlapping chunks of exactly the requested size."""
    arr = np.arange(10)
    chunks = list(_iter_chunks(arr, 3))
    assert len(chunks) == 4
    np.testing.assert_array_equal(chunks[0], [0, 1, 2])
    np.testing.assert_array_equal(chunks[1], [3, 4, 5])
    np.testing.assert_array_equal(chunks[2], [6, 7, 8])
    np.testing.assert_array_equal(chunks[3], [9])


def test_iter_chunks_exact_even():
    """_iter_chunks with an evenly divisible length produces no remainder chunk."""
    arr = np.arange(6)
    chunks = list(_iter_chunks(arr, 3))
    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0], [0, 1, 2])
    np.testing.assert_array_equal(chunks[1], [3, 4, 5])


def test_distances_exact_euclidean():
    """_distances returns correct Euclidean distances for a known input."""
    ls_t = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ls_cp = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    result = _distances(ls_t, ls_cp)

    expected = np.array([
        [0.0,        np.sqrt(2), np.sqrt(2)],
        [np.sqrt(2), 0.0,        np.sqrt(2)],
    ])
    np.testing.assert_array_almost_equal(result, expected)


def test_distances_exact_with_weights():
    """_distances with weights matches manually scaled cdist."""
    from scipy.spatial.distance import cdist

    ls_t  = np.array([[1.0, 2.0], [3.0, 4.0]])
    ls_cp = np.array([[1.5, 2.5], [0.5, 1.5], [2.5, 3.5]])
    weights = np.array([2.0, 0.5])

    result = _distances(ls_t, ls_cp, weights=weights)

    expected = cdist(ls_t * weights, ls_cp * weights)
    np.testing.assert_array_almost_equal(result, expected)


def test_distances_chunking_consistency():
    """Chunked and non-chunked _distances produce identical results (no weights)."""
    rng = np.random.default_rng(42)
    ls_t  = rng.random((5, 4))
    ls_cp = rng.random((11, 4))

    result_full   = _distances(ls_t, ls_cp, n_meters_per_chunk=1000)
    result_chunked = _distances(ls_t, ls_cp, n_meters_per_chunk=3)

    np.testing.assert_array_almost_equal(result_full, result_chunked)


def test_distances_chunking_consistency_with_weights():
    """Chunked and non-chunked _distances produce identical results (with weights)."""
    rng = np.random.default_rng(42)
    ls_t    = rng.random((5, 4))
    ls_cp   = rng.random((11, 4))
    weights = rng.random(4) + 0.1  # positive weights

    result_full    = _distances(ls_t, ls_cp, weights=weights, n_meters_per_chunk=1000)
    result_chunked = _distances(ls_t, ls_cp, weights=weights, n_meters_per_chunk=3)

    np.testing.assert_array_almost_equal(result_full, result_chunked)


def test_closest_idx_duplicates_allowed_exact():
    """_closest_idx_duplicates_allowed returns the correct top-k indices."""
    distances = np.array([
        [5.0, 1.0, 3.0, 2.0],  # closest 2: indices 1 (1.0) and 3 (2.0)
        [2.0, 3.0, 1.0, 4.0],  # closest 2: indices 2 (1.0) and 0 (2.0)
    ])
    settings = Settings(n_matches_per_treatment=2, allow_duplicate_matches=True)
    imm = DistanceMatching(settings=settings)

    idx = imm._closest_idx_duplicates_allowed(distances, n_match=2)

    # argpartition order within top-k is not guaranteed; sort before comparing
    assert set(idx[0]) == {1, 3}
    assert set(idx[1]) == {0, 2}


def test_closest_idx_duplicates_allowed_exact_multi():
    """_closest_idx_duplicates_allowed handles n_match strictly less than pool size."""
    # 5 pool meters, request 4 — kth=4 is valid for a 5-element array
    distances = np.array([[5.0, 1.0, 3.0, 2.0, 4.0]])
    settings = Settings(n_matches_per_treatment=4, allow_duplicate_matches=True)
    imm = DistanceMatching(settings=settings)

    idx = imm._closest_idx_duplicates_allowed(distances, n_match=4)

    assert idx.shape == (1, 4)
    # 4 closest: indices 1(1.0), 3(2.0), 2(3.0), 4(4.0); index 0 (5.0) is excluded
    assert set(idx[0]) == {1, 3, 2, 4}


def test_closest_idx_duplicates_allowed_n_match_equals_pool_size():
    """_closest_idx_duplicates_allowed works when n_match equals pool size."""
    distances = np.array([[3.0, 1.0, 2.0]])  # 1 treatment, 3 pool meters
    settings = Settings(n_matches_per_treatment=3, allow_duplicate_matches=True)
    imm = DistanceMatching(settings=settings)

    idx = imm._closest_idx_duplicates_allowed(distances, n_match=3)

    assert idx.shape == (1, 3)
    assert set(idx[0]) == {0, 1, 2}


def test_get_comparison_group_exact_no_duplicates():
    """get_comparison_group returns exact rows for a fully determined 2×2 case."""
    # "far"   (10) vs match_1 (2)  → dist=8,  match_2 (100) → dist=90
    # "close" (1)  vs match_1 (2)  → dist=1,  match_2 (100) → dist=99
    # linear_sum_assignment (n_match=1): assigns "far"→match_2 (90), "close"→match_1 (1)
    treatment_group = pd.DataFrame(
        {"id": ["far", "close"], "month_1": [10.0, 1.0]}
    ).set_index("id")
    comparison_pool = pd.DataFrame(
        {"id": ["match_1", "match_2"], "month_1": [2.0, 100.0]}
    ).set_index("id")

    settings = Settings(
        selection_method="minimize_meter_distance",
        n_matches_per_treatment=1,
        allow_duplicate_matches=False,
    )
    cg = DistanceMatching(settings=settings).get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool,
    )

    expected = pd.DataFrame({
        "id":          ["match_2", "match_1"],
        "treatment":   ["far",     "close"],
        "distance":    [90.0,      1.0],
        "duplicated":  [False,     False],
    })
    pd.testing.assert_frame_equal(
        cg.reset_index(drop=True),
        expected,
        check_like=False,  # preserve column and row order
        check_dtype=False,  # distances are now float32 for memory efficiency
    )


def test_get_comparison_group_exact_with_max_distance():
    """max_distance_threshold filters rows whose distance exceeds the limit."""
    treatment_group = pd.DataFrame(
        {"id": ["far", "close"], "month_1": [10.0, 1.0]}
    ).set_index("id")
    comparison_pool = pd.DataFrame(
        {"id": ["match_1", "match_2"], "month_1": [2.0, 100.0]}
    ).set_index("id")

    settings = Settings(
        selection_method="minimize_meter_distance",
        n_matches_per_treatment=1,
        allow_duplicate_matches=False,
        max_distance_threshold=50.0,  # filters out the far→match_2 pair (dist=90)
    )
    cg = DistanceMatching(settings=settings).get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool,
    )

    assert len(cg) == 1
    assert cg.iloc[0]["id"] == "match_1"
    assert cg.iloc[0]["treatment"] == "close"
    assert cg.iloc[0]["distance"] == pytest.approx(1.0)


def test_get_comparison_group_n_match_reduction():
    """n_match is silently reduced when n_match * n_treatment > n_pool."""
    # n_treatment=3, n_pool=5, n_match=4 → capped to floor(5/3)=1
    rng = np.random.default_rng(0)
    n_treatment, n_pool = 3, 5
    treatment_group = pd.DataFrame(
        rng.random((n_treatment, 2)),
        index=[f"t_{i}" for i in range(n_treatment)],
        columns=["a", "b"],
    )
    comparison_pool = pd.DataFrame(
        rng.random((n_pool, 2)),
        index=[f"c_{i}" for i in range(n_pool)],
        columns=["a", "b"],
    )
    settings = Settings(
        selection_method="minimize_meter_distance",
        n_matches_per_treatment=4,
        allow_duplicate_matches=False,
    )
    cg = DistanceMatching(settings=settings).get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool,
    )
    # Effective n_match = floor(5/3) = 1 → 3 rows total
    assert len(cg) == n_treatment * 1
    assert not cg["duplicated"].any()


def test_get_comparison_group_duplicated_flag():
    """The 'duplicated' column is True when a pool meter is matched more than once.

    3 treatments × 1 match from a 2-meter pool guarantees at least one pool
    meter is re-used (pigeonhole principle). n_pool=2 keeps kth=1 in bounds.
    """
    # t_1(1.0) and t_2(1.1) both pull toward c_1(1.05)/c_2(1.08);
    # t_3(5.0) also pulls c_2 — by distance, c_2 ends up matched twice.
    treatment_group = pd.DataFrame(
        {"id": ["t_1", "t_2", "t_3"], "month_1": [1.0, 1.1, 5.0]}
    ).set_index("id")
    comparison_pool = pd.DataFrame(
        {"id": ["c_1", "c_2"], "month_1": [1.05, 1.08]}
    ).set_index("id")

    settings = Settings(
        selection_method="minimize_meter_distance",
        n_matches_per_treatment=1,
        allow_duplicate_matches=True,
    )
    cg = DistanceMatching(settings=settings).get_comparison_group(
        treatment_group=treatment_group,
        comparison_pool=comparison_pool,
    )
    assert len(cg) == 3
    assert cg["duplicated"].any()


def test_no_duplicates_greedy_fills_all_treatments():
    """Regression: no treatment is silently dropped when its nearest-candidate
    block is exhausted by earlier assignments. Every treatment must be filled
    via the full-pool fallback while the pool lasts.

    All treatments and a 'close' pool block sit near 0, so every treatment's
    candidate block is the same close meters. The first treatments consume them;
    the rest must fall back to the 'far' pool rather than be dropped."""
    rng = np.random.default_rng(0)
    n_treatment = 20
    n_match = 2

    treatment = pd.DataFrame(
        {"id": [f"t_{i}" for i in range(n_treatment)], "month_1": rng.normal(0, 0.01, n_treatment)}
    ).set_index("id")
    close = rng.normal(0, 0.01, 20)
    far = rng.normal(100, 0.01, 980)
    pool = pd.DataFrame(
        {"id": [f"c_{i}" for i in range(1000)], "month_1": np.concatenate([close, far])}
    ).set_index("id")

    settings = Settings(
        selection_method="minimize_meter_distance",
        n_matches_per_treatment=n_match,
        allow_duplicate_matches=False,
        candidate_multiplier=None,  # isolate the matching logic from the prefilter
    )
    cg = DistanceMatching(settings=settings).get_comparison_group(treatment, pool)

    assert cg["treatment"].nunique() == n_treatment
    assert len(cg) == n_treatment * n_match
    assert not cg["duplicated"].any()


def test_prefilter_keeps_neighbors_for_multimodal_treatments():
    """Regression: the per-treatment kNN prefilter preserves the true nearest
    neighbours of treatments in different regions. A single-centroid prefilter
    would keep the meters near the mean (~50) and drop each cluster's real
    matches; per-treatment kNN keeps both, so matches stay within-cluster."""
    treatment = pd.DataFrame(
        {"id": ["a0", "a1", "b0", "b1"], "month_1": [0.0, 0.0, 100.0, 100.0]}
    ).set_index("id")
    pool_vals = [0.0] * 10 + [100.0] * 10 + [50.0] * 80  # decoys cluster at the centroid
    pool = pd.DataFrame(
        {"id": [f"c_{i}" for i in range(len(pool_vals))], "month_1": pool_vals}
    ).set_index("id")

    settings = Settings(
        selection_method="minimize_meter_distance",
        n_matches_per_treatment=1,
        allow_duplicate_matches=False,
        candidate_multiplier=2,  # n_treatment*k = 8 < pool of 100, so prefilter activates
    )
    cg = DistanceMatching(settings=settings).get_comparison_group(treatment, pool)

    assert (cg["distance"] < 1.0).all()


def test_n_match_reduction_warns(caplog):
    """A pool too small to supply unique matches reduces n_match and warns."""
    treatment = generate_group(4, make_random=True)
    pool = generate_group(6, make_random=True, id_prefix="c")
    settings = Settings(
        selection_method="minimize_meter_distance",
        n_matches_per_treatment=4,
        allow_duplicate_matches=False,
        candidate_multiplier=None,
    )

    with caplog.at_level(logging.WARNING):
        DistanceMatching(settings=settings).get_comparison_group(treatment, pool)

    assert any("Reduced matches per treatment" in r.message for r in caplog.records)
