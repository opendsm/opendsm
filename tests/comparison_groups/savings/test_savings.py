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

from opendsm.common.const import default_season_def
from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.population import ComparisonPool, TreatmentGroup
from opendsm.comparison_groups.savings.correction import CorrectionResult, correct_reporting
from opendsm.comparison_groups.savings.savings import SavingsResult, compute_savings
from opendsm.comparison_groups.selection import (
    ComparisonGroupSelection,
    SelectionMethod,
    _finalize_clusters,
    _normalize_treatment_weights,
)



_TZ = "America/Chicago"


def _index(n, start="2019-01-01", freq="h"):
    return pd.date_range(start, periods=n, freq=freq, tz=_TZ)


def _meter_frame(
    mid, index, observed, corrected, corrected_unc, modeled=None, modeled_unc=None, observed_unc=0.0
):
    if modeled is None:
        modeled = corrected

    if modeled_unc is None:
        modeled_unc = corrected_unc

    n = len(index)
    frame = pd.DataFrame(
        {
            "id": [mid] * n,
            "datetime": index,
            "observed": np.asarray(observed, dtype=np.float64) * np.ones(n),
            "modeled": np.asarray(modeled, dtype=np.float64) * np.ones(n),
            "modeled_unc": np.asarray(modeled_unc, dtype=np.float64) * np.ones(n),
            "corrected": np.asarray(corrected, dtype=np.float64) * np.ones(n),
            "corrected_unc": np.asarray(corrected_unc, dtype=np.float64) * np.ones(n),
            "observed_unc": np.asarray(observed_unc, dtype=np.float64) * np.ones(n),
        }
    )

    return frame


def _empty_corrected():
    frame = pd.DataFrame(
        {
            "id": pd.Series([], dtype="object"),
            "datetime": pd.Series([], dtype=f"datetime64[ns, {_TZ}]"),
            "observed": pd.Series([], dtype="float64"),
            "modeled": pd.Series([], dtype="float64"),
            "modeled_unc": pd.Series([], dtype="float64"),
            "corrected": pd.Series([], dtype="float64"),
            "corrected_unc": pd.Series([], dtype="float64"),
            "observed_unc": pd.Series([], dtype="float64"),
        }
    )

    return frame


def _correction(
    corrected, tz=_TZ, granularity="hourly", correction_periods=None, meter_id="m1", cg_acf=None
):
    result = CorrectionResult(
        meter_id=meter_id,
        corrected=corrected,
        cg_usage=pd.DataFrame(columns=["id", "datetime", "cg_usage_fraction"]),
        exclusions=exclusions.empty_ledger(),
        settings={},
        covered_window=None,
        tz=tz,
        granularity=granularity,
        fingerprint="test-fingerprint",
        correction_periods=correction_periods,
        cg_acf=cg_acf,
    )

    return result


def _billing_read(mid, start, end, observed, corrected, corrected_unc):
    """One read-period row for a billing treatment meter: ``datetime`` is the
    period start, and the matching ``correction_periods`` entry supplies the inclusive
    ``(start, end)`` daily bounds the day-distribution expands over."""
    start = pd.Timestamp(start, tz=_TZ)
    end = pd.Timestamp(end, tz=_TZ)
    frame = _meter_frame(mid, pd.DatetimeIndex([start]), observed, corrected, corrected_unc)
    periods = [(start, end)]

    return frame, periods


def _row(frame, mid, period):
    matches = frame[(frame["id"] == mid) & (frame["period"] == period)]
    assert len(matches) == 1, f"expected exactly one row for ({mid}, {period}), got {len(matches)}"

    return matches.iloc[0]


# ── sign convention & closed-form uncertainty ────────────────────────────────


def test_savings_sign_is_corrected_minus_observed():
    corrected = _meter_frame("m1", _index(3), observed=10.0, corrected=12.0, corrected_unc=0.0)
    result = compute_savings(_correction(corrected))

    assert np.allclose(result.savings["savings"].to_numpy(), 2.0)


def test_negative_savings_when_corrected_below_observed():
    corrected = _meter_frame("m1", _index(3), observed=12.0, corrected=10.0, corrected_unc=0.0)
    result = compute_savings(_correction(corrected))

    assert np.allclose(result.savings["savings"].to_numpy(), -2.0)


def test_savings_unc_matches_quadrature_closed_form():
    corrected = _meter_frame("m1", _index(1), observed=0.0, corrected=10.0, corrected_unc=3.0)
    result = compute_savings(_correction(corrected), observed_unc={"m1": 4.0})

    assert result.savings["savings_unc"].to_numpy()[0] == pytest.approx(5.0)


def test_observed_unc_defaults_to_zero_when_not_supplied():
    corrected = _meter_frame("m1", _index(1), observed=0.0, corrected=10.0, corrected_unc=3.0)
    result = compute_savings(_correction(corrected))

    assert result.savings["savings_unc"].to_numpy()[0] == pytest.approx(3.0)


def test_observed_unc_missing_id_defaults_to_zero():
    corrected = _meter_frame("m1", _index(1), observed=0.0, corrected=10.0, corrected_unc=3.0)
    result = compute_savings(_correction(corrected), observed_unc={"some_other_meter": 100.0})

    assert result.savings["savings_unc"].to_numpy()[0] == pytest.approx(3.0)


# ── per-timestep DataFrame observed_unc alignment ────────────────────────────


def test_dataframe_observed_unc_aligns_distinct_values_per_timestep():
    """A ``[id, datetime, observed_unc]`` frame supplies a distinct observed
    uncertainty per timestep, matched on ``[id, datetime]``. With zero corrected
    uncertainty the native savings_unc equals the supplied per-timestep value."""
    index = _index(3, freq="D")
    frame = _meter_frame("m1", index, observed=0.0, corrected=10.0, corrected_unc=0.0)
    observed_unc = pd.DataFrame(
        {"id": ["m1", "m1", "m1"], "datetime": index, "observed_unc": [3.0, 4.0, 5.0]}
    )
    result = compute_savings(_correction(frame), observed_unc=observed_unc, aggregation="native")

    savings_rows = result.savings.sort_values("period").reset_index(drop=True)
    np.testing.assert_allclose(savings_rows["savings_unc"].to_numpy(), [3.0, 4.0, 5.0])


def test_dataframe_observed_unc_unmatched_rows_default_to_zero():
    """Timesteps absent from the observed_unc frame contribute 0 uncertainty."""
    index = _index(3, freq="D")
    frame = _meter_frame("m1", index, observed=0.0, corrected=10.0, corrected_unc=0.0)
    observed_unc = pd.DataFrame(
        {"id": ["m1"], "datetime": [index[1]], "observed_unc": [4.0]}
    )
    result = compute_savings(_correction(frame), observed_unc=observed_unc, aggregation="native")

    savings_rows = result.savings.sort_values("period").reset_index(drop=True)
    np.testing.assert_allclose(savings_rows["savings_unc"].to_numpy(), [0.0, 4.0, 0.0])


def test_dataframe_observed_unc_extra_rows_are_ignored():
    """Rows in the observed_unc frame with no matching meter-timestep in the
    correction are dropped by the left join, not injected as new rows."""
    index = _index(2, freq="D")
    frame = _meter_frame("m1", index, observed=0.0, corrected=10.0, corrected_unc=0.0)
    ghost = _index(1, start="2020-06-01", freq="D")
    observed_unc = pd.DataFrame(
        {
            "id": ["m1", "m1", "ghost"],
            "datetime": [index[0], index[1], ghost[0]],
            "observed_unc": [3.0, 5.0, 999.0],
        }
    )
    result = compute_savings(_correction(frame), observed_unc=observed_unc, aggregation="native")

    savings_rows = result.savings.sort_values("period").reset_index(drop=True)
    assert len(savings_rows) == 2
    np.testing.assert_allclose(savings_rows["savings_unc"].to_numpy(), [3.0, 5.0])


# ── pct_savings and its zero-corrected guard ─────────────────────────────────


def test_pct_savings_is_savings_over_corrected():
    corrected = _meter_frame("m1", _index(1), observed=8.0, corrected=10.0, corrected_unc=0.0)
    result = compute_savings(_correction(corrected))

    assert result.savings["pct_savings"].to_numpy()[0] == pytest.approx(0.2)


def test_pct_savings_is_nan_when_corrected_is_zero():
    corrected = _meter_frame("m1", _index(1), observed=0.0, corrected=0.0, corrected_unc=0.0)
    result = compute_savings(_correction(corrected))

    assert np.isnan(result.savings["pct_savings"].to_numpy()[0])
    assert result.savings["savings"].to_numpy()[0] == pytest.approx(0.0)


# ── float64 accumulation ──────────────────────────────────────────────────────


def test_time_rollup_accumulates_in_float64_not_float32():
    n = 200_000
    corrected32 = np.full(n, 0.1, dtype=np.float32)
    frame = pd.DataFrame(
        {
            "id": ["m1"] * n,
            "datetime": _index(n),
            "observed": np.zeros(n, dtype=np.float32),
            "modeled": corrected32,
            "modeled_unc": np.zeros(n, dtype=np.float32),
            "corrected": corrected32,
            "corrected_unc": np.zeros(n, dtype=np.float32),
            "observed_unc": np.zeros(n, dtype=np.float32),
        }
    )
    result = compute_savings(_correction(frame), aggregation="total")

    total_savings = result.savings["savings"].to_numpy()[0]
    # A naive float32 accumulator drifts by ~2e-3 over this many additions;
    # float64 accumulation stays within a few parts in 1e4.
    assert total_savings == pytest.approx(n * 0.1, abs=5e-4)


# ── coverage and NaN propagation ──────────────────────────────────────────────


def test_nan_corrected_timestep_excluded_from_sum_with_partial_coverage():
    index = _index(4, freq="D")
    corrected = np.array([10.0, np.nan, 10.0, 10.0])
    corrected_unc = np.array([1.0, np.nan, 1.0, 1.0])
    frame = _meter_frame(
        "m1", index, observed=np.zeros(4), corrected=corrected, corrected_unc=corrected_unc
    )
    result = compute_savings(_correction(frame), aggregation="total")
    row = result.savings.iloc[0]

    assert row["coverage"] == pytest.approx(0.75)
    assert row["savings"] == pytest.approx(30.0)
    assert row["corrected"] == pytest.approx(30.0)
    assert row["savings_unc"] == pytest.approx(np.sqrt(3.0))


def test_nan_observed_timestep_excluded_from_sum_with_partial_coverage():
    """A missing observed value under a finite correction is a coverage hole,
    not a poisoned period: the timestep leaves every sum, the band included,
    and ``coverage`` reports it."""
    index = _index(4, freq="D")
    observed = np.array([2.0, np.nan, 2.0, 2.0])
    frame = _meter_frame("m1", index, observed=observed, corrected=10.0, corrected_unc=1.0)
    result = compute_savings(_correction(frame), aggregation="total")
    row = result.savings.iloc[0]

    assert row["coverage"] == pytest.approx(0.75)
    assert row["observed"] == pytest.approx(6.0)
    assert row["corrected"] == pytest.approx(30.0)
    assert row["savings"] == pytest.approx(24.0)
    assert row["savings_unc"] == pytest.approx(np.sqrt(3.0))


def test_all_nan_period_reports_nan_not_zero():
    index = _index(3, freq="D")
    corrected = np.full(3, np.nan)
    frame = _meter_frame(
        "m1", index, observed=np.zeros(3), corrected=corrected, corrected_unc=np.full(3, np.nan)
    )
    result = compute_savings(_correction(frame), aggregation="total")
    row = result.savings.iloc[0]

    assert row["coverage"] == pytest.approx(0.0)
    assert np.isnan(row["savings"])
    assert np.isnan(row["corrected"])
    assert np.isnan(row["observed"])
    assert np.isnan(row["pct_savings"])
    assert np.isnan(row["savings_unc"])


def test_nan_unc_timestep_propagates_nan_to_aggregate_unc_even_with_full_coverage():
    index = _index(2, freq="D")
    corrected = np.array([10.0, 10.0])
    corrected_unc = np.array([1.0, np.nan])
    frame = _meter_frame(
        "m1", index, observed=np.zeros(2), corrected=corrected, corrected_unc=corrected_unc
    )
    result = compute_savings(_correction(frame), aggregation="total")
    row = result.savings.iloc[0]

    # Both timesteps have finite corrected values, so coverage is full and the
    # point savings sum is unaffected...
    assert row["coverage"] == pytest.approx(1.0)
    assert row["savings"] == pytest.approx(20.0)
    # ...but one non-finite uncertainty timestep still poisons the aggregate band.
    assert np.isnan(row["savings_unc"])


# ── aggregation boundaries ────────────────────────────────────────────────────


def test_monthly_aggregation_groups_by_calendar_month():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2019-01-15", tz=_TZ),
            pd.Timestamp("2019-01-20", tz=_TZ),
            pd.Timestamp("2019-02-01", tz=_TZ),
        ]
    )
    frame = _meter_frame(
        "m1", index, observed=np.zeros(3), corrected=np.array([10.0, 10.0, 5.0]), corrected_unc=np.zeros(3)
    )
    result = compute_savings(_correction(frame), aggregation="monthly")

    assert _row(result.savings, "m1", "2019-01")["savings"] == pytest.approx(20.0)
    assert _row(result.savings, "m1", "2019-02")["savings"] == pytest.approx(5.0)


def test_seasonal_aggregation_uses_default_season_definition():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2019-01-15", tz=_TZ),  # winter
            pd.Timestamp("2019-11-15", tz=_TZ),  # winter
            pd.Timestamp("2019-06-15", tz=_TZ),  # summer
        ]
    )
    frame = _meter_frame(
        "m1", index, observed=np.zeros(3), corrected=np.array([10.0, 30.0, 20.0]), corrected_unc=np.zeros(3)
    )
    result = compute_savings(_correction(frame), aggregation="seasonal")

    assert _row(result.savings, "m1", "2019-winter")["savings"] == pytest.approx(40.0)
    assert _row(result.savings, "m1", "2019-summer")["savings"] == pytest.approx(20.0)


def test_seasonal_aggregation_rejects_season_def_with_unmapped_month():
    frame = _meter_frame("m1", _index(1), observed=0.0, corrected=1.0, corrected_unc=0.0)
    season_def = {k: v for k, v in default_season_def.items() if k != "June"}

    with pytest.raises(ValueError, match="June"):
        compute_savings(_correction(frame), aggregation="seasonal", season_def=season_def)


def test_annual_aggregation_groups_by_calendar_year():
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2019-12-31", tz=_TZ),
            pd.Timestamp("2020-01-01", tz=_TZ),
        ]
    )
    frame = _meter_frame(
        "m1", index, observed=np.zeros(2), corrected=np.array([10.0, 5.0]), corrected_unc=np.zeros(2)
    )
    result = compute_savings(_correction(frame), aggregation="annual")

    assert _row(result.savings, "m1", "2019")["savings"] == pytest.approx(10.0)
    assert _row(result.savings, "m1", "2020")["savings"] == pytest.approx(5.0)


def test_total_aggregation_collapses_to_a_single_period_per_meter():
    frame = _meter_frame(
        "m1", _index(5, freq="D"), observed=np.zeros(5), corrected=np.full(5, 2.0), corrected_unc=np.zeros(5)
    )
    result = compute_savings(_correction(frame), aggregation="total")

    assert len(result.savings) == 1
    assert result.savings["savings"].to_numpy()[0] == pytest.approx(10.0)


def test_tables_exposes_the_savings_frame_as_a_copy():
    index = _index(2, freq="D")
    frame = _meter_frame("m1", index, observed=1.0, corrected=2.0, corrected_unc=0.5)
    result = compute_savings(_correction(frame))

    tables = result.tables
    tables["savings"].loc[0, "savings"] = 99.0

    assert set(tables) == {"savings"}
    assert result.savings.loc[0, "savings"] == pytest.approx(1.0)


def test_invalid_aggregation_raises():
    frame = _meter_frame("m1", _index(1), observed=0.0, corrected=1.0, corrected_unc=0.0)

    with pytest.raises(ValueError):
        compute_savings(_correction(frame), aggregation="not_a_real_aggregation")


# ── billing day-distribution ──────────────────────────────────────────────────


def test_billing_native_output_is_per_read_rows():
    """Native cadence equals correction cadence: a billing meter's native output
    is one row per read period, each carrying that read's own total savings."""
    frame1, period1 = _billing_read("m1", "2019-01-01", "2019-01-31", 10.0, 15.0, 1.0)
    frame2, period2 = _billing_read("m1", "2019-02-01", "2019-02-28", 20.0, 25.0, 1.0)
    frame = pd.concat([frame1, frame2], ignore_index=True)
    correction_periods = period1 + period2
    result = compute_savings(
        _correction(frame, granularity="billing", correction_periods=correction_periods),
        aggregation="native",
    )

    assert len(result.savings) == 2
    savings = result.savings.sort_values("period")["savings"].to_numpy()
    np.testing.assert_allclose(savings, [5.0, 5.0])


def test_billing_straddling_read_prorated_into_monthly_buckets():
    """A read spanning Jan 20–Feb 8 (12 Jan days, 8 Feb days) pro-rates its
    savings and variance by day count into each calendar month. Variance is
    spread uniformly, so each month's band is the day-count share of the read's
    variance; coverage is full because every expanded day is finite."""
    frame, period = _billing_read(
        "m1", "2019-01-20", "2019-02-08", observed=60.0, corrected=100.0, corrected_unc=np.sqrt(20.0)
    )
    correction = _correction(frame, granularity="billing", correction_periods=period)
    result = compute_savings(correction, aggregation="monthly")

    jan = _row(result.savings, "m1", "2019-01")
    feb = _row(result.savings, "m1", "2019-02")

    assert jan["savings"] == pytest.approx(24.0)  # 40 * 12/20
    assert jan["corrected"] == pytest.approx(60.0)  # 100 * 12/20
    assert jan["observed"] == pytest.approx(36.0)  # 60 * 12/20
    assert jan["savings_unc"] == pytest.approx(np.sqrt(12.0))  # 20 * 12/20
    assert jan["coverage"] == pytest.approx(1.0)
    assert jan["pct_savings"] == pytest.approx(0.4)

    assert feb["savings"] == pytest.approx(16.0)  # 40 * 8/20
    assert feb["corrected"] == pytest.approx(40.0)  # 100 * 8/20
    assert feb["savings_unc"] == pytest.approx(np.sqrt(8.0))  # 20 * 8/20
    assert feb["coverage"] == pytest.approx(1.0)


def test_billing_calendar_rollup_preserves_read_total_and_variance():
    """The day-distribution is additive: monthly bucket variances sum back to the
    single read's variance, and a full-span total reproduces the read's own
    savings and band exactly (no pro-ration when nothing straddles the bucket)."""
    frame, period = _billing_read(
        "m1", "2019-01-20", "2019-02-08", observed=60.0, corrected=100.0, corrected_unc=np.sqrt(20.0)
    )
    correction_periods = period

    monthly = compute_savings(
        _correction(frame, granularity="billing", correction_periods=correction_periods), aggregation="monthly"
    ).savings
    total = compute_savings(
        _correction(frame, granularity="billing", correction_periods=correction_periods), aggregation="total"
    ).savings

    bucket_var = np.sum(monthly["savings_unc"].to_numpy() ** 2)
    assert bucket_var == pytest.approx(20.0)

    assert total["savings"].to_numpy()[0] == pytest.approx(40.0)
    assert total["savings_unc"].to_numpy()[0] == pytest.approx(np.sqrt(20.0))
    assert total["corrected"].to_numpy()[0] == pytest.approx(100.0)


def test_billing_read_within_single_month_is_exact():
    """A read that lands entirely inside one calendar month contributes its
    total and full variance to that month, unmodified by the day-distribution."""
    frame, period = _billing_read(
        "m1", "2019-02-01", "2019-02-28", observed=50.0, corrected=80.0, corrected_unc=3.0
    )
    result = compute_savings(
        _correction(frame, granularity="billing", correction_periods=period), aggregation="monthly"
    )
    feb = _row(result.savings, "m1", "2019-02")

    assert feb["savings"] == pytest.approx(30.0)
    assert feb["savings_unc"] == pytest.approx(3.0)
    assert feb["coverage"] == pytest.approx(1.0)


def test_billing_multiple_reads_in_bucket_combine_variance_in_quadrature():
    """Two finite reads landing in the same month sum their savings and combine
    their variances in quadrature (variances add additively across reads)."""
    frame1, period1 = _billing_read("m1", "2019-01-01", "2019-01-15", 0.0, 15.0, 3.0)
    frame2, period2 = _billing_read("m1", "2019-01-16", "2019-01-31", 0.0, 16.0, 4.0)
    frame = pd.concat([frame1, frame2], ignore_index=True)
    correction_periods = period1 + period2
    result = compute_savings(
        _correction(frame, granularity="billing", correction_periods=correction_periods), aggregation="monthly"
    )
    jan = _row(result.savings, "m1", "2019-01")

    assert jan["savings"] == pytest.approx(31.0)
    assert jan["savings_unc"] == pytest.approx(5.0)  # sqrt(9 + 16)
    assert jan["coverage"] == pytest.approx(1.0)


def test_billing_nan_read_lowers_day_grain_coverage_and_leaves_the_band_to_the_finite_read():
    """Coverage is measured at day grain: a NaN read's days are non-finite, so a
    month split between a finite and a NaN read reports the finite day fraction
    and excludes the NaN read from the point sum and from the band alike."""
    frame1, period1 = _billing_read("m1", "2019-01-01", "2019-01-15", 0.0, 15.0, 1.0)
    frame2, period2 = _billing_read("m1", "2019-01-16", "2019-01-31", 0.0, np.nan, np.nan)
    frame = pd.concat([frame1, frame2], ignore_index=True)
    correction_periods = period1 + period2
    result = compute_savings(
        _correction(frame, granularity="billing", correction_periods=correction_periods), aggregation="monthly"
    )
    jan = _row(result.savings, "m1", "2019-01")

    assert jan["coverage"] == pytest.approx(15.0 / 31.0)
    assert jan["savings"] == pytest.approx(15.0)
    assert jan["savings_unc"] == pytest.approx(1.0)


# ── calendar-rollup day-distribution, end to end (slow, ComStock) ────────────


def _one_cluster_selection(treatment, pool):
    """A hand-built one-cluster selection (all pool meters, equal weight) so
    the fixture doesn't depend on the clustering algorithm's minimum sample
    size."""
    clusters_flat = pd.DataFrame(
        {"id": pool.ids, "cluster": 0, "weight": 1.0, "treatment": pd.NA}
    )
    clusters = _finalize_clusters(clusters_flat, SelectionMethod.CG_CLUSTERING)

    weights_frame = pd.DataFrame(
        {tid: {"pct_cluster_0": 1.0} for tid in treatment.ids}
    ).T
    weights_frame.index.name = "id"
    treatment_weights = _normalize_treatment_weights(weights_frame)

    selection = ComparisonGroupSelection(
        clusters=clusters,
        treatment_weights=treatment_weights,
        method="cg_clustering",
        basis="error",
        method_settings={},
        data_settings=None,
        exclusions=exclusions.empty_ledger(),
        treatment_ids=list(treatment.ids),
        pool_ids=list(pool.ids),
        tz=treatment.tz,
    )

    return selection


@pytest.fixture(scope="module")
def billing_rollup_result(comstock, model_bank):
    """A real ComStock billing correction with explicit read boundaries offset
    from calendar-month starts, so every read straddles two calendar months and
    the monthly calendar rollup genuinely pro-rates rather than passing reads
    through unmodified."""
    ids = comstock.ids("billing")

    treatment = TreatmentGroup.from_fit_models(comstock.meters(model_bank, "billing", ids[:2]))
    pool = ComparisonPool.from_fit_models(comstock.meters(model_bank, "billing", ids[2:8]))
    selection = _one_cluster_selection(treatment, pool)

    boundaries = [pd.Timestamp(f"2019-{month:02d}-15", tz=treatment.tz) for month in range(1, 13)]
    correction = correct_reporting(
        selection, treatment, pool, treatment.ids[0], read_boundaries=boundaries
    )

    return correction


def test_billing_native_cadence_equals_correction_cadence(billing_rollup_result):
    """Native aggregation carries one row per read period of the corrected
    meter, matching the correction's received cadence exactly."""
    native = compute_savings(billing_rollup_result, aggregation="native").savings
    meter_id = billing_rollup_result.meter_id

    assert len(native[native["id"] == meter_id]) == len(billing_rollup_result.correction_periods)


def test_billing_calendar_rollup_reproduces_native_total_end_to_end(billing_rollup_result):
    """A real correction whose reads straddle calendar months: the monthly
    calendar rollup pro-rates each read across buckets, but every treatment
    meter's monthly savings sum back to its native (per-read) total and its
    single ``total`` aggregation, exactly as the day-distribution's additivity
    guarantees."""
    native = compute_savings(billing_rollup_result, aggregation="native").savings
    monthly = compute_savings(billing_rollup_result, aggregation="monthly").savings
    total = compute_savings(billing_rollup_result, aggregation="total").savings

    # exactly one calendar year of monthly buckets per meter, despite the
    # read boundaries never falling on a calendar-month start
    assert (monthly.groupby("id").size() == 12).all()

    for tid in billing_rollup_result.corrected["id"].unique():
        native_total = native.loc[native["id"] == tid, "savings"].sum()
        monthly_total = monthly.loc[monthly["id"] == tid, "savings"].sum()
        expected_total = total.loc[total["id"] == tid, "savings"].to_numpy()[0]

        assert monthly_total == pytest.approx(native_total, rel=1e-6)
        assert monthly_total == pytest.approx(expected_total, rel=1e-6)

    assert monthly["coverage"].between(0.0, 1.0).all()


# ── empty / degenerate ────────────────────────────────────────────────────────


def test_empty_correction_yields_empty_frames_without_error():
    result = compute_savings(_correction(_empty_corrected()))

    assert result.savings.empty
    assert list(result.savings.columns) == [
        "id",
        "period",
        "observed",
        "corrected",
        "savings",
        "savings_unc",
        "pct_savings",
        "coverage",
    ]


# ── NDID long-run variance ────────────────────────────────────────────────────


def _ndid_frame(index, corrected_unc, cg_var, observed_unc=0.0):
    n = len(index)
    corrected = np.linspace(10.0, 20.0, n)
    frame = _meter_frame(
        "m1",
        index,
        observed=corrected - 1.0,
        corrected=corrected,
        corrected_unc=corrected_unc,
        observed_unc=observed_unc,
    )
    frame["cg_var"] = np.asarray(cg_var, dtype=np.float64) * np.ones(n)

    return frame


def _savings_unc(frame, aggregation="total", **kwargs):
    result = compute_savings(_correction(frame, **kwargs), aggregation=aggregation)
    unc = result.savings["savings_unc"].to_numpy(dtype=np.float64)

    return unc


@pytest.mark.parametrize("aggregation", ["native", "monthly", "seasonal", "total"])
def test_ndid_zero_lag_acf_reproduces_quadrature_path(aggregation):
    rng = np.random.default_rng(0)
    index = _index(24 * 70)
    corrected_unc = rng.uniform(1.0, 2.0, len(index))
    cg_var = corrected_unc**2 * rng.uniform(0.1, 0.9, len(index))
    frame = _ndid_frame(index, corrected_unc, cg_var, observed_unc=0.3)

    ndid = _savings_unc(frame, aggregation, cg_acf=np.zeros(4))
    legacy = _savings_unc(frame.drop(columns=["cg_var"]), aggregation)

    np.testing.assert_allclose(ndid, legacy, rtol=1e-12, atol=0.0)


def test_ndid_billing_empty_acf_reproduces_quadrature_path_with_read_expansion():
    frame1, period1 = _billing_read("m1", "2019-01-20", "2019-02-08", 10.0, 15.0, 2.0)
    frame2, period2 = _billing_read("m1", "2019-02-09", "2019-03-10", 20.0, 25.0, 3.0)
    frame = pd.concat([frame1, frame2], ignore_index=True)
    frame["cg_var"] = [1.0, 4.0]
    periods = period1 + period2

    ndid = compute_savings(
        _correction(frame, granularity="billing", correction_periods=periods, cg_acf=np.array([])),
        aggregation="monthly",
    )
    legacy = compute_savings(
        _correction(
            frame.drop(columns=["cg_var"]), granularity="billing", correction_periods=periods
        ),
        aggregation="monthly",
    )

    assert len(ndid.savings) == 3
    pd.testing.assert_frame_equal(ndid.savings, legacy.savings)


def test_ndid_constant_sigma_ar1_matches_truncated_closed_form():
    n, k, phi = 50, 5, 0.6
    model_unc, cg_unc, observed_unc = 1.5, 2.0, 0.5
    corrected_unc = np.sqrt(model_unc**2 + cg_unc**2)
    frame = _ndid_frame(_index(n), corrected_unc, cg_unc**2)
    cg_acf = phi ** np.arange(1, k + 1)

    result = compute_savings(
        _correction(frame, cg_acf=cg_acf), observed_unc={"m1": observed_unc}, aggregation="total"
    )

    lags = np.arange(1, k + 1)
    long_run = n + 2.0 * np.sum(phi**lags * (n - lags))
    expected = np.sqrt(n * (model_unc**2 + observed_unc**2) + cg_unc**2 * long_run)

    assert result.savings["savings_unc"].iloc[0] == pytest.approx(expected, rel=1e-9)


def test_ndid_negative_lag_sum_falls_back_to_quadrature():
    frame = _ndid_frame(_index(10), corrected_unc=2.0, cg_var=3.0)

    ndid = _savings_unc(frame, cg_acf=np.array([-0.9, 0.1]))

    assert ndid[0] == pytest.approx(np.sqrt(10 * 4.0), rel=1e-12)


def test_ndid_lag_pairs_count_only_within_a_period_in_time_order():
    """Jan 30, Jan 31, Feb 1, Feb 2 at sigma 1 and rho_1 = 0.5: each month holds
    one adjacent pair (adding 2 * 0.5 * 1 * 1), the Jan 31 / Feb 1 pair spans
    the boundary and enters neither, and row order in the frame is irrelevant."""
    index = _index(4, start="2019-01-30", freq="D")
    frame = _ndid_frame(index, corrected_unc=2.0, cg_var=1.0)
    shuffled = frame.iloc[[2, 0, 3, 1]].reset_index(drop=True)

    ndid = _savings_unc(shuffled, "monthly", cg_acf=np.array([0.5]))

    np.testing.assert_allclose(ndid, np.sqrt([2 * 4.0 + 1.0, 2 * 4.0 + 1.0]), rtol=1e-12)


def test_ndid_native_aggregation_equals_per_row_quadrature():
    frame = _ndid_frame(_index(6), corrected_unc=np.arange(1.0, 7.0), cg_var=0.5, observed_unc=0.2)

    ndid = _savings_unc(frame, "native", cg_acf=np.array([0.8, 0.4]))

    np.testing.assert_allclose(ndid, np.sqrt(np.arange(1.0, 7.0) ** 2 + 0.04), rtol=1e-12)


def test_ndid_nan_cg_var_row_contributes_zero_sigma():
    """sigma = [1, 0, 1, 1] at rho_1 = 0.5: only the (2, 3) pair survives."""
    frame = _ndid_frame(_index(4), corrected_unc=2.0, cg_var=np.array([1.0, np.nan, 1.0, 1.0]))

    ndid = _savings_unc(frame, cg_acf=np.array([0.5]))

    assert ndid[0] == pytest.approx(np.sqrt(4 * 4.0 + 1.0), rel=1e-12)


def test_ndid_uncovered_row_contributes_zero_sigma():
    """An uncovered row leaves the quadrature sum and breaks its lag pairs."""
    frame = _ndid_frame(_index(4), corrected_unc=2.0, cg_var=1.0)
    frame.loc[1, "observed"] = np.nan

    result = compute_savings(_correction(frame, cg_acf=np.array([0.5])), aggregation="total")
    row = result.savings.iloc[0]

    assert row["coverage"] == pytest.approx(0.75)
    assert row["savings_unc"] == pytest.approx(np.sqrt(3 * 4.0 + 1.0), rel=1e-12)


@pytest.mark.parametrize("column", ["corrected_unc", "observed_unc"])
def test_ndid_nan_unc_on_covered_row_makes_period_non_finite(column):
    frame = _ndid_frame(_index(4), corrected_unc=2.0, cg_var=1.0)
    frame.loc[2, column] = np.nan

    result = compute_savings(_correction(frame, cg_acf=np.array([0.5])), aggregation="total")
    row = result.savings.iloc[0]

    assert row["coverage"] == pytest.approx(1.0)
    assert np.isnan(row["savings_unc"])


def test_result_without_cg_var_takes_quadrature_path_even_with_acf():
    frame = _meter_frame(
        "m1", _index(48), observed=1.0, corrected=np.linspace(2.0, 5.0, 48), corrected_unc=1.5
    )

    with_acf = compute_savings(_correction(frame, cg_acf=np.array([0.9])), aggregation="total")
    legacy = compute_savings(_correction(frame), aggregation="total")

    pd.testing.assert_frame_equal(with_acf.savings, legacy.savings)
    assert with_acf.savings["savings_unc"].iloc[0] == pytest.approx(np.sqrt(48 * 1.5**2))


# ── serialization ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("aggregation", ["native", "monthly", "seasonal", "annual", "total"])
def test_json_roundtrip_preserves_tables(aggregation):
    index = pd.DatetimeIndex(
        [pd.Timestamp("2019-01-15", tz=_TZ), pd.Timestamp("2019-06-15", tz=_TZ)]
    )
    frame = _meter_frame(
        "m1", index, observed=np.array([1.0, 2.0]), corrected=np.array([3.0, 5.0]), corrected_unc=np.array([0.5, np.nan])
    )
    result = compute_savings(_correction(frame), aggregation=aggregation)

    restored = SavingsResult.from_json(result.to_json())

    pd.testing.assert_frame_equal(
        restored.savings.reset_index(drop=True), result.savings.reset_index(drop=True)
    )
    assert restored.tz == result.tz
    assert restored.aggregation == result.aggregation
