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

"""Variant environments behind the correction/savings equivalence snapshot.

The suite's real-data populations are built from a bank of pinned fitted models
(``fixtures/equivalence_models.json.gz``, written by ``write_models_fixture``)
with the ComStock data re-attached; the suite never fits those models. Model
fitting converges to different optima on different machines, so the correction
and savings values pinned in ``fixtures/equivalence_snapshot.json.gz`` are only
reproducible against pinned models, while prediction from a deserialized model
is deterministic. ``ComStockData`` caches the data objects so fixtures share
them, and ``build_variant`` rebuilds each equivalence variant from the bank and
its pinned selection.
"""

from __future__ import annotations

import gzip
import json
import pathlib
import subprocess
from io import StringIO

import pandas as pd

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.population import ComparisonPool, TreatmentGroup
from opendsm.comparison_groups.selection import (
    ComparisonGroupSelection,
    SelectionMethod,
    _finalize_clusters,
    _normalize_treatment_weights,
    select_comparison_group,
)
from opendsm.eemeter.models import (
    BillingBaselineData,
    BillingModel,
    BillingReportingData,
    DailyBaselineData,
    DailyModel,
    DailyReportingData,
    HourlyBaselineData,
    HourlyModel,
    HourlyReportingData,
)



SNAPSHOT_PATH = pathlib.Path(__file__).parent / "fixtures" / "equivalence_snapshot.json.gz"

SNAPSHOT_SCHEMA_VERSION = 1

MODELS_PATH = pathlib.Path(__file__).parent / "fixtures" / "equivalence_models.json.gz"

MODELS_SCHEMA_VERSION = 2


VARIANT_NAMES = (
    "daily_cg_clustering",
    "billing_cg_clustering",
    "hourly_pool_daily_treatment",
    "daily_pool_observed_unc",
    "billing_pool_observed_unc",
    "billing_imm",
)

# A deterministic, non-zero scalar observed uncertainty for every pool meter of
# the ``*_observed_unc`` variants: the correction's comparison-group correlation
# path only activates when a pool meter carries observed uncertainty.
POOL_OBSERVED_UNC = 1.0


# ── ComStock data and the pinned model bank ──────────────────────────────────

# The first meters of each granularity, covering every real-data population the
# suite builds. Daily and hourly frames carry the same meter ids; billing is the
# monthly aggregate of those meters.
BANK_SIZES = {"daily": 68, "billing": 46, "hourly": 8}

_MODEL_CLASSES = {"daily": DailyModel, "billing": BillingModel, "hourly": HourlyModel}

_DATA_CLASSES = {
    "daily": (DailyBaselineData, DailyReportingData),
    "billing": (BillingBaselineData, BillingReportingData),
    "hourly": (HourlyBaselineData, HourlyReportingData),
}


class ComStockData:
    """The ComStock ``(baseline, reporting)`` frame pairs per granularity, with a
    cache of the data objects built from them, so every population shares one
    baseline and one reporting object per meter instead of rebuilding them."""

    def __init__(self, daily, monthly, hourly):
        self._frames = {"daily": daily, "billing": monthly, "hourly": hourly}
        self._cache = {}

    def frames(self, granularity):
        return self._frames[granularity]

    def ids(self, granularity):
        df_b, _ = self._frames[granularity]
        ids = sorted(df_b.index.get_level_values("id").unique())

        return ids

    def _data(self, role, granularity, mid):
        key = (role, granularity, str(mid))

        if key not in self._cache:
            df_b, df_r = self._frames[granularity]
            baseline_cls, reporting_cls = _DATA_CLASSES[granularity]

            if role == "baseline":
                raw = df_b.xs(int(mid), level="id").reset_index()
                self._cache[key] = baseline_cls(raw, is_electricity_data=True)
            else:
                raw = df_r.xs(int(mid), level="id").reset_index()
                self._cache[key] = reporting_cls(raw, is_electricity_data=True)

        return self._cache[key]

    def baseline(self, granularity, mid):
        return self._data("baseline", granularity, mid)

    def reporting(self, granularity, mid):
        return self._data("reporting", granularity, mid)

    def meters(self, bank, granularity, ids, observed_unc=None, reporting=True):
        """A ``from_fit_models`` meters mapping over ``ids``: the pinned models
        from ``bank`` with the cached data objects attached."""
        model_cls = _MODEL_CLASSES[granularity]
        meters = {}

        for mid in ids:
            key = str(mid)
            entry = {
                "model": model_cls.from_json(bank[granularity][key]),
                "baseline_data": self.baseline(granularity, key),
                "observed_unc": observed_unc,
            }
            if reporting:
                entry["reporting_data"] = self.reporting(granularity, key)
            meters[key] = entry

        return meters


def fit_bank(comstock):
    """Fit the first ``BANK_SIZES`` meters of each granularity and return their
    serialized models. Generator only: fits converge to different optima on
    different machines, so the suite never fits these."""
    bank = {}

    for granularity, size in BANK_SIZES.items():
        model_cls = _MODEL_CLASSES[granularity]
        models = {}

        for mid in comstock.ids(granularity)[:size]:
            models[str(mid)] = model_cls().fit(comstock.baseline(granularity, mid)).to_json()

        bank[granularity] = models

    return bank


def load_models_fixture():
    """The pinned model bank (``bank[granularity][id]`` model payloads) and the
    per-variant selection payloads."""
    payload = json.loads(gzip.decompress(MODELS_PATH.read_bytes()))

    if payload["schema_version"] != MODELS_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported equivalence models schema_version {payload['schema_version']}; "
            f"expected {MODELS_SCHEMA_VERSION}."
        )

    return payload


def _manual_clustering_selection(treatment_ids, pool_ids, cluster_of, weights, tz):
    """A hand-built clustering selection so guard behavior can be exercised
    without depending on the clustering algorithm's data-driven output."""
    clusters_flat = pd.DataFrame(
        {
            "id": [str(p) for p in pool_ids],
            "cluster": [cluster_of[str(p)] for p in pool_ids],
            "weight": 1.0,
            "treatment": pd.NA,
        }
    )
    clusters = _finalize_clusters(clusters_flat, SelectionMethod.CG_CLUSTERING)

    weights_frame = pd.DataFrame(weights).T
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
        treatment_ids=[str(t) for t in treatment_ids],
        pool_ids=[str(p) for p in pool_ids],
        tz=tz,
    )

    return selection


def _mixed_selection(treatment, pool):
    """A hand-built one-cluster selection (all pool meters, equal weight) so a
    mixed-granularity pair doesn't depend on the clustering algorithm."""
    cluster_of = {p: 0 for p in pool.ids}
    weights = {t: {"pct_cluster_0": 1.0} for t in treatment.ids}

    return _manual_clustering_selection(treatment.ids, pool.ids, cluster_of, weights, treatment.tz)


# ── variant environments ─────────────────────────────────────────────────────

# Per variant: the treatment (granularity, start, stop), the pool (granularity,
# start, stop, observed_unc) as slices of the sorted meter ids, and the selection
# method; "mixed" is the hand-built one-cluster selection for a pool at a finer
# granularity than its treatment.
_VARIANT_SPECS = {
    "daily_cg_clustering": (("daily", 0, 8), ("daily", 8, 68, None), "cg_clustering"),
    "billing_cg_clustering": (("billing", 0, 6), ("billing", 6, 46, None), "cg_clustering"),
    "hourly_pool_daily_treatment": (("daily", 0, 2), ("hourly", 2, 8, None), "mixed"),
    "daily_pool_observed_unc": (
        ("daily", 0, 8),
        ("daily", 8, 68, POOL_OBSERVED_UNC),
        "cg_clustering",
    ),
    "billing_pool_observed_unc": (
        ("billing", 0, 6),
        ("billing", 6, 46, POOL_OBSERVED_UNC),
        "cg_clustering",
    ),
    "billing_imm": (("billing", 0, 6), ("billing", 6, 46, None), "individual_meter_matching"),
}


def variant_populations(name, comstock, bank, cache=None):
    """The treatment and pool populations of a variant from the pinned bank.
    Populations with the same specification are shared through ``cache`` across
    variants, so each is predicted once."""
    if cache is None:
        cache = {}

    treatment_spec, pool_spec, _ = _VARIANT_SPECS[name]
    treatment_key = ("treatment",) + treatment_spec
    pool_key = ("pool",) + pool_spec

    if treatment_key not in cache:
        granularity, start, stop = treatment_spec
        ids = comstock.ids(granularity)[start:stop]
        cache[treatment_key] = TreatmentGroup.from_fit_models(
            comstock.meters(bank, granularity, ids)
        )

    if pool_key not in cache:
        granularity, start, stop, observed_unc = pool_spec
        ids = comstock.ids(granularity)[start:stop]
        cache[pool_key] = ComparisonPool.from_fit_models(
            comstock.meters(bank, granularity, ids, observed_unc=observed_unc)
        )

    return cache[treatment_key], cache[pool_key]


def fit_variant(name, comstock, bank, cache=None):
    """A variant with its selection computed live. Generator only."""
    treatment, pool = variant_populations(name, comstock, bank, cache)
    method = _VARIANT_SPECS[name][2]

    if method == "mixed":
        selection = _mixed_selection(treatment, pool)
    else:
        selection = select_comparison_group(treatment, pool, method=method)

    env = {"treatment": treatment, "pool": pool, "selection": selection}

    return env


def build_variant(name, comstock, models, cache=None):
    """A variant from the pinned models fixture: populations from the bank with
    the real ComStock data attached, and the pinned selection."""
    treatment, pool = variant_populations(name, comstock, models["bank"], cache)
    selection = ComparisonGroupSelection.from_json(models["selections"][name])
    env = {"treatment": treatment, "pool": pool, "selection": selection}

    return env


def write_models_fixture(daily, monthly, hourly):
    """Fit the model bank and the six selections and write the pinned models
    fixture. Run manually through the guarded ``test_generate_equivalence_models``;
    regenerate the equivalence snapshot alongside whenever the fits change."""
    comstock = ComStockData(daily, monthly, hourly)
    bank = fit_bank(comstock)
    selections = {}
    cache = {}

    for name in VARIANT_NAMES:
        selections[name] = fit_variant(name, comstock, bank, cache)["selection"].to_json()

    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    payload = {
        "schema_version": MODELS_SCHEMA_VERSION,
        "git_sha": sha,
        "bank": bank,
        "selections": selections,
    }
    blob = json.dumps(payload, allow_nan=False).encode("utf-8")
    MODELS_PATH.write_bytes(gzip.compress(blob, compresslevel=9, mtime=0))


# ── snapshot artifact ────────────────────────────────────────────────────────


def _read_frame(payload, tz, datetime_columns):
    frame = pd.read_json(StringIO(payload), orient="table")
    frame["id"] = frame["id"].astype(str)

    for column in datetime_columns:
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert(tz)

    return frame


def _read_periods(payload, tz):
    periods = [
        (pd.Timestamp(start).tz_convert(tz), pd.Timestamp(end).tz_convert(tz))
        for start, end in payload
    ]

    return periods


def load_equivalence_snapshot():
    """The pinned correction/savings output, per variant and treatment meter,
    with every table restored as a DataFrame (NaN preserved) and every read
    period as a ``(start, end)`` Timestamp pair. Five variants carry the
    batch-path pin; the hourly-pool variant is pinned from the per-meter path
    (its ``pinned_from`` entry names the commit), because its hourly pool
    models carry the deterministic curvature floor, for which no batch-path
    values exist."""
    payload = json.loads(gzip.decompress(SNAPSHOT_PATH.read_bytes()))

    if payload["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported equivalence snapshot schema_version {payload['schema_version']}; "
            f"expected {SNAPSHOT_SCHEMA_VERSION}."
        )

    variants = {}

    for name, variant in payload["variants"].items():
        tz = variant["tz"]
        meters = {}

        for mid, meter in variant["meters"].items():
            savings = {}

            for aggregation, table in meter["savings"].items():
                if aggregation == "native":
                    savings[aggregation] = _read_frame(table, tz, ["period"])
                else:
                    savings[aggregation] = _read_frame(table, tz, [])

            meters[mid] = {
                "corrected": _read_frame(meter["corrected"], tz, ["datetime"]),
                "cg_usage": _read_frame(meter["cg_usage"], tz, ["datetime"]),
                "exclusions": _read_frame(meter["exclusions"], tz, []),
                "correction_periods": _read_periods(meter["correction_periods"], tz),
                "savings": savings,
            }

        variants[name] = {
            "method": variant["method"],
            "granularity": variant["granularity"],
            "tz": tz,
            "treatment_ids": variant["treatment_ids"],
            "pool_ids": variant["pool_ids"],
            "meters": meters,
        }

    snapshot = {"git_sha": payload["git_sha"], "variants": variants}

    return snapshot
