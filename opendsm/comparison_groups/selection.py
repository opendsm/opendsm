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

from __future__ import annotations

import hashlib
import json
from enum import Enum
from io import StringIO

import numpy as np
import pandas as pd

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const
from opendsm.comparison_groups.population import _DEFAULT_TIME_PERIOD
from opendsm.comparison_groups.cg_clustering.create_comparison_groups import CG_Clustering
from opendsm.comparison_groups.cg_clustering.settings import CG_Clustering_Settings
from opendsm.comparison_groups.individual_meter_matching.create_comparison_groups import (
    Individual_Meter_Matching,
)
from opendsm.comparison_groups.random_sampling.create_comparison_groups import Random_Sampling
from opendsm.comparison_groups.stratified_sampling.create_comparison_groups import Stratified_Sampling
from opendsm.comparison_groups.stratified_sampling.settings import (
    DistanceStratifiedSamplingSettings,
)



SCHEMA_VERSION = 1


class SelectionMethod(str, Enum):
    CG_CLUSTERING = "cg_clustering"
    INDIVIDUAL_METER_MATCHING = "individual_meter_matching"
    STRATIFIED_SAMPLING = "stratified_sampling"
    RANDOM_SAMPLING = "random_sampling"


_ALGORITHM = {
    SelectionMethod.CG_CLUSTERING: CG_Clustering,
    SelectionMethod.INDIVIDUAL_METER_MATCHING: Individual_Meter_Matching,
    SelectionMethod.STRATIFIED_SAMPLING: Stratified_Sampling,
    SelectionMethod.RANDOM_SAMPLING: Random_Sampling,
}

# Default loadshape basis per method: CG clustering matches on modeling error,
# every other method on the modeled load.
_DEFAULT_BASIS = {
    SelectionMethod.CG_CLUSTERING: _const.LoadshapeType.ERROR,
    SelectionMethod.INDIVIDUAL_METER_MATCHING: _const.LoadshapeType.MODELED,
    SelectionMethod.STRATIFIED_SAMPLING: _const.LoadshapeType.MODELED,
    SelectionMethod.RANDOM_SAMPLING: _const.LoadshapeType.MODELED,
}

# Canonical dtypes for the normalized clusters table (id is the index, str).
_CLUSTER_DTYPES = {
    "cluster": "int64",
    "weight": "float64",
    "treatment": "string",
    "distance": "float64",
    "duplicated": "bool",
}

_BASE_CLUSTER_COLUMNS = ["cluster", "weight", "treatment"]
_IMM_CLUSTER_COLUMNS = ["cluster", "weight", "treatment", "distance", "duplicated"]

_FINGERPRINT_SIG_DIGITS = 9


def _default_settings(method):
    if method == SelectionMethod.CG_CLUSTERING:
        return CG_Clustering_Settings()

    if method == SelectionMethod.INDIVIDUAL_METER_MATCHING:
        return Individual_Meter_Matching().settings

    if method == SelectionMethod.STRATIFIED_SAMPLING:
        return DistanceStratifiedSamplingSettings()

    return Random_Sampling().settings


def _build_data(population, method, basis, data_settings):
    if method != SelectionMethod.STRATIFIED_SAMPLING:
        return population.loadshape_data(basis, data_settings)

    if population.features is None:
        raise ValueError(
            f"stratified_sampling requires stratification features on the {population.role} "
            "population, but none were provided."
        )

    loadshape = population.loadshape_data(basis, data_settings).loadshape
    features = population.features.rename_axis("id").reset_index()
    settings = Data_Settings(agg_type=None, loadshape_type=None, time_period=None)
    data = Data(
        loadshape_df=loadshape.reset_index(),
        features_df=features,
        settings=settings,
    )

    return data


def _finalize_clusters(flat, method):
    """Coerce a flat clusters frame (id column present) to the canonical
    normalized schema: id-indexed, method-specific columns, pinned dtypes."""
    if method == SelectionMethod.INDIVIDUAL_METER_MATCHING:
        columns = _IMM_CLUSTER_COLUMNS
    else:
        columns = _BASE_CLUSTER_COLUMNS

    frame = flat.copy()
    frame["id"] = frame["id"].astype(str)

    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA

        frame[column] = frame[column].astype(_CLUSTER_DTYPES[column])

    ordered = frame[["id"] + columns]
    ordered.columns = pd.Index(["id"] + columns)
    clusters = ordered.set_index("id")

    return clusters


def _normalize_clusters(method, clusters_raw):
    """Map each algorithm's clusters output onto the normalized schema the
    correction kernel consumes. The IMM duplicate id index is preserved."""
    if method == SelectionMethod.CG_CLUSTERING:
        flat = clusters_raw.reset_index()
        flat["weight"] = 1.0
        flat["treatment"] = pd.NA

    elif method == SelectionMethod.INDIVIDUAL_METER_MATCHING:
        flat = clusters_raw.reset_index()

    elif method in (SelectionMethod.RANDOM_SAMPLING, SelectionMethod.STRATIFIED_SAMPLING):
        flat = clusters_raw.reset_index()
        flat["treatment"] = pd.NA

    else:
        raise ValueError(f"Unknown selection method {method!r}.")

    clusters = _finalize_clusters(flat, method)

    return clusters


def _normalize_treatment_weights(treatment_weights_raw):
    weights = treatment_weights_raw.copy()
    weights.index = weights.index.astype(str)
    weights.index.name = "id"
    weights.columns = pd.Index([str(column) for column in weights.columns])

    for column in weights.columns:
        weights[column] = weights[column].astype("float64")

    return weights


def _stage_exclusions(treatment_data, pool_data, treatment_weights):
    """Selection-stage ledger: ``Data`` validation exclusions (verbatim reason
    strings) plus treatment meters whose loadshape produced no usable cluster
    weights. Pool-trim rows live on the pool population's own ledger."""
    ledger = exclusions.empty_ledger()

    for data in (treatment_data, pool_data):
        for reason, group in data.excluded_ids.groupby("reason"):
            ledger = exclusions.append(
                ledger, group["id"], "selection", "data_validation", str(reason)
            )

    nan_rows = treatment_weights.index[treatment_weights.isna().all(axis=1)]
    ledger = exclusions.append(
        ledger,
        nan_rows,
        "selection",
        "treatment_fit",
        "treatment loadshape invalid (all-NaN cluster weights)",
    )

    ledger = ledger.drop_duplicates(ignore_index=True)

    return ledger


# Native period cadence per granularity for the coverage denominator; billing
# rides a daily substrate, so its window length is counted in days.
COVERAGE_FREQ = {"hourly": "h", "daily": "D", "billing": "D"}

# Higher fineness = finer native cadence. A pool must be at least as fine as
# the treatment: fineness(pool) >= fineness(treatment).
FINENESS = {"billing": 0, "daily": 1, "hourly": 2}


def check_granularity_fineness(treatment, pool):
    """Raise ``ValueError`` when ``pool`` is coarser than ``treatment``: a
    comparison pool must be at the treatment's cadence or finer so it can be
    aggregated up to it."""
    if FINENESS[pool.granularity] < FINENESS[treatment.granularity]:
        raise ValueError(
            f"pool granularity {pool.granularity!r} is coarser than treatment granularity "
            f"{treatment.granularity!r}; the comparison pool must be same-or-finer."
        )


def window_coverage(baseline_data, window, freq):
    """Fraction of the baseline group ``window`` a meter covers with finite
    observed AND temperature: its finite in-window native rows over the window's
    length in native periods (so both interior holes and a short span count
    against it)."""
    start, end = window
    n_periods = len(pd.date_range(start, end, freq=freq))
    df = baseline_data.df
    in_window = (df.index >= start) & (df.index <= end)
    observed = df.loc[in_window, "observed"].to_numpy(dtype=float)
    temperature = df.loc[in_window, "temperature"].to_numpy(dtype=float)
    finite = np.isfinite(observed) & np.isfinite(temperature)
    coverage = float(finite.sum()) / n_periods

    return coverage


def _coverage_exclusions(population, ids, window, min_window_coverage):
    """Selection-stage baseline-coverage ledger: the meters among ``ids`` whose
    finite observed-and-temperature coverage over the group ``window`` falls
    below ``min_window_coverage``. The computed coverage goes in ``detail``, with
    any construction-time eemeter disqualification warnings appended verbatim.
    Returns the failing ids (for pruning) and their ledger rows."""
    freq = COVERAGE_FREQ[population.granularity]
    ledger = exclusions.empty_ledger()
    failed = []

    for mid in ids:
        rec = population._meters.get(mid)
        if rec is None:
            continue

        coverage = window_coverage(rec.baseline_data, window, freq)
        if coverage >= min_window_coverage:
            continue

        detail = f"baseline window coverage {coverage:.4f} < {min_window_coverage}"
        warning_text = exclusions.format_warnings(rec.baseline_data.disqualification)
        if warning_text:
            detail = f"{detail}; {warning_text}"

        ledger = exclusions.append(
            ledger,
            [mid],
            "selection",
            "baseline_coverage",
            "baseline coverage below minimum over the group window",
            detail=detail,
        )
        failed.append(mid)

    return failed, ledger


def _round_sig(value):
    return float(f"{float(value):.{_FINGERPRINT_SIG_DIGITS}g}")


def _canonical_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return _round_sig(value)

    return str(value)


def _canonical_rows(flat):
    records = []

    for row in flat.to_dict(orient="records"):
        record = {str(key): _canonical_value(val) for key, val in row.items()}
        records.append(record)

    records.sort(key=lambda item: json.dumps(item, sort_keys=True))

    return records


def table_fingerprint(tables):
    """SHA-256 digest of the canonical form of ``tables`` (a name-to-frame
    mapping), with floats rounded to nine significant digits so a JSON round
    trip reproduces it."""
    canonical = {name: _canonical_rows(flat) for name, flat in tables.items()}
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()

    return digest


class ComparisonGroupSelection:
    """The comparison group chosen for a treatment population, in the normalized
    form the correction kernel consumes.

    ``clusters`` is id-indexed (duplicate ids allowed for individual meter
    matching) with columns ``cluster``, ``weight``, ``treatment`` and, for IMM,
    ``distance`` and ``duplicated``. ``treatment_weights`` is id-indexed with
    ``pct_cluster_*`` columns and may contain all-NaN rows for treatment meters
    whose loadshape could not be matched (these ids are also recorded in
    ``exclusions``, the selection-stage disqualification ledger).
    """

    def __init__(
        self,
        clusters,
        treatment_weights,
        method,
        basis,
        method_settings,
        data_settings,
        exclusions,
        treatment_ids,
        pool_ids,
        tz,
    ):
        self.clusters = clusters
        self.treatment_weights = treatment_weights
        self.method = SelectionMethod(method)
        self.basis = _const.LoadshapeType(basis).value
        self.method_settings = method_settings
        self.data_settings = data_settings
        self.exclusions = exclusions
        self.treatment_ids = [str(x) for x in treatment_ids]
        self.pool_ids = [str(x) for x in pool_ids]
        self.tz = tz
        self.fingerprint = table_fingerprint(self._fingerprint_tables())

    # -- accessors -----------------------------------------------------------

    def _fingerprint_tables(self):
        """The tables the fingerprint covers: the selection's substance
        (clusters and treatment weights). The exclusions ledger is provenance
        and deliberately not fingerprinted."""
        frames = {
            "clusters": self.clusters.reset_index(),
            "treatment_weights": self.treatment_weights.reset_index(),
        }

        return frames

    @property
    def tables(self):
        """Flat frames (id as a column) for tabular storage."""
        frames = {**self._fingerprint_tables(), "exclusions": self.exclusions.copy()}

        return frames

    # -- serialization -------------------------------------------------------

    def _table_payload(self):
        # double_precision at its 15-place maximum so serialized floats round-trip
        # to the same 9-significant-digit fingerprint (the default 10 places
        # truncates small cluster weights below that resolution).
        payload = {
            "clusters": self.clusters.reset_index().to_json(orient="table", double_precision=15),
            "treatment_weights": self.treatment_weights.reset_index().to_json(
                orient="table", double_precision=15
            ),
            "exclusions": self.exclusions.to_json(orient="table"),
        }

        return payload

    def to_json(self):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "method": self.method.value,
            "basis": self.basis,
            "tz": self.tz,
            "fingerprint": self.fingerprint,
            "treatment_ids": self.treatment_ids,
            "pool_ids": self.pool_ids,
            "method_settings": self.method_settings,
            "data_settings": self.data_settings,
            "tables": self._table_payload(),
        }

        return json.dumps(payload, allow_nan=False)

    @classmethod
    def from_json(cls, s):
        payload = json.loads(s)

        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {payload['schema_version']}; "
                f"expected {SCHEMA_VERSION}."
            )

        method = SelectionMethod(payload["method"])
        tables = payload["tables"]

        clusters_flat = pd.read_json(StringIO(tables["clusters"]), orient="table")
        clusters = _finalize_clusters(clusters_flat, method)

        weights_flat = pd.read_json(StringIO(tables["treatment_weights"]), orient="table")
        treatment_weights = _normalize_treatment_weights(weights_flat.set_index("id"))

        ledger = exclusions.read_table_json(tables["exclusions"])

        selection = cls(
            clusters=clusters,
            treatment_weights=treatment_weights,
            method=method,
            basis=payload["basis"],
            method_settings=payload["method_settings"],
            data_settings=payload["data_settings"],
            exclusions=ledger,
            treatment_ids=payload["treatment_ids"],
            pool_ids=payload["pool_ids"],
            tz=payload["tz"],
        )

        if selection.fingerprint != payload["fingerprint"]:
            raise ValueError(
                "Deserialized selection fingerprint does not match the stored value; "
                "the payload tables are inconsistent."
            )

        return selection


def select_comparison_group(
    treatment,
    pool,
    method="cg_clustering",
    method_settings=None,
    basis=None,
    data_settings=None,
    min_window_coverage=0.9,
):
    """Select a comparison group for ``treatment`` from ``pool``.

    A fresh algorithm instance is built per call because CG clustering mutates its
    own settings during ``get_labels``. ``method_settings`` is captured (dumped)
    before that mutation so the selection can be re-run deterministically.

    ``min_window_coverage`` is the baseline-coverage floor: each treatment meter
    and each selected pool meter must carry finite observed and temperature over
    at least this fraction of the treatment population's baseline group window.
    Meters below it are pruned (dropped from ``treatment_weights`` or
    ``clusters``) and recorded on the selection ledger with their coverage,
    because the correction loop skips only absent meters — a ledger-only row
    would leave the meter corrected.

    Loadshapes are always built at the TREATMENT population's granularity: when
    ``data_settings`` is not given, its ``time_period`` defaults per treatment
    granularity and that same setting is passed to both populations, so a finer
    pool aggregates to the treatment's cadence rather than defaulting to its own
    (mismatched loadshape lengths otherwise). The data settings actually used
    are stored on the selection. The pool must be same-or-finer
    than the treatment (``fineness(pool) >= fineness(treatment)``); a coarser
    pool raises.

    Args:
        treatment: ``TreatmentGroup`` with baseline data attached.
        pool: ``ComparisonPool`` with baseline data attached, at the treatment's
            cadence or finer.
        method: ``"cg_clustering"``, ``"individual_meter_matching"``,
            ``"stratified_sampling"`` or ``"random_sampling"``.
        method_settings: the chosen method's settings object; the method's
            defaults when omitted.
        basis: loadshape basis; per-method default when omitted (modeling error
            for CG clustering, modeled load otherwise).
        data_settings: ``Data_Settings`` for loadshape construction; defaults to
            the treatment granularity's time period.
        min_window_coverage: baseline-coverage floor over the treatment group
            window, applied to treatment and selected pool meters.

    Returns:
        ``ComparisonGroupSelection``.
    """
    method = SelectionMethod(method)

    if treatment.tz != pool.tz:
        raise ValueError(
            f"treatment timezone {treatment.tz} does not match pool timezone {pool.tz}."
        )

    check_granularity_fineness(treatment, pool)

    if basis is None:
        basis = _DEFAULT_BASIS[method]
    basis = _const.LoadshapeType(basis)

    if method_settings is None:
        method_settings = _default_settings(method)

    settings_dump = method_settings.model_dump(mode="json")

    if data_settings is None:
        data_settings = Data_Settings(time_period=_DEFAULT_TIME_PERIOD[treatment.granularity])

    data_settings_dump = data_settings.model_dump(mode="json")

    treatment_data = _build_data(treatment, method, basis, data_settings)
    pool_data = _build_data(pool, method, basis, data_settings)

    algorithm = _ALGORITHM[method](method_settings)
    clusters_raw, treatment_weights_raw = algorithm.get_comparison_group(treatment_data, pool_data)

    clusters = _normalize_clusters(method, clusters_raw)
    treatment_weights = _normalize_treatment_weights(treatment_weights_raw)
    stage_ledger = _stage_exclusions(treatment_data, pool_data, treatment_weights)

    window = treatment.baseline_window
    pool_ids_selected = list(dict.fromkeys(clusters.index.astype(str)))
    treatment_failed, treatment_cov_ledger = _coverage_exclusions(
        treatment, list(treatment_weights.index), window, min_window_coverage
    )
    pool_failed, pool_cov_ledger = _coverage_exclusions(
        pool, pool_ids_selected, window, min_window_coverage
    )

    treatment_weights = treatment_weights.drop(index=treatment_failed)
    clusters = clusters.drop(index=pool_failed)

    stage_ledger = pd.concat(
        [stage_ledger, treatment_cov_ledger, pool_cov_ledger], ignore_index=True
    )
    stage_ledger = stage_ledger.drop_duplicates(ignore_index=True)

    selection = ComparisonGroupSelection(
        clusters=clusters,
        treatment_weights=treatment_weights,
        method=method,
        basis=basis,
        method_settings=settings_dump,
        data_settings=data_settings_dump,
        exclusions=stage_ledger,
        treatment_ids=treatment.ids,
        pool_ids=pool.ids,
        tz=treatment.tz,
    )

    return selection
