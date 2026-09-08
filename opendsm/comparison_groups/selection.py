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
from opendsm.comparison_groups.common import const as _const
from opendsm.comparison_groups.population import default_data_settings
from opendsm.comparison_groups.cg_clustering.create_comparison_groups import CG_Clustering
from opendsm.comparison_groups.cg_clustering.settings import CG_Clustering_Settings
from opendsm.comparison_groups.individual_meter_matching.create_comparison_groups import (
    Individual_Meter_Matching,
)
from opendsm.comparison_groups.individual_meter_matching.settings import (
    Settings as Individual_Meter_Matching_Settings,
)
from opendsm.comparison_groups.random_sampling.create_comparison_groups import Random_Sampling
from opendsm.comparison_groups.random_sampling.settings import Settings as Random_Sampling_Settings
from opendsm.comparison_groups.stratified_sampling.create_comparison_groups import Stratified_Sampling
from opendsm.comparison_groups.stratified_sampling.settings import (
    DistanceStratifiedSamplingSettings,
)



SCHEMA_VERSION = 1


class SelectionMethod(str, Enum):
    """Comparison-group selection methods that ``select_comparison_group`` runs."""

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

_DEFAULT_SETTINGS = {
    SelectionMethod.CG_CLUSTERING: CG_Clustering_Settings,
    SelectionMethod.INDIVIDUAL_METER_MATCHING: Individual_Meter_Matching_Settings,
    SelectionMethod.STRATIFIED_SAMPLING: DistanceStratifiedSamplingSettings,
    SelectionMethod.RANDOM_SAMPLING: Random_Sampling_Settings,
}

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


def _default_method_settings(method):
    """A new default settings object for ``method``.

    Raises:
        ValueError: ``method`` has no default settings.
    """
    if method not in _DEFAULT_SETTINGS:
        raise ValueError(f"Unknown selection method {method!r}.")

    settings = _DEFAULT_SETTINGS[method]()

    return settings


def window_to_json(window):
    """``window`` as a ``[start, end]`` pair of ISO-8601 strings, or ``None``
    when ``window`` is ``None``."""
    if window is None:
        return None

    start, end = window
    bounds = [start.isoformat(), end.isoformat()]

    return bounds


def coerce_window(window, tz):
    """Normalize a ``(start, end)`` window of datetime-likes or ISO-8601 strings
    to a tuple of tz-aware ``Timestamp`` in ``tz``: naive bounds are localized
    to ``tz``, aware ones converted. ``None`` passes through."""
    if window is None:
        return None

    start, end = window
    bounds = []

    for value in (start, end):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
        bounds.append(ts)

    window = (bounds[0], bounds[1])

    return window


def _build_data(population, method, basis, data_settings, exclude):
    """The selection input for ``population`` with the ``exclude`` ids left out
    of its loadshapes; for stratified sampling it also carries the population's
    stratification features."""
    if method != SelectionMethod.STRATIFIED_SAMPLING:
        return population.loadshape_data(basis, data_settings, exclude=exclude)

    if population.features is None:
        raise ValueError(
            f"stratified_sampling requires stratification features on the {population.role} "
            "population, but none were provided."
        )

    data = population.loadshape_data(
        basis, data_settings, exclude=exclude, features=population.features
    )

    return data


def _finalize_clusters(flat, method):
    """Coerce a flat clusters frame (id column present) to the canonical
    normalized schema: str id index, the method's columns in order, pinned
    dtypes. A missing ``treatment`` column is filled with NA.

    Raises:
        ValueError: ``flat`` lacks any other column the method's schema holds.
    """
    if method == SelectionMethod.INDIVIDUAL_METER_MATCHING:
        columns = _IMM_CLUSTER_COLUMNS
    else:
        columns = _BASE_CLUSTER_COLUMNS

    frame = flat.copy()
    if "treatment" not in frame.columns:
        frame["treatment"] = pd.NA

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{method.value} clusters require columns {missing}.")

    frame["id"] = frame["id"].astype(str)
    dtypes = {column: _CLUSTER_DTYPES[column] for column in columns}
    clusters = frame[["id"] + columns].astype(dtypes).set_index("id")
    # Fresh labels, so the column Index dtype does not depend on the input frame's.
    clusters.columns = pd.Index(columns)

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


def _short_baseline_exclusions(population, time_period):
    """Selection-stage ledger rows for the meters of ``population`` whose
    baseline spans fewer than 12 months, which a ``MONTH`` loadshape cannot
    use; empty for any other ``time_period``. The month count goes in
    ``detail``."""
    ledger = exclusions.empty_ledger()

    if time_period != _const.TimePeriod.MONTH:
        return ledger

    for mid, months in population.short_baseline_months().items():
        ledger = exclusions.append(
            ledger,
            [mid],
            "selection",
            "baseline_data",
            "fewer than 12 baseline months; cannot support a MONTH loadshape",
            detail=f"baseline months: {months}",
        )

    return ledger


def _pool_overlap_exclusions(treatment, pool):
    """Selection-stage ledger rows for the pool meters that are also in
    ``treatment``, in pool order. Each row concerns the pool copy; the
    treatment copy stays in selection."""
    treatment_ids = {str(mid) for mid in treatment.ids}
    overlap = [str(mid) for mid in pool.ids if str(mid) in treatment_ids]
    ledger = exclusions.append(
        exclusions.empty_ledger(),
        overlap,
        "selection",
        "pool_overlap",
        "pool copy excluded: meter is in the treatment group",
    )

    return ledger


# Native period cadence per model frequency for the coverage denominator; billing
# rides a daily substrate, so its window length is counted in days.
COVERAGE_FREQ = {"hourly": "h", "daily": "D", "billing": "D"}

FINENESS = {"billing": 0, "daily": 1, "hourly": 2}


def check_model_frequency_fineness(treatment, pool):
    """Raise ``ValueError`` when ``pool`` is coarser than ``treatment``: a
    comparison pool must be at the treatment's cadence or finer so it can be
    aggregated up to it.

    Args:
        treatment: The treatment population.
        pool: The comparison pool.

    Raises:
        ValueError: ``pool`` is coarser than ``treatment``.
    """
    if FINENESS[pool.model_frequency] < FINENESS[treatment.model_frequency]:
        raise ValueError(
            f"pool model frequency {pool.model_frequency!r} is coarser than treatment model frequency "
            f"{treatment.model_frequency!r}; the comparison pool must be same-or-finer."
        )


def window_coverage(data, window, freq):
    """Fraction of ``window`` a meter's data covers with finite observed AND
    temperature: its finite in-window native rows over the window's length in
    native periods (so both interior holes and a short span count against it).

    Args:
        data: The meter's baseline or reporting data object.
        window: ``(start, end)`` timestamps, inclusive.
        freq: Pandas frequency of the meter's native periods.

    Returns:
        The covered fraction.
    """
    start, end = window
    n_periods = len(pd.date_range(start, end, freq=freq))
    df = data.df
    in_window = (df.index >= start) & (df.index <= end)
    observed = df.loc[in_window, "observed"].to_numpy(dtype=float)
    temperature = df.loc[in_window, "temperature"].to_numpy(dtype=float)
    finite = np.isfinite(observed) & np.isfinite(temperature)
    coverage = float(finite.sum()) / n_periods

    return coverage


def _coverage_exclusions(population, ids, window, min_baseline_coverage):
    """Selection-stage baseline-coverage ledger rows for the meters among ``ids``
    whose finite observed-and-temperature coverage over the group ``window``
    falls below ``min_baseline_coverage``. The computed coverage goes in
    ``detail``, with any construction-time disqualification records appended
    verbatim."""
    freq = COVERAGE_FREQ[population.model_frequency]
    ledger = exclusions.empty_ledger()

    for mid in ids:
        baseline_data = population.meter(mid).baseline_data
        coverage = window_coverage(baseline_data, window, freq)
        if coverage >= min_baseline_coverage:
            continue

        detail = f"baseline coverage {coverage:.4f} < min_baseline_coverage {min_baseline_coverage}"
        warning_text = exclusions.format_warnings(baseline_data.compliance.disqualification)
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

    return ledger


def _ids_outside(population, ledger):
    """The ids of ``population`` that ``ledger`` does not name, in population order."""
    excluded = set(ledger["id"])
    ids = [mid for mid in population.ids if mid not in excluded]

    return ids


def _round_sig(value):
    rounded = float(f"{float(value):.{_FINGERPRINT_SIG_DIGITS}g}")

    return rounded


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


def _table_fingerprint(tables):
    """SHA-256 hex digest of the canonical form of ``tables`` (a name-to-frame
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
    ``exclusions``, the selection-stage disqualification ledger). The
    constructor coerces both tables to that form: str ids, the method's columns
    in order, pinned dtypes, and NA for an absent ``treatment`` column; a
    ``clusters`` frame lacking any other of the method's columns raises
    ``ValueError``.

    ``baseline_window`` is the tz-aware ``(start, end)`` the baseline-coverage
    check judged against, or ``None``. It records what selection used and is
    never reapplied; like ``data_settings`` it lies outside the fingerprint.
    """

    def __init__(
        self,
        clusters,
        treatment_weights,
        method,
        basis,
        method_settings,
        data_settings,
        baseline_window,
        exclusions,
        treatment_ids,
        pool_ids,
        tz,
    ):
        self.method = SelectionMethod(method)
        self.basis = _const.LoadshapeType(basis)
        self.clusters = _finalize_clusters(clusters.rename_axis("id").reset_index(), self.method)
        self.treatment_weights = _normalize_treatment_weights(treatment_weights)
        self.method_settings = method_settings
        self.data_settings = data_settings
        self.baseline_window = coerce_window(baseline_window, tz)
        self.exclusions = exclusions
        self.treatment_ids = [str(x) for x in treatment_ids]
        self.pool_ids = [str(x) for x in pool_ids]
        self.tz = tz
        self.fingerprint = _table_fingerprint(self._fingerprint_tables())

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

    def verify_fingerprint(self):
        """Check that ``clusters`` and ``treatment_weights`` still match
        ``fingerprint``.

        Raises:
            ValueError: Either table changed after construction.
        """
        if _table_fingerprint(self._fingerprint_tables()) != self.fingerprint:
            raise ValueError(
                "selection fingerprint does not match its tables; the selection has been mutated."
            )

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
        """Serialize to JSON.

        Returns:
            A JSON string holding the method, basis, timezone, fingerprint, member ids,
            settings, baseline window and tables.
        """
        payload = {
            "schema_version": SCHEMA_VERSION,
            "method": self.method.value,
            "basis": self.basis.value,
            "tz": self.tz,
            "fingerprint": self.fingerprint,
            "treatment_ids": self.treatment_ids,
            "pool_ids": self.pool_ids,
            "method_settings": self.method_settings,
            "data_settings": self.data_settings,
            "baseline_window": window_to_json(self.baseline_window),
            "tables": self._table_payload(),
        }

        return json.dumps(payload, allow_nan=False)

    @classmethod
    def from_json(cls, s):
        """Rebuild from ``to_json()`` output.

        Args:
            s: JSON string written by ``to_json()``.

        Returns:
            The rebuilt ``ComparisonGroupSelection``.

        Raises:
            ValueError: The payload's ``schema_version`` is not supported, or its
                stored fingerprint does not match its tables.
        """
        payload = json.loads(s)

        if payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {payload['schema_version']}; "
                f"expected {SCHEMA_VERSION}."
            )

        tables = payload["tables"]
        clusters = pd.read_json(StringIO(tables["clusters"]), orient="table")
        treatment_weights = pd.read_json(StringIO(tables["treatment_weights"]), orient="table")

        selection = cls(
            clusters=clusters.set_index("id"),
            treatment_weights=treatment_weights.set_index("id"),
            method=payload["method"],
            basis=payload["basis"],
            method_settings=payload["method_settings"],
            data_settings=payload["data_settings"],
            baseline_window=payload["baseline_window"],
            exclusions=exclusions.read_table_json(tables["exclusions"]),
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
    baseline_window=None,
    min_baseline_coverage=0.9,
):
    """Select a comparison group for ``treatment`` from ``pool``.

    ``method_settings`` is stored as dumped before the run, so the selection can
    be re-run deterministically.

    Before the selection algorithm runs, meters that fail any of these checks are
    left out of it and recorded on the selection ledger; neither population is
    changed, and an excluded meter is absent from the selection's tables:

    - Baseline coverage: a treatment or pool meter whose finite observed and
      temperature cover less than ``min_baseline_coverage`` of ``baseline_window``
      gets a ``baseline_coverage`` row carrying its coverage.
    - For a ``MONTH`` loadshape time period, a meter with fewer than 12 baseline
      months gets a ``baseline_data`` row carrying its month count.
    - A meter in both populations stays in the treatment and is left out of the
      pool side, recorded once with origin ``pool_overlap``; that row supersedes
      a short-baseline row for its pool copy. A treatment copy short of 12
      baseline months keeps its own short-baseline row, so such a meter has one
      row per copy.

    Both populations' loadshapes use the same ``data_settings``, so a finer pool
    aggregates to the treatment's cadence (its own default would give mismatched
    loadshape lengths); the settings used are stored on the selection.

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
        data_settings: ``Data_Settings`` for loadshape construction;
            ``default_data_settings`` of the treatment's model frequency when
            omitted.
        baseline_window: ``(start, end)`` the baseline-coverage check judges
            against; naive bounds are localized to the treatment timezone. When
            omitted, the union of the baseline spans of every treatment meter
            given, including meters the checks then exclude.
        min_baseline_coverage: baseline-coverage floor over ``baseline_window``,
            a fraction applied to every treatment and pool meter.

    Returns:
        ``ComparisonGroupSelection``.

    Raises:
        ValueError: A meter in either population has no baseline data attached
            (every such meter is named), the populations' timezones differ, the
            pool is coarser than the treatment, the loadshape time period is
            ``MONTH`` and every treatment meter has fewer than 12 baseline months,
            or the checks above exclude every treatment meter or every pool meter.
    """
    missing = {
        population.role: population.ids_without_baseline() for population in (treatment, pool)
    }
    missing = {role: ids for role, ids in missing.items() if ids}
    if missing:
        raise ValueError(
            f"Selection requires baseline data for every meter; missing for {missing}."
        )

    method = SelectionMethod(method)

    if treatment.tz != pool.tz:
        raise ValueError(
            f"treatment timezone {treatment.tz} does not match pool timezone {pool.tz}."
        )

    check_model_frequency_fineness(treatment, pool)

    if basis is None:
        basis = _DEFAULT_BASIS[method]
    basis = _const.LoadshapeType(basis)

    if method_settings is None:
        method_settings = _default_method_settings(method)

    settings_dump = method_settings.model_dump(mode="json")

    if data_settings is None:
        data_settings = default_data_settings(treatment.model_frequency)

    data_settings_dump = data_settings.model_dump(mode="json")

    if baseline_window is None:
        window = treatment.baseline_span()
    else:
        window = coerce_window(baseline_window, treatment.tz)

    treatment_short = _short_baseline_exclusions(treatment, data_settings.time_period)

    if not treatment_short.empty and len(treatment_short) == len(treatment):
        summary = ", ".join(
            f"{mid}: {detail}"
            for mid, detail in zip(treatment_short["id"], treatment_short["detail"])
        )
        raise ValueError(
            "MONTH loadshapes require at least 12 baseline months per meter; "
            f"all treatment meters excluded ({summary})."
        )

    treatment_coverage = _coverage_exclusions(
        treatment, _ids_outside(treatment, treatment_short), window, min_baseline_coverage
    )
    treatment_excluded = pd.concat([treatment_short, treatment_coverage], ignore_index=True)

    pool_overlap = _pool_overlap_exclusions(treatment, pool)
    pool_short = _short_baseline_exclusions(pool, data_settings.time_period)
    pool_short = pool_short[~pool_short["id"].isin(pool_overlap["id"])]
    pool_excluded = pd.concat([pool_overlap, pool_short], ignore_index=True)
    pool_coverage = _coverage_exclusions(
        pool, _ids_outside(pool, pool_excluded), window, min_baseline_coverage
    )
    pool_excluded = pd.concat([pool_excluded, pool_coverage], ignore_index=True)

    if len(treatment_excluded) == len(treatment):
        raise ValueError(
            "No treatment meters remain: every treatment meter has fewer than 12 baseline "
            "months under a MONTH time period or baseline coverage below "
            f"min_baseline_coverage ({min_baseline_coverage})."
        )

    if len(pool_excluded) == len(pool):
        raise ValueError(
            "No comparison pool meters remain: every pool meter is also in the treatment "
            "group, has fewer than 12 baseline months under a MONTH time period, or has "
            f"baseline coverage below min_baseline_coverage ({min_baseline_coverage})."
        )

    treatment_data = _build_data(
        treatment, method, basis, data_settings, list(treatment_excluded["id"])
    )
    pool_data = _build_data(pool, method, basis, data_settings, list(pool_excluded["id"]))

    # A new algorithm instance per call: CG clustering rebuilds its own settings
    # during ``get_labels``.
    algorithm = _ALGORITHM[method](method_settings)
    clusters, treatment_weights = algorithm.get_comparison_group(treatment_data, pool_data)

    if method == SelectionMethod.CG_CLUSTERING:
        clusters = clusters.assign(weight=1.0)

    stage_ledger = pd.concat(
        [
            treatment_excluded,
            pool_excluded,
            _stage_exclusions(treatment_data, pool_data, treatment_weights),
        ],
        ignore_index=True,
    )

    selection = ComparisonGroupSelection(
        clusters=clusters,
        treatment_weights=treatment_weights,
        method=method,
        basis=basis,
        method_settings=settings_dump,
        data_settings=data_settings_dump,
        baseline_window=window,
        exclusions=stage_ledger,
        treatment_ids=treatment.ids,
        pool_ids=pool.ids,
        tz=treatment.tz,
    )

    return selection
