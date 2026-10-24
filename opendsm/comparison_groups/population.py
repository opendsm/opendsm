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
from io import StringIO

import numpy as np
import pandas as pd

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const
from opendsm.comparison_groups.warnings import ComparisonGroupWarning
from opendsm.eemeter.common.compliance import (
    normalize_enforce_compliance,
    raise_if_model_disqualified,
)
from opendsm.eemeter.common.data_processor_utilities import require_dataframe
from opendsm.eemeter.common.exceptions import (
    ComplianceError,
    DataSufficiencyError,
    DisqualifiedModelError,
    InfeasibleDataError,
    ModelFitError,
    NonstandardSettingsError,
)
from opendsm.eemeter.models import (
    BillingModel,
    DailyModel,
    HourlyModel,
)



SCHEMA_VERSION = 1

# Above this many models to_json() raises and points to to_ndjson().
_MAX_JSON_MODELS = 2000

_LOADSHAPE_SEED = 0

_MAX_POOL_SIZE = Data_Settings.model_fields["max_pool_size"].default

_MODEL_CLASSES = {
    model_cls.model_frequency.value: model_cls
    for model_cls in (HourlyModel, DailyModel, BillingModel)
}

_MODEL_CHOICES = {model_cls.model_choice: model_cls for model_cls in _MODEL_CLASSES.values()}

_DEFAULT_TIME_PERIOD = {
    "hourly": _const.TimePeriod.SEASONAL_HOURLY_DAY_OF_WEEK,
    "daily": _const.TimePeriod.SEASONAL_DAY_OF_WEEK,
    "billing": _const.TimePeriod.MONTH,
}

# Base input columns needed to reconstruct a Data object for a meter.
_INPUT_COLS = {
    "hourly": ("temperature", "ghi", "observed"),
    "daily": ("temperature", "observed"),
    "billing": ("temperature", "observed"),
}


def _fit_failure_origin_and_reason(exc) -> tuple:
    """The ledger origin naming the part of the fit a failure came from, and the
    ledger reason describing the failure."""
    if isinstance(exc, InfeasibleDataError):
        return "baseline_data", "no feasible model for the baseline data"

    if isinstance(exc, DataSufficiencyError):
        return "baseline_data", "insufficient baseline data"

    if isinstance(exc, NonstandardSettingsError):
        return "settings", "nonstandard settings"

    if isinstance(exc, DisqualifiedModelError):
        return "model", "model disqualified"

    if isinstance(exc, ModelFitError):
        return "model", "model fit failed"

    raise TypeError(f"Unrecognized fit failure {type(exc).__name__}")


def _model_frequency_of_model(model) -> str:
    # exact type, because BillingModel subclasses DailyModel
    for frequency, model_cls in _MODEL_CLASSES.items():
        if type(model) is model_cls:
            return frequency

    raise TypeError(
        f"Unsupported model type {type(model).__name__}; "
        f"expected one of {[cls.__name__ for cls in _MODEL_CLASSES.values()]}"
    )


def _checked_model_frequency(mid, model, model_frequency) -> str:
    """The model's frequency; raises ``TypeError`` when ``model_frequency`` is given and differs."""
    inferred = _model_frequency_of_model(model)
    if model_frequency is not None and inferred != model_frequency:
        raise TypeError(
            f"Meter {mid}: model frequency {inferred!r} does not match "
            f"requested {model_frequency!r}."
        )

    return inferred


def default_data_settings(model_frequency):
    """The ``Data_Settings`` a population's loadshapes use when the caller gives
    none: the model frequency's time period (``SEASONAL_HOURLY_DAY_OF_WEEK``,
    ``SEASONAL_DAY_OF_WEEK`` or ``MONTH``) and ``Data_Settings``'s defaults for
    every other field.

    Args:
        model_frequency: ``"hourly"``, ``"daily"`` or ``"billing"``.

    Raises:
        ValueError: ``model_frequency`` is not one of the three.
    """
    if model_frequency not in _DEFAULT_TIME_PERIOD:
        raise ValueError(
            f"model_frequency must be one of {list(_DEFAULT_TIME_PERIOD)}, got {model_frequency!r}"
        )

    settings = Data_Settings(time_period=_DEFAULT_TIME_PERIOD[model_frequency])

    return settings


def _fit_meters(
    baseline, model_frequency, is_electricity_data, settings, enforce_compliance, reporting
):
    """Fit one model per meter, serially. Returns ``(records, ledger)``: a record
    per meter that fits, carrying its ``reporting`` entry, and a population-stage
    ledger row per meter whose fit raises a compliance or model-fit error."""
    if model_frequency not in _MODEL_CLASSES:
        raise ValueError(
            f"model_frequency must be one of {list(_MODEL_CLASSES)}, got {model_frequency!r}"
        )

    reporting = {str(k): v for k, v in (reporting or {}).items()}
    unknown = set(reporting) - {str(mid) for mid in baseline}
    if unknown:
        raise KeyError(f"Reporting data for meters not in baseline: {sorted(unknown)}.")

    model_cls = _MODEL_CLASSES[model_frequency]
    records = {}
    ledger = exclusions.empty_ledger()

    for mid, df in baseline.items():
        require_dataframe(df, f"Meter {mid}: baseline data")

        model = model_cls(settings=settings)

        try:
            baseline_data, n_leading, n_trailing = model._build_baseline_data(
                df, bool(is_electricity_data)
            )
            model._fit_baseline(
                baseline_data,
                n_leading,
                n_trailing,
                bool(is_electricity_data),
                enforce_compliance,
            )
        except (ComplianceError, ModelFitError) as exc:
            origin, reason = _fit_failure_origin_and_reason(exc)
            detail = str(exc)
            if isinstance(exc, ComplianceError):
                detail = exclusions.format_warnings(exc.entries) or detail

            ledger = exclusions.append(ledger, [mid], "population", origin, reason, detail=detail)
            continue

        record = _MeterRecord(id=mid, model=model, baseline_data=baseline_data)

        reporting_df = reporting.get(str(mid))
        if reporting_df is not None:
            require_dataframe(reporting_df, f"Meter {mid}: reporting data")
            record.reporting_data = model._reporting_data(reporting_df)

        records[str(mid)] = record

    if not records:
        summary = "; ".join(f"{row.id}: {row.reason}" for row in ledger.itertuples())
        raise ValueError(f"All meters failed to fit: {summary}")

    return records, ledger


def _pool_trim(ids, max_pool_size, seed):
    """Split ``ids`` into ``(kept, trimmed)``, each in input order. The kept ids
    are the ``max_pool_size`` with the smallest SHA-256 of seed and id, so
    membership depends on neither input order nor which other ids are present
    beyond the cut; ``None`` keeps every id."""
    ids = [str(mid) for mid in ids]
    if max_pool_size is None or len(ids) <= max_pool_size:
        return ids, []

    def rank(mid):
        return hashlib.sha256(f"{seed}:{mid}".encode()).digest()

    kept_set = set(sorted(ids, key=rank)[:max_pool_size])
    kept = [mid for mid in ids if mid in kept_set]
    trimmed = [mid for mid in ids if mid not in kept_set]

    return kept, trimmed


def _record_pool_trim(ledger, trimmed):
    ledger = exclusions.append(
        ledger,
        trimmed,
        "population",
        "pool_trim",
        "outside the max_pool_size lowest seeded id hashes",
    )

    return ledger


def _serialize_unc(unc):
    if unc is None:
        return None

    if isinstance(unc, pd.Series):
        table = unc.rename("observed_unc").to_frame().to_json(orient="table")

        return {"series": table}

    return float(unc)


def _deserialize_unc(payload, tz):
    if payload is None:
        return None

    if isinstance(payload, dict):
        frame = pd.read_json(StringIO(payload["series"]), orient="table")
        series = frame["observed_unc"]
        series.index = series.index.tz_convert(tz)

        return series

    return float(payload)


class _MeterRecord:
    __slots__ = ("id", "model", "baseline_data", "reporting_data", "observed_unc")

    def __init__(self, id, model, baseline_data=None, reporting_data=None, observed_unc=None):
        self.id = str(id)
        self.model = model
        self.baseline_data = baseline_data
        self.reporting_data = reporting_data
        self.observed_unc = observed_unc


class MeterPopulation:
    """A set of per-meter energy models sharing one model frequency, timezone, and
    fuel type, plus their baseline and reporting data. Baseline data is optional
    per meter only on a ``from_json``/``from_ndjson`` reload; selection and
    baseline predictions require it.

    Uncertainty columns (``modeled_unc``) are the models' t-scaled ASHRAE-style
    bands, not calibrated 1-sigma intervals.

    ``ledger`` is the population's initial exclusions ledger, in the shared
    ledger schema; empty when omitted.

    ``enforce_compliance`` is the caller's policy, stored and serialized as passed. Every
    prediction the population makes enforces it less ``model``, with ``data`` judged on
    each model's fit records only, so a population built on relaxed enforcement keeps
    predicting on the meters it admitted.
    """

    _role = "population"

    def __init__(
        self,
        meters,
        model_frequency,
        features=None,
        ledger=None,
        enforce_compliance=True,
    ):
        if not meters:
            raise ValueError("A population must contain at least one meter.")

        if model_frequency not in _MODEL_CLASSES:
            raise ValueError(
                f"model_frequency must be one of {list(_MODEL_CLASSES)}, got {model_frequency!r}"
            )

        self.model_frequency = model_frequency
        self._meters = {str(k): v for k, v in meters.items()}
        # Raises on a bad policy at construction rather than at the first prediction.
        normalize_enforce_compliance(enforce_compliance)
        self.enforce_compliance = enforce_compliance
        self.features = self._validate_features(features)

        if ledger is None:
            ledger = exclusions.empty_ledger()

        self._exclusions = ledger
        self._baseline_pred = {}
        self._reporting_pred = {}

        self.tz, self.is_electricity_data = self._validate_meters()

    # -- construction --------------------------------------------------------

    @classmethod
    def from_fit_models(cls, meters, model_frequency=None, features=None, **kwargs):
        """Build a population from per-meter fitted models (instances or model
        payloads).

        Args:
            meters: ``{id: {model, baseline_df, reporting_df=None,
                observed_unc=None}}``. ``model`` may be a fitted instance, a
                ``to_dict()`` payload, or a ``to_json()`` string. A payload that
                lacks ``is_electricity_data`` raises ``ValueError``; load it with its
                model's ``from_dict(payload, is_electricity_data=...)`` and pass the
                instance. ``baseline_df`` and ``reporting_df`` are the meter's
                dataframes; the model builds the data objects from them. An entry
                without ``baseline_df`` raises ``ValueError``.
            model_frequency: Inferred from each model: a fitted instance's type or a
                payload's ``model_choice``. Required only for a payload without
                ``model_choice``; when supplied it is validated against the inferred
                value (mismatch raises ``TypeError``). A payload whose ``model_choice``
                names no supported model raises ``ValueError``.
        """
        records = {}
        resolved = model_frequency

        for mid, entry in meters.items():
            model_obj = entry["model"]

            if isinstance(model_obj, (dict, str)):
                payload = model_obj
                if isinstance(payload, str):
                    payload = json.loads(payload)

                choice = payload.get("model_choice")
                if choice is None:
                    if model_frequency is None:
                        raise ValueError(
                            "model_frequency is required when models are passed as dict or "
                            "JSON payloads without a model_choice."
                        )
                    model_cls = _MODEL_CLASSES[model_frequency]

                elif choice not in _MODEL_CHOICES:
                    raise ValueError(
                        f"Meter {mid}: payload model_choice {choice!r} is not supported; "
                        f"expected one of {sorted(_MODEL_CHOICES)}."
                    )

                else:
                    model_cls = _MODEL_CHOICES[choice]

                model = model_cls.from_dict(payload)

            else:
                model = model_obj

            inferred = _checked_model_frequency(mid, model, model_frequency)
            if resolved is None:
                resolved = inferred

            if "baseline_df" not in entry:
                raise ValueError(f"Meter {mid}: baseline_df is required.")

            baseline_df = entry["baseline_df"]
            require_dataframe(baseline_df, f"Meter {mid}: baseline_df")
            baseline_data, _, _ = model._baseline_data(baseline_df)
            reporting_df = entry.get("reporting_df")

            record = _MeterRecord(
                id=mid,
                model=model,
                baseline_data=baseline_data,
                observed_unc=entry.get("observed_unc"),
            )

            if reporting_df is not None:
                require_dataframe(reporting_df, f"Meter {mid}: reporting_df")
                record.reporting_data = model._reporting_data(reporting_df)

            records[str(mid)] = record

        if resolved is None:
            raise ValueError("Could not resolve model_frequency; pass it explicitly.")

        return cls(records, resolved, features=features, **kwargs)

    @classmethod
    def from_data(
        cls,
        baseline,
        model_frequency,
        *,
        is_electricity_data,
        settings=None,
        enforce_compliance=True,
        features=None,
        reporting=None,
    ):
        """Fit one model per meter, serially.

        A meter whose fit raises a compliance or model-fit error is excluded and
        recorded on the population's exclusions ledger under the leaf exception
        name while the loop continues.

        Args:
            baseline: ``{id: baseline_df}`` for the requested model frequency.
            model_frequency: ``"hourly"``, ``"daily"``, or ``"billing"``.
            is_electricity_data: Whether the observed values are electricity usage.
            enforce_compliance: Which compliance members to enforce, passed through
                to each fit unchanged and kept for the population's predictions.
            reporting: Optional ``{id: reporting_df}``. Reporting for a meter
                that is excluded is ignored.

        Raises:
            ValueError: ``model_frequency`` is unknown, or every meter fails to fit.
            TypeError: A baseline value is not the frequency's baseline-data class.
            KeyError: A ``reporting`` id is not in ``baseline``.
        """
        records, ledger = _fit_meters(
            baseline,
            model_frequency,
            is_electricity_data,
            settings,
            enforce_compliance,
            reporting,
        )
        population = cls(
            records,
            model_frequency,
            features=features,
            ledger=ledger,
            enforce_compliance=enforce_compliance,
        )

        return population

    # -- validation ----------------------------------------------------------

    def _validate_features(self, features):
        if features is None:
            return None

        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame indexed by meter id.")

        validated = features.copy()
        validated.index = validated.index.astype(str)

        return validated

    def _validate_meters(self):
        timezones = set()
        fuels = set()
        ghi_flags = set()
        disqualified = []

        for mid, rec in self._meters.items():
            _checked_model_frequency(mid, rec.model, self.model_frequency)
            tz = str(rec.model.baseline_timezone)
            baseline = rec.baseline_data
            disqualification = rec.model.compliance.disqualification

            # the model builds its meter's data objects, so its own fuel flag is
            # every meter's, with or without baseline data
            fuels.add(bool(rec.model.is_electricity_data))

            if baseline is not None:
                if tz != str(baseline.tz):
                    raise ValueError(
                        f"Meter {mid}: model timezone {tz} does not match baseline data "
                        f"timezone {baseline.tz}."
                    )

                disqualification = disqualification or baseline.compliance.disqualification

            if rec.reporting_data is not None and str(rec.reporting_data.tz) != tz:
                raise ValueError(
                    f"Meter {mid}: reporting data timezone {rec.reporting_data.tz} does not "
                    f"match model timezone {tz}."
                )

            # an hourly model fit with train_features omitting ghi ignores a ghi
            # column its data carries, so solar is what the model uses; daily and
            # billing models and data carry no ghi
            if self.model_frequency == "hourly":
                ghi_flags.add("ghi" in rec.model._ts_features)

            timezones.add(tz)

            if disqualification:
                disqualified.append(mid)

        if len(timezones) != 1:
            raise ValueError(f"All meters must share one timezone; found {sorted(timezones)}.")

        if len(fuels) != 1:
            raise ValueError(
                f"All meters must share one is_electricity_data flag; found {sorted(fuels)}."
            )

        if len(ghi_flags) > 1:
            raise ValueError("Population mixes solar (using ghi) and non-solar meters.")

        if disqualified:
            entry = ComparisonGroupWarning(
                qualified_name="comparison_groups.population.disqualified_meters",
                description=(
                    f"Population includes disqualified meters or baselines: {disqualified}."
                ),
                severity="advisory",
                data={"ids": disqualified},
            )
            entry.warn()

        tz = timezones.pop()
        is_electricity_data = fuels.pop()

        return tz, is_electricity_data

    def baseline_span(self):
        """The union of the meters' baseline spans as ``(min start, max end)``
        over the meters that carry baseline data, or ``None`` when none do."""
        starts = []
        ends = []

        for rec in self._meters.values():
            if rec.baseline_data is None:
                continue

            index = rec.baseline_data.df.index
            starts.append(index.min())
            ends.append(index.max())

        if not starts:
            return None

        window = (min(starts), max(ends))

        return window

    def ids_without_baseline(self):
        """Ids of the meters carrying no baseline data, in ``ids`` order."""
        ids = [mid for mid, rec in self._meters.items() if rec.baseline_data is None]

        return ids

    # -- accessors -----------------------------------------------------------

    @property
    def ids(self):
        """Meter ids, in the order the meters were added."""
        return list(self._meters.keys())

    def meter(self, mid):
        """The record of meter ``mid`` (compared as a string), with ``id``,
        ``model``, ``baseline_data``, ``reporting_data`` and ``observed_unc``
        attributes, or ``None`` when the population holds no such meter. The
        record is the population's own: callers must not modify it."""
        return self._meters.get(str(mid))

    @property
    def role(self):
        """Population role: ``"treatment"`` for a ``TreatmentGroup``, ``"pool"`` for a
        ``ComparisonPool``."""
        return self._role

    @property
    def exclusions(self):
        """Ledger of meters dropped at the population stage, in the shared
        ``[id, stage, origin, reason, detail]`` schema."""
        return self._exclusions

    def __len__(self):
        return len(self._meters)

    # -- predictions ---------------------------------------------------------

    def _predict_record(self, rec, period):
        """Predict ``period`` data for one meter. A billing model predicts
        unaggregated, on its daily substrate: a billing meter's ``MONTH``
        loadshape is therefore a daily-mean per calendar month, not a monthly
        total. That reshaping only touches selection features (loadshapes),
        never a corrected savings number."""
        if period == "baseline":
            data = rec.baseline_data
        else:
            data = rec.reporting_data

        if data is None:
            raise ValueError(f"Meter {rec.id}: no {period} data attached.")

        # A population may admit a meter whose fit metrics failed (`_validate_meters`),
        # so enforcing `model` here would undo that admission. Correction applies its
        # own observed and joint pruning to reporting data, so `data` is enforced on
        # the model's fit records and not on the data predicted on.
        members = normalize_enforce_compliance(self.enforce_compliance) - {"model"}
        raise_if_model_disqualified(rec.model.compliance, members)

        pred = rec.model._predict_data(data, enforce_compliance=sorted(members - {"data"}))

        frame = pd.DataFrame(
            {
                "observed": pred["observed"].to_numpy(),
                "modeled": pred["predicted"].to_numpy(),
                "modeled_unc": pred["predicted_unc"].to_numpy(),
            },
            index=pred.index,
        )
        frame.index.name = "datetime"

        return frame

    def prediction_frames(self, period, ids=None):
        """Per-meter prediction frames ``{id: frame}``, each indexed by tz-aware
        ``datetime`` with columns ``observed``, ``modeled`` and ``modeled_unc``. A
        meter not yet predicted for ``period`` is predicted and cached; a model
        prediction error propagates and caches nothing for that meter. The
        frames are the cached objects: callers must not modify them.

        Args:
            period: ``"baseline"`` or ``"reporting"``.
            ids: Meter ids to return, in the given order; every meter when omitted.

        Raises:
            ValueError: ``period`` is neither ``"baseline"`` nor ``"reporting"``, or a
                requested meter has no ``period`` data attached.
            KeyError: An id is not in the population.
        """
        if period == "baseline":
            cache = self._baseline_pred
        elif period == "reporting":
            cache = self._reporting_pred
        else:
            raise ValueError(f"period must be 'baseline' or 'reporting', got {period!r}.")

        if ids is None:
            ids = list(self._meters)

        for mid in ids:
            if mid not in cache:
                cache[mid] = self._predict_record(self._meters[mid], period)

        preds = {mid: cache[mid] for mid in ids}

        return preds

    def observed_unc_at(self, mid, index):
        """Meter ``mid``'s observed uncertainty at each timestamp of ``index``, as
        a float array: zeros when the meter carries none, its scalar repeated, or
        its series read at ``index``.

        Raises:
            ValueError: The meter's observed-uncertainty series does not cover
                every timestamp of ``index``.
        """
        unc = self._meters[mid].observed_unc

        if unc is None:
            values = np.zeros(len(index), dtype=float)
        elif isinstance(unc, pd.Series):
            if not index.isin(unc.index).all():
                raise ValueError(
                    f"observed_unc series for meter {mid} does not cover the requested "
                    f"window {index[0]} to {index[-1]}."
                )

            values = unc.reindex(index).to_numpy(dtype=float)
        else:
            values = np.full(len(index), float(unc))

        return values

    def prediction_matrices(self, period, ids=None):
        """Predictions as dense arrays over the union of the meters' timestamps:
        ``(index, ids, observed, modeled, modeled_unc, observed_unc)``, each array
        float32 ``[T, M]`` with NaN where a meter has no row (``observed_unc``: 0).
        The arrays are new on every call; predictions come from
        ``prediction_frames``.

        Args:
            period: ``"baseline"`` or ``"reporting"``.
            ids: Meter ids for the columns, in the given order; every meter when
                omitted.

        Returns:
            The sorted union ``DatetimeIndex``, the column ids, and the four
            arrays; ``observed_unc`` is None when no included meter carries
            observed uncertainty.

        Raises:
            ValueError: As ``prediction_frames``, or a meter's observed-uncertainty
                series does not cover its prediction index.
        """
        preds = self.prediction_frames(period, ids)
        ids = list(preds)

        index = None
        for frame in preds.values():
            if index is None:
                index = frame.index
            else:
                index = index.union(frame.index)
        index = index.sort_values()

        n_times = len(index)
        n_meters = len(ids)
        observed = np.full((n_times, n_meters), np.nan, dtype=np.float32)
        modeled = np.full((n_times, n_meters), np.nan, dtype=np.float32)
        modeled_unc = np.full((n_times, n_meters), np.nan, dtype=np.float32)

        any_observed_unc = any(self._meters[mid].observed_unc is not None for mid in ids)
        if any_observed_unc:
            observed_unc = np.zeros((n_times, n_meters), dtype=np.float32)
        else:
            observed_unc = None

        for col, mid in enumerate(ids):
            frame = preds[mid]
            rows = index.get_indexer(frame.index)
            observed[rows, col] = frame["observed"].to_numpy(dtype=np.float32)
            modeled[rows, col] = frame["modeled"].to_numpy(dtype=np.float32)
            modeled_unc[rows, col] = frame["modeled_unc"].to_numpy(dtype=np.float32)

            if observed_unc is not None:
                values = self.observed_unc_at(mid, frame.index)
                observed_unc[rows, col] = values.astype(np.float32)

        return index, ids, observed, modeled, modeled_unc, observed_unc

    # -- loadshape data ------------------------------------------------------

    def short_baseline_months(self):
        """Baseline month counts, as ``{id: months}``, of the meters with fewer
        than 12 baseline months, which cannot support a ``MONTH`` loadshape. Every
        meter must carry baseline data."""
        short = {}

        for mid, rec in self._meters.items():
            naive_index = rec.baseline_data.df.index.tz_localize(None)
            months = naive_index.to_period("M").nunique()
            if months < 12:
                short[mid] = months

        return short

    def _loadshape_frames(self, basis, ids):
        frames = []

        if basis == _const.LoadshapeType.OBSERVED:
            for mid in ids:
                observed = self._meters[mid].baseline_data.df["observed"]
                frame = pd.DataFrame(
                    {
                        "id": mid,
                        "datetime": observed.index.tz_localize(None),
                        "observed": observed.to_numpy(),
                    }
                )
                frames.append(frame)

            return frames

        preds = self.prediction_frames("baseline", ids)

        for mid, pred in preds.items():
            observed = pred["observed"].to_numpy().copy()
            modeled = pred["modeled"].to_numpy().copy()

            if basis == _const.LoadshapeType.ERROR:
                zero_modeled = modeled == 0
                observed[zero_modeled] = np.nan
                modeled[zero_modeled] = np.nan

            frame = pd.DataFrame(
                {
                    "id": mid,
                    "datetime": pred.index.tz_localize(None),
                    "observed": observed,
                    "modeled": modeled,
                }
            )
            frames.append(frame)

        return frames

    def loadshape_data(self, basis, data_settings=None, exclude=(), features=None):
        """Build a ``common.Data`` from baseline predictions. Datetimes are made
        tz-naive (local clock) so ``Data`` bins on local time. ``data_settings``
        supplies the aggregation, time period, interpolation, minimum data
        fraction, season and weekday/weekend definitions; the loadshape type is
        ``basis``. Pool trimming happens at ``ComparisonPool`` construction, so
        ``Data``'s own trim is disabled here.

        Args:
            basis: Loadshape type: ``"observed"``, ``"modeled"`` or ``"error"``.
            data_settings: Optional ``Data_Settings`` supplying the fields above;
                ``default_data_settings`` of the population's model frequency when
                omitted.
            exclude: Meter ids left out of the loadshapes, compared as strings. The
                population itself is unchanged.
            features: Optional stratification features indexed by meter id, compared
                as strings. Only the rows of meters with a loadshape are kept, so a
                row with no meter behind it (absent, excluded, or dropped by the
                loadshape build) cannot be drawn.

        Returns:
            The ``common.Data`` built from the baseline predictions.

        Raises:
            ValueError: A meter has no baseline data attached.
        """
        missing = self.ids_without_baseline()
        if missing:
            raise ValueError(f"Loadshapes require baseline data; none attached for {missing}.")

        basis = _const.LoadshapeType(basis)

        if data_settings is None:
            data_settings = default_data_settings(self.model_frequency)

        caller_settings = data_settings.model_dump(
            include={
                "agg_type",
                "time_period",
                "interpolate_missing",
                "min_data_pct_required",
                "season",
                "weekday_weekend",
            }
        )

        exclude = {str(mid) for mid in exclude}
        ids = [mid for mid in self._meters if str(mid) not in exclude]
        frames = self._loadshape_frames(basis, ids)
        time_series = pd.concat(frames, ignore_index=True)

        features_df = None
        if features is not None:
            features_df = features.rename_axis("id").reset_index()
            features_df["id"] = features_df["id"].astype(str)
            features_df = features_df[features_df["id"].isin(ids)]

        settings = Data_Settings(
            **caller_settings,
            loadshape_type=basis,
            max_pool_size=len(ids),
            seed=_LOADSHAPE_SEED,
        )
        data = Data(time_series_df=time_series, features_df=features_df, settings=settings)

        return data

    # -- serialization -------------------------------------------------------

    def _header(self):
        header = {
            "schema_version": SCHEMA_VERSION,
            "model_frequency": self.model_frequency,
            "role": self._role,
            "tz": self.tz,
            "is_electricity_data": bool(self.is_electricity_data),
            "exclusions": self._exclusions.to_json(orient="table"),
            "enforce_compliance": self.enforce_compliance,
        }

        return header

    def _serialize_record(self, rec):
        record = {
            "id": rec.id,
            "model": rec.model.to_json(),
            "observed_unc": _serialize_unc(rec.observed_unc),
        }

        return record

    def to_json(self):
        """Header plus per-meter models and observed uncertainty. Raw meter data
        is not embedded; refuses above the JSON model ceiling."""
        if len(self._meters) > _MAX_JSON_MODELS:
            raise ValueError(
                f"Population has {len(self._meters)} models, above the {_MAX_JSON_MODELS} JSON "
                "ceiling; use to_ndjson() for pool-scale persistence."
            )

        payload = {
            "header": self._header(),
            "meters": [self._serialize_record(rec) for rec in self._meters.values()],
        }

        return json.dumps(payload, allow_nan=False)

    def to_ndjson(self):
        """One JSON object per line: the header first, then one meter record per
        line; no model-count ceiling."""
        lines = [json.dumps({"header": self._header()}, allow_nan=False)]

        for rec in self._meters.values():
            lines.append(json.dumps(self._serialize_record(rec), allow_nan=False))

        text = "\n".join(lines)

        return text

    @classmethod
    def _from_payload(cls, header, meter_dicts, baseline, reporting, features):
        if header["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {header['schema_version']}; "
                f"expected {SCHEMA_VERSION}."
            )

        if header["role"] != cls._role:
            raise ValueError(
                f"Serialized role {header['role']!r} does not match {cls.__name__} "
                f"role {cls._role!r}."
            )

        model_frequency = header["model_frequency"]
        model_cls = _MODEL_CLASSES[model_frequency]
        tz = header["tz"]
        baseline = {str(k): v for k, v in (baseline or {}).items()}
        reporting = {str(k): v for k, v in (reporting or {}).items()}
        records = {}

        for meter in meter_dicts:
            mid = str(meter["id"])

            # from_dict rejects a supplied flag when the payload stores one
            payload = json.loads(meter["model"])
            if (payload.get("info") or {}).get("is_electricity_data") is not None:
                model = model_cls.from_dict(payload)
            else:
                model = model_cls.from_dict(
                    payload, is_electricity_data=header["is_electricity_data"]
                )

            record = _MeterRecord(
                id=mid,
                model=model,
                observed_unc=_deserialize_unc(meter["observed_unc"], tz),
            )

            baseline_df = baseline.get(mid)
            if baseline_df is not None:
                require_dataframe(baseline_df, f"Meter {mid}: baseline data")
                record.baseline_data, _, _ = model._baseline_data(baseline_df)

            reporting_df = reporting.get(mid)
            if reporting_df is not None:
                require_dataframe(reporting_df, f"Meter {mid}: reporting data")
                record.reporting_data = model._reporting_data(reporting_df)

            records[mid] = record

        population = cls(
            records,
            model_frequency,
            features=features,
            ledger=exclusions.read_table_json(header["exclusions"]),
            **cls._header_settings(header),
        )

        return population

    @classmethod
    def _header_settings(cls, header):
        """Constructor arguments a subclass restores from its serialized header."""
        return {"enforce_compliance": header["enforce_compliance"]}

    @classmethod
    def from_json(cls, s, baseline=None, reporting=None, features=None):
        """Rebuild from ``to_json()`` output, re-attaching caller-supplied
        baseline and reporting frames by id.

        Args:
            s: JSON string written by ``to_json()``.
            baseline: Optional mapping of meter id to baseline frame. A meter
                without an entry takes its timezone, fuel flag and
                disqualification from its model, and cannot be selected or
                baseline-predicted.
            reporting: Optional mapping of meter id to reporting frame.
            features: Optional feature frame indexed by meter id.

        Returns:
            The rebuilt population.

        Raises:
            ValueError: The ``schema_version`` is not supported, or the serialized
                role differs from this class's.
        """
        payload = json.loads(s)
        population = cls._from_payload(
            payload["header"], payload["meters"], baseline, reporting, features
        )

        return population

    @classmethod
    def from_ndjson(cls, text, baseline=None, reporting=None, features=None):
        """Rebuild from ``to_ndjson()`` output (header line then meter lines).

        Args:
            text: NDJSON text written by ``to_ndjson()``.
            baseline: Optional mapping of meter id to baseline frame, as for
                ``from_json``.
            reporting: Optional mapping of meter id to reporting frame.
            features: Optional feature frame indexed by meter id.

        Returns:
            The rebuilt population.

        Raises:
            ValueError: The ``schema_version`` is not supported, or the serialized
                role differs from this class's.
        """
        lines = [line for line in text.splitlines() if line.strip()]
        header = json.loads(lines[0])["header"]
        meter_dicts = [json.loads(line) for line in lines[1:]]

        return cls._from_payload(header, meter_dicts, baseline, reporting, features)


class TreatmentGroup(MeterPopulation):
    """A population playing the treatment role in a savings analysis."""

    _role = "treatment"


class ComparisonPool(MeterPopulation):
    """A population playing the comparison-pool role. Sole owner of the
    ``max_pool_size`` trim, recorded in its ``exclusions`` ledger.

    ``max_pool_size`` caps the pool's meters: omitted, it is 10000; ``None``
    means no cap. For a given ``seed`` the same meters are kept whatever the
    input order and whether the pool is built by ``from_data`` or
    ``from_fit_models``; ``ids`` keeps the input order. The trimmed meters are
    appended to ``ledger``. A ``from_json``/``from_ndjson`` reload restores the
    saved meters, ledger, ``max_pool_size`` and ``seed``; a payload written by
    ``to_json``/``to_ndjson`` holds no more meters than its cap, so the reload
    trims none.
    """

    _role = "pool"

    def __init__(
        self,
        meters,
        model_frequency,
        features=None,
        ledger=None,
        max_pool_size=_MAX_POOL_SIZE,
        seed=0,
        enforce_compliance=True,
    ):
        super().__init__(
            meters,
            model_frequency,
            features=features,
            ledger=ledger,
            enforce_compliance=enforce_compliance,
        )
        self.max_pool_size = max_pool_size
        self.seed = seed

        _, trimmed = _pool_trim(self._meters, max_pool_size, seed)
        for mid in trimmed:
            del self._meters[mid]

        self._exclusions = _record_pool_trim(self._exclusions, trimmed)

    @classmethod
    def from_data(
        cls,
        baseline,
        model_frequency,
        *,
        is_electricity_data,
        settings=None,
        enforce_compliance=True,
        features=None,
        reporting=None,
        max_pool_size=_MAX_POOL_SIZE,
        seed=0,
    ):
        """``MeterPopulation.from_data`` fitting only the meters the
        ``max_pool_size`` trim keeps; the others are ledgered as ``pool_trim``
        without a fit. A kept meter that fails to fit is not replaced, so the
        pool can end below the cap."""
        _, trimmed = _pool_trim(baseline, max_pool_size, seed)
        trimmed_set = set(trimmed)
        kept_baseline = {
            mid: bdata for mid, bdata in baseline.items() if str(mid) not in trimmed_set
        }
        # reporting for an id outside ``baseline`` still reaches the fit's KeyError
        kept_reporting = {
            mid: data for mid, data in (reporting or {}).items() if str(mid) not in trimmed_set
        }

        records, ledger = _fit_meters(
            kept_baseline,
            model_frequency,
            is_electricity_data,
            settings,
            enforce_compliance,
            kept_reporting,
        )
        ledger = _record_pool_trim(ledger, trimmed)
        pool = cls(
            records,
            model_frequency,
            features=features,
            ledger=ledger,
            max_pool_size=max_pool_size,
            seed=seed,
            enforce_compliance=enforce_compliance,
        )

        return pool

    def _header(self):
        header = super()._header()
        header["max_pool_size"] = self.max_pool_size
        header["seed"] = self.seed

        return header

    @classmethod
    def _header_settings(cls, header):
        settings = {
            **super()._header_settings(header),
            "max_pool_size": header["max_pool_size"],
            "seed": header["seed"],
        }

        return settings
