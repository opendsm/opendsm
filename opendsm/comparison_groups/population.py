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
import warnings
from io import StringIO

import numpy as np
import pandas as pd

from opendsm.comparison_groups import exclusions
from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)
from opendsm.eemeter.models import (
    BillingModel,
    DailyModel,
    HourlyModel,
)



SCHEMA_VERSION = 1

# Above this many models, JSON serialization is refused in favor of the
# streamed one-record-per-line NDJSON path (pool-scale persistence).
_MAX_JSON_MODELS = 2000

_LOADSHAPE_SEED = 0

# Pool meters kept when the caller names no cap; ``Data_Settings.max_pool_size``'s
# default.
_MAX_POOL_SIZE = 10000

# Stands in for an omitted ``max_pool_size`` so ``_MAX_POOL_SIZE`` is read at call
# time; ``None`` already means no cap.
_DEFAULT_POOL_SIZE = object()

_MODEL_CLASSES = {
    model_cls.model_frequency.value: model_cls
    for model_cls in (HourlyModel, DailyModel, BillingModel)
}

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


def _model_frequency_of_model(model) -> str:
    # exact type, because BillingModel subclasses DailyModel
    for frequency, model_cls in _MODEL_CLASSES.items():
        if type(model) is model_cls:
            return frequency

    raise TypeError(
        f"Unsupported model type {type(model).__name__}; "
        f"expected one of {[cls.__name__ for cls in _MODEL_CLASSES.values()]}"
    )


def _model_from_payload(model_cls, payload):
    if isinstance(payload, str):
        return model_cls.from_json(payload)

    if isinstance(payload, dict):
        return model_cls.from_dict(payload)

    raise TypeError(f"Model payload must be a dict or JSON string, got {type(payload).__name__}")


def _require_frame(mid, name, value):
    """Reject a data object where a dataframe is expected."""
    if not isinstance(value, pd.DataFrame) and hasattr(value, "df"):
        raise TypeError(
            f"Meter {mid}: {name} must be a pandas DataFrame, got a "
            f"{type(value).__name__}. Populations build the data objects themselves."
        )


def _payload_has_is_electricity_data(payload):
    """Whether a dict/JSON model payload already stores its own fuel flag."""
    if isinstance(payload, str):
        payload = json.loads(payload)

    info = payload.get("info") or {}

    return "is_electricity_data" in info


def _window_to_json(window):
    if window is None:
        return None

    start, end = window
    bounds = [start.isoformat(), end.isoformat()]

    return bounds


def _coerce_window(window, tz):
    """Normalize a ``(start, end)`` window (datetime-likes or serialized ISO
    strings) to a tuple of tz-aware ``Timestamp`` in ``tz``. ``None`` passes
    through."""
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


def _resolve_max_pool_size(max_pool_size):
    if max_pool_size is _DEFAULT_POOL_SIZE:
        return _MAX_POOL_SIZE

    return max_pool_size


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
    """

    _role = "population"

    def __init__(
        self,
        meters,
        model_frequency,
        features=None,
    ):
        if not meters:
            raise ValueError("A population must contain at least one meter.")

        if model_frequency not in _MODEL_CLASSES:
            raise ValueError(
                f"model_frequency must be one of {list(_MODEL_CLASSES)}, got {model_frequency!r}"
            )

        self.model_frequency = model_frequency
        self._meters = {str(k): v for k, v in meters.items()}
        self.features = self._validate_features(features)
        self._exclusions = exclusions.empty_ledger()
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
                dataframes; the model builds the data objects from them.
            model_frequency: Inferred from each fitted instance's type, so optional
                for instances, and validated against it when supplied (mismatch
                raises ``TypeError``). Required when models are passed as payloads.
        """
        records = {}
        resolved = model_frequency

        for mid, entry in meters.items():
            model_obj = entry["model"]

            if isinstance(model_obj, (dict, str)):
                if model_frequency is None:
                    raise ValueError(
                        "model_frequency is required when models are passed as dict or JSON "
                        "payloads."
                    )
                inferred = model_frequency
                model = _model_from_payload(_MODEL_CLASSES[inferred], model_obj)

            else:
                inferred = _model_frequency_of_model(model_obj)
                if model_frequency is not None and inferred != model_frequency:
                    raise TypeError(
                        f"Meter {mid}: model frequency {inferred!r} does not match "
                        f"requested {model_frequency!r}."
                    )
                model = model_obj

            if resolved is None:
                resolved = inferred

            baseline_df = entry["baseline_df"]
            _require_frame(mid, "baseline_df", baseline_df)
            reporting_df = entry.get("reporting_df")

            record = _MeterRecord(
                id=mid,
                model=model,
                baseline_data=model._baseline_data(baseline_df),
                observed_unc=entry.get("observed_unc"),
            )

            if reporting_df is not None:
                _require_frame(mid, "reporting_df", reporting_df)
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
        ignore_disqualification=False,
        features=None,
        reporting=None,
        **kwargs,
    ):
        """Fit one model per meter serially (never in parallel).

        A meter whose fit raises a data-sufficiency or model-disqualification
        error is excluded and recorded on the population's exclusions ledger
        while the loop continues; raises only when every meter fails.

        Args:
            baseline: ``{id: baseline_df}`` for the requested model frequency.
            model_frequency: ``"hourly"``, ``"daily"``, or ``"billing"``.
            is_electricity_data: Whether the observed values are electricity usage.
            reporting: Optional ``{id: reporting_df}``. Reporting for a meter
                that is excluded is ignored.

        Raises:
            KeyError: A ``reporting`` id is not in ``baseline``.
        """
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
        dropped = exclusions.empty_ledger()

        for mid, df in baseline.items():
            _require_frame(mid, "baseline data", df)

            model = model_cls(settings=settings)

            try:
                model.fit(
                    df,
                    is_electricity_data=is_electricity_data,
                    ignore_disqualification=ignore_disqualification,
                )
            except DataSufficiencyError as exc:
                detail = exclusions.format_warnings(exc.disqualification) or str(exc)
                dropped = exclusions.append(
                    dropped,
                    [mid],
                    "population",
                    "baseline_data",
                    "baseline data insufficient to fit a model",
                    detail=detail,
                )
                continue
            except DisqualifiedModelError as exc:
                detail = exclusions.format_warnings(model.disqualification) or str(exc)
                dropped = exclusions.append(
                    dropped,
                    [mid],
                    "population",
                    "model",
                    "fitted model disqualified",
                    detail=detail,
                )
                continue

            record = _MeterRecord(id=mid, model=model, baseline_data=model._baseline_data(df))

            reporting_df = reporting.get(str(mid))
            if reporting_df is not None:
                _require_frame(mid, "reporting data", reporting_df)
                record.reporting_data = model._reporting_data(reporting_df)

            records[str(mid)] = record

        if not records:
            summary = "; ".join(f"{row.id}: {row.reason}" for row in dropped.itertuples())
            raise ValueError(f"All meters failed to fit: {summary}")

        population = cls(records, model_frequency, features=features, **kwargs)
        if not dropped.empty:
            population._exclusions = pd.concat(
                [dropped, population._exclusions], ignore_index=True
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
            if _model_frequency_of_model(rec.model) != self.model_frequency:
                raise TypeError(
                    f"Meter {mid}: model type does not match population model frequency "
                    f"{self.model_frequency!r}."
                )

            tz = str(rec.model.baseline_timezone)
            baseline = rec.baseline_data
            disqualification = rec.model.disqualification

            # the model builds its meter's data objects, so its own fuel flag is
            # every meter's, with or without baseline data
            fuels.add(bool(rec.model._is_electricity_data))

            if baseline is not None:
                if tz != str(baseline.tz):
                    raise ValueError(
                        f"Meter {mid}: model timezone {tz} does not match baseline data "
                        f"timezone {baseline.tz}."
                    )

                disqualification = disqualification or baseline.disqualification

            if rec.reporting_data is not None and str(rec.reporting_data.tz) != tz:
                raise ValueError(
                    f"Meter {mid}: reporting data timezone {rec.reporting_data.tz} does not "
                    f"match model timezone {tz}."
                )

            # an hourly model fit with train_features omitting ghi ignores a ghi
            # column its data carries, so solar is what the model uses
            if self.model_frequency == "hourly":
                ghi_flags.add("ghi" in rec.model._ts_features)
            elif baseline is not None:
                ghi_flags.add("ghi" in baseline.df.columns)

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
            warnings.warn(
                f"Population includes disqualified meters or baselines: {disqualified}.",
                stacklevel=2,
            )

        tz = timezones.pop()
        is_electricity_data = fuels.pop()

        return tz, is_electricity_data

    def _baseline_span(self):
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

    def _ids_without_baseline(self):
        ids = [mid for mid, rec in self._meters.items() if rec.baseline_data is None]

        return ids

    # -- accessors -----------------------------------------------------------

    @property
    def ids(self):
        """Meter ids, in the order the meters were added."""
        return list(self._meters.keys())

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
        """Predict ``period`` data for one meter. Billing predictions ride the
        daily substrate (``aggregation=None``): a billing meter's ``MONTH``
        loadshape is therefore a daily-mean per calendar month, not a monthly
        total. That reshaping only touches selection features (loadshapes),
        never a corrected savings number."""
        if period == "baseline":
            data = rec.baseline_data
        else:
            data = rec.reporting_data

        if data is None:
            raise ValueError(f"Meter {rec.id}: no {period} data attached.")

        if self.model_frequency == "billing":
            pred = rec.model._predict_data(data, aggregation=None)
        else:
            pred = rec.model._predict_data(data)

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

    def _ensure_pred(self, period, ids=None):
        """Per-meter prediction frames for ``ids`` (default: all meters),
        predicting and caching any meter not yet cached."""
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

    def _materialize_observed_unc(self, mid, index):
        unc = self._meters[mid].observed_unc

        if unc is None:
            return np.zeros(len(index), dtype=float)

        if isinstance(unc, pd.Series):
            if not index.isin(unc.index).all():
                raise ValueError(
                    f"observed_unc series for meter {mid} does not cover the requested "
                    f"window {index[0]} to {index[-1]}."
                )

            return unc.reindex(index).to_numpy(dtype=float)

        return np.full(len(index), float(unc))

    def predictions(self, period):
        """Long frame ``[id, datetime, observed, modeled, modeled_unc,
        observed_unc]`` at treatment scale. Datetimes are tz-aware.

        Args:
            period: ``"baseline"`` or ``"reporting"``.

        Returns:
            One row per meter and timestep.

        Raises:
            ValueError: ``period`` is neither ``"baseline"`` nor ``"reporting"``.
        """
        preds = self._ensure_pred(period)
        frames = []

        for mid, frame in preds.items():
            long = frame.reset_index()
            long.insert(0, "id", mid)
            long["observed_unc"] = self._materialize_observed_unc(mid, frame.index)
            frames.append(long)

        result = pd.concat(frames, ignore_index=True)
        result = result[["id", "datetime", "observed", "modeled", "modeled_unc", "observed_unc"]]

        return result

    def _prediction_matrices(self, period, ids=None):
        """``(index, ids, observed, modeled, modeled_unc, observed_unc)`` where
        the arrays are float32 ``[T, M]`` written per meter into preallocated
        buffers. ``ids`` optionally restricts the columns to a meter subset (in
        the given order). ``observed_unc`` is None when no included meter
        carries observed uncertainty."""
        preds = self._ensure_pred(period, ids)
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
                values = self._materialize_observed_unc(mid, frame.index)
                observed_unc[rows, col] = values.astype(np.float32)

        return index, ids, observed, modeled, modeled_unc, observed_unc

    # -- loadshape data ------------------------------------------------------

    def _short_baseline_months(self):
        """Baseline month counts, as ``{id: months}``, of the meters with fewer
        than 12 baseline months, which cannot support a ``MONTH`` loadshape.
        Reads only; the population is unchanged."""
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

        preds = self._ensure_pred("baseline", ids)

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

    def loadshape_data(self, basis, data_settings=None, exclude=()):
        """Build a ``common.Data`` from baseline predictions. Datetimes are made
        tz-naive (local clock) so ``Data`` bins on local time. ``data_settings``
        supplies the aggregation, time period, interpolation, minimum data
        fraction, season and weekday/weekend definitions; the loadshape type is
        ``basis``. This population owns no pool trim, so ``Data``'s own trim is
        disabled here.

        Args:
            basis: Loadshape type: ``"observed"``, ``"modeled"`` or ``"error"``.
            data_settings: Optional ``Data_Settings`` supplying the fields above; the
                defaults suit the population's model frequency.
            exclude: Meter ids left out of the loadshapes, compared as strings. The
                population itself is unchanged.

        Returns:
            The ``common.Data`` built from the baseline predictions.

        Raises:
            ValueError: A meter has no baseline data attached.
        """
        missing = self._ids_without_baseline()
        if missing:
            raise ValueError(f"Loadshapes require baseline data; none attached for {missing}.")

        basis = _const.LoadshapeType(basis)

        if data_settings is None:
            caller_settings = {
                "agg_type": _const.AggType.MEAN,
                "time_period": _DEFAULT_TIME_PERIOD[self.model_frequency],
            }
        else:
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

        settings = Data_Settings(
            **caller_settings,
            loadshape_type=basis,
            max_pool_size=len(ids),
            seed=_LOADSHAPE_SEED,
        )
        data = Data(time_series_df=time_series, settings=settings)

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
        line, for streamed pool-scale persistence."""
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

            if _payload_has_is_electricity_data(meter["model"]):
                model = model_cls.from_json(meter["model"])
            else:
                model = model_cls.from_json(
                    meter["model"], is_electricity_data=header["is_electricity_data"]
                )

            record = _MeterRecord(
                id=mid,
                model=model,
                observed_unc=_deserialize_unc(meter["observed_unc"], tz),
            )

            baseline_df = baseline.get(mid)
            if baseline_df is not None:
                _require_frame(mid, "baseline data", baseline_df)
                record.baseline_data = model._baseline_data(baseline_df)

            reporting_df = reporting.get(mid)
            if reporting_df is not None:
                _require_frame(mid, "reporting data", reporting_df)
                record.reporting_data = model._reporting_data(reporting_df)

            records[mid] = record

        population = cls._restored(records, model_frequency, features, header)
        population._exclusions = exclusions.read_table_json(header["exclusions"])

        return population

    @classmethod
    def _restored(cls, records, model_frequency, features, header):
        population = cls(records, model_frequency, features=features)

        return population

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
    ``from_fit_models``; ``ids`` keeps the input order. A
    ``from_json``/``from_ndjson`` reload never trims: it restores the saved
    meters, ledger, ``max_pool_size`` and ``seed``.
    """

    _role = "pool"

    def __init__(
        self,
        meters,
        model_frequency,
        features=None,
        max_pool_size=_DEFAULT_POOL_SIZE,
        seed=0,
    ):
        super().__init__(meters, model_frequency, features=features)
        self.max_pool_size = _resolve_max_pool_size(max_pool_size)
        self.seed = seed

        _, trimmed = _pool_trim(self._meters, self.max_pool_size, seed)
        for mid in trimmed:
            del self._meters[mid]

        self._exclusions = _record_pool_trim(self._exclusions, trimmed)

    @classmethod
    def from_data(
        cls,
        baseline,
        model_frequency,
        *,
        max_pool_size=_DEFAULT_POOL_SIZE,
        seed=0,
        reporting=None,
        **kwargs,
    ):
        """``MeterPopulation.from_data`` fitting only the meters the
        ``max_pool_size`` trim keeps; the others are ledgered as ``pool_trim``
        without a fit. A kept meter that fails to fit is not replaced, so the
        pool can end below the cap."""
        max_pool_size = _resolve_max_pool_size(max_pool_size)
        _, trimmed = _pool_trim(baseline, max_pool_size, seed)
        trimmed_set = set(trimmed)
        kept_baseline = {
            mid: bdata for mid, bdata in baseline.items() if str(mid) not in trimmed_set
        }
        # reporting for an id outside ``baseline`` still reaches the parent's KeyError
        kept_reporting = {
            mid: data for mid, data in (reporting or {}).items() if str(mid) not in trimmed_set
        }

        pool = super().from_data(
            kept_baseline,
            model_frequency,
            reporting=kept_reporting,
            max_pool_size=max_pool_size,
            seed=seed,
            **kwargs,
        )
        pool._exclusions = _record_pool_trim(pool._exclusions, trimmed)

        return pool

    def _header(self):
        header = super()._header()
        header["max_pool_size"] = self.max_pool_size
        header["seed"] = self.seed

        return header

    @classmethod
    def _restored(cls, records, model_frequency, features, header):
        # built uncapped so a reload keeps every saved meter; the saved cap is
        # restored afterwards as the record of the original trim
        pool = cls(
            records,
            model_frequency,
            features=features,
            max_pool_size=None,
            seed=header["seed"],
        )
        pool.max_pool_size = header["max_pool_size"]

        return pool
