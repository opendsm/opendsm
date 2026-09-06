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



SCHEMA_VERSION = 1

# Above this many models, JSON serialization is refused in favor of the
# streamed one-record-per-line NDJSON path (pool-scale persistence).
_MAX_JSON_MODELS = 2000

_LOADSHAPE_SEED = 0

_MODEL_TYPES = {
    "hourly": HourlyModel,
    "daily": DailyModel,
    "billing": BillingModel,
}

_BASELINE_TYPES = {
    "hourly": HourlyBaselineData,
    "daily": DailyBaselineData,
    "billing": BillingBaselineData,
}

_REPORTING_TYPES = {
    "hourly": HourlyReportingData,
    "daily": DailyReportingData,
    "billing": BillingReportingData,
}

# BillingModel subclasses DailyModel and their payloads are structurally
# identical, so dispatch is by exact type with billing checked first.
_GRANULARITY_ORDER = ("billing", "daily", "hourly")

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


def _granularity_of_model(model) -> str:
    for name in _GRANULARITY_ORDER:
        if type(model) is _MODEL_TYPES[name]:
            return name

    raise TypeError(
        f"Unsupported model type {type(model).__name__}; "
        f"expected one of {[cls.__name__ for cls in _MODEL_TYPES.values()]}"
    )


def _model_from_payload(model_cls, payload):
    if isinstance(payload, str):
        return model_cls.from_json(payload)

    if isinstance(payload, dict):
        return model_cls.from_dict(payload)

    raise TypeError(f"Model payload must be a dict or JSON string, got {type(payload).__name__}")


def _payload_model_type(payload):
    """The ``model_type`` granularity tag on a dict/JSON model payload, or
    ``None`` for a legacy payload written before the tag existed."""
    if isinstance(payload, str):
        payload = json.loads(payload)

    tag = payload.get("model_type")

    return tag


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
    """A set of per-meter energy models sharing one granularity, timezone, and
    fuel type, plus their baseline (and optionally reporting) data.

    Uncertainty columns (``modeled_unc``) are the models' t-scaled ASHRAE-style
    bands, not calibrated 1-sigma intervals.
    """

    _role = "population"

    def __init__(
        self,
        meters,
        granularity,
        features=None,
        baseline_window=None,
        reporting_window=None,
    ):
        if not meters:
            raise ValueError("A population must contain at least one meter.")

        if granularity not in _MODEL_TYPES:
            raise ValueError(
                f"granularity must be one of {list(_MODEL_TYPES)}, got {granularity!r}"
            )

        self.granularity = granularity
        self._meters = {str(k): v for k, v in meters.items()}
        self.features = self._validate_features(features)
        self._exclusions = exclusions.empty_ledger()
        self._baseline_pred = {}
        self._reporting_pred = {}

        self.tz, self.is_electricity_data = self._validate_meters()
        self._baseline_window_arg = _coerce_window(baseline_window, self.tz)
        self._reporting_window_arg = _coerce_window(reporting_window, self.tz)
        self.baseline_window = self._resolve_window("baseline")
        self.reporting_window = self._resolve_window("reporting")

    # -- construction --------------------------------------------------------

    @classmethod
    def from_fit_models(cls, meters, granularity=None, features=None, **kwargs):
        """Build a population from per-meter fitted models (instances or model
        payloads).

        Args:
            meters: ``{id: {model, baseline_data, reporting_data=None,
                observed_unc=None}}``. ``model`` may be a fitted instance, a
                ``to_dict()`` payload, or a ``to_json()`` string.
            granularity: Inferred from each model — from a fitted instance's type
                or a payload's ``model_type`` tag — so it is optional for tagged
                inputs. When supplied it is validated against the inferred value
                (mismatch raises ``TypeError``). A legacy payload written before
                the tag existed carries no ``model_type`` and still requires it.
        """
        records = {}
        resolved = granularity

        for mid, entry in meters.items():
            model_obj = entry["model"]

            if isinstance(model_obj, (dict, str)):
                inferred = _payload_model_type(model_obj)

                if inferred is None:
                    if granularity is None:
                        raise ValueError(
                            "granularity is required when models are passed as dict or JSON "
                            "payloads without a model_type tag."
                        )
                    inferred = granularity

                elif inferred not in _MODEL_TYPES:
                    raise ValueError(
                        f"Meter {mid}: unknown model_type {inferred!r} in payload."
                    )

                elif granularity is not None and inferred != granularity:
                    raise TypeError(
                        f"Meter {mid}: payload model_type {inferred!r} does not match "
                        f"requested {granularity!r}."
                    )

                model = _model_from_payload(_MODEL_TYPES[inferred], model_obj)

            else:
                inferred = _granularity_of_model(model_obj)
                if granularity is not None and inferred != granularity:
                    raise TypeError(
                        f"Meter {mid}: model granularity {inferred!r} does not match "
                        f"requested {granularity!r}."
                    )
                model = model_obj

            if resolved is None:
                resolved = inferred

            records[str(mid)] = _MeterRecord(
                id=mid,
                model=model,
                baseline_data=entry.get("baseline_data"),
                reporting_data=entry.get("reporting_data"),
                observed_unc=entry.get("observed_unc"),
            )

        if resolved is None:
            raise ValueError("Could not resolve granularity; pass it explicitly.")

        return cls(records, resolved, features=features, **kwargs)

    @classmethod
    def from_data(
        cls,
        baseline,
        model_type,
        settings=None,
        ignore_disqualification=False,
        features=None,
        **kwargs,
    ):
        """Fit one model per meter serially (never in parallel).

        A meter whose fit raises a data-sufficiency or model-disqualification
        error is excluded and recorded on the population's exclusions ledger
        while the loop continues; raises only when every meter fails.

        Args:
            baseline: ``{id: BaselineData}`` for the requested granularity.
            model_type: ``"hourly"``, ``"daily"``, or ``"billing"``.
        """
        if model_type not in _MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {list(_MODEL_TYPES)}, got {model_type!r}"
            )

        baseline_cls = _BASELINE_TYPES[model_type]
        model_cls = _MODEL_TYPES[model_type]
        records = {}
        dropped = exclusions.empty_ledger()

        for mid, bdata in baseline.items():
            if not isinstance(bdata, baseline_cls):
                raise TypeError(f"Meter {mid}: baseline data must be {baseline_cls.__name__}.")

            model = model_cls(settings=settings)

            try:
                model.fit(bdata, ignore_disqualification=ignore_disqualification)
            except DataSufficiencyError as exc:
                detail = exclusions.format_warnings(bdata.disqualification) or str(exc)
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

            records[str(mid)] = _MeterRecord(id=mid, model=model, baseline_data=bdata)

        if not records:
            summary = "; ".join(f"{row.id}: {row.reason}" for row in dropped.itertuples())
            raise ValueError(f"All meters failed to fit: {summary}")

        population = cls(records, model_type, features=features, **kwargs)
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
        baseline_cls = _BASELINE_TYPES[self.granularity]
        reporting_cls = _REPORTING_TYPES[self.granularity]

        timezones = set()
        fuels = set()
        ghi_flags = set()
        disqualified = []

        for mid, rec in self._meters.items():
            if _granularity_of_model(rec.model) != self.granularity:
                raise TypeError(
                    f"Meter {mid}: model type does not match population granularity "
                    f"{self.granularity!r}."
                )

            if rec.baseline_data is None:
                raise ValueError(f"Meter {mid}: baseline_data is required.")

            if not isinstance(rec.baseline_data, baseline_cls):
                raise TypeError(f"Meter {mid}: baseline_data must be {baseline_cls.__name__}.")

            if rec.reporting_data is not None and not isinstance(rec.reporting_data, reporting_cls):
                raise TypeError(f"Meter {mid}: reporting_data must be {reporting_cls.__name__}.")

            if str(rec.model.baseline_timezone) != str(rec.baseline_data.tz):
                raise ValueError(
                    f"Meter {mid}: model timezone {rec.model.baseline_timezone} does not match "
                    f"baseline data timezone {rec.baseline_data.tz}."
                )

            timezones.add(str(rec.model.baseline_timezone))
            fuels.add(bool(rec.baseline_data.is_electricity_data))
            ghi_flags.add("ghi" in rec.baseline_data.df.columns)

            if rec.model.disqualification or rec.baseline_data.disqualification:
                disqualified.append(mid)

        if len(timezones) != 1:
            raise ValueError(f"All meters must share one timezone; found {sorted(timezones)}.")

        if len(fuels) != 1:
            raise ValueError(
                f"All meters must share one is_electricity_data flag; found {sorted(fuels)}."
            )

        if len(ghi_flags) > 1:
            raise ValueError("Population mixes solar (ghi present) and non-solar meters.")

        if disqualified:
            warnings.warn(
                f"Population includes disqualified meters or baselines: {disqualified}.",
                stacklevel=2,
            )

        tz = timezones.pop()
        is_electricity_data = fuels.pop()

        return tz, is_electricity_data

    def _compute_window(self, period):
        """The attached-data span for ``period`` as ``(min start, max end)`` over
        the meters that carry that data, or ``None`` when none do. Meter spans may
        differ; the group window is their union."""
        starts = []
        ends = []

        for rec in self._meters.values():
            if period == "baseline":
                data = rec.baseline_data
            else:
                data = rec.reporting_data

            if data is None:
                continue

            index = data.df.index
            starts.append(index.min())
            ends.append(index.max())

        if not starts:
            return None

        window = (min(starts), max(ends))

        return window

    def _resolve_window(self, period):
        """The group window for ``period``: an explicit override when one was
        supplied and the data is attached, else the attached-data span. ``None``
        when no data is attached, so ``reporting_window`` stays the
        reporting-attached sentinel even under an explicit override."""
        if period == "baseline":
            override = self._baseline_window_arg
        elif period == "reporting":
            override = self._reporting_window_arg
        else:
            raise ValueError(f"period must be 'baseline' or 'reporting', got {period!r}.")

        span = self._compute_window(period)

        if span is None:
            return None

        if override is not None:
            return override

        return span

    # -- accessors -----------------------------------------------------------

    @property
    def ids(self):
        return list(self._meters.keys())

    @property
    def role(self):
        return self._role

    @property
    def exclusions(self):
        """Ledger of meters dropped at the population stage, in the shared
        ``[id, stage, origin, reason, detail]`` schema."""
        return self._exclusions

    def __len__(self):
        return len(self._meters)

    # -- reporting data ------------------------------------------------------

    def add_reporting_data(self, reporting):
        """Attach or replace per-meter reporting data (caller passes cumulative
        data) and invalidate the reporting prediction cache."""
        reporting_cls = _REPORTING_TYPES[self.granularity]

        for rid, data in reporting.items():
            mid = str(rid)
            if mid not in self._meters:
                raise KeyError(f"Meter {mid} is not part of this population.")

            if not isinstance(data, reporting_cls):
                raise TypeError(f"Meter {mid}: reporting data must be {reporting_cls.__name__}.")

            if str(data.tz) != self.tz:
                raise ValueError(
                    f"Meter {mid}: reporting timezone {data.tz} does not match population "
                    f"timezone {self.tz}."
                )

            self._meters[mid].reporting_data = data

        self._reporting_pred = {}
        self.reporting_window = self._resolve_window("reporting")

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

        if self.granularity == "billing":
            pred = rec.model.predict(data, aggregation=None)
        else:
            pred = rec.model.predict(data)

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
        observed_unc]`` at treatment scale. Datetimes are tz-aware."""
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

    def _require_full_year_baseline(self):
        """Meters with fewer than 12 baseline months can't support a ``MONTH``
        loadshape. Pool meters are excluded and recorded on the ledger; treatment
        meters are too, except that a population would raise rather than
        silently exclude every treatment meter."""
        offenders = []

        for mid, rec in self._meters.items():
            naive_index = rec.baseline_data.df.index.tz_localize(None)
            months = naive_index.to_period("M").nunique()
            if months < 12:
                offenders.append((mid, months))

        if not offenders:
            return

        if self._role == "treatment" and len(offenders) == len(self._meters):
            summary = ", ".join(f"{mid}: {months} months" for mid, months in offenders)
            raise ValueError(
                "Billing month loadshapes require at least 12 baseline months per meter; "
                f"all treatment meters excluded ({summary})."
            )

        offender_ids = [mid for mid, _ in offenders]
        for mid in offender_ids:
            del self._meters[mid]

        self._exclusions = exclusions.append(
            self._exclusions,
            offender_ids,
            "population",
            "baseline_data",
            "fewer than 12 baseline months; cannot support a MONTH loadshape",
            detail="; ".join(f"{mid}: {months} months" for mid, months in offenders),
        )

        self._baseline_pred = {}
        self._reporting_pred = {}

    def _loadshape_frames(self, basis):
        frames = []

        if basis == _const.LoadshapeType.OBSERVED:
            for mid, rec in self._meters.items():
                observed = rec.baseline_data.df["observed"]
                frame = pd.DataFrame(
                    {
                        "id": mid,
                        "datetime": observed.index.tz_localize(None),
                        "observed": observed.to_numpy(),
                    }
                )
                frames.append(frame)

            return frames

        preds = self._ensure_pred("baseline")

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

    def loadshape_data(self, basis, data_settings=None):
        """Build a ``common.Data`` from baseline predictions. Datetimes are made
        tz-naive (local clock) so ``Data`` bins on local time. This population
        owns no pool trim, so ``Data``'s own trim is disabled here."""
        basis = _const.LoadshapeType(basis)

        if data_settings is None:
            agg_type = _const.AggType.MEAN
            time_period = _DEFAULT_TIME_PERIOD[self.granularity]
        else:
            agg_type = data_settings.agg_type
            time_period = data_settings.time_period

        if time_period == _const.TimePeriod.MONTH:
            self._require_full_year_baseline()

        frames = self._loadshape_frames(basis)
        time_series = pd.concat(frames, ignore_index=True)

        settings = Data_Settings(
            agg_type=agg_type,
            loadshape_type=basis,
            time_period=time_period,
            max_pool_size=len(self._meters),
            seed=_LOADSHAPE_SEED,
        )
        data = Data(time_series_df=time_series, settings=settings)

        return data

    # -- serialization -------------------------------------------------------

    def _header(self):
        header = {
            "schema_version": SCHEMA_VERSION,
            "granularity": self.granularity,
            "role": self._role,
            "tz": self.tz,
            "is_electricity_data": bool(self.is_electricity_data),
            "baseline_window": _window_to_json(self.baseline_window),
            "reporting_window": _window_to_json(self.reporting_window),
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
    def _from_payload(cls, header, meter_dicts, baseline, reporting, features, **kwargs):
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

        if baseline is None:
            raise ValueError("baseline data mapping is required to rebuild a population.")

        granularity = header["granularity"]
        model_cls = _MODEL_TYPES[granularity]
        tz = header["tz"]
        baseline = {str(k): v for k, v in baseline.items()}
        reporting = {str(k): v for k, v in (reporting or {}).items()}
        records = {}

        for meter in meter_dicts:
            mid = str(meter["id"])
            if mid not in baseline:
                raise KeyError(f"Meter {mid}: baseline data not provided for re-attach.")

            model = model_cls.from_json(meter["model"])
            observed_unc = _deserialize_unc(meter["observed_unc"], tz)
            records[mid] = _MeterRecord(
                id=mid,
                model=model,
                baseline_data=baseline[mid],
                reporting_data=reporting.get(mid),
                observed_unc=observed_unc,
            )

        population = cls(
            records,
            granularity,
            features=features,
            baseline_window=header["baseline_window"],
            reporting_window=header["reporting_window"],
            **kwargs,
        )
        # the serialized ledger comes first; rows a rebuild adds (a pool trim
        # requested on this call) follow it rather than being lost
        restored = exclusions.read_table_json(header["exclusions"])
        if population._exclusions.empty:
            population._exclusions = restored
        else:
            population._exclusions = pd.concat(
                [restored, population._exclusions], ignore_index=True
            )

        return population

    @classmethod
    def from_json(cls, s, baseline=None, reporting=None, features=None, **kwargs):
        """Rebuild from ``to_json()`` output, re-attaching caller-supplied
        baseline (and optional reporting) data by id."""
        payload = json.loads(s)
        population = cls._from_payload(
            payload["header"], payload["meters"], baseline, reporting, features, **kwargs
        )

        return population

    @classmethod
    def from_ndjson(cls, text, baseline=None, reporting=None, features=None, **kwargs):
        """Rebuild from ``to_ndjson()`` output (header line then meter lines)."""
        lines = [line for line in text.splitlines() if line.strip()]
        header = json.loads(lines[0])["header"]
        meter_dicts = [json.loads(line) for line in lines[1:]]

        return cls._from_payload(header, meter_dicts, baseline, reporting, features, **kwargs)


class TreatmentGroup(MeterPopulation):
    """A population playing the treatment role in a savings analysis."""

    _role = "treatment"


class ComparisonPool(MeterPopulation):
    """A population playing the comparison-pool role. Sole owner of the
    ``max_pool_size`` trim, recorded in its ``exclusions`` ledger."""

    _role = "pool"

    def __init__(
        self,
        meters,
        granularity,
        features=None,
        max_pool_size=None,
        seed=0,
        baseline_window=None,
        reporting_window=None,
    ):
        super().__init__(
            meters,
            granularity,
            features=features,
            baseline_window=baseline_window,
            reporting_window=reporting_window,
        )
        self.max_pool_size = max_pool_size
        self.seed = seed

        if max_pool_size is not None:
            self._trim(max_pool_size, seed)

    def _trim(self, max_pool_size, seed):
        ids = list(self._meters.keys())
        if len(ids) <= max_pool_size:
            return

        rng = np.random.RandomState(seed)
        excluded = rng.choice(ids, len(ids) - max_pool_size, replace=False)

        for mid in excluded:
            del self._meters[mid]

        self._exclusions = exclusions.append(
            self._exclusions,
            excluded,
            "population",
            "pool_trim",
            "randomly selected to reduce pool size",
        )

        self._baseline_pred = {}
        self._reporting_pred = {}
        self.baseline_window = self._resolve_window("baseline")
        self.reporting_window = self._resolve_window("reporting")
