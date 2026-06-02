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

"""Top-level adaptive P-spline model with EM-based regime discovery."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)
from opendsm.eemeter.common.warnings import EEMeterWarning
from opendsm.common.metrics import BaselineMetrics, BaselineMetricsFromDict

from opendsm.eemeter.models.daily_pspline.data import (
    DailyBaselineData,
    DailyReportingData,
)
from opendsm.eemeter.models.daily.data import (
    DailyBaselineData as _DailyBaselineDataAlt,
    DailyReportingData as _DailyReportingDataAlt,
)
from opendsm.eemeter.models.daily_pspline.spline import PSpline

from .settings import DailyAdaptivePSplineSettings
from .regime_discovery import discover_regimes
from .classifier import RegimeClassifier, build_features


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _baseload(spl: PSpline) -> float:
    """Predicted value at the midpoint between balance points."""
    mid = 0.5 * (spl.bp[0] + spl.bp[1])
    return float(spl.predict(np.array([mid]))[0])


def _model_type(spl: PSpline) -> str:
    """Infer functional form from effective balance points."""
    ebp = spl.eff_bp
    bp = spl.bp
    rtol = 1e-6
    has_hdd = bp[0] - ebp[0] > rtol
    has_cdd = ebp[1] - bp[1] > rtol
    if has_hdd and has_cdd:
        return "hdd_tidd_cdd"
    elif has_hdd:
        return "hdd_tidd"
    elif has_cdd:
        return "tidd_cdd"
    return "tidd"


# ------------------------------------------------------------------
# Model class
# ------------------------------------------------------------------

class DailyAdaptivePSplineModel:
    """Fits multiple PSplines via EM regime discovery.

    Discovers the optimal number of temperature-energy regimes from
    training data, fits a PSpline per regime, and trains a lightweight
    classifier to assign new days to regimes at prediction time.

    Attributes
    ----------
    settings : DailyAdaptivePSplineSettings
    model : dict[int, PSpline]
        Regime index -> fitted spline, set after fit().
    classifier : RegimeClassifier
        Assigns new days to regimes based on (date, temperature).
    best_k : int
        Number of discovered regimes.
    baseline_metrics : BaselineMetrics
    is_fit : bool
    """

    _baseline_data_type = (DailyBaselineData, _DailyBaselineDataAlt)
    _reporting_data_type = (DailyReportingData, _DailyReportingDataAlt)

    def __init__(
        self,
        settings: DailyAdaptivePSplineSettings | dict | None = None,
        verbose: bool = False,
    ):
        if isinstance(settings, dict):
            settings = DailyAdaptivePSplineSettings(**settings)
        self.settings = settings or DailyAdaptivePSplineSettings()
        self.verbose = verbose
        self.is_fit = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        baseline_data: DailyBaselineData,
        ignore_disqualification: bool = False,
    ) -> DailyAdaptivePSplineModel:
        if not isinstance(baseline_data, self._baseline_data_type):
            names = " or ".join(t.__name__ for t in self._baseline_data_type)
            raise TypeError(f"baseline_data must be a {names} object")
        baseline_data.log_warnings()
        if baseline_data.disqualification and not ignore_disqualification:
            raise DataSufficiencyError("Can't fit model on disqualified baseline data")

        self.baseline_timezone = baseline_data.tz
        self.warnings = baseline_data.warnings
        self.disqualification = baseline_data.disqualification

        df = baseline_data.df
        if df is None:
            return self

        self._fit(df)
        return self

    def predict(
        self,
        reporting_data: DailyBaselineData | DailyReportingData,
        ignore_disqualification: bool = False,
    ) -> pd.DataFrame:
        if not self.is_fit:
            raise RuntimeError("Model must be fit before predictions can be made.")
        if self.disqualification and not ignore_disqualification:
            raise DisqualifiedModelError(
                "Attempting to predict using disqualified model without setting "
                "ignore_disqualification=True"
            )
        if str(self.baseline_timezone) != str(reporting_data.tz):
            raise ValueError(
                "Reporting data must use the same timezone the model was fit on."
            )
        all_types = self._baseline_data_type + self._reporting_data_type
        if not isinstance(reporting_data, all_types):
            names = " or ".join(t.__name__ for t in all_types)
            raise TypeError(f"reporting_data must be a {names} object")

        df = reporting_data.df
        if df is None:
            return pd.DataFrame()
        return self._predict(df)

    @property
    def eff_bp(self) -> dict[int, np.ndarray]:
        """Effective balance points per regime."""
        return {k: spl.eff_bp for k, spl in self.model.items()}

    @property
    def model_type(self) -> dict[int, str]:
        """Model type per regime."""
        return {k: _model_type(spl) for k, spl in self.model.items()}

    def plot(self, data, ax=None, **kwargs):
        try:
            from opendsm.eemeter.models.daily.plot import plot
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")
        df = data.df
        if df is None:
            return
        return plot(self, self._predict(df), ax=ax, **kwargs)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "submodels": {str(k): spl.to_dict() for k, spl in self.model.items()},
            "classifier": self.classifier.to_dict(),
            "best_k": self.best_k,
            "settings": self.settings.model_dump(),
            "info": {
                "baseline_timezone": str(self.baseline_timezone),
                "metrics": self.baseline_metrics.model_dump(),
                "disqualification": [dq.json() for dq in self.disqualification],
                "warnings": [w.json() for w in self.warnings],
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> DailyAdaptivePSplineModel:
        m = cls(settings=data.get("settings"))
        m.model = {int(k): PSpline.from_dict(v) for k, v in data["submodels"].items()}
        m.classifier = RegimeClassifier.from_dict(data["classifier"])
        m.best_k = data["best_k"]

        info = data.get("info", {})
        m.baseline_timezone = info.get("baseline_timezone")
        m.baseline_metrics = BaselineMetricsFromDict(info.get("metrics", {}))

        def _deser(lst):
            return [
                EEMeterWarning(
                    qualified_name=w.get("qualified_name"),
                    description=w.get("description"),
                    data=w.get("data"),
                )
                for w in (lst or [])
            ]

        m.disqualification = _deser(info.get("disqualification"))
        m.warnings = _deser(info.get("warnings"))
        m.is_fit = True
        return m

    @classmethod
    def from_json(cls, json_str: str) -> DailyAdaptivePSplineModel:
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> None:
        result = discover_regimes(df, self.settings)

        self.model = result.splines
        self.classifier = result.classifier
        self.best_k = result.k

        if self.verbose:
            print(f"Selected K={result.k} regimes (BIC={result.bic:.1f})")
            for k, spl in self.model.items():
                n_days = np.sum(result.assignments == k)
                print(f"  regime {k}: {n_days} days, type={_model_type(spl)}")

        # Aggregate metrics across all regimes
        obs_all, pred_all = [], []
        for k, spl in self.model.items():
            obs_all.append(spl.y)
            pred_all.append(spl.predict(spl.x))
        obs = np.concatenate(obs_all)
        pred = np.concatenate(pred_all)
        num_coeffs = sum(len(spl.c) for spl in self.model.values())

        self.baseline_metrics = BaselineMetrics(
            df=pd.DataFrame({"observed": obs, "predicted": pred}),
            num_model_params=num_coeffs,
        )
        self.is_fit = True

    def _predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        df_out["predicted"] = np.nan
        df_out["predicted_unc_lower"] = np.nan
        df_out["predicted_unc_upper"] = np.nan
        df_out["heating_load"] = 0.0
        df_out["cooling_load"] = 0.0
        df_out["model_split"] = None
        df_out["model_type"] = None

        # Assign each day to a regime via the classifier
        X = build_features(df_out, self.settings)
        regime_labels = self.classifier.predict(X)

        for k, spl in self.model.items():
            mask = regime_labels == k
            if not np.any(mask):
                continue

            seg = df_out.loc[mask]
            T = seg["temperature"].values
            pred = spl.predict(T)
            df_out.loc[mask, "predicted"] = pred

            if spl.uncertainty is not None:
                lower, upper = spl.prediction_uncertainty(T, predicted=pred)
                df_out.loc[mask, "predicted_unc_lower"] = lower
                df_out.loc[mask, "predicted_unc_upper"] = upper

            baseload = _baseload(spl)
            heating = np.where(T < spl.bp[0], pred - baseload, 0.0)
            cooling = np.where(T > spl.bp[1], pred - baseload, 0.0)
            df_out.loc[mask, "heating_load"] = np.maximum(heating, 0.0)
            df_out.loc[mask, "cooling_load"] = np.maximum(cooling, 0.0)

            df_out.loc[mask, "model_split"] = f"regime_{k}"
            df_out.loc[mask, "model_type"] = _model_type(spl)

        return df_out
