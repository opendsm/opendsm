"""Top-level P-spline daily model with season/weekday-weekend split selection."""

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
from opendsm.eemeter.models.daily.utilities.selection_criteria import selection_criteria

from opendsm.eemeter.models.daily_pspline.data import DailyBaselineData, DailyReportingData
from opendsm.eemeter.models.daily.data import (
    DailyBaselineData as _DailyBaselineDataAlt,
    DailyReportingData as _DailyReportingDataAlt,
)
from opendsm.eemeter.models.daily_pspline.settings import DailyPSplineSettings
from opendsm.eemeter.models.daily_pspline.spline import PSpline
from opendsm.eemeter.models.daily_pspline.fitting import fit_segment
from opendsm.eemeter.models.daily_pspline.split_selection import SplitSelector, segment


# ------------------------------------------------------------------
# Helpers (used by _predict)
# ------------------------------------------------------------------

def _baseload(spl: PSpline) -> float:
    """Baseload: predicted value at midpoint between balance points."""
    mid = 0.5 * (spl.bp[0] + spl.bp[1])
    return float(spl.predict(np.array([mid]))[0])


def _model_type(spl: PSpline) -> str:
    """Infer functional form from effective balance points.

    Returns 'hdd_tidd_cdd', 'hdd_tidd', 'tidd_cdd', or 'tidd'.
    """
    ebp = spl.eff_bp
    bp = spl.bp
    rtol = 1e-6

    # eff_bp moves away from bp when a meaningful slope is found.
    # If eff_bp stayed at bp, the zone is flat (no effective zone).
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

class DailyPSplineModel:
    """Fits a penalized B-spline energy model per season/day segment.

    Uses season and weekday/weekend split-selection logic shared with
    DailyModel, but fits a monotone P-spline per segment.

    Attributes
    ----------
    settings : DailyPSplineSettings
    model : dict[str, PSpline]
        Component key → fitted spline, set after fit().
    best_combination : str
        Winning split, e.g. 'wd-su__we-su__fw-sh_wi'.
    baseline_metrics : BaselineMetrics
    is_fit : bool
    """

    _baseline_data_type = (DailyBaselineData, _DailyBaselineDataAlt)
    _reporting_data_type = (DailyReportingData, _DailyReportingDataAlt)

    def __init__(
        self,
        settings: DailyPSplineSettings | dict | None = None,
        verbose: bool = False,
    ):
        if isinstance(settings, dict):
            settings = DailyPSplineSettings(**settings)
        self.settings = settings or DailyPSplineSettings()
        self.verbose = verbose
        self.is_fit = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        baseline_data: DailyBaselineData,
        ignore_disqualification: bool = False,
    ) -> DailyPSplineModel:
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

        self._fit(df, time_order=getattr(baseline_data, '_time_order', None))
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
    def eff_bp(self) -> dict[str, np.ndarray]:
        """Effective balance points per model split."""
        return {key: spl.eff_bp for key, spl in self.model.items()}

    @property
    def model_type(self) -> dict[str, str]:
        """Model type per model split (e.g. 'tidd_cdd', 'hdd_tidd_cdd')."""
        return {key: _model_type(spl) for key, spl in self.model.items()}

    def plot(self, data, ax=None, **kwargs):
        try:
            from opendsm.eemeter.models.daily.plot import plot
        except ImportError:  # pragma: no cover
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
            "submodels": {k: spl.to_dict() for k, spl in self.model.items()},
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
    def from_dict(cls, data: dict) -> DailyPSplineModel:
        m = cls(settings=data.get("settings"))
        m.model = {k: PSpline.from_dict(v) for k, v in data["submodels"].items()}
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
        m.best_combination = "__".join(m.model.keys())
        m.is_fit = True
        return m

    @classmethod
    def from_json(cls, json_str: str) -> DailyPSplineModel:
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame, time_order=None) -> None:
        df = df.copy()
        for col in ("season", "weekday_weekend"):
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].astype("category")

        selector = SplitSelector(df, self.settings.split_selection)
        self._selector = selector

        combinations = selector.combinations()
        components = selector.required_components(combinations)

        segments: dict[str, tuple] = {}
        for comp in components:
            seg = selector.segment(comp)
            if len(seg) >= 2:
                # Time-order index within this segment: argsort of original
                # positions gives the permutation that restores time order.
                if time_order is not None:
                    seg_positions = time_order.get_indexer(seg.index)
                    time_sort = np.argsort(seg_positions)
                else:
                    time_sort = None
                segments[comp] = (seg["temperature"].values, seg["observed"].values, time_sort)

        fit_components: dict[str, PSpline] = {}
        for comp in components:
            if comp not in segments:
                if self.verbose:
                    print(f"Skipping {comp}: too few data points")
                continue
            x, y, time_sort = segments[comp]
            fitted = fit_segment(x, y, self.settings, time_sort=time_sort)
            fit_components[comp] = fitted
            if self.verbose:
                rmse = np.sqrt(np.mean((y - fitted.predict(x)) ** 2))
                print(f"{comp}: n={len(x)}, RMSE={rmse:.4f}")

        fitted_keys = set(fit_components.keys())
        combinations = [
            c for c in combinations
            if all(p in fitted_keys for p in c.split("__"))
        ]

        self.model, obs, pred = self._select_best(combinations, fit_components)
        self.best_combination = "__".join(self.model.keys())

        num_coeffs = sum(len(spl.c) for spl in self.model.values())
        self.baseline_metrics = BaselineMetrics(
            df=pd.DataFrame({"observed": obs, "predicted": pred}),
            num_model_params=num_coeffs,
        )
        self.is_fit = True

    def _predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        df_out["predicted"] = np.nan
        df_out["predicted_unc"] = np.nan
        df_out["heating_load"] = 0.0
        df_out["cooling_load"] = 0.0
        df_out["model_split"] = None
        df_out["model_type"] = None

        for key, spl in self.model.items():
            seg = segment(key, df_out)
            T = seg["temperature"].values
            pred = spl.predict(T)
            df_out.loc[seg.index, "predicted"] = pred

            if spl.uncertainty is not None:
                df_out.loc[seg.index, "predicted_unc"] = spl.prediction_uncertainty(T)

            # Load decomposition uses fitted BPs
            baseload = _baseload(spl)
            heating = np.where(T < spl.bp[0], pred - baseload, 0.0)
            cooling = np.where(T > spl.bp[1], pred - baseload, 0.0)
            df_out.loc[seg.index, "heating_load"] = np.maximum(heating, 0.0)
            df_out.loc[seg.index, "cooling_load"] = np.maximum(cooling, 0.0)

            # Model type uses effective balance points
            df_out.loc[seg.index, "model_split"] = key
            df_out.loc[seg.index, "model_type"] = _model_type(spl)

        return df_out

    def _select_best(
        self,
        combinations: list[str],
        fit_components: dict[str, PSpline],
    ) -> tuple[dict[str, PSpline], np.ndarray, np.ndarray]:
        ss = self.settings.split_selection

        base = fit_components["fw-su_sh_wi"]
        obs_base = base.y
        pred_base = base.predict(base.x)
        wRMSE_base = float(np.sqrt(np.mean((obs_base - pred_base) ** 2)))
        N = len(obs_base)

        best_combo, best_sc = None, np.inf
        best_obs, best_pred = obs_base, pred_base

        for combo in combinations:
            parts = combo.split("__")
            if combo == "fw-su_sh_wi":
                obs, pred = obs_base, pred_base
            else:
                obs = np.hstack([fit_components[c].y for c in parts])
                pred = np.hstack([fit_components[c].predict(fit_components[c].x) for c in parts])

            wRMSE = float(np.sqrt(np.mean((obs - pred) ** 2)))
            TSS = sum(
                float(np.sum((fit_components[c].y - np.mean(fit_components[c].y)) ** 2))
                for c in parts
            )
            sc = selection_criteria(
                wRMSE / wRMSE_base, TSS, N, len(parts),
                ss.criteria, ss.penalty_multiplier, ss.penalty_power,
            )
            if self.verbose:
                print(f"{combo:>50s}  {sc:>10.4f}")
            if sc < best_sc:
                best_combo, best_sc = combo, sc
                best_obs, best_pred = obs, pred

        if best_combo is None:
            raise RuntimeError("No valid combinations found.")

        return {c: fit_components[c] for c in best_combo.split("__")}, best_obs, best_pred
