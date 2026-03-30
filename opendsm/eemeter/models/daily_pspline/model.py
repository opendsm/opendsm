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

import itertools
import json

import numpy as np
import pandas as pd

from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    DisqualifiedModelError,
)
from opendsm.eemeter.common.warnings import EEMeterWarning
from opendsm.eemeter.models.daily_pspline.data import DailyBaselineData, DailyReportingData
from opendsm.eemeter.models.daily.data import (
    DailyBaselineData as _DailyBaselineDataAlt,
    DailyReportingData as _DailyReportingDataAlt,
)
from opendsm.eemeter.models.daily_pspline.pspline import DailyPSpline
from opendsm.common.metrics import BaselineMetrics, BaselineMetricsFromDict
from opendsm.eemeter.models.daily.utilities.selection_criteria import selection_criteria
from opendsm.eemeter.models.daily.utilities.ellipsoid_test import ellipsoid_split_filter
from opendsm.eemeter.models.daily_pspline.settings import DailyPSplineSettings


class DailyPSplineModel:
    """Fits a penalized B-spline (P-spline) energy model per season/weekday-weekend segment.

    Uses the same season and weekday/weekend split-selection logic as DailyModel,
    but fits a DailyPSpline (monotone P-spline) to the temperature-indexed baseline
    data in each segment rather than the piecewise-linear HDD/TIDD/CDD model.

    Attributes:
        settings: Model hyperparameters and split-selection configuration.
        best_combination: Component string for the winning season/day split, e.g.
            ``"wd-su__we-su__fw-sh_wi"``.  Set after ``fit()``.
        model: Mapping of component key → fitted ``DailyPSpline`` for the best
            combination.  Set after ``fit()``.
        baseline_metrics: Fit quality metrics computed on the training data.
            Set after ``fit()``.
        is_fit: ``True`` once the model has been successfully fitted.
    """

    _baseline_data_type = (DailyBaselineData, _DailyBaselineDataAlt)
    _reporting_data_type = (DailyReportingData, _DailyReportingDataAlt)

    # Short-key → label mappings used when parsing component strings.
    _SEASONS = {"su": "summer", "sh": "shoulder", "wi": "winter"}
    _DAYS    = {"fw": ["weekday", "weekend"], "wd": ["weekday"], "we": ["weekend"]}

    # Candidate season and day-type partitions (identical for all instances).
    _SEASONAL_OPTIONS = [
        ["su_sh_wi"],
        ["su", "sh_wi"],
        ["su_sh", "wi"],
        ["su_wi", "sh"],
        ["su", "sh", "wi"],
    ]
    _DAY_OPTIONS = [["wd", "we"]]

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
    ) -> "DailyPSplineModel":
        """Fit the model to baseline data.

        Args:
            baseline_data: DailyBaselineData object.
            ignore_disqualification: Whether to ignore disqualification errors.

        Returns:
            The fitted model (self).
        """
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
        """Predict energy consumption for reporting data.

        Args:
            reporting_data: DailyBaselineData or DailyReportingData object.
            ignore_disqualification: Whether to ignore model disqualification.

        Returns:
            Copy of the reporting DataFrame with three columns added:
            ``predicted`` (modelled energy), ``model_split`` (component key that
            covered each row), and ``model_type`` (``"pspline"``).
        """
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

        all_reporting_types = self._baseline_data_type + self._reporting_data_type
        if not isinstance(reporting_data, all_reporting_types):
            names = " or ".join(t.__name__ for t in all_reporting_types)
            raise TypeError(f"reporting_data must be a {names} object")

        df = reporting_data.df
        if df is None:
            return pd.DataFrame()
        
        return self._predict(df)

    def plot(
        self,
        data: DailyBaselineData | DailyReportingData,
        ax=None,
        **kwargs,
    ):
        """Plot a model fit with baseline or reporting data. Requires matplotlib to use.

        Args:
            data: The baseline or reporting data object to plot.
            ax: Optional existing matplotlib Axes to plot onto. Creates a new figure if None.
            **kwargs: Additional keyword arguments forwarded to the plot function.
        """
        try:
            from opendsm.eemeter.models.daily.plot import plot
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib is required for plotting.")

        df = data.df
        if df is None:
            return

        return plot(self, self._predict(df), ax=ax, **kwargs)

    def to_dict(self) -> dict:
        """Serialize model to a dictionary."""
        submodels = {k: spl.to_dict() for k, spl in self.model.items()}
        return {
            "submodels": submodels,
            "settings": self.settings.model_dump(),
            "info": {
                "baseline_timezone": str(self.baseline_timezone),
                "metrics": self.baseline_metrics.model_dump(),
                "disqualification": [dq.json() for dq in self.disqualification],
                "warnings": [w.json() for w in self.warnings],
            },
        }

    def to_json(self) -> str:
        """Serialize model to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "DailyPSplineModel":
        """Reconstruct a fitted model from a dictionary."""
        m = cls(settings=data.get("settings"))
        m.model = {k: DailyPSpline.from_dict(v) for k, v in data["submodels"].items()}

        info = data.get("info", {})
        m.baseline_timezone = info.get("baseline_timezone")
        m.baseline_metrics = BaselineMetricsFromDict(info.get("metrics", {}))

        def _deserialize_warnings(lst):
            return [
                EEMeterWarning(
                    qualified_name=w.get("qualified_name"),
                    description=w.get("description"),
                    data=w.get("data"),
                )
                for w in (lst or [])
            ]

        m.disqualification = _deserialize_warnings(info.get("disqualification"))
        m.warnings = _deserialize_warnings(info.get("warnings"))
        m.best_combination = "__".join(m.model.keys())
        m.is_fit = True
        return m

    @classmethod
    def from_json(cls, json_str: str) -> "DailyPSplineModel":
        """Reconstruct a fitted model from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Internal fit
    # ------------------------------------------------------------------

    def _fit(self, df: pd.DataFrame) -> "DailyPSplineModel":
        """Core fit logic, called by ``fit()`` after input validation."""
        self.df_meter = df

        # Determine valid combinations before fitting.
        self.combinations = self._combinations()

        # Needed components = union of all surviving combination parts.
        self.components = self._components(self.combinations)

        # Pre-extract segment arrays once to avoid repeated pandas boolean-index filtering.
        segments: dict[str, tuple] = {}
        for component in self.components:
            seg = self._meter_segment(component)
            if len(seg) >= 2:
                segments[component] = (seg["temperature"].values, seg["observed"].values)

        self.fit_components = self._fit_components(segments)

        # Post-filter: drop any combination whose component couldn't be fitted (< 2 pts).
        if len(self.fit_components) < len(self.components):
            fitted = set(self.fit_components.keys())
            self.combinations = [
                c for c in self.combinations
                if all(p in fitted for p in c.split("__"))
            ]

        self.model, obs, pred = self._best_combination()
        self.best_combination = "__".join(self.model.keys())

        num_coeffs = sum(len(spl.c) for spl in self.model.values())
        self.baseline_metrics = BaselineMetrics(
            df=pd.DataFrame({"observed": obs, "predicted": pred}),
            num_model_params=num_coeffs,
        )

        self.is_fit = True

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fit model to a pre-validated DataFrame."""
        df_out = df.copy()
        df_out["predicted"] = np.nan
        df_out["model_split"] = None
        df_out["model_type"] = "pspline"

        for component_key, spl in self.model.items():
            segment = self._meter_segment(component_key, df_out)
            T = segment["temperature"].values
            df_out.loc[segment.index, "predicted"] = spl.predict(T)
            df_out.loc[segment.index, "model_split"] = component_key

        return df_out

    # ------------------------------------------------------------------
    # Combination and component generation
    # ------------------------------------------------------------------

    def _combinations(self, split_min_days: int = 30) -> list[str]:
        """Return all valid season × day combination strings for the current data.

        Generates the full candidate set then trims to those that have sufficient
        data and are allowed by the current settings.
        """

        def _get_combinations() -> list[str]:
            combos: set[str] = set()
            for day_types in self._DAY_OPTIONS:
                for season_partition in self._SEASONAL_OPTIONS:
                    for choices in itertools.product(["fw", "split"], repeat=len(season_partition)):
                        fw_parts: list[str] = []
                        day_parts: dict[str, list[str]] = {dt: [] for dt in day_types}
                        for season_group, choice in zip(season_partition, choices):
                            if choice == "fw":
                                fw_parts.append(f"fw-{season_group}")
                            else:
                                for day_type in day_types:
                                    day_parts[day_type].append(f"{day_type}-{season_group}")
                        components = fw_parts + [c for dt in day_types for c in day_parts[dt]]
                        combos.add("__".join(components))

            return sorted(combos, key=lambda x: (len(x), x))

        def _trim_combinations(combo_list: list[str]) -> list[str]:
            meter = self.df_meter
            ss = self.settings.split_selection
            we_days = self._DAYS["we"]

            allow = {
                "su": ss.allow_separate_summer and (meter["season"] == "summer").sum() >= split_min_days,
                "sh": ss.allow_separate_shoulder and (meter["season"] == "shoulder").sum() >= split_min_days,
                "wi": ss.allow_separate_winter and (meter["season"] == "winter").sum() >= split_min_days,
            }
            allow_wd = ss.allow_separate_weekday_weekend

            if ss.reduce_splits_by_gaussian:
                gaussian = ellipsoid_split_filter(meter, n_std=ss.reduce_splits_num_std)
                allow = {k: allow[k] and gaussian[self._SEASONS[k]] for k in allow}
                allow_wd = allow_wd and gaussian["weekday_weekend"]

            we_counts = {
                s: ((meter["season"] == self._SEASONS[s]) & meter["weekday_weekend"].isin(we_days)).sum()
                for s in ("su", "sh", "wi")
            }
            we_min = split_min_days / 3.75

            trimmed = []
            for combo in combo_list:
                if "wd" in combo and not allow_wd:
                    continue

                valid = True
                for component in combo.split("__"):
                    season_keys = component[3:].split("_")
                    if len(season_keys) == 1 and not allow.get(season_keys[0], True):
                        valid = False
                        break
                    if sum(we_counts[s] for s in season_keys) < we_min:
                        valid = False
                        break

                if valid:
                    trimmed.append(combo)

            return trimmed

        combinations = _get_combinations()
        combinations = _trim_combinations(combinations)

        return combinations

    def _components(self, combinations: list[str]) -> list[str]:
        """Return the sorted list of component keys needed to evaluate all combinations.

        Always includes ``fw-su_sh_wi`` so the all-merged baseline RMSE can be
        computed for normalisation even when that combo is trimmed.
        """
        needed: set[str] = {"fw-su_sh_wi"}
        for combo in combinations:
            needed.update(combo.split("__"))

        return sorted(needed, key=lambda x: (len(x), x))

    # ------------------------------------------------------------------
    # Segment helpers
    # ------------------------------------------------------------------

    def _meter_segment(self, component: str, meter: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the rows of meter matching the given component key.

        Component keys have the form  ``<day_prefix>-<season1>_<season2>``
        e.g. ``fw-su_sh_wi``, ``wd-su``, ``we-sh_wi``.
        """
        if meter is None:
            meter = self.df_meter

        day_key = component[:2]          # "fw", "wd", or "we"
        season_keys = component[3:].split("_")  # ["su", "sh", "wi"] etc.

        days = self._DAYS[day_key]
        seasons = [self._SEASONS[k] for k in season_keys]

        return meter[
            meter["season"].isin(seasons) & meter["weekday_weekend"].isin(days)
        ]

    # ------------------------------------------------------------------
    # Component fitting
    # ------------------------------------------------------------------

    def _fit_components(self, segments: dict[str, tuple]) -> dict[str, DailyPSpline]:
        """Fit a DailyPSpline for each component.

        Args:
            segments: Pre-extracted ``{component: (x_array, y_array)}`` mapping.
                Components absent from the dict had fewer than 2 data points and
                are silently skipped (or logged when ``verbose=True``).

        Returns:
            Mapping of component key → fitted ``DailyPSpline``.
        """
        s = self.settings
        fit_components = {}
        for component in self.components:
            if component not in segments:
                if self.verbose:
                    print(f"Skipping {component}: too few data points")
                continue
            x, y = segments[component]

            spl = DailyPSpline(
                bspline_degree=s.bspline_degree,
                bc_type=s.bc_type,
                n_min=s.zone.n_min,
                lambda_smoothing=s.lambda_smoothing,
                kappa_penalty=s.kappa_penalty,
                maxiter=s.maxiter,
                adaptive_iterations=s.adaptive_iterations,
                zone_knot_count_max=s.zone.knot_count_max,
                allow_heating_zone=s.zone.allow_heating_zone,
                allow_cooling_zone=s.zone.allow_cooling_zone,
                zone_criteria=s.zone.criteria,
                zone_penalty_multiplier=s.zone.penalty_multiplier,
                zone_penalty_power=s.zone.penalty_power,
                regularization_alpha=s.regularization_alpha,
                regularization_percent_lasso=s.regularization_percent_lasso,
                freeze_bp_on_convergence=s.freeze_bp_on_convergence,
            )
            spl.fit(x, y)
            fit_components[component] = spl

            if self.verbose:
                rmse = np.sqrt(np.mean((y - spl.predict(x)) ** 2))
                print(f"{component}: n={len(x)}, RMSE={rmse:.4f}")

        return fit_components

    # ------------------------------------------------------------------
    # Combination selection
    # ------------------------------------------------------------------

    def _best_combination(
        self,
    ) -> tuple[dict[str, DailyPSpline], np.ndarray, np.ndarray]:
        """Select the best season/day combination using the configured selection criteria.

        Iterates over ``self.combinations``, computing obs/pred for each combo once.
        The winner's arrays are cached so the caller can build ``BaselineMetrics``
        without an additional predict pass.

        Returns:
            A 3-tuple of ``(model_dict, obs, pred)`` where ``model_dict`` maps each
            winning component key to its fitted ``DailyPSpline``, and ``obs``/``pred``
            are the stacked training observations and predictions across all components.
        """
        ss = self.settings.split_selection

        # Pre-compute base RMSE (all-merged combo) for normalization.
        base_spl  = self.fit_components["fw-su_sh_wi"]
        obs_base  = base_spl.y
        pred_base = base_spl.predict(base_spl.x)
        wRMSE_base = float(np.sqrt(np.mean((obs_base - pred_base) ** 2)))
        N = len(obs_base)

        best_combo, best_sc = None, np.inf
        best_obs, best_pred = obs_base, pred_base
        for combo in self.combinations:
            parts = combo.split("__")
            if combo == "fw-su_sh_wi":
                obs, pred = obs_base, pred_base
            else:
                obs  = np.hstack([self.fit_components[c].y for c in parts])
                pred = np.hstack([self.fit_components[c].predict(self.fit_components[c].x) for c in parts])

            wRMSE = float(np.sqrt(np.mean((obs - pred) ** 2)))
            TSS = sum(
                float(np.sum((self.fit_components[c].y - np.mean(self.fit_components[c].y)) ** 2))
                for c in parts
            )
            # Split selection penalizes the number of structural components
            # (segments), not the total model complexity.  This matches the
            # DailyModel's approach: the question here is "should I fit
            # separate summer/winter curves?" — a structural choice penalized
            # by len(parts).  Effective degrees of freedom (edf) is used
            # separately inside the zone knot-count scan where the question
            # is "how flexible should each curve be?".
            sc = selection_criteria(
                wRMSE / wRMSE_base,
                TSS,
                N,
                len(parts),
                ss.criteria,
                ss.penalty_multiplier,
                ss.penalty_power,
            )
            if self.verbose:
                print(f"{combo:>50s}  {sc:>10.4f}")
            if sc < best_sc:
                best_combo, best_sc = combo, sc
                best_obs, best_pred = obs, pred

        if best_combo is None:
            raise RuntimeError("No valid combinations found — combinations list was empty.")

        return {c: self.fit_components[c] for c in best_combo.split("__")}, best_obs, best_pred
