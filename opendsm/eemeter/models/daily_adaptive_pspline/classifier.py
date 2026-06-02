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

"""Regime assignment classifier and feature engineering.

Trains a lightweight classifier on (trailing_avg_temp, sin/cos doy,
day_of_week, [daily_temp]) to predict regime labels discovered by EM.
Used at prediction time to assign new days to regimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    settings,
    trailing_temps: np.ndarray | None = None,
) -> np.ndarray:
    """Build classifier feature matrix from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex and a 'temperature' column.
    settings : DailyAdaptivePSplineSettings
        Controls which features are included.
    trailing_temps : array, optional
        Pre-computed trailing average temperatures.  If None, computed
        from the temperature column using ``settings.trailing_avg_window``.

    Returns
    -------
    X : ndarray of shape (n_days, n_features)
    """
    features = []
    temp = df["temperature"].values
    idx = df.index

    # Day of year (1-366)
    doy = idx.dayofyear.values.astype(float)

    if settings.use_trailing_avg_temp:
        if trailing_temps is None:
            trailing_temps = _trailing_avg(temp, idx, settings.trailing_avg_window)
        features.append(trailing_temps.reshape(-1, 1))

    if settings.use_doy_harmonics:
        angle = 2.0 * np.pi * doy / 365.25
        features.append(np.column_stack([np.sin(angle), np.cos(angle)]))

    if settings.use_day_of_week:
        dow = idx.dayofweek.values  # 0=Mon .. 6=Sun
        # One-hot encode, drop last column to avoid collinearity
        dow_onehot = np.zeros((len(dow), 6), dtype=float)
        for i in range(6):
            dow_onehot[:, i] = (dow == i).astype(float)
        features.append(dow_onehot)

    if settings.use_daily_temp:
        features.append(temp.reshape(-1, 1))

    if not features:
        raise ValueError("No classifier features enabled in settings.")

    return np.hstack(features)


def _trailing_avg(
    temp: np.ndarray,
    index: pd.DatetimeIndex,
    window: int,
) -> np.ndarray:
    """Trailing average temperature over a fixed window.

    For the first ``window`` days, uses all available prior data
    (partial window) rather than producing NaN.
    """
    series = pd.Series(temp, index=index).sort_index()
    trailing = series.rolling(window, min_periods=1).mean()
    # Re-align to original index order
    return trailing.reindex(index).values


def feature_names(settings) -> list[str]:
    """Human-readable names for each column of build_features output."""
    names = []
    if settings.use_trailing_avg_temp:
        names.append("trailing_avg_temp")
    if settings.use_doy_harmonics:
        names.extend(["sin_doy", "cos_doy"])
    if settings.use_day_of_week:
        names.extend([f"dow_{i}" for i in range(6)])
    if settings.use_daily_temp:
        names.append("daily_temp")
    return names


# ------------------------------------------------------------------
# Classifier
# ------------------------------------------------------------------

@dataclass
class RegimeClassifier:
    """Multinomial logistic classifier for regime assignment.

    Stores weights and intercept directly (no sklearn dependency at
    predict time) for portable serialization.

    Attributes
    ----------
    weights : ndarray of shape (n_classes, n_features)
    intercept : ndarray of shape (n_classes,)
    classes : ndarray of shape (n_classes,)
        Regime labels (int).
    feature_scales : ndarray of shape (n_features,)
        Per-feature standard deviations used for normalization.
    feature_means : ndarray of shape (n_features,)
        Per-feature means used for normalization.
    """

    weights: np.ndarray = field(default_factory=lambda: np.empty(0))
    intercept: np.ndarray = field(default_factory=lambda: np.empty(0))
    classes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    feature_means: np.ndarray = field(default_factory=lambda: np.empty(0))
    feature_scales: np.ndarray = field(default_factory=lambda: np.empty(0))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        settings,
    ) -> RegimeClassifier:
        """Train logistic regression on features X and regime labels y.

        Uses sklearn for training but extracts weights for serialization.
        The ``daily_temp_penalty`` in settings applies heavier L2
        regularization to the daily-temperature column.
        """
        from sklearn.linear_model import LogisticRegression

        # Standardize features
        self.feature_means = X.mean(axis=0)
        self.feature_scales = X.std(axis=0)
        self.feature_scales[self.feature_scales < 1e-10] = 1.0
        X_norm = (X - self.feature_means) / self.feature_scales

        # Per-feature penalty: scale daily_temp column by 1/penalty so
        # the L2 norm penalizes it more heavily.
        if settings.use_daily_temp and settings.daily_temp_penalty > 1.0:
            temp_col = X_norm.shape[1] - 1  # daily_temp is always last
            X_norm[:, temp_col] /= settings.daily_temp_penalty

        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            # Only one regime — trivial classifier
            self.classes = unique_classes
            self.weights = np.zeros((1, X.shape[1]))
            self.intercept = np.zeros(1)
            return self

        clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
        )
        clf.fit(X_norm, y)

        self.weights = clf.coef_.copy()
        self.intercept = clf.intercept_.copy()
        self.classes = clf.classes_.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regime labels for new days."""
        if len(self.classes) == 1:
            return np.full(len(X), self.classes[0], dtype=int)

        X_norm = (X - self.feature_means) / self.feature_scales
        logits = X_norm @ self.weights.T + self.intercept

        if len(self.classes) == 2:
            # Binary case: sklearn stores one row; positive logit → class 1
            return self.classes[(logits.ravel() > 0).astype(int)]

        return self.classes[np.argmax(logits, axis=1)]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "intercept": self.intercept.tolist(),
            "classes": self.classes.tolist(),
            "feature_means": self.feature_means.tolist(),
            "feature_scales": self.feature_scales.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RegimeClassifier:
        return cls(
            weights=np.array(data["weights"]),
            intercept=np.array(data["intercept"]),
            classes=np.array(data["classes"], dtype=int),
            feature_means=np.array(data["feature_means"]),
            feature_scales=np.array(data["feature_scales"]),
        )
