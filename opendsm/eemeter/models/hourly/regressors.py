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

import numpy as np

from copy import deepcopy as copy

from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso

from opendsm.common.stats.adaptive_loss import adaptive_weights


class _NullFitMixin:
    """Mixin that short-circuits fit() when y is all zeros."""

    def _null_fit(self, X, y):
        """Set all coefficients and intercepts to zero for zero-valued targets."""
        n_features = X.shape[1]
        if y.ndim > 1:
            n_targets = y.shape[1]
            self.coef_ = np.zeros((n_targets, n_features))
            self.intercept_ = np.zeros(n_targets)
        else:
            self.coef_ = np.zeros(n_features)
            self.intercept_ = 0.0
        return self

    def fit(self, X, y, **kwargs):
        if not np.any(y):
            return self._null_fit(X, y)
        return super().fit(X, y, **kwargs)


class SafeLinearRegression(_NullFitMixin, LinearRegression):
    pass


class SafeRidge(_NullFitMixin, Ridge):
    pass


class SafeLasso(_NullFitMixin, Lasso):
    pass


class SafeElasticNet(_NullFitMixin, ElasticNet):
    pass


class AdaptiveElasticNetRegressor:
    def __init__(self, base_model, settings):
        self.settings = settings

        self.base_model = base_model
        self.base_model.warm_start = True

        self._hour_model = copy(self.base_model)

    def _null_fit(self, X, num_hours):
        """Set all coefficients and intercepts to zero for zero-valued targets."""
        self.base_model.coef_ = np.zeros((num_hours, X.shape[1]))
        self.base_model.intercept_ = np.zeros(num_hours)
        self.base_model.adaptive_iterations = 0
        self.base_model.adaptive_alpha = np.full(num_hours, 2.0)
        self.base_model.adaptive_weights = np.ones((X.shape[0], num_hours))

        return self

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model with X, y data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            The target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns:
        --------
        self : returns an instance of self.
        """
        settings = self.settings.adaptive_weights
        window_size = self.settings.adaptive_weights.window_size - 1
        tol = self.settings.adaptive_weights.tol

        num_hours = y.shape[1]

        if not np.any(y):
            return self._null_fit(X, num_hours)

        # fit the base model as an initial guess
        self.base_model.fit(X, y, sample_weight=sample_weight)

        if sample_weight is None:
            weights = np.ones((X.shape[0], num_hours))
        else:
            weights = sample_weight

        hour_fit = [False for _ in range(num_hours)]
        alpha_prior = np.array([2.0 for _ in range(num_hours)])
        alpha_min = alpha_prior.copy()
        for i in range(settings.max_iter):
            if all(hour_fit):
                i -= 1
                break

            # get prediction and residuals for all hours
            y_fit = self.base_model.predict(X)
            resid = y - y_fit

            for hour in range(num_hours):
                # if hour_fit[hour]:
                #     continue

                # Update weights
                # Calculate weights using window of hours
                window_idx = np.arange(hour - window_size, hour + window_size + 1)

                # if idx_i < 0, roll to the end or if idx_i >= num_hours, roll to the beginning
                for idx_i in range(len(window_idx)):
                    if window_idx[idx_i] < 0:
                        window_idx[idx_i] = num_hours + window_idx[idx_i]

                    if window_idx[idx_i] >= num_hours:
                        window_idx[idx_i] = window_idx[idx_i] - num_hours

                # unique values in idx only
                window_idx = list(set(window_idx))

                # calculate weights
                weights_update, _, alpha = adaptive_weights(
                    resid[:,window_idx].flatten(),
                    alpha="adaptive",
                    sigma=settings.sigma,
                    quantile=0.25,
                    min_weight=0.0,
                    C_algo=settings.c_algo,
                )

                # break criteria
                if (alpha == 2) or (np.abs(alpha - alpha_prior[hour]) <= tol):
                    hour_fit[hour] = True
                    continue
                else:
                    hour_fit[hour] = False

                # update weights and alpha_prior
                alpha_prior[hour] = alpha
                alpha_min[hour] = min(alpha_min[hour], alpha)

                # trim weights to hour size
                if window_size > 0:
                    # get index of hour in window_idx
                    idx = window_idx.index(hour)
                    hour_len = int(len(weights_update)/len(window_idx))

                    weights_update = weights_update[idx*hour_len:(idx+1)*hour_len]

                weights[:, hour] *= weights_update

                # update hour model from base model
                self._hour_model.coef_ = self.base_model.coef_[hour,:]
                self._hour_model.intercept_ = self.base_model.intercept_[hour]

                # fit
                self._hour_model.fit(
                    X,
                    y[:, hour],
                    sample_weight=weights[:, hour]
                )

                # update base model from refit hour model
                self.base_model.coef_[hour,:] = self._hour_model.coef_
                self.base_model.intercept_[hour] = self._hour_model.intercept_

        # save info to base_model
        self.base_model.adaptive_iterations = i
        self.base_model.adaptive_alpha = alpha_min
        self.base_model.adaptive_weights = weights

        return self

    @property
    def is_fit(self):
        """Check if the model is fitted."""
        is_fit = True

        if not hasattr(self.base_model, "coef_"):
            is_fit = False

        if not hasattr(self.base_model, "intercept_"):
            is_fit = False

        return is_fit

    def predict(self, X):
        """
        Predict using the model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y : array of shape (n_samples,) or (n_samples, n_targets)
            The predicted values.
        """
        if not self.is_fit:
            raise RuntimeError("Model must be fit before predictions can be made.")

        y = self.base_model.predict(X)

        return y

    @property
    def coef_(self):
        """Get model coefficients."""
        if not hasattr(self.base_model, "coef_"):
            raise RuntimeError("Model coefficients must be set before accessed.")

        return self.base_model.coef_

    @coef_.setter
    def coef_(self, val):
        self.base_model.coef_ = val

    @property
    def intercept_(self):
        """Get model intercepts."""
        if not hasattr(self.base_model, "intercept_"):
            raise RuntimeError("Model intercepts must be set before accessed.")

        return self.base_model.intercept_

    @intercept_.setter
    def intercept_(self, val):
        """Set model intercepts"""
        self.base_model.intercept_ = val