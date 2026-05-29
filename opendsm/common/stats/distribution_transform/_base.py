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

"""Abstract base class for per-dimension distribution transforms.

All transform classes (Standardize, Bisymlog, YeoJohnson, BoxCox) inherit
from :class:`TransformBase`, which provides shared logic for input
validation, the fit/transform/inverse_transform pipeline, and serialisation.

Subclasses implement three hooks:

- ``_init_params(D)`` — allocate per-dimension fitted arrays
- ``_fit_dim(d, col_finite, finite_mask, X, return_transformed, out)``
  — fit one dimension and optionally write to ``out``
- ``_transform_dim(col_finite, d)`` / ``_inverse_transform_dim(col_finite, d)``
  — forward / inverse on a single dimension's finite values
"""

import json
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class TransformBase(ABC):
    """Base class for per-dimension transforms with invertibility.

    Parameters
    ----------
    min_variance : float
        Dimensions with range below this are skipped.
    min_samples : int
        Dimensions with fewer finite samples are skipped.
    """

    def __init__(self, min_variance=1e-10, min_samples=5):
        self.min_variance = min_variance
        self.min_samples = min_samples
        self.skip_dims_ = None
        self.n_features_ = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check_is_fitted(self):
        if self.n_features_ is None:
            raise RuntimeError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call fit() or fit_transform() first."
            )

    @staticmethod
    def _to_2d(X):
        """Convert to float64 2-D; return (X_2d, was_1d)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return X.reshape(-1, 1), True
        if X.ndim == 2:
            return X, False
        raise ValueError(f"Expected 1-D or 2-D input, got {X.ndim}-D")

    def _validate(self, X, reset=False):
        X, is_1d = self._to_2d(X)
        if not reset and self.n_features_ is not None:
            if X.shape[1] != self.n_features_:
                raise ValueError(
                    f"Expected {self.n_features_} features, got {X.shape[1]}"
                )
        return X, is_1d

    # ------------------------------------------------------------------
    # Hooks (subclass must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def _init_params(self, D):
        """Allocate per-dimension parameter arrays for D features."""

    @abstractmethod
    def _fit_dim(self, d, col_finite, finite_mask, X, return_transformed, out):
        """Fit dimension *d*.

        Must set ``self.skip_dims_[d] = True`` on failure and return.
        When *return_transformed* is True, write transformed finite values
        into ``out[finite_mask, d]``.
        """

    @abstractmethod
    def _transform_dim(self, col_finite, d):
        """Forward-transform finite values of dimension *d*."""

    @abstractmethod
    def _inverse_transform_dim(self, col_finite, d):
        """Inverse-transform finite values of dimension *d*."""

    @abstractmethod
    def _serialise_params(self):
        """Return a dict of fitted parameter arrays (lists, not ndarray)."""

    @abstractmethod
    def _deserialise_params(self, d):
        """Restore fitted parameter arrays from dict *d*."""

    def _serialise_hyperparams(self):
        """Return a dict of constructor kwargs for serialisation."""
        return {
            "min_variance": self.min_variance,
            "min_samples": self.min_samples,
        }

    # ------------------------------------------------------------------
    # fit / transform pipeline
    # ------------------------------------------------------------------

    def fit(self, X):
        X, _ = self._validate(X, reset=True)
        self._fit_core(X)
        return self

    def fit_transform(self, X):
        X_orig = np.asarray(X)
        is_1d = X_orig.ndim == 1
        X, _ = self._validate(X_orig, reset=True)
        out = self._fit_core(X, return_transformed=True)
        return out.ravel() if is_1d else out

    # Whether to wrap the fit loop in np.seterr(all="warn").
    # Only needed for transforms that call code which may leave
    # numpy error state elevated (e.g. Huber M-estimator via numba).
    _GUARD_NUMPY_ERRORS = False

    def _fit_core(self, X, return_transformed=False):
        N, D = X.shape
        self.n_features_ = D
        self.skip_dims_ = np.zeros(D, dtype=bool)
        self._init_params(D)

        out = X.copy() if return_transformed else None

        old_err = np.seterr(all="warn") if self._GUARD_NUMPY_ERRORS else None
        try:
            for d in range(D):
                col = X[:, d]
                fm = np.isfinite(col)
                col_f = col[fm]

                if len(col_f) < self.min_samples:
                    self.skip_dims_[d] = True
                    continue
                if col_f.max() - col_f.min() < self.min_variance:
                    self.skip_dims_[d] = True
                    continue

                try:
                    self._fit_dim(d, col_f, fm, X, return_transformed, out)
                except Exception:
                    warnings.warn(
                        f"{type(self).__name__}: fitting failed for dim {d}; "
                        f"passing through.",
                        RuntimeWarning, stacklevel=4,
                    )
                    self.skip_dims_[d] = True
        finally:
            if old_err is not None:
                np.seterr(**old_err)
        return out

    def transform(self, X):
        self._check_is_fitted()
        X, is_1d = self._validate(X)
        out = X.copy()
        for d in np.where(~self.skip_dims_)[0]:
            col = out[:, d]
            fm = np.isfinite(col)
            col[fm] = self._transform_dim(col[fm], d)
        return out.ravel() if is_1d else out

    def inverse_transform(self, X):
        self._check_is_fitted()
        X, is_1d = self._validate(X)
        out = X.copy()
        for d in np.where(~self.skip_dims_)[0]:
            col = out[:, d]
            fm = np.isfinite(col)
            col[fm] = self._inverse_transform_dim(col[fm], d)
        return out.ravel() if is_1d else out

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self):
        self._check_is_fitted()
        d = self._serialise_hyperparams()
        d["n_features"] = self.n_features_
        d["skip_dims"] = self.skip_dims_.tolist()
        d.update(self._serialise_params())
        return d

    # Subclasses should list constructor kwarg names here for from_dict.
    _HYPERPARAM_KEYS = ("min_variance", "min_samples")

    @classmethod
    def from_dict(cls, d):
        ctor_kwargs = {k: d[k] for k in cls._HYPERPARAM_KEYS if k in d}
        obj = cls(**ctor_kwargs)
        obj.n_features_ = d["n_features"]
        obj.skip_dims_ = np.array(d["skip_dims"], dtype=bool)
        obj._deserialise_params(d)
        return obj

    def to_json(self, path=None):
        s = json.dumps(self.to_dict())
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
            return None
        return s

    @classmethod
    def from_json(cls, path_or_string):
        try:
            p = Path(path_or_string)
            if p.exists() and p.is_file():
                text = p.read_text(encoding="utf-8")
            else:
                text = str(path_or_string)
        except OSError:
            text = str(path_or_string)
        return cls.from_dict(json.loads(text))


class PowerTransformMixin:
    """Shared fitted-parameter storage for YeoJohnson and BoxCox.

    Both power transforms store the same five per-dimension arrays:
    lambdas, pre_mu, pre_sigma, post_mu, post_sigma.  This mixin
    provides _init_params, _serialise_params, and _deserialise_params
    so that neither class duplicates the logic.
    """

    _HYPERPARAM_KEYS = ("robust", "Q_perc", "outlier_alpha", "min_variance", "min_samples")
    _GUARD_NUMPY_ERRORS = True

    def _init_params(self, D):
        self.lambdas_ = np.ones(D)
        self.pre_mu_ = np.zeros(D)
        self.pre_sigma_ = np.ones(D)
        self.post_mu_ = np.zeros(D)
        self.post_sigma_ = np.ones(D)

    def _serialise_params(self):
        return {
            "lambdas": self.lambdas_.tolist(),
            "pre_mu": self.pre_mu_.tolist(),
            "pre_sigma": self.pre_sigma_.tolist(),
            "post_mu": self.post_mu_.tolist(),
            "post_sigma": self.post_sigma_.tolist(),
        }

    def _deserialise_params(self, d):
        self.lambdas_ = np.array(d["lambdas"])
        self.pre_mu_ = np.array(d["pre_mu"])
        self.pre_sigma_ = np.array(d["pre_sigma"])
        self.post_mu_ = np.array(d["post_mu"])
        self.post_sigma_ = np.array(d["post_sigma"])

    def _serialise_hyperparams(self):
        hp = super()._serialise_hyperparams()
        hp.update({
            "robust": self.robust,
            "Q_perc": self.Q_perc,
            "outlier_alpha": self.outlier_alpha,
        })
        return hp
