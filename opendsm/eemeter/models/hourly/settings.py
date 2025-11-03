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
import pandas as pd

import pydantic

from enum import Enum
from typing import Optional, Literal, Union, TypeVar, Dict

import pywt

from opendsm.common.base_settings import BaseSettings
from opendsm.common.clustering.settings import ClusteringSettings
from opendsm.common.metrics import BaselineMetrics
from opendsm.common.const import CAlgoChoice

from opendsm.eemeter.common.warnings import EEMeterWarning

# from opendsm.common.const import CountryCode


class SelectionChoice(str, Enum):
    CYCLIC = "cyclic"
    RANDOM = "random"


class ScalingChoice(str, Enum):
    ROBUST_SCALER = "robustscaler"
    STANDARD_SCALER = "standardscaler"


class BinningChoice(str, Enum):
    EQUAL_SAMPLE_COUNT = "equal_sample_count"
    EQUAL_BIN_WIDTH = "equal_bin_width"
    SET_BIN_WIDTH = "set_bin_width"
    FIXED_BINS = "fixed_bins"


class DefaultTrainingFeatures(str, Enum):
    SOLAR = ["temperature", "ghi"]
    NONSOLAR = ["temperature"]


class AggregationMethod(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


class BaseModel(str, Enum):
    ELASTICNET = "elasticnet"
    KERNEL_RIDGE = "kernel_ridge"


class TemperatureBinSettings(BaseSettings):
    """how to bin temperature data"""
    method: BinningChoice = pydantic.Field(
        default=BinningChoice.FIXED_BINS,
    )

    """number of temperature bins"""
    n_bins: Optional[int] = pydantic.Field(
        default=None,
        ge=1,
    )

    """temperature bin width in fahrenheit"""
    bin_width: Optional[float] = pydantic.Field(
        default=25,
        ge=1,
    )

    """specified fixed temperature bins in fahrenheit"""
    fixed_bins: Optional[list[float]] = pydantic.Field(
        default=[10, 30, 50, 65, 75, 90, 105],
    )

    "minimum bin count"
    min_bin_count: Optional[int] = pydantic.Field(
        default=20,
        ge=1,
    )

    """use edge bins bool"""
    include_edge_bins: bool = pydantic.Field(
        default=True, 
    )

    """rate for edge temperature bins"""
    edge_bin_rate: Optional[Union[float, Literal["heuristic"]]] = pydantic.Field(
        default="heuristic",
    )

    """percent of total data in edge bins"""
    edge_bin_percent: Optional[float] = pydantic.Field(
        default=None,
        gt=0,
        le=0.45,
    )

    """offset normalized temperature range for edge bins (keeps exp from blowing up)"""
    edge_bin_temperature_range_offset: Optional[float] = pydantic.Field(
        default=1.0, # prior 1.0
        ge=0,
    )

    @pydantic.model_validator(mode="after")
    def _check_temperature_bins(self):
        if self.method == BinningChoice.EQUAL_SAMPLE_COUNT:
            if self.n_bins is None:
                raise ValueError(
                    "'n_bins' must be specified if 'method' is 'equal_sample_count'."
                )
            if self.n_bins < 1:
                raise ValueError("'n_bins' must be greater than 0.")

        elif self.method == BinningChoice.EQUAL_BIN_WIDTH:
            if self.bin_width is None:
                raise ValueError(
                    "'bin_width' must be specified if 'method' is 'equal_bin_width'."
                )
            if self.bin_width < 1:
                raise ValueError("'bin_width' must be greater than 0.")

        elif self.method == BinningChoice.SET_BIN_WIDTH:
            if self.bin_width is None:
                raise ValueError(
                    "'bin_width' must be specified if 'method' is 'set_bin_width'."
                )
            if self.bin_width < 1:
                raise ValueError("'bin_width' must be greater than 0.")

        elif self.method == BinningChoice.FIXED_BINS:
            if self.fixed_bins is None:
                raise ValueError(
                    "'fixed_bins' must be specified if 'method' is 'fixed_bins'."
                )

        else:
            raise ValueError(f"Invalid method: {self.method}")

        return self

    @pydantic.model_validator(mode="after")
    def _check_edge_bins(self):
        if self.include_edge_bins:
            if self.edge_bin_rate is None:
                raise ValueError(
                    "'edge_bin_rate' must be specified if 'include_edge_bins' is True."
                )
            if self.edge_bin_percent is None and self.method != BinningChoice.FIXED_BINS:
                raise ValueError(
                    "'edge_bin_days' must be specified if 'include_edge_bins' is True."
                )
            if self.edge_bin_temperature_range_offset is None:
                raise ValueError(
                    "'edge_bin_temperature_range_offset' must be specified if 'include_edge_bins' is True."
                )

        else:
            if self.edge_bin_rate is not None:
                raise ValueError(
                    "'edge_bin_rate' must be None if 'include_edge_bins' is False."
                )
            if self.edge_bin_percent is not None:
                raise ValueError(
                    "'edge_bin_days' must be None if 'include_edge_bins' is False."
                )
            if self.edge_bin_temperature_range_offset is not None:
                raise ValueError(
                    "'edge_bin_temperature_range_offset' must be None if 'include_edge_bins' is False."
                )

        return self


class ElasticNetSettings(BaseSettings):
    """ElasticNet alpha parameter"""

    alpha: float = pydantic.Field(
        default=0.0139,
        ge=0,
    )

    """ElasticNet l1_ratio parameter"""
    l1_ratio: float = pydantic.Field(
        default=0.871,
        ge=0,
        le=1,
    )

    """ElasticNet fit_intercept parameter"""
    fit_intercept: bool = pydantic.Field(
        default=True,
    )

    """ElasticNet parameter to precompute Gram matrix"""
    precompute: bool = pydantic.Field(
        default=False,
    )

    """ElasticNet max_iter parameter"""
    max_iter: int = pydantic.Field(
        default=3000,
        ge=1,
        le=2**32 - 1,
    )

    """ElasticNet copy_X parameter"""
    copy_x: bool = pydantic.Field(
        default=True,
    )

    """ElasticNet tol parameter"""
    tol: float = pydantic.Field(
        default=1e-3,
        gt=0,
    )

    """ElasticNet selection parameter"""
    selection: SelectionChoice = pydantic.Field(
        default=SelectionChoice.CYCLIC,
    )

    """ElasticNet warm_start parameter"""
    warm_start: bool = pydantic.Field(
        default=False,
    )


class KernelRidgeSettings(BaseSettings):
    """Kernel Ridge alpha parameter"""
    alpha: float = pydantic.Field(
        default=0.0425,
        ge=0,
    )

    """Kernel Ridge kernel parameter"""
    kernel: str = pydantic.Field(
        default="rbf",
    )

    """Kernel Ridge gamma parameter"""
    gamma: Optional[float] = pydantic.Field(
        default=None,
        gt=0,
    )


class AdaptiveWeightsSettings(BaseSettings):
    """Adaptive Weights for ElasticNet"""
    enabled: bool = pydantic.Field(
        default=True,
    )

    """Sigma threshold for calculating C"""
    sigma: Optional[float] = pydantic.Field(
        default=4.55,
        gt=0,
    )

    """Adaptive weights window size"""
    window_size: Optional[int] = pydantic.Field(
        default=3,
        ge=1,
        le=12,
    )

    """Algorithm to use for calculating C"""
    c_algo: Optional[CAlgoChoice] = pydantic.Field(
        default=CAlgoChoice.IQR,
    )

    """Number of iterations to iterate weights"""
    max_iter: Optional[int] = pydantic.Field(
        default=100,   # Exits early based on tol
        ge=1,
    )

    """Relative difference in weights to stop iteration"""
    tol: Optional[float] = pydantic.Field(
        default=1E-3,   # Previously was using 1e-4
        ge=0,
    )

    @pydantic.model_validator(mode="after")
    def _check_adaptive_weights(self):
        if self.enabled:
            # iterate through all the parameters to check if they are set
            # if any are None, raise an error
            pass
        else:
            # iterate through all the parameters to check if they are set
            # if any are not None, raise an error
            pass

        return self


class Criterion(str, Enum):
    AIC = "aic"
    BIC = "bic"


# analytic_features = ['GHI', 'Temperature', 'DHI', 'DNI', 'Relative Humidity', 'Wind Speed', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type']
class BaseHourlySettings(BaseSettings):
    """train features used within the model"""

    train_features: Optional[list[str]] = None

    """CVRMSE threshold for model disqualification"""
    cvrmse_threshold: float = pydantic.Field(
        default=1.4,
    )

    """PNRMSE threshold for model disqualification"""
    pnrmse_threshold: float = pydantic.Field(
        default=2.2,
    )

    """minimum number of training hours per day below which a day is excluded"""
    min_daily_training_hours: int = pydantic.Field(
        default=12,
        ge=0,
        le=24,
    )

    """temperature bin settings"""
    temperature_bin: Optional[TemperatureBinSettings] = pydantic.Field(
        default_factory=TemperatureBinSettings,
    )

    """settings for temporal clustering"""
    temporal_cluster: ClusteringSettings = pydantic.Field(
        default_factory=ClusteringSettings,
    )

    """temporal cluster aggregation method"""
    temporal_cluster_aggregation: AggregationMethod = pydantic.Field(
        default=AggregationMethod.MEDIAN,
    )

    """temporal cluster/temperature bin/temperature interaction scalar"""
    interaction_scalar: float = pydantic.Field(
        default=0.524,
        gt=0,
    )

    """supplemental time series column names"""
    supplemental_time_series_columns: Optional[list] = pydantic.Field(
        default=None,
    )

    """supplemental categorical column names"""
    supplemental_categorical_columns: Optional[list] = pydantic.Field(
        default=None,
    )

    """base model type"""
    base_model: BaseModel = pydantic.Field(
        default=BaseModel.ELASTICNET,
    )

    """ElasticNet settings"""
    elasticnet: Optional[ElasticNetSettings] = pydantic.Field(
        default_factory=ElasticNetSettings,
    )

    """Kernel Ridge settings"""
    kernel_ridge: Optional[KernelRidgeSettings] = pydantic.Field(
        default_factory=KernelRidgeSettings,
    )

    """Adaptive Weights settings"""
    adaptive_weights: AdaptiveWeightsSettings = pydantic.Field(
        default_factory=AdaptiveWeightsSettings,
    )

    """Feature scaling method"""
    scaling_method: ScalingChoice = pydantic.Field(
        default=ScalingChoice.STANDARD_SCALER,
    )

    """Significance level used for uncertainty calculations"""
    uncertainty_alpha: float = pydantic.Field(
        default=0.1,
        ge=0,
        le=1,
        description="Significance level used for uncertainty calculations",
    )

    """seed for any random state assignment (ElasticNet, Clustering)"""
    seed: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
    )

    @pydantic.model_validator(mode="after")
    def _check_seed(self):
        if self.seed is None:
            self._seed = np.random.randint(0, 2**32 - 1, dtype=np.int64)
        else:
            self._seed = self.seed

        self.elasticnet._seed = self._seed
        self.temporal_cluster._seed = self._seed

        return self

    @pydantic.model_validator(mode="after")
    def _remove_unselected_model_settings(self):
        self.model_config["frozen"] = False
        
        if self.base_model == BaseModel.ELASTICNET:
            self.kernel_ridge = None
        elif self.base_model == BaseModel.KERNEL_RIDGE:
            self.elasticnet = None

        self.model_config["frozen"] = True

        return self

    def add_default_features(self, incoming_columns: list[str]):
        """ "called prior fit step to set default training features"""
        if "ghi" in incoming_columns:
            default_features = ["temperature", "ghi"]
        else:
            default_features = ["temperature"]
        return self.model_copy(update={"train_features": default_features})


class HourlySolarSettings(BaseHourlySettings):
    """train features used within the model"""

    train_features: list[str] = pydantic.Field(
        default=["temperature", "ghi"],
    )

    @pydantic.field_validator("train_features", mode="after")
    def _add_required_features(cls, v):
        required_features = ["ghi", "temperature"]
        for feature in required_features:
            if feature not in v:
                v.insert(0, feature)
        return v


class HourlyNonSolarSettings(BaseHourlySettings):
    """number of temperature bins"""

    # TEMPERATURE_BIN_COUNT: Optional[int] = pydantic.Field(
    #     default=10,
    #     ge=1,
    # )
    train_features: list[str] = pydantic.Field(
        default=["temperature"],
    )

    @pydantic.field_validator("train_features", mode="after")
    def _add_required_features(cls, v):
        if "temperature" not in v:
            v.insert(0, "temperature")
        return v


class ModelInfo(pydantic.BaseModel):
    """additional information about the model"""

    warnings: list[EEMeterWarning]
    disqualification: list[EEMeterWarning]
    error: dict
    baseline_timezone: str
    version: str


class SerializeModel(BaseSettings):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    settings: Optional[BaseHourlySettings] = None
    temporal_clusters: Optional[list[list[int]]] = None
    temperature_bin_edges: Optional[list] = None
    temperature_edge_bin_coefficients: Optional[Dict[int, Dict[str, float]]] = None
    ts_features: Optional[list] = None
    categorical_features: Optional[list] = None
    feature_scaler: Optional[Dict[str, list[float]]] = None
    catagorical_scaler: Optional[Dict[str, list[float]]] = None
    y_scaler: Optional[list[float]] = None
    coefficients: Optional[list[list[float]]] = None
    intercept: Optional[list[float]] = None
    baseline_metrics: Optional[BaselineMetrics] = None
    info: ModelInfo
