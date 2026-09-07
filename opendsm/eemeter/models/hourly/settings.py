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

from opendsm.common.base_settings import BaseSettings, DeveloperField, UserField
from opendsm.common.clustering.settings import ClusteringSettings
from opendsm.common.metrics import BaselineMetrics
from opendsm.common.const import CAlgoChoice

from opendsm.eemeter.common.data_settings import HourlyDataSettings
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
    method: BinningChoice = DeveloperField(
        default=BinningChoice.FIXED_BINS,
    )

    """number of temperature bins"""
    n_bins: Optional[int] = DeveloperField(
        default=None,
        ge=1,
    )

    """temperature bin width in fahrenheit"""
    bin_width: Optional[float] = DeveloperField(
        default=25,
        ge=1,
    )

    """specified fixed temperature bins in fahrenheit"""
    fixed_bins: Optional[list[float]] = DeveloperField(
        default=[10, 30, 50, 65, 75, 90, 105],
    )

    "minimum bin count"
    min_bin_count: Optional[int] = DeveloperField(
        default=20,
        ge=1,
    )

    """use edge bins bool"""
    include_edge_bins: bool = DeveloperField(
        default=True, 
    )

    """rate for edge temperature bins"""
    edge_bin_rate: Optional[Union[float, Literal["heuristic"]]] = DeveloperField(
        default="heuristic",
    )

    """percent of total data in edge bins"""
    edge_bin_percent: Optional[float] = DeveloperField(
        default=None,
        gt=0,
        le=0.45,
    )

    """offset normalized temperature range for edge bins (keeps exp from blowing up)"""
    edge_bin_temperature_range_offset: Optional[float] = DeveloperField(
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

    alpha: float = DeveloperField(
        default=0.0139,
        ge=0,
    )

    """ElasticNet l1_ratio parameter"""
    l1_ratio: float = DeveloperField(
        default=0.871,
        ge=0,
        le=1,
    )

    """ElasticNet fit_intercept parameter"""
    fit_intercept: bool = DeveloperField(
        default=True,
    )

    """ElasticNet parameter to precompute Gram matrix"""
    precompute: bool = DeveloperField(
        default=False,
    )

    """ElasticNet max_iter parameter"""
    max_iter: int = DeveloperField(
        default=3000,
        ge=1,
        le=2**32 - 1,
    )

    """ElasticNet copy_X parameter"""
    copy_x: bool = DeveloperField(
        default=True,
    )

    """ElasticNet tol parameter"""
    tol: float = DeveloperField(
        default=1e-3,
        gt=0,
    )

    """ElasticNet selection parameter"""
    selection: SelectionChoice = DeveloperField(
        default=SelectionChoice.CYCLIC,
    )

    """ElasticNet warm_start parameter"""
    warm_start: bool = DeveloperField(
        default=False,
    )


class KernelRidgeSettings(BaseSettings):
    """Kernel Ridge alpha parameter"""
    alpha: float = DeveloperField(
        default=0.0425,
        ge=0,
    )

    """Kernel Ridge kernel parameter"""
    kernel: str = DeveloperField(
        default="rbf",
    )

    """Kernel Ridge gamma parameter"""
    gamma: Optional[float] = DeveloperField(
        default=None,
        gt=0,
    )


class AdaptiveWeightsSettings(BaseSettings):
    """Adaptive Weights for ElasticNet"""
    enabled: bool = DeveloperField(
        default=True,
    )

    """Sigma threshold for calculating C"""
    sigma: Optional[float] = DeveloperField(
        default=4.55,
        gt=0,
    )

    """Adaptive weights window size"""
    window_size: Optional[int] = DeveloperField(
        default=3,
        ge=1,
        le=12,
    )

    """Algorithm to use for calculating C"""
    c_algo: Optional[CAlgoChoice] = DeveloperField(
        default=CAlgoChoice.IQR,
    )

    """Number of iterations to iterate weights"""
    max_iter: Optional[int] = DeveloperField(
        default=100,   # Exits early based on tol
        ge=1,
    )

    """Relative difference in weights to stop iteration"""
    tol: Optional[float] = DeveloperField(
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

    train_features: Optional[list[str]] = UserField(default=None)

    data: HourlyDataSettings = DeveloperField(default_factory=HourlyDataSettings)

    """CVRMSE threshold for model disqualification"""
    cvrmse_threshold: float = DeveloperField(
        default=1.4,
    )

    """PNRMSE threshold for model disqualification"""
    pnrmse_threshold: float = DeveloperField(
        default=2.2,
    )

    """minimum number of training hours per day below which a day is excluded"""
    min_daily_training_hours: int = DeveloperField(
        default=12,
        ge=0,
        le=24,
    )

    """temperature bin settings"""
    temperature_bin: Optional[TemperatureBinSettings] = DeveloperField(
        default_factory=TemperatureBinSettings,
    )

    """settings for temporal clustering"""
    temporal_cluster: ClusteringSettings = DeveloperField(
        default_factory=lambda: ClusteringSettings(
            # the RBF affinity and the median magnitude feature are tuned as a pair
            min_cluster_size=2,
            small_cluster_mode="outlier",
            outlier_removal_sigma=None,
            algorithm_selection="spectral",
            spectral={
                "affinity": "rbf",
                "scoring": {
                    "weights": {"calinski_harabasz_index": 1.0},
                    "k_penalty": {"enabled": False},
                },
                "eigengap_weight": 0.0,
            },
            spectral_divisive={
                "recluster_count": 0,
                "scoring": {
                    "weights": {"calinski_harabasz_index": 1.0},
                    "k_penalty": {"enabled": False},
                },
            },
            feature_transform={
                "normalize": {"method": "standardize", "scope": "global"},
                "wavelet": {
                    "enabled": True,
                    "pca_scope": "global",
                    "pca_n_components": "mle",
                    "variance_weighted": False,
                },
                "fpca": {"enabled": False},
                "magnitude_features": {"features": ["median"]},
            },
        ),
    )

    """temporal cluster aggregation method"""
    temporal_cluster_aggregation: AggregationMethod = DeveloperField(
        default=AggregationMethod.MEDIAN,
    )

    """temporal cluster/temperature bin/temperature interaction scalar"""
    interaction_scalar: float = DeveloperField(
        default=0.524,
        gt=0,
    )

    """scalar for ghi feature"""
    ghi_scalar: float = DeveloperField(
        default=1.0,
        gt=0,
    )

    """supplemental time series column names; the model does not impute them: a day missing a
    value is left out of the fit, and a missing value at prediction takes the baseline median for
    its hour"""
    supplemental_time_series_columns: Optional[list] = UserField(
        default=None,
    )

    """supplemental categorical column names, as numeric codes; a day takes its first value, a
    day with none is left out of the fit, and a day with none at prediction takes the most common
    baseline value"""
    supplemental_categorical_columns: Optional[list] = UserField(
        default=None,
    )

    """base model type"""
    base_model: BaseModel = DeveloperField(
        default=BaseModel.ELASTICNET,
    )

    """ElasticNet settings"""
    elasticnet: Optional[ElasticNetSettings] = DeveloperField(
        default_factory=ElasticNetSettings,
    )

    """Kernel Ridge settings"""
    kernel_ridge: Optional[KernelRidgeSettings] = DeveloperField(
        default_factory=KernelRidgeSettings,
    )

    """Adaptive Weights settings"""
    adaptive_weights: AdaptiveWeightsSettings = DeveloperField(
        default_factory=AdaptiveWeightsSettings,
    )

    """Feature scaling method"""
    scaling_method: ScalingChoice = DeveloperField(
        default=ScalingChoice.STANDARD_SCALER,
    )

    """Significance level used for uncertainty calculations"""
    uncertainty_alpha: float = UserField(
        default=0.1,
        ge=0,
        le=1,
        description="Significance level used for uncertainty calculations",
    )

    """Multiplicative scale factor for per-point uncertainty"""
    uncertainty_scale_factor: float = DeveloperField(
        default=1.0,
        ge=0.1,
        description=(
            "Multiplicative scale factor applied to per-point uncertainty. "
            "Compensates for in-sample RMSE bias from pipeline stages not "
            "captured by elastic-net edf (clustering, PCA, temperature binning). "
            "Adjustable via post-deployment coverage calibration."
        ),
    )

    """seed for any random state assignment (ElasticNet, Clustering)"""
    seed: Optional[int] = UserField(
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

        feature_transform = self.temporal_cluster.feature_transform
        for transform in (feature_transform.wavelet, feature_transform.fpca):
            if transform is not None:
                transform._seed = self._seed

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

    train_features: list[str] = UserField(
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
    train_features: list[str] = UserField(
        default=["temperature"],
    )

    @pydantic.field_validator("train_features", mode="after")
    def _add_required_features(cls, v):
        if "temperature" not in v:
            v.insert(0, "temperature")
        return v


class ModelInfo(pydantic.BaseModel):
    """additional information about the model"""

    baseline_timezone: str
    disqualification: list[EEMeterWarning]
    warnings: list[EEMeterWarning]
    version: str
    is_electricity_data: bool
    pv_start: Optional[str] = None


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
    supplemental_fill_values: Optional[Dict[str, list[float]]] = None
    baseline_metrics: Optional[BaselineMetrics] = None
    baseline_hour_metrics: Optional[Dict[str, BaselineMetrics]] = None
    info: ModelInfo
