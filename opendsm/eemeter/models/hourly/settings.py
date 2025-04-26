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


class DefaultTrainingFeatures(str, Enum):
    SOLAR = ["temperature", "ghi"]
    NONSOLAR = ["temperature"]


class AggregationMethod(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


class BaseModel(str, Enum):
    ELASTICNET = "elasticnet"
    SGDREGRESSOR = "sgdregressor"
    KERNEL_RIDGE = "kernel_ridge"
    LASSO_LARS = "lasso_lars"


class TemperatureBinSettings(BaseSettings):
    """how to bin temperature data"""
    method: BinningChoice = pydantic.Field(
        default=BinningChoice.SET_BIN_WIDTH,
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

    """use edge bins bool"""
    include_edge_bins: bool = pydantic.Field(
        default=True, 
    )

    """rate for edge temperature bins"""
    edge_bin_rate: Optional[Union[float, Literal["heuristic"]]] = pydantic.Field(
        default="heuristic", # prior "heuristic"
    )

    """percent of total data in edge bins"""
    edge_bin_percent: Optional[float] = pydantic.Field(
        default=0.087329, # prior 0.045
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
        # check that temperature bin count is set based on binning method
        if self.method is None:
            if self.n_bins is not None:
                raise ValueError("'n_bins' must be None if 'method' is None.")
            if self.bin_width is not None:
                raise ValueError("'n_bins' must be None if 'method' is None.")
        else:
            if self.method == BinningChoice.SET_BIN_WIDTH:
                if self.bin_width is None:
                    raise ValueError(
                        "'n_bins' must be specified if 'method' is 'set_bin_width'."
                    )
                elif isinstance(self.bin_width, float):
                    if self.bin_width <= 0:
                        raise ValueError("'bin_width' must be greater than 0.")

                if self.n_bins is not None:
                    raise ValueError(
                        "'n_bins' must be None if 'method' is 'set_bin_width'."
                    )
            else:
                if self.n_bins is None:
                    raise ValueError(
                        "'n_bins' must be specified if 'method' is not None."
                    )
                if self.bin_width is not None:
                    raise ValueError("'n_bins' must be None if 'method' is not None.")

        return self

    @pydantic.model_validator(mode="after")
    def _check_edge_bins(self):
        if self.method != BinningChoice.SET_BIN_WIDTH:
            if self.include_edge_bins:
                raise ValueError(
                    "'include_edge_bins' must be False if 'method' is not 'set_bin_width'."
                )

        if self.include_edge_bins:
            if self.edge_bin_rate is None:
                raise ValueError(
                    "'edge_bin_rate' must be specified if 'include_edge_bins' is True."
                )
            if self.edge_bin_percent is None:
                raise ValueError(
                    "'edge_bin_days' must be specified if 'include_edge_bins' is True."
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

        return self


class ElasticNetSettings(BaseSettings):
    """ElasticNet alpha parameter"""

    alpha: float = pydantic.Field(
        default=0.009613,
        ge=0,
    )

    """ElasticNet l1_ratio parameter"""
    l1_ratio: float = pydantic.Field(
        default=1.0,
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


class LossChoice(str, Enum):
    SQUARED_ERROR = "squared_error"
    HUBER = "huber"
    EPSILON_INSENSITIVE = "epsilon_insensitive"
    SQUARED_EPSILON_INSENSITIVE = "squared_epsilon_insensitive"


class LearningRateChoice(str, Enum):
    CONSTANT = "constant"
    OPTIMAL = "optimal"
    INVSCALING = "invscaling"
    ADAPTIVE = "adaptive"


class SGDSettings(BaseSettings):
    loss: LossChoice = pydantic.Field(
        default=LossChoice.HUBER,
    )

    alpha: float = pydantic.Field(
        default=0.0175,
        ge=0,
    )

    l1_ratio: float = pydantic.Field(
        default=1.0,
        ge=0,
        le=1,
    )

    epsilon: float = pydantic.Field(
        default=5.0,
        gt=0,
    )

    adaptive_epsilon_enabled: bool = pydantic.Field(
        default=True,
    )

    adaptive_epsilon_sigma_threshold: float = pydantic.Field(
        default=8.9,
        gt=0,
    )

    adaptive_epsilon_iter: int = pydantic.Field(
        default=10,
        ge=1,
        le=2**32 - 1,
    )

    adaptive_epsilon_tolerance: float = pydantic.Field(
        default=0.05,
        gt=0,
    )

    fit_intercept: bool = pydantic.Field(
        default=True,
    )

    max_iter: int = pydantic.Field(
        default=3000,
        ge=1,
        le=2**32 - 1,
    )

    tol: float = pydantic.Field(
        default=1e-4,
        gt=0,
    )

    learning_rate: LearningRateChoice = pydantic.Field(
        default=LearningRateChoice.CONSTANT,
    )

    eta0: float = pydantic.Field(
        default=0.01,
        gt=0,
    )

    power_t: float = pydantic.Field(
        default=0.5,
    )
    
    shuffle: bool = pydantic.Field(
        default=True,
    )

    early_stopping: bool = pydantic.Field(
        default=False,
    )

    validation_fraction: float = pydantic.Field(
        default=0.1,
        gt=0,
        le=1,
    )

    n_iter_no_change: int = pydantic.Field(
        default=10,
        gt=0,
        le=2**32 - 1,
    )

    warm_start: bool = pydantic.Field(
        default=True,
    )

    
    @pydantic.model_validator(mode="after")
    def _set_penalty(self):
        if self.alpha == 0:
            self._penalty = None
        elif self.l1_ratio == 0:
            self._penalty = "l2"
        elif self.l1_ratio == 1:
            self._penalty = "l1"
        else:
            self._penalty = "elasticnet"

        return self

    @pydantic.model_validator(mode="after")
    def _check_n_iter_no_change(self):
        if self.n_iter_no_change > self.max_iter:
            raise ValueError("'n_iter_no_change' must be less than 'max_iter'.")

        return self


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


class Criterion(str, Enum):
    AIC = "aic"
    BIC = "bic"


class LassoLarsICSettings(BaseSettings):
    """criterion"""
    criterion: Criterion = pydantic.Field(
        default=Criterion.AIC,
    )

    """noise variance"""
    noise_variance: Optional[float] = pydantic.Field(
        default=None,
        gt=0,
    )

    """fit_intercept"""
    fit_intercept: bool = pydantic.Field(
        default=True,
    )

    """Force positive coefficients"""
    positive: bool = pydantic.Field(
        default=False,
    )


    """epsilon-precision regularization for Cholesky diagonal factors"""
    eps: float = pydantic.Field(
        default=1E-6, #np.finfo(float).eps,
        gt=0,
    )

    """copy X to prevent overwriting"""
    copy_x: bool = pydantic.Field(
        default=True,
    )

    """maximum number of iterations"""
    max_iter: int = pydantic.Field(
        default=1000,
        ge=1,
    )

    @pydantic.model_validator(mode="after")
    def _check_positive(self):
        if self.positive:
            if self.fit_intercept:
                raise ValueError(
                    "'fit_intercept' must be False if 'positive' is True."
                )
        return self


class AdaptiveDaysSettings(BaseSettings):
    """Adaptive Daily Weights for ElasticNet"""
    enabled: bool = pydantic.Field(
        default=True,
    )

    """Number of iterations to iterate weights"""
    max_iter: Optional[int] = pydantic.Field(
        default=25,   # Previously was using 100 as it exits early where appropriate
        ge=1,
    )

    """Relative difference in weights to stop iteration"""
    tol: Optional[float] = pydantic.Field(
        default=1E-4,   # Previously was using 1e-4
        ge=0,
    )

    @pydantic.model_validator(mode="after")
    def _check_adaptive_weights(self):
        if self.enabled:
            if self.max_iter is None:
                raise ValueError(
                    "'max_iter' must be specified if 'adaptive_weights' is True."
                )
            if self.tol is None:
                raise ValueError(
                    "'tol' must be specified if 'adaptive_weights' is True."
                )
        else:
            if self.max_iter is not None:
                raise ValueError(
                    "'max_iter' must be None if 'adaptive_weights' is False."
                )
            if self.tol is not None:
                raise ValueError(
                    "'tol' must be None if 'adaptive_weights' is False."
                )

        return self


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
        default=0.109947,
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

    """SGDRegressor settings"""
    sgd_regressor: Optional[SGDSettings] = pydantic.Field(
        default_factory=SGDSettings,
    )

    """Kernel Ridge settings"""
    kernel_ridge: Optional[KernelRidgeSettings] = pydantic.Field(
        default_factory=KernelRidgeSettings,
    )

    """LassoLarsIC settings"""
    lasso_lars: Optional[LassoLarsICSettings] = pydantic.Field(
        default_factory=LassoLarsICSettings,
    )

    """adaptive days settings"""
    adaptive_weighted_days: AdaptiveDaysSettings = pydantic.Field(
        default_factory=AdaptiveDaysSettings,
    )

    """Feature scaling method"""
    scaling_method: ScalingChoice = pydantic.Field(
        default=ScalingChoice.STANDARD_SCALER,
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
        self.sgd_regressor._seed = self._seed

        return self

    @pydantic.model_validator(mode="after")
    def _check_adaptive_sgdregressor(self):
        if (self.base_model == BaseModel.SGDREGRESSOR) and self.adaptive_weighted_days.enabled:
            raise ValueError("'adaptive_weighted_days' must be False if 'base_model' is 'sgdregressor'.")

        return self

    @pydantic.model_validator(mode="after")
    def _remove_unselected_model_settings(self):
        self.model_config["frozen"] = False
        
        if self.base_model == BaseModel.ELASTICNET:
            self.sgd_regressor = None
            self.kernel_ridge = None
            self.lasso_lars = None
        elif self.base_model == BaseModel.SGDREGRESSOR:
            self.kernel_ridge = None
            self.lasso_lars = None
        elif self.base_model == BaseModel.KERNEL_RIDGE:
            self.elasticnet = None
            self.sgd_regressor = None
            self.lasso_lars = None
        elif self.base_model == BaseModel.LASSO_LARS:
            self.elasticnet = None
            self.sgd_regressor = None
            self.kernel_ridge = None

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
