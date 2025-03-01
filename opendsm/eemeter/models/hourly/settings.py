from __future__ import annotations

import numpy as np
import pandas as pd

import pydantic

from enum import Enum
from typing import Optional, Literal, Union, TypeVar, Dict

import pywt

from opendsm.common.base_settings import BaseSettings
from opendsm.common.metrics import BaselineMetrics

from opendsm.eemeter.common.warnings import EEMeterWarning

# from opendsm.common.const import CountryCode


class SelectionChoice(str, Enum):
    CYCLIC = "cyclic"
    RANDOM = "random"


class ScalingChoice(str, Enum):
    ROBUSTSCALER = "robustscaler"
    STANDARDSCALER = "standardscaler"


class BinningChoice(str, Enum):
    EQUAL_SAMPLE_COUNT = "equal_sample_count"
    EQUAL_BIN_WIDTH = "equal_bin_width"
    SET_BIN_WIDTH = "set_bin_width"


class ClusterScoringMetric(str, Enum):
    SILHOUETTE = "silhouette"
    SILHOUETTE_MEDIAN = "silhouette_median"
    VARIANCE_RATIO = "variance_ratio"
    DAVIES_BOULDIN = "davies-bouldin"


class DistanceMetric(str, Enum):
    """
    what distance method to use
    """

    EUCLIDEAN = "euclidean"
    SEUCLIDEAN = "seuclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"


class DefaultTrainingFeatures(str, Enum):
    SOLAR = ["temperature", "ghi"]
    NONSOLAR = ["temperature"]


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
        default=12,
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
        default=0.0425,
        ge=0,
        le=0.45,
    )

    """offset normalized temperature range for edge bins (keeps exp from blowing up)"""
    edge_bin_temperature_range_offset: Optional[float] = pydantic.Field(
        default=1.0,
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


class TemporalClusteringSettings(BaseSettings):
    """wavelet decomposition level"""

    wavelet_n_levels: int = pydantic.Field(
        default=5,
        ge=1,
        # le=5,  #TODO investigate upper limit
    )

    """wavelet choice for wavelet decomposition"""
    wavelet_name: str = pydantic.Field(
        default="haar",  # maybe db3?
    )

    """signal extension mode for wavelet decomposition"""
    wavelet_mode: str = pydantic.Field(
        default="periodization",
    )

    """minimum variance ratio for PCA clustering"""
    pca_min_variance_ratio_explained: float = pydantic.Field(
        default=0.725,
        ge=0.5,
        le=1,
    )

    """number of times to recluster"""
    recluster_count: int = pydantic.Field(
        default=3,
        ge=1,
    )

    """lower bound for number of clusters"""
    n_cluster_lower: int = pydantic.Field(
        default=12,
        ge=2,
    )

    """upper bound for number of clusters"""
    n_cluster_upper: int = pydantic.Field(
        default=24,
        ge=2,
    )

    """minimum cluster size"""
    min_cluster_size: int = pydantic.Field(
        default=1,
        ge=1,
    )

    """scoring method for clustering"""
    score_metric: ClusterScoringMetric = pydantic.Field(
        default=ClusterScoringMetric.VARIANCE_RATIO,
    )

    """distance metric for clustering"""
    distance_metric: DistanceMetric = pydantic.Field(
        default=DistanceMetric.EUCLIDEAN,
    )

    @pydantic.model_validator(mode="after")
    def _check_wavelet(self):
        all_wavelets = pywt.wavelist(kind="discrete")
        if self.wavelet_name not in all_wavelets:
            raise ValueError(
                f"'wavelet_name' must be a valid wavelet in PyWavelets: \n{all_wavelets}"
            )

        all_modes = pywt.Modes.modes
        if self.wavelet_mode not in all_modes:
            raise ValueError(
                f"'wavelet_mode' must be a valid mode in PyWavelets: \n{all_modes}"
            )

        return self


class ElasticNetSettings(BaseSettings):
    """ElasticNet alpha parameter"""

    alpha: float = pydantic.Field(
        default=0.0425,
        ge=0,
    )

    """ElasticNet l1_ratio parameter"""
    l1_ratio: float = pydantic.Field(
        default=0.5,
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
        default=1000,
        ge=1,
        le=2**32 - 1,
    )

    """ElasticNet copy_X parameter"""
    copy_x: bool = pydantic.Field(
        default=True,
    )

    """ElasticNet tol parameter"""
    tol: float = pydantic.Field(
        default=1e-4,
        gt=0,
    )

    """ElasticNet selection parameter"""
    selection: SelectionChoice = pydantic.Field(
        default=SelectionChoice.CYCLIC,
    )

    """Adaptive Daily Weights for ElasticNet"""
    adaptive_weights: bool = pydantic.Field(
        default=False,
    )

    """Number of iterations to iterate weights"""
    adaptive_weight_max_iter: Optional[int] = pydantic.Field(
        default=None,   # Previously was using 100 as it exits early where appropriate
        ge=1,
    )

    """Relative difference in weights to stop iteration"""
    adaptive_weight_tol: Optional[float] = pydantic.Field(
        default=None,   # Previously was using 1e-4
        ge=0,
    )

    @pydantic.model_validator(mode="after")
    def _check_adaptive_weights(self):
        if self.adaptive_weights:
            if self.adaptive_weight_max_iter is None:
                raise ValueError(
                    "'adaptive_weight_iter' must be specified if 'adaptive_weights' is True."
                )
            if self.adaptive_weight_tol is None:
                raise ValueError(
                    "'adaptive_weight_tol' must be specified if 'adaptive_weights' is True."
                )
        else:
            if self.adaptive_weight_max_iter is not None:
                raise ValueError(
                    "'adaptive_weight_iter' must be None if 'adaptive_weights' is False."
                )
            if self.adaptive_weight_tol is not None:
                raise ValueError(
                    "'adaptive_weight_tol' must be None if 'adaptive_weights' is False."
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
    temporal_cluster: TemporalClusteringSettings = pydantic.Field(
        default_factory=TemporalClusteringSettings,
    )

    """supplemental time series column names"""
    supplemental_time_series_columns: Optional[list] = pydantic.Field(
        default=None,
    )

    """supplemental categorical column names"""
    supplemental_categorical_columns: Optional[list] = pydantic.Field(
        default=None,
    )

    """ElasticNet settings"""
    elasticnet: ElasticNetSettings = pydantic.Field(
        default_factory=ElasticNetSettings,
    )

    """Feature scaling method"""
    scaling_method: ScalingChoice = pydantic.Field(
        default=ScalingChoice.STANDARDSCALER,
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
