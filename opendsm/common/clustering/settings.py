from __future__ import annotations

import numpy as np

import pydantic

from enum import Enum
from typing import Optional, Literal, Union

import pywt

from opendsm.common.base_settings import BaseSettings


class NormalizeChoice(str, Enum):
    MIN_MAX_QUANTILE = "min_max_quantile"
    STANDARDIZE = "standardize"
    MED_MAD = "med_mad"


class NormalizeSettings(BaseSettings):
    """normalization method for data"""
    method: Optional[NormalizeChoice] = pydantic.Field(
        default=NormalizeChoice.STANDARDIZE,
    )

    pre_transform: bool = pydantic.Field(
        default=True,
    )

    post_transform: bool = pydantic.Field(
        default=True,
    )

    quantile: Optional[float] = pydantic.Field(
        default=None,
        gt=0.0,
        lt=0.5,
    )

    axis: Optional[int] = pydantic.Field(
        default=None,
    )

    @pydantic.model_validator(mode="after")
    def _check_quantile(self):
        if self.method == NormalizeChoice.MIN_MAX_QUANTILE:
            if self.quantile is None:
                raise ValueError(
                    "'quantile' must be specified when 'method' is 'min_max_quantile'"
                )
        else:
            if self.quantile is not None:
                raise ValueError(
                    "'quantile' should only be specified when 'method' is 'min_max_quantile'"
                )

        return self

    @pydantic.model_validator(mode="after")
    def _check_enable(self):
        if self.method is None:
            if self.pre_transform or self.post_transform:
                raise ValueError(
                    "'method' cannot be None if 'pre_transform' or 'post_transform' is True"
                )
        else:
            if not (self.pre_transform or self.post_transform):
                raise ValueError(
                    "'pre_transform' or 'post_transform' must be True if 'method' is specified"
                )

        return self


class TransformChoice(str, Enum):
    FPCA = "fpca"
    WAVELET = "wavelet"
    

class fPCATransformSettings(BaseSettings):
    """explained variance ratio for fPCA clustering"""
    min_var_ratio: float = pydantic.Field(
        default=0.97,
        ge=0.5,
        le=1.0,
    )


class PCASelection(str, Enum):
    PCA = "pca"
    KERNEL_PCA = "kernel_pca"


class WaveletSelection(str, Enum):
    BIOR1_1 = "bior1.1"
    COIF6 = "coif6"
    COIF17 = "coif17"    # Best error/speed mix
    DB1 = "db1"          # Best error metrics
    DB16 = "db16"
    DB26 = "db26"
    DB29 = "db29"
    HAAR = "haar"
    RBIO1_1 = "rbio1.1"
    SYM11 = "sym11"


class WaveletTransformSettings(BaseSettings):
    """wavelet decomposition level"""
    wavelet_n_levels: int = pydantic.Field(
        default=4,
        ge=1,
        # le=5,  #TODO investigate upper limit
    )

    """wavelet choice for wavelet decomposition"""
    wavelet_name: WaveletSelection = pydantic.Field(
        default=WaveletSelection.DB1,
    )

    """signal extension mode for wavelet decomposition"""
    wavelet_mode: str = pydantic.Field(
        default="smooth",
    )

    """PCA method"""
    pca_method: PCASelection = pydantic.Field(
        default=PCASelection.PCA,
    )

    """minimum variance ratio for PCA clustering"""
    pca_min_variance_ratio_explained: Optional[float] = pydantic.Field(
        default=None,
    )

    """number of components to keep for PCA clustering"""
    pca_n_components: Optional[Union[int, Literal["mle"]]] = pydantic.Field(
        default="mle",
    )

    """add mean to pca components"""
    pca_include_median: bool = pydantic.Field(
        default=True,
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

    @pydantic.model_validator(mode="after")
    def _check_pca_settings(self):
        if self.pca_n_components is None and self.pca_min_variance_ratio_explained is None:
            raise ValueError(
                "Must specify either 'pca_min_variance_ratio_explained' or 'pca_n_components'"
            )

        if self.pca_n_components is not None:
            if self.pca_min_variance_ratio_explained is not None:
                raise ValueError(
                    "Cannot specify both 'pca_min_variance_ratio_explained' and 'pca_n_components'"
                )
            
            if isinstance(self.pca_n_components, int):
                if self.pca_n_components < 1:
                    raise ValueError(
                        "'pca_n_components' must be >= 1"
                    )

            if (self.pca_n_components == "mle") and (self.pca_method == PCASelection.KERNEL_PCA):
                raise ValueError(
                    "Cannot use 'mle' with 'kernel_pca'"
                )

        if self.pca_min_variance_ratio_explained is not None:
            if not 0.5 <= self.pca_min_variance_ratio_explained <= 1:
                raise ValueError(
                    "'pca_min_variance_ratio_explained' must be between 0.5 and 1"
                )
            
        return self


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

class ScoreSettings(BaseSettings):
    """minimum cluster size"""
    min_cluster_size: int = pydantic.Field(
        default=1,
        ge=1,
    )

    """maximum number of non-outlier clusters"""
    max_non_outlier_cluster_count: int = pydantic.Field(
        default=200,
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


class ClusterRangeSettings(BaseSettings):
    """lower bound for number of clusters"""
    lower: int = pydantic.Field(
        default=2,
        ge=2,
    )

    """upper bound for number of clusters"""
    upper: int = pydantic.Field(
        default=24,
        ge=2,
    )

    @pydantic.model_validator(mode="after")
    def _check_n_cluster_range(self):
        if self.lower > self.upper:
            raise ValueError(
                "'n_cluster_lower' must be <= 'n_cluster_upper'"
            )

        return self


class BiKmeansInnerAlgorithms(str, Enum):
    ELKAN = "elkan"
    LLOYD = "lloyd"


class BiKmeansBisectingStrategies(str, Enum):
    BIGGEST_INERTIA = "biggest_inertia"
    LARGEST_CLUSTER = "largest_cluster"


class BisectingKMeansSettings(BaseSettings):
    """number of times to recluster"""
    recluster_count: int = pydantic.Field(
        default=3,
        ge=1,
    )

    """number of times to recluster internally"""
    internal_recluster_count: int = pydantic.Field(
        default=5,
        ge=1,
    )

    """Inner KMeans algorithm used in bisection"""
    inner_algorithm: BiKmeansInnerAlgorithms = pydantic.Field(
        default=BiKmeansInnerAlgorithms.ELKAN,
    )

    """Bisection strategy"""
    bisecting_strategy: BiKmeansBisectingStrategies = pydantic.Field(
        default=BiKmeansBisectingStrategies.LARGEST_CLUSTER,
    )

    n_cluster: ClusterRangeSettings = pydantic.Field(
        default_factory=ClusterRangeSettings
    )

    scoring: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings
    )

    
class BirchSettings(BaseSettings):
    """radius of the subcluster to merge a new sample in"""
    threshold: float = pydantic.Field(
        default=0.5,
        ge=0,
    )

    """maximum number of CF subclusters in each node"""
    branching_factor: int = pydantic.Field(
        default=50,
        ge=1,
    )

    n_cluster: ClusterRangeSettings = pydantic.Field(
        default_factory=ClusterRangeSettings
    )

    scoring: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings
    )


class DbscanDistanceAlgorithm(str, Enum):
    AUTO = "auto"
    BRUTE = "brute"
    KD_TREE = "kd_tree"
    BALL_TREE = "ball_tree"

class DBSCANSettings(BaseSettings):
    """maximum distance between two samples for one to be considered as in the neighborhood of the other"""
    epsilon: float = pydantic.Field(
        default=0.5,
        gt=0,
    )

    """minimum number of samples in a neighborhood for a point to be considered as a cluster"""
    min_samples: int = pydantic.Field(
        default=1, # sklearn default is 5
        ge=1,
    )

    """distance metric for calculating distance between samples"""
    distance_metric: DistanceMetric = pydantic.Field(
        default=DistanceMetric.EUCLIDEAN,
    )

    """distance algorithm to use for nearest neighbors"""
    nearest_neighbors_algorithm: DbscanDistanceAlgorithm = pydantic.Field(
        default=DbscanDistanceAlgorithm.AUTO,
    )

    """leaf size for KDTree or BallTree"""
    leaf_size: Optional[int] = pydantic.Field(
        default=30,
    )

    """Minkowski p-norm distance power"""
    minkowski_p: float = pydantic.Field(
        default=2,
        ge=1,
    )


class HdbscanClusterSelectionMethod(str, Enum):
    LEAF = "leaf"
    EXCESS_OF_MASS = "eom"


class HDBSCANSettings(BaseSettings):
    """allow single cluster"""
    allow_single_cluster: bool = pydantic.Field(
        default=True,
    )

    """maximum cluster count"""
    max_cluster_size: Optional[int] = pydantic.Field(
        default=None,
    )

    """minimum number of samples in a group for it to be considered as a cluster"""
    min_samples: int = pydantic.Field(
        default=1,
        ge=1,
    )

    """distance metric for calculating distance between samples"""
    distance_metric: DistanceMetric = pydantic.Field(
        default=DistanceMetric.EUCLIDEAN,
    )

    """samples to calculate distance between neighbors"""
    scoring_sample_count: Optional[int] = pydantic.Field(
        default=None,
    )

    """clusters below this distance threshold will be merged"""
    cluster_selection_epsilon: float = pydantic.Field(
        default=0.0,
        ge=0,
    )

    """distance scaling factor for robust single linkage"""
    robust_single_linkage_scaling: float = pydantic.Field(
        default=1.0,
        gt=0,
    )

    """distance algorithm to use"""
    nearest_neighbors_algorithm: DbscanDistanceAlgorithm = pydantic.Field(
        default=DbscanDistanceAlgorithm.AUTO,
    )

    """leaf size for KDTree or BallTree"""
    leaf_size: Optional[int] = pydantic.Field(
        default=40,
    )

    """cluster selection method"""
    cluster_selection_method: HdbscanClusterSelectionMethod = pydantic.Field(
        default=HdbscanClusterSelectionMethod.EXCESS_OF_MASS,
    )


class SpectralEigenSolver(str, Enum):
    ARPACK = "arpack"
    LOBPCG = "lobpcg"
    # AMG = "amg" # disabled due to additional installation requirements

class AffinityMatrixOptions(str, Enum):
    NEAREST_NEIGHBORS = "nearest_neighbors"
    RBF = "rbf"
    ADDITIVE_CHI2 = "additive_chi2"
    CHI2 = "chi2"
    LINEAR = "linear"
    POLY = "poly"
    POLYNOMIAL = "polynomial"
    LAPLACIAN = "laplacian"
    SIGMOID = "sigmoid"
    COSINE = "cosine"

class SpectralAssignLabels(str, Enum):
    KMEANS = "kmeans"
    DISCRETIZE = "discretize"
    CLUSTER_QR = "cluster_qr"
    
class SpectralSettings(BaseSettings):
    """eigen solver to use"""
    eigen_solver: Optional[SpectralEigenSolver] = pydantic.Field(
        default=SpectralEigenSolver.ARPACK,
    )

    """number of eigenvectors to use, defaults to n_clusters"""
    n_components: Optional[int] = pydantic.Field(
        default=None,
    )

    """affinity matrix algorithm to use"""
    affinity: AffinityMatrixOptions = pydantic.Field(
        default=AffinityMatrixOptions.RBF,
    )

    """number of nearest neighbors to use for nearest neighbors kernel"""
    nearest_neighbors: int = pydantic.Field(
        default=5,
        ge=1,
    )

    """gamma for RBF, polynomial, sigmoid, laplacian, and chi2 kernels"""
    gamma: float = pydantic.Field(
        default=1.05,
        ge=0, # could be wrong? maybe gt?
    )

    """stopping criterion for eigen decomposition"""
    eigen_tol: Union[float, Literal["auto"]] = pydantic.Field(
        default="auto",
    )

    """label assignment method"""
    assign_labels: SpectralAssignLabels = pydantic.Field(
        default=SpectralAssignLabels.CLUSTER_QR,
    )

    n_cluster: ClusterRangeSettings = pydantic.Field(
        default_factory=ClusterRangeSettings
    )

    scoring: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings
    )

    @pydantic.model_validator(mode="after")
    def _check_eigen_tol(self):
        if self.eigen_tol != "auto":
            if self.eigen_tol < 0:
                raise ValueError(
                    "'eigen_tol' must be >= 0"
                )

        return self


class SortMethod(str, Enum):
    SIZE = "size"
    PEAK = "peak"
    # VARIANCE = "variance"


class AggregateMethod(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


class ClusterSortSettings(BaseSettings):
    """sort method"""
    method: SortMethod = pydantic.Field(
        default=SortMethod.PEAK,
    )

    """aggregate method"""
    aggregation: AggregateMethod = pydantic.Field(
        default=AggregateMethod.MEAN
    )

    """sort order"""
    reverse: bool = pydantic.Field(
        default=False,
    )


class ClusterAlgorithms(str, Enum):
    BISECTING_KMEANS = "bisecting_kmeans"
    BIRCH = "birch"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    SPECTRAL = "spectral"


class ClusteringSettings(BaseSettings):
    """pretransform data normalization settings"""
    normalize: NormalizeSettings = pydantic.Field(
        default_factory=NormalizeSettings
    )

    """transform method"""
    transform_selection: Optional[TransformChoice] = pydantic.Field(
        default=TransformChoice.WAVELET,
    )

    """fPCA transform settings"""
    fpca_transform: Optional[fPCATransformSettings] = pydantic.Field(
        default_factory=fPCATransformSettings
    )

    """wavelet transform settings"""
    wavelet_transform: Optional[WaveletTransformSettings] = pydantic.Field(
        default_factory=WaveletTransformSettings
    )

    """clustering choice"""
    algorithm_selection: ClusterAlgorithms = pydantic.Field(
        default=ClusterAlgorithms.SPECTRAL,
    )

    """BisectingKMeans settings"""
    bisecting_kmeans: Optional[BisectingKMeansSettings] = pydantic.Field(
        default_factory=BisectingKMeansSettings,
    )

    """Birch settings"""
    birch: Optional[BirchSettings] = pydantic.Field(
        default_factory=BirchSettings,
    )

    """DBSCAN settings"""
    dbscan: Optional[DBSCANSettings] = pydantic.Field(
        default_factory=DBSCANSettings,
    )

    """HDBSCAN settings"""
    hdbscan: Optional[HDBSCANSettings] = pydantic.Field(
        default_factory=HDBSCANSettings,
    )

    """Spectral settings"""
    spectral: Optional[SpectralSettings] = pydantic.Field(
        default_factory=SpectralSettings,
    )

    """sort clusters boolean"""
    sort_clusters: bool = pydantic.Field(
        default=False,
    )

    """sort clusters """
    cluster_sort_options: ClusterSortSettings = pydantic.Field(
        default_factory=ClusterSortSettings,
    )

    """seed for random state assignment"""
    seed: Optional[int] = pydantic.Field(
        default=None,
        ge=0,
    )

    _seed: Optional[int] = pydantic.PrivateAttr(
        default=None
    )

    @pydantic.model_validator(mode="after")
    def _check_seed(self):
        if self.seed is None and self._seed is None:
            self._seed = np.random.randint(0, 2**32 - 1, dtype=np.int64)
        else:
            self._seed = self.seed

        return self

    @pydantic.model_validator(mode="after")
    def _remove_unselected_algorithms(self):
        self.model_config["frozen"] = False

        algo_dict = {
            ClusterAlgorithms.BISECTING_KMEANS: self.bisecting_kmeans,
            ClusterAlgorithms.BIRCH: self.birch,
            ClusterAlgorithms.DBSCAN: self.dbscan,
            ClusterAlgorithms.HDBSCAN: self.hdbscan,
            ClusterAlgorithms.SPECTRAL: self.spectral,
        }

        for k in algo_dict.keys():
            if k != self.algorithm_selection:
                setattr(self, k, None)

        self.model_config["frozen"] = True

        return self

    @pydantic.model_validator(mode="after")
    def _remove_unselected_transform(self):
        self.model_config["frozen"] = False

        transform_dict = {
            TransformChoice.WAVELET: self.wavelet_transform,
            TransformChoice.FPCA: self.fpca_transform,
        }

        for k in transform_dict.keys():
            if k != self.transform_selection:
                setattr(self, f"{k.value}_transform", None)

        self.model_config["frozen"] = True

        return self


if __name__ == "__main__":
    settings = ClusteringSettings()

    print(settings)

    print(settings._algorithm)