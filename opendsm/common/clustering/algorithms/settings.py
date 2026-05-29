from __future__ import annotations

import pydantic

from enum import Enum
from typing import Literal

from opendsm.common.base_settings import BaseSettings

from opendsm.common.clustering.metrics.settings import ScoreSettings, ClusterRangeSettings


class BiKmeansInnerAlgorithms(str, Enum):
    ELKAN = "elkan"
    LLOYD = "lloyd"


class BiKmeansBisectingStrategies(str, Enum):
    BIGGEST_INERTIA = "biggest_inertia"
    LARGEST_CLUSTER = "largest_cluster"


class BisectingKMeansSettings(BaseSettings):
    recluster_count: int = pydantic.Field(
        default=3,
        ge=1,
        description="number of times to recluster",
    )

    internal_recluster_count: int = pydantic.Field(
        default=5,
        ge=1,
        description="number of times to recluster internally",
    )

    inner_algorithm: BiKmeansInnerAlgorithms = pydantic.Field(
        default=BiKmeansInnerAlgorithms.ELKAN,
        description="Inner KMeans algorithm used in bisection",
    )

    bisecting_strategy: BiKmeansBisectingStrategies = pydantic.Field(
        default=BiKmeansBisectingStrategies.LARGEST_CLUSTER,
        description="Bisection strategy",
    )

    n_cluster: ClusterRangeSettings = pydantic.Field(
        default_factory=ClusterRangeSettings
    )

    scoring: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings
    )


class BirchSettings(BaseSettings):
    threshold: float = pydantic.Field(
        default=0.5,
        ge=0,
        description="radius of the subcluster to merge a new sample in",
    )

    branching_factor: int = pydantic.Field(
        default=50,
        ge=1,
        description="maximum number of CF subclusters in each node",
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
    epsilon: float = pydantic.Field(
        default=0.5,
        gt=0,
        description="maximum distance between two samples for one to be considered as in the neighborhood of the other",
    )

    min_samples: int = pydantic.Field(
        default=1, # sklearn default is 5
        ge=1,
        description="minimum number of samples in a neighborhood for a point to be considered as a cluster",
    )

    nearest_neighbors_algorithm: DbscanDistanceAlgorithm = pydantic.Field(
        default=DbscanDistanceAlgorithm.AUTO,
        description="distance algorithm to use for nearest neighbors",
    )

    leaf_size: int | None = pydantic.Field(
        default=30,
        description="leaf size for KDTree or BallTree",
    )

    minkowski_p: float = pydantic.Field(
        default=2,
        ge=1,
        description="Minkowski p-norm distance power",
    )

    scoring: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings
    )


class HdbscanClusterSelectionMethod(str, Enum):
    LEAF = "leaf"
    EXCESS_OF_MASS = "eom"


class HDBSCANSettings(BaseSettings):
    allow_single_cluster: bool = pydantic.Field(
        default=True,
        description="allow single cluster",
    )

    max_cluster_size: int | None = pydantic.Field(
        default=None,
        description="maximum cluster count",
    )

    min_samples: int = pydantic.Field(
        default=1,
        ge=1,
        description="minimum number of samples in a group for it to be considered as a cluster",
    )

    neighborhood_min_samples: int | None = pydantic.Field(
        default=None,
        description="neighborhood size for density estimation; maps to sklearn's min_samples parameter",
    )

    cluster_selection_epsilon: float = pydantic.Field(
        default=0.0,
        ge=0,
        description="clusters below this distance threshold will be merged",
    )

    robust_single_linkage_scaling: float = pydantic.Field(
        default=1.0,
        gt=0,
        description="distance scaling factor for robust single linkage",
    )

    nearest_neighbors_algorithm: DbscanDistanceAlgorithm = pydantic.Field(
        default=DbscanDistanceAlgorithm.AUTO,
        description="distance algorithm to use",
    )

    leaf_size: int | None = pydantic.Field(
        default=40,
        description="leaf size for KDTree or BallTree",
    )

    cluster_selection_method: HdbscanClusterSelectionMethod = pydantic.Field(
        default=HdbscanClusterSelectionMethod.EXCESS_OF_MASS,
        description="cluster selection method",
    )

    scoring: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings
    )


class SpectralEigenSolver(str, Enum):
    ARPACK = "arpack"
    LOBPCG = "lobpcg"
    # AMG = "amg" # disabled due to additional installation requirements

class AffinityMatrixOptions(str, Enum):
    NEAREST_NEIGHBORS = "nearest_neighbors"
    RBF = "rbf"
    CHI2 = "chi2"
    LAPLACIAN = "laplacian"
    SELF_TUNING = "self_tuning"
    DIFFUSION = "diffusion"
    ANISOTROPIC = "anisotropic"

class SpectralAssignLabels(str, Enum):
    KMEANS = "kmeans"
    DISCRETIZE = "discretize"
    CLUSTER_QR = "cluster_qr"

class SpectralSettings(BaseSettings):
    recluster_count: int = pydantic.Field(
        default=1,
        ge=0,
        description="number of times to recluster",
    )

    eigen_solver: SpectralEigenSolver | None = pydantic.Field(
        default=SpectralEigenSolver.ARPACK,
        description="eigen solver to use",
    )

    n_components: int | None = pydantic.Field(
        default=None,
        description="number of eigenvectors to use, defaults to n_clusters",
    )

    affinity: AffinityMatrixOptions = pydantic.Field(
        default=AffinityMatrixOptions.SELF_TUNING,
        description="affinity matrix algorithm to use",
    )

    nearest_neighbors: int = pydantic.Field(
        default=5,
        ge=1,
        description="number of nearest neighbors to use for nearest neighbors kernel",
    )

    gamma: float = pydantic.Field(
        default=1.05,
        gt=0,
        description="gamma for RBF, polynomial, sigmoid, laplacian, and chi2 kernels",
    )

    eigen_tol: float | Literal["auto"] = pydantic.Field(
        default="auto",
        description="stopping criterion for eigen decomposition",
    )

    assign_labels: SpectralAssignLabels = pydantic.Field(
        default=SpectralAssignLabels.CLUSTER_QR,
        description="label assignment method",
    )

    n_cluster: ClusterRangeSettings = pydantic.Field(
        default_factory=ClusterRangeSettings
    )

    scoring: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings
    )

    eigengap_weight: float = pydantic.Field(
        default=1.0,
        ge=0,
        description="weight for eigengap heuristic voter in k-selection (spectral-specific)",
    )

    diffusion_time: int | None = pydantic.Field(
        default=None,
        description=(
            "Number of diffusion steps for affinity='diffusion'. Higher values "
            "reveal coarser, more global cluster structure by allowing probability "
            "to flow through chains of similar points. t=1 is equivalent to "
            "self_tuning. None (default) auto-selects t from the spectral gap: "
            "t = ceil(log(0.01) / log(λ_below_gap)), clamped to [2, 10]. "
            "Only used when affinity='diffusion'."
        ),
    )

    local_scale_neighbors: int = pydantic.Field(
        default=7,
        ge=1,
        description=(
            "number of nearest neighbors used to compute per-point local scale σ_i for "
            "self-tuning affinity (Zelnik-Manor & Perona 2004). σ_i is the distance from "
            "point i to its local_scale_neighbors-th nearest neighbor. Only used when "
            "affinity='self_tuning'. Default k=7 follows the original paper; optimal value "
            "for energy profile data is not yet benchmarked — see ROADMAP."
        ),
    )

    diffusion_alpha: float = pydantic.Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Coifman & Lafon (2006) alpha-normalization for diffusion maps. "
            "Controls density dependence: alpha=1.0 removes all density influence "
            "(standard normalized Laplacian), alpha=0.5 (default) preserves some "
            "density information for varying-density clusters, alpha=0.0 is "
            "unnormalized. Only used when affinity='diffusion'."
        ),
    )

    use_pic: bool = pydantic.Field(
        default=True,
        description=(
            "Use power iteration clustering (Lin & Cohen 2010) instead of eigsh "
            "for computing the Fiedler vector in per-split bisection. Faster for "
            "large sparse matrices (O(n * nnz_per_row * iter) vs O(n * k²)). "
            "Slightly less accurate due to deflation approximation. "
            "Only affects the per-split Fiedler path; embedding paths are unaffected."
        ),
    )

    split_lambda2_threshold: float = pydantic.Field(
        default=1.0,
        gt=0,
        le=1.0,
        description=(
            "Stop bisecting when the best remaining split has λ₂ ≥ this value. "
            "λ₂ ≈ 0 means near-disconnected (natural split); λ₂ → 1 means "
            "well-connected (forced split). Default 1.0 disables the check. "
            "Used only by spectral_divisive; ignored by spectral."
        ),
    )

    nystrom_samples: int | None = pydantic.Field(
        default=50_000,
        ge=100,
        description=(
            "Nyström approximation threshold for eigsh.  When a sub-cluster "
            "exceeds this size during Fiedler bisection, the eigenvectors are "
            "approximated: compute exact eigsh on a random sample of this size, "
            "then extend to all points via the Nyström formula.  None disables "
            "the approximation (always exact).  Default 50,000."
        ),
    )

    refinement_enabled: bool = pydantic.Field(
        default=True,
        description=(
            "Apply k-medians refinement after greedy bisection to allow points "
            "to migrate between clusters. Corrects suboptimal early splits."
        ),
    )

    refinement_max_iter: int = pydantic.Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Maximum iterations for k-medians refinement. Convergence is checked "
            "each iteration; the loop exits early if no points change assignment."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _check_self_tuning_gamma(self):
        if self.affinity in (AffinityMatrixOptions.SELF_TUNING, AffinityMatrixOptions.DIFFUSION,
                             AffinityMatrixOptions.ANISOTROPIC):
            gamma_default = type(self).model_fields["gamma"].default
            if self.gamma != gamma_default:
                raise ValueError(
                    "gamma has no effect when affinity='self_tuning'. "
                    "Self-tuning affinity derives per-point scale automatically "
                    "from the local_scale_neighbors-th nearest neighbor distance — "
                    "there is no global scale parameter. "
                    "Remove the gamma argument or use affinity='rbf' to apply a global scale."
                )

        return self

    @pydantic.model_validator(mode="after")
    def _check_eigen_tol(self):
        if self.eigen_tol != "auto":
            if self.eigen_tol < 0:
                raise ValueError(
                    "'eigen_tol' must be >= 0"
                )

        return self

    @pydantic.model_validator(mode="after")
    def _check_diffusion_time(self):
        if self.diffusion_time is not None:
            if self.diffusion_time < 1 or self.diffusion_time > 20:
                raise ValueError(
                    "diffusion_time must be between 1 and 20, or None for auto-selection"
                )
        return self


class KMediansSettings(BaseSettings):
    """Direct KMedians clustering configuration.

    Runs KMedians at each k independently with KMeans++ initialization,
    producing balanced partitions for the scoring council.
    """

    recluster_count: int = pydantic.Field(
        default=3,
        ge=0,
        description="Number of outer restarts with different seeds. "
        "Default 3 (4 total runs). Outer restarts add seed diversity "
        "beyond what mixed init strategies provide within each run.",
    )

    n_init: int = pydantic.Field(
        default=5,
        ge=1,
        description="Total restarts per k. Default 5 uses mixed init "
        "(1 farthest-first, 1 bisecting, 2 KMeans++, 1 random).",
    )

    max_iter: int = pydantic.Field(
        default=30,
        ge=1,
        description="Max KMedians iterations per restart.",
    )

    early_stop_inits: bool = pydantic.Field(
        default=True,
        description="Stop restarts early when consecutive inits converge "
        "to the same inertia (within 1%%). Saves ~30%% on average.",
    )

    adaptive_n_init: bool = pydantic.Field(
        default=False,
        description="Reduce restarts for high k where the solution space "
        "is constrained. k=2-4: full n_init, k=5-12: ceil(n_init*0.6), "
        "k=13+: ceil(n_init*0.4). Can miss good partitions at moderate k "
        "on some data. Enable for large k_upper (>50) where speed matters.",
    )

    n_cluster: ClusterRangeSettings = pydantic.Field(
        default_factory=ClusterRangeSettings,
    )

    scoring: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings,
    )


class BisectingKMediansSettings(BaseSettings):
    """Bisecting KMedians clustering configuration (legacy).

    Top-down bisection where each split uses KMedians(k=2) with KMeans++
    initialization. Produces partitions at every k for the scoring council.
    Prefer KMediansSettings (algorithm_selection="kmedians") for balanced
    partitions without catch-all clusters.
    """

    recluster_count: int = pydantic.Field(
        default=3,
        ge=0,
        description="Number of outer restarts with different seeds.",
    )

    n_init: int = pydantic.Field(
        default=10,
        ge=1,
        description="Number of KMeans++ restarts per bisection split.",
    )

    max_iter: int = pydantic.Field(
        default=30,
        ge=1,
        description="Max KMedians iterations per restart.",
    )

    bisecting_strategy: BiKmeansBisectingStrategies = pydantic.Field(
        default=BiKmeansBisectingStrategies.BIGGEST_INERTIA,
        description=(
            "Which cluster to split next. 'biggest_inertia' splits the "
            "most spread-out cluster first (avoids catch-all clusters). "
            "'largest_cluster' splits the biggest cluster first."
        ),
    )

    refinement_enabled: bool = pydantic.Field(
        default=True,
        description=(
            "Apply global k-medians refinement after each bisection step "
            "to allow points to migrate between clusters."
        ),
    )

    refinement_max_iter: int = pydantic.Field(
        default=10,
        ge=1,
        le=100,
        description="Max iterations for global k-medians refinement.",
    )

    n_cluster: ClusterRangeSettings = pydantic.Field(
        default_factory=ClusterRangeSettings,
    )

    scoring: ScoreSettings = pydantic.Field(
        default_factory=ScoreSettings,
    )


class SortMethod(str, Enum):
    SIZE = "size"
    PEAK = "peak"
    # VARIANCE = "variance"


class AggregateMethod(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"


class ClusterSortSettings(BaseSettings):
    enable: bool = pydantic.Field(
        default=True,
        description="enable cluster sorting",
    )

    method: SortMethod = pydantic.Field(
        default=SortMethod.SIZE,
        description="sort method",
    )

    reverse: bool = pydantic.Field(
        default=False,
        description="sort order",
    )


class ClusterAlgorithms(str, Enum):
    KMEDIANS = "kmedians"
    BISECTING_KMEDIANS = "bisecting_kmedians"
    BISECTING_KMEANS = "bisecting_kmeans"
    BIRCH = "birch"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"
    SPECTRAL = "spectral"
    SPECTRAL_DIVISIVE = "spectral_divisive"
