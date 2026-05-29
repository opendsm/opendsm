from __future__ import annotations

import warnings

import numpy as np

import pydantic

from enum import Enum
from typing import Literal

import pywt

from opendsm.common.base_settings import BaseSettings
from opendsm.common.clustering.transform.normalize_settings import NormalizeSettings


class TransformChoice(str, Enum):
    FPCA = "fpca"
    WAVELET = "wavelet"


class WaveletSelection(str, Enum):
    # Daubechies — filter length 2N; use DB1–DB6 for 24h (T=24), DB8+ for 168h+
    HAAR    = "haar"     # alias for db1
    DB1     = "db1"      # length 2;  piecewise-constant
    DB2     = "db2"      # length 4
    DB3     = "db3"      # length 6
    DB4     = "db4"      # length 8
    DB6     = "db6"      # length 12; good balance for smooth 24h profiles
    DB8     = "db8"      # length 16
    DB10    = "db10"     # length 20
    DB12    = "db12"     # length 24; max safe length for T=24
    DB16    = "db16"     # length 32
    DB26    = "db26"     # length 52
    DB29    = "db29"     # length 58
    # Symlets — near-symmetric; filter length 2N; same length guidance as DB
    SYM2    = "sym2"     # length 4
    SYM4    = "sym4"     # length 8
    SYM6    = "sym6"     # length 12
    SYM8    = "sym8"     # length 16
    SYM10   = "sym10"    # length 20
    SYM11   = "sym11"    # length 22
    # Coiflets — near-symmetric; filter length 6N
    COIF1   = "coif1"    # length 6
    COIF2   = "coif2"    # length 12
    COIF3   = "coif3"    # length 18
    COIF4   = "coif4"    # length 24; max safe for T=24
    COIF6   = "coif6"    # length 36
    COIF17  = "coif17"   # length 102; best error/speed mix for 504h+
    # Biorthogonal / reverse biorthogonal
    BIOR1_1 = "bior1.1"
    RBIO1_1 = "rbio1.1"
    # Discrete Meyer — good frequency localisation
    DMEY    = "dmey"


class MagnitudeFeature(str, Enum):
    MEAN = "mean"
    STDEV = "stdev"
    MEDIAN = "median"
    MAD = "mad"
    QUANTILE_RANGE = "quantile_range"
    BASELOAD = "baseload"
    PEAK = "peak"


class FeatureMagnitudeSettings(BaseSettings):
    """Magnitude descriptors computed from original data pre-normalization/transform.

    These features are automatically enabled when centering normalization is
    used (standardize, med_mad, min_max_quantile) and disabled otherwise.
    Not user-configurable — the gate logic is in ``transform.py``.
    """

    features: list[MagnitudeFeature] = pydantic.Field(
        default=[
            MagnitudeFeature.MEDIAN,
            MagnitudeFeature.QUANTILE_RANGE,
            MagnitudeFeature.BASELOAD,
            MagnitudeFeature.PEAK,
        ],
    )

    quantile_range_q: float = pydantic.Field(
        default=0.1,
        gt=0.0,
        lt=0.5,
    )

    baseload_q: float = pydantic.Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
    )

    peak_q: float = pydantic.Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
    )

    weight: float = pydantic.Field(
        default=1.00,
        ge=0.0,
        description=(
            "Balance factor for magnitude vs shape features. "
            "Actual per-feature scale is sqrt(n_shape / n_mag) * weight. "
            "1.0 = equal total variance; >1 emphasizes level; <1 emphasizes shape."
        ),
    )


class fPCATransformSettings(BaseSettings):
    """fPCA transform configuration."""

    enabled: bool = pydantic.Field(
        default=False,
        description="Enable fPCA transform",
    )

    """explained variance ratio for fPCA clustering"""
    min_var_ratio: float = pydantic.Field(
        default=0.97,
        ge=0.5,
        le=1.0,
    )

    """use parallel analysis instead of variance-ratio threshold for n_components"""
    use_parallel_analysis: bool = pydantic.Field(
        default=True,
    )


class WaveletTransformSettings(BaseSettings):
    """Wavelet transform configuration."""

    enabled: bool = pydantic.Field(
        default=True,
        description="Enable wavelet transform",
    )

    """wavelet decomposition level"""
    wavelet_n_levels: int | None = pydantic.Field(
        default=None,
        ge=1,
    )

    """wavelet choice for wavelet decomposition"""
    wavelet_name: WaveletSelection = pydantic.Field(
        default=WaveletSelection.DB1,
    )

    """signal extension mode for wavelet decomposition"""
    wavelet_mode: str = pydantic.Field(
        default="smooth",
    )

    """PCA scope: 'global' flattens all subbands then applies PCA (legacy);
    'per_level' applies PCA independently to each wavelet level, preserving
    scale separation.  Per-level is recommended — it retains the multi-scale
    structure that the wavelet decomposition was designed to capture."""
    pca_scope: Literal["global", "per_level"] = pydantic.Field(
        default="per_level",
    )

    """When True and pca_scope='per_level', scale each level's post-normalize
    features by sqrt(level_variance / total_variance) before concatenation.
    Preserves multi-scale separation while preventing noise amplification
    in low-energy levels."""
    variance_weighted: bool = pydantic.Field(
        default=True,
    )

    """minimum variance ratio for PCA clustering"""
    pca_min_variance_ratio_explained: float | None = pydantic.Field(
        default=None,
    )

    """number of components to keep for PCA clustering"""
    pca_n_components: int | Literal["mle", "parallel_analysis"] | None = pydantic.Field(
        default="parallel_analysis",
    )

    """seed for random state assignment"""
    seed: int | None = pydantic.Field(
        default=None,
        ge=0,
    )

    _seed: int | None = pydantic.PrivateAttr(
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

        if self.pca_min_variance_ratio_explained is not None:
            if not 0.5 <= self.pca_min_variance_ratio_explained <= 1:
                raise ValueError(
                    "'pca_min_variance_ratio_explained' must be between 0.5 and 1"
                )

        return self


class FeatureTransformSettings(BaseSettings):
    """Feature preprocessing and transformation pipeline."""

    normalize: NormalizeSettings = pydantic.Field(
        default_factory=NormalizeSettings,
    )

    fpca: fPCATransformSettings = pydantic.Field(
        default_factory=fPCATransformSettings,
    )

    wavelet: WaveletTransformSettings = pydantic.Field(
        default_factory=WaveletTransformSettings,
    )

    magnitude_features: FeatureMagnitudeSettings = pydantic.Field(
        default_factory=FeatureMagnitudeSettings,
    )

    @pydantic.model_validator(mode="after")
    def _check_exclusive_transforms(self):
        if self.fpca.enabled and self.wavelet.enabled:
            raise ValueError(
                "Only one of fpca or wavelet can be enabled at a time"
            )
        return self

    @pydantic.model_validator(mode="after")
    def _warn_magnitude_without_normalization(self):
        default_mag = FeatureMagnitudeSettings()
        is_customized = self.magnitude_features != default_mag
        if is_customized and not self.normalize.enabled:
            warnings.warn(
                "magnitude_features was customized but normalization is disabled. "
                "Magnitude features are only appended when normalization is enabled.",
                UserWarning,
                stacklevel=2,
            )
        return self
