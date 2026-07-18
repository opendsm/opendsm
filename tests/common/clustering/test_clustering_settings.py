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

"""Validator and derived-state tests for the clustering settings models."""

import pytest

from pydantic import ValidationError

from opendsm.common.stats.basic import MAD_k
from opendsm.common.clustering.settings import ClusteringSettings
from opendsm.common.clustering.metrics.settings import (
    ClusterRangeSettings,
    ScoreSettings,
)
from opendsm.common.clustering.algorithms.settings import (
    DBSCANSettings,
    SpectralSettings,
)
from opendsm.common.clustering.transform.normalize_settings import NormalizeSettings



class TestClusterRange:
    """Cluster-count bounds and ordering."""

    def test_default_constructs(self):
        """Defaults (1, 24) construct without error."""
        cr = ClusterRangeSettings()
        assert cr.lower == 1
        assert cr.upper == 24

    def test_lower_above_upper_raises(self):
        """lower > upper is rejected."""
        with pytest.raises(ValidationError, match="must be <="):
            ClusterRangeSettings(lower=5, upper=3)

    @pytest.mark.parametrize("field", ["lower", "upper"])
    def test_zero_bound_raises(self, field):
        """Each bound must be >= 1."""
        with pytest.raises(ValidationError):
            ClusterRangeSettings(**{field: 0})


class TestClusteringSettings:
    """Top-level settings: enums, bounds, derived state, coupling."""

    def test_default_constructs(self):
        """Default construction succeeds."""
        cs = ClusteringSettings()
        assert cs.algorithm_selection.value == "kmedians"

    def test_bad_algorithm_enum_raises(self):
        """An unknown algorithm name is rejected."""
        with pytest.raises(ValidationError):
            ClusteringSettings(algorithm_selection="not_an_algorithm")

    def test_min_cluster_size_below_one_raises(self):
        """min_cluster_size must be >= 1."""
        with pytest.raises(ValidationError):
            ClusteringSettings(min_cluster_size=0)

    def test_outlier_sigma_zero_raises(self):
        """outlier_removal_sigma must be > 0 when set."""
        with pytest.raises(ValidationError):
            ClusteringSettings(outlier_removal_sigma=0.0)

    def test_outlier_sigma_none_disables_threshold(self):
        """None outlier sigma leaves the derived MAD threshold unset."""
        cs = ClusteringSettings(outlier_removal_sigma=None)
        assert cs._outlier_mad_threshold is None

    def test_outlier_sigma_value_sets_threshold(self):
        """A finite sigma is converted to a MAD threshold (sigma / MAD_k)."""
        cs = ClusteringSettings(outlier_removal_sigma=3.0)
        assert cs._outlier_mad_threshold == pytest.approx(3.0 / MAD_k)

    def test_explicit_seed_is_deterministic(self):
        """An explicit seed is copied verbatim into the private seed."""
        cs = ClusteringSettings(seed=123)
        assert cs._seed == 123

    def test_none_seed_draws_random(self):
        """seed=None draws a concrete random seed that varies per instance."""
        a = ClusteringSettings(seed=None)
        b = ClusteringSettings(seed=None)
        assert a._seed is not None
        assert a._seed != b._seed

    def test_seed_propagates_to_transforms(self):
        """The resolved seed is pushed onto the fpca and wavelet sub-settings."""
        cs = ClusteringSettings(seed=777)
        assert cs.feature_transform.fpca._seed == 777
        assert cs.feature_transform.wavelet._seed == 777

    def test_unselected_algorithms_nulled(self):
        """Only the selected algorithm keeps its sub-settings; others are None."""
        cs = ClusteringSettings(algorithm_selection="dbscan")
        assert cs.dbscan is not None
        assert cs.kmedians is None
        assert cs.spectral is None

    def test_min_cluster_size_keep_coupling_raises(self):
        """min_cluster_size >= 2 with KEEP mode is contradictory."""
        with pytest.raises(ValidationError, match="min_cluster_size=1"):
            ClusteringSettings(min_cluster_size=3, small_cluster_mode="keep")

    def test_outlier_mode_requires_min_cluster_size(self):
        """small_cluster_mode='outlier' with min_cluster_size=1 is contradictory."""
        with pytest.raises(ValidationError, match="small_cluster_mode='keep'"):
            ClusteringSettings(min_cluster_size=1, small_cluster_mode="outlier")


class TestSpectralSettings:
    """Spectral-specific cross-field validators and bounds."""

    @pytest.mark.parametrize("affinity", ["self_tuning", "diffusion", "anisotropic"])
    def test_gamma_with_scale_free_affinity_raises(self, affinity):
        """A non-default gamma has no meaning for scale-free affinities."""
        with pytest.raises(ValidationError, match="gamma has no effect"):
            SpectralSettings(affinity=affinity, gamma=2.0)

    def test_negative_eigen_tol_raises(self):
        """A negative eigen_tol is rejected; 'auto' is accepted."""
        with pytest.raises(ValidationError, match="eigen_tol"):
            SpectralSettings(eigen_tol=-0.1)
        assert SpectralSettings(eigen_tol="auto").eigen_tol == "auto"

    @pytest.mark.parametrize("bad_time", [0, 21])
    def test_diffusion_time_out_of_range_raises(self, bad_time):
        """diffusion_time must lie in [1, 20]."""
        with pytest.raises(ValidationError, match="diffusion_time"):
            SpectralSettings(diffusion_time=bad_time)

    @pytest.mark.parametrize("bad_alpha", [-0.1, 1.1])
    def test_diffusion_alpha_out_of_range_raises(self, bad_alpha):
        """diffusion_alpha is bounded to [0, 1]."""
        with pytest.raises(ValidationError):
            SpectralSettings(diffusion_alpha=bad_alpha)


class TestDBSCANSettings:
    """DBSCAN parameter bounds."""

    def test_epsilon_zero_raises(self):
        """epsilon must be strictly positive."""
        with pytest.raises(ValidationError):
            DBSCANSettings(epsilon=0.0)

    def test_min_samples_zero_raises(self):
        """min_samples must be >= 1."""
        with pytest.raises(ValidationError):
            DBSCANSettings(min_samples=0)


class TestScoreSettings:
    """Scoring-weight and alpha validators."""

    def test_unknown_metric_raises(self):
        """An unrecognised index name in the weights is rejected."""
        with pytest.raises(ValidationError, match="Unknown metric"):
            ScoreSettings(weights={"not_a_real_index": 1.0})

    def test_all_zero_weights_raises(self):
        """At least one weight must be positive."""
        with pytest.raises(ValidationError, match="greater than 0"):
            ScoreSettings(weights={"silhouette_index": 0.0})

    def test_negative_weight_raises(self):
        """Negative weights are rejected."""
        with pytest.raises(ValidationError):
            ScoreSettings(weights={"silhouette_index": -1.0})

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0])
    def test_null_test_alpha_bounds(self, bad_alpha):
        """null_test_alpha must be strictly inside (0, 1)."""
        with pytest.raises(ValidationError):
            ScoreSettings(null_test_alpha=bad_alpha)


class TestNormalizeSettings:
    """Normalization cross-field validators and derived axis."""

    def test_min_max_quantile_requires_quantile(self):
        """MIN_MAX_QUANTILE without a quantile is rejected."""
        with pytest.raises(ValidationError, match="quantile"):
            NormalizeSettings(method="min_max_quantile", quantile=None)

    def test_enabled_requires_method(self):
        """Enabling normalization without a method is rejected."""
        with pytest.raises(ValidationError, match="method"):
            NormalizeSettings(enabled=True, method=None)

    @pytest.mark.parametrize("scope,expected_axis", [("sample", 1), ("global", None)])
    def test_scope_sets_axis(self, scope, expected_axis):
        """scope maps to the private normalization axis."""
        ns = NormalizeSettings(scope=scope)
        assert ns._axis == expected_axis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
