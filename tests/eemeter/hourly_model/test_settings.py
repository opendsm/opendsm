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

import pydantic
import pytest

from opendsm.eemeter.models.hourly.settings import (
    BinningChoice,
    HourlyNonSolarSettings,
    HourlySolarSettings,
    TemperatureBinSettings,
)



def test_default_binning_settings_construct():
    """Defaults use fixed bins and validate cleanly."""
    settings = TemperatureBinSettings()

    assert settings.method == BinningChoice.FIXED_BINS
    assert settings.fixed_bins is not None


def test_equal_sample_count_requires_n_bins():
    """EQUAL_SAMPLE_COUNT without n_bins is rejected."""
    with pytest.raises(pydantic.ValidationError, match="n_bins"):
        TemperatureBinSettings(method=BinningChoice.EQUAL_SAMPLE_COUNT, n_bins=None)


def test_fixed_bins_required_when_method_fixed():
    """FIXED_BINS with no fixed_bins list is rejected."""
    with pytest.raises(pydantic.ValidationError, match="fixed_bins"):
        TemperatureBinSettings(method=BinningChoice.FIXED_BINS, fixed_bins=None)


def test_n_bins_must_be_positive():
    """A non-positive n_bins violates the field bound."""
    with pytest.raises(pydantic.ValidationError):
        TemperatureBinSettings(method=BinningChoice.EQUAL_SAMPLE_COUNT, n_bins=0)


@pytest.mark.parametrize("bad_percent", [0.0, 0.5])
def test_edge_bin_percent_out_of_range_rejected(bad_percent):
    """edge_bin_percent must lie in (0, 0.45]."""
    with pytest.raises(pydantic.ValidationError):
        TemperatureBinSettings(edge_bin_percent=bad_percent)


def test_edge_bin_percent_upper_boundary_accepted():
    """edge_bin_percent at the 0.45 ceiling is accepted."""
    settings = TemperatureBinSettings(edge_bin_percent=0.45)

    assert settings.edge_bin_percent == 0.45


@pytest.mark.parametrize("settings_cls", [HourlyNonSolarSettings, HourlySolarSettings])
def test_temporal_cluster_disables_outlier_removal(settings_cls):
    """Temporal clustering runs without post-clustering outlier removal."""
    settings = settings_cls()

    assert settings.temporal_cluster.outlier_removal_sigma is None, (
        f"expected outlier_removal_sigma None, got "
        f"{settings.temporal_cluster.outlier_removal_sigma}"
    )


@pytest.mark.parametrize("settings_cls", [HourlyNonSolarSettings, HourlySolarSettings])
def test_temporal_cluster_uses_rbf_affinity(settings_cls):
    """Temporal clustering builds its spectral affinity with the RBF kernel."""
    settings = settings_cls()

    affinity = settings.temporal_cluster.spectral.affinity
    message = f"expected spectral affinity 'rbf', got {affinity!r}"
    assert affinity == "rbf", message


@pytest.mark.parametrize("settings_cls", [HourlyNonSolarSettings, HourlySolarSettings])
def test_temporal_cluster_appends_median_magnitude_feature(settings_cls):
    """Temporal clustering appends the profile median as its only magnitude feature."""
    settings = settings_cls()

    features = settings.temporal_cluster.feature_transform.magnitude_features.features
    message = f"expected magnitude features ['median'], got {features!r}"
    assert features == ["median"], message
