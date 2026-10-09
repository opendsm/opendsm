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

import pytest

from opendsm.comparison_groups.savings.settings import (
    CGCorrectionSettings,
    CorrectionAlgorithm,
    CorrectionCapSettings,
    CorrectionCapChoice,
    NDIDSettings,
    OutlierRejectionSettings,
    WeightClusterAggChoice,
)



def test_correction_cap_defaults_are_valid():
    cap = CorrectionCapSettings()

    assert cap.type == CorrectionCapChoice.SOLAR
    assert cap.solar_threshold is not None


def test_solar_cap_requires_threshold():
    with pytest.raises(ValueError):
        CorrectionCapSettings(type=CorrectionCapChoice.SOLAR, solar_threshold=None)


def test_global_cap_rejects_solar_threshold():
    with pytest.raises(ValueError):
        CorrectionCapSettings(type=CorrectionCapChoice.GLOBAL)


def test_global_cap_with_null_threshold_is_valid():
    cap = CorrectionCapSettings(type=CorrectionCapChoice.GLOBAL, solar_threshold=None)

    assert cap.solar_threshold is None


def test_alpha_out_of_range_rejected():
    with pytest.raises(ValueError):
        CGCorrectionSettings(alpha=1.5)


def test_outlier_quantile_out_of_range_rejected():
    with pytest.raises(ValueError):
        OutlierRejectionSettings(quantile=0.6)


def test_weight_cluster_aggregation_defaults_to_model():
    settings = CGCorrectionSettings()

    assert settings.weight_cluster_aggregation == WeightClusterAggChoice.MODEL


def test_weight_cap_defaults_to_one_half():
    settings = CGCorrectionSettings()

    assert settings.weight_cap == 0.5


def test_weight_cap_rejects_zero():
    with pytest.raises(ValueError):
        CGCorrectionSettings(weight_cap=0.0)


def test_weight_cap_rejects_above_one():
    with pytest.raises(ValueError):
        CGCorrectionSettings(weight_cap=1.1)


def test_weight_cap_accepts_upper_bound():
    settings = CGCorrectionSettings(weight_cap=1.0)

    assert settings.weight_cap == 1.0


def test_default_algorithm_is_abspctdid_without_ndid():
    settings = CGCorrectionSettings()

    assert settings.algorithm == CorrectionAlgorithm.ABSPCTDID
    assert settings.ndid is None


def test_ndid_settings_defaults():
    ndid = NDIDSettings(sector="commercial")

    assert ndid.state_bandwidth_factor == 1.0
    assert ndid.size_bandwidth is None
    assert ndid.gamma is None
    assert ndid.cluster_pseudo_count == 20.0


@pytest.mark.parametrize(
    "field, value",
    [
        ("gamma", 0.0),
        ("gamma", 0.5),
        ("cluster_pseudo_count", 0.0),
        ("state_bandwidth_factor", 1e-9),
        ("size_bandwidth", 1e-9),
    ],
)
def test_ndid_settings_accepts_bound(field, value):
    ndid = NDIDSettings(sector="residential", **{field: value})

    assert getattr(ndid, field) == value


@pytest.mark.parametrize(
    "field, value",
    [
        ("gamma", -1e-9),
        ("gamma", 0.5 + 1e-9),
        ("cluster_pseudo_count", -1e-9),
        ("state_bandwidth_factor", 0.0),
        ("size_bandwidth", 0.0),
    ],
)
def test_ndid_settings_rejects_out_of_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        NDIDSettings(sector="residential", **{field: value})


def test_ndid_settings_requires_sector_when_gamma_unset():
    with pytest.raises(ValueError, match="sector"):
        NDIDSettings()


def test_ndid_settings_sector_optional_when_gamma_set():
    ndid = NDIDSettings(gamma=0.25)

    assert ndid.sector is None


def test_ndid_settings_rejects_unknown_sector():
    with pytest.raises(ValueError):
        NDIDSettings(sector="industrial")


def test_ndid_algorithm_requires_ndid_settings():
    with pytest.raises(ValueError, match="'ndid' must be specified"):
        CGCorrectionSettings(algorithm=CorrectionAlgorithm.NDID)


@pytest.mark.parametrize(
    "algorithm",
    [a for a in CorrectionAlgorithm if a != CorrectionAlgorithm.NDID],
)
def test_ndid_settings_rejected_with_other_algorithm(algorithm):
    with pytest.raises(ValueError, match="'ndid' should only be specified"):
        CGCorrectionSettings(algorithm=algorithm, ndid=NDIDSettings(gamma=0.0))


@pytest.mark.parametrize(
    "field, value",
    [
        ("weight_cluster_aggregation", None),
        ("weight_cap", 0.25),
        ("outlier_rejection", OutlierRejectionSettings(enabled=True)),
        ("correction_cap", CorrectionCapSettings(enabled=False)),
    ],
)
def test_ndid_rejects_non_default_legacy_field(field, value):
    with pytest.raises(ValueError, match=field):
        CGCorrectionSettings(
            algorithm=CorrectionAlgorithm.NDID,
            ndid=NDIDSettings(gamma=0.0),
            **{field: value},
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("weight_cluster_aggregation", WeightClusterAggChoice.MODEL),
        ("weight_cap", 0.5),
        ("outlier_rejection", OutlierRejectionSettings()),
        ("correction_cap", CorrectionCapSettings()),
    ],
)
def test_ndid_accepts_legacy_field_at_its_default(field, value):
    settings = CGCorrectionSettings(
        algorithm=CorrectionAlgorithm.NDID,
        ndid=NDIDSettings(gamma=0.0),
        **{field: value},
    )

    assert settings.algorithm == CorrectionAlgorithm.NDID


def test_ndid_keeps_alpha_and_min_window_coverage():
    settings = CGCorrectionSettings(
        algorithm=CorrectionAlgorithm.NDID,
        ndid=NDIDSettings(gamma=0.0),
        alpha=0.05,
        min_window_coverage=0.8,
    )

    assert settings.alpha == 0.05
    assert settings.min_window_coverage == 0.8


def test_ndid_settings_roundtrip_through_json_dump():
    ndid = NDIDSettings(
        state_bandwidth_factor=2.0,
        size_bandwidth=0.7,
        gamma=0.25,
        sector="residential",
        cluster_pseudo_count=5.0,
    )

    restored = NDIDSettings(**ndid.model_dump(mode="json"))

    assert restored == ndid


def test_ndid_correction_settings_roundtrip_through_set_fields_dump():
    settings = CGCorrectionSettings(
        algorithm=CorrectionAlgorithm.NDID,
        ndid=NDIDSettings(gamma=0.25, size_bandwidth=0.7),
        alpha=0.05,
    )

    dumped = settings.model_dump(mode="json", exclude_unset=True)
    restored = CGCorrectionSettings(**dumped)

    assert dumped["algorithm"] == "normalized_difference_in_differences"
    assert restored == settings


def test_ndid_correction_settings_roundtrip_through_full_dump():
    settings = CGCorrectionSettings(
        algorithm=CorrectionAlgorithm.NDID,
        ndid=NDIDSettings(gamma=0.25),
    )

    restored = CGCorrectionSettings(**settings.model_dump(mode="json"))

    assert restored == settings
