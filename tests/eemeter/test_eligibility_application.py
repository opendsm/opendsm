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

import copy

import numpy as np
import pytest

from opendsm.eemeter.common.compliance import Compliance
from opendsm.eemeter.common.exceptions import (
    DataSufficiencyError,
    NonstandardSettingsError,
)
from opendsm.eemeter.models.billing.model import BillingModel
from opendsm.eemeter.models.daily.model import DailyModel
from opendsm.eemeter.models.hourly.model import HourlyModel



_MODEL_CLASSES = {
    "daily": DailyModel,
    "billing": BillingModel,
    "hourly": HourlyModel,
}

_COMSTOCK_FIXTURES = {
    "daily": "comstock_daily",
    "billing": "comstock_monthly",
    "hourly": "comstock_hourly",
}


@pytest.fixture(scope="module")
def baseline_model_hourly(comstock_hourly):
    """The hourly model fit once on the ComStock baseline meter; tests read it and never
    mutate it."""
    df_b, _ = comstock_hourly
    model = HourlyModel().fit(df_b.copy(), is_electricity_data=True, enforce_compliance=False)

    return model


@pytest.fixture(params=list(_MODEL_CLASSES))
def model_name(request):
    return request.param


@pytest.fixture
def fitted_model(model_name, request):
    """The shared fitted model for `model_name`; copy it before mutating."""
    model = request.getfixturevalue(f"baseline_model_{model_name}")

    return model


@pytest.fixture
def comstock_frames(model_name, request):
    """The ComStock baseline and reporting frames at `model_name`'s frequency."""
    frames = request.getfixturevalue(_COMSTOCK_FIXTURES[model_name])

    return frames


def _plain(entries) -> list[dict]:
    """Compliance records as JSON-ready mappings."""
    records = [
        {
            "qualified_name": entry.qualified_name,
            "severity": entry.severity,
            "member": entry.member,
            "data": entry.data,
        }
        for entry in entries
    ]

    return records


def _one_step_past(threshold: float) -> float:
    step = float(np.nextafter(threshold, np.inf))

    return step


# Fit-metric check


def _fit_metric_outcome(model, cvrmse_adj: float, pnrmse_adj: float) -> dict:
    """Run the model's fit-metric check on a copy of `model` whose baseline metrics carry
    these values; `model` is left unchanged."""
    checked = copy.copy(model)
    checked.compliance = Compliance()
    checked.baseline_metrics = copy.copy(model.baseline_metrics)
    # the metrics are cached computed fields, read from the instance dict; pydantic 2.4 drops
    # an attribute assignment to them, so the values go into the dict as a loaded model's do
    checked.baseline_metrics.__dict__.update(cvrmse_adj=cvrmse_adj, pnrmse_adj=pnrmse_adj)
    checked._check_model_fit()
    outcome = {
        "passed": checked.compliance.qualified,
        "records": _plain(checked.compliance.all),
    }

    return outcome


@pytest.mark.slow
def test_fit_metric_check_passes_while_either_metric_is_within_its_threshold(
    fitted_model, snapshot
):
    """The fit-metric check passes when `cvrmse_adj` or `pnrmse_adj` is at most its default
    threshold, and disqualifies the model only when both are one step past."""
    cvrmse_threshold = fitted_model.settings.cvrmse_threshold
    pnrmse_threshold = fitted_model.settings.pnrmse_threshold
    cvrmse_past = _one_step_past(cvrmse_threshold)
    pnrmse_past = _one_step_past(pnrmse_threshold)

    outcomes = {
        "both_at_threshold": _fit_metric_outcome(
            fitted_model, cvrmse_threshold, pnrmse_threshold
        ),
        "cvrmse_past_threshold": _fit_metric_outcome(
            fitted_model, cvrmse_past, pnrmse_threshold
        ),
        "pnrmse_past_threshold": _fit_metric_outcome(
            fitted_model, cvrmse_threshold, pnrmse_past
        ),
        "both_past_threshold": _fit_metric_outcome(fitted_model, cvrmse_past, pnrmse_past),
    }

    assert outcomes == snapshot


# Timezone check


def _prediction_outcome(model, df) -> dict:
    """Predict on `df` with data compliance enforced, returning whether the prediction is
    allowed and the reporting data's compliance records."""
    data = model._reporting_data(df)
    try:
        model._predict_data(data, enforce_compliance=["data"])
    except DataSufficiencyError:
        passed = False
    else:
        passed = True

    outcome = {"passed": passed, "records": _plain(data.compliance.all)}

    return outcome


@pytest.mark.slow
def test_timezone_check_refuses_reporting_data_in_another_timezone(
    fitted_model, comstock_frames, snapshot
):
    """Prediction passes on reporting data in the model's timezone and is refused on the same
    data in another timezone, which carries a timezone-mismatch record. The mismatched frame
    keeps the local wall-clock timestamps so the mismatch is its only disqualification."""
    _, df_r = comstock_frames
    wall_clock_in_utc = df_r.tz_localize(None).tz_localize("UTC")

    outcomes = {
        "matching": _prediction_outcome(fitted_model, df_r.copy()),
        "mismatched": _prediction_outcome(fitted_model, wall_clock_in_utc),
    }

    assert outcomes == snapshot


# Settings-deviation check


def _settings_fit_outcome(model, df) -> dict:
    """Fit `model` on `df` with settings compliance enforced, returning whether the fit is
    allowed and its settings records."""
    try:
        model.fit(df, is_electricity_data=True, enforce_compliance=["settings"])
    except NonstandardSettingsError as error:
        outcome = {"passed": False, "records": _plain(error.entries)}
    else:
        outcome = {
            "passed": True,
            "records": _plain(model.compliance.disqualified_in({"settings"})),
        }

    return outcome


@pytest.mark.slow
def test_settings_deviation_check_refuses_a_changed_developer_setting(
    model_name, comstock_frames, snapshot
):
    """A settings-enforced fit passes with default settings and is refused with a changed
    developer-tier setting, carrying the deviation as a settings record."""
    model_class = _MODEL_CLASSES[model_name]
    df_b, _ = comstock_frames
    changed = {"cvrmse_threshold": 2 * model_class().settings.cvrmse_threshold}

    outcomes = {
        "default": _settings_fit_outcome(model_class(), df_b.copy()),
        "changed": _settings_fit_outcome(model_class(settings=changed), df_b.copy()),
    }

    assert outcomes == snapshot
