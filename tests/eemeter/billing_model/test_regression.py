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
"""Regression tests pinning BillingModel fit & predict outputs."""

import pytest

from opendsm.eemeter.models.billing.model import BillingModel

from eemeter.perturbations import bimonthly_reads, estimated_read, offcycle_read
from regression_metrics import regression_block



# scenario -> (perturbation, {is_electricity_data: qualified names of the disqualifications
# on the DEFAULT meter's monthly frame}); an empty set means the scenario stays qualified.
BILLING_SCHEDULE_SCENARIOS = {
    "offcycle_read": (
        offcycle_read,
        {
            True: {"eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data"},
            False: {"eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data"},
        },
    ),
    "bimonthly_reads": (
        bimonthly_reads,
        {
            True: {
                "eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
            },
            False: {
                "eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
            },
        },
    ),
    "estimated_read": (
        estimated_read,
        {
            True: {
                "eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data",
                "eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data",
            },
            False: set(),
        },
    ),
}


@pytest.fixture(scope="session")
def billing_baseline_df(comstock_monthly_diverse):
    df_b, _ = comstock_monthly_diverse

    return df_b.reset_index()


@pytest.fixture(scope="session")
def billing_reporting_df(comstock_monthly_diverse):
    _, df_r = comstock_monthly_diverse

    return df_r.reset_index()


@pytest.fixture(scope="session")
def billing_model_fit(billing_baseline_df):
    model = BillingModel().fit(
        billing_baseline_df, is_electricity_data=True, enforce_compliance=False
    )

    return model


@pytest.mark.slow
@pytest.mark.regression
def test_billing_baseline_predict_regression(
    billing_model_fit, billing_baseline_df, snapshot
):
    result = billing_model_fit.predict(billing_baseline_df)

    assert regression_block(result, freq="daily") == snapshot(name="regression")


@pytest.mark.slow
@pytest.mark.regression
def test_billing_reporting_predict_regression(
    billing_model_fit, billing_reporting_df, snapshot
):
    result = billing_model_fit.predict(billing_reporting_df)

    assert regression_block(result, freq="daily") == snapshot(name="regression")


IS_ELECTRICITY_BY_FUEL = {"electricity": True, "gas": False}


@pytest.fixture(
    scope="session",
    params=[
        (scenario, fuel)
        for scenario in BILLING_SCHEDULE_SCENARIOS
        for fuel in IS_ELECTRICITY_BY_FUEL
    ],
    ids="-".join,
)
def billing_schedule_scenario_fit(request, comstock_monthly):
    """Fit the DEFAULT meter's monthly baseline, perturbed by one schedule scenario, with model
    and settings compliance enforced; return the expected disqualifications, the model and the
    unperturbed reporting frame.
    """
    scenario_name, fuel = request.param
    is_electricity_data = IS_ELECTRICITY_BY_FUEL[fuel]
    scenario, expected_by_fuel = BILLING_SCHEDULE_SCENARIOS[scenario_name]
    baseline, reporting = comstock_monthly

    model = BillingModel().fit(
        scenario(baseline).reset_index(),
        is_electricity_data=is_electricity_data,
        enforce_compliance=["model", "settings"],
    )
    expected = expected_by_fuel[is_electricity_data]
    reporting = reporting.reset_index()

    return expected, model, reporting


@pytest.mark.slow
@pytest.mark.regression
def test_billing_schedule_scenario_disqualifications(billing_schedule_scenario_fit):
    """A schedule scenario disqualifies the fit exactly by its table entry."""
    expected, model, _ = billing_schedule_scenario_fit
    actual = {entry.qualified_name for entry in model.compliance.disqualification}

    assert actual == expected
    assert model.compliance.qualified == (not expected)


@pytest.mark.slow
@pytest.mark.regression
def test_billing_schedule_scenario_prediction(billing_schedule_scenario_fit, snapshot):
    """Predictions on the unperturbed reporting frame match the snapshot."""
    _, model, reporting = billing_schedule_scenario_fit
    result = model.predict(reporting, enforce_compliance=["model", "settings"])

    assert regression_block(result, freq="daily") == snapshot(name="regression")
