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


from opendsm.eemeter.models.daily.model import DailyModel

__all__ = ("metered_savings", "modeled_savings")


def metered_savings(
    baseline_model,
    reporting_meter_data,
    temperature_data,
    with_disaggregated=False,
    confidence_level=0.90,
    predict_kwargs=None,
    degc: bool = False,
    billing_data: bool = False,
):
    """Compute metered savings, i.e., savings in which the baseline model
    is used to calculate the modeled usage in the reporting period. This
    modeled usage is then compared to the actual usage from the reporting period.
    Also compute two measures of the uncertainty of the aggregate savings estimate,
    a fractional savings uncertainty (FSU) error band and an OLS error band. (To convert
    the FSU error band into FSU, divide by total estimated savings.)

    Parameters
    ----------
    baseline_model : :any:`eemeter.CalTRACKUsagePerDayModelResults`
        Object to use for predicting pre-intervention usage.
    reporting_meter_data : :any:`pandas.DataFrame`
        The observed reporting period data (totals). Savings will be computed for the
        periods supplied in the reporting period data.
    temperature_data : :any:`pandas.Series`
        Hourly-frequency timeseries of temperature data during the reporting
        period.
    with_disaggregated : :any:`bool`, optional
        If True, calculate baseline counterfactual disaggregated usage
        estimates. Savings cannot be disaggregated for metered savings. For
        that, use :any:`eemeter.modeled_savings`.
    confidence_level : :any:`float`, optional
        The two-tailed confidence level used to calculate the t-statistic used
        in calculation of the error bands.

        Ignored if not computing error bands.
    predict_kwargs : :any:`dict`, optional
        Extra kwargs to pass to the baseline_model.predict method.
    degc : :any 'bool'
        Relevant temperature units; defaults to False (i.e. Fahrenheit).

    Returns
    -------
    results : :any:`pandas.DataFrame`
        DataFrame with metered savings, indexed with
        ``reporting_meter_data.index``. Will include the following columns:

        - ``counterfactual_usage`` (baseline model projected into reporting period)
        - ``reporting_observed`` (given by reporting_meter_data)
        - ``metered_savings``

        If `with_disaggregated` is set to True, the following columns will also
        be in the results DataFrame:

        - ``counterfactual_base_load``
        - ``counterfactual_heating_load``
        - ``counterfactual_cooling_load``

    error_bands : ``None``
        Always ``None``.
    """
    if degc == True:
        temperature_data = 32 + (temperature_data * 1.8)

    if predict_kwargs is None:
        predict_kwargs = {}

    model_type = None
    if isinstance(baseline_model, DailyModel):
        raise NotImplementedError(
            "Use predict() with daily and billing models to compute metered savings."
        )

    prediction_index = reporting_meter_data.index
    model_prediction = baseline_model.predict(
        prediction_index, temperature_data, **predict_kwargs
    )

    predicted_baseline_usage = model_prediction.result

    # CalTrack 3.5.1
    counterfactual_usage = predicted_baseline_usage["predicted_usage"].to_frame(
        "counterfactual_usage"
    )

    reporting_observed = reporting_meter_data["value"].to_frame("reporting_observed")

    def metered_savings_func(row):
        return row.counterfactual_usage - row.reporting_observed

    results = reporting_observed.join(counterfactual_usage).assign(
        metered_savings=metered_savings_func
    )

    results = results.dropna().reindex(results.index)  # carry NaNs

    return results, None


def modeled_savings(
    baseline_model,
    reporting_model,
    result_index,
    temperature_data,
    with_disaggregated=False,
    confidence_level=0.90,
    predict_kwargs=None,
    degc: bool = False,
):
    """Compute modeled savings, i.e., savings in which baseline and reporting
    usage values are based on models. This is appropriate for annualizing or
    weather normalizing models.

    Parameters
    ----------
    baseline_model : :any:`eemeter.CalTRACKUsagePerDayCandidateModel`
        Model to use for predicting pre-intervention usage.
    reporting_model : :any:`eemeter.CalTRACKUsagePerDayCandidateModel`
        Model to use for predicting post-intervention usage.
    result_index : :any:`pandas.DatetimeIndex`
        The dates for which usage should be modeled.
    temperature_data : :any:`pandas.Series`
        Hourly-frequency timeseries of temperature data during the modeled
        period.
    with_disaggregated : :any:`bool`, optional
        If True, calculate modeled disaggregated usage estimates and savings.
    confidence_level : :any:`float`, optional
        The two-tailed confidence level used to calculate the t-statistic used
        in calculation of the error bands.

        Ignored if not computing error bands.
    predict_kwargs : :any:`dict`, optional
        Extra kwargs to pass to the baseline_model.predict and
        reporting_model.predict methods.
    degc : :any 'bool'
        Relevant temperature units; defaults to False (i.e. Fahrenheit).


    Returns
    -------
    results : :any:`pandas.DataFrame`
        DataFrame with modeled savings, indexed with the result_index. Will
        include the following columns:

        - ``modeled_baseline_usage``
        - ``modeled_reporting_usage``
        - ``modeled_savings``

        If `with_disaggregated` is set to True, the following columns will also
        be in the results DataFrame:

        - ``modeled_baseline_base_load``
        - ``modeled_baseline_cooling_load``
        - ``modeled_baseline_heating_load``
        - ``modeled_reporting_base_load``
        - ``modeled_reporting_cooling_load``
        - ``modeled_reporting_heating_load``
        - ``modeled_base_load_savings``
        - ``modeled_cooling_load_savings``
        - ``modeled_heating_load_savings``
    error_bands : ``None``
        Always ``None``.
    """
    if degc == True:
        temperature_data = 32 + (temperature_data * 1.8)

    prediction_index = result_index

    if predict_kwargs is None:
        predict_kwargs = {}

    model_type = None  # generic
    if isinstance(baseline_model, DailyModel) or isinstance(
        reporting_model, DailyModel
    ):
        raise NotImplementedError(
            "Use predict() with daily and billing models to compute modeled savings."
        )

    def _predicted_usage(model):
        model_prediction = model.predict(
            prediction_index, temperature_data, **predict_kwargs
        )
        predicted_usage = model_prediction.result
        return predicted_usage

    predicted_baseline_usage = _predicted_usage(baseline_model)
    predicted_reporting_usage = _predicted_usage(reporting_model)
    modeled_baseline_usage = predicted_baseline_usage["predicted_usage"].to_frame(
        "modeled_baseline_usage"
    )
    modeled_reporting_usage = predicted_reporting_usage["predicted_usage"].to_frame(
        "modeled_reporting_usage"
    )

    def modeled_savings_func(row):
        return row.modeled_baseline_usage - row.modeled_reporting_usage

    results = modeled_baseline_usage.join(modeled_reporting_usage).assign(
        modeled_savings=modeled_savings_func
    )

    results = results.dropna().reindex(results.index)  # carry NaNs

    return results, None
