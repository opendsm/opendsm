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

from scipy.stats import t

from opendsm.eemeter.models.daily.model import DailyModel

__all__ = ("metered_savings", "modeled_savings")


def _compute_ols_error(
    t_stat,
    rmse_base_residuals,
    post_obs,
    base_obs,
    base_avg,
    post_avg,
    base_var,
    nprime,
):
    ols_model_agg_error = (
        (t_stat * rmse_base_residuals * post_obs)
        / (base_obs**0.5)
        * (1.0 + ((base_avg - post_avg) ** 2.0 / base_var)) ** 0.5
    )

    ols_noise_agg_error = (
        t_stat * rmse_base_residuals * (post_obs * base_obs / nprime) ** 0.5
    )

    ols_total_agg_error = (ols_model_agg_error**2.0 + ols_noise_agg_error**2.0) ** 0.5

    return ols_total_agg_error, ols_model_agg_error, ols_noise_agg_error


def _compute_fsu_error(
    t_stat,
    interval,
    post_obs,
    total_base_energy,
    rmse_base_residuals,
    base_avg,
    base_obs,
    nprime,
):
    if interval.startswith("billing"):
        a_coeff = -0.00022
        b_coeff = 0.03306
        c_coeff = 0.94054
        months_reporting = float(post_obs)
    else:  # daily
        a_coeff = -0.00024
        b_coeff = 0.03535
        c_coeff = 1.00286
        months_reporting = float(post_obs) / 30.0

    fsu_error_band = total_base_energy * (
        t_stat
        * (a_coeff * months_reporting**2.0 + b_coeff * months_reporting + c_coeff)
        * (rmse_base_residuals / base_avg)
        * ((base_obs / nprime) * (1.0 + (2.0 / nprime)) * (1.0 / post_obs)) ** 0.5
    )

    return fsu_error_band


def _compute_error_bands_metered_savings(
    totals_metrics, results, interval, confidence_level
):
    num_parameters = float(totals_metrics.num_parameters)

    base_obs = float(totals_metrics.observed_length)
    if (interval.startswith("billing")) & (len(results.dropna().index) > 0):
        post_obs = float(round((results.index[-1] - results.index[0]).days / 30.0))
    else:
        post_obs = float(results["reporting_observed"].dropna().shape[0])

    degrees_of_freedom = float(base_obs - num_parameters)
    single_tailed_confidence_level = 1 - ((1 - confidence_level) / 2)
    t_stat = t.ppf(single_tailed_confidence_level, degrees_of_freedom)

    rmse_base_residuals = float(totals_metrics.rmse_adj)
    autocorr_resid = totals_metrics.autocorr_resid

    base_avg = float(totals_metrics.observed_mean)
    post_avg = float(results["reporting_observed"].mean())

    base_var = float(totals_metrics.observed_variance)

    # these result in division by zero error for fsu_error_band
    if (
        post_obs == 0
        or autocorr_resid is None
        or abs(autocorr_resid) == 1
        or base_obs == 0
        or base_avg == 0
        or base_var == 0
    ):
        return None

    autocorr_resid = float(autocorr_resid)

    nprime = float(base_obs * (1 - autocorr_resid) / (1 + autocorr_resid))

    total_base_energy = float(base_avg * base_obs)

    ols_total_agg_error, ols_model_agg_error, ols_noise_agg_error = _compute_ols_error(
        t_stat,
        rmse_base_residuals,
        post_obs,
        base_obs,
        base_avg,
        post_avg,
        base_var,
        nprime,
    )

    fsu_error_band = _compute_fsu_error(
        t_stat,
        interval,
        post_obs,
        total_base_energy,
        rmse_base_residuals,
        base_avg,
        base_obs,
        nprime,
    )

    return {
        "FSU Error Band": fsu_error_band,
        "OLS Error Band": ols_total_agg_error,
        "OLS Error Band: Model Error": ols_model_agg_error,
        "OLS Error Band: Noise": ols_noise_agg_error,
    }


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

    error_bands : :any:`dict`, optional
        If baseline_model is an instance of CalTRACKUsagePerDayModelResults,
        will also return a dictionary of FSU and OLS error bands for the
        aggregated energy savings over the post period.
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

    # compute t-statistic associated with n degrees of freedom
    # and a two-tailed confidence level.
    error_bands = None
    return results, error_bands


def _compute_error_bands_modeled_savings(
    totals_metrics_baseline,
    totals_metrics_reporting,
    results,
    interval_baseline,
    interval_reporting,
    confidence_level,
):
    num_parameters_baseline = float(totals_metrics_baseline.num_parameters)
    num_parameters_reporting = float(totals_metrics_reporting.num_parameters)

    base_obs_baseline = float(totals_metrics_baseline.observed_length)
    base_obs_reporting = float(totals_metrics_reporting.observed_length)

    if (interval_baseline.startswith("billing")) & (len(results.dropna().index) > 0):
        post_obs_baseline = float(
            round((results.index[-1] - results.index[0]).days / 30.0)
        )
    else:
        post_obs_baseline = float(results["modeled_baseline_usage"].dropna().shape[0])

    if (interval_reporting.startswith("billing")) & (len(results.dropna().index) > 0):
        post_obs_reporting = float(
            round((results.index[-1] - results.index[0]).days / 30.0)
        )
    else:
        post_obs_reporting = float(results["modeled_reporting_usage"].dropna().shape[0])

    degrees_of_freedom_baseline = float(base_obs_baseline - num_parameters_baseline)
    degrees_of_freedom_reporting = float(base_obs_reporting - num_parameters_reporting)
    single_tailed_confidence_level = 1 - ((1 - confidence_level) / 2)
    t_stat_baseline = t.ppf(single_tailed_confidence_level, degrees_of_freedom_baseline)
    t_stat_reporting = t.ppf(
        single_tailed_confidence_level, degrees_of_freedom_reporting
    )

    rmse_base_residuals_baseline = float(totals_metrics_baseline.rmse_adj)
    rmse_base_residuals_reporting = float(totals_metrics_reporting.rmse_adj)
    autocorr_resid_baseline = totals_metrics_baseline.autocorr_resid
    autocorr_resid_reporting = totals_metrics_reporting.autocorr_resid

    base_avg_baseline = float(totals_metrics_baseline.observed_mean)
    base_avg_reporting = float(totals_metrics_reporting.observed_mean)

    # these result in division by zero error for fsu_error_band
    if (
        post_obs_baseline == 0
        or autocorr_resid_baseline is None
        or abs(autocorr_resid_baseline) == 1
        or base_obs_baseline == 0
        or base_avg_baseline == 0
        or post_obs_reporting == 0
        or autocorr_resid_reporting is None
        or abs(autocorr_resid_reporting) == 1
        or base_obs_reporting == 0
        or base_avg_reporting == 0
    ):
        return None

    autocorr_resid_baseline = float(autocorr_resid_baseline)
    autocorr_resid_reporting = float(autocorr_resid_reporting)

    nprime_baseline = float(
        base_obs_baseline
        * (1 - autocorr_resid_baseline)
        / (1 + autocorr_resid_baseline)
    )
    nprime_reporting = float(
        base_obs_reporting
        * (1 - autocorr_resid_reporting)
        / (1 + autocorr_resid_reporting)
    )

    total_base_energy_baseline = float(base_avg_baseline * base_obs_baseline)
    total_base_energy_reporting = float(base_avg_reporting * base_obs_reporting)

    fsu_error_band_baseline = _compute_fsu_error(
        t_stat_baseline,
        interval_baseline,
        post_obs_baseline,
        total_base_energy_baseline,
        rmse_base_residuals_baseline,
        base_avg_baseline,
        base_obs_baseline,
        nprime_baseline,
    )

    fsu_error_band_reporting = _compute_fsu_error(
        t_stat_reporting,
        interval_reporting,
        post_obs_reporting,
        total_base_energy_reporting,
        rmse_base_residuals_reporting,
        base_avg_reporting,
        base_obs_reporting,
        nprime_reporting,
    )

    return {
        "FSU Error Band: Baseline": fsu_error_band_baseline,
        "FSU Error Band: Reporting": fsu_error_band_reporting,
        "FSU Error Band": (fsu_error_band_baseline**2.0 + fsu_error_band_reporting**2.0)
        ** 0.5,
    }


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
    error_bands : :any:`dict`, optional
        If baseline_model and reporting_model are instances of
        CalTRACKUsagePerDayModelResults, will also return a dictionary of
        FSU and error bands for the aggregated energy savings over the
        normal year period.
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

    error_bands = None
    return results, error_bands
