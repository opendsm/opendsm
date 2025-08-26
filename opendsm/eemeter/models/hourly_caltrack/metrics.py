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

import numpy as np
import pandas as pd
from scipy.stats import t

from opendsm.eemeter.common.warnings import EEMeterWarning

__all__ = ("ModelMetrics",)


def _compute_r_squared(combined):
    return combined[["predicted", "observed"]].corr().iloc[0, 1] ** 2


def _compute_r_squared_adj(r_squared, length, num_parameters):
    return 1 - (1 - r_squared) * (length - 1) / (length - num_parameters - 1)


def _compute_rmse(combined):
    return (combined["residuals"].astype(float) ** 2).mean() ** 0.5


def _compute_rmse_adj(combined, length, num_parameters):
    if length > num_parameters:
        return (
            (combined["residuals"].astype(float) ** 2).sum() / (length - num_parameters)
        ) ** 0.5
    else:
        return np.nan


def _compute_cvrmse(rmse, observed_mean):
    return rmse / observed_mean


def _compute_cvrmse_adj(rmse_adj, observed_mean):
    return rmse_adj / observed_mean


def _compute_mape(combined):
    return (combined["residuals"] / combined["observed"]).abs().mean()


def _compute_nmae(combined):
    return (combined["residuals"].astype(float).abs().sum()) / (
        combined["observed"].sum()
    )


def _compute_nmbe(combined):
    return combined["residuals"].astype(float).sum() / combined["observed"].sum()


def _compute_autocorr_resid(combined, autocorr_lags):
    return combined["residuals"].autocorr(lag=autocorr_lags)


def _json_safe_float(number):
    """
    JSON serialization for infinity can be problematic.
    See https://docs.python.org/2/library/json.html#basic-usage
    This function returns None if `number` is infinity or negative infinity.

    If the `number` cannot be converted to float, this will raise an exception.
    """
    if number is None:
        return None

    if isinstance(number, float):
        return None if np.isinf(number) or np.isnan(number) else number

    # errors if number is not float compatible
    return float(number)


class ModelMetricsFromJson(object):
    def __init__(
        self,
        observed_length,
        predicted_length,
        merged_length,
        num_parameters,
        observed_mean,
        predicted_mean,
        observed_variance,
        predicted_variance,
        observed_skew,
        predicted_skew,
        observed_kurtosis,
        predicted_kurtosis,
        observed_cvstd,
        predicted_cvstd,
        r_squared,
        r_squared_adj,
        rmse,
        rmse_adj,
        cvrmse,
        cvrmse_adj,
        mape,
        mape_no_zeros,
        num_meter_zeros,
        nmae,
        nmbe,
        autocorr_resid,
        confidence_level,
        n_prime,
        single_tailed_confidence_level,
        degrees_of_freedom,
        t_stat,
        cvrmse_auto_corr_correction,
        approx_factor_auto_corr_correction,
        fsu_base_term,
    ):
        self.observed_length = observed_length
        self.predicted_length = predicted_length
        self.merged_length = merged_length
        self.num_parameters = num_parameters
        self.observed_mean = observed_mean
        self.predicted_mean = predicted_mean
        self.observed_variance = observed_variance
        self.predicted_variance = predicted_variance
        self.observed_skew = observed_skew
        self.predicted_skew = predicted_skew
        self.observed_kurtosis = observed_kurtosis
        self.predicted_kurtosis = predicted_kurtosis
        self.observed_cvstd = observed_cvstd
        self.predicted_cvstd = predicted_cvstd
        self.r_squared = r_squared
        self.r_squared_adj = r_squared_adj
        self.rmse = rmse
        self.rmse_adj = rmse_adj
        self.cvrmse = cvrmse
        self.cvrmse_adj = cvrmse_adj
        self.mape = mape
        self.mape_no_zeros = mape_no_zeros
        self.num_meter_zeros = num_meter_zeros
        self.nmae = nmae
        self.nmbe = nmbe
        self.autocorr_resid = autocorr_resid
        self.confidence_level = confidence_level
        self.n_prime = n_prime
        self.single_tailed_confidence_level = single_tailed_confidence_level
        self.degrees_of_freedom = degrees_of_freedom
        self.t_stat = t_stat
        self.cvrmse_auto_corr_correction = cvrmse_auto_corr_correction
        self.approx_factor_auto_corr_correction = approx_factor_auto_corr_correction
        self.fsu_base_term = fsu_base_term


class ModelMetrics(object):
    """Contains measures of model fit and summary statistics on the input series.

    Parameters
    ----------
    observed_input : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with a set of electricity or
        gas meter values.
    predicted_input : :any:`pandas.Series`
        Series with :any:`pandas.DatetimeIndex` with a set of electricity or
        gas meter values.
    num_parameters : :any:`int`, optional
        The number of parameters (excluding the intercept) used in the
        regression from which the predictions were derived.
    autocorr_lags : :any:`int`, optional
        The number of lags to use when calculating the autocorrelation of the
        residuals.
    confidence_level : :any:`int`, optional
        Confidence level used in fractional savings uncertainty computations.

    Attributes
    ----------
    observed_length : :any:`int`
        The length of the observed_input series.
    predicted_length : :any:`int`
        The length of the predicted_input series.
    merged_length : :any:`int`
        The length of the dataframe resulting from the inner join of the
        observed_input series and the predicted_input series.
    observed_mean : :any:`float`
        The mean of the observed_input series.
    predicted_mean : :any:`float`
        The mean of the predicted_input series.
    observed_skew : :any:`float`
        The skew of the observed_input series.
    predicted_skew : :any:`float`
        The skew of the predicted_input series.
    observed_kurtosis : :any:`float`
        The excess kurtosis of the observed_input series.
    predicted_kurtosis : :any:`float`
        The excess kurtosis of the predicted_input series.
    observed_cvstd : :any:`float`
        The coefficient of standard deviation of the observed_input series.
    predicted_cvstd : :any:`float`
        The coefficient of standard deviation of the predicted_input series.
    r_squared : :any:`float`
        The r-squared of the model from which the predicted_input series was
        produced.
    r_squared_adj : :any:`float`
        The r-squared of the predicted_input series relative to the
        observed_input series, adjusted by the number of parameters in the model.
    cvrmse : :any:`float`
        The coefficient of variation (root-mean-squared error) of the
        predicted_input series relative to the observed_input series.
    cvrmse_adj : :any:`float`
        The coefficient of variation (root-mean-squared error) of the
        predicted_input series relative to the observed_input series, adjusted
        by the number of parameters in the model.
    mape : :any:`float`
        The mean absolute percent error of the predicted_input series relative
        to the observed_input series.
    mape_no_zeros : :any:`float`
        The mean absolute percent error of the predicted_input series relative
        to the observed_input series, with all time periods dropped where the
        observed_input series was not greater than zero.
    num_meter_zeros : :any:`int`
        The number of time periods for which the observed_input series was not
        greater than zero.
    nmae : :any:`float`
        The normalized mean absolute error of the predicted_input series
        relative to the observed_input series.
    nmbe : :any:`float`
        The normalized mean bias error of the predicted_input series relative
        to the observed_input series.
    autocorr_resid : :any:`float`
        The autocorrelation of the residuals (where the residuals equal the
        predicted_input series minus the observed_input series), measured
        using a number of lags equal to autocorr_lags.
    n_prime: :any:`float`
        The number of baseline inputs corrected for autocorrelation -- used
        in fractional savings uncertainty computation.
    single_tailed_confidence_level: :any:`float`
        The adjusted confidence level for use in single-sided tests.
    degrees_of_freedom: :any:`float
        Maxmimum number of independent variables which have the freedom to vary
    t_stat: :any:`float
        t-statistic, used for hypothesis testing
    cvrmse_auto_corr_correction: :any:`float
        Correctoin factor the apply to cvrmse to account for autocorrelation of inputs.
    approx_factor_auto_corr_correction: :any:`float
        Approximation factor used in ashrae 14 guideline for uncertainty computation.
    fsu_base_term: :any:`float
        Base term used in fractional savings uncertainty computation.

    """

    def __init__(
        self,
        observed_input,
        predicted_input,
        num_parameters=1,
        autocorr_lags=1,
        confidence_level=0.90,
    ):
        if num_parameters < 0:
            raise ValueError("num_parameters must be greater than or equal to zero")
        if autocorr_lags <= 0:
            raise ValueError("autocorr_lags must be greater than zero")
        if (confidence_level <= 0) or (confidence_level >= 1):
            raise ValueError("confidence_level must be between zero and one.")

        self.warnings = []

        observed = observed_input.to_frame().dropna()
        predicted = predicted_input.to_frame().dropna()
        observed.columns = ["observed"]
        predicted.columns = ["predicted"]

        self.observed_length = observed.shape[0]
        self.predicted_length = predicted.shape[0]

        # Do an inner join on the two input series to make sure that we only
        # use observations with the same time stamps.
        combined = observed.merge(predicted, left_index=True, right_index=True)
        self.merged_length = len(combined)

        if self.observed_length != self.predicted_length:
            self.warnings.append(
                EEMeterWarning(
                    qualified_name="eemeter.metrics.input_series_are_of_different_lengths",
                    description="Input series are of different lengths.",
                    data={
                        "observed_input_length": len(observed_input),
                        "predicted_input_length": len(predicted_input),
                        "observed_length_without_nan": self.observed_length,
                        "predicted_length_without_nan": self.predicted_length,
                        "merged_length": self.merged_length,
                    },
                )
            )

        # Calculate residuals because these are an input for most of the metrics.
        combined["residuals"] = combined.predicted - combined.observed

        self.num_parameters = num_parameters
        self.autocorr_lags = autocorr_lags

        # to account for solar usage the cvrmse should be calculated as the
        # mean of the absolute value of observed.

        self.observed_mean = abs(combined["observed"]).mean()
        self.predicted_mean = abs(combined["predicted"]).mean()

        self.observed_variance = combined["observed"].var(ddof=0)
        self.predicted_variance = combined["predicted"].var(ddof=0)

        self.observed_skew = combined["observed"].skew()
        self.predicted_skew = combined["predicted"].skew()

        self.observed_kurtosis = combined["observed"].kurtosis()
        self.predicted_kurtosis = combined["predicted"].kurtosis()

        self.r_squared = _compute_r_squared(combined)
        self.r_squared_adj = _compute_r_squared_adj(
            self.r_squared, self.merged_length, self.num_parameters
        )

        self.rmse = _compute_rmse(combined)
        self.rmse_adj = _compute_rmse_adj(
            combined, self.merged_length, self.num_parameters
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            self.observed_cvstd = combined["observed"].std() / self.observed_mean
            self.predicted_cvstd = combined["predicted"].std() / self.predicted_mean

            self.cvrmse = _compute_cvrmse(self.rmse, self.observed_mean)
            self.cvrmse_adj = _compute_cvrmse_adj(self.rmse_adj, self.observed_mean)

        # Create a new DataFrame with all rows removed where observed is
        # zero, so we can calculate a version of MAPE with the zeros excluded.
        # (Without the zeros excluded, MAPE becomes infinite when one observed
        # value is zero.)
        no_observed_zeros = combined[combined["observed"] > 0]

        self.mape = _compute_mape(combined)
        self.mape_no_zeros = _compute_mape(no_observed_zeros)

        self.num_meter_zeros = (self.merged_length) - no_observed_zeros.shape[0]

        self.nmae = _compute_nmae(combined)

        self.nmbe = _compute_nmbe(combined)

        self.autocorr_resid = _compute_autocorr_resid(combined, autocorr_lags)

        # ** Compute terms needed for fractional savings uncertainty computation.

        self.confidence_level = confidence_level
        self.n_prime = float(
            self.observed_length * (1 - self.autocorr_resid) / (1 + self.autocorr_resid)
        )
        self.single_tailed_confidence_level = 1 - ((1 - self.confidence_level) / 2)

        # convert to integer degrees of freedom, because n_prime could be non-integer
        if pd.isnull(self.n_prime) or not np.isfinite(self.n_prime):
            self.degrees_of_freedom = None
            self.t_stat = None
        else:
            self.degrees_of_freedom = round(self.n_prime - self.num_parameters)
            self.t_stat = t.ppf(
                self.single_tailed_confidence_level, self.degrees_of_freedom
            )

        if (
            self.n_prime == 0
            or pd.isnull(self.n_prime)
            or not np.isfinite(self.n_prime)
            or self.n_prime - self.num_parameters == 0
            or self.degrees_of_freedom < 1
            or self.observed_length < self.num_parameters
        ):
            self.cvrmse_auto_corr_correction = None
            self.approx_factor_auto_corr_correction = None
            self.fsu_base_term = None
        else:
            # factor to correct cvrmse_adj for autocorrelation of inputs
            # i.e., divide by (n' - n_param) instead of by (n - n_param)
            self.cvrmse_auto_corr_correction = (
                (self.observed_length - self.num_parameters)
                / (self.n_prime - self.num_parameters)
            ) ** 0.5

            # part of approximation factor used in ashrae 14 guideline
            self.approx_factor_auto_corr_correction = (
                1.0 + (2.0 / self.n_prime)
            ) ** 0.5

            # all the following values are unitless
            self.fsu_base_term = (
                self.t_stat
                * self.cvrmse_adj
                * self.cvrmse_auto_corr_correction
                * self.approx_factor_auto_corr_correction
            )

    def __repr__(self):
        return (
            "ModelMetrics(merged_length={}, r_squared_adj={}, cvrmse_adj={}, "
            "mape_no_zeros={}, nmae={}, nmbe={}, autocorr_resid={}, confidence_level={})".format(
                self.merged_length,
                round(self.r_squared_adj, 3),
                round(self.cvrmse_adj, 3),
                round(self.mape_no_zeros, 3),
                round(self.nmae, 3),
                round(self.nmbe, 3),
                round(self.autocorr_resid, 3),
                round(self.confidence_level, 3),
            )
        )

    def json(self):
        """Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """
        return {
            "observed_length": _json_safe_float(self.observed_length),
            "predicted_length": _json_safe_float(self.predicted_length),
            "merged_length": _json_safe_float(self.merged_length),
            "num_parameters": _json_safe_float(self.num_parameters),
            "observed_mean": _json_safe_float(self.observed_mean),
            "predicted_mean": _json_safe_float(self.predicted_mean),
            "observed_variance": _json_safe_float(self.observed_variance),
            "predicted_variance": _json_safe_float(self.predicted_variance),
            "observed_skew": _json_safe_float(self.observed_skew),
            "predicted_skew": _json_safe_float(self.predicted_skew),
            "observed_kurtosis": _json_safe_float(self.observed_kurtosis),
            "predicted_kurtosis": _json_safe_float(self.predicted_kurtosis),
            "observed_cvstd": _json_safe_float(self.observed_cvstd),
            "predicted_cvstd": _json_safe_float(self.predicted_cvstd),
            "r_squared": _json_safe_float(self.r_squared),
            "r_squared_adj": _json_safe_float(self.r_squared_adj),
            "rmse": _json_safe_float(self.rmse),
            "rmse_adj": _json_safe_float(self.rmse_adj),
            "cvrmse": _json_safe_float(self.cvrmse),
            "cvrmse_adj": _json_safe_float(self.cvrmse_adj),
            "mape": _json_safe_float(self.mape),
            "mape_no_zeros": _json_safe_float(self.mape_no_zeros),
            "num_meter_zeros": _json_safe_float(self.num_meter_zeros),
            "nmae": _json_safe_float(self.nmae),
            "nmbe": _json_safe_float(self.nmbe),
            "autocorr_resid": _json_safe_float(self.autocorr_resid),
            "confidence_level": _json_safe_float(self.confidence_level),
            "n_prime": _json_safe_float(self.n_prime),
            "single_tailed_confidence_level": _json_safe_float(
                self.single_tailed_confidence_level
            ),
            "degrees_of_freedom": _json_safe_float(self.degrees_of_freedom),
            "t_stat": _json_safe_float(self.t_stat),
            "cvrmse_auto_corr_correction": _json_safe_float(
                self.cvrmse_auto_corr_correction
            ),
            "approx_factor_auto_corr_correction": _json_safe_float(
                self.approx_factor_auto_corr_correction
            ),
            "fsu_base_term": _json_safe_float(self.fsu_base_term),
        }

    @classmethod
    def from_json(cls, data):
        """Loads a JSON-serializable representation into the model state.

        The input of this function is a dict which can be the result
        of :any:`json.loads`.
        """

        c = ModelMetricsFromJson(
            observed_length=data.get("observed_length"),
            predicted_length=data.get("predicted_length"),
            merged_length=data.get("merged_length"),
            num_parameters=data.get("num_parameters"),
            observed_mean=data.get("observed_mean"),
            predicted_mean=data.get("predicted_mean"),
            observed_variance=data.get("observed_variance"),
            predicted_variance=data.get("predicted_variance"),
            observed_skew=data.get("observed_skew"),
            predicted_skew=data.get("predicted_skew"),
            observed_kurtosis=data.get("observed_kurtosis"),
            predicted_kurtosis=data.get("predicted_kurtosis"),
            observed_cvstd=data.get("observed_cvstd"),
            predicted_cvstd=data.get("predicted_cvstd"),
            r_squared=data.get("r_squared"),
            r_squared_adj=data.get("r_squared_adj"),
            rmse=data.get("rmse"),
            rmse_adj=data.get("rmse_adj"),
            cvrmse=data.get("cvrmse"),
            cvrmse_adj=data.get("cvrmse_adj"),
            mape=data.get("mape"),
            mape_no_zeros=data.get("mape_no_zeros"),
            num_meter_zeros=data.get("num_meter_zeros"),
            nmae=data.get("nmae"),
            nmbe=data.get("nmbe"),
            autocorr_resid=data.get("autocorr_resid"),
            confidence_level=data.get("confidence_level"),
            n_prime=data.get("n_prime"),
            single_tailed_confidence_level=data.get("single_tailed_confidence_level"),
            degrees_of_freedom=data.get("degrees_of_freedom"),
            t_stat=data.get("t_stat"),
            cvrmse_auto_corr_correction=data.get("cvrmse_auto_corr_correction"),
            approx_factor_auto_corr_correction=data.get(
                "approx_factor_auto_corr_correction"
            ),
            fsu_base_term=data.get("fsu_base_term"),
        )

        return c
