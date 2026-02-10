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
from typing import Union, Optional
from enum import Enum
from functools import cached_property

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from opendsm.common.utils import safe_divide
from opendsm.common.stats.basic import (
    median_absolute_deviation,
    t_stat,
)
from opendsm.common.pydantic_utils import (
    ArbitraryPydanticModel,
    PydanticDf,
    PydanticFromDict,
    computed_field_cached_property,
)



_MIN_DENOMINATOR: float = 1e-3


class ColumnMetrics(ArbitraryPydanticModel):
    """Statistical metrics for a single pandas Series.

    Computes various statistical measures including mean, variance, standard
    deviation, median, and distribution characteristics for a data series.

    Parameters
    ----------
    series : pd.Series
        Input data series for metric calculations.
    """
    series: pd.Series = pydantic.Field(
        exclude=True,
        repr=False,
    )

    @computed_field_cached_property()
    def sum(self) -> float:
        """Calculate the sum of all values in the series.

        Returns
        -------
        float
            Sum of series values.
        """
        return self.series.sum()

    @computed_field_cached_property()
    def mean(self) -> float:
        """Calculate the arithmetic mean of the series.

        Returns
        -------
        float
            Mean value, or 0.0 if series is empty.
        """
        n = len(self.series)
        if n == 0:
            return 0.0
        return self.sum / n

    @computed_field_cached_property()
    def variance(self) -> float:
        """Calculate the variance of the series.

        Uses population variance (ddof=0).

        Returns
        -------
        float
            Variance value.
        """
        return self.series.var(ddof=0)

    @computed_field_cached_property()
    def std(self) -> float:
        """Calculate the standard deviation of the series.

        Returns
        -------
        float
            Standard deviation value.
        """
        return self.variance**0.5

    @computed_field_cached_property()
    def cvstd(self) -> float:
        """Calculate coefficient of variation of standard deviation.

        Ratio of standard deviation to mean, providing a normalized
        measure of dispersion. Useful for comparing variability across
        datasets with different scales.

        Returns
        -------
        float
            Coefficient of variation.
        """
        return safe_divide(self.std, self.mean, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def sum_squared(self) -> float:
        """Calculate the sum of squared values.

        Used in various statistical calculations including variance and SSE.

        Returns
        -------
        float
            Sum of squared values.
        """
        return (self.series**2).sum()

    @computed_field_cached_property()
    def median(self) -> float:
        """Calculate the median of the series.

        The middle value that separates the higher half from the lower half.

        Returns
        -------
        float
            Median value.
        """
        return self.series.median()

    @computed_field_cached_property()
    def MAD_scaled(self) -> float:
        """Calculate scaled Median Absolute Deviation (MAD).

        A robust measure of statistical dispersion that is less sensitive
        to outliers than standard deviation. Scaled to be consistent with
        the standard deviation for normally distributed data.

        Returns
        -------
        float
            Scaled MAD value.
        """
        return median_absolute_deviation(self.series, self.median)

    @computed_field_cached_property()
    def iqr(self) -> float:
        """Calculate the interquartile range (IQR).

        The difference between the 75th and 25th percentiles, providing a robust
        measure of spread that is resistant to outliers.

        Returns
        -------
        float
            Interquartile range.
        """
        return np.diff(np.quantile(self.series, [0.25, 0.75]))[0]

    @computed_field_cached_property()
    def skew(self) -> float:
        """Calculate the skewness of the distribution.

        Measures the asymmetry of the distribution. Positive values indicate
        right skew, negative values indicate left skew.

        Returns
        -------
        float
            Skewness value.
        """
        return self.series.skew()

    @computed_field_cached_property()
    def kurtosis(self) -> float:
        """Calculate the kurtosis of the distribution.

        Measures the "tailedness" of the distribution. Higher values indicate
        heavier tails and more outliers.

        Returns
        -------
        float
            Kurtosis value (excess kurtosis, where normal distribution = 0).
        """
        return self.series.kurtosis()


def A_n(x: np.ndarray, n: float) -> float:
    """Calculate the proportion of values in x that are less than or equal to n.

    Parameters
    ----------
    x : np.ndarray
        Input array
    n : float
        Threshold value

    Returns
    -------
    float
        Proportion of values <= n
    """
    return np.mean(x <= n)

class BaselineMetrics(ArbitraryPydanticModel):
    """Comprehensive baseline model evaluation metrics.

    Calculates a wide range of statistical measures and goodness-of-fit metrics
    for evaluating baseline energy model performance. Includes error metrics,
    normalized metrics, percentage error metrics, efficiency metrics, and
    autocorrelation-adjusted variants following ASHRAE Guideline 14 methodology.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'observed' and 'predicted' columns containing model
        baseline period data.
    num_model_params : int
        Number of parameters in the baseline model (used for degrees of freedom
        adjustments). Must be >= 1.

    Attributes
    ----------
    n : float
        Number of valid observations
    n_prime : float
        Autocorrelation-adjusted sample size
    ddof : float
        Delta degrees of freedom
    ddof_autocorr : float
        Autocorrelation-adjusted degrees of freedom
    observed : ColumnMetrics
        Statistical metrics for observed data
    predicted : ColumnMetrics
        Statistical metrics for predicted data
    residuals : ColumnMetrics
        Statistical metrics for residuals
    max_error : float
        Maximum absolute error
    mae : float
        Mean absolute error
    nmae : float
        Normalized mean absolute error (by mean)
    pnmae : float
        Percentile normalized MAE (by IQR)
    medae : float
        Median absolute error
    mbe : float
        Mean bias error
    nmbe : float
        Normalized mean bias error (by mean)
    pnmbe : float
        Percentile normalized MBE (by IQR)
    sse : float
        Sum of squared errors
    mse : float
        Mean squared error
    rmse : float
        Root mean squared error
    rmse_autocorr : float
        Autocorrelation-corrected RMSE
    rmse_adj : float
        Adjusted RMSE (by ddof)
    rmse_autocorr_adj : float
        Fully adjusted RMSE
    cvrmse : float
        Coefficient of variation RMSE
    cvrmse_autocorr : float
        Autocorrelation-corrected CVRMSE
    cvrmse_adj : float
        Adjusted CVRMSE
    cvrmse_autocorr_adj : float
        Fully adjusted CVRMSE
    pnrmse : float
        Percentile normalized RMSE
    pnrmse_autocorr : float
        Autocorrelation-corrected PNRMSE
    pnrmse_adj : float
        Adjusted PNRMSE
    pnrmse_autocorr_adj : float
        Fully adjusted PNRMSE
    r_squared : float
        Coefficient of determination
    r_squared_adj : float
        Adjusted R-squared
    mape : float
        Mean absolute percentage error
    smape : float
        Symmetric MAPE
    wape : float
        Weighted absolute percentage error
    swape : float
        Symmetric WAPE
    maape : float
        Mean arctangent absolute percentage error
    nse : float
        Nash-Sutcliffe efficiency
    nnse : float
        Normalized Nash-Sutcliffe efficiency
    kge : Optional[float]
        Kling-Gupta efficiency
    a10 : float
        Proportion within 10% accuracy
    a20 : float
        Proportion within 20% accuracy
    a30 : float
        Proportion within 30% accuracy
    wi : float
        Willmott index
    index_of_agreement : float
        Refined Willmott index
    pearson_r : float
        Pearson correlation coefficient
    pi : float
        Performance index
    pi_rating : str
        Performance rating (excellent/very good/good/satisfactory/poor/bad/very bad)
    explained_variance_score : float
        Explained variance score

    Notes
    -----
    All metrics are computed on valid (finite) observations only. Autocorrelation
    adjustments follow ASHRAE Guideline 14 methodology for M&V applications.
    """

    df: pd.DataFrame = pydantic.Field(
        exclude=True,
        repr=False,
        description="Input dataframe with 'observed' and 'predicted' columns",
    )

    num_model_params: int = pydantic.Field(
        ge=1,
        validate_default=True,
        description="Number of parameters in the baseline model",
    )

    @cached_property
    def _df(self) -> pd.DataFrame:
        """Prepare and validate the input dataframe.

        Validates column types, filters non-finite values, and computes residuals.

        Returns
        -------
        pd.DataFrame
            Processed dataframe with 'observed', 'predicted', and 'residuals' columns.

        Raises
        ------
        ValueError
            If input dataframe is empty.
        """
        _df = self.df[["observed", "predicted"]].copy()

        if len(_df) < 1:
            raise ValueError("Input dataframe must have at least one row")

        # Check dataframe
        expected_columns = {"observed": "float", "predicted": "float"}
        _df = PydanticDf(df=_df, column_types=expected_columns).df

        # drop non finite values from df
        _df = _df[np.isfinite(_df["observed"]) & np.isfinite(_df["predicted"])]

        # get residuals
        _df["residuals"] = _df["observed"] - _df["predicted"]

        return _df

    @computed_field_cached_property()
    def n(self) -> float:
        """Calculate the number of observations.

        Returns the count of valid observations after filtering non-finite values.

        Returns
        -------
        float
            Number of observations.
        """
        return len(self._df)

    @computed_field_cached_property()
    def n_prime(self) -> float:
        """Calculate effective sample size corrected for autocorrelation.

        Adjusts the sample size to account for autocorrelation in residuals,
        following ASHRAE Guideline 14 methodology. Uses lag-1 autocorrelation
        as recommended in LBNL technical report.

        Reference: https://www.osti.gov/servlets/purl/1366449

        Returns
        -------
        float
            Effective sample size (minimum value of 1).
        """
        # lag should be 1 according to LBNL guidance
        autocorr = acf(self._df["residuals"].values, lag_n=1, ac_type="moving_stats")[1]

        numerator = self.n * (1 - autocorr)
        denominator = 1 + autocorr
        _n_prime = safe_divide(numerator, denominator, _MIN_DENOMINATOR)

        # Ensure valid result
        if not np.isfinite(_n_prime) or _n_prime < 1:
            _n_prime = 1

        return _n_prime

    @computed_field_cached_property()
    def ddof(self) -> float:
        """Calculate delta degrees of freedom (ddof).

        The number of independent observations minus the number of model
        parameters. Used in adjusted statistical calculations like
        adjusted RMSE and R-squared.

        Returns
        -------
        float
            Delta degrees of freedom (minimum value of 1).
        """
        _ddof = self.n - self.num_model_params

        if _ddof < 1:
            _ddof = 1

        return _ddof

    @computed_field_cached_property()
    def ddof_autocorr(self) -> float:
        """Calculate autocorrelation-adjusted delta degrees of freedom.

        Similar to ddof but uses the effective sample size (n_prime) that
        accounts for autocorrelation in residuals. Used in ASHRAE Guideline 14
        uncertainty calculations.

        Returns
        -------
        float
            Autocorrelation-adjusted ddof (minimum value of 1).
        """
        _ddof_autocorr = self.n_prime - self.num_model_params

        if _ddof_autocorr < 1:
            _ddof_autocorr = 1

        return _ddof_autocorr

    @computed_field_cached_property()
    def observed(self) -> ColumnMetrics:
        """Calculate statistical metrics for observed values.

        Returns
        -------
        ColumnMetrics
            Statistical metrics for the observed data column.
        """
        return ColumnMetrics(series=self._df["observed"])

    @computed_field_cached_property()
    def predicted(self) -> ColumnMetrics:
        """Calculate statistical metrics for predicted values.

        Returns
        -------
        ColumnMetrics
            Statistical metrics for the predicted data column.
        """
        return ColumnMetrics(series=self._df["predicted"])

    @computed_field_cached_property()
    def residuals(self) -> ColumnMetrics:
        """Calculate statistical metrics for residuals.

        Returns
        -------
        ColumnMetrics
            Statistical metrics for the residuals (observed - predicted).
        """
        return ColumnMetrics(series=self._df["residuals"])

    @computed_field_cached_property()
    def max_error(self) -> float:
        """Calculate maximum absolute error.

        The largest absolute difference between predicted and observed values.
        Useful for understanding worst-case prediction errors.

        Returns
        -------
        float
            Maximum absolute error.
        """
        return np.max(np.abs(self._df["residuals"].values))

    @computed_field_cached_property()
    def mae(self) -> float:
        """Calculate Mean Absolute Error (MAE).

        The average of absolute differences between predicted and observed values.
        Provides a straightforward measure of prediction accuracy.

        Returns
        -------
        float
            Mean absolute error.
        """
        return np.mean(np.abs(self._df["residuals"].values))

    @computed_field_cached_property()
    def nmae(self) -> float:
        """Calculate Normalized Mean Absolute Error (NMAE).

        Normalizes MAE by the mean of observed values. Commonly used in
        ASHRAE Guideline 14 for model validation. Lower values indicate
        better performance.

        Returns
        -------
        float
            NMAE value.
        """
        return safe_divide(self.mae, self.observed.mean, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def pnmae(self) -> float:
        """Calculate Percentile Normalized Mean Absolute Error (PNMAE).

        Normalizes MAE by the interquartile range (IQR) of observed values
        instead of the mean, making it more robust to outliers and extreme values.

        Returns
        -------
        float
            PNMAE value.
        """
        return safe_divide(self.mae, self.observed.iqr, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def medae(self) -> float:
        """Calculate Median Absolute Error (MedAE).

        The median of absolute errors, providing a robust central measure of
        prediction error that is less sensitive to outliers than MAE.

        Returns
        -------
        float
            Median absolute error.
        """
        return np.median(np.abs(self._df["residuals"].values))

    @computed_field_cached_property()
    def mbe(self) -> float:
        """Calculate Mean Bias Error (MBE).

        The average of residuals (observed - predicted), indicating systematic
        bias in predictions. Positive values indicate under-prediction, negative
        values indicate over-prediction.

        Returns
        -------
        float
            Mean bias error.
        """
        return self.residuals.mean

    @computed_field_cached_property()
    def nmbe(self) -> float:
        """Calculate Normalized Mean Bias Error (NMBE).

        Normalizes MBE by the mean of observed values. Measures systematic
        bias in predictions (over- or under-prediction). Used in ASHRAE
        Guideline 14 for model validation. Values near zero indicate
        unbiased predictions.

        Returns
        -------
        float
            NMBE value.
        """
        return safe_divide(self.mbe, self.observed.mean, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def pnmbe(self) -> float:
        """Calculate Percentile Normalized Mean Bias Error (PNMBE).

        Normalizes MBE by the interquartile range (IQR) of observed values
        instead of the mean, providing a robust measure of bias that is less
        sensitive to outliers.

        Returns
        -------
        float
            PNMBE value.
        """
        return safe_divide(self.mbe, self.observed.iqr, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def sse(self) -> float:
        """Calculate Sum of Squared Errors (SSE).

        The sum of squared residuals, a fundamental measure used in many
        statistical calculations and goodness-of-fit metrics.

        Returns
        -------
        float
            Sum of squared errors.
        """
        return self.residuals.sum_squared

    @computed_field_cached_property()
    def mse(self) -> float:
        """Calculate Mean Squared Error (MSE).

        The average of squared residuals, penalizing larger errors more than
        smaller ones. Square root of MSE gives RMSE.

        Returns
        -------
        float
            Mean squared error.
        """
        return self.sse / self.n

    @computed_field_cached_property()
    def rmse(self) -> float:
        """Calculate Root Mean Squared Error (RMSE).

        The square root of MSE, providing an error metric in the same units
        as the original data. Commonly used for model evaluation.

        Returns
        -------
        float
            Root mean squared error.
        """
        return self.mse**0.5

    @computed_field_cached_property()
    def rmse_autocorr(self) -> float:
        """Calculate autocorrelation-corrected RMSE.

        RMSE adjusted for autocorrelation in residuals using the effective
        sample size (n_prime). More accurate for time-series data.

        Returns
        -------
        float
            Autocorrelation-corrected RMSE.
        """
        return (self.sse / self.n_prime) ** 0.5

    @computed_field_cached_property()
    def rmse_adj(self) -> float:
        """Calculate adjusted RMSE.

        RMSE adjusted for degrees of freedom to account for model complexity.
        Penalizes models with more parameters.

        Returns
        -------
        float
            Adjusted RMSE.
        """
        return (self.sse / self.ddof) ** 0.5

    @computed_field_cached_property()
    def rmse_autocorr_adj(self) -> float:
        """Calculate autocorrelation-corrected and adjusted RMSE.

        RMSE with both autocorrelation and degrees-of-freedom adjustments,
        providing the most robust error metric for time-series modeling.

        Returns
        -------
        float
            Autocorrelation-corrected and adjusted RMSE.
        """
        return (self.sse / self.ddof_autocorr) ** 0.5

    @computed_field_cached_property()
    def cvrmse(self) -> float:
        """Calculate Coefficient of Variation of Root Mean Squared Error (CVRMSE).

        Normalizes RMSE by the mean of observed values, making it a
        dimensionless measure of model fit quality. Commonly used in
        ASHRAE Guideline 14 for M&V applications. Lower values indicate
        better performance.

        Returns
        -------
        float
            CVRMSE value.
        """
        return safe_divide(self.rmse, self.observed.mean, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def cvrmse_autocorr(self) -> float:
        """Calculate autocorrelation-corrected CVRMSE.

        CVRMSE using autocorrelation-adjusted RMSE for better handling of
        time-series data with correlated residuals.

        Returns
        -------
        float
            Autocorrelation-corrected CVRMSE value.
        """
        return safe_divide(self.rmse_autocorr, self.observed.mean, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def cvrmse_adj(self) -> float:
        """Calculate adjusted CVRMSE.

        CVRMSE using degrees-of-freedom adjusted RMSE to account for
        model complexity. Used in ASHRAE Guideline 14 uncertainty calculations.

        Returns
        -------
        float
            Adjusted CVRMSE value.
        """
        return safe_divide(self.rmse_adj, self.observed.mean, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def cvrmse_autocorr_adj(self) -> float:
        """Calculate autocorrelation-corrected and adjusted CVRMSE.

        CVRMSE using both autocorrelation and degrees-of-freedom adjustments
        for the most robust normalized error metric in time-series modeling.

        Returns
        -------
        float
            Autocorrelation-corrected and adjusted CVRMSE value.
        """
        return safe_divide(
            self.rmse_autocorr_adj, self.observed.mean, _MIN_DENOMINATOR
        )

    @computed_field_cached_property()
    def pnrmse(self) -> float:
        """Calculate Percentile Normalized Root Mean Squared Error (PNRMSE).

        Normalizes RMSE by the interquartile range (IQR) instead of the mean,
        providing a robust dimensionless error metric that is less sensitive
        to outliers.

        Returns
        -------
        float
            PNRMSE value.
        """
        return safe_divide(self.rmse, self.observed.iqr, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def pnrmse_autocorr(self) -> float:
        """Calculate autocorrelation-corrected PNRMSE.

        PNRMSE using autocorrelation-adjusted RMSE for better handling of
        time-series data with correlated residuals.

        Returns
        -------
        float
            Autocorrelation-corrected PNRMSE value.
        """
        return safe_divide(self.rmse_autocorr, self.observed.iqr, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def pnrmse_adj(self) -> float:
        """Calculate adjusted PNRMSE.

        PNRMSE using degrees-of-freedom adjusted RMSE to account for
        model complexity.

        Returns
        -------
        float
            Adjusted PNRMSE value.
        """
        return safe_divide(self.rmse_adj, self.observed.iqr, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def pnrmse_autocorr_adj(self) -> float:
        """Calculate autocorrelation-corrected and adjusted PNRMSE.

        PNRMSE using both autocorrelation and degrees-of-freedom adjustments
        for the most robust error metric in time-series with model complexity.

        Returns
        -------
        float
            Autocorrelation-corrected and adjusted PNRMSE value.
        """
        return safe_divide(
            self.rmse_autocorr_adj, self.observed.iqr, _MIN_DENOMINATOR
        )

    @computed_field_cached_property()
    def r_squared(self) -> float:
        """Calculate coefficient of determination (R²).

        Represents the proportion of variance in the observed data that is
        predictable from the model. Ranges from 0 to 1, with 1 indicating
        perfect prediction.

        Returns
        -------
        float
            R-squared value.
        """
        return self._df[["predicted", "observed"]].corr().iloc[0, 1] ** 2

    @computed_field_cached_property()
    def r_squared_adj(self) -> float:
        """Calculate adjusted R-squared.

        Adjusts R-squared for the number of model parameters, penalizing
        model complexity. More appropriate than R-squared when comparing
        models with different numbers of parameters.

        Returns
        -------
        float
            Adjusted R-squared value.
        """
        n = self.n
        n_adj = self.ddof

        num = (1 - self.r_squared) * (n - 1)
        den = n_adj - 1

        res = safe_divide(num, den, _MIN_DENOMINATOR)

        return 1 - res

    @computed_field_cached_property()
    def mape(self) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE).

        Expresses prediction accuracy as a percentage of the observed values.
        Lower values indicate better performance. Can be problematic when
        observed values are close to zero.

        Returns
        -------
        float
            Mean absolute percentage error.
        """
        df = self._df

        num = np.abs(df["residuals"].values)
        den = np.abs(df["observed"].values)

        inner = safe_divide(num, den, _MIN_DENOMINATOR)

        return np.mean(inner)

    @computed_field_cached_property()
    def smape(self) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

        A symmetric alternative to MAPE that treats over- and under-predictions
        equally by using the average of observed and predicted values in the
        denominator. More robust when values approach zero.

        Returns
        -------
        float
            Symmetric mean absolute percentage error.
        """
        df = self._df

        num = np.abs(df["residuals"].values)
        obs = np.abs(df["observed"].values)
        pred = np.abs(df["predicted"].values)
        den = (obs + pred) / 2

        inner = safe_divide(num, den, _MIN_DENOMINATOR)

        return np.mean(inner)

    @computed_field_cached_property()
    def wape(self) -> float:
        """Calculate Weighted Absolute Percentage Error (WAPE).

        Also known as MAD/Mean ratio. Weights errors by the magnitude of
        observations, making it more robust to outliers than MAPE.

        Returns
        -------
        float
            Weighted absolute percentage error.
        """
        df = self._df

        num = self.mae * self.n
        den = np.sum(np.abs(df["observed"].values))

        return safe_divide(num, den, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def swape(self) -> float:
        """Calculate Symmetric Weighted Absolute Percentage Error (SWAPE).

        Combines the symmetry of SMAPE with the weighting approach of WAPE,
        providing a balanced metric that is robust to both outliers and
        near-zero values.

        Returns
        -------
        float
            Symmetric weighted absolute percentage error.
        """
        df = self._df

        num = self.mae * self.n
        obs = np.abs(df["observed"].values)
        pred = np.abs(df["predicted"].values)
        den = np.sum((obs + pred) / 2)

        return safe_divide(num, den, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def maape(self) -> float:
        """Calculate Mean Arctangent Absolute Percentage Error (MAAPE).

        Uses arctangent transformation to bound percentage errors, making it
        highly robust to outliers and extreme values. Returns values in the
        range [0, π/2].

        Returns
        -------
        float
            Mean arctangent absolute percentage error.
        """
        df = self._df

        num = df["residuals"].values
        den = df["observed"].values

        inner = safe_divide(num, den, _MIN_DENOMINATOR)
        inner = np.arctan(np.abs(inner))

        return np.mean(inner)

    @computed_field_cached_property()
    def nse(self) -> float:
        """Calculate Nash-Sutcliffe Efficiency (NSE).

        Measures how well predictions match observations relative to using the
        mean as a predictor. Ranges from -∞ to 1, with 1 being perfect match,
        0 meaning the model is no better than the mean, and negative values
        indicating worse performance than using the mean.

        Returns
        -------
        float
            Nash-Sutcliffe Efficiency value.
        """
        df = self._df

        num = self.sse
        den = np.sum((df["observed"].values - self.observed.mean)**2)

        return 1 - safe_divide(num, den, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def nnse(self) -> float:
        """Calculate Normalized Nash-Sutcliffe Efficiency (NNSE).

        A normalized version of NSE that transforms the range to [0, 1],
        making it easier to interpret. Values closer to 1 indicate better
        model performance.

        Returns
        -------
        float
            Normalized Nash-Sutcliffe Efficiency value.
        """
        return safe_divide(1.0, 2 - self.nse, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def kge(self) -> Optional[float]:
        """Calculate Kling-Gupta Efficiency (KGE).

        A comprehensive goodness-of-fit measure that decomposes into
        correlation, bias, and variability components. Ranges from -∞ to 1,
        with 1 being perfect agreement.

        Returns
        -------
        Optional[float]
            Kling-Gupta Efficiency value, or None if calculation fails.
        """
        r = self.pearson_r
        bias_ratio = safe_divide(self.predicted.mean, self.observed.mean, _MIN_DENOMINATOR)
        variability_ratio = safe_divide(self.predicted.cvstd, self.observed.cvstd, _MIN_DENOMINATOR)

        # Check if all components are finite
        if not np.isfinite(r) or not np.isfinite(bias_ratio) or not np.isfinite(variability_ratio):
            return None

        result = 1 - np.sqrt((r - 1)**2 + (bias_ratio - 1)**2 + (variability_ratio - 1)**2)

        if not np.isfinite(result):
            return None

        return result

    @cached_property
    def _relative_errors(self) -> np.ndarray:
        """Cache the relative error calculation used by a10, a20, a30 metrics."""
        numerator = np.abs(self._df["residuals"].values)
        denominator = np.abs(self._df["observed"].values)

        return safe_divide(numerator, denominator, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def a10(self) -> float:
        """Calculate A10 metric (proportion of predictions within 10% of observed).

        Returns the fraction of predictions where the absolute percentage error
        is less than or equal to 10%. Higher values indicate better performance.

        Returns
        -------
        float
            Proportion of predictions within 10% accuracy.
        """
        return A_n(self._relative_errors, 0.1)

    @computed_field_cached_property()
    def a20(self) -> float:
        """Calculate A20 metric (proportion of predictions within 20% of observed).

        Returns the fraction of predictions where the absolute percentage error
        is less than or equal to 20%. Higher values indicate better performance.

        Returns
        -------
        float
            Proportion of predictions within 20% accuracy.
        """
        return A_n(self._relative_errors, 0.2)

    @computed_field_cached_property()
    def a30(self) -> float:
        """Calculate A30 metric (proportion of predictions within 30% of observed).

        Returns the fraction of predictions where the absolute percentage error
        is less than or equal to 30%. Higher values indicate better performance.

        Returns
        -------
        float
            Proportion of predictions within 30% accuracy.
        """
        return A_n(self._relative_errors, 0.3)

    @computed_field_cached_property()
    def wi(self) -> float:
        """Calculate the Willmott Index of Agreement.

        Measures the degree of model prediction error relative to potential error.
        Ranges from 0 to 1, with 1 indicating perfect agreement.

        Returns
        -------
        float
            Willmott Index value.
        """
        df = self._df

        num = self.sse

        mean_obs = self.observed.mean
        pred_shifted = df["predicted"].values - mean_obs
        obs_shifted = df["observed"].values - mean_obs
        den = np.sum((np.abs(pred_shifted) + np.abs(obs_shifted))**2)

        return 1 - safe_divide(num, den, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def index_of_agreement(self) -> float:
        """Calculate the refined Index of Agreement (d_r).

        A refinement of the Willmott Index that is more sensitive to systematic
        over- or under-prediction. Ranges from -1 to 1, with 1 indicating
        perfect agreement.

        Reference: Willmott et al. (2012), https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.2419

        Returns
        -------
        float
            Refined index of agreement value.
        """
        df = self._df

        num = self.mae * self.n
        den = 2 * np.sum(np.abs(df["observed"].values - self.observed.mean))

        if num <= den:
            return 1 - safe_divide(num, den, _MIN_DENOMINATOR)

        return safe_divide(den, num, _MIN_DENOMINATOR) - 1
    
    @computed_field_cached_property()
    def pearson_r(self) -> float:
        """Calculate Pearson correlation coefficient.

        Measures the linear correlation between observed and predicted values.
        Ranges from -1 to 1, with 1 indicating perfect positive correlation,
        -1 perfect negative correlation, and 0 no linear correlation.

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        return pearsonr(self._df["observed"].values, self._df["predicted"].values)[0]

    @computed_field_cached_property()
    def pi(self) -> float:
        """Calculate Performance Index (PI).

        Combines Pearson correlation and Willmott Index to provide a
        comprehensive model performance metric. Ranges from -1 to 1,
        with higher values indicating better performance.

        Reference: https://doi.org/10.1016/j.asoc.2021.107282

        Returns
        -------
        float
            Performance Index value.
        """
        return self.pearson_r * self.wi

    @computed_field_cached_property()
    def pi_rating(self) -> str:
        """Classify model performance based on Performance Index (PI).

        Returns a qualitative rating of the model performance based on
        the Performance Index value according to established thresholds.

        Returns
        -------
        str
            Performance rating: 'excellent', 'very good', 'good', 'satisfactory',
            'poor', 'bad', or 'very bad'.
        """
        pi = self.pi

        if pi >= 0.85:
            return "excellent"
        elif pi >= 0.75:
            return "very good"
        elif pi >= 0.65:
            return "good"
        elif pi >= 0.60:
            return "satisfactory"
        elif pi >= 0.50:
            return "poor"
        elif pi >= 0.40:
            return "bad"
        else:
            return "very bad"
        
    @computed_field_cached_property()
    def explained_variance_score(self) -> float:
        """Calculate the explained variance score.

        Measures the proportion of variance in the observed data that is
        explained by the model predictions. Ranges from -∞ to 1, with 1
        indicating perfect prediction and 0 indicating no explanatory power.

        Returns
        -------
        float
            Explained variance score.
        """
        num = self.residuals.variance
        den = self.observed.variance

        return 1 - safe_divide(num, den, _MIN_DENOMINATOR)
    

def BaselineMetricsFromDict(input_dict: dict) -> BaselineMetrics:
    """Construct a BaselineMetrics instance from a dictionary.

    Parameters
    ----------
    input_dict : dict
        Dictionary containing BaselineMetrics data, with optional nested
        ColumnMetrics data for 'observed', 'predicted', and 'residuals' keys.

    Returns
    -------
    BaselineMetrics
        Constructed BaselineMetrics instance.
    """
    for k in ["observed", "predicted", "residuals"]:
        if k in input_dict:
            input_dict[k] = PydanticFromDict(input_dict[k], name="ColumnMetrics")

    return PydanticFromDict(input_dict, name="BaselineMetrics")


class ModelChoice(str, Enum):
    """Data frequency choices for baseline models.

    Determines the time granularity of the baseline model, which affects
    uncertainty calculations in ASHRAE Guideline 14 methodology.

    Attributes
    ----------
    HOURLY : str
        Hourly data frequency.
    HOURLYSOLAR : str
        Hourly solar data frequency (mapped to "hourly").
    DAILY : str
        Daily data frequency.
    BILLING : str
        Billing period data frequency.
    """
    HOURLY = "hourly"
    HOURLYSOLAR = "hourly"
    DAILY = "daily"
    BILLING = "billing"


class ReportingMetrics(pydantic.BaseModel):
    """Reporting period metrics for energy savings calculations.

    Calculates savings, uncertainty, and fractional savings uncertainty (FSU)
    for a reporting period based on baseline model metrics and reporting data.
    Follows ASHRAE Guideline 14 methodology.
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    baseline_metrics: Union[BaselineMetrics, pydantic.BaseModel] = pydantic.Field(
        exclude=True,
        description="Baseline model metrics instance",
    )

    reporting_df: pd.DataFrame = pydantic.Field(
        exclude=True,
        description="Reporting period dataframe with 'observed' and 'predicted' columns",
    )

    data_frequency: ModelChoice = pydantic.Field(
        exclude=False,
        description="Data frequency of the model (hourly, daily, or billing)",
    )

    confidence_level: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        default=0.90,
        validate_default=True,
        description="Confidence level for uncertainty calculations",
    )

    t_tail: int = pydantic.Field(
        ge=1,
        le=2,
        default=2,
        validate_default=True,
        description="Number of tails for hypothesis testing (1 or 2)",
    )

    @property
    def _baseline(self) -> BaselineMetrics:
        """Convenience property to access baseline metrics."""
        return self.baseline_metrics

    @cached_property
    def _df(self) -> pd.DataFrame:
        """Prepare and validate the reporting period dataframe.

        Validates column types and filters non-finite values.

        Returns
        -------
        pd.DataFrame
            Processed dataframe with 'observed' and 'predicted' columns.

        Raises
        ------
        ValueError
            If reporting dataframe is empty.
        """
        _df = self.reporting_df[["observed", "predicted"]].copy()

        if len(_df) < 1:
            raise ValueError("Input dataframe must have at least one row")

        # Check dataframe
        expected_columns = {"observed": "float", "predicted": "float"}
        _df = PydanticDf(df=_df, column_types=expected_columns).df

        # drop non finite values from df
        _df = _df[np.isfinite(_df["observed"]) & np.isfinite(_df["predicted"])]

        return _df

    @computed_field_cached_property()
    def n(self) -> float:
        """Calculate the number of observations in the reporting period.

        Returns the count of valid observations after filtering non-finite values.

        Returns
        -------
        float
            Number of observations.
        """
        return len(self._df)

    @computed_field_cached_property()
    def observed_sum(self) -> float:
        """Calculate total observed energy consumption.

        Sum of all observed values in the reporting period.

        Returns
        -------
        float
            Total observed energy.
        """
        return self._df["observed"].sum()

    @computed_field_cached_property()
    def predicted_sum(self) -> float:
        """Calculate total predicted energy consumption.

        Sum of all predicted values in the reporting period (baseline forecast).

        Returns
        -------
        float
            Total predicted energy.
        """
        return self._df["predicted"].sum()

    @computed_field_cached_property()
    def t_stat(self) -> float:
        """Calculate t-statistic for uncertainty calculations.

        Returns the t-statistic value based on confidence level, degrees of
        freedom, and number of tails for hypothesis testing.

        Returns
        -------
        float
            t-statistic value.
        """
        return t_stat(1 - self.confidence_level, self._baseline.ddof, tail=self.t_tail)

    @computed_field_cached_property()
    def savings(self) -> float:
        """Calculate energy savings.

        The difference between predicted (baseline) and observed energy
        consumption. Positive values indicate energy savings.

        Returns
        -------
        float
            Energy savings.
        """
        return self.predicted_sum - self.observed_sum

    @computed_field_cached_property()
    def total_savings_uncertainty(self) -> Optional[float]:
        """Calculate total savings uncertainty following ASHRAE Guideline 14.

        Computes uncertainty in energy savings predictions accounting for
        autocorrelation, sample size, and data frequency effects.

        Returns
        -------
        Optional[float]
            Total savings uncertainty, or None if calculation fails.
        """
        E_reporting = self.predicted_sum
        n = self._baseline.n
        n_prime = self._baseline.n_prime
        m = self.n
        t = self.t_stat
        cvrmse_adj = self._baseline.cvrmse_adj

        # Approximation factor from ASHRAE Guideline 14
        n_ratio = safe_divide(n, n_prime, _MIN_DENOMINATOR)
        n_prime_term = safe_divide(2.0, n_prime, _MIN_DENOMINATOR)
        approx_factor = np.sqrt(n_ratio * (1 + n_prime_term) * m)

        try:
            e_per_m = safe_divide(E_reporting, m, _MIN_DENOMINATOR)
            s_unc_base = np.abs(e_per_m * cvrmse_adj) * t * approx_factor
        except (ZeroDivisionError, FloatingPointError, ValueError):
            return None

        if self.data_frequency == "hourly":
            # ASHRAE 14 hourly data correction factor
            s_unc = 1.26 * s_unc_base

        elif self.data_frequency in ["daily", "billing"]:
            M = len(self._df.index.month.unique())

            # Sun & Baltazar 2013 polynomial corrections
            if self.data_frequency == "daily":
                coefs = [-0.00024, 0.03535, 1.00286]
            else:
                coefs = [-0.00022, 0.03306, 0.94054]

            s_unc = np.polyval(coefs, M) * s_unc_base

        else:
            raise ValueError("model_type must be 'hourly', 'daily', or 'billing'")

        return s_unc

    @computed_field_cached_property()
    def fsu(self) -> float:
        """Calculate Fractional Savings Uncertainty (FSU).

        The ratio of total savings uncertainty to actual savings, expressed
        as a fraction. Used to assess the reliability of savings estimates.

        Returns
        -------
        float
            Fractional savings uncertainty.
        """
        return safe_divide(self.total_savings_uncertainty, self.savings, _MIN_DENOMINATOR)

    @computed_field_cached_property()
    def predicted_data_point_unc(self) -> Optional[float]:
        """Calculate uncertainty per predicted data point.

        Normalizes total savings uncertainty by the square root of the number
        of reporting period observations.

        Returns
        -------
        Optional[float]
            Per-point uncertainty, or None if total uncertainty cannot be calculated.
        """
        if self.total_savings_uncertainty is None:
            return None

        return self.total_savings_uncertainty / np.sqrt(self.n)
    

class AutocorrelationMethod(Enum):
    """Methods for computing autocorrelation function.

    Attributes
    ----------
    MOVING_STATS : str
        Compute mean and standard deviation in a rolling window.
    STATIONARY_CORRELATE : str
        Compute over entire series using correlate.
    STATIONARY_STATS_FFT : str
        Compute over entire series using FFT for efficiency.
    """
    MOVING_STATS = "moving_stats"
    STATIONARY_CORRELATE = "stationary_correlate"
    STATIONARY_STATS_FFT = "stationary_stats_fft"


def acf(
    x: np.ndarray,
    lag_n: Optional[int] = None,
    ac_type: AutocorrelationMethod = AutocorrelationMethod.MOVING_STATS
) -> np.ndarray:
    """Compute the autocorrelation function (ACF) of a time series.

    The ACF measures the correlation of a signal with a delayed copy of itself
    as a function of delay. It helps identify repeating patterns, periodic signals
    obscured by noise, or missing fundamental frequencies implied by harmonics.

    Parameters
    ----------
    x : np.ndarray
        The time series data.
    lag_n : int, optional
        The number of lags to compute the ACF for. If None, computes the ACF
        for all possible lags.
    ac_type : AutocorrelationMethod, optional
        Method to compute the ACF. Default is MOVING_STATS.

    Returns
    -------
    np.ndarray
        The autocorrelation function values for the given time series and lags.
    """
    if isinstance(ac_type, AutocorrelationMethod):
        ac_type = ac_type.value

    if lag_n is None:
        lags = range(len(x) - 1)
    else:
        lags = range(lag_n + 1)

    if ac_type == AutocorrelationMethod.MOVING_STATS.value:
        # mean and std are computed in a rolling window
        corr = [1.0 if l == 0 else np.corrcoef(x[l:], x[:-l])[0][1] for l in lags]
        corr = np.array(corr)

    elif "stationary" in ac_type:
        # mean and std are computed over the entire series
        n = len(x)
        mean = x.mean()
        var = np.var(x)
        xc = x - mean

        if ac_type == AutocorrelationMethod.STATIONARY_CORRELATE.value:
            corr = np.correlate(xc, xc, "full")[(n - 1):] / var / n

        elif ac_type == AutocorrelationMethod.STATIONARY_STATS_FFT.value:
            cf = np.fft.fft(xc)
            sf = cf.conjugate() * cf
            corr = np.fft.ifft(sf).real / var / len(x)

        corr = corr[:len(lags)]

    return corr