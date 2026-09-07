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

from __future__ import annotations

import numpy as np
import pandas as pd

from opendsm.eemeter.models.billing.data import (
    BillingBaselineData,
    BillingReportingData,
)
from opendsm.eemeter.models.daily.model import DailyModel
from opendsm.eemeter.models.billing.settings import BillingSettings



class BillingModel(DailyModel):
    """A class to fit a model to the input meter data.

    BillingModel is a wrapper for the DailyModel class using billing presets.

    Billing rows are period starts: each row's 'observed' value covers the span from that row
    to the next one, and a trailing row with a NaN 'observed' value closes the last period.

    Attributes:
        settings (dict): A dictionary of settings.
        seasonal_options (list): A list of seasonal options (su: Summer, sh: Shoulder, wi: Winter).
            Elements in the list are seasons separated by '_' that represent a model split.
            For example, a list of ['su_sh', 'wi'] represents two splits: summer/shoulder and winter.
        day_options (list): A list of day options.
        combo_dictionary (dict): A dictionary of combinations.
        df_meter (pandas.DataFrame): A dataframe of meter data.
        error (dict): A dictionary of error metrics.
        combinations (list): A list of combinations.
        components (list): A list of components.
        fit_components (list): A list of fit components.
        wRMSE_base (float): The mean bias error for no splits.
        best_combination (list): The best combination of splits.
        model (sklearn.pipeline.Pipeline): The final fitted model.
        id (str): The index of the meter data.
    """
    _baseline_data_type = BillingBaselineData
    _reporting_data_type = BillingReportingData
    _data_df_name = "df"
    _trim_trailing_columns = ()
    _check_timezone = False
    _settings_class = BillingSettings

    def _reference_settings(self) -> BillingSettings:
        return BillingSettings()

    def _predict_data(
        self,
        data,
        aggregation: str | None = None,
        ignore_disqualification: bool = False,
    ) -> pd.DataFrame:
        """Predict on an already built reporting or baseline data object.

        Args:
            data: The billing data object to predict on.
            aggregation: The aggregation level for the prediction. One of [None, 'none', 'monthly', 'bimonthly'].

        Returns:
            Dataframe with input data along with predicted energy consumption.

        Raises:
            ValueError: If the aggregation is not one of [None, 'none', 'monthly', 'bimonthly'].
        """
        self._check_predictable(data, ignore_disqualification)
        df = getattr(data, self._data_df_name)
        df_res = self._predict(df)

        if aggregation is None:
            agg = None
        elif aggregation.lower() == "none":
            agg = None
        elif aggregation == "monthly":
            agg = "MS"
        elif aggregation == "bimonthly":
            agg = "2MS"
        else:
            raise ValueError(
                "aggregation must be one of [None, 'monthly', 'bimonthly']"
            )

        if agg is not None:
            sum_quad = lambda x: np.sqrt(np.sum(np.square(x)))

            season = df_res["season"].resample(agg).first()
            temperature = df_res["temperature"].resample(agg).mean()
            observed = df_res["observed"].resample(agg).sum()
            predicted = df_res["predicted"].resample(agg).sum()
            predicted_unc = df_res["predicted_unc"].resample(agg).apply(sum_quad)
            heating_load = df_res["heating_load"].resample(agg).sum()
            cooling_load = df_res["cooling_load"].resample(agg).sum()
            model_split = df_res["model_split"].resample(agg).first()
            model_type = df_res["model_type"].resample(agg).first()

            df_res = pd.concat(
                [
                    season,
                    temperature,
                    observed,
                    predicted,
                    predicted_unc,
                    heating_load,
                    cooling_load,
                    model_split,
                    model_type,
                ],
                axis=1,
            )

        return df_res

    def predict(
        self,
        df: pd.DataFrame,
        *,
        aggregation: str | None = None,
        ignore_disqualification: bool = False,
    ) -> pd.DataFrame:
        """Predicts the energy consumption using the fitted model.

        Args:
            df: Reporting data indexed by a tz-aware DatetimeIndex, or containing a tz-aware
                'datetime' column, with a 'temperature' column. Rows are period starts.
            aggregation: The aggregation level for the prediction. One of [None, 'none', 'monthly', 'bimonthly'].
            ignore_disqualification: Whether to ignore model disqualification. Defaults to False.

        Returns:
            Dataframe with input data along with predicted energy consumption.

        Raises:
            RuntimeError: If the model is not fitted.
            DisqualifiedModelError: If the model is disqualified and ignore_disqualification is False.
            TypeError: If df is not a dataframe.
            ValueError: If the aggregation is not one of [None, 'none', 'monthly', 'bimonthly'].
        """
        self._reject_data_object(df)
        if not self.is_fitted:
            raise RuntimeError("Model must be fit before predictions can be made.")

        df_res = self._predict_data(
            self._reporting_data(df),
            aggregation=aggregation,
            ignore_disqualification=ignore_disqualification,
        )

        return df_res

    def plot(
        self,
        df: pd.DataFrame,
        aggregation: str | None = None,
    ):
        """Plot a model fit with baseline or reporting data. Requires matplotlib to use.

        Args:
            df: Baseline or reporting data indexed by a tz-aware DatetimeIndex, or containing
                a tz-aware 'datetime' column, with a 'temperature' column.
            aggregation: The aggregation level for the prediction. One of [None, 'none', 'monthly', 'bimonthly'].
        """
        try:
            from opendsm.eemeter.models.billing.plot import plot
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib is required for plotting.")

        # TODO: pass more kwargs to plotting function

        plot(self, self.predict(df, aggregation=aggregation))

    def to_dict(self) -> dict:
        """Returns a dictionary of model parameters.

        Returns:
            Model parameters.
        """
        model_dict = super().to_dict()
        model_dict["model_type"] = "billing"

        return model_dict
