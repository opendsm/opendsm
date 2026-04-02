"""Data classes for the P-spline daily model.

Thin subclass of the DailyModel data classes. The only difference is
that baseline data drops NaN rows and sorts by temperature for the
spline fitter.
"""

from __future__ import annotations

import pandas as pd

from opendsm.eemeter.models.daily.data import (
    DailyBaselineData as _DailyBaselineData,
    DailyReportingData,
)

__all__ = ("DailyBaselineData", "DailyReportingData")


class DailyBaselineData(_DailyBaselineData):
    """P-spline baseline data: drops NaN observations and sorts by temperature."""

    def __init__(
        self,
        df: pd.DataFrame,
        is_electricity_data: bool,
        settings: dict | None = None,
    ):
        super().__init__(df, is_electricity_data, settings=settings)
        if self._df is not None:
            self._df = (
                self._df
                .dropna(subset=["temperature", "observed"])
            )
            # Store time-ordering before temperature sort (for ACF computation)
            self._time_order = self._df.index.copy()
            self._df = self._df.sort_values("temperature")
