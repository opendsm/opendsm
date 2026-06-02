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
import pytest

from opendsm.comparison_groups.common import Data, Data_Settings
from opendsm.comparison_groups.common import const as _const



def _treatment_time_series(comstock_hourly_all, n_ids=3):
    df_baseline, _ = comstock_hourly_all
    ids = sorted(df_baseline.index.get_level_values("id").unique())[:n_ids]
    time_series = df_baseline.loc[ids].reset_index()[["id", "datetime", "observed"]]

    return time_series, ids


def test_hour_time_period_keys_present():
    """Regression: TimePeriod.HOUR must be a valid key in the row-count and
    granularity dicts. The dicts previously keyed the standalone hour period as
    'hourly', mismatching the enum value 'hour', so any lookup raised KeyError."""
    assert _const.time_period_row_counts[_const.TimePeriod.HOUR] == 24
    assert _const.min_granularity_per_time_period[_const.TimePeriod.HOUR] == 60


def test_time_series_ingestion_hourly(_comstock_hourly_all):
    """Regression: building Data from a time series exercises loadshape_type /
    time_period reads (previously uppercased -> AttributeError) and the HOUR
    row-count lookup (previously KeyError). Should produce a 24-column loadshape."""
    time_series, ids = _treatment_time_series(_comstock_hourly_all)
    settings = Data_Settings(
        agg_type=_const.AggType.MEAN,
        loadshape_type=_const.LoadshapeType.OBSERVED,
        time_period=_const.TimePeriod.HOUR,
    )

    data = Data(time_series_df=time_series, settings=settings)

    assert data.loadshape is not None
    assert data.loadshape.shape[1] == 24
    assert set(data.loadshape.index).issubset(set(ids))
    assert np.isfinite(data.loadshape.to_numpy()).all()
