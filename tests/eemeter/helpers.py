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

"""Helpers shared by the eemeter model tests."""

import pandas as pd



def meter_temperature_frame(meter, temperature):
    """Combine meter data and temperature into one model input frame.

    `meter` is a one-column frame, a Series, or None; `temperature` is a Series or a
    one-column frame. The result has columns 'observed' then 'temperature' over the
    sorted union of both indexes, with temperature converted to the meter's tz. With
    `meter` None the result holds the temperature column alone, in its own tz and order.
    """
    if isinstance(temperature, pd.DataFrame):
        temperature = temperature.iloc[:, 0]
    temperature = temperature.rename("temperature")

    if meter is None:
        frame = temperature.to_frame()

        return frame

    if isinstance(meter, pd.DataFrame):
        meter = meter.iloc[:, 0]
    meter = meter.rename("observed")
    temperature = temperature.tz_convert(meter.index.tz)
    frame = pd.concat([meter, temperature], axis=1, sort=True)

    return frame


def nonstandard_settings_warnings(warnings):
    """Return the warnings in `warnings` whose qualified name is
    'eemeter.settings.nonstandard'."""
    matching = [w for w in warnings if w.qualified_name == "eemeter.settings.nonstandard"]

    return matching
