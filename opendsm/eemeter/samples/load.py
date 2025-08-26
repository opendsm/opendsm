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

import json

import pytz
from dateutil.parser import parse as parse_date
import importlib.resources

from ..utilities.io import meter_data_from_csv, temperature_data_from_csv

__all__ = ("samples", "load_sample")


def _load_sample_metadata():
    with importlib.resources.files("opendsm.eemeter.samples").joinpath(
        "metadata.json"
    ).open("rb") as f:
        metadata = json.loads(f.read())
    return metadata


def samples():
    """Load a list of sample data identifiers.

    Returns
    -------
    samples : :any:`list` of :any:`str`
        List of sample identifiers for use with :any:`opendsm.eemeter.load_sample`.
    """
    sample_metadata = _load_sample_metadata()
    return list(sorted(sample_metadata.keys()))


def load_sample(sample, tempF=True):
    """Load meter data, temperature data, and metadata for associated with a
    particular sample identifier. Note: samples are simulated, not real, data.

    Parameters
    ----------
    sample : :any:`str`
        Identifier of sample. Complete list can be obtained with
        :any:`opendsm.eemeter.samples`.
    tempF : :any 'bool'
        Flag regarding whether the sample temperature dataset is associated with Fahrenheit or Celsius.

    Returns
    -------
    meter_data, temperature_data, metadata : :any:`tuple` of :any:`pandas.DataFrame`, :any:`pandas.Series`, and :any:`dict`
        Meter data, temperature data, and metadata for this sample identifier.
    """
    if tempF == True:
        temp_units = "tempF"
    else:
        temp_units = "tempC"

    sample_metadata = _load_sample_metadata()
    metadata = sample_metadata.get(sample)
    if metadata is None:
        raise ValueError(
            "Sample not found: {}. Try one of these?\n{}".format(
                sample,
                "\n".join(
                    [" - {}".format(key) for key in sorted(sample_metadata.keys())]
                ),
            )
        )

    freq = metadata.get("freq")
    if freq not in ("hourly", "daily"):
        freq = None

    meter_data_filename = metadata["meter_data_filename"]
    with importlib.resources.files("opendsm.eemeter.samples").joinpath(
        meter_data_filename
    ).open("rb") as f:
        meter_data = meter_data_from_csv(f, gzipped=True, freq=freq)

    temperature_filename = metadata["temperature_filename"]
    with importlib.resources.files("opendsm.eemeter.samples").joinpath(
        temperature_filename
    ).open("rb") as f:
        temperature_data = temperature_data_from_csv(
            f, gzipped=True, freq="hourly", temp_col=temp_units
        )

    metadata["blackout_start_date"] = pytz.UTC.localize(
        parse_date(metadata["blackout_start_date"])
    )
    metadata["blackout_end_date"] = pytz.UTC.localize(
        parse_date(metadata["blackout_end_date"])
    )

    return meter_data, temperature_data, metadata
