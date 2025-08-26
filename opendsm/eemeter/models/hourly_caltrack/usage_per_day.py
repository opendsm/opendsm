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

import pytz

from opendsm.eemeter.common.transform import day_counts
from opendsm.eemeter.common.warnings import EEMeterWarning

__all__ = (
    "DataSufficiency",
    "caltrack_sufficiency_criteria",
)


class DataSufficiency(object):
    """Contains the result of a data sufficiency check.

    Attributes
    ----------
    status : :any:`str`
        A string indicating the status of this result. Possible statuses:

        - ``'NO DATA'``: No baseline data was available.
        - ``'FAIL'``: Data did not meet criteria.
        - ``'PASS'``: Data met criteria.
    criteria_name : :any:`str`
        The name of the criteria method used to check for baseline data sufficiency.
    warnings : :any:`list` of :any:`eemeter.EEMeterWarning`
        A list of any warnings reported during the check for baseline data sufficiency.
    data : :any:`dict`
        A dictionary of data related to determining whether a warning should be generated.
    settings : :any:`dict`
        A dictionary of settings (keyword arguments) used.
    """

    def __init__(self, status, criteria_name, warnings=None, data=None, settings=None):
        self.status = status  # NO DATA | FAIL | PASS
        self.criteria_name = criteria_name

        if warnings is None:
            warnings = []
        self.warnings = warnings

        if data is None:
            data = {}
        self.data = data

        if settings is None:
            settings = {}
        self.settings = settings

    def __repr__(self):
        return (
            "DataSufficiency("
            "status='{status}', criteria_name='{criteria_name}')".format(
                status=self.status, criteria_name=self.criteria_name
            )
        )

    def json(self):
        """Return a JSON-serializable representation of this result.

        The output of this function can be converted to a serialized string
        with :any:`json.dumps`.
        """
        return {
            "status": self.status,
            "criteria_name": self.criteria_name,
            "warnings": [w.json() for w in self.warnings],
            "data": self.data,
            "settings": self.settings,
        }


def caltrack_sufficiency_criteria(
    data_quality,
    requested_start,
    requested_end,
    num_days=365,
    min_fraction_daily_coverage=0.9,  # TODO: needs to be per year
    min_fraction_hourly_temperature_coverage_per_period=0.9,
):
    """CalTRACK daily data sufficiency criteria.

    .. note::

        For CalTRACK compliance, ``min_fraction_daily_coverage`` must be set
        at ``0.9`` (section 2.2.1.2), and requested_start and requested_end must
        not be None (section 2.2.4).


    Parameters
    ----------
    data_quality : :any:`pandas.DataFrame`
        A DataFrame containing at least the column ``meter_value`` and the two
        columns ``temperature_null``, containing a count of null hourly
        temperature values for each meter value, and ``temperature_not_null``,
        containing a count of not-null hourly temperature values for each
        meter value. Should have a :any:`pandas.DatetimeIndex`.
    requested_start : :any:`datetime.datetime`, timezone aware (or :any:`None`)
        The desired start of the period, if any, especially if this is
        different from the start of the data. If given, warnings
        are reported on the basis of this start date instead of data start
        date. Must be explicitly set to ``None`` in order to use data start date.
    requested_end : :any:`datetime.datetime`, timezone aware (or :any:`None`)
        The desired end of the period, if any, especially if this is
        different from the end of the data. If given, warnings
        are reported on the basis of this end date instead of data end date.
        Must be explicitly set to ``None`` in order to use data end date.
    num_days : :any:`int`, optional
        Exact number of days allowed in data, including extent given by
        ``requested_start`` or ``requested_end``, if given.
    min_fraction_daily_coverage : :any:, optional
        Minimum fraction of days of data in total data extent for which data
        must be available.
    min_fraction_hourly_temperature_coverage_per_period=0.9,
        Minimum fraction of hours of temperature data coverage in a particular
        period. Anything below this causes the whole period to be considered
        considered missing.

    Returns
    -------
    data_sufficiency : :any:`eemeter.DataSufficiency`
        The an object containing sufficiency status and warnings for this data.
    """
    criteria_name = "caltrack_sufficiency_criteria"

    if data_quality.dropna().empty:
        return DataSufficiency(
            status="NO DATA",
            criteria_name=criteria_name,
            warnings=[
                EEMeterWarning(
                    qualified_name="eemeter.caltrack_sufficiency_criteria.no_data",
                    description=("No data available."),
                    data={},
                )
            ],
        )

    data_start = data_quality.index.min().tz_convert("UTC")
    data_end = data_quality.index.max().tz_convert("UTC")
    n_days_data = (data_end - data_start).days

    if requested_start is not None:
        # check for gap at beginning
        requested_start = requested_start.astimezone(pytz.UTC)
        n_days_start_gap = (data_start - requested_start).days
    else:
        n_days_start_gap = 0

    if requested_end is not None:
        # check for gap at end
        requested_end = requested_end.astimezone(pytz.UTC)
        n_days_end_gap = (requested_end - data_end).days
    else:
        n_days_end_gap = 0

    critical_warnings = []

    if n_days_end_gap < 0:
        # CalTRACK 2.2.4
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".extra_data_after_requested_end_date"
                ),
                description=("Extra data found after requested end date."),
                data={
                    "requested_end": requested_end.isoformat(),
                    "data_end": data_end.isoformat(),
                },
            )
        )
        n_days_end_gap = 0

    if n_days_start_gap < 0:
        # CalTRACK 2.2.4
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".extra_data_before_requested_start_date"
                ),
                description=("Extra data found before requested start date."),
                data={
                    "requested_start": requested_start.isoformat(),
                    "data_start": data_start.isoformat(),
                },
            )
        )
        n_days_start_gap = 0

    n_days_total = n_days_data + n_days_start_gap + n_days_end_gap

    n_negative_meter_values = data_quality.meter_value[
        data_quality.meter_value < 0
    ].shape[0]

    if n_negative_meter_values > 0:
        # CalTrack 2.3.5
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria" ".negative_meter_values"
                ),
                description=(
                    "Found negative meter data values, which may indicate presence"
                    " of solar net metering."
                ),
                data={"n_negative_meter_values": n_negative_meter_values},
            )
        )

    # TODO(philngo): detect and report unsorted or repeated values.

    # create masks showing which daily or billing periods meet criteria
    valid_meter_value_rows = data_quality.meter_value.notnull()
    valid_temperature_rows = (
        data_quality.temperature_not_null
        / (data_quality.temperature_not_null + data_quality.temperature_null)
    ) > min_fraction_hourly_temperature_coverage_per_period
    valid_rows = valid_meter_value_rows & valid_temperature_rows

    # get number of days per period - for daily this should be a series of ones
    row_day_counts = day_counts(data_quality.index)

    # apply masks, giving total
    n_valid_meter_value_days = int((valid_meter_value_rows * row_day_counts).sum())
    n_valid_temperature_days = int((valid_temperature_rows * row_day_counts).sum())
    n_valid_days = int((valid_rows * row_day_counts).sum())

    median = data_quality.meter_value.median()
    upper_quantile = data_quality.meter_value.quantile(0.75)
    lower_quantile = data_quality.meter_value.quantile(0.25)
    iqr = upper_quantile - lower_quantile
    extreme_value_limit = median + (3 * iqr)
    n_extreme_values = data_quality.meter_value[
        data_quality.meter_value > extreme_value_limit
    ].shape[0]
    max_value = float(data_quality.meter_value.max())

    if n_days_total > 0:
        fraction_valid_meter_value_days = n_valid_meter_value_days / float(n_days_total)
        fraction_valid_temperature_days = n_valid_temperature_days / float(n_days_total)
        fraction_valid_days = n_valid_days / float(n_days_total)
    else:
        # unreachable, I think.
        fraction_valid_meter_value_days = 0
        fraction_valid_temperature_days = 0
        fraction_valid_days = 0

    if n_days_total != num_days:
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".incorrect_number_of_total_days"
                ),
                description=("Total data span does not match the required value."),
                data={"num_days": num_days, "n_days_total": n_days_total},
            )
        )

    if fraction_valid_days < min_fraction_daily_coverage:
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".too_many_days_with_missing_data"
                ),
                description=(
                    "Too many days in data have missing meter data or"
                    " temperature data."
                ),
                data={"n_valid_days": n_valid_days, "n_days_total": n_days_total},
            )
        )

    if fraction_valid_meter_value_days < min_fraction_daily_coverage:
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".too_many_days_with_missing_meter_data"
                ),
                description=("Too many days in data have missing meter data."),
                data={
                    "n_valid_meter_data_days": n_valid_meter_value_days,
                    "n_days_total": n_days_total,
                },
            )
        )

    if fraction_valid_temperature_days < min_fraction_daily_coverage:
        critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria"
                    ".too_many_days_with_missing_temperature_data"
                ),
                description=("Too many days in data have missing temperature data."),
                data={
                    "n_valid_temperature_data_days": n_valid_temperature_days,
                    "n_days_total": n_days_total,
                },
            )
        )

    if len(critical_warnings) > 0:
        status = "FAIL"
    else:
        status = "PASS"

    non_critical_warnings = []
    if n_extreme_values > 0:
        # CalTRACK 2.3.6
        non_critical_warnings.append(
            EEMeterWarning(
                qualified_name=(
                    "eemeter.caltrack_sufficiency_criteria" ".extreme_values_detected"
                ),
                description=(
                    "Extreme values (greater than (median + (3 * IQR)),"
                    " must be flagged for manual review."
                ),
                data={
                    "n_extreme_values": n_extreme_values,
                    "median": median,
                    "upper_quantile": upper_quantile,
                    "lower_quantile": lower_quantile,
                    "extreme_value_limit": extreme_value_limit,
                    "max_value": max_value,
                },
            )
        )

    warnings = critical_warnings + non_critical_warnings
    sufficiency_data = {
        "extra_data_after_requested_end_date": {
            "requested_end": requested_end.isoformat() if requested_end else None,
            "data_end": data_end.isoformat(),
            "n_days_end_gap": n_days_end_gap,
        },
        "extra_data_before_requested_start_date": {
            "requested_start": requested_start.isoformat() if requested_start else None,
            "data_start": data_start.isoformat(),
            "n_days_start_gap": n_days_start_gap,
        },
        "negative_meter_values": {"n_negative_meter_values": n_negative_meter_values},
        "incorrect_number_of_total_days": {
            "num_days": num_days,
            "n_days_total": n_days_total,
        },
        "too_many_days_with_missing_data": {
            "n_valid_days": n_valid_days,
            "n_days_total": n_days_total,
        },
        "too_many_days_with_missing_meter_data": {
            "n_valid_meter_data_days": n_valid_meter_value_days,
            "n_days_total": n_days_total,
        },
        "too_many_days_with_missing_temperature_data": {
            "n_valid_temperature_data_days": n_valid_temperature_days,
            "n_days_total": n_days_total,
        },
        "extreme_values_detected": {
            "n_extreme_values": n_extreme_values,
            "median": median,
            "upper_quantile": upper_quantile,
            "lower_quantile": lower_quantile,
            "extreme_value_limit": extreme_value_limit,
            "max_value": max_value,
        },
    }

    return DataSufficiency(
        status=status,
        criteria_name=criteria_name,
        warnings=warnings,
        data=sufficiency_data,
        settings={
            "num_days": num_days,
            "min_fraction_daily_coverage": min_fraction_daily_coverage,
            "min_fraction_hourly_temperature_coverage_per_period": min_fraction_hourly_temperature_coverage_per_period,
        },
    )
