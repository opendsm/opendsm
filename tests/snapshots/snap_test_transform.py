# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_as_freq_daily_all_nones as_daily_shape'] = [
    701
]

snapshots['test_as_freq_daily_all_nones meter_data_shape'] = [
    24,
    1
]

snapshots['test_as_freq_daily_all_nones_instantaneous as_daily_shape'] = [
    701
]

snapshots['test_as_freq_daily_all_nones_instantaneous meter_data_shape'] = [
    24,
    1
]

snapshots['test_as_freq_daily_temperature as_daily_shape'] = [
    732
]

snapshots['test_as_freq_daily_temperature temperature_data_shape'] = [
    17518
]

snapshots['test_as_freq_daily_temperature_monthly as_daily_mean___round'] = 121.0

snapshots['test_as_freq_daily_temperature_monthly as_daily_shape'] = [
    731
]

snapshots['test_as_freq_daily_temperature_monthly temperature_data_shape'] = [
    25
]

snapshots['test_as_freq_month_start_temperature as_month_start_mean___round'] = 119.8

snapshots['test_as_freq_month_start_temperature as_month_start_shape'] = [
    26
]

snapshots['test_as_freq_month_start_temperature temperature_data_shape'] = [
    17518
]

snapshots['test_as_freq_not_series meter_data_shape'] = [
    24,
    1
]

snapshots['test_clean_caltrack_billing_daily_data_daily cleaned_data_shape'] = [
    730,
    1
]

snapshots['test_clean_caltrack_billing_daily_data_daily_local_tz cleaned_data_shape'] = [
    730,
    1
]

snapshots['test_clean_caltrack_billing_daily_data_hourly cleaned_data_shape'] = [
    732,
    1
]

snapshots['test_clean_caltrack_daily_data_hourly cleaned_data_shape'] = [
    732,
    1
]

snapshots['test_clean_caltrack_daily_data_hourly_local_tz cleaned_data_shape'] = [
    731,
    1
]

snapshots['test_day_counts_empty_series counts_shape'] = [
    0
]

snapshots['test_format_energy_data_for_caltrack_billing df_reformatted_columns_len'] = 1

snapshots['test_format_energy_data_for_caltrack_daily df_reformatted_columns_len'] = 1

snapshots['test_format_energy_data_for_caltrack_hourly df_reformatted_columns_len'] = 1

snapshots['test_get_baseline_data warnings_len'] = 0

snapshots['test_get_baseline_data_with_timezones warnings_len'] = 0

snapshots['test_get_reporting_data warnings_len'] = 0

snapshots['test_get_reporting_data_with_timezones warnings_len'] = 0

snapshots['test_get_terms_empty_index_input terms_len'] = 0

snapshots['test_remove_duplicates_df df_dedupe_shape'] = [
    2,
    1
]

snapshots['test_remove_duplicates_df df_shape'] = [
    3,
    1
]

snapshots['test_remove_duplicates_series series_dedupe_shape'] = [
    2
]

snapshots['test_remove_duplicates_series series_shape'] = [
    3
]
