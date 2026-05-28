# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_daily_baseline_data_with_missing_specific_daily_input df_length'] = 365

snapshots['test_daily_baseline_data_with_missing_specific_daily_input disqualification'] = [
]

snapshots['test_daily_baseline_data_with_missing_specific_daily_input observed_sum'] = 316473.43

snapshots['test_daily_baseline_data_with_missing_specific_daily_input warnings'] = [
    'eemeter.data_quality.utc_index',
    'eemeter.sufficiency_criteria.extreme_values_detected'
]

snapshots['test_daily_baseline_data_with_specific_daily_input df_length'] = 365

snapshots['test_daily_baseline_data_with_specific_daily_input disqualification'] = [
]

snapshots['test_daily_baseline_data_with_specific_daily_input observed_sum'] = 346171.72

snapshots['test_daily_baseline_data_with_specific_daily_input warnings'] = [
    'eemeter.data_quality.utc_index',
    'eemeter.sufficiency_criteria.extreme_values_detected'
]

snapshots['test_daily_baseline_data_with_specific_hourly_input df_length'] = 365

snapshots['test_daily_baseline_data_with_specific_hourly_input disqualification'] = [
]

snapshots['test_daily_baseline_data_with_specific_hourly_input observed_sum'] = 346099.72

snapshots['test_daily_baseline_data_with_specific_hourly_input warnings'] = [
    'eemeter.data_quality.utc_index',
    'eemeter.sufficiency_criteria.extreme_values_detected'
]

snapshots['test_non_ns_datetime_index df_length'] = 365

snapshots['test_offset_aggregations_hourly df_length'] = 365

snapshots['test_offset_temperature_aggregations[Europe/London-13] abs_diff_Europe/London_13'] = 0.7619

snapshots['test_offset_temperature_aggregations[US/Eastern-8] abs_diff_US/Eastern_8'] = 0.4583

snapshots['test_offset_temperature_aggregations[US/Pacific-3] abs_diff_US/Pacific_3'] = 0.3706
