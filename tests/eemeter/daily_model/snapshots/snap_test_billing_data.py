# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_billing_baseline_data_with_specific_daily_input disqualification'] = [
    'eemeter.sufficiency_criteria.incorrect_number_of_total_days',
    'eemeter.sufficiency_criteria.missing_monthly_temperature_data'
]

snapshots['test_billing_baseline_data_with_specific_daily_input observed_sum'] = 346171.72

snapshots['test_billing_baseline_data_with_specific_daily_input warnings'] = [
    'eemeter.data_quality.utc_index',
    'eemeter.sufficiency_criteria.inferior_model_usage',
    'eemeter.sufficiency_criteria.missing_high_frequency_temperature_data'
]

snapshots['test_billing_baseline_data_with_specific_hourly_input disqualification'] = [
    'eemeter.sufficiency_criteria.missing_monthly_temperature_data',
    'eemeter.sufficiency_criteria.no_data',
    'eemeter.sufficiency_criteria.offcycle_reads_in_billing_monthly_data',
    'eemeter.sufficiency_criteria.too_many_days_with_missing_joint_data',
    'eemeter.sufficiency_criteria.too_many_days_with_missing_observed_data',
    'eemeter.sufficiency_criteria.too_many_days_with_missing_temperature_data'
]

snapshots['test_billing_baseline_data_with_specific_hourly_input observed_sum'] = 345816.84

snapshots['test_billing_baseline_data_with_specific_hourly_input warnings'] = [
    'eemeter.data_quality.utc_index',
    'eemeter.sufficiency_criteria.inferior_model_usage',
    'eemeter.sufficiency_criteria.missing_high_frequency_temperature_data'
]

snapshots['test_billing_baseline_data_with_specific_missing_daily_input disqualification'] = [
    'eemeter.sufficiency_criteria.incorrect_number_of_total_days',
    'eemeter.sufficiency_criteria.missing_monthly_temperature_data'
]

snapshots['test_billing_baseline_data_with_specific_missing_daily_input warnings'] = [
    'eemeter.data_quality.utc_index',
    'eemeter.sufficiency_criteria.inferior_model_usage',
    'eemeter.sufficiency_criteria.missing_high_frequency_temperature_data'
]

snapshots['test_billing_baseline_data_with_specific_monthly_input disqualification'] = [
]

snapshots['test_billing_baseline_data_with_specific_monthly_input observed_sum'] = 306958.19

snapshots['test_billing_baseline_data_with_specific_monthly_input warnings'] = [
    'eemeter.data_quality.utc_index'
]
