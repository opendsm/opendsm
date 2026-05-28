# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_create_caltrack_billing_design_matrix design_matrix_shape'] = [
    274,
    6
]

snapshots['test_create_caltrack_billing_design_matrix design_matrix_sum'] = 293106.29

snapshots['test_create_caltrack_daily_design_matrix design_matrix_sum'] = 158592.6

snapshots['test_create_caltrack_hourly_preliminary_design_matrix design_matrix_sum'] = 252544.4

snapshots['test_create_caltrack_hourly_segmented_design_matrices dec-jan-feb-weighted_columns'] = [
    'bin_0_occupied',
    'bin_0_unoccupied',
    'bin_1_occupied',
    'bin_1_unoccupied',
    'bin_2_occupied',
    'bin_2_unoccupied',
    'bin_3_occupied',
    'bin_3_unoccupied',
    'bin_4_occupied',
    'bin_4_unoccupied',
    'bin_5_occupied',
    'bin_5_unoccupied',
    'bin_6_occupied',
    'bin_6_unoccupied',
    'hour_of_week',
    'meter_value',
    'weight'
]

snapshots['test_create_caltrack_hourly_segmented_design_matrices dec-jan-feb-weighted_shape'] = [
    1000,
    17
]

snapshots['test_create_caltrack_hourly_segmented_design_matrices dec-jan-feb-weighted_sum'] = 235311.32

snapshots['test_create_caltrack_hourly_segmented_design_matrices mar-apr-may-weighted_columns'] = [
    'bin_0_occupied',
    'bin_0_unoccupied',
    'hour_of_week',
    'meter_value',
    'weight'
]

snapshots['test_create_caltrack_hourly_segmented_design_matrices mar-apr-may-weighted_shape'] = [
    1000,
    5
]

snapshots['test_create_caltrack_hourly_segmented_design_matrices mar-apr-may-weighted_sum'] = 304168.01
