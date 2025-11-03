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

import pandas as pd
from pathlib import Path

# Define the current directory
current_dir = Path(__file__).parent
data_dir = current_dir.parents[2] / "data"

def load_tutorial_data(data_type: str):
    data_type = data_type.lower()

    if data_type == "features":
        df = pd.read_csv(data_dir / "features.csv")
    
    elif data_type == "seasonal_hourly_day_of_week_loadshape":
        df = pd.read_csv(data_dir / "seasonal_hourly_day_of_week_loadshape.csv")
    
    elif data_type == "seasonal_day_of_week_loadshape":
        df = pd.read_csv(data_dir / "seasonal_day_of_week_loadshape.csv")
    
    elif data_type == "month_loadshape":
        df = pd.read_csv(data_dir / "month_loadshape.csv")
    
    elif data_type == "hourly_data":
        df = pd.read_parquet(data_dir / "hourly_data.parquet")

    else:
        raise ValueError(f"Data type {data_type} not recognized.")
    
    if data_type not in "hourly_data":
        df = df.set_index("id")
    
    return df