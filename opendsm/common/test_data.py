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

from __future__ import annotations

from pathlib import Path

import pandas as pd

import platformdirs
import requests

from opendsm import __file__ as opendsm_file_path
from opendsm import __version__
from opendsm.common.const import TutorialDataChoice

DOWNLOAD_TIMEOUT_S = 60


def _data_ref(version: str) -> str:
    """Git ref holding the data files for a package version: its release tag, or master."""
    if version == "unknown":
        return "master"

    return f"v{version}"


# data/ ships only with the source repo, not the installed wheel. _resolve_file uses
# the in-repo copy when present (source checkouts, CI) and otherwise downloads the
# file from the release tag matching the installed version into a per-version cache
# directory, so one version's file is never served to another.
repo_data_dir = Path(opendsm_file_path).resolve().parent.parent / "data"
cache_dir = Path(platformdirs.user_cache_dir("opendsm")) / "data"
_raw_url = "https://raw.githubusercontent.com/opendsm/opendsm"
base_url = f"{_raw_url}/{_data_ref(__version__)}/data"
# unreleased installs can reference files added after their version's tag
fallback_url = f"{_raw_url}/master/data"


comparison_group_time_series = [
    TutorialDataChoice.HOURLY_COMPARISON_GROUP_DATA,
    TutorialDataChoice.DAILY_COMPARISON_GROUP_DATA,
    TutorialDataChoice.MONTHLY_COMPARISON_GROUP_DATA,
]

treatment_time_series = [
    TutorialDataChoice.HOURLY_TREATMENT_DATA,
    TutorialDataChoice.DAILY_TREATMENT_DATA,
    TutorialDataChoice.MONTHLY_TREATMENT_DATA,
]


def load_test_data(data_type: str):
    """Returns back tutorial data of the given data type as a dataframe

    Args:
        data_type (str): Must be one of the following:
            - "features"
            - "seasonal_hourly_day_of_week_loadshape"
            - "seasonal_day_of_week_loadshape"
            - "month_loadshape"
            - "hourly_data"
            - "daily_treatment_data"
            - "monthly_treatment_data"

    Returns:
        (dataframe): Returns a dataframe
    """

    # remove all "_" and " " from string and convert to lowercase
    data_type = data_type.lower()
    data_type = data_type.replace("_", "").replace(" ", "")

    valid_list = [k.value for k in TutorialDataChoice]
    keys = [k.lower() for k in TutorialDataChoice.__members__.keys()]

    if data_type not in valid_list:
        raise ValueError(
            f"Data type {data_type} not recognized. \nMust be one of {keys}."
        )

    if data_type in [*comparison_group_time_series, *treatment_time_series]:
        return _load_time_series_data(data_type)

    else:
        return _load_other_data(data_type)


def _load_time_series_data(data_type):
    if data_type in treatment_time_series:
        df = _load_file("hourly_data_0.parquet")

    elif data_type in comparison_group_time_series:
        raise NotImplementedError(
            "Comparison-group tutorial data (hourly_data_1.parquet, hourly_data_2.parquet) "
            "is not yet available."
        )

    # localize datetime and convert to CST
    df = df.reset_index()
    df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    df["datetime"] = df["datetime"] + pd.Timedelta(hours=5)
    df["datetime"] = df["datetime"].dt.tz_convert("America/Chicago")
    df = df.set_index(["id", "datetime"])

    df_baseline = df[["temperature", "ghi_baseline", "observed_baseline"]]
    df_baseline = df_baseline.rename(columns={"observed_baseline": "observed", "ghi_baseline": "ghi"})

    df_reporting = df[["temperature", "ghi_reporting", "observed_reporting"]]
    df_reporting = df_reporting.rename(columns={"observed_reporting": "observed", "ghi_reporting": "ghi"})

    df_reporting = df_reporting.reset_index()
    df_reporting["datetime"] = df_reporting["datetime"] + pd.Timedelta(days=365)
    df_reporting = df_reporting.set_index(["id", "datetime"])

    if "daily" in data_type:
        df_baseline = _aggregate_hourly_data(df_baseline, "D")
        df_reporting = _aggregate_hourly_data(df_reporting, "D")

    elif "monthly" in data_type:
        df_baseline = _aggregate_hourly_data(df_baseline, "MS")
        df_reporting = _aggregate_hourly_data(df_reporting, "MS")

    return df_baseline, df_reporting


def _aggregate_hourly_data(df, agg):
    df_agg = df.reset_index().set_index("datetime").groupby("id")
    df_agg_temperature = df_agg["temperature"].resample("D").mean()
    df_agg_observed = df_agg["observed"].resample(agg).sum()

    if agg == "MS":
        df_agg_observed = df_agg_observed.reindex(df_agg_temperature.index)

    df = pd.concat([df_agg_temperature, df_agg_observed], axis=1)
    df = df.reset_index().set_index(["id", "datetime"])

    return df


def _load_other_data(data_type):
    if data_type == TutorialDataChoice.FEATURES:
        df = _load_file("features.csv")

    elif data_type == TutorialDataChoice.SEASONAL_HOUR_DAY_WEEK_LOADSHAPE:
        df = _load_file("seasonal_hourly_day_of_week_loadshape.csv")

    elif data_type == TutorialDataChoice.SEASONAL_DAY_WEEK_LOADSHAPE:
        df = _load_file("seasonal_day_of_week_loadshape.csv")

    elif data_type == TutorialDataChoice.MONTH_LOADSHAPE:
        df = _load_file("month_loadshape.csv")

    df = df.set_index("id")

    return df


def _load_file(name: str):
    if name.endswith(".parquet"):
        # checked before _resolve_file so nothing is downloaded that cannot be read
        try:
            import pyarrow  # noqa: F401  # optional dependency from the tutorial extra
        except ImportError as e:
            raise ImportError(
                f'Reading {name} requires pyarrow: pip install "opendsm[tutorial]"'
            ) from e

    source = _resolve_file(name)

    if name.endswith(".csv"):
        df = pd.read_csv(source)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(source, engine="pyarrow")
    else:
        raise ValueError(f"Unsupported tutorial-data file type: {name}")

    return df


def _resolve_file(name: str) -> Path:
    """Return a local path to the data file, downloading + caching it if absent.

    The in-repo data/ copy is used in source checkouts and CI; otherwise the file
    is fetched once from `base_url` into `cache_dir / __version__`, or from
    `fallback_url` when the release tag does not hold it.
    """

    repo_file = repo_data_dir / name
    if repo_file.exists():
        return repo_file

    cache_file = cache_dir / __version__ / name
    if not cache_file.exists():
        try:
            _download(f"{base_url}/{name}", cache_file)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                raise
            _download(f"{fallback_url}/{name}", cache_file)

    return cache_file


def _download(url: str, dest: Path) -> None:
    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_S)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    # written to a temporary name so an interrupted write never leaves a partial cache entry
    partial = dest.with_name(dest.name + ".part")
    partial.write_bytes(response.content)
    partial.replace(dest)


if __name__ == "__main__":
    df = load_test_data("hourly_treatment_data")
    print(df.index.get_level_values(0).nunique())
    print(df.head())
