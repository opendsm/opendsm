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

import sys

import pandas as pd
import pytest
import requests

from opendsm.common import test_data



@pytest.fixture
def fake_download(monkeypatch):
    """Replace the network download with one that records calls and writes a tiny CSV."""
    calls = []

    def _fake(url, dest):
        calls.append((url, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"id,x\n1,2\n")

    monkeypatch.setattr(test_data, "_download", _fake)

    return calls


@pytest.fixture
def no_repo_copy(tmp_path, monkeypatch):
    """Point the loader at an absent data/ dir and an empty cache; return the cache dir."""
    cache = tmp_path / "cache"
    monkeypatch.setattr(test_data, "repo_data_dir", tmp_path / "repo")
    monkeypatch.setattr(test_data, "cache_dir", cache)

    return cache


def test_resolve_file_prefers_in_repo_copy(tmp_path, monkeypatch, fake_download):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "features.csv").write_bytes(b"id,x\n1,2\n")
    monkeypatch.setattr(test_data, "repo_data_dir", repo)
    monkeypatch.setattr(test_data, "cache_dir", tmp_path / "cache")

    assert test_data._resolve_file("features.csv") == repo / "features.csv"
    assert fake_download == [], f"unexpected download: {fake_download}"


def test_resolve_file_downloads_and_caches_when_absent(no_repo_copy, fake_download):
    first = test_data._resolve_file("features.csv")

    assert first.read_bytes() == b"id,x\n1,2\n"
    assert len(fake_download) == 1


def test_resolve_file_uses_cache_on_second_call(no_repo_copy, fake_download):
    test_data._resolve_file("features.csv")
    test_data._resolve_file("features.csv")

    assert len(fake_download) == 1


def test_resolve_file_downloads_from_base_url(no_repo_copy, fake_download, monkeypatch):
    monkeypatch.setattr(test_data, "base_url", "https://example.invalid/data")

    test_data._resolve_file("features.csv")

    url, _ = fake_download[0]
    assert url == "https://example.invalid/data/features.csv"


@pytest.mark.parametrize(
    "version, ref",
    [("1.2.7", "v1.2.7"), ("2.0.0", "v2.0.0"), ("unknown", "master")],
)
def test_data_ref_pins_release_tag_or_master(version, ref):
    assert test_data._data_ref(version) == ref


def test_default_base_url_is_pinned_to_installed_version():
    expected = test_data._data_ref(test_data.__version__)

    assert test_data.base_url == (
        f"https://raw.githubusercontent.com/opendsm/opendsm/{expected}/data"
    )


def _http_error(status):
    response = requests.Response()
    response.status_code = status

    return requests.HTTPError(response=response)


def test_resolve_file_falls_back_to_master_when_tag_lacks_file(
    no_repo_copy, monkeypatch
):
    urls = []

    def _download(url, dest):
        urls.append(url)
        if url.startswith(test_data.base_url):
            raise _http_error(404)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"id,x\n1,2\n")

    monkeypatch.setattr(test_data, "_download", _download)

    path = test_data._resolve_file("features.csv")

    assert urls == [
        f"{test_data.base_url}/features.csv",
        f"{test_data.fallback_url}/features.csv",
    ]
    assert path.exists()


def test_resolve_file_does_not_fall_back_on_other_http_errors(
    no_repo_copy, monkeypatch
):
    def _download(url, dest):
        raise _http_error(500)

    monkeypatch.setattr(test_data, "_download", _download)

    with pytest.raises(requests.HTTPError):
        test_data._resolve_file("features.csv")


def test_download_failure_leaves_no_cache_file_and_sets_timeout(tmp_path, monkeypatch):
    seen = {}

    class _Response:
        content = b"<html>Not Found</html>"

        def raise_for_status(self):
            raise _http_error(404)

    def _get(url, **kwargs):
        seen.update(kwargs)

        return _Response()

    monkeypatch.setattr(test_data.requests, "get", _get)
    dest = tmp_path / "v" / "features.csv"

    with pytest.raises(requests.HTTPError):
        test_data._download("https://example.invalid/features.csv", dest)

    assert not dest.exists(), "error body was written to the cache path"
    assert seen["timeout"] == test_data.DOWNLOAD_TIMEOUT_S


def test_download_writes_file_without_leaving_partial(tmp_path, monkeypatch):
    class _Response:
        content = b"id,x\n1,2\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(test_data.requests, "get", lambda url, **kw: _Response())
    dest = tmp_path / "v" / "features.csv"

    test_data._download("https://example.invalid/features.csv", dest)

    assert dest.read_bytes() == b"id,x\n1,2\n"
    assert [p.name for p in dest.parent.iterdir()] == ["features.csv"]


@pytest.mark.parametrize("version", ["1.2.7", "unknown"])
def test_cache_path_is_per_version(no_repo_copy, fake_download, monkeypatch, version):
    monkeypatch.setattr(test_data, "__version__", version)

    path = test_data._resolve_file("features.csv")

    assert path == no_repo_copy / version / "features.csv"
    assert fake_download[0][1] == path


def test_cached_file_of_another_version_is_not_reused(
    no_repo_copy, fake_download, monkeypatch
):
    stale = no_repo_copy / "1.0.0" / "features.csv"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"id,x\n9,9\n")
    monkeypatch.setattr(test_data, "__version__", "1.2.7")

    path = test_data._resolve_file("features.csv")

    assert path == no_repo_copy / "1.2.7" / "features.csv"
    assert len(fake_download) == 1


def test_parquet_without_pyarrow_names_extra_and_skips_download(
    no_repo_copy, fake_download, monkeypatch
):
    monkeypatch.setitem(sys.modules, "pyarrow", None)

    with pytest.raises(ImportError, match=r"opendsm\[tutorial\]") as excinfo:
        test_data._load_file("hourly_data_0.parquet")

    assert isinstance(excinfo.value.__cause__, ImportError)
    assert fake_download == [], f"download attempted without pyarrow: {fake_download}"


def test_load_test_data_unknown_type_raises():
    """An unrecognized data_type raises ValueError before any file access."""
    with pytest.raises(ValueError, match="not recognized"):
        test_data.load_test_data("not_a_real_dataset")


def test_load_test_data_comparison_group_not_implemented():
    """Comparison-group tutorial data is not yet available and raises."""
    with pytest.raises(NotImplementedError, match="not yet available"):
        test_data.load_test_data("hourly_comparison_group_data")


def test_load_file_rejects_unsupported_extension(tmp_path, monkeypatch):
    """A resolved file with an unsupported extension raises ValueError."""
    target = tmp_path / "data.txt"
    target.write_text("ignored")
    monkeypatch.setattr(test_data, "_resolve_file", lambda name: target)

    with pytest.raises(ValueError, match="Unsupported tutorial-data file type"):
        test_data._load_file("data.txt")


@pytest.mark.parametrize(
    "data_type",
    ["month_loadshape", "seasonal_day_of_week_loadshape",
     "seasonal_hourly_day_of_week_loadshape"],
)
def test_load_other_data_from_repo(data_type):
    """The CSV loadshape datasets load from the in-repo copy, indexed by id."""
    df = test_data.load_test_data(data_type)

    assert df.index.name == "id"
    assert len(df) > 0


# The fingerprint tests below pin the shape/scale of the committed tutorial
# datasets. They are deliberately exact so that any change to the in-repo data
# files (data/features.csv, data/hourly_data_0.parquet) is caught here rather
# than silently shifting downstream snapshots.

def test_features_dataset_fingerprint():
    """The features dataset has its committed shape and id count."""
    df = test_data.load_test_data("features")

    assert df.shape == (1200, 3)
    assert df.index.name == "id"
    assert df.index.nunique() == 1200


@pytest.mark.slow
def test_hourly_treatment_dataset_fingerprint():
    """Hourly treatment data has its committed structure, scale and date range."""
    baseline, reporting = test_data.load_test_data("hourly_treatment_data")

    assert list(baseline.columns) == ["temperature", "ghi", "observed"]
    assert baseline.index.names == ["id", "datetime"]
    assert baseline.index.get_level_values("id").nunique() == 100
    assert len(baseline) == 875900
    assert len(reporting) == len(baseline)

    assert baseline["observed"].sum() == pytest.approx(85072318.08, rel=1e-6)
    datetimes = baseline.index.get_level_values("datetime")
    assert datetimes.min().year == 2018
    assert datetimes.max().year == 2018


@pytest.mark.slow
@pytest.mark.parametrize("data_type", ["daily_treatment_data", "monthly_treatment_data"])
def test_aggregated_treatment_data_loads(data_type):
    """Daily/monthly treatment data aggregates the hourly series and stays non-empty."""
    baseline, reporting = test_data.load_test_data(data_type)

    assert "observed" in baseline.columns
    assert len(baseline) > 0


@pytest.mark.slow
def test_daily_aggregation_equals_sum_of_hourly():
    """A daily observed value equals the sum of that day's hourly observations."""
    hourly, _ = test_data.load_test_data("hourly_treatment_data")
    daily, _ = test_data.load_test_data("daily_treatment_data")

    meter = hourly.index.get_level_values("id")[0]
    hourly_meter = hourly.xs(meter, level="id")
    day = hourly_meter.index[0].floor("D")

    same_day = (hourly_meter.index >= day) & (hourly_meter.index < day + pd.Timedelta("1D"))
    hourly_day_sum = hourly_meter.loc[same_day, "observed"].sum()
    daily_value = daily.xs(meter, level="id").loc[day, "observed"]

    assert daily_value == pytest.approx(hourly_day_sum)
