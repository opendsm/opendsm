import pandas as pd
import pytest

from opendsm.common import test_data



class _FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def test_resolve_file_prefers_in_repo_copy(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "features.csv").write_bytes(b"id,x\n1,2\n")
    monkeypatch.setattr(test_data, "repo_data_dir", repo)
    monkeypatch.setattr(test_data, "cache_dir", tmp_path / "cache")

    def _no_network(url):
        raise AssertionError(f"unexpected download: {url}")

    monkeypatch.setattr(test_data.requests, "get", _no_network)

    assert test_data._resolve_file("features.csv") == repo / "features.csv"


def test_resolve_file_downloads_and_caches_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(test_data, "repo_data_dir", tmp_path / "repo")
    cache = tmp_path / "cache"
    monkeypatch.setattr(test_data, "cache_dir", cache)

    calls = []

    def _fake_get(url):
        calls.append(url)
        return _FakeResponse(b"id,x\n1,2\n")

    monkeypatch.setattr(test_data.requests, "get", _fake_get)

    first = test_data._resolve_file("features.csv")

    assert first == cache / "features.csv"
    assert first.read_bytes() == b"id,x\n1,2\n"
    assert len(calls) == 1
    assert calls[0].endswith("/data/features.csv")


def test_resolve_file_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(test_data, "repo_data_dir", tmp_path / "repo")
    monkeypatch.setattr(test_data, "cache_dir", tmp_path / "cache")

    calls = []

    def _fake_get(url):
        calls.append(url)
        return _FakeResponse(b"id,x\n1,2\n")

    monkeypatch.setattr(test_data.requests, "get", _fake_get)

    test_data._resolve_file("features.csv")
    test_data._resolve_file("features.csv")

    assert len(calls) == 1


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
