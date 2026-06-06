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
    ["features", "month_loadshape", "seasonal_day_of_week_loadshape",
     "seasonal_hourly_day_of_week_loadshape"],
)
def test_load_other_data_from_repo(data_type):
    """The CSV tutorial datasets load from the in-repo copy, indexed by id."""
    df = test_data.load_test_data(data_type)

    assert df.index.name == "id"
    assert len(df) > 0


def test_load_hourly_treatment_data_splits_baseline_reporting():
    """Hourly treatment data returns a (baseline, reporting) pair on an id/datetime index."""
    baseline, reporting = test_data.load_test_data("hourly_treatment_data")

    assert list(baseline.columns) == ["temperature", "ghi", "observed"]
    assert baseline.index.names == ["id", "datetime"]
    assert len(reporting) == len(baseline)


@pytest.mark.parametrize("data_type", ["daily_treatment_data", "monthly_treatment_data"])
def test_load_aggregated_treatment_data(data_type):
    """Daily/monthly treatment data aggregates the hourly series without error."""
    baseline, reporting = test_data.load_test_data(data_type)

    assert "observed" in baseline.columns
    assert len(baseline) > 0
