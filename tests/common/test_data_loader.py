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
