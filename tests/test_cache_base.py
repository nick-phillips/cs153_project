"""Tests for the disk cache and Tool base helpers."""

from biomarker_agent import cache as cache_mod
from biomarker_agent.tools import base


def test_disk_cache_roundtrip(tmp_path):
    c = cache_mod.DiskCache(tmp_path / "c")
    calls = {"n": 0}

    def produce():
        calls["n"] += 1
        return {"v": 42}

    assert c.get_or_set("k1", produce) == {"v": 42}
    assert c.get_or_set("k1", produce) == {"v": 42}
    assert calls["n"] == 1  # second call served from cache


def test_disk_cache_does_not_persist_errors(tmp_path):
    c = cache_mod.DiskCache(tmp_path / "c")
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        # first call "fails", second succeeds
        return {"error": "boom"} if calls["n"] == 1 else {"v": 7}

    assert c.get_or_set("k", flaky) == {"error": "boom"}  # not cached
    assert c.get_or_set("k", flaky) == {"v": 7}  # re-run, real value now cached
    assert c.get_or_set("k", flaky) == {"v": 7}  # served from cache
    assert calls["n"] == 2


def test_tool_run_catches_errors():
    def boom(**kwargs):
        raise RuntimeError("kaboom")

    t = base.Tool(name="boom", description="d", input_schema={"type": "object", "properties": {}}, handler=boom)
    out = t.run({})
    assert "error" in out
    assert "kaboom" in out["error"]


def test_http_get_json_mock(monkeypatch):
    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(base.requests, "get", lambda *a, **k: FakeResp())
    assert base.http_get_json("http://x", params={"a": 1}) == {"ok": True}
