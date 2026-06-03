"""Tests for the literature tool (PubMed default, HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import literature


def test_pubmed_counts(monkeypatch, tmp_path):
    resp = {"esearchresult": {"count": "37", "idlist": ["111", "222"]}}

    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return resp

        return R()

    monkeypatch.setattr(literature.base.requests, "get", fake_get)
    tool = literature.make_tool(DiskCache(tmp_path), backend="pubmed")
    out = tool.run({"gene": "ITGA1", "context_terms": ["cancer", "apatinib"]})
    assert out["count"] == 37
    assert out["pmids"] == ["111", "222"]
    assert "ITGA1" in out["query"]
