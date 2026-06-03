"""Tests for the literature tool (PubMed default, HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import literature


def _patch(monkeypatch, search_resp, summary_resp):
    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return summary_resp if "esummary" in url else search_resp

        return R()

    monkeypatch.setattr(literature.base.requests, "get", fake_get)


def test_pubmed_counts_and_titles(monkeypatch, tmp_path):
    search = {"esearchresult": {"count": "37", "idlist": ["111", "222"]}}
    summary = {"result": {"uids": ["111", "222"],
                          "111": {"title": "MRE11 in mitosis", "pubdate": "2021 May"},
                          "222": {"title": "Spindle stress and DDR", "pubdate": "2019"}}}
    _patch(monkeypatch, search, summary)
    tool = literature.make_tool(DiskCache(tmp_path), backend="pubmed")
    out = tool.run({"gene": "MRE11", "context_terms": ["cancer", "mitosis"]})
    assert out["count"] == 37
    assert out["pmids"] == ["111", "222"]
    assert "MRE11" in out["query"]
    # now returns actual titles + years the agent can cite
    titles = [p["title"] for p in out["top_papers"]]
    assert "MRE11 in mitosis" in titles
    assert out["top_papers"][0]["year"] == "2021"


def test_pubmed_zero_hits_no_summary(monkeypatch, tmp_path):
    search = {"esearchresult": {"count": "0", "idlist": []}}
    _patch(monkeypatch, search, {})
    tool = literature.make_tool(DiskCache(tmp_path), backend="pubmed")
    out = tool.run({"gene": "NOPE", "context_terms": ["drugX"]})
    assert out["count"] == 0
    assert out["pmids"] == []
    assert out["top_papers"] == []
