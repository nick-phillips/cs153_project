"""Tests for the literature tool (PubMed default, HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import literature


_EFETCH_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>111</PMID>
      <Article><Abstract>
        <AbstractText Label="BACKGROUND">MRE11 acts in the MRN complex.</AbstractText>
        <AbstractText Label="RESULTS">It is required for mitotic fidelity.</AbstractText>
      </Abstract></Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>222</PMID>
      <Article><Abstract>
        <AbstractText>Spindle stress triggers the DNA damage response.</AbstractText>
      </Abstract></Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def _patch(monkeypatch, search_resp, summary_resp, efetch_xml=_EFETCH_XML):
    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200
            text = efetch_xml

            def raise_for_status(self):
                pass

            def json(self):
                return summary_resp if "esummary" in url else search_resp

        return R()

    monkeypatch.setattr(literature.base.requests, "get", fake_get)


def test_pubmed_counts_titles_and_abstracts(monkeypatch, tmp_path):
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
    # returns titles + years the agent can cite
    titles = [p["title"] for p in out["top_papers"]]
    assert "MRE11 in mitosis" in titles
    assert out["top_papers"][0]["year"] == "2021"
    # and now the actual abstract text (labeled sections concatenated)
    p111 = next(p for p in out["top_papers"] if p["pmid"] == "111")
    assert "MRN complex" in p111["abstract"]
    assert "mitotic fidelity" in p111["abstract"]
    p222 = next(p for p in out["top_papers"] if p["pmid"] == "222")
    assert "DNA damage response" in p222["abstract"]


def test_pubmed_abstract_truncated(monkeypatch, tmp_path):
    long_xml = (
        '<?xml version="1.0"?><PubmedArticleSet><PubmedArticle><MedlineCitation>'
        '<PMID>111</PMID><Article><Abstract><AbstractText>'
        + ("A" * 5000)
        + "</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    )
    search = {"esearchresult": {"count": "1", "idlist": ["111"]}}
    summary = {"result": {"uids": ["111"], "111": {"title": "T", "pubdate": "2020"}}}
    _patch(monkeypatch, search, summary, efetch_xml=long_xml)
    tool = literature.make_tool(DiskCache(tmp_path), backend="pubmed")
    out = tool.run({"gene": "MRE11"})
    abstract = out["top_papers"][0]["abstract"]
    assert len(abstract) <= literature.ABSTRACT_MAXLEN + 1  # +1 for the ellipsis
    assert abstract.endswith("…")


def test_pubmed_zero_hits_no_summary(monkeypatch, tmp_path):
    search = {"esearchresult": {"count": "0", "idlist": []}}
    _patch(monkeypatch, search, {})
    tool = literature.make_tool(DiskCache(tmp_path), backend="pubmed")
    out = tool.run({"gene": "NOPE", "context_terms": ["drugX"]})
    assert out["count"] == 0
    assert out["pmids"] == []
    assert out["top_papers"] == []
