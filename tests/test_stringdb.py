"""Tests for the STRING-DB tool (HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import stringdb


def test_stringdb_enrichment(monkeypatch, tmp_path):
    network = [{"preferredName_A": "ITGA1", "preferredName_B": "SMARCD1", "score": 0.6}]
    enrich = [
        {"category": "Process", "term": "GO:1", "description": "cell adhesion",
         "number_of_genes": 2, "fdr": 1e-4, "preferredNames": ["ITGA1", "SMARCD1"]}
    ]

    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return enrich if "enrichment" in url else network

        return R()

    monkeypatch.setattr(stringdb.base.requests, "get", fake_get)
    tool = stringdb.make_tool(DiskCache(tmp_path))
    out = tool.run({"genes": ["ITGA1", "SMARCD1"]})
    assert out["n_genes"] == 2
    assert out["n_interactions"] == 1
    assert any("adhesion" in e["description"] for e in out["enrichment"])
