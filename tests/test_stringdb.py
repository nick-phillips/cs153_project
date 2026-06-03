"""Tests for the STRING-DB tool (HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import stringdb


def _patch(monkeypatch, network, enrich):
    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return enrich if "enrichment" in url else network

        return R()

    monkeypatch.setattr(stringdb.base.requests, "get", fake_get)


def test_stringdb_enrichment(monkeypatch, tmp_path):
    network = [{"preferredName_A": "ITGA1", "preferredName_B": "SMARCD1", "score": 0.6}]
    enrich = [
        {"category": "Process", "term": "GO:1", "description": "cell adhesion",
         "number_of_genes": 2, "fdr": 1e-4, "preferredNames": ["ITGA1", "SMARCD1"]}
    ]
    _patch(monkeypatch, network, enrich)
    tool = stringdb.make_tool(DiskCache(tmp_path))
    out = tool.run({"genes": ["ITGA1", "SMARCD1"]})
    assert out["submitted_genes"] == ["ITGA1", "SMARCD1"]
    assert out["n_interactions"] == 1
    assert set(out["neighborhood"]) == {"ITGA1", "SMARCD1"}
    assert any("adhesion" in e["description"] for e in out["enrichment"])


def test_stringdb_single_gene_expands_neighborhood(monkeypatch, tmp_path):
    # one submitted gene; STRING returns partners via add_nodes expansion
    network = [
        {"preferredName_A": "MRE11", "preferredName_B": "RAD50", "score": 0.99},
        {"preferredName_A": "MRE11", "preferredName_B": "NBN", "score": 0.97},
    ]
    enrich = [{"category": "Process", "term": "GO:2", "description": "double-strand break repair",
               "number_of_genes": 3, "fdr": 1e-8, "preferredNames": ["MRE11", "RAD50", "NBN"]}]
    _patch(monkeypatch, network, enrich)
    tool = stringdb.make_tool(DiskCache(tmp_path))
    out = tool.run({"genes": ["MRE11"]})
    # neighborhood now includes the partners discovered by expansion
    assert set(out["neighborhood"]) == {"MRE11", "RAD50", "NBN"}
    assert any("break repair" in e["description"] for e in out["enrichment"])


def test_stringdb_empty_genes(monkeypatch, tmp_path):
    _patch(monkeypatch, [], [])
    tool = stringdb.make_tool(DiskCache(tmp_path))
    assert "error" in tool.run({"genes": []})
