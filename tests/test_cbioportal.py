"""Tests for the cBioPortal tool (HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import cbioportal


def test_cbioportal(monkeypatch, tmp_path):
    gene_resp = {"entrezGeneId": 3672, "hugoGeneSymbol": "ITGA1"}
    mut_resp = [{"sampleId": "S1"}, {"sampleId": "S2"}, {"sampleId": "S1"}]

    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return gene_resp if "/genes/" in url else mut_resp

        return R()

    monkeypatch.setattr(cbioportal.base.requests, "get", fake_get)
    tool = cbioportal.make_tool(DiskCache(tmp_path))
    out = tool.run({"gene": "ITGA1"})
    assert out["entrez_id"] == 3672
    assert out["n_mutated_samples"] == 2  # S1, S2 unique
