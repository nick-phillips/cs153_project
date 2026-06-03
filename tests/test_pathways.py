"""Tests for the Reactome pathways tool (HTTP mocked).

Mirrors the real two-call flow: search → resolve human UniProt accession, then
the mapping endpoint → pathways. Response shapes match the live Reactome API
(species is a list; pathway names may carry HTML highlighting tags).
"""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import pathways

SEARCH_RESP = {
    "results": [
        {
            "entries": [
                # non-human entry that must be skipped
                {"databaseName": "UniProt", "species": ["Gallus gallus"],
                 "referenceIdentifier": "X0X0X0"},
                # the human ReferenceGeneProduct we want
                {"databaseName": "UniProt", "species": ["Homo sapiens"],
                 "referenceIdentifier": "P56199", "name": "<span>ITGA1</span>"},
            ]
        }
    ]
}

MAPPING_RESP = [
    {"stId": "R-HSA-216083", "displayName": "Integrin cell surface interactions"},
    {"stId": "R-HSA-3000157", "displayName": "Laminin interactions"},
]


def _fake_get_factory():
    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return MAPPING_RESP if "/mapping/" in url else SEARCH_RESP

        return R()

    return fake_get


def test_pathways_resolves_uniprot_then_maps(monkeypatch, tmp_path):
    monkeypatch.setattr(pathways.base.requests, "get", _fake_get_factory())
    tool = pathways.make_tool(DiskCache(tmp_path))
    out = tool.run({"gene": "ITGA1"})
    assert out["uniprot"] == "P56199"  # picked the human accession, not the chicken one
    assert out["n_pathways"] == 2
    assert "Integrin cell surface interactions" in [p["name"] for p in out["pathways"]]


def test_pathways_no_human_uniprot(monkeypatch, tmp_path):
    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {"results": []}

        return R()

    monkeypatch.setattr(pathways.base.requests, "get", fake_get)
    tool = pathways.make_tool(DiskCache(tmp_path))
    out = tool.run({"gene": "NOSUCHGENE"})
    assert out["n_pathways"] == 0
    assert out["pathways"] == []
