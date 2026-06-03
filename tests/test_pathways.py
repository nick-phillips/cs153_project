"""Tests for the Reactome pathways tool (HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import pathways


def test_pathways(monkeypatch, tmp_path):
    resp = {
        "results": [
            {"typeName": "Pathway", "entries": [
                {"name": "Integrin signaling", "id": "R-HSA-1", "species": "Homo sapiens"},
                {"name": "ECM interactions", "id": "R-HSA-2", "species": "Homo sapiens"},
            ]}
        ]
    }

    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return resp

        return R()

    monkeypatch.setattr(pathways.base.requests, "get", fake_get)
    tool = pathways.make_tool(DiskCache(tmp_path))
    out = tool.run({"gene": "ITGA1"})
    assert "Integrin signaling" in [p["name"] for p in out["pathways"]]
