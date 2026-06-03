"""Tests for the Open Targets tool (HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import opentargets


def test_opentargets(monkeypatch, tmp_path):
    search_resp = {"data": {"search": {"hits": [{"id": "ENSG0001", "name": "ITGA1"}]}}}
    target_resp = {
        "data": {
            "target": {
                "approvedSymbol": "ITGA1",
                "tractability": [{"modality": "SM", "label": "Approved Drug", "value": True}],
                "associatedDiseases": {
                    "rows": [
                        {"disease": {"name": "cancer", "therapeuticAreas": [{"name": "neoplasm"}]},
                         "score": 0.42}
                    ]
                },
            }
        }
    }
    responses = iter([search_resp, target_resp])

    def fake_post(url, json=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self_inner):
                return next(responses)

        return R()

    monkeypatch.setattr(opentargets.base.requests, "post", fake_post)
    tool = opentargets.make_tool(DiskCache(tmp_path))
    out = tool.run({"gene": "ITGA1"})
    assert out["ensembl_id"] == "ENSG0001"
    assert out["tractability"][0]["modality"] == "SM"
    assert out["cancer_associations"][0]["score"] == 0.42
