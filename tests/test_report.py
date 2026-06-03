"""Tests for report rendering from the structured submit_report payload."""

import json

from biomarker_agent import report


SAMPLE = {
    "summary": "Apatinib response tracks an integrin/SWI-SNF axis.",
    "hypotheses": [
        {
            "rank": 1,
            "title": "SMARCD1 dependency sensitizes to apatinib",
            "features": ["CRISPR_SMARCD1"],
            "mechanism": "SWI-SNF subunit loss alters VEGFR signaling.",
            "novelty": "off-MOA",
            "confidence": 0.6,
            "evidence": {"internal": "r=0.30", "depmap": "selective", "literature": "3 papers"},
        }
    ],
    "caveats": ["small n"],
}


def test_render_markdown_and_json(tmp_path):
    out_dir = tmp_path / "interpretation"
    paths = report.write_report(SAMPLE, out_dir, compound_id="BRD:TEST-1")
    md = paths["markdown"].read_text()
    assert "SMARCD1 dependency" in md
    assert "off-MOA" in md
    assert "BRD:TEST-1" in md
    loaded = json.loads(paths["json"].read_text())
    assert loaded["hypotheses"][0]["rank"] == 1


def test_report_schema_is_valid_json_schema():
    from biomarker_agent import prompts
    assert prompts.REPORT_TOOL["name"] == "submit_report"
    assert "hypotheses" in prompts.REPORT_TOOL["input_schema"]["properties"]


def test_render_embeds_figures(tmp_path):
    from biomarker_agent import report
    payload = {
        "summary": "s",
        "hypotheses": [{
            "rank": 1, "title": "H", "features": ["CRISPR_SMARCD1"], "mechanism": "m",
            "novelty": "off-MOA", "confidence": 0.6,
            "evidence": {"internal": "r=0.3"},
            "figures": [{"path": "figures/feature_response__CRISPR_SMARCD1.png",
                         "caption": "SMARCD1 vs response"}],
        }],
    }
    out = report.write_report(payload, tmp_path / "interp", compound_id="BRD:TEST-1")
    md = out["markdown"].read_text()
    assert "![SMARCD1 vs response](figures/feature_response__CRISPR_SMARCD1.png)" in md


def test_report_schema_has_figures():
    from biomarker_agent import prompts
    hyp = prompts.REPORT_TOOL["input_schema"]["properties"]["hypotheses"]["items"]
    assert "figures" in hyp["properties"]
    assert "figures" not in hyp["required"]  # optional
