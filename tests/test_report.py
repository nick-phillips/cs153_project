"""Tests for report rendering from the structured submit_report payload."""

import json

from biomarker_agent import report


SAMPLE = {
    "summary": "Apatinib response tracks an integrin/SWI-SNF axis.",
    "clear_hypothesis": True,
    "proposed_mechanisms": ["SWI/SNF (BAF) dependency creates a vulnerability apatinib exploits."],
    "proposed_biomarkers": ["High SMARCD1 CRISPR dependency → greater sensitivity."],
    "hypotheses": [
        {
            "rank": 1,
            "title": "SMARCD1 dependency sensitizes to apatinib",
            "kind": "both",
            "features": ["CRISPR_SMARCD1"],
            "mechanism": "SWI-SNF subunit loss alters VEGFR signaling.",
            "novelty": "off-MOA",
            "confidence": 0.6,
            "evidence": {"internal": "r=0.30", "depmap": "selective", "literature": "3 papers"},
        }
    ],
    "caveats": ["small n"],
}

META = {"drug_name": "APATINIB", "moa": "RET TKI", "targets": "RET, KDR, KIT",
        "n_samples": 530,
        "performance": {"selected_refit_oob_pearson": 0.323, "bootstrap_pred_pearson": 0.204,
                        "baseline_pred_pearson": 0.177}}


def test_render_markdown_and_json(tmp_path):
    out_dir = tmp_path / "interpretation"
    paths = report.write_report(SAMPLE, out_dir, compound_id="BRD:TEST-1", meta=META)
    md = paths["markdown"].read_text()
    assert "SMARCD1 dependency" in md
    assert "off-MOA" in md
    assert "BRD:TEST-1" in md
    # header: performance + drug context surfaced deterministically
    assert "APATINIB" in md
    assert "selected-refit r=0.323" in md
    assert "n=530 cell lines" in md
    # headline sections present
    assert "## Summary" in md
    assert "### Proposed mechanism(s) of anticancer action" in md
    assert "### Proposed biomarker(s) of response" in md
    assert "## Supporting evidence" in md
    loaded = json.loads(paths["json"].read_text())
    assert loaded["hypotheses"][0]["rank"] == 1
    assert loaded["meta"]["drug_name"] == "APATINIB"


def test_render_no_clear_hypothesis(tmp_path):
    payload = {
        "summary": "Associations are weak and features incoherent; no confident hypothesis.",
        "clear_hypothesis": False,
        "proposed_mechanisms": [],
        "proposed_biomarkers": [],
        "hypotheses": [],
        "caveats": ["bootstrap r=0.05; near noise"],
    }
    out = report.write_report(payload, tmp_path / "i", compound_id="BRD:X", meta={})
    md = out["markdown"].read_text()
    assert "_No clear mechanism hypothesis is supported by the evidence._" in md
    assert "_No clear biomarker hypothesis is supported by the evidence._" in md
    # no supporting-evidence section when there are no hypotheses
    assert "## Supporting evidence" not in md


def test_report_schema_is_valid_json_schema():
    from biomarker_agent import prompts
    assert prompts.REPORT_TOOL["name"] == "submit_report"
    props = prompts.REPORT_TOOL["input_schema"]["properties"]
    assert "hypotheses" in props
    assert "proposed_mechanisms" in props
    assert "proposed_biomarkers" in props
    assert "clear_hypothesis" in prompts.REPORT_TOOL["input_schema"]["required"]


def test_render_embeds_figures_sized(tmp_path):
    payload = {
        "summary": "s", "clear_hypothesis": True,
        "hypotheses": [{
            "rank": 1, "title": "H", "features": ["CRISPR_SMARCD1"], "mechanism": "m",
            "novelty": "off-MOA", "confidence": 0.6,
            "evidence": {"internal": "r=0.3"},
            "figures": [{"path": "figures/feature_response__CRISPR_SMARCD1.png",
                         "caption": "SMARCD1 vs response"}],
        }],
    }
    out = report.write_report(payload, tmp_path / "interp", compound_id="BRD:TEST-1", meta={})
    md = out["markdown"].read_text()
    # markdown image syntax (renders reliably in VS Code preview), empty alt,
    # visible italic caption below
    assert "![](figures/feature_response__CRISPR_SMARCD1.png)" in md
    assert "<img" not in md  # no raw HTML img tags
    assert "*SMARCD1 vs response*" in md


def test_render_header_shap_section(tmp_path):
    payload = {"summary": "s", "clear_hypothesis": True, "hypotheses": []}
    meta = {"header_figures": [
        {"path": "figures/shap__selected_refit_model_significant_features.png",
         "caption": "Selected-refit model (significant features) — SHAP feature importance"},
        {"path": "figures/shap__baseline_model_random_forest.png",
         "caption": "Baseline model: random_forest — SHAP feature importance"},
    ]}
    out = report.write_report(payload, tmp_path / "i", compound_id="BRD:X", meta=meta)
    md = out["markdown"].read_text()
    assert "## Model feature attributions (SHAP)" in md
    # rendered side by side as a markdown table: labels header row + image row
    assert "| --- | --- |" in md
    assert "![](figures/shap__selected_refit_model_significant_features.png)" in md
    assert "![](figures/shap__baseline_model_random_forest.png)" in md
    assert "Baseline model: random_forest" in md
    assert "<img" not in md  # no raw HTML


def test_report_schema_has_figures():
    from biomarker_agent import prompts
    hyp = prompts.REPORT_TOOL["input_schema"]["properties"]["hypotheses"]["items"]
    assert "figures" in hyp["properties"]
    assert "figures" not in hyp["required"]  # optional


def test_render_feature_dispositions_table(tmp_path):
    payload = {
        "summary": "s", "clear_hypothesis": True, "hypotheses": [],
        "feature_dispositions": [
            {"feature": "GE_LOW", "rank": 2, "importance_ratio": 0.25, "r": 0.31,
             "disposition": "uninterpretable", "note": "no literature"},
            {"feature": "GE_TOP", "rank": 1, "importance_ratio": 1.0, "r": -0.20,
             "disposition": "centered", "note": "on-MOA target"},
        ],
    }
    out = report.write_report(payload, tmp_path / "i", compound_id="BRD:X", meta={})
    md = out["markdown"].read_text()
    assert "### Feature dispositions (ranked by model importance)" in md
    # rank-1 row renders before rank-2 (sorted by rank)
    assert md.index("GE_TOP") < md.index("GE_LOW")
    assert "centered" in md and "uninterpretable" in md
    assert "+0.31" in md and "-0.20" in md  # signed r formatting


def test_report_schema_has_feature_dispositions():
    from biomarker_agent import prompts
    props = prompts.REPORT_TOOL["input_schema"]["properties"]
    assert "feature_dispositions" in props
    item = props["feature_dispositions"]["items"]
    assert set(item["required"]) == {"feature", "rank", "disposition"}


def test_render_hypothesis_strength(tmp_path):
    payload = {"summary": "s", "clear_hypothesis": True, "hypothesis_strength": 0.82,
               "hypotheses": []}
    out = report.write_report(payload, tmp_path / "i", compound_id="BRD:X", meta={})
    md = out["markdown"].read_text()
    assert "**Hypothesis strength:** 0.82 / 1.00" in md


def test_report_schema_requires_hypothesis_strength():
    from biomarker_agent import prompts
    schema = prompts.REPORT_TOOL["input_schema"]
    assert "hypothesis_strength" in schema["properties"]
    assert "hypothesis_strength" in schema["required"]


def test_render_feature_comparison(tmp_path):
    payload = {"summary": "s", "clear_hypothesis": True, "hypotheses": []}
    meta = {"feature_comparison": {
        "baseline_model": "random_forest", "n_refit": 6, "n_baseline_top": 10,
        "shared": ["GE_ITGA1"], "refit_only": ["CRISPR_SMARCD1"],
        "baseline_only": ["GE_FOO"], "divergence": "high"}}
    out = report.write_report(payload, tmp_path / "i", compound_id="BRD:X", meta=meta)
    md = out["markdown"].read_text()
    assert "### Refit vs baseline top features" in md
    assert "substantially different" in md
    assert "Emphasized by the refit only: CRISPR_SMARCD1" in md
