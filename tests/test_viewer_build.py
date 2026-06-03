"""Tests for the viewer data build step."""

import json

from biomarker_agent.viewer_build import build, index_entry, parse_gene


def test_parse_gene_strips_class_prefix():
    assert parse_gene("shRNA_MDM4") == "MDM4"
    assert parse_gene("GE_KRT20") == "KRT20"
    assert parse_gene("CRISPR_TP53") == "TP53"
    assert parse_gene("PLAIN") == "PLAIN"


def test_index_entry_extracts_genes_and_flags():
    report = {
        "compound_id": "BRD:1",
        "clear_hypothesis": True,
        "meta": {
            "drug_name": "DrugA",
            "moa": "KINASE INHIBITOR",
            "targets": "ABL1",
            "performance": {
                "selected_refit_oob_pearson": 0.33,
                "bootstrap_pred_pearson": 0.12,
                "baseline_pred_pearson": 0.08,
            },
            "feature_comparison": {
                "baseline_model": "random_forest",
                "n_refit": 2, "n_baseline_top": 10,
                "shared": ["shRNA_MDM4"],
                "refit_only": ["GE_KRT20"],
                "baseline_only": ["CRISPR_TP53"],
                "divergence": "moderate",
            },
        },
        "hypotheses": [{"rank": 1, "title": "MDM4 axis", "features": ["shRNA_MDM4", "GE_EMSY"]}],
    }
    e = index_entry(report, "C1")
    assert e["id"] == "C1"
    assert e["compound_id"] == "BRD:1"
    assert e["drug_name"] == "DrugA"
    assert e["has_hypothesis"] is True
    assert e["top_hypothesis_title"] == "MDM4 axis"
    assert e["divergence"] == "moderate"
    assert e["performance"]["refit"] == 0.33
    assert set(e["refit_features"]) == {"shRNA_MDM4", "GE_KRT20"}
    assert set(e["baseline_features"]) == {"shRNA_MDM4", "CRISPR_TP53"}
    assert {"MDM4", "KRT20", "TP53", "EMSY"} <= set(e["search_genes"])
    assert "EMSY" in e["hypothesis_genes"]


def test_index_entry_no_hypothesis():
    report = {"compound_id": "BRD:2", "clear_hypothesis": False,
              "meta": {"drug_name": "DrugB"}, "hypotheses": []}
    e = index_entry(report, "C2")
    assert e["has_hypothesis"] is False
    assert e["top_hypothesis_title"] is None
    assert e["refit_features"] == []
    assert e["search_genes"] == []


def _write_compound(d, report, trace=None, figures=()):
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps(report))
    if trace is not None:
        (d / "trace.json").write_text(json.dumps(trace))
    if figures:
        fd = d / "figures"
        fd.mkdir()
        for name in figures:
            (fd / name).write_bytes(b"\x89PNG\r\n")


def test_build_writes_bundle_and_copies_figures(tmp_path):
    results = tmp_path / "results"
    _write_compound(
        results / "C1",
        {
            "compound_id": "BRD:1", "clear_hypothesis": True,
            "meta": {"drug_name": "DrugA", "feature_comparison": {
                "baseline_model": "random_forest", "n_refit": 1, "n_baseline_top": 10,
                "shared": ["shRNA_MDM4"], "refit_only": [], "baseline_only": [],
                "divergence": "low"}},
            "hypotheses": [{"rank": 1, "title": "T", "features": ["shRNA_MDM4"],
                            "figures": [{"path": "figures/a.png", "caption": "c"}]}],
        },
        trace={"compound_id": "BRD:1", "model": "m", "usage": {"cost_usd": 0.1},
               "seed_context": "s", "transcript": [{"event": "assistant_text", "text": "hi"}]},
        figures=["a.png"],
    )
    _write_compound(
        results / "C2",
        {"compound_id": "BRD:2", "clear_hypothesis": False,
         "meta": {"drug_name": "DrugB"}, "hypotheses": []},
    )
    (results / "junk").mkdir()  # no report.json -> skipped

    out = tmp_path / "out"
    summary = build(results, out)
    assert summary["n_compounds"] == 2

    index = json.loads((out / "index.json").read_text())
    assert {e["id"] for e in index} == {"C1", "C2"}

    c1 = json.loads((out / "C1.json").read_text())
    assert c1["trace"]["model"] == "m"
    assert c1["id"] == "C1"
    assert (out / "C1" / "figures" / "a.png").exists()

    c2 = json.loads((out / "C2.json").read_text())
    assert c2["trace"] is None
    entry2 = next(e for e in index if e["id"] == "C2")
    assert entry2["has_hypothesis"] is False
