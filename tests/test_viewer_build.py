"""Tests for the viewer data build step."""

from biomarker_agent.viewer_build import build, index_entry, parse_gene  # noqa: F401


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
