"""Tests for the output-directory loader."""

from biomarker_agent import loader


def test_parse_feature_name():
    assert loader.parse_feature_name("CRISPR_SMARCD1") == ("CRISPR", "SMARCD1")
    assert loader.parse_feature_name("GE_ITGA1") == ("GE", "ITGA1")
    # genes can contain underscores/hyphens; only the first token is the class
    assert loader.parse_feature_name("PROT_PDLIM5") == ("PROT", "PDLIM5")


def test_find_compounds_batch(sample_dir):
    dirs = loader.find_compounds(sample_dir)
    assert len(dirs) == 10
    assert all(d.name.startswith("BRD_") for d in dirs)


def test_find_compounds_single(sample_compound):
    dirs = loader.find_compounds(sample_compound)
    assert dirs == [sample_compound]


def test_load_compound(sample_compound):
    res = loader.load_compound(sample_compound)
    assert res.compound_id == "BRD:BRD-K25244359-066-03-4"
    assert res.n_samples == 530
    assert res.metrics["selected_refit_oob_pearson"] > 0.3
    names = [f.name for f in res.passing_features]
    assert "CRISPR_SMARCD1" in names
    smarcd1 = next(f for f in res.passing_features if f.name == "CRISPR_SMARCD1")
    assert smarcd1.gene == "SMARCD1"
    assert smarcd1.feature_class == "CRISPR"
    assert smarcd1.q_value < 1e-10
    # baseline top features present for at least random_forest
    assert "random_forest" in res.baseline_top_features
    assert len(res.baseline_top_features["random_forest"]) > 0
    # SHAP summary images collected: selected-refit + baselines, all existing files
    from pathlib import Path as _P
    labels = [s["label"] for s in res.shap_summaries]
    assert any("Selected-refit" in l for l in labels)
    assert any("random_forest" in l for l in labels)
    assert all(_P(s["source"]).exists() for s in res.shap_summaries)
