"""Tests for the viewer data build step."""

import json

import pytest

from biomarker_agent.viewer_build import (
    _fullres_shap_source,
    build,
    index_entry,
    parse_gene,
    ranked_refit_features,
)
from biomarker_agent.viewer_figures import pred_vs_actual_plot, response_histogram


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


def _write_sig_csv(d, rows):
    """rows: list of (feature_name, feature_class, importance, passes)."""
    sig = d / "refract" / "significant"
    sig.mkdir(parents=True)
    lines = ["feature_name,feature_class,mean_real_importance,passes"]
    lines += [f"{n},{k},{imp},{p}" for (n, k, imp, p) in rows]
    (sig / "significant_features.csv").write_text("\n".join(lines) + "\n")


def test_ranked_refit_features_orders_and_filters(tmp_path):
    _write_sig_csv(tmp_path / "C1", [
        ("GE_KRT20", "GE", 0.008, "True"),
        ("shRNA_MDM4", "shRNA", 0.005, "True"),
        ("GE_NOISE", "GE", 0.001, "False"),  # not significant -> excluded
    ])
    feats = ranked_refit_features(tmp_path, "C1")
    assert [f["gene"] for f in feats] == ["KRT20", "MDM4"]
    assert feats[0]["klass"] == "GE"
    assert feats[0]["name"] == "GE_KRT20"
    assert ranked_refit_features(None, "C1") == []
    assert ranked_refit_features(tmp_path, "missing") == []


def test_build_attaches_ranked_top_features_from_source(tmp_path):
    results = tmp_path / "results"
    _write_compound(results / "C1", {"compound_id": "BRD:1", "clear_hypothesis": True,
                                     "meta": {"drug_name": "DrugA"}, "hypotheses": []})
    source = tmp_path / "source"
    _write_sig_csv(source / "C1", [
        ("GE_KRT20", "GE", 0.008, "True"),
        ("shRNA_MDM4", "shRNA", 0.005, "True"),
    ])
    out = tmp_path / "out"
    build(results, out, source_dir=source)
    index = json.loads((out / "index.json").read_text())
    tf = index[0]["top_features"]
    assert [f["gene"] for f in tf] == ["KRT20", "MDM4"]


def test_fullres_shap_source_maps_captions(tmp_path):
    base = tmp_path / "C1"
    sel = base / "refract" / "significant" / "selected_shap_summary.png"
    sel.parent.mkdir(parents=True)
    sel.write_bytes(b"\x89PNG")
    rf = base / "baselines" / "random_forest" / "shap_summary.png"
    rf.parent.mkdir(parents=True)
    rf.write_bytes(b"\x89PNG")

    assert _fullres_shap_source("Selected-refit model (significant features) — SHAP",
                                tmp_path, "C1") == sel
    assert _fullres_shap_source("Baseline model: random_forest — SHAP feature importance",
                                tmp_path, "C1") == rf
    # baseline model with no file present -> None
    assert _fullres_shap_source("Baseline model: xgboost — SHAP", tmp_path, "C1") is None
    assert _fullres_shap_source("something else", tmp_path, "C1") is None
    assert _fullres_shap_source("Selected-refit", None, "C1") is None


def test_response_histogram(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    cid = "BRD:BRD-1"
    df = pd.DataFrame(
        {cid: [0.1, 0.2, None, 0.5, -0.3], "BRD:BRD-2": [1.0, 2.0, 3.0, 4.0, 5.0]},
        index=[f"ACH-{i}" for i in range(5)],
    )
    pkl = tmp_path / "responses.pkl"
    df.to_pickle(pkl)
    out_png = tmp_path / "fig" / "hist.png"
    n = response_histogram(pkl, cid, out_png)
    assert n == 4  # one NaN dropped
    assert out_png.exists() and out_png.stat().st_size > 0
    assert response_histogram(pkl, "BRD:missing", out_png) is None


def test_pred_vs_actual_plot(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    csv = tmp_path / "pred_vs_actual.csv"
    csv.write_text(
        "sample,observed,predicted\n"
        "ACH-1,0.1,0.2\nACH-2,0.5,0.4\nACH-3,-0.3,-0.1\nACH-4,0.8,0.6\n"
    )
    out_png = tmp_path / "fig" / "pva.png"
    n = pred_vs_actual_plot(csv, out_png)
    assert n == 4
    assert out_png.exists() and out_png.stat().st_size > 0
    assert pred_vs_actual_plot(tmp_path / "missing.csv", out_png) is None
