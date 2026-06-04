"""Build a static JSON bundle for the results viewer from the agent's outputs.

Reads each compound directory under a results dir (each containing report.json,
optional trace.json, and a figures/ dir) and emits, into an output dir:
  - index.json: a compact, searchable list (one entry per compound)
  - <dir>.json: the full report payload plus the trace, per compound
  - <dir>/figures/*.png: the compound's figures, copied verbatim
The viewer SPA fetches these at runtime.

When a source pipeline dir is supplied, each compound's refit-model significant
features (with SHAP importance) are read from
``<source>/<dir>/refract/significant/significant_features.csv`` so the viewer
can rank them — this signal isn't carried in the agent's report.json.
"""

import csv
import json
import shutil
from pathlib import Path

from .viewer_figures import pred_vs_actual_plot, response_histogram

TOP_FEATURES = 5  # ranked refit features shown per compound card
BASELINE_SHAP = "shap_summary.png"


def _fullres_shap_source(caption: str, source_dir, dir_name: str):
    """Locate the full-resolution SHAP PNG in the source pipeline dir for a header
    figure, matched by its caption. Returns a Path or None.

    The bundle's copied SHAP panels are downscaled (small) for fast pages; the
    expanded lightbox view uses these crisp ~1400px originals instead.
    """
    if not source_dir:
        return None
    base = Path(source_dir) / dir_name
    cap = caption or ""
    if "Selected-refit" in cap:
        p = base / "refract" / "significant" / "selected_shap_summary.png"
        return p if p.exists() else None
    marker = "Baseline model:"
    if marker in cap:
        model = cap.split(marker, 1)[1].strip().split()[0]
        p = base / "baselines" / model / BASELINE_SHAP
        return p if p.exists() else None
    return None


def parse_gene(token: str) -> str:
    """'shRNA_MDM4' -> 'MDM4'. No class prefix -> token unchanged."""
    return token.split("_", 1)[1] if "_" in token else token


def ranked_refit_features(source_dir, dir_name: str, top_n: int = TOP_FEATURES) -> list:
    """Top refit-model features by SHAP importance, ranked desc, from the source dir.

    Returns ``[{name, gene, klass, importance}]``; empty if no source dir or CSV.
    """
    if not source_dir:
        return []
    csv_path = (Path(source_dir) / dir_name / "refract" / "significant"
                / "significant_features.csv")
    if not csv_path.exists():
        return []
    rows: list = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("passes", "")).strip().lower() != "true":
                continue
            try:
                imp = float(r["mean_real_importance"])
            except (KeyError, ValueError):
                continue
            name = r.get("feature_name", "")
            rows.append({"name": name, "gene": parse_gene(name),
                         "klass": r.get("feature_class", ""), "importance": imp})
    rows.sort(key=lambda x: x["importance"], reverse=True)
    return rows[:top_n]


def _genes(tokens: list) -> list:
    """Bare gene symbols for a list of feature tokens, order-preserving + de-duped."""
    out: list = []
    for t in tokens:
        g = parse_gene(t)
        if g and g not in out:
            out.append(g)
    return out


def index_entry(report: dict, dir_name: str, top_features: list | None = None) -> dict:
    """Compact, searchable index record for one compound's report payload."""
    meta = report.get("meta", {}) or {}
    perf = meta.get("performance", {}) or {}
    cmp = meta.get("feature_comparison") or {}
    shared = cmp.get("shared") or []
    refit_features = sorted(set(shared) | set(cmp.get("refit_only") or []))
    baseline_features = sorted(set(shared) | set(cmp.get("baseline_only") or []))
    hyps = report.get("hypotheses") or []
    hyp_features: list = []
    for h in hyps:
        for f in h.get("features", []) or []:
            if f not in hyp_features:
                hyp_features.append(f)
    return {
        "id": dir_name,
        "compound_id": report.get("compound_id", dir_name),
        "drug_name": meta.get("drug_name") or "",
        "moa": meta.get("moa") or "",
        "targets": meta.get("targets") or "",
        "has_hypothesis": bool(report.get("clear_hypothesis") and hyps),
        "performance": {
            "refit": perf.get("selected_refit_oob_pearson"),
            "bootstrap": perf.get("bootstrap_pred_pearson"),
            "baseline": perf.get("baseline_pred_pearson"),
        },
        "divergence": cmp.get("divergence"),
        "top_hypothesis_title": hyps[0].get("title") if hyps else None,
        "refit_features": refit_features,
        "baseline_features": baseline_features,
        "hypothesis_genes": _genes(hyp_features),
        "search_genes": _genes(refit_features + baseline_features + hyp_features),
        "top_features": top_features or [],
    }


def build(results_dir, out_dir, source_dir=None, responses_file=None) -> dict:
    """Scan results_dir, write the viewer bundle into out_dir. Returns a summary.

    ``source_dir`` (optional) is the pipeline output root holding the per-compound
    ``refract/significant/significant_features.csv`` (used to rank refit features)
    and the full-resolution SHAP PNGs (swapped in for the expanded view).
    ``responses_file`` (optional) is the pickled response matrix used to render a
    ground-truth toxicity histogram alongside the SHAP panels.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index: list = []
    for sub in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        report_path = sub / "report.json"
        if not report_path.exists():
            print(f"skip {sub.name}: no report.json")
            continue
        report = json.loads(report_path.read_text())
        compound_id = report.get("compound_id", sub.name)
        top_features = ranked_refit_features(source_dir, sub.name)
        index.append(index_entry(report, sub.name, top_features))

        trace = None
        trace_path = sub / "trace.json"
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
        compound = {**report, "id": sub.name, "trace": trace, "top_features": top_features}

        # Copy the agent's figures (incl. downscaled SHAP panels).
        fig_src = sub / "figures"
        if fig_src.is_dir():
            fig_dest = out_dir / sub.name / "figures"
            fig_dest.mkdir(parents=True, exist_ok=True)
            for png in sorted(fig_src.glob("*.png")):
                shutil.copy2(png, fig_dest / png.name)

        meta = compound.setdefault("meta", {})
        header_figs = list(meta.get("header_figures") or [])

        # Swap downscaled SHAP panels for full-resolution originals (crisp expand).
        if source_dir:
            for hf in header_figs:
                full = _fullres_shap_source(hf.get("caption", ""), source_dir, sub.name)
                if full and hf.get("path"):
                    dest = out_dir / sub.name / hf["path"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full, dest)

        # Top row of the 2x2 header grid: refit predicted-vs-actual + the
        # ground-truth toxicity distribution (SHAP panels go on the bottom row).
        extras: list = []
        if source_dir:
            pva_csv = (Path(source_dir) / sub.name / "refract" / "significant"
                       / "pred_vs_actual.csv")
            pva_rel = "figures/pred_vs_actual.png"
            n = pred_vs_actual_plot(pva_csv, out_dir / sub.name / pva_rel)
            if n:
                extras.append({"path": pva_rel,
                               "caption": f"Predicted vs actual — selected-refit model "
                                          f"(n={n} cell lines)"})
        if responses_file:
            hist_rel = "figures/response_hist.png"
            n = response_histogram(responses_file, compound_id, out_dir / sub.name / hist_rel)
            if n:
                extras.append({"path": hist_rel,
                               "caption": f"Ground-truth response (toxicity) distribution "
                                          f"— n={n} cell lines"})
        if extras:
            meta["header_figures"] = [*extras, *header_figs]

        (out_dir / f"{sub.name}.json").write_text(json.dumps(compound, indent=2))

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return {"n_compounds": len(index), "out_dir": str(out_dir)}
