"""Build a static JSON bundle for the results viewer from the agent's outputs.

Reads each compound directory under a results dir (each containing report.json,
optional trace.json, and a figures/ dir) and emits, into an output dir:
  - index.json: a compact, searchable list (one entry per compound)
  - <dir>.json: the full report payload plus the trace, per compound
  - <dir>/figures/*.png: the compound's figures, copied verbatim
The viewer SPA fetches these at runtime.
"""

import json
import shutil
from pathlib import Path


def parse_gene(token: str) -> str:
    """'shRNA_MDM4' -> 'MDM4'. No class prefix -> token unchanged."""
    return token.split("_", 1)[1] if "_" in token else token


def _genes(tokens: list) -> list:
    """Bare gene symbols for a list of feature tokens, order-preserving + de-duped."""
    out: list = []
    for t in tokens:
        g = parse_gene(t)
        if g and g not in out:
            out.append(g)
    return out


def index_entry(report: dict, dir_name: str) -> dict:
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
    }


def build(results_dir, out_dir) -> dict:
    """Scan results_dir, write the viewer bundle into out_dir. Returns a summary."""
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
        index.append(index_entry(report, sub.name))

        trace = None
        trace_path = sub / "trace.json"
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
        compound = {**report, "id": sub.name, "trace": trace}
        (out_dir / f"{sub.name}.json").write_text(json.dumps(compound, indent=2))

        fig_src = sub / "figures"
        if fig_src.is_dir():
            fig_dest = out_dir / sub.name / "figures"
            fig_dest.mkdir(parents=True, exist_ok=True)
            for png in sorted(fig_src.glob("*.png")):
                shutil.copy2(png, fig_dest / png.name)

    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    return {"n_compounds": len(index), "out_dir": str(out_dir)}
