"""Render the structured report payload to markdown + JSON on disk."""

import json
from pathlib import Path

# Display widths (px) for embedded figures. Source PNGs are 300-DPI; constraining
# the rendered width keeps the report compact instead of showing huge images.
IMG_WIDTH = 420       # agent-generated evidence figures
SHAP_WIDTH = 360      # header SHAP-summary panels (several shown together)


def _performance_line(meta: dict) -> str | None:
    perf = meta.get("performance") or {}
    parts = []
    labels = [("selected_refit_oob_pearson", "selected-refit"),
              ("bootstrap_pred_pearson", "bootstrap"),
              ("baseline_pred_pearson", "baseline")]
    for key, label in labels:
        v = perf.get(key)
        if isinstance(v, (int, float)):
            parts.append(f"{label} r={v:.3f}")
    if not parts:
        return None
    n = meta.get("n_samples")
    suffix = f" · n={n} cell lines" if n else ""
    return "**Model performance (Pearson):** " + ", ".join(parts) + suffix


def render_markdown(payload: dict, compound_id: str, meta: dict | None = None) -> str:
    meta = meta or {}
    lines = [f"# Interpretation report — {compound_id}", ""]

    # --- Header: drug context + deterministic model performance ---
    drug = meta.get("drug_name")
    moa = meta.get("moa")
    targets = meta.get("targets")
    if drug or moa or targets:
        bits = []
        if drug:
            bits.append(f"**Drug:** {drug}")
        if moa:
            bits.append(f"**Known MOA:** {moa}")
        if targets:
            bits.append(f"**Target(s):** {targets}")
        lines.append("  ·  ".join(bits))
    perf_line = _performance_line(meta)
    if perf_line:
        lines.append(perf_line)
    lines.append("")

    # --- Header: model SHAP feature-attribution summaries (pipeline artifacts),
    # rendered side by side in a single-row HTML table ---
    header_figs = [f for f in (meta.get("header_figures") or []) if f.get("path")]
    if header_figs:
        lines.append("## Model feature attributions (SHAP)")
        lines.append("")
        lines.append("<table><tr>")
        for fig in header_figs:
            cap = fig.get("caption", "")
            lines.append(f'<td align="center"><img src="{fig["path"]}" alt="{cap}" '
                         f'width="{SHAP_WIDTH}"><br><em>{cap}</em></td>')
        lines.append("</tr></table>")
        lines.append("")

    # --- Header: refit vs baseline top-feature comparison ---
    cmp = meta.get("feature_comparison")
    if cmp:
        _DIV = {"high": "substantially different", "moderate": "partially overlapping",
                "low": "largely consistent"}
        bm = cmp.get("baseline_model", "baseline")
        lines.append("### Refit vs baseline top features")
        lines.append(
            f"The resampled refit model selected {cmp['n_refit']} significant feature(s); "
            f"{len(cmp['shared'])} overlap with the {bm} baseline's top {cmp['n_baseline_top']}. "
            f"Top features are **{_DIV.get(cmp.get('divergence'), 'compared')}** between the two models.")
        if cmp.get("shared"):
            lines.append(f"- Selected by both: {', '.join(cmp['shared'])}")
        if cmp.get("refit_only"):
            lines.append(f"- Emphasized by the refit only: {', '.join(cmp['refit_only'])}")
        if cmp.get("baseline_only"):
            lines.append(f"- Top baseline features not selected by refit: {', '.join(cmp['baseline_only'])}")
        lines.append("")

    # --- Summary + headline conclusions ---
    lines.append("## Summary")
    lines.append(payload.get("summary", ""))
    lines.append("")

    mechs = payload.get("proposed_mechanisms") or []
    lines.append("### Proposed mechanism(s) of anticancer action")
    if mechs:
        lines.extend(f"- {m}" for m in mechs)
    else:
        lines.append("_No clear mechanism hypothesis is supported by the evidence._")
    lines.append("")

    bios = payload.get("proposed_biomarkers") or []
    lines.append("### Proposed biomarker(s) of response")
    if bios:
        lines.extend(f"- {b}" for b in bios)
    else:
        lines.append("_No clear biomarker hypothesis is supported by the evidence._")
    lines.append("")

    # --- Supporting evidence ---
    hyps = sorted(payload.get("hypotheses", []), key=lambda x: x.get("rank", 999))
    if hyps:
        lines.append("## Supporting evidence")
        lines.append("")
        for h in hyps:
            kind = h.get("kind")
            kind_tag = f" · _{kind}_" if kind else ""
            lines.append(f"### {h.get('rank')}. {h.get('title')}  ·  _{h.get('novelty')}_"
                         f"{kind_tag}  (confidence {h.get('confidence')})")
            lines.append(f"**Features:** {', '.join(h.get('features', []))}")
            lines.append("")
            lines.append(h.get("mechanism", ""))
            ev = h.get("evidence") or {}
            if ev:
                lines.append("")
                lines.append("**Evidence:**")
                for k, v in ev.items():
                    lines.append(f"- _{k}_: {v}")
            for fig in (h.get("figures") or []):
                fpath = fig.get("path")
                if fpath:
                    cap = fig.get("caption", "")
                    lines.append("")
                    lines.append(f'<img src="{fpath}" alt="{cap}" width="{IMG_WIDTH}">')
                    lines.append("")
                    lines.append(f"*{cap}*")
            lines.append("")

    caveats = payload.get("caveats") or []
    if caveats:
        lines.append("## Caveats")
        lines.extend(f"- {c}" for c in caveats)
    return "\n".join(lines).rstrip() + "\n"


def write_report(payload: dict, out_dir: Path, compound_id: str, meta: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    md_path.write_text(render_markdown(payload, compound_id, meta))
    json_path.write_text(json.dumps(
        {"compound_id": compound_id, "meta": meta or {}, **payload}, indent=2))
    return {"markdown": md_path, "json": json_path}
