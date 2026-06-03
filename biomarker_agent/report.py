"""Render the structured report payload to markdown + JSON on disk."""

import json
from pathlib import Path


def render_markdown(payload: dict, compound_id: str) -> str:
    lines = [f"# Interpretation report — {compound_id}", "", payload.get("summary", ""), ""]
    for h in sorted(payload.get("hypotheses", []), key=lambda x: x.get("rank", 999)):
        lines.append(f"## {h.get('rank')}. {h.get('title')}  ·  _{h.get('novelty')}_  "
                     f"(confidence {h.get('confidence')})")
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
                lines.append(f"![{cap}]({fpath})")
                lines.append(f"*Figure: {cap}*")
        lines.append("")
    caveats = payload.get("caveats") or []
    if caveats:
        lines.append("## Caveats")
        lines.extend(f"- {c}" for c in caveats)
    return "\n".join(lines).rstrip() + "\n"


def write_report(payload: dict, out_dir: Path, compound_id: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    md_path.write_text(render_markdown(payload, compound_id))
    json_path.write_text(json.dumps({"compound_id": compound_id, **payload}, indent=2))
    return {"markdown": md_path, "json": json_path}
