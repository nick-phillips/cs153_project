"""Command-line entry point: point at an output dir, get interpretation reports."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from . import context, report
from .agent import DEFAULT_MODEL, run_agent
from .datactx import DataContext
from .loader import find_compounds, load_compound
from .prompts import SYSTEM_PROMPT
from .tools import build_registry

DATA = Path("data")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# Per-provider default model id (Anthropic-native vs OpenRouter slug).
DEFAULT_MODELS = {
    "anthropic": DEFAULT_MODEL,
    "openrouter": "anthropic/claude-sonnet-4.6",
}


def run_one(compound_dir, out_dir, feature_file, response_file, treatment_info, cache_dir,
            client, model=DEFAULT_MODEL, literature_backend="pubmed", max_tool_calls=40):
    """Analyze a single compound dir and write its report + trace. Returns paths."""
    result = load_compound(compound_dir)
    data_ctx = DataContext(feature_file, response_file)
    registry = build_registry(
        data_ctx=data_ctx, treatment_info=treatment_info, cache_dir=cache_dir,
        literature_backend=literature_backend,
        figures_dir=Path(out_dir) / "figures", figures_rel_prefix="figures",
        compound_result=result,
    )
    drug_info = registry.dispatch("drug_context", {"compound_id": result.compound_id})
    internal = context.precompute_internal(result, data_ctx)
    seed = context.build_seed_context(result, drug_info=drug_info, internal=internal)
    payload, transcript = run_agent(
        client=client, registry=registry, system_prompt=SYSTEM_PROMPT,
        seed_context=seed, model=model, max_tool_calls=max_tool_calls,
    )
    header_figures = _copy_shap_summaries(result, Path(out_dir))
    meta = {
        "drug_name": drug_info.get("drug_name"),
        "moa": drug_info.get("moa"),
        "targets": drug_info.get("targets"),
        "n_samples": result.n_samples,
        "performance": result.metrics,
        "header_figures": header_figures,
    }
    paths = report.write_report(payload, Path(out_dir), result.compound_id, meta=meta)
    trace_path = Path(out_dir) / "trace.json"
    trace_path.write_text(json.dumps(
        {"compound_id": result.compound_id, "model": model, "seed_context": seed,
         "transcript": transcript}, indent=2))
    paths["trace"] = trace_path
    return paths


def _copy_shap_summaries(result, out_dir: Path) -> list:
    """Copy the pipeline's SHAP-summary PNGs into the report's figures dir.

    Returns header figures [{path, caption}] with paths relative to out_dir so
    they embed in report.md.
    """
    figs_dir = Path(out_dir) / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for item in getattr(result, "shap_summaries", []):
        src = Path(item["source"])
        if not src.exists():
            continue
        slug = re.sub(r"[^A-Za-z0-9]+", "_", item["label"]).strip("_").lower()
        dest = figs_dir / f"shap__{slug}.png"
        shutil.copyfile(src, dest)
        out.append({"path": f"figures/{dest.name}",
                    "caption": f"{item['label']} — SHAP feature importance"})
    return out


def _make_client(provider: str, base_url: str | None = None):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    # OpenAI-compatible providers (OpenRouter, DigitalOcean serverless, ...)
    from .providers import OpenAICompatClient
    return OpenAICompatClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=base_url or OPENROUTER_BASE,
        extra_headers={"X-Title": "biomarker_agent"},
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Biological interpretation of biomarker-discovery outputs")
    p.add_argument("target", help="Output dir (batch w/ MANIFEST.csv) or a single BRD_* compound dir")
    p.add_argument("--out", default=None, help="Output root (default: <compound>/interpretation)")
    p.add_argument("--feature-file", default=str(DATA / "x-all_v4.pkl"))
    p.add_argument("--response-file", default=str(DATA / "responses_primary_v4.pkl"))
    p.add_argument("--treatment-info", default=str(DATA / "primary_screen_treatment_info.csv"))
    p.add_argument("--cache-dir", default=".biomarker_agent_cache")
    p.add_argument("--provider", choices=["anthropic", "openrouter"], default="anthropic",
                   help="LLM backend (default: anthropic)")
    p.add_argument("--base-url", default=None,
                   help="Override the OpenAI-compatible base URL (e.g. a DigitalOcean endpoint)")
    p.add_argument("--model", default=None,
                   help="Model id; defaults per provider")
    p.add_argument("--literature", choices=["pubmed", "paperclip"], default="pubmed")
    p.add_argument("--max-tool-calls", type=int, default=40)
    args = p.parse_args(argv)

    required_key = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENROUTER_API_KEY"
    if not os.environ.get(required_key):
        raise SystemExit(f"ERROR: {required_key} is not set (required for --provider {args.provider}).")

    model = args.model or DEFAULT_MODELS[args.provider]
    client = _make_client(args.provider, args.base_url)
    compounds = find_compounds(Path(args.target))
    index_lines = ["# Interpretation index", ""]
    for cdir in compounds:
        out_dir = Path(args.out) / cdir.name if args.out else cdir / "interpretation"
        paths = run_one(
            compound_dir=cdir, out_dir=out_dir,
            feature_file=Path(args.feature_file), response_file=Path(args.response_file),
            treatment_info=Path(args.treatment_info), cache_dir=Path(args.cache_dir),
            client=client, model=model, literature_backend=args.literature,
            max_tool_calls=args.max_tool_calls,
        )
        print(f"[done] {cdir.name} -> {paths['markdown']}")
        index_lines.append(f"- {cdir.name}: {paths['markdown']}")

    if len(compounds) > 1 and args.out:
        (Path(args.out) / "interpretation_index.md").write_text("\n".join(index_lines) + "\n")


if __name__ == "__main__":
    main()
