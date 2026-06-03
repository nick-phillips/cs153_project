"""Command-line entry point: point at an output dir, get interpretation reports."""

import argparse
import os
from pathlib import Path

from . import context, report
from .agent import DEFAULT_MODEL, run_agent
from .datactx import DataContext
from .loader import find_compounds, load_compound
from .prompts import SYSTEM_PROMPT
from .tools import build_registry

DATA = Path("data")


def run_one(compound_dir, out_dir, feature_file, response_file, treatment_info, cache_dir,
            client, model=DEFAULT_MODEL, literature_backend="pubmed", max_tool_calls=40):
    """Analyze a single compound dir and write its report. Returns report paths."""
    result = load_compound(compound_dir)
    data_ctx = DataContext(feature_file, response_file)
    registry = build_registry(
        data_ctx=data_ctx, treatment_info=treatment_info, cache_dir=cache_dir,
        literature_backend=literature_backend,
    )
    drug_info = registry.dispatch("drug_context", {"compound_id": result.compound_id})
    internal = context.precompute_internal(result, data_ctx)
    seed = context.build_seed_context(result, drug_info=drug_info, internal=internal)
    payload, _ = run_agent(
        client=client, registry=registry, system_prompt=SYSTEM_PROMPT,
        seed_context=seed, model=model, max_tool_calls=max_tool_calls,
    )
    return report.write_report(payload, Path(out_dir), result.compound_id)


def _make_client():
    import anthropic
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def main(argv=None):
    p = argparse.ArgumentParser(description="Biological interpretation of biomarker-discovery outputs")
    p.add_argument("target", help="Output dir (batch w/ MANIFEST.csv) or a single BRD_* compound dir")
    p.add_argument("--out", default=None, help="Output root (default: <compound>/interpretation)")
    p.add_argument("--feature-file", default=str(DATA / "x-all_v4.pkl"))
    p.add_argument("--response-file", default=str(DATA / "responses_primary_v4.pkl"))
    p.add_argument("--treatment-info", default=str(DATA / "primary_screen_treatment_info.csv"))
    p.add_argument("--cache-dir", default=".biomarker_agent_cache")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--literature", choices=["pubmed", "paperclip"], default="pubmed")
    p.add_argument("--max-tool-calls", type=int, default=40)
    args = p.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ERROR: ANTHROPIC_API_KEY is not set.")

    client = _make_client()
    compounds = find_compounds(Path(args.target))
    index_lines = ["# Interpretation index", ""]
    for cdir in compounds:
        out_dir = Path(args.out) / cdir.name if args.out else cdir / "interpretation"
        paths = run_one(
            compound_dir=cdir, out_dir=out_dir,
            feature_file=Path(args.feature_file), response_file=Path(args.response_file),
            treatment_info=Path(args.treatment_info), cache_dir=Path(args.cache_dir),
            client=client, model=args.model, literature_backend=args.literature,
            max_tool_calls=args.max_tool_calls,
        )
        print(f"[done] {cdir.name} -> {paths['markdown']}")
        index_lines.append(f"- {cdir.name}: {paths['markdown']}")

    if len(compounds) > 1 and args.out:
        (Path(args.out) / "interpretation_index.md").write_text("\n".join(index_lines) + "\n")


if __name__ == "__main__":
    main()
