#!/usr/bin/env python3
"""CLI: build the viewer data bundle from the agent's interpretation results.

    python viewer/scripts/build_data.py \
        --results data/interpretation_results --out viewer/public/data
"""

import argparse
import sys
from pathlib import Path

# Allow running directly (python viewer/scripts/build_data.py) without install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from biomarker_agent.viewer_build import build  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="data/interpretation_results",
                    help="directory of per-compound result dirs")
    ap.add_argument("--out", default="viewer/public/data",
                    help="output directory for the viewer bundle")
    ap.add_argument("--source", default="data/small_test_sample",
                    help="pipeline output root (per-compound refract/significant CSVs + "
                         "full-resolution SHAP PNGs); ignored if absent")
    ap.add_argument("--responses", default="data/responses_primary_v4.pkl",
                    help="pickled response matrix for the toxicity histogram; ignored if "
                         "absent or if pandas/matplotlib aren't installed. Run under `uv run` "
                         "to enable.")
    args = ap.parse_args()
    if not Path(args.results).is_dir():
        sys.exit(f"error: results directory not found: {args.results}\n"
                 "Run the biomarker agent first, or pass --results.")
    source = args.source if Path(args.source).is_dir() else None
    if source is None:
        print(f"note: source dir {args.source!r} not found; refit features won't be ranked "
              "and SHAP panels stay downscaled")
    responses = args.responses if Path(args.responses).is_file() else None
    if responses is None:
        print(f"note: responses file {args.responses!r} not found; no toxicity histogram")
    summary = build(args.results, args.out, source_dir=source, responses_file=responses)
    print(f"Wrote {summary['n_compounds']} compounds to {summary['out_dir']}")


if __name__ == "__main__":
    main()
