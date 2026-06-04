# Problem Description
Interpretation of large-scale cancer cell line screening data is challenging. Elucidating biomarkers and drug mechanisms of action often requires domain expertise and extensive research.

Artificial intelligence tools have the potential to significantly increase the throughput and depth of such analyses. In this project, we develop an agentic system for biological discovery using data from the PRISM multiplexed drug screen. We demonstrate the system's utility in generating novel hypotheses through the use of custom analysis tools and biological data APIs.

We envision this system can be applied to aid in the characterization of novel mechanisms of cancer drug activity and to nominate candidates for experimental validation.

## Statistical Analysis for Biomarker Discovery

This code is used to generate upstream feature importance values for the agentic workflow. It benchmarks drug response prediction baselines (ElasticNet, RF, XGBoost,
CatBoost) with correlation-based feature selection and SHAP feature importance.

This step is computationally expensive and pre-computed results are available on request.

## Usage
To run a single compound:
```
uv run scripts/run_benchmark.py \
    --response_id BRD:BRD-K49049886-001-08-7 \
    --feature_file data/x-all_v4.pkl \
    --response_file data/responses_primary_v4.pkl \
    --output_dir single_compound_test \
    --n_folds 5
```

## Interpretation Agent + Viewer

After running the benchmark, interpret the selected biomarkers with an LLM agent
that proposes the mechanism/biomarker the model's output supports, scores it with a
`hypothesis_strength` (0–1), and backs it with evidence — then browse the results in
a searchable web viewer.

Full docs: [`biomarker_agent/README.md`](biomarker_agent/README.md)
(deeper tool/architecture notes in [`docs/biomarker_agent.md`](docs/biomarker_agent.md)).

```bash
# interpret a batch (needs ANTHROPIC_API_KEY, or --provider openrouter + OPENROUTER_API_KEY)
uv run biomarker-analyze data/priority_samples --out data/interpretation_results

# build + serve the viewer
uv run python viewer/scripts/build_data.py --results data/interpretation_results \
    --source data/priority_samples --responses data/responses_primary_v4.pkl
cd viewer && npm install && npm run dev      # → http://localhost:5173/
```

## AI Usage Disclosure
AI assistance with Claude Code was used extensively in generating all code for this project. External data sources and APIs are described in the biomarker agent documentation.