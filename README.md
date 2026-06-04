# biomarker-discovery

Benchmarks drug response prediction baselines (ElasticNet, RF, XGBoost,
CatBoost) with correlation-based feature selection and SHAP feature importance.

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

## Interpretation agent + viewer

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
