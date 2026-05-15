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
