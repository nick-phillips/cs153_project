/*
 * Process: BENCHMARK_TRAIN
 * Run all models (ElasticNet, RF, XGBoost, CatBoost) for one response ID.
 */

process BENCHMARK_TRAIN {
    tag "${response_id}"
    label 'process_high'
    maxForks 200

    publishDir "${params.outdir}/${response_id}", mode: 'copy'

    input:
    val response_id
    path feature_file
    path response_file

    output:
    tuple val(response_id), path("benchmark"), emit: benchmark_results

    script:
    def pythonCmd = workflow.containerEngine ? "python" : "uv run --frozen -- python"
    """
    set -euo pipefail
    ${pythonCmd} -m scripts.run_benchmark \
        --response_id '${response_id}' \
        --feature_file ${feature_file} \
        --response_file ${response_file} \
        --output_dir benchmark \
        --n_folds ${params.n_folds}
    """
}
