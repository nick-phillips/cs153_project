/*
 * Biomarker Discovery Pipeline Workflow
 *
 * For each response ID, runs the full benchmark:
 *   ElasticNet, RF, XGBoost, CatBoost
 */

include { BENCHMARK_TRAIN } from '../modules/benchmark_train'

workflow BIOMARKER_DISCOVERY_PIPELINE {
    take:
    response_ids
    feature_file
    response_file

    main:
    benchmark_results = BENCHMARK_TRAIN(
        response_ids,
        feature_file,
        response_file
    )

    emit:
    benchmarks = benchmark_results.benchmark_results
}
