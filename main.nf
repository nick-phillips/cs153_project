#!/usr/bin/env nextflow

/*
 * biomarker-discovery
 *
 * Benchmarks drug response prediction models:
 *   ElasticNet, RF, XGBoost, CatBoost
 *
 * Usage:
 *   nextflow run main.nf \
 *     --response_ids response_ids.txt \
 *     --feature_file features.pkl \
 *     --response_file responses.pkl \
 *     --outdir results/
 *
 * response_ids.txt should contain one response ID per line, e.g.:
 *   BRD:BRD-K49049886-001-08-7
 *   BRD:BRD-K12345678-001-01-1
 */

nextflow.enable.dsl = 2

include { BIOMARKER_DISCOVERY_PIPELINE } from './workflows/biomarker_discovery_pipeline'


def validateParams() {
    def errors = []
    if (!file(params.response_ids).exists())
        errors << "Response IDs file not found: ${params.response_ids}"
    if (!file(params.feature_file).exists())
        errors << "Feature file not found: ${params.feature_file}"
    if (!file(params.response_file).exists())
        errors << "Response file not found: ${params.response_file}"

    if (errors) {
        log.error """
        ============================================================
        ERROR: Input files not found
        ============================================================
        ${errors.join('\n        ')}
        ============================================================
        """.stripIndent()
        exit 1
    }
}


workflow {
    validateParams()

    log.info """
    ============================================================
    Biomarker Discovery Benchmark Pipeline
    ============================================================
    response_ids  : ${params.response_ids}
    feature_file  : ${params.feature_file}
    response_file : ${params.response_file}
    outdir        : ${params.outdir}
    n_folds       : ${params.n_folds}
    ============================================================
    """.stripIndent()

    response_ids_ch = Channel
        .fromPath(params.response_ids)
        .splitText()
        .map { it.trim() }
        .filter { it.length() > 0 }

    feature_file = file(params.feature_file)
    response_file = file(params.response_file)

    BIOMARKER_DISCOVERY_PIPELINE(
        response_ids_ch,
        feature_file,
        response_file
    )
}


workflow.onComplete {
    log.info """
    ============================================================
    Pipeline ${workflow.success ? 'SUCCESS' : 'FAILED'}
    Duration : ${workflow.duration}
    Output   : ${params.outdir}
    ============================================================
    """.stripIndent()
}
