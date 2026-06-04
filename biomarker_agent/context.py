"""Build the grounded opening context handed to the agent for one compound."""

import json

from .loader import CompoundResult, refit_vs_baseline


def build_seed_context(result: CompoundResult, drug_info: dict, internal: dict) -> str:
    """Render a compact, information-dense text block describing the compound."""
    lines = []
    lines.append(f"## Compound {result.compound_id} ({result.dir_name})")
    lines.append(
        f"Drug: {drug_info.get('drug_name','?')} | "
        f"Known MOA: {drug_info.get('moa') or 'unknown'} | "
        f"Known target(s): {drug_info.get('targets') or 'unknown'}"
    )
    m = result.metrics
    lines.append(
        "Model performance (Pearson): "
        f"resampled refit={m.get('selected_refit_oob_pearson')}, "
        f"bootstrap={m.get('bootstrap_pred_pearson')}, "
        f"baseline={m.get('baseline_pred_pearson')} "
        "(CONTEXT ONLY — report it honestly, but do NOT decide whether to propose a "
        "hypothesis based on the refit-vs-baseline lift. Judge by the strength and "
        "plausibility of the biology/evidence in the feature set.)"
    )

    lines.append(
        "RESPONSE CONVENTION: response is a drug-response score where LOWER = greater "
        "SENSITIVITY (more cell killing) and HIGHER = RESISTANCE. So a NEGATIVE "
        "feature-response correlation means higher feature value -> greater sensitivity; "
        "a POSITIVE correlation means higher feature value -> greater resistance. State "
        "biomarker directions in sensitivity/resistance terms and double-check the sign."
    )
    lines.append(f"Passing features by class: {json.dumps(result.passing_by_class)}")
    lines.append(
        "\n### Passing features — RANKED BY MODEL IMPORTANCE (mean_real_importance, descending)"
    )
    lines.append(
        "This is the full feature set the model selected, ordered by what it weights. "
        "Interpret the SET as a scientist — a single dominant feature or a coherent "
        "multi-feature signature may be the hypothesis. Weight by importance, but do not "
        "silently ignore a high-ranked feature or falsely call a low-ranked one 'strongest'. "
        "imp_ratio = this feature's importance / the top feature's importance."
    )
    ranked = sorted(result.passing_features, key=lambda f: f.mean_real_importance, reverse=True)
    top_imp = ranked[0].mean_real_importance if ranked else 0.0
    for rank, f in enumerate(ranked, start=1):
        assoc = internal.get(f.name, {})
        implies = assoc.get("higher_feature_implies")
        implies_txt = f"; higher {f.gene} -> {implies}" if implies else ""
        ratio = (f.mean_real_importance / top_imp) if top_imp else 0.0
        lines.append(
            f"- #{rank} {f.name} (class={f.feature_class}, gene={f.gene}): "
            f"real_importance={f.mean_real_importance:.4f} (imp_ratio={ratio:.2f}, "
            f"null={f.mean_null_importance:.4f}), reproducibility={f.reproducibility}, "
            f"q={f.q_value:.2e}; internal assoc r={assoc.get('pearson_r')} "
            f"({assoc.get('direction')}, n={assoc.get('n')}){implies_txt}"
        )
    if result.baseline_top_features:
        lines.append("\n### Top baseline-model SHAP features (context/contrast)")
        for model, feats in result.baseline_top_features.items():
            top = ", ".join(name for name, _ in feats[:8])
            lines.append(f"- {model}: {top}")

    cmp = refit_vs_baseline(result)
    lines.append("\n### Refit vs baseline top features")
    lines.append(
        f"Refit-selected vs {cmp['baseline_model']} baseline top-{cmp['n_baseline_top']} "
        f"divergence: {cmp['divergence']}. "
        f"Shared: {cmp['shared'] or 'none'}. "
        f"Refit-only: {cmp['refit_only'] or 'none'}. "
        f"Baseline-only (dropped by refit): {cmp['baseline_only'] or 'none'}. "
        "Note in your report whether the refit emphasizes different features than the baseline.")
    return "\n".join(lines)


def precompute_internal(result: CompoundResult, data_ctx) -> dict:
    """Run the internal association for each passing feature up front."""
    out = {}
    for f in result.passing_features:
        out[f.name] = data_ctx.associate(f.name, result.compound_id)
    return out
