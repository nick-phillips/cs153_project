"""Build the grounded opening context handed to the agent for one compound."""

import json

from .loader import CompoundResult


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
        f"baseline={m.get('baseline_pred_pearson')}"
    )
    lines.append(f"Passing features by class: {json.dumps(result.passing_by_class)}")
    lines.append("\n### Passing features (resampled model: significant + stable)")
    for f in result.passing_features:
        assoc = internal.get(f.name, {})
        lines.append(
            f"- {f.name} (class={f.feature_class}, gene={f.gene}): "
            f"reproducibility={f.reproducibility}, q={f.q_value:.2e}, "
            f"real_importance={f.mean_real_importance:.4f} (null={f.mean_null_importance:.4f}); "
            f"internal assoc r={assoc.get('pearson_r')} ({assoc.get('direction')}, n={assoc.get('n')})"
        )
    if result.baseline_top_features:
        lines.append("\n### Top baseline-model SHAP features (context/contrast)")
        for model, feats in result.baseline_top_features.items():
            top = ", ".join(name for name, _ in feats[:8])
            lines.append(f"- {model}: {top}")
    return "\n".join(lines)


def precompute_internal(result: CompoundResult, data_ctx) -> dict:
    """Run the internal association for each passing feature up front."""
    out = {}
    for f in result.passing_features:
        out[f.name] = data_ctx.associate(f.name, result.compound_id)
    return out
