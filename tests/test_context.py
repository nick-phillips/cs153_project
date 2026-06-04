"""Tests for the seed-context builder."""

from biomarker_agent import context
from biomarker_agent.loader import CompoundResult, PassingFeature


def test_build_seed_context():
    cr = CompoundResult(
        compound_id="BRD:TEST-1",
        dir_name="BRD_TEST-1",
        path=None,
        n_samples=100,
        metrics={"selected_refit_oob_pearson": 0.32, "baseline_pred_pearson": 0.17},
        passing_features=[
            PassingFeature("CRISPR_SMARCD1", "CRISPR", "SMARCD1", 0.74, 0.022, 0.002, 1e-32, 4e-28)
        ],
        passing_by_class={"CRISPR": 1},
        baseline_top_features={"random_forest": [("CRISPR_SMARCD1", 0.08), ("GE_X", 0.05)]},
    )
    text = context.build_seed_context(cr, drug_info={"drug_name": "APATINIB", "moa": "RET TKI", "targets": "RET"},
                                      internal={"CRISPR_SMARCD1": {"pearson_r": 0.3, "direction": "positive"}})
    assert "APATINIB" in text
    assert "SMARCD1" in text
    assert "RET TKI" in text
    assert "0.32" in text  # metric surfaced
    # features are explicitly ranked by model importance, not |r|
    assert "RANKED BY MODEL IMPORTANCE" in text
    assert "#1 CRISPR_SMARCD1" in text
    assert "imp_ratio" in text


def test_seed_context_orders_by_importance_and_lift_is_context_only():
    """Features list in importance order; performance is framed as context, not a gate."""
    cr = CompoundResult(
        compound_id="BRD:TEST-2",
        dir_name="BRD_TEST-2",
        path=None,
        n_samples=100,
        metrics={"selected_refit_oob_pearson": 0.30, "bootstrap_pred_pearson": 0.138,
                 "baseline_pred_pearson": 0.158},
        passing_features=[
            # deliberately out of importance order to prove the builder re-sorts
            PassingFeature("GE_LOW", "GE", "LOW", 0.5, 0.002, 0.001, 1e-3, 2e-2),
            PassingFeature("GE_TOP", "GE", "TOP", 0.7, 0.020, 0.001, 1e-9, 1e-6),
        ],
        passing_by_class={"GE": 2},
        baseline_top_features={},
    )
    internal = {"GE_LOW": {"pearson_r": 0.24, "direction": "positive"},
                "GE_TOP": {"pearson_r": 0.20, "direction": "positive"}}
    text = context.build_seed_context(cr, drug_info={"drug_name": "POSACONAZOLE"}, internal=internal)
    # performance present but explicitly framed as context, not a decision gate
    assert "CONTEXT ONLY" in text
    assert "do NOT decide whether to propose a hypothesis based on the refit-vs-baseline lift" in text
    # no lift-tier machinery leaks into the prompt anymore
    assert "SIGNAL ASSESSMENT" not in text
    assert "tier =" not in text
    # GE_TOP (higher importance) listed before GE_LOW despite GE_LOW's larger |r|
    assert text.index("#1 GE_TOP") < text.index("#2 GE_LOW")
