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
