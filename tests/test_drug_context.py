"""Tests for the drug-context (MOA) lookup tool."""

import pandas as pd

from biomarker_agent.tools import drug_context


def _info(tmp_path):
    df = pd.DataFrame(
        {
            "IDs": ["BRD:BRD-K25244359-066-03-4", "BRD:OTHER"],
            "Drug.Name": ["APATINIB", "OTHERDRUG"],
            "MOA": ["RET TYROSINE KINASE INHIBITOR", "X"],
            "repurposing_target": ["CSK, KDR, KIT, RET", ""],
        }
    )
    p = tmp_path / "ti.csv"
    df.to_csv(p, index=False)
    return p


def test_lookup_hit(tmp_path):
    fn = drug_context.make_handler(_info(tmp_path))
    out = fn(compound_id="BRD:BRD-K25244359-066-03-4")
    assert out["drug_name"] == "APATINIB"
    assert "RET" in out["moa"]
    assert "RET" in out["targets"]


def test_lookup_miss(tmp_path):
    fn = drug_context.make_handler(_info(tmp_path))
    out = fn(compound_id="BRD:NOPE")
    assert "error" in out
