"""Tests for DataContext internal-stats primitives."""

import math

from biomarker_agent.datactx import DataContext


def test_associate_recovers_signal(synthetic_data):
    ff, rf, cid = synthetic_data
    ctx = DataContext(ff, rf)
    res = ctx.associate("GE_AAA", cid)
    assert res["n"] > 40
    assert res["pearson_r"] > 0.7
    assert res["pearson_p"] < 1e-6
    assert res["direction"] == "positive"
    # unrelated feature
    null = ctx.associate("GE_CCC", cid)
    assert abs(null["pearson_r"]) < 0.4


def test_associate_missing_feature(synthetic_data):
    ff, rf, cid = synthetic_data
    ctx = DataContext(ff, rf)
    res = ctx.associate("GE_NOPE", cid)
    assert "error" in res


def test_dependency_profile(synthetic_data):
    ff, rf, cid = synthetic_data
    ctx = DataContext(ff, rf)
    prof = ctx.dependency_profile("BBB")
    # 15/60 lines strongly dependent (gene effect < -0.5)
    assert prof["frac_dependent"] > 0.2
    assert prof["mean_gene_effect"] < -0.1
    assert prof["is_selective"] is True


def test_codependencies(synthetic_data):
    ff, rf, cid = synthetic_data
    ctx = DataContext(ff, rf)
    cod = ctx.codependencies("BBB", top=2)
    assert isinstance(cod, list)
    assert all("gene" in c and "r" in c for c in cod)
