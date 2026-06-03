"""Tests for data-driven pure plotting functions."""

import numpy as np
import pandas as pd

from biomarker_agent import plots

PNG = b"\x89PNG\r\n\x1a\n"


def _xy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(size=n), name="GE_AAA")
    y = pd.Series(x.values * 0.8 + rng.normal(scale=0.3, size=n), name="BRD:TEST-1")
    return x, y


def _is_png(p):
    from pathlib import Path
    p = Path(p)
    return p.exists() and p.stat().st_size > 500 and p.read_bytes()[:8] == PNG


def test_feature_response(tmp_path):
    x, y = _xy()
    out = plots.feature_response(x, y, "GE_AAA", "BRD:TEST-1", tmp_path / "fr.png")
    assert _is_png(out)


def test_two_feature_response(tmp_path):
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(size=50), name="GE_A")
    b = pd.Series(rng.normal(size=50), name="GE_B")
    resp = pd.Series(rng.normal(size=50), name="BRD:TEST-1")
    out = plots.two_feature_response(a, b, resp, "GE_A", "GE_B", "BRD:TEST-1", tmp_path / "tf.png")
    assert _is_png(out)


def test_feature_panel(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"GE_AAA": rng.normal(size=50), "GE_BBB": rng.normal(size=50),
                       "response": rng.normal(size=50)})
    corr = df.corr()
    out = plots.feature_panel(corr, "BRD:TEST-1", tmp_path / "fp.png")
    assert _is_png(out)


def test_dependency_distribution(tmp_path):
    rng = np.random.default_rng(2)
    eff = pd.Series(np.concatenate([rng.normal(-1.0, 0.1, 15), rng.normal(0.0, 0.1, 45)]))
    out = plots.dependency_distribution(eff, "BBB", tmp_path / "dd.png", threshold=-0.5)
    assert _is_png(out)


def test_codependency_bar(tmp_path):
    codeps = [{"gene": "BRD9", "r": 0.72}, {"gene": "BICRA", "r": 0.60},
              {"gene": "FOO", "r": -0.4}]
    out = plots.codependency_bar(codeps, "SMARCD1", tmp_path / "cb.png")
    assert _is_png(out)


def test_passing_importance(tmp_path):
    feats = [{"name": "CRISPR_SMARCD1", "mean_real_importance": 0.022, "mean_null_importance": 0.002},
             {"name": "GE_ITGA1", "mean_real_importance": 0.008, "mean_null_importance": 0.0026}]
    out = plots.passing_importance(feats, tmp_path / "pi.png")
    assert _is_png(out)


def test_codependency_bar_empty_errors(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        plots.codependency_bar([], "X", tmp_path / "x.png")
