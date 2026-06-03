"""Shared pytest fixtures."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "data" / "small_test_sample"
SAMPLE_COMPOUND = SAMPLE_DIR / "BRD_BRD-K25244359-066-03-4"


@pytest.fixture
def sample_dir() -> Path:
    return SAMPLE_DIR


@pytest.fixture
def sample_compound() -> Path:
    return SAMPLE_COMPOUND


@pytest.fixture
def synthetic_data(tmp_path):
    """Small feature + response frames with a planted signal.

    Returns (feature_file, response_file, compound_id). GE_AAA is positively
    correlated with the response of compound BRD:TEST-1; CRISPR_BBB is a
    selective dependency.
    """
    rng = np.random.default_rng(0)
    lines = [f"ACH-{i:06d}" for i in range(60)]
    signal = rng.normal(size=60)
    feats = pd.DataFrame(
        {
            "GE_AAA": signal + rng.normal(scale=0.1, size=60),
            "GE_CCC": rng.normal(size=60),
            "CRISPR_BBB": np.where(np.arange(60) < 15, -1.2, 0.05) + rng.normal(scale=0.05, size=60),
            "CRISPR_DDD": rng.normal(scale=0.05, size=60),
        },
        index=pd.Index(lines, name="ModelID"),
    )
    resp = pd.DataFrame({"BRD:TEST-1": signal * 0.8 + rng.normal(scale=0.2, size=60)}, index=lines)
    ff = tmp_path / "x.pkl"
    rf = tmp_path / "y.pkl"
    feats.to_pickle(ff)
    resp.to_pickle(rf)
    return ff, rf, "BRD:TEST-1"
