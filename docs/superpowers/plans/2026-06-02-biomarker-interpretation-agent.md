# Biomarker Interpretation Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `biomarker_agent`, a standalone CLI that points an LLM agent at a biomarker-discovery output directory and produces ranked, evidence-backed biological-mechanism hypotheses per compound.

**Architecture:** A deterministic data/tool layer (loader + internal stats + external API clients, all behind a uniform cached `Tool` interface) feeds a single-agent Anthropic tool-use loop. The agent investigates each compound's "passing" features, then emits a forced structured report rendered to markdown + JSON.

**Tech Stack:** Python 3.12, `anthropic` SDK, `requests`, `pandas`/`numpy`/`scipy` (already present), `pytest`. Managed by `uv`.

---

## File Structure

```
biomarker_agent/
  __init__.py        # package marker + version
  loader.py          # output dir → CompoundResult (Task 2)
  datactx.py         # DataContext: lazy pkl loading + internal stats primitives (Task 3)
  cache.py           # on-disk JSON cache for external calls (Task 4)
  tools/
    __init__.py      # TOOL REGISTRY: build anthropic schemas + dispatch (Task 13)
    base.py          # Tool dataclass, http_get_json, graceful errors (Task 4)
    drug_context.py  # treatment_info MOA/target lookup (Task 5)
    internal_assoc.py# feature↔response association tool (Task 6)
    depmap.py        # local dependency/co-dependency from CRISPR cols (Task 7)
    stringdb.py      # STRING interactions + enrichment (Task 8)
    opentargets.py   # Open Targets target↔disease + tractability (Task 9)
    cbioportal.py    # cBioPortal alteration frequency (Task 10)
    pathways.py      # Reactome pathway membership (Task 11)
    literature.py    # PubMed (default) + paperclip (optional) (Task 12)
  context.py         # build grounded seed context for a compound (Task 14)
  prompts.py         # system prompt + report JSON schema (Task 15)
  report.py          # structured report → report.md + report.json (Task 15)
  agent.py           # Anthropic tool-use loop, DI client, prompt caching (Task 16)
  cli.py             # arg parsing, batch/single, orchestration (Task 17)
tests/
  conftest.py        # fixtures: sample dir path, synthetic frames (Task 1)
  test_loader.py             (Task 2)
  test_datactx.py            (Task 3)
  test_cache_base.py         (Task 4)
  test_drug_context.py       (Task 5)
  test_internal_assoc.py     (Task 6)
  test_depmap.py             (Task 7)
  test_stringdb.py           (Task 8)
  test_opentargets.py        (Task 9)
  test_cbioportal.py         (Task 10)
  test_pathways.py           (Task 11)
  test_literature.py         (Task 12)
  test_registry.py           (Task 13)
  test_context.py            (Task 14)
  test_report.py             (Task 15)
  test_agent.py              (Task 16)
  test_cli_e2e.py            (Task 17)
```

**Conventions** (match existing repo): module docstring at top; `import numpy as np` / `import pandas as pd`; type hints on public functions; small focused files. Tests use `pytest`. External HTTP is always mocked in tests via `monkeypatch` on the module's `requests` calls — **no live network in CI**.

---

## Task 1: Scaffolding, dependencies, test fixtures

**Files:**
- Modify: `pyproject.toml`
- Create: `biomarker_agent/__init__.py`
- Create: `biomarker_agent/tools/__init__.py` (temporary empty stub, filled in Task 13)
- Create: `tests/conftest.py`

- [ ] **Step 1: Add dependencies and package to `pyproject.toml`**

In `[project] dependencies`, add `"anthropic>=0.40"` and `"requests>=2.31"`. In `[project.optional-dependencies]` change `dev` to `["pytest", "ipykernel"]` (pytest already listed — keep). In `[tool.setuptools]` change `packages` to list both:

```toml
[tool.setuptools]
packages = ["biomarker_discovery", "biomarker_agent", "biomarker_agent.tools"]
```

Add a console script:

```toml
[project.scripts]
biomarker-analyze = "biomarker_agent.cli:main"
```

- [ ] **Step 2: Create package markers**

`biomarker_agent/__init__.py`:

```python
"""biomarker_agent: agentic biological interpretation of model outputs."""

__version__ = "0.1.0"
```

`biomarker_agent/tools/__init__.py`:

```python
"""Analysis tools available to the interpretation agent."""
```

- [ ] **Step 3: Create `tests/conftest.py`**

```python
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
```

- [ ] **Step 4: Sync env and verify pytest runs**

Run: `uv sync --frozen 2>/dev/null || uv sync` then `uv run pytest -q`
Expected: pytest collects 0 tests, exits 0 (no tests yet) — or "no tests ran". If `uv sync --frozen` fails because the lockfile lacks new deps, run plain `uv sync` (regenerates lock) and proceed.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock biomarker_agent tests
git commit -m "Scaffold biomarker_agent package, deps, and test fixtures"
```

---

## Task 2: Output loader

**Files:**
- Create: `biomarker_agent/loader.py`
- Test: `tests/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the output-directory loader."""

from biomarker_agent import loader


def test_parse_feature_name():
    assert loader.parse_feature_name("CRISPR_SMARCD1") == ("CRISPR", "SMARCD1")
    assert loader.parse_feature_name("GE_ITGA1") == ("GE", "ITGA1")
    # genes can contain underscores/hyphens; only the first token is the class
    assert loader.parse_feature_name("PROT_PDLIM5") == ("PROT", "PDLIM5")


def test_find_compounds_batch(sample_dir):
    dirs = loader.find_compounds(sample_dir)
    assert len(dirs) == 10
    assert all(d.name.startswith("BRD_") for d in dirs)


def test_find_compounds_single(sample_compound):
    dirs = loader.find_compounds(sample_compound)
    assert dirs == [sample_compound]


def test_load_compound(sample_compound):
    res = loader.load_compound(sample_compound)
    assert res.compound_id == "BRD:BRD-K25244359-066-03-4"
    assert res.n_samples == 530
    assert res.metrics["selected_refit_oob_pearson"] > 0.3
    names = [f.name for f in res.passing_features]
    assert "CRISPR_SMARCD1" in names
    smarcd1 = next(f for f in res.passing_features if f.name == "CRISPR_SMARCD1")
    assert smarcd1.gene == "SMARCD1"
    assert smarcd1.feature_class == "CRISPR"
    assert smarcd1.q_value < 1e-10
    # baseline top features present for at least random_forest
    assert "random_forest" in res.baseline_top_features
    assert len(res.baseline_top_features["random_forest"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_loader.py -q`
Expected: FAIL — `ModuleNotFoundError: biomarker_agent.loader` (or AttributeError).

- [ ] **Step 3: Write the implementation**

`biomarker_agent/loader.py`:

```python
"""Parse a biomarker-discovery output directory into typed result objects."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

BASELINE_MODELS = ["random_forest", "elasticnet", "lightgbm", "xgboost", "catboost"]


def parse_feature_name(name: str) -> tuple[str, str]:
    """Split a '<CLASS>_<GENE>' feature name into (class, gene).

    Only the first underscore-delimited token is the feature class; the
    remainder (which may itself contain underscores) is the gene/identifier.
    """
    cls, _, gene = name.partition("_")
    return cls, gene


@dataclass
class PassingFeature:
    name: str
    feature_class: str
    gene: str
    reproducibility: float
    mean_real_importance: float
    mean_null_importance: float
    p_value: float
    q_value: float


@dataclass
class CompoundResult:
    compound_id: str
    dir_name: str
    path: Path
    n_samples: int
    metrics: dict
    passing_features: list[PassingFeature]
    passing_by_class: dict
    baseline_top_features: dict = field(default_factory=dict)


def find_compounds(root: Path) -> list[Path]:
    """Return compound dirs. Batch if root has MANIFEST.csv; else single dir."""
    root = Path(root)
    if (root / "MANIFEST.csv").exists():
        return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("BRD_"))
    if (root / "summary.json").exists() or (root / "refract").exists():
        return [root]
    raise ValueError(f"{root} is neither a batch dir (MANIFEST.csv) nor a compound dir")


def _load_baseline_top(path: Path, top_n: int = 15) -> dict:
    out: dict = {}
    for model in BASELINE_MODELS:
        fi = path / "baselines" / model / "feature_importance.csv"
        if not fi.exists():
            continue
        df = pd.read_csv(fi).sort_values("mean_abs_shap", ascending=False).head(top_n)
        out[model] = list(zip(df["feature"].astype(str), df["mean_abs_shap"].astype(float)))
    return out


def load_compound(path: Path) -> CompoundResult:
    path = Path(path)
    summary = json.loads((path / "summary.json").read_text()) if (path / "summary.json").exists() \
        else json.loads((path / "refract" / "summary.json").read_text())

    sig_path = path / "refract" / "significant" / "significant_features.csv"
    passing: list[PassingFeature] = []
    if sig_path.exists():
        df = pd.read_csv(sig_path)
        for _, r in df.iterrows():
            cls, gene = parse_feature_name(str(r["feature_name"]))
            passing.append(
                PassingFeature(
                    name=str(r["feature_name"]),
                    feature_class=cls,
                    gene=gene,
                    reproducibility=float(r["reproducibility"]),
                    mean_real_importance=float(r["mean_real_importance"]),
                    mean_null_importance=float(r["mean_null_importance"]),
                    p_value=float(r["p_value"]),
                    q_value=float(r["q_value"]),
                )
            )

    metrics = {
        k: summary[k]
        for k in (
            "bootstrap_pred_pearson",
            "baseline_pred_pearson",
            "selected_refit_oob_pearson",
        )
        if k in summary
    }
    return CompoundResult(
        compound_id=summary["drug"],
        dir_name=path.name,
        path=path,
        n_samples=int(summary.get("n_samples", 0)),
        metrics=metrics,
        passing_features=passing,
        passing_by_class=summary.get("passing_by_class", {}),
        baseline_top_features=_load_baseline_top(path),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_loader.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/loader.py tests/test_loader.py
git commit -m "Add output-directory loader"
```

---

## Task 3: DataContext (lazy pkl loading + stats primitives)

**Files:**
- Create: `biomarker_agent/datactx.py`
- Test: `tests/test_datactx.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_datactx.py -q`
Expected: FAIL — `ModuleNotFoundError: biomarker_agent.datactx`.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/datactx.py`:

```python
"""Lazy access to the feature/response matrices and internal-stats primitives."""

from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


class DataContext:
    """Holds the big pkl matrices, loaded once, and computes feature stats."""

    def __init__(self, feature_file: Path, response_file: Path):
        self.feature_file = Path(feature_file)
        self.response_file = Path(response_file)

    @cached_property
    def features(self) -> pd.DataFrame:
        return pd.read_pickle(self.feature_file)

    @cached_property
    def responses(self) -> pd.DataFrame:
        return pd.read_pickle(self.response_file)

    def associate(self, feature_name: str, compound_id: str) -> dict:
        """Correlate one feature with one drug's response across shared lines."""
        if feature_name not in self.features.columns:
            return {"error": f"feature {feature_name!r} not in feature matrix"}
        if compound_id not in self.responses.columns:
            return {"error": f"compound {compound_id!r} not in response matrix"}

        x = self.features[feature_name]
        y = self.responses[compound_id]
        df = pd.concat([x, y], axis=1, join="inner").dropna()
        df.columns = ["x", "y"]
        n = len(df)
        if n < 10:
            return {"error": f"only {n} shared non-NaN samples", "n": n}

        pr, pp = stats.pearsonr(df["x"], df["y"])
        sr, sp = stats.spearmanr(df["x"], df["y"])
        # differential activity: top vs bottom tertile of the feature
        lo, hi = df["x"].quantile([1 / 3, 2 / 3])
        high = df.loc[df["x"] >= hi, "y"]
        low = df.loc[df["x"] <= lo, "y"]
        if len(high) >= 3 and len(low) >= 3:
            mw_u, mw_p = stats.mannwhitneyu(high, low, alternative="two-sided")
            diff_high, diff_low = float(high.mean()), float(low.mean())
        else:
            mw_p, diff_high, diff_low = float("nan"), float("nan"), float("nan")

        return {
            "feature": feature_name,
            "compound": compound_id,
            "n": int(n),
            "pearson_r": round(float(pr), 4),
            "pearson_p": float(pp),
            "spearman_r": round(float(sr), 4),
            "spearman_p": float(sp),
            "diff_high_resp_mean": round(diff_high, 4) if diff_high == diff_high else None,
            "diff_low_resp_mean": round(diff_low, 4) if diff_low == diff_low else None,
            "diff_mannwhitney_p": float(mw_p) if mw_p == mw_p else None,
            "direction": "positive" if pr >= 0 else "negative",
        }

    def dependency_profile(self, gene: str, threshold: float = -0.5) -> dict:
        """From the CRISPR_<gene> column: how dependent are cell lines?"""
        col = f"CRISPR_{gene}"
        if col not in self.features.columns:
            return {"error": f"{col} not in feature matrix"}
        v = self.features[col].dropna()
        frac = float((v < threshold).mean())
        return {
            "gene": gene,
            "n_lines": int(len(v)),
            "mean_gene_effect": round(float(v.mean()), 4),
            "frac_dependent": round(frac, 4),
            "is_selective": bool(0.01 < frac < 0.5 and v.mean() > -0.5),
        }

    def codependencies(self, gene: str, top: int = 10) -> list:
        """Top CRISPR co-dependencies (correlated gene-effect profiles)."""
        col = f"CRISPR_{gene}"
        if col not in self.features.columns:
            return []
        crispr_cols = [c for c in self.features.columns if c.startswith("CRISPR_") and c != col]
        target = self.features[col]
        out = []
        sub = self.features[crispr_cols]
        corr = sub.corrwith(target)
        corr = corr.dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(top)
        for c, r in corr.items():
            out.append({"gene": c.replace("CRISPR_", ""), "r": round(float(r), 4)})
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_datactx.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/datactx.py tests/test_datactx.py
git commit -m "Add DataContext stats primitives"
```

---

## Task 4: Cache + Tool base (HTTP + graceful errors)

**Files:**
- Create: `biomarker_agent/cache.py`
- Create: `biomarker_agent/tools/base.py`
- Test: `tests/test_cache_base.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the disk cache and Tool base helpers."""

from biomarker_agent import cache as cache_mod
from biomarker_agent.tools import base


def test_disk_cache_roundtrip(tmp_path):
    c = cache_mod.DiskCache(tmp_path / "c")
    calls = {"n": 0}

    def produce():
        calls["n"] += 1
        return {"v": 42}

    assert c.get_or_set("k1", produce) == {"v": 42}
    assert c.get_or_set("k1", produce) == {"v": 42}
    assert calls["n"] == 1  # second call served from cache


def test_tool_run_catches_errors():
    def boom(**kwargs):
        raise RuntimeError("kaboom")

    t = base.Tool(name="boom", description="d", input_schema={"type": "object", "properties": {}}, handler=boom)
    out = t.run({})
    assert "error" in out
    assert "kaboom" in out["error"]


def test_http_get_json_mock(monkeypatch):
    class FakeResp:
        status_code = 200

        def json(self):
            return {"ok": True}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(base.requests, "get", lambda *a, **k: FakeResp())
    assert base.http_get_json("http://x", params={"a": 1}) == {"ok": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cache_base.py -q`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write the implementations**

`biomarker_agent/cache.py`:

```python
"""Simple content-addressed JSON disk cache for external API calls."""

import hashlib
import json
from pathlib import Path
from typing import Callable


class DiskCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.root / f"{h}.json"

    def get_or_set(self, key: str, produce: Callable[[], dict]) -> dict:
        p = self._path(key)
        if p.exists():
            return json.loads(p.read_text())
        value = produce()
        p.write_text(json.dumps(value))
        return value
```

`biomarker_agent/tools/base.py`:

```python
"""Tool abstraction: JSON-schema declaration + graceful-failure handler."""

from dataclasses import dataclass
from typing import Callable

import requests

DEFAULT_TIMEOUT = 20


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., dict]

    def run(self, arguments: dict) -> dict:
        try:
            return self.handler(**arguments)
        except Exception as exc:  # noqa: BLE001 - tools must degrade gracefully
            return {"error": f"{type(exc).__name__}: {exc}"}

    def to_anthropic(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def http_get_json(url: str, params: dict | None = None, headers: dict | None = None,
                  timeout: int = DEFAULT_TIMEOUT) -> dict:
    resp = requests.get(url, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def http_post_json(url: str, json_body: dict, headers: dict | None = None,
                   timeout: int = DEFAULT_TIMEOUT) -> dict:
    resp = requests.post(url, json=json_body, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cache_base.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/cache.py biomarker_agent/tools/base.py tests/test_cache_base.py
git commit -m "Add disk cache and Tool base"
```

---

## Task 5: drug_context tool

**Files:**
- Create: `biomarker_agent/tools/drug_context.py`
- Test: `tests/test_drug_context.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_drug_context.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/tools/drug_context.py`:

```python
"""Tool: look up a compound's known MOA / target from treatment info."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .base import Tool

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "compound_id": {"type": "string", "description": "BRD id, e.g. 'BRD:BRD-K25244359-066-03-4'"}
    },
    "required": ["compound_id"],
}


@lru_cache(maxsize=4)
def _load(info_path: str) -> pd.DataFrame:
    return pd.read_csv(info_path)


def make_handler(info_path: Path):
    info_path = str(info_path)

    def handler(compound_id: str) -> dict:
        df = _load(info_path)
        hit = df[df["IDs"] == compound_id]
        if hit.empty:
            return {"error": f"{compound_id} not found in treatment info"}
        row = hit.iloc[0]
        return {
            "compound_id": compound_id,
            "drug_name": str(row.get("Drug.Name", "")),
            "moa": str(row.get("MOA", "") or ""),
            "targets": str(row.get("repurposing_target", "") or ""),
        }

    return handler


def make_tool(info_path: Path) -> Tool:
    return Tool(
        name="drug_context",
        description=(
            "Look up the known mechanism of action (MOA) and protein target(s) for a "
            "compound by its BRD id. Use this first to decide whether a selected feature "
            "is on-MOA (expected) or off-MOA (potentially novel)."
        ),
        input_schema=INPUT_SCHEMA,
        handler=make_handler(info_path),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_drug_context.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/drug_context.py tests/test_drug_context.py
git commit -m "Add drug_context tool"
```

---

## Task 6: internal_assoc tool

**Files:**
- Create: `biomarker_agent/tools/internal_assoc.py`
- Test: `tests/test_internal_assoc.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the internal-association tool wrapper."""

from biomarker_agent.datactx import DataContext
from biomarker_agent.tools import internal_assoc


def test_tool_wraps_datactx(synthetic_data):
    ff, rf, cid = synthetic_data
    tool = internal_assoc.make_tool(DataContext(ff, rf))
    assert tool.name == "internal_association"
    out = tool.run({"feature_name": "GE_AAA", "compound_id": cid})
    assert out["pearson_r"] > 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_internal_assoc.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/tools/internal_assoc.py`:

```python
"""Tool: associate a feature with the compound's response in the training data."""

from ..datactx import DataContext
from .base import Tool

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "feature_name": {"type": "string", "description": "Full feature name, e.g. 'GE_ITGA1'"},
        "compound_id": {"type": "string", "description": "BRD id of the compound"},
    },
    "required": ["feature_name", "compound_id"],
}


def make_tool(ctx: DataContext) -> Tool:
    return Tool(
        name="internal_association",
        description=(
            "Quantify how a feature relates to this compound's response across cell lines "
            "in the actual training data: Pearson/Spearman correlation, differential activity "
            "(high vs low feature tertile), and direction. Use to confirm a selected feature "
            "tracks response and in which direction."
        ),
        input_schema=INPUT_SCHEMA,
        handler=lambda feature_name, compound_id: ctx.associate(feature_name, compound_id),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_internal_assoc.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/internal_assoc.py tests/test_internal_assoc.py
git commit -m "Add internal_association tool"
```

---

## Task 7: depmap tool (local dependency analysis)

**Files:**
- Create: `biomarker_agent/tools/depmap.py`
- Test: `tests/test_depmap.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the local DepMap dependency tool."""

from biomarker_agent.datactx import DataContext
from biomarker_agent.tools import depmap


def test_depmap_tool(synthetic_data):
    ff, rf, cid = synthetic_data
    tool = depmap.make_tool(DataContext(ff, rf))
    out = tool.run({"gene": "BBB"})
    assert out["profile"]["is_selective"] is True
    assert "codependencies" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_depmap.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/tools/depmap.py`:

```python
"""Tool: DepMap-style dependency analysis computed from the local CRISPR features."""

from ..datactx import DataContext
from .base import Tool

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "gene": {"type": "string", "description": "Gene symbol, e.g. 'SMARCD1' (no class prefix)"},
        "top_codeps": {"type": "integer", "description": "How many co-dependencies to return", "default": 8},
    },
    "required": ["gene"],
}


def make_tool(ctx: DataContext) -> Tool:
    def handler(gene: str, top_codeps: int = 8) -> dict:
        profile = ctx.dependency_profile(gene)
        if "error" in profile:
            return profile
        return {"profile": profile, "codependencies": ctx.codependencies(gene, top=top_codeps)}

    return Tool(
        name="depmap_dependency",
        description=(
            "Analyze a gene's CRISPR knockout dependency profile across cancer cell lines "
            "(computed from the modeling data): how selectively cells depend on it, and its "
            "top co-dependencies (genes with correlated dependency, hinting at shared pathway). "
            "Use to judge whether a CRISPR/shRNA feature reflects a real, selective vulnerability."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_depmap.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/depmap.py tests/test_depmap.py
git commit -m "Add local depmap_dependency tool"
```

---

## Task 8: STRING-DB tool

Reference (consult at implementation time): the `string-database` skill. Endpoints used:
`https://string-db.org/api/json/network` and `.../api/json/enrichment`, param `identifiers` (newline-joined genes), `species=9606`, `caller_identity=biomarker_agent`.

**Files:**
- Create: `biomarker_agent/tools/stringdb.py`
- Test: `tests/test_stringdb.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the STRING-DB tool (HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import stringdb


def test_stringdb_enrichment(monkeypatch, tmp_path):
    network = [{"preferredName_A": "ITGA1", "preferredName_B": "SMARCD1", "score": 0.6}]
    enrich = [
        {"category": "Process", "term": "GO:1", "description": "cell adhesion",
         "number_of_genes": 2, "fdr": 1e-4, "preferredNames": ["ITGA1", "SMARCD1"]}
    ]

    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return enrich if "enrichment" in url else network

        return R()

    monkeypatch.setattr(stringdb.base.requests, "get", fake_get)
    tool = stringdb.make_tool(DiskCache(tmp_path))
    out = tool.run({"genes": ["ITGA1", "SMARCD1"]})
    assert out["n_genes"] == 2
    assert out["n_interactions"] == 1
    assert any("adhesion" in e["description"] for e in out["enrichment"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_stringdb.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/tools/stringdb.py`:

```python
"""Tool: STRING-DB protein interactions and functional enrichment for a gene set."""

from ..cache import DiskCache
from . import base
from .base import Tool

API = "https://string-db.org/api/json"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "genes": {"type": "array", "items": {"type": "string"},
                  "description": "Gene symbols (no class prefix), e.g. ['ITGA1','SMARCD1']"},
        "species": {"type": "integer", "description": "NCBI taxon id", "default": 9606},
    },
    "required": ["genes"],
}


def make_tool(cache: DiskCache) -> Tool:
    def handler(genes: list, species: int = 9606) -> dict:
        ids = "%0d".join(genes)
        params = {"identifiers": ids, "species": species, "caller_identity": "biomarker_agent"}
        network = cache.get_or_set(
            f"string:net:{species}:{','.join(sorted(genes))}",
            lambda: base.http_get_json(f"{API}/network", params=params),
        )
        enrichment = cache.get_or_set(
            f"string:enr:{species}:{','.join(sorted(genes))}",
            lambda: base.http_get_json(f"{API}/enrichment", params=params),
        )
        interactions = [
            {"a": e.get("preferredName_A"), "b": e.get("preferredName_B"),
             "score": round(float(e.get("score", 0)), 3)}
            for e in (network if isinstance(network, list) else [])
        ]
        enr = sorted(
            (
                {"category": e.get("category"), "term": e.get("term"),
                 "description": e.get("description"), "n_genes": e.get("number_of_genes"),
                 "fdr": e.get("fdr"), "genes": e.get("preferredNames")}
                for e in (enrichment if isinstance(enrichment, list) else [])
            ),
            key=lambda e: e["fdr"] if e["fdr"] is not None else 1.0,
        )[:15]
        return {
            "n_genes": len(genes),
            "n_interactions": len(interactions),
            "interactions": interactions[:25],
            "enrichment": enr,
        }

    return Tool(
        name="string_enrichment",
        description=(
            "Query STRING-DB for protein-protein interactions among a set of genes and their "
            "shared functional/GO/pathway enrichment. Use to see whether several selected genes "
            "physically/functionally connect or converge on a common process."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_stringdb.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/stringdb.py tests/test_stringdb.py
git commit -m "Add string_enrichment tool"
```

---

## Task 9: Open Targets tool

Reference: `opentargets-database` skill. GraphQL endpoint `https://api.platform.opentargets.org/api/v4/graphql`. Resolve symbol→Ensembl via the `search` query, then fetch `target` association to cancer EFO terms + tractability.

**Files:**
- Create: `biomarker_agent/tools/opentargets.py`
- Test: `tests/test_opentargets.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the Open Targets tool (HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import opentargets


def test_opentargets(monkeypatch, tmp_path):
    search_resp = {"data": {"search": {"hits": [{"id": "ENSG0001", "name": "ITGA1"}]}}}
    target_resp = {
        "data": {
            "target": {
                "approvedSymbol": "ITGA1",
                "tractability": [{"modality": "SM", "label": "Approved Drug", "value": True}],
                "associatedDiseases": {
                    "rows": [
                        {"disease": {"name": "cancer", "therapeuticAreas": [{"name": "neoplasm"}]},
                         "score": 0.42}
                    ]
                },
            }
        }
    }
    responses = iter([search_resp, target_resp])

    def fake_post(url, json=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self_inner):
                return next(responses)

        return R()

    monkeypatch.setattr(opentargets.base.requests, "post", fake_post)
    tool = opentargets.make_tool(DiskCache(tmp_path))
    out = tool.run({"gene": "ITGA1"})
    assert out["ensembl_id"] == "ENSG0001"
    assert out["tractability"][0]["modality"] == "SM"
    assert out["cancer_associations"][0]["score"] == 0.42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_opentargets.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/tools/opentargets.py`:

```python
"""Tool: Open Targets cancer association, tractability, and known drugs for a gene."""

from ..cache import DiskCache
from . import base
from .base import Tool

API = "https://api.platform.opentargets.org/api/v4/graphql"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {"gene": {"type": "string", "description": "Gene symbol, e.g. 'ITGA1'"}},
    "required": ["gene"],
}

_SEARCH = """query ($q:String!){ search(queryString:$q, entityNames:["target"]){
  hits{ id name } } }"""

_TARGET = """query ($id:String!){ target(ensemblId:$id){
  approvedSymbol
  tractability{ modality label value }
  associatedDiseases(page:{index:0,size:25}){
    rows{ score disease{ name therapeuticAreas{ name } } } } } }"""


def _is_cancer(row: dict) -> bool:
    areas = " ".join(a.get("name", "") for a in row.get("disease", {}).get("therapeuticAreas", []))
    name = row.get("disease", {}).get("name", "")
    blob = f"{areas} {name}".lower()
    return any(k in blob for k in ("neoplasm", "cancer", "carcinoma", "tumor", "tumour", "leukemia", "lymphoma"))


def make_tool(cache: DiskCache) -> Tool:
    def handler(gene: str) -> dict:
        sr = cache.get_or_set(
            f"ot:search:{gene}",
            lambda: base.http_post_json(API, {"query": _SEARCH, "variables": {"q": gene}}),
        )
        hits = sr.get("data", {}).get("search", {}).get("hits", [])
        if not hits:
            return {"error": f"no Open Targets hit for {gene}"}
        ens = hits[0]["id"]
        tr = cache.get_or_set(
            f"ot:target:{ens}",
            lambda: base.http_post_json(API, {"query": _TARGET, "variables": {"id": ens}}),
        )
        tgt = tr.get("data", {}).get("target") or {}
        rows = tgt.get("associatedDiseases", {}).get("rows", [])
        cancer = [
            {"disease": r["disease"]["name"], "score": round(float(r["score"]), 3)}
            for r in rows if _is_cancer(r)
        ][:10]
        return {
            "gene": gene,
            "ensembl_id": ens,
            "tractability": [
                {"modality": t.get("modality"), "label": t.get("label")}
                for t in (tgt.get("tractability") or []) if t.get("value")
            ],
            "cancer_associations": cancer,
            "max_cancer_score": max((c["score"] for c in cancer), default=0.0),
        }

    return Tool(
        name="opentargets_target",
        description=(
            "Query Open Targets for a gene's association with cancers, its druggability "
            "(tractability modalities), and whether it is an established drug target. Use to "
            "judge plausibility and novelty of a feature as an anti-cancer target."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_opentargets.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/opentargets.py tests/test_opentargets.py
git commit -m "Add opentargets_target tool"
```

---

## Task 10: cBioPortal tool

Reference: `cbioportal-database` skill. REST base `https://www.cbioportal.org/api`. v1 scope = **alteration frequency** for a gene across a default study (mutations). Survival is deferred (note below). Resolve gene→Entrez via `/genes/{symbol}`, then count mutations in a study's mutation molecular profile.

**Files:**
- Create: `biomarker_agent/tools/cbioportal.py`
- Test: `tests/test_cbioportal.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the cBioPortal tool (HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import cbioportal


def test_cbioportal(monkeypatch, tmp_path):
    gene_resp = {"entrezGeneId": 3672, "hugoGeneSymbol": "ITGA1"}
    mut_resp = [{"sampleId": "S1"}, {"sampleId": "S2"}, {"sampleId": "S1"}]

    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return gene_resp if "/genes/" in url else mut_resp

        return R()

    monkeypatch.setattr(cbioportal.base.requests, "get", fake_get)
    tool = cbioportal.make_tool(DiskCache(tmp_path))
    out = tool.run({"gene": "ITGA1"})
    assert out["entrez_id"] == 3672
    assert out["n_mutated_samples"] == 2  # S1, S2 unique
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cbioportal.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/tools/cbioportal.py`:

```python
"""Tool: cBioPortal somatic mutation frequency for a gene in a tumor study.

v1 scope: mutation frequency in a single configurable pan-cancer study. Copy-number
and survival association are deferred to a future iteration.
"""

from ..cache import DiskCache
from . import base
from .base import Tool

API = "https://www.cbioportal.org/api"
DEFAULT_STUDY = "msk_impact_2017"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "gene": {"type": "string", "description": "Gene symbol, e.g. 'ITGA1'"},
        "study_id": {"type": "string", "description": "cBioPortal study id", "default": DEFAULT_STUDY},
    },
    "required": ["gene"],
}


def make_tool(cache: DiskCache) -> Tool:
    def handler(gene: str, study_id: str = DEFAULT_STUDY) -> dict:
        g = cache.get_or_set(
            f"cbio:gene:{gene}",
            lambda: base.http_get_json(f"{API}/genes/{gene}"),
        )
        entrez = g.get("entrezGeneId")
        if not entrez:
            return {"error": f"no Entrez id for {gene}"}
        profile = f"{study_id}_mutations"
        muts = cache.get_or_set(
            f"cbio:mut:{study_id}:{entrez}",
            lambda: base.http_get_json(
                f"{API}/molecular-profiles/{profile}/mutations",
                params={"sampleListId": f"{study_id}_all", "entrezGeneId": entrez,
                        "projection": "SUMMARY"},
            ),
        )
        samples = {m.get("sampleId") for m in (muts if isinstance(muts, list) else [])}
        return {
            "gene": gene,
            "entrez_id": entrez,
            "study_id": study_id,
            "n_mutated_samples": len(samples),
        }

    return Tool(
        name="cbioportal_mutations",
        description=(
            "Query cBioPortal for how often a gene is somatically mutated in patient tumors "
            "(a configurable pan-cancer study). Use as tumor-level evidence that a gene is "
            "cancer-relevant in patients, not just cell lines."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cbioportal.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/cbioportal.py tests/test_cbioportal.py
git commit -m "Add cbioportal_mutations tool"
```

---

## Task 11: pathways tool (Reactome)

Reference: `reactome-database` skill. Use Reactome ContentService:
`https://reactome.org/ContentService/data/mapping/UniProt/{gene}/pathways` is gene-symbol-unfriendly; instead use the search/query mapping endpoint
`https://reactome.org/ContentService/search/query?query={gene}&species=Homo+sapiens&types=Pathway`. v1: return pathway names a gene maps to.

**Files:**
- Create: `biomarker_agent/tools/pathways.py`
- Test: `tests/test_pathways.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the Reactome pathways tool (HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import pathways


def test_pathways(monkeypatch, tmp_path):
    resp = {
        "results": [
            {"typeName": "Pathway", "entries": [
                {"name": "Integrin signaling", "id": "R-HSA-1", "species": "Homo sapiens"},
                {"name": "ECM interactions", "id": "R-HSA-2", "species": "Homo sapiens"},
            ]}
        ]
    }

    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return resp

        return R()

    monkeypatch.setattr(pathways.base.requests, "get", fake_get)
    tool = pathways.make_tool(DiskCache(tmp_path))
    out = tool.run({"gene": "ITGA1"})
    assert "Integrin signaling" in [p["name"] for p in out["pathways"]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pathways.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/tools/pathways.py`:

```python
"""Tool: Reactome pathway membership for a gene."""

from ..cache import DiskCache
from . import base
from .base import Tool

API = "https://reactome.org/ContentService/search/query"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {"gene": {"type": "string", "description": "Gene symbol, e.g. 'ITGA1'"}},
    "required": ["gene"],
}


def make_tool(cache: DiskCache) -> Tool:
    def handler(gene: str) -> dict:
        data = cache.get_or_set(
            f"reactome:{gene}",
            lambda: base.http_get_json(
                API,
                params={"query": gene, "species": "Homo sapiens", "types": "Pathway"},
            ),
        )
        pathways_out = []
        for group in data.get("results", []):
            if group.get("typeName") != "Pathway":
                continue
            for e in group.get("entries", []):
                if e.get("species") in (None, "Homo sapiens"):
                    pathways_out.append({"name": e.get("name"), "id": e.get("id")})
        return {"gene": gene, "n_pathways": len(pathways_out), "pathways": pathways_out[:20]}

    return Tool(
        name="reactome_pathways",
        description=(
            "List the Reactome pathways a gene participates in. Use to test whether several "
            "selected genes converge on a common pathway, suggesting a coherent mechanism."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pathways.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/pathways.py tests/test_pathways.py
git commit -m "Add reactome_pathways tool"
```

---

## Task 12: literature tool (PubMed default + paperclip optional)

Reference: `pubmed-database` skill. NCBI E-utilities `esearch`:
`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=...&retmode=json&retmax=5`. Returns `esearchresult.count` + `idlist`. Paperclip backend (paperclip.gxl.ai) is enabled only when `PAPERCLIP_API_KEY` is set or `backend='paperclip'`; built behind the same return shape.

**Files:**
- Create: `biomarker_agent/tools/literature.py`
- Test: `tests/test_literature.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the literature tool (PubMed default, HTTP mocked)."""

from biomarker_agent.cache import DiskCache
from biomarker_agent.tools import literature


def test_pubmed_counts(monkeypatch, tmp_path):
    resp = {"esearchresult": {"count": "37", "idlist": ["111", "222"]}}

    def fake_get(url, params=None, headers=None, timeout=20):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return resp

        return R()

    monkeypatch.setattr(literature.base.requests, "get", fake_get)
    tool = literature.make_tool(DiskCache(tmp_path), backend="pubmed")
    out = tool.run({"gene": "ITGA1", "context_terms": ["cancer", "apatinib"]})
    assert out["count"] == 37
    assert out["pmids"] == ["111", "222"]
    assert "ITGA1" in out["query"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_literature.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/tools/literature.py`:

```python
"""Tool: literature co-mention search. PubMed by default; paperclip optional."""

import os

from ..cache import DiskCache
from . import base
from .base import Tool

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PAPERCLIP_API = "https://paperclip.gxl.ai/api/search"
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "gene": {"type": "string", "description": "Gene symbol to search, e.g. 'ITGA1'"},
        "context_terms": {"type": "array", "items": {"type": "string"},
                          "description": "Optional extra terms ANDed in, e.g. ['cancer','apatinib']"},
    },
    "required": ["gene"],
}


def _build_query(gene: str, context_terms: list | None) -> str:
    parts = [gene] + list(context_terms or [])
    return " AND ".join(parts)


def make_tool(cache: DiskCache, backend: str = "pubmed", paperclip_key: str | None = None) -> Tool:
    paperclip_key = paperclip_key or os.environ.get("PAPERCLIP_API_KEY")

    def _pubmed(query: str) -> dict:
        data = cache.get_or_set(
            f"pubmed:{query}",
            lambda: base.http_get_json(
                EUTILS,
                params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 5},
            ),
        )
        r = data.get("esearchresult", {})
        return {"backend": "pubmed", "query": query,
                "count": int(r.get("count", 0)), "pmids": r.get("idlist", [])}

    def _paperclip(query: str) -> dict:
        data = cache.get_or_set(
            f"paperclip:{query}",
            lambda: base.http_get_json(
                PAPERCLIP_API,
                params={"q": query},
                headers={"Authorization": f"Bearer {paperclip_key}"},
            ),
        )
        return {"backend": "paperclip", "query": query,
                "count": data.get("total", len(data.get("results", []))),
                "results": data.get("results", [])[:5]}

    def handler(gene: str, context_terms: list | None = None) -> dict:
        query = _build_query(gene, context_terms)
        if backend == "paperclip":
            if not paperclip_key:
                return {"error": "paperclip backend requested but PAPERCLIP_API_KEY not set"}
            return _paperclip(query)
        return _pubmed(query)

    return Tool(
        name="literature_search",
        description=(
            "Search the biomedical literature for papers co-mentioning a gene with optional "
            "context terms (e.g. cancer + drug name). Returns hit count and identifiers. Use to "
            "gauge whether a gene-mechanism link is established (many hits) or novel (few/none)."
        ),
        input_schema=INPUT_SCHEMA,
        handler=handler,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_literature.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/literature.py tests/test_literature.py
git commit -m "Add literature_search tool (PubMed + paperclip)"
```

---

## Task 13: Tool registry

**Files:**
- Modify: `biomarker_agent/tools/__init__.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for assembling the tool registry."""

from pathlib import Path

from biomarker_agent.datactx import DataContext
from biomarker_agent.tools import build_registry


def test_build_registry(synthetic_data, tmp_path):
    ff, rf, cid = synthetic_data
    ti = tmp_path / "ti.csv"
    ti.write_text("IDs,Drug.Name,MOA,repurposing_target\nBRD:TEST-1,DRUG,MOA,GENE\n")
    reg = build_registry(
        data_ctx=DataContext(ff, rf),
        treatment_info=ti,
        cache_dir=tmp_path / "cache",
        literature_backend="pubmed",
    )
    names = set(reg.names())
    assert {
        "drug_context", "internal_association", "depmap_dependency", "string_enrichment",
        "opentargets_target", "cbioportal_mutations", "reactome_pathways", "literature_search",
    } <= names
    schemas = reg.anthropic_schemas()
    assert all("input_schema" in s for s in schemas)
    # dispatch works and degrades gracefully
    out = reg.dispatch("drug_context", {"compound_id": "BRD:TEST-1"})
    assert out["drug_name"] == "DRUG"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py -q`
Expected: FAIL — `build_registry` not defined.

- [ ] **Step 3: Write the implementation**

Replace `biomarker_agent/tools/__init__.py` with:

```python
"""Analysis tools available to the interpretation agent + registry assembly."""

from dataclasses import dataclass
from pathlib import Path

from ..cache import DiskCache
from ..datactx import DataContext
from .base import Tool
from . import (
    cbioportal,
    depmap,
    drug_context,
    internal_assoc,
    literature,
    opentargets,
    pathways,
    stringdb,
)


@dataclass
class Registry:
    tools: dict  # name -> Tool

    def names(self):
        return list(self.tools)

    def anthropic_schemas(self) -> list:
        return [t.to_anthropic() for t in self.tools.values()]

    def dispatch(self, name: str, arguments: dict) -> dict:
        if name not in self.tools:
            return {"error": f"unknown tool {name!r}"}
        return self.tools[name].run(arguments)


def build_registry(data_ctx: DataContext, treatment_info: Path, cache_dir: Path,
                   literature_backend: str = "pubmed") -> Registry:
    cache = DiskCache(cache_dir)
    tools = [
        drug_context.make_tool(treatment_info),
        internal_assoc.make_tool(data_ctx),
        depmap.make_tool(data_ctx),
        stringdb.make_tool(cache),
        opentargets.make_tool(cache),
        cbioportal.make_tool(cache),
        pathways.make_tool(cache),
        literature.make_tool(cache, backend=literature_backend),
    ]
    return Registry(tools={t.name: t for t in tools})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/__init__.py tests/test_registry.py
git commit -m "Add tool registry assembly"
```

---

## Task 14: Seed context builder

**Files:**
- Create: `biomarker_agent/context.py`
- Test: `tests/test_context.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_context.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/context.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_context.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/context.py tests/test_context.py
git commit -m "Add seed-context builder"
```

---

## Task 15: Prompts + report rendering

**Files:**
- Create: `biomarker_agent/prompts.py`
- Create: `biomarker_agent/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for report rendering from the structured submit_report payload."""

import json

from biomarker_agent import report


SAMPLE = {
    "summary": "Apatinib response tracks an integrin/SWI-SNF axis.",
    "hypotheses": [
        {
            "rank": 1,
            "title": "SMARCD1 dependency sensitizes to apatinib",
            "features": ["CRISPR_SMARCD1"],
            "mechanism": "SWI-SNF subunit loss alters VEGFR signaling.",
            "novelty": "off-MOA",
            "confidence": 0.6,
            "evidence": {"internal": "r=0.30", "depmap": "selective", "literature": "3 papers"},
        }
    ],
    "caveats": ["small n"],
}


def test_render_markdown_and_json(tmp_path):
    out_dir = tmp_path / "interpretation"
    paths = report.write_report(SAMPLE, out_dir, compound_id="BRD:TEST-1")
    md = paths["markdown"].read_text()
    assert "SMARCD1 dependency" in md
    assert "off-MOA" in md
    assert "BRD:TEST-1" in md
    loaded = json.loads(paths["json"].read_text())
    assert loaded["hypotheses"][0]["rank"] == 1


def test_report_schema_is_valid_json_schema():
    from biomarker_agent import prompts
    assert prompts.REPORT_TOOL["name"] == "submit_report"
    assert "hypotheses" in prompts.REPORT_TOOL["input_schema"]["properties"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -q`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write the implementations**

`biomarker_agent/prompts.py`:

```python
"""System prompt and the forced structured-output report tool schema."""

SYSTEM_PROMPT = """You are a cancer-biology analyst interpreting the outputs of a \
drug-response prediction model. For one compound, you are given the features that a \
resampling-based model (bootstrap ensemble + stability/significance selection) found \
reproducibly predictive of response, plus context features from baseline models.

Your job: identify the most interesting, plausible biological mechanisms by which these \
features could relate to the drug's anti-cancer activity — and flag any that look novel \
(off the drug's known mechanism of action).

Method:
1. Start from the passing features and the pre-computed internal associations provided.
2. Use the tools to triangulate evidence per gene/gene-set: known MOA (drug_context), \
internal association strength/direction, dependency selectivity (depmap_dependency), \
interactions/enrichment across the set (string_enrichment), cancer relevance + druggability \
(opentargets_target), tumor mutation frequency (cbioportal_mutations), pathway convergence \
(reactome_pathways), and literature support (literature_search).
3. Prefer hypotheses supported by MULTIPLE independent sources. Be explicit about novelty: \
on-MOA (expected given the known target) vs off-MOA (potentially novel).
4. Do not overclaim. Note when evidence is weak, conflicting, or a tool returned an error.

When finished, call submit_report exactly once with your ranked hypotheses. Do not write \
prose outside the tool call."""

REPORT_TOOL = {
    "name": "submit_report",
    "description": "Submit the final ranked interpretation. Call exactly once when done.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "2-3 sentence overall takeaway."},
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "title": {"type": "string"},
                        "features": {"type": "array", "items": {"type": "string"}},
                        "mechanism": {"type": "string",
                                      "description": "Plain-language proposed mechanism."},
                        "novelty": {"type": "string", "enum": ["on-MOA", "off-MOA", "unknown"]},
                        "confidence": {"type": "number",
                                       "description": "0-1 confidence given the evidence."},
                        "evidence": {"type": "object",
                                     "description": "Per-source evidence summary (free-form keys)."},
                    },
                    "required": ["rank", "title", "features", "mechanism", "novelty", "confidence"],
                },
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "hypotheses"],
    },
}
```

`biomarker_agent/report.py`:

```python
"""Render the structured report payload to markdown + JSON on disk."""

import json
from pathlib import Path


def render_markdown(payload: dict, compound_id: str) -> str:
    lines = [f"# Interpretation report — {compound_id}", "", payload.get("summary", ""), ""]
    for h in sorted(payload.get("hypotheses", []), key=lambda x: x.get("rank", 999)):
        lines.append(f"## {h.get('rank')}. {h.get('title')}  ·  _{h.get('novelty')}_  "
                     f"(confidence {h.get('confidence')})")
        lines.append(f"**Features:** {', '.join(h.get('features', []))}")
        lines.append("")
        lines.append(h.get("mechanism", ""))
        ev = h.get("evidence") or {}
        if ev:
            lines.append("")
            lines.append("**Evidence:**")
            for k, v in ev.items():
                lines.append(f"- _{k}_: {v}")
        lines.append("")
    caveats = payload.get("caveats") or []
    if caveats:
        lines.append("## Caveats")
        lines.extend(f"- {c}" for c in caveats)
    return "\n".join(lines).rstrip() + "\n"


def write_report(payload: dict, out_dir: Path, compound_id: str) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    md_path.write_text(render_markdown(payload, compound_id))
    json_path.write_text(json.dumps({"compound_id": compound_id, **payload}, indent=2))
    return {"markdown": md_path, "json": json_path}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_report.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/prompts.py biomarker_agent/report.py tests/test_report.py
git commit -m "Add prompts and report rendering"
```

---

## Task 16: Agent loop (Anthropic tool-use, DI client, prompt caching)

Reference: `claude-api` skill (apply prompt caching to the system block). The client is injected so tests use a fake — **no live API in CI**.

**Files:**
- Create: `biomarker_agent/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the agent loop using a fake Anthropic client."""

from biomarker_agent.agent import run_agent


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Msg:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class FakeClient:
    """Turn 1: call a tool. Turn 2: call submit_report."""

    def __init__(self):
        self.calls = 0
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _Msg(
                [
                    _Block(type="text", text="checking"),
                    _Block(type="tool_use", id="t1", name="drug_context",
                           input={"compound_id": "BRD:TEST-1"}),
                ],
                "tool_use",
            )
        return _Msg(
            [_Block(type="tool_use", id="t2", name="submit_report",
                    input={"summary": "done", "hypotheses": [
                        {"rank": 1, "title": "H", "features": ["GE_AAA"],
                         "mechanism": "m", "novelty": "off-MOA", "confidence": 0.5}]})],
            "tool_use",
        )


class FakeRegistry:
    def anthropic_schemas(self):
        return [{"name": "drug_context", "description": "d",
                 "input_schema": {"type": "object", "properties": {}}}]

    def dispatch(self, name, arguments):
        return {"drug_name": "DRUG"}


def test_run_agent_returns_report():
    client = FakeClient()
    payload, transcript = run_agent(
        client=client, registry=FakeRegistry(), system_prompt="sys",
        seed_context="ctx", model="fake", max_tool_calls=10,
    )
    assert payload["summary"] == "done"
    assert payload["hypotheses"][0]["features"] == ["GE_AAA"]
    assert client.calls == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/agent.py`:

```python
"""Anthropic tool-use loop driving one compound's investigation."""

import json

from .prompts import REPORT_TOOL

DEFAULT_MODEL = "claude-sonnet-4-6"


def _content_blocks(message):
    """Normalize message.content into a list of plain dicts for the next turn."""
    out = []
    for b in message.content:
        btype = getattr(b, "type", None)
        if btype == "text":
            out.append({"type": "text", "text": getattr(b, "text", "")})
        elif btype == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
    return out


def run_agent(client, registry, system_prompt: str, seed_context: str,
              model: str = DEFAULT_MODEL, max_tool_calls: int = 40):
    """Run the loop until submit_report is called or the call budget is hit.

    Returns (report_payload, transcript). `client` must expose
    `client.messages.create(...)` like the anthropic SDK.
    """
    tools = registry.anthropic_schemas() + [REPORT_TOOL]
    system = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": seed_context}]
    transcript = []
    used = 0

    while used < max_tool_calls:
        resp = client.messages.create(
            model=model, max_tokens=4096, system=system, tools=tools, messages=messages,
        )
        blocks = _content_blocks(resp)
        messages.append({"role": "assistant", "content": blocks})
        tool_uses = [b for b in blocks if b["type"] == "tool_use"]
        if not tool_uses:
            # model stopped without a tool call; nudge once then stop
            transcript.append({"event": "no_tool_use", "stop_reason": resp.stop_reason})
            break

        results = []
        for tu in tool_uses:
            if tu["name"] == "submit_report":
                return tu["input"], transcript
            out = registry.dispatch(tu["name"], tu["input"])
            used += 1
            transcript.append({"tool": tu["name"], "input": tu["input"], "output": out})
            results.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": json.dumps(out)[:6000],
            })
        messages.append({"role": "user", "content": results})

    # budget exhausted without report: ask once for the report explicitly
    messages.append({"role": "user",
                     "content": "Tool budget reached. Call submit_report now with your best hypotheses."})
    resp = client.messages.create(
        model=model, max_tokens=4096, system=system, tools=tools, messages=messages,
    )
    for b in _content_blocks(resp):
        if b["type"] == "tool_use" and b["name"] == "submit_report":
            return b["input"], transcript
    return {"summary": "No report produced.", "hypotheses": [],
            "caveats": ["agent did not call submit_report"]}, transcript
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/agent.py tests/test_agent.py
git commit -m "Add Anthropic tool-use agent loop"
```

---

## Task 17: CLI wiring + offline end-to-end

**Files:**
- Create: `biomarker_agent/cli.py`
- Test: `tests/test_cli_e2e.py`

- [ ] **Step 1: Write the failing test**

```python
"""End-to-end CLI test using a fake client injected via run_one."""

from pathlib import Path

from biomarker_agent import cli


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Msg:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason


class FakeClient:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        return _Msg([_Block(type="tool_use", id="t", name="submit_report", input={
            "summary": "ok",
            "hypotheses": [{"rank": 1, "title": "H", "features": ["CRISPR_SMARCD1"],
                            "mechanism": "m", "novelty": "off-MOA", "confidence": 0.5}],
        })])


def test_run_one_writes_report(sample_compound, tmp_path):
    data = Path("data")
    out = cli.run_one(
        compound_dir=sample_compound,
        out_dir=tmp_path / "out",
        feature_file=data / "x-all_v4.pkl",
        response_file=data / "responses_primary_v4.pkl",
        treatment_info=data / "primary_screen_treatment_info.csv",
        cache_dir=tmp_path / "cache",
        client=FakeClient(),
        model="fake",
        literature_backend="pubmed",
        max_tool_calls=5,
    )
    assert out["markdown"].exists()
    assert "BRD:BRD-K25244359-066-03-4" in out["markdown"].read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_e2e.py -q`
Expected: FAIL — `run_one` not defined.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/cli.py`:

```python
"""Command-line entry point: point at an output dir, get interpretation reports."""

import argparse
import os
from pathlib import Path

from . import context, report
from .agent import DEFAULT_MODEL, run_agent
from .datactx import DataContext
from .loader import find_compounds, load_compound
from .prompts import SYSTEM_PROMPT
from .tools import build_registry

DATA = Path("data")


def run_one(compound_dir, out_dir, feature_file, response_file, treatment_info, cache_dir,
            client, model=DEFAULT_MODEL, literature_backend="pubmed", max_tool_calls=40):
    """Analyze a single compound dir and write its report. Returns report paths."""
    result = load_compound(compound_dir)
    data_ctx = DataContext(feature_file, response_file)
    registry = build_registry(
        data_ctx=data_ctx, treatment_info=treatment_info, cache_dir=cache_dir,
        literature_backend=literature_backend,
    )
    drug_info = registry.dispatch("drug_context", {"compound_id": result.compound_id})
    internal = context.precompute_internal(result, data_ctx)
    seed = context.build_seed_context(result, drug_info=drug_info, internal=internal)
    payload, _ = run_agent(
        client=client, registry=registry, system_prompt=SYSTEM_PROMPT,
        seed_context=seed, model=model, max_tool_calls=max_tool_calls,
    )
    return report.write_report(payload, Path(out_dir), result.compound_id)


def _make_client():
    import anthropic
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def main(argv=None):
    p = argparse.ArgumentParser(description="Biological interpretation of biomarker-discovery outputs")
    p.add_argument("target", help="Output dir (batch w/ MANIFEST.csv) or a single BRD_* compound dir")
    p.add_argument("--out", default=None, help="Output root (default: <compound>/interpretation)")
    p.add_argument("--feature-file", default=str(DATA / "x-all_v4.pkl"))
    p.add_argument("--response-file", default=str(DATA / "responses_primary_v4.pkl"))
    p.add_argument("--treatment-info", default=str(DATA / "primary_screen_treatment_info.csv"))
    p.add_argument("--cache-dir", default=".biomarker_agent_cache")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--literature", choices=["pubmed", "paperclip"], default="pubmed")
    p.add_argument("--max-tool-calls", type=int, default=40)
    args = p.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ERROR: ANTHROPIC_API_KEY is not set.")

    client = _make_client()
    compounds = find_compounds(Path(args.target))
    index_lines = ["# Interpretation index", ""]
    for cdir in compounds:
        out_dir = Path(args.out) / cdir.name if args.out else cdir / "interpretation"
        paths = run_one(
            compound_dir=cdir, out_dir=out_dir,
            feature_file=Path(args.feature_file), response_file=Path(args.response_file),
            treatment_info=Path(args.treatment_info), cache_dir=Path(args.cache_dir),
            client=client, model=args.model, literature_backend=args.literature,
            max_tool_calls=args.max_tool_calls,
        )
        print(f"[done] {cdir.name} -> {paths['markdown']}")
        index_lines.append(f"- {cdir.name}: {paths['markdown']}")

    if len(compounds) > 1 and args.out:
        (Path(args.out) / "interpretation_index.md").write_text("\n".join(index_lines) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_e2e.py -q`
Expected: PASS. (Loads the real pkls — may take a few seconds.)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add biomarker_agent/cli.py tests/test_cli_e2e.py
git commit -m "Add CLI entry point and offline end-to-end test"
```

---

## Task 18: Live smoke test + docs

**Files:**
- Create: `docs/biomarker_agent.md`
- Modify: `README.md`

- [ ] **Step 1: Live smoke test (manual, not CI)**

Only if `ANTHROPIC_API_KEY` is set. Run a single real compound and eyeball the report:

```bash
uv run biomarker-analyze data/small_test_sample/BRD_BRD-K25244359-066-03-4 \
    --out /tmp/bioagent_smoke --max-tool-calls 12
```

Expected: completes without error; `/tmp/bioagent_smoke/BRD_BRD-K25244359-066-03-4/report.md` exists, names real genes (e.g. SMARCD1, ITGA1), labels at least one hypothesis on-/off-MOA relative to Apatinib's RET/KDR/KIT MOA, and cites tool evidence. If a tool errored on live data, fix that tool's parsing against the real response and re-run (cache makes re-runs cheap). Document any API quirks found in `docs/biomarker_agent.md`.

- [ ] **Step 2: Write `docs/biomarker_agent.md`**

```markdown
# Biomarker Interpretation Agent

Points an LLM agent at a biomarker-discovery output directory and produces ranked,
evidence-backed biological-mechanism hypotheses per compound.

## Usage

```bash
export ANTHROPIC_API_KEY=sk-...           # required
# optional: export PAPERCLIP_API_KEY=...  # enables --literature paperclip

# whole batch (dir containing MANIFEST.csv)
uv run biomarker-analyze data/small_test_sample --out results/interpretation

# one compound (writes to <dir>/interpretation by default)
uv run biomarker-analyze data/small_test_sample/BRD_BRD-K25244359-066-03-4
```

Flags: `--model`, `--literature {pubmed,paperclip}`, `--max-tool-calls`,
`--feature-file`, `--response-file`, `--treatment-info`, `--cache-dir`.

## Tools the agent can call

drug_context, internal_association, depmap_dependency, string_enrichment,
opentargets_target, cbioportal_mutations, reactome_pathways, literature_search.

All external calls are cached under `--cache-dir` (default `.biomarker_agent_cache`),
so re-runs are cheap and resilient to transient API failures. Each tool degrades
gracefully — a failing API returns an `{"error": ...}` the agent works around.

## Output

Per compound: `report.md` (human) + `report.json` (machine). Batch runs with `--out`
also write `interpretation_index.md`.
```

- [ ] **Step 3: Add a pointer to `README.md`**

Append to `README.md`:

```markdown

## Interpretation agent

After running the benchmark, interpret the selected biomarkers with an LLM agent:
see [`docs/biomarker_agent.md`](docs/biomarker_agent.md). Quick start:
`uv run biomarker-analyze <output_dir>` (needs `ANTHROPIC_API_KEY`).
```

- [ ] **Step 4: Add cache dir to `.gitignore`**

Append to `.gitignore`:

```
# biomarker_agent external-API cache
.biomarker_agent_cache/
```

- [ ] **Step 5: Commit**

```bash
git add docs/biomarker_agent.md README.md .gitignore
git commit -m "Add biomarker_agent docs and ignore cache dir"
```

---

## Final verification

- [ ] Run full suite: `uv run pytest -q` → all green.
- [ ] Confirm no live network in CI tests: `grep -rL "monkeypatch" tests/test_stringdb.py tests/test_opentargets.py tests/test_cbioportal.py tests/test_pathways.py tests/test_literature.py` returns nothing (every external-tool test mocks HTTP).
- [ ] Confirm graceful degradation: every external tool returns `{"error": ...}` on exception (via `Tool.run`).
```
