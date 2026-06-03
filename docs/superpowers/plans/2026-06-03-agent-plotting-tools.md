# Agent Plotting Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the interpretation agent publication-quality plotting tools that save PNGs and get embedded per-hypothesis in the final report.

**Architecture:** A shared `plotstyle` module (one consistent matplotlib look) + a pure `plots` module (data → PNG) + thin `tools/figures.py` `Tool` wrappers that bind the output dir and data sources. The agent attaches returned figure paths to hypotheses in `submit_report`; `report.render_markdown` embeds them.

**Tech Stack:** Python 3.12, matplotlib 3.10 (Agg backend), pandas/numpy/scipy (present), pytest. Managed by `uv`.

---

## File Structure

```
biomarker_agent/
  plotstyle.py        # apply_style(), PALETTE/OKABE_ITO, finalize(fig, path)   (Task 1)
  plots.py            # 8 pure plotting functions: data -> PNG -> path          (Tasks 2,3)
  tools/
    figures.py        # make_figure_tools(...) -> [Tool]; per-plot wrappers     (Task 4)
  tools/__init__.py   # build_registry gains figures_dir/compound_result        (Task 5)
  cli.py              # run_one passes figures dir + compound_result            (Task 5)
  prompts.py          # hypothesis schema gains optional `figures`; prompt note (Task 6)
  report.py           # render_markdown embeds attached figures                 (Task 6)
tests/
  test_plotstyle.py        (Task 1)
  test_plots_data.py       (Task 2)
  test_plots_external.py   (Task 3)
  test_figures.py          (Task 4)
  test_registry.py (modify) + test_cli_e2e.py (modify)   (Task 5)
  test_report.py (modify) + test_prompts via test_report (Task 6)
```

**Conventions:** module docstring at top; type hints on public functions; small focused files. All figure files are written under `<out_dir>/figures/` and tools return paths **relative to `out_dir`** (`figures/<slug>.png`).

---

## Task 1: Shared plot style

**Files:**
- Create: `biomarker_agent/plotstyle.py`
- Test: `tests/test_plotstyle.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the shared matplotlib style helpers."""

from pathlib import Path

from biomarker_agent import plotstyle


def test_palette_and_cycle():
    assert plotstyle.PALETTE["positive"] != plotstyle.PALETTE["negative"]
    assert len(plotstyle.OKABE_ITO) >= 7
    # all palette/cycle colors are hex strings
    for c in plotstyle.OKABE_ITO:
        assert c.startswith("#") and len(c) == 7


def test_apply_style_sets_publication_rcparams():
    plotstyle.apply_style()
    import matplotlib as mpl
    assert mpl.get_backend().lower() == "agg"
    assert mpl.rcParams["axes.spines.top"] is False
    assert mpl.rcParams["axes.spines.right"] is False
    assert mpl.rcParams["savefig.dpi"] == 300


def test_finalize_writes_png_and_closes(tmp_path):
    plotstyle.apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "f.png"
    returned = plotstyle.finalize(fig, out)
    assert Path(returned) == out
    assert out.exists() and out.stat().st_size > 500
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature
    assert plt.fignum_exists(fig.number) is False  # figure closed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plotstyle.py -q`
Expected: FAIL — `ModuleNotFoundError: biomarker_agent.plotstyle`.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/plotstyle.py`:

```python
"""Shared, publication-quality matplotlib styling for agent figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from cycler import cycler  # noqa: E402

# Okabe-Ito colorblind-safe qualitative palette.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]

PALETTE = {
    "primary": "#0072B2",    # blue
    "secondary": "#999999",  # grey (context)
    "accent": "#D55E00",     # vermillion
    "positive": "#009E73",   # green
    "negative": "#D55E00",   # vermillion
    "neutral": "#444444",
}


def apply_style() -> None:
    """Apply a clean, consistent look used by every figure."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.constrained_layout.use": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "medium",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "axes.prop_cycle": cycler(color=OKABE_ITO),
    })


def finalize(fig, path) -> str:
    """Save at 300 DPI and close the figure. Returns the path as a string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_plotstyle.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/plotstyle.py tests/test_plotstyle.py
git commit -m "Add shared publication-quality plot style"
```

---

## Task 2: Data-driven plotting functions

**Files:**
- Create: `biomarker_agent/plots.py`
- Test: `tests/test_plots_data.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plots_data.py -q`
Expected: FAIL — `ModuleNotFoundError: biomarker_agent.plots`.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/plots.py`:

```python
"""Pure, publication-quality plotting functions: data in, PNG path out.

Each function builds one figure with the shared style and saves it. No knowledge
of the agent or tool layer — fully unit-testable.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .plotstyle import PALETTE, apply_style, finalize

apply_style()

import matplotlib.pyplot as plt  # noqa: E402


def feature_response(x: pd.Series, y: pd.Series, feature_name: str,
                     compound_id: str, path) -> str:
    """Scatter of a feature vs a compound's response, with OLS fit and stats."""
    df = pd.concat([x, y], axis=1, join="inner").dropna()
    df.columns = ["x", "y"]
    if len(df) < 3:
        raise ValueError(f"need >=3 shared points, got {len(df)}")
    pr, pp = stats.pearsonr(df["x"], df["y"])
    sr, _ = stats.spearmanr(df["x"], df["y"])
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    xs = np.linspace(df["x"].min(), df["x"].max(), 100)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df["x"], df["y"], s=22, color=PALETTE["primary"], alpha=0.7,
               edgecolor="white", linewidth=0.4)
    ax.plot(xs, slope * xs + intercept, color=PALETTE["accent"], linewidth=2)
    ax.set_xlabel(feature_name)
    ax.set_ylabel(f"{compound_id} response")
    ax.set_title(f"{feature_name} vs response")
    ax.annotate(f"Pearson r = {pr:.2f} (p = {pp:.1e})\nSpearman r = {sr:.2f}\nn = {len(df)}",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top", ha="left",
                fontsize=9, color=PALETTE["neutral"],
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PALETTE["secondary"], alpha=0.8))
    return finalize(fig, path)


def feature_panel(corr: pd.DataFrame, compound_id: str, path) -> str:
    """Heatmap of a correlation matrix among features (+ a response row/col)."""
    labels = list(corr.columns)
    fig, ax = plt.subplots(figsize=(0.7 * len(labels) + 2.5, 0.7 * len(labels) + 2))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(v) > 0.6 else PALETTE["neutral"], fontsize=8)
    ax.set_title(f"Feature correlations · {compound_id}")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r")
    return finalize(fig, path)


def dependency_distribution(effect: pd.Series, gene: str, path, threshold: float = -0.5) -> str:
    """Histogram of CRISPR gene-effect with the dependency threshold marked."""
    eff = pd.Series(effect).dropna()
    if len(eff) < 3:
        raise ValueError("need >=3 values")
    frac = float((eff < threshold).mean())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(eff, bins=40, color=PALETTE["primary"], alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.axvline(threshold, color=PALETTE["accent"], linestyle="--", linewidth=1.5,
               label=f"dependency threshold ({threshold})")
    ax.set_xlabel(f"CRISPR gene effect · {gene}")
    ax.set_ylabel("cell lines")
    ax.set_title(f"{gene} dependency across cell lines")
    ax.annotate(f"{frac*100:.1f}% dependent\nn = {len(eff)}", xy=(0.03, 0.97),
                xycoords="axes fraction", va="top", fontsize=9, color=PALETTE["neutral"])
    ax.legend(loc="upper right")
    return finalize(fig, path)


def codependency_bar(codeps: list, gene: str, path) -> str:
    """Horizontal bar of top CRISPR co-dependencies by signed correlation."""
    if not codeps:
        raise ValueError("no co-dependencies to plot")
    items = sorted(codeps, key=lambda d: abs(d["r"]))
    names = [d["gene"] for d in items]
    vals = [d["r"] for d in items]
    colors = [PALETTE["positive"] if v >= 0 else PALETTE["negative"] for v in vals]
    fig, ax = plt.subplots(figsize=(5, 0.4 * len(items) + 1.5))
    ax.barh(range(len(items)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(items)), names)
    ax.axvline(0, color=PALETTE["neutral"], linewidth=0.8)
    ax.set_xlabel("co-dependency (Pearson r of gene-effect)")
    ax.set_title(f"{gene} top co-dependencies")
    ax.grid(axis="y", visible=False)
    return finalize(fig, path)


def passing_importance(features: list, path) -> str:
    """Grouped horizontal bars of passing features' real vs null importance."""
    if not features:
        raise ValueError("no features to plot")
    feats = sorted(features, key=lambda f: f["mean_real_importance"])
    names = [f["name"] for f in feats]
    real = [f["mean_real_importance"] for f in feats]
    null = [f["mean_null_importance"] for f in feats]
    y = np.arange(len(feats))
    h = 0.38
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(feats) + 1.6))
    ax.barh(y + h / 2, real, height=h, color=PALETTE["primary"], label="real")
    ax.barh(y - h / 2, null, height=h, color=PALETTE["secondary"], label="null")
    ax.set_yticks(y, names)
    ax.set_xlabel("mean importance")
    ax.set_title("Passing features: real vs null importance")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return finalize(fig, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_plots_data.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/plots.py tests/test_plots_data.py
git commit -m "Add data-driven plotting functions"
```

---

## Task 3: External-output plotting functions

**Files:**
- Modify: `biomarker_agent/plots.py` (append functions)
- Test: `tests/test_plots_external.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for plotting functions that visualize external-tool outputs."""

from pathlib import Path

from biomarker_agent import plots

PNG = b"\x89PNG\r\n\x1a\n"


def _is_png(p):
    p = Path(p)
    return p.exists() and p.stat().st_size > 500 and p.read_bytes()[:8] == PNG


def test_string_network(tmp_path):
    interactions = [{"a": "TP53", "b": "MDM2", "score": 0.99},
                    {"a": "TP53", "b": "CDKN1A", "score": 0.9}]
    out = plots.string_network(["TP53", "MDM2", "CDKN1A"], interactions, tmp_path / "sn.png")
    assert _is_png(out)


def test_string_network_no_edges_still_plots_nodes(tmp_path):
    out = plots.string_network(["A", "B"], [], tmp_path / "sn2.png")
    assert _is_png(out)


def test_mutation_frequency(tmp_path):
    out = plots.mutation_frequency([("TP53", 4538), ("ITGA1", 0)], tmp_path / "mf.png")
    assert _is_png(out)


def test_pathway_membership(tmp_path):
    mapping = {"ITGA1": ["Integrin interactions", "Laminin interactions"],
               "PDLIM5": ["Integrin interactions"]}
    out = plots.pathway_membership(mapping, tmp_path / "pm.png")
    assert _is_png(out)


def test_pathway_membership_empty_errors(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        plots.pathway_membership({}, tmp_path / "x.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plots_external.py -q`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Append the implementation to `biomarker_agent/plots.py`**

```python
def string_network(genes: list, interactions: list, path) -> str:
    """Circular-layout interaction graph among genes; edge width ~ score."""
    n = len(genes)
    if n == 0:
        raise ValueError("no genes")
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {g: (np.cos(a), np.sin(a)) for g, a in zip(genes, angles)}
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for e in interactions:
        a, b, s = e.get("a"), e.get("b"), float(e.get("score", 0))
        if a in pos and b in pos:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            ax.plot([x1, x2], [y1, y2], color=PALETTE["secondary"],
                    linewidth=0.5 + 3 * s, alpha=0.6, zorder=1)
    xs = [pos[g][0] for g in genes]
    ys = [pos[g][1] for g in genes]
    ax.scatter(xs, ys, s=320, color=PALETTE["primary"], zorder=2, edgecolor="white", linewidth=1.2)
    for g in genes:
        ax.annotate(g, pos[g], ha="center", va="center", fontsize=9,
                    color="white", zorder=3, fontweight="bold")
    ax.set_title(f"STRING interactions ({len(interactions)} edges)")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    return finalize(fig, path)


def mutation_frequency(counts: list, path) -> str:
    """Bar of per-gene mutated-sample counts (cBioPortal)."""
    if not counts:
        raise ValueError("no counts")
    items = sorted(counts, key=lambda c: c[1], reverse=True)
    names = [c[0] for c in items]
    vals = [c[1] for c in items]
    fig, ax = plt.subplots(figsize=(0.6 * len(items) + 2.5, 4))
    bars = ax.bar(range(len(items)), vals, color=PALETTE["primary"], alpha=0.85)
    ax.set_xticks(range(len(items)), names, rotation=45, ha="right")
    ax.set_ylabel("mutated samples")
    ax.set_title("cBioPortal somatic mutation frequency")
    ax.grid(axis="x", visible=False)
    ax.bar_label(bars, fmt="%d", fontsize=8, padding=2)
    return finalize(fig, path)


def pathway_membership(mapping: dict, path, max_pathways: int = 15) -> str:
    """Genes x pathways binary membership heatmap (pathway convergence)."""
    if not mapping:
        raise ValueError("no pathway mapping")
    from collections import Counter
    freq = Counter(p for paths in mapping.values() for p in paths)
    pathways = [p for p, _ in freq.most_common(max_pathways)]
    if not pathways:
        raise ValueError("no pathways")
    genes = list(mapping)
    mat = np.array([[1 if p in mapping[g] else 0 for p in pathways] for g in genes], dtype=float)
    fig, ax = plt.subplots(figsize=(0.45 * len(pathways) + 3, 0.5 * len(genes) + 2))
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pathways)), pathways, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(genes)), genes)
    ax.set_title("Reactome pathway membership")
    ax.grid(False)
    return finalize(fig, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_plots_external.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/plots.py tests/test_plots_external.py
git commit -m "Add external-output plotting functions"
```

---

## Task 4: Figure tool wrappers

**Files:**
- Create: `biomarker_agent/tools/figures.py`
- Test: `tests/test_figures.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the figure Tool wrappers."""

from pathlib import Path

from biomarker_agent.cache import DiskCache
from biomarker_agent.datactx import DataContext
from biomarker_agent.loader import CompoundResult, PassingFeature
from biomarker_agent.tools import figures

PNG = b"\x89PNG\r\n\x1a\n"


def _png_ok(out_dir, rel):
    p = Path(out_dir) / rel
    return p.exists() and p.read_bytes()[:8] == PNG


def _compound_result():
    return CompoundResult(
        compound_id="BRD:TEST-1", dir_name="BRD_TEST-1", path=None, n_samples=60,
        metrics={}, passing_features=[
            PassingFeature("CRISPR_BBB", "CRISPR", "BBB", 0.7, 0.02, 0.002, 1e-9, 1e-6),
            PassingFeature("GE_AAA", "GE", "AAA", 0.6, 0.01, 0.003, 1e-6, 1e-4),
        ], passing_by_class={"CRISPR": 1, "GE": 1})


def _tools(tmp_path, synthetic_data):
    ff, rf, cid = synthetic_data
    figs_dir = tmp_path / "out" / "figures"
    reg = figures.make_figure_tools(
        figures_dir=figs_dir, rel_prefix="figures",
        data_ctx=DataContext(ff, rf), compound_result=_compound_result(),
        cache=DiskCache(tmp_path / "cache"),
    )
    return {t.name: t for t in reg}, tmp_path / "out"


def test_make_figure_tools_names(tmp_path, synthetic_data):
    tools, _ = _tools(tmp_path, synthetic_data)
    assert {
        "plot_feature_response", "plot_feature_panel", "plot_dependency_distribution",
        "plot_codependency_bar", "plot_passing_importance", "plot_string_network",
        "plot_mutation_frequency", "plot_pathway_membership",
    } == set(tools)


def test_feature_response_tool(tmp_path, synthetic_data):
    tools, out = _tools(tmp_path, synthetic_data)
    res = tools["plot_feature_response"].run({"feature": "GE_AAA"})
    assert res["figure"].startswith("figures/")
    assert "caption" in res
    assert _png_ok(out, res["figure"])


def test_dependency_tool(tmp_path, synthetic_data):
    tools, out = _tools(tmp_path, synthetic_data)
    res = tools["plot_dependency_distribution"].run({"gene": "BBB"})
    assert _png_ok(out, res["figure"])


def test_passing_importance_tool(tmp_path, synthetic_data):
    tools, out = _tools(tmp_path, synthetic_data)
    res = tools["plot_passing_importance"].run({})
    assert _png_ok(out, res["figure"])


def test_feature_response_unknown_feature_errors(tmp_path, synthetic_data):
    tools, _ = _tools(tmp_path, synthetic_data)
    res = tools["plot_feature_response"].run({"feature": "GE_NOPE"})
    assert "error" in res


def test_mutation_frequency_tool_mocked(tmp_path, synthetic_data, monkeypatch):
    from biomarker_agent.tools import cbioportal
    monkeypatch.setattr(cbioportal, "make_tool",
                        lambda cache: _FakeTool({"gene": "X", "n_mutated_samples": 5}))
    tools, out = _tools(tmp_path, synthetic_data)
    res = tools["plot_mutation_frequency"].run({"genes": ["AAA", "BBB"]})
    assert _png_ok(out, res["figure"])


class _FakeTool:
    def __init__(self, payload):
        self._p = payload

    def run(self, args):
        return dict(self._p, gene=args.get("gene", "X"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_figures.py -q`
Expected: FAIL — `biomarker_agent.tools.figures` not found.

- [ ] **Step 3: Write the implementation**

`biomarker_agent/tools/figures.py`:

```python
"""Figure-generating Tool wrappers.

Each tool computes a deterministic filename, calls a pure function in
`biomarker_agent.plots`, and returns {"figure": "<rel_prefix>/<slug>.png",
"caption": ...} or {"error": ...}. Figures are written under `figures_dir`.
"""

import re
from pathlib import Path

from .. import plots
from ..cache import DiskCache
from ..datactx import DataContext
from . import cbioportal, pathways, stringdb
from .base import Tool


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def make_figure_tools(figures_dir, rel_prefix: str, data_ctx: DataContext,
                      compound_result, cache: DiskCache) -> list:
    figures_dir = Path(figures_dir)
    cid = compound_result.compound_id

    def _abs(name: str):
        return figures_dir / name, f"{rel_prefix}/{name}"

    def _feature_response(feature: str) -> dict:
        if feature not in data_ctx.features.columns:
            return {"error": f"feature {feature!r} not in matrix"}
        if cid not in data_ctx.responses.columns:
            return {"error": f"compound {cid!r} not in responses"}
        ap, rel = _abs(f"feature_response__{_slug(feature)}.png")
        plots.feature_response(data_ctx.features[feature], data_ctx.responses[cid],
                               feature, cid, ap)
        return {"figure": rel, "caption": f"{feature} vs {cid} response across cell lines"}

    def _feature_panel(features: list) -> dict:
        cols = [f for f in features if f in data_ctx.features.columns]
        if len(cols) < 2:
            return {"error": "need >=2 known features"}
        sub = data_ctx.features[cols].join(
            data_ctx.responses[cid].rename("response"), how="inner").dropna()
        if len(sub) < 3:
            return {"error": "too few shared samples"}
        ap, rel = _abs(f"feature_panel__{_slug('_'.join(cols))}.png")
        plots.feature_panel(sub.corr(), cid, ap)
        return {"figure": rel, "caption": f"Correlations among {len(cols)} features and response"}

    def _dependency(gene: str) -> dict:
        col = f"CRISPR_{gene}"
        if col not in data_ctx.features.columns:
            return {"error": f"{col} not in matrix"}
        ap, rel = _abs(f"dependency__{_slug(gene)}.png")
        plots.dependency_distribution(data_ctx.features[col], gene, ap)
        return {"figure": rel, "caption": f"{gene} CRISPR dependency distribution"}

    def _codependency(gene: str) -> dict:
        codeps = data_ctx.codependencies(gene)
        if not codeps:
            return {"error": f"no co-dependencies for {gene}"}
        ap, rel = _abs(f"codependency__{_slug(gene)}.png")
        plots.codependency_bar(codeps, gene, ap)
        return {"figure": rel, "caption": f"{gene} top CRISPR co-dependencies"}

    def _passing_importance() -> dict:
        feats = [{"name": f.name, "mean_real_importance": f.mean_real_importance,
                  "mean_null_importance": f.mean_null_importance}
                 for f in compound_result.passing_features]
        if not feats:
            return {"error": "no passing features"}
        ap, rel = _abs("passing_importance.png")
        plots.passing_importance(feats, ap)
        return {"figure": rel, "caption": "Passing features: real vs null importance"}

    def _string_network(genes: list) -> dict:
        data = stringdb.make_tool(cache).run({"genes": genes})
        if "error" in data:
            return data
        ap, rel = _abs(f"string__{_slug('_'.join(genes))}.png")
        plots.string_network(genes, data.get("interactions", []), ap)
        return {"figure": rel, "caption": f"STRING interactions among {len(genes)} genes"}

    def _mutation_frequency(genes: list) -> dict:
        tool = cbioportal.make_tool(cache)
        counts = []
        for g in genes:
            out = tool.run({"gene": g})
            counts.append((g, int(out.get("n_mutated_samples", 0)) if "error" not in out else 0))
        ap, rel = _abs(f"mutations__{_slug('_'.join(genes))}.png")
        plots.mutation_frequency(counts, ap)
        return {"figure": rel, "caption": "cBioPortal mutation frequency per gene"}

    def _pathway_membership(genes: list) -> dict:
        tool = pathways.make_tool(cache)
        mapping = {}
        for g in genes:
            out = tool.run({"gene": g})
            if "error" not in out:
                mapping[g] = [p["name"] for p in out.get("pathways", [])]
        if not any(mapping.values()):
            return {"error": "no pathways found for given genes"}
        ap, rel = _abs(f"pathways__{_slug('_'.join(genes))}.png")
        plots.pathway_membership(mapping, ap)
        return {"figure": rel, "caption": "Reactome pathway membership across genes"}

    feat_schema = {"type": "object", "properties": {
        "feature": {"type": "string", "description": "Full feature name, e.g. 'GE_ITGA1'"}},
        "required": ["feature"]}
    gene_schema = {"type": "object", "properties": {
        "gene": {"type": "string", "description": "Gene symbol (no class prefix)"}},
        "required": ["gene"]}
    genes_schema = {"type": "object", "properties": {
        "genes": {"type": "array", "items": {"type": "string"},
                  "description": "Gene symbols (no class prefix)"}},
        "required": ["genes"]}
    features_schema = {"type": "object", "properties": {
        "features": {"type": "array", "items": {"type": "string"},
                     "description": "Full feature names, e.g. ['GE_ITGA1','PROT_PDLIM5']"}},
        "required": ["features"]}
    empty_schema = {"type": "object", "properties": {}}

    return [
        Tool("plot_feature_response",
             "Scatter of a feature vs this compound's response with regression line and r/p/n. "
             "Use to visualize and adjudicate the direction/strength of a feature's association.",
             feat_schema, _feature_response),
        Tool("plot_feature_panel",
             "Correlation heatmap among several features and the response together. Use to show "
             "how a combination of features relates to response and to each other (collinearity).",
             features_schema, _feature_panel),
        Tool("plot_dependency_distribution",
             "Histogram of a gene's CRISPR knockout effect across cell lines with the dependency "
             "threshold marked. Use to visualize selective dependency.",
             gene_schema, _dependency),
        Tool("plot_codependency_bar",
             "Bar chart of a gene's top CRISPR co-dependencies. Use to show shared-complex/pathway "
             "structure behind a dependency.",
             gene_schema, _codependency),
        Tool("plot_passing_importance",
             "Bar chart of the passing features' real vs null importance for this compound. Use as "
             "an overview of what the resampled model selected.",
             empty_schema, lambda: _passing_importance()),
        Tool("plot_string_network",
             "STRING protein-interaction network among a set of genes. Use to visualize whether "
             "selected genes physically/functionally connect.",
             genes_schema, _string_network),
        Tool("plot_mutation_frequency",
             "Bar of cBioPortal somatic mutation counts per gene. Use to visualize tumor-level "
             "alteration frequency across selected genes.",
             genes_schema, _mutation_frequency),
        Tool("plot_pathway_membership",
             "Heatmap of which Reactome pathways each gene belongs to. Use to visualize pathway "
             "convergence across selected genes.",
             genes_schema, _pathway_membership),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_figures.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add biomarker_agent/tools/figures.py tests/test_figures.py
git commit -m "Add figure tool wrappers"
```

---

## Task 5: Wire figure tools into the registry and CLI

**Files:**
- Modify: `biomarker_agent/tools/__init__.py`
- Modify: `biomarker_agent/cli.py`
- Test: modify `tests/test_registry.py`, `tests/test_cli_e2e.py`

- [ ] **Step 1: Write the failing test (registry)**

Append to `tests/test_registry.py`:

```python
def test_build_registry_with_figures(synthetic_data, tmp_path):
    from biomarker_agent.loader import CompoundResult
    ff, rf, cid = synthetic_data
    ti = tmp_path / "ti.csv"
    ti.write_text("IDs,Drug.Name,MOA,repurposing_target\nBRD:TEST-1,DRUG,MOA,GENE\n")
    cr = CompoundResult(compound_id=cid, dir_name="d", path=None, n_samples=60,
                        metrics={}, passing_features=[], passing_by_class={})
    reg = build_registry(
        data_ctx=DataContext(ff, rf), treatment_info=ti, cache_dir=tmp_path / "c",
        figures_dir=tmp_path / "out" / "figures", compound_result=cr,
    )
    names = set(reg.names())
    assert "plot_feature_response" in names
    assert "plot_string_network" in names
    # data tools still present
    assert "internal_association" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_registry.py::test_build_registry_with_figures -q`
Expected: FAIL — `build_registry() got an unexpected keyword argument 'figures_dir'`.

- [ ] **Step 3: Modify `build_registry`**

In `biomarker_agent/tools/__init__.py`, add the import near the other tool imports:

```python
from . import figures as figures_mod
```

Replace the `build_registry` function with:

```python
def build_registry(data_ctx: DataContext, treatment_info: Path, cache_dir: Path,
                   literature_backend: str = "pubmed", figures_dir=None,
                   figures_rel_prefix: str = "figures", compound_result=None) -> Registry:
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
    if figures_dir is not None and compound_result is not None:
        tools += figures_mod.make_figure_tools(
            figures_dir=figures_dir, rel_prefix=figures_rel_prefix,
            data_ctx=data_ctx, compound_result=compound_result, cache=cache,
        )
    return Registry(tools={t.name: t for t in tools})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_registry.py -q`
Expected: PASS (all registry tests).

- [ ] **Step 5: Wire `run_one` in `biomarker_agent/cli.py`**

Replace the body of `run_one` (the part that builds the registry) so it computes the figures dir and passes the compound result. The full updated function:

```python
def run_one(compound_dir, out_dir, feature_file, response_file, treatment_info, cache_dir,
            client, model=DEFAULT_MODEL, literature_backend="pubmed", max_tool_calls=40):
    """Analyze a single compound dir and write its report + trace. Returns paths."""
    result = load_compound(compound_dir)
    data_ctx = DataContext(feature_file, response_file)
    registry = build_registry(
        data_ctx=data_ctx, treatment_info=treatment_info, cache_dir=cache_dir,
        literature_backend=literature_backend,
        figures_dir=Path(out_dir) / "figures", figures_rel_prefix="figures",
        compound_result=result,
    )
    drug_info = registry.dispatch("drug_context", {"compound_id": result.compound_id})
    internal = context.precompute_internal(result, data_ctx)
    seed = context.build_seed_context(result, drug_info=drug_info, internal=internal)
    payload, transcript = run_agent(
        client=client, registry=registry, system_prompt=SYSTEM_PROMPT,
        seed_context=seed, model=model, max_tool_calls=max_tool_calls,
    )
    paths = report.write_report(payload, Path(out_dir), result.compound_id)
    trace_path = Path(out_dir) / "trace.json"
    trace_path.write_text(json.dumps(
        {"compound_id": result.compound_id, "model": model, "seed_context": seed,
         "transcript": transcript}, indent=2))
    paths["trace"] = trace_path
    return paths
```

- [ ] **Step 6: Run the existing e2e test (back-compat)**

Run: `uv run pytest tests/test_cli_e2e.py -q`
Expected: PASS — the FakeClient submits immediately, so no figure tools are called, but the registry now also builds them (exercises the wiring without LLM).

- [ ] **Step 7: Run full suite**

Run: `uv run pytest -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add biomarker_agent/tools/__init__.py biomarker_agent/cli.py tests/test_registry.py
git commit -m "Wire figure tools into registry and CLI"
```

---

## Task 6: Report integration (schema + render + prompt)

**Files:**
- Modify: `biomarker_agent/prompts.py`
- Modify: `biomarker_agent/report.py`
- Test: modify `tests/test_report.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_report.py` (and update the `SAMPLE` dict's first hypothesis to include figures):

```python
def test_render_embeds_figures(tmp_path):
    from biomarker_agent import report
    payload = {
        "summary": "s",
        "hypotheses": [{
            "rank": 1, "title": "H", "features": ["CRISPR_SMARCD1"], "mechanism": "m",
            "novelty": "off-MOA", "confidence": 0.6,
            "evidence": {"internal": "r=0.3"},
            "figures": [{"path": "figures/feature_response__CRISPR_SMARCD1.png",
                         "caption": "SMARCD1 vs response"}],
        }],
    }
    out = report.write_report(payload, tmp_path / "interp", compound_id="BRD:TEST-1")
    md = out["markdown"].read_text()
    assert "![SMARCD1 vs response](figures/feature_response__CRISPR_SMARCD1.png)" in md


def test_report_schema_has_figures():
    from biomarker_agent import prompts
    hyp = prompts.REPORT_TOOL["input_schema"]["properties"]["hypotheses"]["items"]
    assert "figures" in hyp["properties"]
    assert "figures" not in hyp["required"]  # optional
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report.py -q`
Expected: FAIL — figures not embedded / schema missing `figures`.

- [ ] **Step 3: Add `figures` to the hypothesis schema in `biomarker_agent/prompts.py`**

In `REPORT_TOOL["input_schema"]["properties"]["hypotheses"]["items"]["properties"]`, add (alongside `evidence`):

```python
                        "figures": {
                            "type": "array",
                            "description": "Figures supporting this hypothesis. ONLY use paths "
                                           "returned by a plot_* tool; never invent a path.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "caption": {"type": "string"},
                                },
                                "required": ["path"],
                            },
                        },
```

(Do not add `figures` to the hypothesis `required` list.)

- [ ] **Step 4: Add a figure instruction to `SYSTEM_PROMPT` in `biomarker_agent/prompts.py`**

Append this sentence to the end of the method bullet list in `SYSTEM_PROMPT` (before the final "When finished" paragraph):

```
4b. After you have supporting evidence for a hypothesis, generate 1-2 figures with the \
plot_* tools (e.g. plot_feature_response for the key association, plot_dependency_distribution \
for a CRISPR dependency, plot_string_network/plot_pathway_membership for the gene set) and \
attach the returned paths to that hypothesis's `figures`. Only attach paths returned by a \
plot tool; never invent one.
```

- [ ] **Step 5: Embed figures in `biomarker_agent/report.py`**

In `render_markdown`, after the evidence block (after the `for k, v in ev.items()` loop and before the trailing `lines.append("")`), insert:

```python
        for fig in (h.get("figures") or []):
            fpath = fig.get("path")
            if fpath:
                cap = fig.get("caption", "")
                lines.append("")
                lines.append(f"![{cap}]({fpath})")
                lines.append(f"*Figure: {cap}*")
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_report.py -q`
Expected: PASS.

- [ ] **Step 7: Run full suite**

Run: `uv run pytest -q`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add biomarker_agent/prompts.py biomarker_agent/report.py tests/test_report.py
git commit -m "Embed agent-generated figures in the report"
```

---

## Task 7: Live smoke + docs

**Files:**
- Modify: `docs/biomarker_agent.md`
- Modify: `pyproject.toml` (add `biomarker_agent.tools` already listed; no change unless missing)

- [ ] **Step 1: Live smoke (manual; needs a key)**

If `OPENROUTER_API_KEY` (or `ANTHROPIC_API_KEY`) is set, run a single compound and confirm figures are produced and embedded:

```bash
uv run biomarker-analyze data/small_test_sample/BRD_BRD-K25244359-066-03-4 \
    --provider openrouter --out /tmp/bioagent_fig_smoke --max-tool-calls 24
```

Expected: `/tmp/bioagent_fig_smoke/BRD_BRD-K25244359-066-03-4/figures/` contains
several PNGs; `report.md` contains `![...](figures/....png)` lines under hypotheses;
opening the report renders the figures. Spot-check one PNG looks clean
(despined, labeled, annotated). If a plot tool errored in the trace, fix that
wrapper and re-run.

- [ ] **Step 2: Update `docs/biomarker_agent.md`**

Under the "Tools the agent can call" table, add a row group for plotting tools:

```markdown
### Figure tools

The agent can also generate publication-quality figures, saved to
`<out>/figures/` and embedded inline in `report.md` under the hypothesis they
support:

| Tool | Shows |
|------|-------|
| `plot_feature_response` | feature vs response scatter + regression |
| `plot_feature_panel` | correlation heatmap among features + response |
| `plot_dependency_distribution` | CRISPR gene-effect histogram + threshold |
| `plot_codependency_bar` | top CRISPR co-dependencies |
| `plot_passing_importance` | passing features' real vs null importance |
| `plot_string_network` | STRING interactions among genes |
| `plot_mutation_frequency` | cBioPortal mutations per gene |
| `plot_pathway_membership` | Reactome pathway convergence |
```

Also update the "## Output" paragraph to mention the `figures/` directory.

- [ ] **Step 3: Commit**

```bash
git add docs/biomarker_agent.md
git commit -m "Document agent figure tools"
```

---

## Final verification

- [ ] `uv run pytest -q` → all green.
- [ ] `uvx ruff check biomarker_agent --select F401,F811,F841` → clean.
- [ ] Confirm no live network in figure unit tests: external-output figure tests monkeypatch the underlying tool/client.
- [ ] Confirm `.biomarker_agent_cache/` and any `/tmp` smoke outputs are not staged.
```
