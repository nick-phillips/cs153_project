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


def string_network(genes: list, interactions: list, path) -> str:
    """Circular-layout interaction graph among genes; edge width ~ score.

    Gene labels are placed radially outside their node so they never overflow the
    marker or clip at the figure edge.
    """
    n = len(genes)
    if n == 0:
        raise ValueError("no genes")
    # Single node sits at the center; otherwise spread around a circle. Start at
    # the top (pi/2) and go clockwise for a tidy layout.
    if n == 1:
        pos = {genes[0]: (0.0, 0.0)}
    else:
        angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
        pos = {g: (np.cos(a), np.sin(a)) for g, a in zip(genes, angles)}

    fig, ax = plt.subplots(figsize=(6, 6))
    for e in interactions:
        a, b, s = e.get("a"), e.get("b"), float(e.get("score", 0))
        if a in pos and b in pos:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            ax.plot([x1, x2], [y1, y2], color=PALETTE["secondary"],
                    linewidth=0.6 + 3 * s, alpha=0.55, zorder=1, solid_capstyle="round")
    xs = [pos[g][0] for g in genes]
    ys = [pos[g][1] for g in genes]
    ax.scatter(xs, ys, s=260, color=PALETTE["primary"], zorder=2,
               edgecolor="white", linewidth=1.5)
    for g in genes:
        x, y = pos[g]
        # push the label radially outward; align away from the marker
        r = (x ** 2 + y ** 2) ** 0.5 or 1.0
        lx, ly = x + 0.16 * x / r, y + 0.16 * y / r
        ha = "center" if abs(x) < 0.3 else ("left" if x > 0 else "right")
        va = "center" if abs(y) < 0.3 else ("bottom" if y > 0 else "top")
        ax.annotate(g, (lx, ly), ha=ha, va=va, fontsize=10,
                    color=PALETTE["neutral"], zorder=3, fontweight="bold")
    ax.set_title(f"STRING interactions ({len(interactions)} edges)")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
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
