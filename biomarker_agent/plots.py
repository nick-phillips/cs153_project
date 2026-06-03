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
