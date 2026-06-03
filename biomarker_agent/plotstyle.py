"""Shared, publication-quality matplotlib styling for agent figures."""

import warnings
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
        # Embeds use markdown image syntax (no width control), so size is the
        # pixel size: a modest DPI keeps figures compact yet crisp on screen.
        "savefig.dpi": 110,
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
    # Figures with many long rotated tick labels can make constrained_layout give
    # up ("axes sizes collapsed to zero"); this is benign because savefig uses
    # bbox_inches="tight", which still captures every label. Silence the noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*constrained_layout.*")
        fig.savefig(path)
    plt.close(fig)
    return str(path)
