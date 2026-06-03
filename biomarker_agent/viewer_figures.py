"""Optional build-time figures for the viewer (needs pandas + matplotlib).

Kept separate from ``viewer_build`` so the core bundle build stays
dependency-light: the heavy imports happen lazily inside the function and any
failure (missing deps, missing data) degrades to "no figure" rather than
breaking the build. Run the build under the project venv (``uv run``) to enable
these figures.
"""

from pathlib import Path


def response_histogram(responses_file, compound_id: str, out_png) -> int | None:
    """Plot the ground-truth response (toxicity) distribution for one compound.

    ``responses_file`` is a pickled DataFrame of [cell lines x compounds]; the
    compound's values are ``responses[compound_id]``. Returns the sample count on
    success, or None if the deps/data are unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return None
    try:
        responses = pd.read_pickle(responses_file)
        if compound_id not in responses.columns:
            return None
        vals = responses[compound_id].dropna().astype(float)
    except Exception:
        return None
    if vals.empty:
        return None

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=150)
    ax.hist(vals, bins=30, color="#2563eb", edgecolor="white", linewidth=0.4)
    ax.axvline(float(vals.median()), color="#b91c1c", linestyle="--", linewidth=1.2,
               label=f"median {vals.median():.2f}")
    ax.set_xlabel("Ground-truth response (toxicity)")
    ax.set_ylabel("Cell lines")
    ax.set_title("Response distribution", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return int(vals.shape[0])
