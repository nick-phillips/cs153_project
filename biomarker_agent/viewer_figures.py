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


def pred_vs_actual_plot(csv_path, out_png) -> int | None:
    """Scatter of the refit model's predicted vs actual response (with identity
    line + Pearson r). ``csv_path`` has columns ``observed``/``predicted``.

    Returns the sample count on success, or None if the deps/data are missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return None
    from pathlib import Path as _Path

    csv_path = _Path(csv_path)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        obs = df["observed"].astype(float)
        pred = df["predicted"].astype(float)
    except Exception:
        return None
    if obs.empty:
        return None
    r = float(obs.corr(pred))

    out_png = _Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=150)
    ax.scatter(obs, pred, s=12, alpha=0.5, color="#2563eb", edgecolors="none")
    lo = float(min(obs.min(), pred.min()))
    hi = float(max(obs.max(), pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#b91c1c", linestyle="--", linewidth=1.0,
            label="y = x")
    ax.set_xlabel("Actual response")
    ax.set_ylabel("Predicted (refit)")
    ax.set_title(f"Predicted vs actual (r = {r:.2f})", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return int(obs.shape[0])
