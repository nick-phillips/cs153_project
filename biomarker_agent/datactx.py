"""Lazy access to the feature/response matrices and internal-stats primitives."""

from functools import cached_property
from pathlib import Path

import pandas as pd
from scipy import stats

# Response values are drug-response scores where LOWER = greater sensitivity
# (more cell killing) and HIGHER = resistance. A NEGATIVE feature-response
# correlation therefore means a higher feature value tracks greater sensitivity.
RESPONSE_CONVENTION = (
    "lower response = greater sensitivity (more cell killing); higher = resistance"
)


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

        # Translate the correlation sign into sensitivity terms using the fixed
        # response convention (lower response = more sensitive). A negative r
        # means higher feature value -> lower response -> greater sensitivity.
        higher_feature_implies = "greater sensitivity" if pr < 0 else "greater resistance"
        if diff_high == diff_high and diff_low == diff_low:
            more_sensitive_tertile = "high" if diff_high < diff_low else "low"
        else:
            more_sensitive_tertile = None

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
            "response_convention": RESPONSE_CONVENTION,
            "higher_feature_implies": higher_feature_implies,
            "more_sensitive_tertile": more_sensitive_tertile,
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
