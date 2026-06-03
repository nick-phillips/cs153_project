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
    shap_summaries: list = field(default_factory=list)


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


def _collect_shap_summaries(path: Path) -> list:
    """Existing pipeline SHAP-summary PNGs: selected-refit model + baselines."""
    out = []
    selected = path / "refract" / "significant" / "selected_shap_summary.png"
    if selected.exists():
        out.append({"label": "Selected-refit model (significant features)", "source": str(selected)})
    for model in BASELINE_MODELS:
        p = path / "baselines" / model / "shap_summary.png"
        if p.exists():
            out.append({"label": f"Baseline model: {model}", "source": str(p)})
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
        shap_summaries=_collect_shap_summaries(path),
    )
