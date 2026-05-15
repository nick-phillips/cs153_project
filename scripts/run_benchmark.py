#!/usr/bin/env python
"""Benchmark baseline models on a single drug response.

Runs 5-fold outer CV for:
  Baselines: ElasticNet, RF, XGBoost, CatBoost

Outputs:
  predictions.csv              — pred vs actual for all models
  comparison_stats.csv         — summary metrics
  pred_vs_actual_baselines.png — scatter plots (baselines)
  shap_summary_*.png/csv       — SHAP plots + ranked feature importance
"""

import argparse
import logging
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

from biomarker_discovery.baseline_trainers import (
    ElasticNetModel, RFModel, XGBoostModel, CatBoostModel,
)

warnings.filterwarnings("ignore")

log = logging.getLogger("biomarker_discovery")


def setup_logging(output_dir):
    """Configure logging to both stderr and a log file in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # stderr
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # file
    fh = logging.FileHandler(os.path.join(output_dir, "benchmark.log"),
                             mode="w")
    fh.setFormatter(fmt)
    root.addHandler(fh)

BASELINES = {
    "ElasticNet": ElasticNetModel,
    "RF": RFModel,
    "XGBoost": XGBoostModel,
    "CatBoost": CatBoostModel,
}


# ---- Evaluation helpers ----

def compute_stats(pred, true):
    if len(true) < 10:
        return None
    r = pearsonr(pred, true)
    rho = spearmanr(pred, true)
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return {
        "n": len(true),
        "pearson_r": r.statistic,
        "pearson_p": r.pvalue,
        "spearman_rho": rho.statistic,
        "rmse": rmse,
    }


def report(label, pred, true):
    if len(true) < 10:
        return
    r = pearsonr(pred, true)
    log.info("  %-30s  r=%.4f  p=%.2e  n=%d",
             label, r.statistic, r.pvalue, len(true))


# ---- Plotting ----

def scatter_with_stats(ax, true, pred, title, color="steelblue"):
    ax.scatter(true, pred, alpha=0.5, s=16, c=color, edgecolors="none")

    if len(true) >= 3:
        z = np.polyfit(true, pred, 1)
        x_line = np.linspace(true.min(), true.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), color="red", lw=1.5, alpha=0.8)
        r = pearsonr(pred, true)
        rho = spearmanr(pred, true)
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        txt = (f"n = {len(true)}\n"
               f"Pearson r = {r.statistic:.3f} (p={r.pvalue:.1e})\n"
               f"Spearman rho = {rho.statistic:.3f}\n"
               f"RMSE = {rmse:.3f}")
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=8,
                va="top", bbox=dict(boxstyle="round,pad=0.3",
                                    facecolor="white", alpha=0.8))
    ax.set_xlabel("True response")
    ax.set_ylabel("Predicted response")
    ax.set_title(title, fontsize=10)


def aggregate_shap(shap_dfs, xtest_dfs, n_samples):
    """Align and concatenate fold-level SHAP DataFrames."""
    feat_set = set()
    for df in shap_dfs:
        feat_set.update(df.columns)
    feat_list = sorted(feat_set)

    shap_concat = pd.DataFrame(0.0, index=range(n_samples), columns=feat_list)
    xtest_concat = pd.DataFrame(np.nan, index=range(n_samples),
                                columns=feat_list)
    row = 0
    for shap_df, xtest_df in zip(shap_dfs, xtest_dfs):
        n = len(shap_df)
        for col in shap_df.columns:
            shap_concat.iloc[row:row+n,
                             shap_concat.columns.get_loc(col)] = \
                shap_df[col].values
        common_cols = [c for c in shap_df.columns if c in xtest_df.columns]
        for col in common_cols:
            xtest_concat.iloc[row:row+n,
                              xtest_concat.columns.get_loc(col)] = \
                xtest_df[col].values
        row += n
    return shap_concat, xtest_concat


def make_shap_plot(shap_concat, xtest_concat, title, save_path,
                   out_dir, n_top=30, csv_tag=None):
    shap_sub = shap_concat
    xtest_sub = xtest_concat

    mean_abs = shap_sub.abs().mean(axis=0).sort_values(ascending=False)
    top = mean_abs.head(n_top).index.tolist()

    log.info("Top %d features by mean |SHAP| (%s):", n_top, title)
    for i, feat in enumerate(top[:10]):
        log.info("  %2d. %-40s mean|SHAP|=%.4f", i+1, feat, mean_abs[feat])

    if csv_tag:
        ranked_df = pd.DataFrame({
            "rank": range(1, len(mean_abs) + 1),
            "feature": mean_abs.index,
            "mean_abs_shap": mean_abs.values,
        })
        csv_path = os.path.join(out_dir, f"shap_ranked_{csv_tag}.csv")
        ranked_df.to_csv(csv_path, index=False)
        log.info("Ranked features saved to %s", csv_path)

    fig, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_sub[top].values, xtest_sub[top].values,
        feature_names=top, show=False, max_display=n_top,
    )
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("SHAP plot saved to %s", save_path)


# ---- Main benchmark ----

def run_benchmark(features, y, out_dir, drug_name, n_folds=5):
    os.makedirs(out_dir, exist_ok=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Accumulators — baselines
    bl_preds = {name: [] for name in BASELINES}
    bl_shap_dfs = {name: [] for name in BASELINES}
    bl_xtest_dfs = {name: [] for name in BASELINES}

    all_true = []
    all_sample_ids = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
        log.info("=" * 60)
        log.info("FOLD %d/%d  (train=%d, test=%d)",
                 fold, n_folds - 1, len(train_idx), len(test_idx))
        log.info("=" * 60)

        X_train = features.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = features.iloc[test_idx]
        y_test = y.iloc[test_idx]

        all_true.extend(y_test.values)
        all_sample_ids.extend(y_test.index.tolist())

        # ---- Baselines ----
        for name, ModelClass in BASELINES.items():
            log.info("[Fold %d] %s: starting...", fold, name)
            model = ModelClass()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            bl_preds[name].extend(preds)
            sv = model.shap_values(X_test)
            bl_shap_dfs[name].append(sv)
            bl_xtest_dfs[name].append(X_test)
            log.info("[Fold %d] %s: done.", fold, name)

    # Convert to arrays
    all_true = np.array(all_true)

    # ---- Predictions table ----
    pred_df = pd.DataFrame({
        "sample_id": all_sample_ids,
        "y_true": all_true,
    })
    for name in BASELINES:
        pred_df[name] = np.array(bl_preds[name])

    pred_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    log.info("Predictions saved to %s", pred_path)

    # ---- Results summary ----
    log.info("=" * 60)
    log.info("Drug: %s | %d total samples", drug_name, len(all_true))
    log.info("=" * 60)

    for name in BASELINES:
        log.info("--- %s ---", name)
        report("Full dataset", np.array(bl_preds[name]), all_true)

    # ---- Comparison stats table ----
    table_rows = []
    for name in BASELINES:
        s = compute_stats(np.array(bl_preds[name]), all_true)
        if s:
            table_rows.append({"model": name, "subset": "all", **s})

    comparison_df = pd.DataFrame(table_rows)
    table_path = os.path.join(out_dir, "comparison_stats.csv")
    comparison_df.to_csv(table_path, index=False)
    log.info("Comparison table saved to %s", table_path)
    log.info("\n%s", comparison_df.to_string(index=False))

    # ---- Pred vs actual: baselines ----
    n_bl = len(BASELINES)
    cols = min(n_bl, 2)
    rows = (n_bl + cols - 1) // cols
    colors = ["steelblue", "forestgreen", "darkorange", "mediumpurple"]
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows),
                             squeeze=False)
    for i, name in enumerate(BASELINES):
        r, c = divmod(i, cols)
        scatter_with_stats(axes[r][c], all_true, np.array(bl_preds[name]),
                           f"{name}: All samples",
                           color=colors[i % len(colors)])
    for j in range(i + 1, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)
    fig.suptitle(f"{drug_name} — Baselines ({n_folds}-fold CV)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "pred_vs_actual_baselines.png"), dpi=150)
    plt.close(fig)
    log.info("Baseline pred-vs-actual plot saved")

    # ---- SHAP: baselines ----
    n_total = len(all_true)
    for name in BASELINES:
        log.info("Aggregating SHAP for %s...", name)
        shap_agg, xtest_agg = aggregate_shap(
            bl_shap_dfs[name], bl_xtest_dfs[name], n_total
        )
        make_shap_plot(
            shap_agg, xtest_agg, f"{drug_name} — SHAP ({name})",
            os.path.join(out_dir, f"shap_summary_{name}.png"),
            out_dir, csv_tag=name,
        )

    log.info("All outputs saved to %s", out_dir)


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark drug response models."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--response_id", type=str,
                       help="Response column ID (e.g. BRD:BRD-K49049886-001-08-7)")
    group.add_argument("--drug_name", type=str,
                       help="Drug name to look up in treatment_info CSV")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing data files (convenience)")
    parser.add_argument("--feature_file", type=str, default=None)
    parser.add_argument("--response_file", type=str, default=None)
    parser.add_argument("--treatment_info", type=str, default=None,
                        help="Only needed when using --drug_name")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    # Resolve file paths: explicit args take precedence over data_dir
    feat_path = args.feature_file
    resp_path = args.response_file
    info_path = args.treatment_info
    if args.data_dir:
        feat_path = feat_path or os.path.join(args.data_dir, "x-all_v4.pkl")
        resp_path = resp_path or os.path.join(
            args.data_dir, "responses_primary_v4.pkl"
        )
        info_path = info_path or os.path.join(
            args.data_dir, "primary_screen_treatment_info.csv"
        )

    if not all([feat_path, resp_path]):
        sys.exit("Provide --data_dir or both --feature_file and "
                 "--response_file")

    # Determine label and response_id
    if args.response_id:
        response_id = args.response_id
        label = response_id
    else:
        if not info_path:
            sys.exit("--treatment_info (or --data_dir) required with "
                     "--drug_name")
        label = args.drug_name

    out_dir = args.output_dir or label
    setup_logging(out_dir)

    log.info("Loading features: %s", feat_path)
    features = pd.read_pickle(feat_path)
    log.info("Loading responses: %s", resp_path)
    responses = pd.read_pickle(resp_path)

    if args.drug_name:
        log.info("Loading treatment info: %s", info_path)
        treatment_info = pd.read_csv(info_path)
        row = treatment_info.loc[
            treatment_info["Drug.Name"] == args.drug_name
        ]
        if row.empty:
            sys.exit(f"Drug '{args.drug_name}' not found in treatment_info.")
        response_id = row.IDs.values[0]

    if response_id not in responses.columns:
        sys.exit(f"Response ID '{response_id}' not found in response file.")
    y = responses[response_id].dropna()

    common = np.intersect1d(y.index.values, features.index.values)
    y = y.loc[common]
    features = features.loc[common]

    log.info("Label: %s | Response ID: %s | Samples: %d | Features: %d",
             label, response_id, len(y), features.shape[1])

    run_benchmark(features, y, out_dir, label, n_folds=args.n_folds)


if __name__ == "__main__":
    main()
