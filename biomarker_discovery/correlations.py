"""Feature selection via column-wise Pearson correlation."""

import numpy as np
import pandas as pd
from scipy import stats


def colwise_corr_with_p(X, y):
    """Column-wise Pearson correlation between X (with NaNs) and y.

    Uses pairwise-complete observations per column.
    Returns (r, p) arrays of shape (n_features,).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    valid = ~np.isnan(X)
    W = valid.astype(float)
    n = W.sum(axis=0)

    X_filled = np.nan_to_num(X, nan=0.0)
    y2d = y[:, None]

    sum_x = (X_filled * W).sum(axis=0)
    sum_x2 = (X_filled * X_filled * W).sum(axis=0)
    sum_y = (y2d * W).sum(axis=0)
    sum_y2 = (y2d * y2d * W).sum(axis=0)
    sum_xy = (X_filled * y2d * W).sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = sum_x / n
        mean_y = sum_y / n

        cov_xy = (sum_xy - n * mean_x * mean_y) / (n - 1)
        var_x = (sum_x2 - n * mean_x * mean_x) / (n - 1)
        var_y = (sum_y2 - n * mean_y * mean_y) / (n - 1)

        r = cov_xy / np.sqrt(var_x * var_y)

        df = n - 2
        valid_df = (df > 0) & np.isfinite(r) & (var_x > 0) & (var_y > 0)

        t = np.full_like(r, np.nan)
        t[valid_df] = r[valid_df] * np.sqrt(df[valid_df] / (1.0 - r[valid_df] ** 2))

        p = np.full_like(r, np.nan)
        p[valid_df] = 2.0 * stats.t.sf(np.abs(t[valid_df]), df[valid_df])

    return r, p


def compute_baseline_correlations(X, y):
    """Compute per-feature Pearson correlation with response.

    Returns DataFrame with columns: feature, r, p — sorted by p ascending.
    """
    r, p = colwise_corr_with_p(X.values, y.values)
    return pd.DataFrame({"feature": X.columns, "r": r, "p": p}).sort_values(
        "p", ascending=True
    ).reset_index(drop=True)
