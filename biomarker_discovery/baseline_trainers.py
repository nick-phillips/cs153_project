"""Baseline model classes for drug response prediction.

Each model: correlation-based feature selection → fit → predict → SHAP.
All single-threaded by default.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from biomarker_discovery.correlations import compute_baseline_correlations

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
log = logging.getLogger(__name__)


def _select_features(X, y, k):
    """Top-k features by Pearson correlation p-value."""
    corr_df = compute_baseline_correlations(X, y)
    return corr_df.head(k).feature.values


class ElasticNetModel:
    def __init__(self, top_k=500):
        self.top_k = top_k
        self.selected_features = None
        self.scaler = None
        self.model = None

    def fit(self, X, y):
        log.info("  ElasticNet: selecting %d features...", self.top_k)
        self.selected_features = _select_features(X, y, self.top_k)
        Xs = X[self.selected_features].fillna(0)
        self.scaler = StandardScaler()
        Xs_scaled = self.scaler.fit_transform(Xs)
        log.info("  ElasticNet: fitting model...")
        self.model = ElasticNetCV(
            l1_ratio=[0.5, 0.9, 1.0],
            alphas=np.logspace(-3, 2, 20),
            cv=5, max_iter=5000, random_state=42, n_jobs=1,
        )
        self.model.fit(Xs_scaled, y.values)
        log.info("  ElasticNet: fit complete (alpha=%.4f, l1_ratio=%.2f)",
                 self.model.alpha_, self.model.l1_ratio_)

    def predict(self, X):
        Xs = X[self.selected_features].fillna(0)
        return self.model.predict(self.scaler.transform(Xs))

    def shap_values(self, X):
        log.info("  ElasticNet: computing SHAP values...")
        Xs = X[self.selected_features].fillna(0)
        Xs_scaled = self.scaler.transform(Xs)
        explainer = shap.LinearExplainer(self.model, Xs_scaled)
        sv = explainer.shap_values(Xs_scaled)
        return pd.DataFrame(sv, index=X.index, columns=self.selected_features)


class RFModel:
    def __init__(self, top_k=500, n_estimators=200):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.selected_features = None
        self.model = None

    def fit(self, X, y):
        log.info("  RF: selecting %d features...", self.top_k)
        self.selected_features = _select_features(X, y, self.top_k)
        Xs = X[self.selected_features].fillna(0).values
        log.info("  RF: fitting %d trees...", self.n_estimators)
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, n_jobs=1, random_state=42,
        )
        self.model.fit(Xs, y.values)
        log.info("  RF: fit complete")

    def predict(self, X):
        return self.model.predict(X[self.selected_features].fillna(0).values)

    def shap_values(self, X):
        log.info("  RF: computing SHAP values...")
        Xs = X[self.selected_features].fillna(0).values
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(Xs)
        return pd.DataFrame(sv, index=X.index, columns=self.selected_features)


class XGBoostModel:
    def __init__(self, top_k=500, n_estimators=200, max_depth=6,
                 learning_rate=0.1):
        self.top_k = top_k
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.selected_features = None
        self.model = None

    def fit(self, X, y):
        log.info("  XGBoost: selecting %d features...", self.top_k)
        self.selected_features = _select_features(X, y, self.top_k)
        Xs = X[self.selected_features].fillna(0).values
        log.info("  XGBoost: fitting %d rounds (depth=%d, lr=%.2f)...",
                 self.n_estimators, self.max_depth, self.learning_rate)
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, n_jobs=1, random_state=42,
            tree_method="hist", verbosity=0,
        )
        self.model.fit(Xs, y.values)
        log.info("  XGBoost: fit complete")

    def predict(self, X):
        return self.model.predict(X[self.selected_features].fillna(0).values)

    def shap_values(self, X):
        log.info("  XGBoost: computing SHAP values...")
        Xs = X[self.selected_features].fillna(0).values
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(Xs)
        return pd.DataFrame(sv, index=X.index, columns=self.selected_features)


class CatBoostModel:
    def __init__(self, top_k=500, iterations=500, depth=6,
                 learning_rate=0.05):
        self.top_k = top_k
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.selected_features = None
        self.model = None

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        log.info("  CatBoost: selecting %d features...", self.top_k)
        self.selected_features = _select_features(X, y, self.top_k)
        Xs = X[self.selected_features].fillna(0).values
        log.info("  CatBoost: fitting %d iterations (depth=%d, lr=%.2f)...",
                 self.iterations, self.depth, self.learning_rate)
        self.model = CatBoostRegressor(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, random_seed=42,
            verbose=0, thread_count=1,
        )
        self.model.fit(Xs, y.values)
        log.info("  CatBoost: fit complete")

    def predict(self, X):
        return self.model.predict(X[self.selected_features].fillna(0).values)

    def shap_values(self, X):
        log.info("  CatBoost: computing SHAP values...")
        Xs = X[self.selected_features].fillna(0).values
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(Xs)
        return pd.DataFrame(sv, index=X.index, columns=self.selected_features)
