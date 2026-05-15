"""Utility functions for data loading and preprocessing."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def align_features_and_response(features: pd.DataFrame, y: pd.Series):
    """Align features and response by common index, dropping NaN responses."""
    y = y.dropna()
    common = np.intersect1d(y.index.values, features.index.values)
    return features.loc[common], y.loc[common]
