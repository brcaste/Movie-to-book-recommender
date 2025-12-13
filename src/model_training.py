import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import cross_val_predict
import joblib


def load_labeled_data(path: str ="data/processed/labeled_pairs.csv") -> pd.DataFrame:
    print(f"Loading labeled data from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} labeled pairs.")
    return df


def prepare_features_and_labels(df: pd.DataFrame):
    # Features
    X = df[['cosine_sim']].values

    # Target
    y = df['label'].values.astype(int)

    print("Features and labels prepared.")
    print(f" - X shape: {X.shape}")
    print(f" - Positive labels: {np.sum(y == 1)}")
    print(f" - Negative labels: {np.sum(y == 0)}")
    return X, y


def cross_validate_model(X, y, n_splits: int = 5):
    print("Running stratified k-fold cross-validation...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # baseline classifier (before hyper-paremeter tuning)
    base_model= LogisticRegression(
        class_weight = 'balanced',
        solver='lbfgs',
        max_iter=1000
    )
    # Cross-validate predictions for each sample
    y_pred = cross_val_predict(base_model, X, y, cv=skf)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print("Cross-Validation Metrics (Baseline Model):")
    print(f" - Accuracy: {acc:.3f}")
    print(f" - Precision: {prec:.3f}")
    print(f" - Recall: {rec:.3f}")
    print(f" - F1 score: {f1:.3f}")
    print("\nDetailed classification report:")
    print(classification_report(y, y_pred, zero_division=0))

