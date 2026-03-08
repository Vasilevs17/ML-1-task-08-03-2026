from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, root_mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.features.preprocessing import make_preprocessor


@dataclass
class TrainResult:
    model: Pipeline
    metrics: dict[str, float]


def infer_problem_type(y) -> str:
    unique = y.dropna().nunique()
    if y.dtype.kind in {"i", "u", "b"} and unique <= 20:
        return "classification"
    if y.dtype.kind == "f" and unique <= 10:
        return "classification"
    return "regression"


def build_estimator(problem_type: str, random_state: int = 42):
    if problem_type == "classification":
        return RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    if problem_type == "regression":
        return RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    raise ValueError(f"Неизвестный problem_type: {problem_type}")


def train_baseline(X, y, test_size: float = 0.2, random_state: int = 42) -> TrainResult:
    problem_type = infer_problem_type(y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if problem_type == "classification" and y.nunique() > 1 else None,
    )

    preprocessor = make_preprocessor(X)
    estimator = build_estimator(problem_type, random_state=random_state)
    model = Pipeline([("prep", preprocessor), ("model", estimator)])
    model.fit(X_train, y_train)

    pred = model.predict(X_valid)
    metrics: dict[str, float]

    if problem_type == "classification":
        metrics = {"accuracy": float(accuracy_score(y_valid, pred))}
        if y.nunique() == 2:
            proba = model.predict_proba(X_valid)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_valid, proba))
    else:
        rmse = float(root_mean_squared_error(y_valid, pred))
        mae = float(mean_absolute_error(y_valid, pred))
        metrics = {"rmse": rmse, "mae": mae}

    metrics["n_train"] = float(len(X_train))
    metrics["n_valid"] = float(len(X_valid))
    metrics["target_mean"] = float(np.mean(y_train))

    return TrainResult(model=model, metrics=metrics)
