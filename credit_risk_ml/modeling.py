from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk_ml.data_contract import DatasetSchema, feature_frame, target_series


MODEL_VERSION = "v1"
DEFAULT_DECISION_THRESHOLD = 0.30


@dataclass(frozen=True)
class TrainingArtifact:
    model_version: str
    pipeline: Pipeline
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    target_column: str
    decision_threshold: float


@dataclass(frozen=True)
class CalibratedRiskArtifact:
    model_version: str
    training_artifact: TrainingArtifact
    calibrator: Any
    calibration_method: str
    decision_threshold: float

    @property
    def feature_columns(self) -> list[str]:
        return self.training_artifact.feature_columns

    def predict_bad_probability(self, frame: pd.DataFrame) -> pd.Series:
        base_probabilities = self.training_artifact.pipeline.predict_proba(frame[self.feature_columns])[:, 1]
        if hasattr(self.calibrator, "predict_proba"):
            calibrated_probabilities = self.calibrator.predict_proba(base_probabilities.reshape(-1, 1))[:, 1]
        else:
            calibrated_probabilities = self.calibrator.transform(base_probabilities)
        return pd.Series(calibrated_probabilities, index=frame.index, name="prob_bad")


def build_pipeline(categorical_columns: list[str], numeric_columns: list[str]) -> Pipeline:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_columns),
            ("numeric", numeric_transformer, numeric_columns),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    random_state=42,
                    solver="liblinear",
                    max_iter=1000,
                    C=0.01,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def infer_column_types(frame: pd.DataFrame, schema: DatasetSchema) -> tuple[list[str], list[str]]:
    categorical_columns: list[str] = []
    numeric_columns: list[str] = []
    for field in schema.fields:
        if field.name == schema.target_column:
            continue
        if field.field_type == "categorical":
            categorical_columns.append(field.name)
        elif field.field_type == "integer":
            numeric_columns.append(field.name)
    return categorical_columns, numeric_columns


def fit_artifact(frame: pd.DataFrame, schema: DatasetSchema) -> tuple[TrainingArtifact, dict[str, Any]]:
    categorical_columns, numeric_columns = infer_column_types(frame, schema)
    pipeline = build_pipeline(categorical_columns, numeric_columns)

    X = feature_frame(frame, schema)
    y = target_series(frame, schema)
    y_binary = (y == 2).astype(int)

    pipeline.fit(X, y_binary)
    probabilities = pipeline.predict_proba(X)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_binary, probabilities)),
        "approval_rate": float((probabilities < DEFAULT_DECISION_THRESHOLD).mean()),
    }

    artifact = TrainingArtifact(
        model_version=MODEL_VERSION,
        pipeline=pipeline,
        feature_columns=list(X.columns),
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        target_column=schema.target_column,
        decision_threshold=DEFAULT_DECISION_THRESHOLD,
    )
    return artifact, metrics


def save_artifact(artifact: TrainingArtifact, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)


def load_artifact(artifact_path: Path) -> TrainingArtifact:
    with artifact_path.open("rb") as handle:
        return pickle.load(handle)


def predict_bad_probability(artifact: TrainingArtifact, frame: pd.DataFrame) -> pd.Series:
    probabilities = artifact.pipeline.predict_proba(frame[artifact.feature_columns])[:, 1]
    return pd.Series(probabilities, index=frame.index, name="prob_bad")


def predict_bad_probability_from_model(model: Any, frame: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_bad_probability"):
        return model.predict_bad_probability(frame)
    if hasattr(model, "pipeline") and hasattr(model, "feature_columns"):
        probabilities = model.pipeline.predict_proba(frame[model.feature_columns])[:, 1]
        return pd.Series(probabilities, index=frame.index, name="prob_bad")
    raise TypeError(f"Unsupported model object: {type(model)!r}")


def evaluate_predictions(artifact: TrainingArtifact, frame: pd.DataFrame, schema: DatasetSchema) -> dict[str, Any]:
    y = target_series(frame, schema)
    y_binary = (y == 2).astype(int)
    probabilities = predict_bad_probability_from_model(artifact, frame)
    approvals = (probabilities < artifact.decision_threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_binary, probabilities)),
        "approval_rate": float(approvals.mean()),
        "decision_threshold": artifact.decision_threshold,
    }