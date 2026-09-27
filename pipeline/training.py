"""
Training stage, with MLflow tracking.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_CONFIGS,
    RANDOM_STATE,
)
from pipeline.preprocessing import build_preprocessor, prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "logreg": LogisticRegression,
    "rf": RandomForestClassifier,
    "hgb": HistGradientBoostingClassifier,
}


def setup_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> str:
    """Point MLflow at the tracking store and make sure the experiment exists."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow tracking at %s, experiment '%s'", tracking_uri, experiment_name)
    return tracking_uri


def build_model(model_type: str, **params: Any) -> Any:
    """Instantiate an estimator by name."""
    cls = MODEL_CLASSES.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available: {sorted(MODEL_CLASSES)}"
        )
    merged = {**MODEL_CONFIGS.get(model_type, {}), **params}
    if "random_state" not in merged:
        merged["random_state"] = RANDOM_STATE
    if model_type == "rf":
        merged.setdefault("n_jobs", -1)
    return cls(**merged)


def build_pipeline(model_type: str, feature_columns: list, **params: Any) -> Pipeline:
    """Preprocessor + estimator as a single fitted-together object."""
    return Pipeline([
        ("preprocess", build_preprocessor(feature_columns)),
        ("classifier", build_model(model_type, **params)),
    ])


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "hgb",
    run_name: Optional[str] = None,
    data_stats: Optional[Dict[str, Any]] = None,
    validation_report: Optional[Dict[str, Any]] = None,
    **params: Any,
) -> Tuple[Pipeline, str]:
    """Fit a pipeline inside an MLflow run and return it with the run id."""
    # Step 1: Derive features and get column list
    X_train_prepared = prepare_features(X_train)
    feature_columns = list(X_train_prepared.columns)

    # Step 2: Open MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Step 3: Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("n_train_rows", len(X_train))
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Step 4: Log data stats and validation report
        if data_stats:
            for key, value in data_stats.items():
                if isinstance(value, (int, float)):
                    mlflow.log_param(f"data_{key}", value)

        if validation_report:
            mlflow.log_dict(validation_report, "validation_report.json")
            mlflow.set_tag(
                "validation_passed", str(validation_report.get("passed", False))
            )

        # Step 5: Log feature columns
        mlflow.log_dict({"features": feature_columns}, "feature_columns.json")

        # Step 6: Build and fit the pipeline
        pipeline = build_pipeline(model_type, feature_columns, **params)
        pipeline.fit(X_train_prepared, y_train)

        # Step 7: Log the model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_train_prepared.head(3).astype("float64"),
        )

        logger.info(
            "Trained %s model (run_id=%s, n_features=%d, n_rows=%d)",
            model_type, run_id, len(feature_columns), len(X_train),
        )

    # Step 8: Return pipeline and run_id
    return pipeline, run_id
