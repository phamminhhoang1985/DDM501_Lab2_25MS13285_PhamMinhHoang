"""
Model registry stage.

MLflow deprecated registry STAGES in 2.9 and will remove them. This module uses
ALIASES instead — a named pointer to one version, repointed atomically.

    champion    what the serving layer loads
    challenger  a candidate that passed the gate and is waiting for a decision
"""

import logging
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from pipeline.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    MAX_FAIRNESS_GAP,
    MIN_PR_AUC,
    MIN_ROC_AUC,
    MLFLOW_EXPERIMENT_NAME,
    PRIMARY_METRIC,
    REGISTERED_MODEL_NAME,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_best_run(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = PRIMARY_METRIC,
    ascending: bool = False,
) -> Dict[str, Any]:
    """Best run in an experiment by a single metric."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")

    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} > 0",
        order_by=[f"metrics.{metric} {order}"],
        max_results=1,
    )
    if not runs:
        # Try without filter in case metric value is exactly 0
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
    if not runs:
        raise ValueError(
            f"No runs found in experiment '{experiment_name}' with metric '{metric}'."
        )

    best = runs[0]
    return {
        "run_id": best.info.run_id,
        "metrics": dict(best.data.metrics),
        "params": dict(best.data.params),
        "artifact_uri": best.info.artifact_uri,
    }


def register_model(
    run_id: str, model_name: str = REGISTERED_MODEL_NAME, artifact_path: str = "model"
) -> str:
    """Register a run's model artifact and return the new version number."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info("Registered model '%s' version %s from run %s", model_name, mv.version, run_id)
    return str(mv.version)


def set_alias(model_name: str, version: str, alias: str) -> None:
    """Point an alias at a version."""
    client = MlflowClient()
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)
    logger.info("Set alias '@%s' -> %s v%s", alias, model_name, version)


def get_model_version_by_alias(
    model_name: str = REGISTERED_MODEL_NAME, alias: str = CHAMPION_ALIAS
) -> Optional[Dict[str, Any]]:
    """Which version does an alias currently point at? None if it is unset."""
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except Exception:  # noqa: BLE001 - alias or model may simply not exist yet
        return None
    return {
        "name": mv.name,
        "version": mv.version,
        "run_id": mv.run_id,
        "aliases": list(mv.aliases),
        "model_uri": f"models:/{model_name}@{alias}",
    }


def passes_quality_gate(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Does this model clear the promotion bar?"""
    # Use 0.0 as default for metrics that must be large (so missing -> fail)
    # Use 1.0 as default for fairness_gap which must be small (so missing -> fail)
    roc_auc = metrics.get("roc_auc", 0.0)
    pr_auc = metrics.get("pr_auc", 0.0)
    fairness_gap_val = metrics.get("fairness_gap", 1.0)  # missing means worst possible

    checks = {
        "roc_auc": {
            "passed": roc_auc >= MIN_ROC_AUC,
            "rule": f"roc_auc >= {MIN_ROC_AUC} (got {roc_auc:.4f})",
        },
        "pr_auc": {
            "passed": pr_auc >= MIN_PR_AUC,
            "rule": f"pr_auc >= {MIN_PR_AUC} (got {pr_auc:.4f})",
        },
        "fairness_gap": {
            "passed": fairness_gap_val <= MAX_FAIRNESS_GAP,
            "rule": f"fairness_gap <= {MAX_FAIRNESS_GAP} (got {fairness_gap_val:.4f})",
        },
    }

    failed_checks = [name for name, check in checks.items() if not check["passed"]]
    passed = len(failed_checks) == 0

    return {
        "passed": passed,
        "failed_checks": failed_checks,
        "detail": checks,
    }


def beats_champion(
    candidate_metrics: Dict[str, float],
    model_name: str = REGISTERED_MODEL_NAME,
    metric: str = PRIMARY_METRIC,
    margin: float = 0.002,
) -> bool:
    """Is the candidate better than what is already live?

    The margin exists so that noise does not trigger a deployment. Shipping a
    model that is 0.0003 AUC better is all risk and no reward.
    """
    current = get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    if current is None:
        logger.info("No champion yet — candidate wins by default")
        return True
    client = MlflowClient()
    run = client.get_run(current["run_id"])
    champion_score = run.data.metrics.get(metric, 0.0)
    candidate_score = candidate_metrics.get(metric, 0.0)
    logger.info("Champion %s=%.4f, candidate %s=%.4f", metric, champion_score, metric, candidate_score)
    return candidate_score >= champion_score + margin


def promote_model(
    run_id: str,
    metrics: Dict[str, float],
    model_name: str = REGISTERED_MODEL_NAME,
) -> Dict[str, Any]:
    """Register a run, then decide what alias it deserves."""
    # Check quality gate FIRST
    gate_result = passes_quality_gate(metrics)

    # Always register the model — rejected models get a version for audit trail
    version = register_model(run_id, model_name)

    client = MlflowClient()

    if gate_result["passed"]:
        if beats_champion(metrics, model_name):
            outcome = "champion"
            set_alias(model_name, version, CHAMPION_ALIAS)
            logger.info("Model v%s promoted to @champion", version)
        else:
            outcome = "challenger"
            set_alias(model_name, version, CHALLENGER_ALIAS)
            logger.info("Model v%s set as @challenger (did not beat champion margin)", version)
    else:
        outcome = "rejected"
        logger.info(
            "Model v%s rejected by quality gate: %s",
            version, gate_result["failed_checks"]
        )

    # Tag the version with the quality gate outcome
    client.set_model_version_tag(model_name, version, "quality_gate", outcome)

    return {
        "run_id": run_id,
        "model_name": model_name,
        "version": version,
        "outcome": outcome,
        "quality_gate": gate_result,
        "metrics": metrics,
    }


# =============================================================================
# Helpers (PROVIDED)
# =============================================================================
def list_registered_models() -> List[Dict[str, Any]]:
    """Every registered model with its versions and aliases."""
    client = MlflowClient()
    out = []
    for model in client.search_registered_models():
        versions = client.search_model_versions(f"name='{model.name}'")
        out.append({
            "name": model.name,
            "aliases": dict(model.aliases or {}),
            "versions": [{"version": v.version, "run_id": v.run_id, "aliases": list(v.aliases)}
                         for v in versions],
        })
    return out


def compare_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    metric: str = PRIMARY_METRIC,
    top_n: int = 10,
    ascending: bool = False,
) -> List[Dict[str, Any]]:
    """Top N runs, for the comparison table in your report."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {order}"],
        max_results=top_n,
    )
    return [
        {"run_id": r.info.run_id, "run_name": r.data.tags.get("mlflow.runName", ""),
         "metrics": dict(r.data.metrics), "params": dict(r.data.params)}
        for r in runs
    ]
