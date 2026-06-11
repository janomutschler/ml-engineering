"""Two-stage model production: walk-forward backtest, then fit the final artifact.

This is the ML core of the training asset, deliberately free of MLflow and the
Dagster execution context so it can be unit-tested directly. The asset wraps it
to log params, metrics, and the model to MLflow.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator

from bike_rental.defs.preprocessing.assembly import to_model_matrix
from bike_rental.defs.resources.training_config import TrainingConfigResource
from bike_rental.defs.training.backtest import walk_forward_backtest
from bike_rental.defs.training.metrics import aggregate_fold_metrics


@dataclass(frozen=True)
class TrainingResult:
    """The evidence and the artifact produced by one training run.

    Attributes
    ----------
    fold_metrics : pd.DataFrame
        Per-fold walk-forward backtest metrics (the generalization evidence).
    aggregates : dict[str, float]
        ``mean_*`` / ``std_*`` summaries of ``fold_metrics``.
    model : BaseEstimator
        The final estimator, fit on all available data — the deployable artifact.
    X_all : pd.DataFrame
        The full feature matrix the final model was fit on, retained so the
        caller can infer a model signature and an input example from it.

    """

    fold_metrics: pd.DataFrame
    aggregates: dict[str, float]
    model: BaseEstimator
    X_all: pd.DataFrame


def backtest_and_fit(
    modeling_feature_set: pd.DataFrame,
    config: TrainingConfigResource,
    metric_names: Sequence[str],
) -> TrainingResult:
    """Evaluate the configured recipe, then fit the final model on all data.

    Stage 1 runs an expanding-window walk-forward backtest and aggregates the
    per-fold metrics — the honest estimate of how this configuration
    generalizes over time. Stage 2 fits the same configuration on the entire
    history to produce the artifact that gets registered and deployed; its
    performance estimate is the backtest, not a held-out slice.

    Parameters
    ----------
    modeling_feature_set : pd.DataFrame
        Modeling-ready dataset containing the feature, target, and time columns.
    config : TrainingConfigResource
        Training configuration (model selection, feature/target/time columns,
        and the number of backtest folds).
    metric_names : Sequence[str]
        Metric columns to aggregate from the backtest.

    Returns
    -------
    TrainingResult
        The per-fold metrics, their aggregates, the fitted final model, and the
        feature matrix it was fit on.

    """
    fold_metrics = walk_forward_backtest(
        modeling_feature_set,
        make_model=config.build_model,
        feature_columns=config.feature_columns,
        target_column=config.target_column,
        time_column=config.time_column,
        n_splits=config.n_splits,
    )
    aggregates = aggregate_fold_metrics(fold_metrics, metric_names)

    ordered = modeling_feature_set.sort_values(config.time_column)
    X_all = to_model_matrix(ordered, config.feature_columns)
    y_all = ordered[config.target_column]

    model = config.build_model()
    model.fit(X_all, y_all)

    return TrainingResult(
        fold_metrics=fold_metrics,
        aggregates=aggregates,
        model=model,
        X_all=X_all,
    )
