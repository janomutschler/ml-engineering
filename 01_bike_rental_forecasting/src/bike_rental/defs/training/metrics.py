"""Regression evaluation metrics for the forecasting workflow."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)


def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """Compute regression metrics on the original target scale.

    Reports a log-scale error (RMSLE) alongside MAE, RMSE, and R². RMSLE
    penalizes relative rather than absolute error, which is informative for
    skewed demand data. It is undefined for negative inputs, so predictions are
    clipped at zero before it is computed; the demand target is itself
    non-negative.

    Parameters
    ----------
    y_true : ArrayLike
        Observed target values.
    y_pred : ArrayLike
        Predicted target values on the original scale.

    Returns
    -------
    dict[str, float]
        Metrics keyed by ``mae``, ``rmse``, ``rmsle``, and ``r2``.

    """
    y_pred_non_negative = np.clip(y_pred, 0, None)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "rmsle": float(root_mean_squared_log_error(y_true, y_pred_non_negative)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def aggregate_fold_metrics(
    fold_metrics: pd.DataFrame,
    metric_names: Sequence[str],
) -> dict[str, float]:
    """Summarize per-fold backtest metrics into run-level aggregates.

    Parameters
    ----------
    fold_metrics : pd.DataFrame
        One row per backtest fold, with a column per metric in ``metric_names``.
    metric_names : Sequence[str]
        Metric columns to summarize.

    Returns
    -------
    dict[str, float]
        For each metric, a ``mean_<metric>`` and ``std_<metric>`` entry. The
        mean is the headline performance estimate; the std exposes stability
        across time, which a single holdout cannot show.

    """
    aggregates = {f"mean_{metric}": float(fold_metrics[metric].mean()) for metric in metric_names}
    aggregates.update(
        {f"std_{metric}": float(fold_metrics[metric].std()) for metric in metric_names}
    )
    return aggregates
