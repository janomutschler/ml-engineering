"""Walk-forward backtesting for time-series forecasting evaluation."""

from collections.abc import Callable

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit

from bike_rental.defs.training.metrics import regression_metrics


def walk_forward_backtest(
    df: pd.DataFrame,
    make_model: Callable[[], BaseEstimator],
    feature_columns: list[str],
    target_column: str,
    time_column: str,
    n_splits: int,
) -> pd.DataFrame:
    """Evaluate a model with expanding-window walk-forward cross-validation.

    The data is sorted chronologically and split into ``n_splits`` expanding
    train/test folds via ``TimeSeriesSplit``. A fresh model is fit on each
    fold's training window and scored on the subsequent test window, so model
    quality is reported as a distribution across time rather than a single
    number.

    Splitting the pre-computed modeling feature set by index does not leak
    future information because its historical-demand features are causal
    (backward-looking only); a row at time t depends only on data up to t.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling feature set containing the feature, target, and time columns.
    make_model : Callable[[], BaseEstimator]
        Factory returning a fresh, unfitted estimator for each fold.
    feature_columns : list[str]
        Model input feature columns.
    target_column : str
        Forecasting target column.
    time_column : str
        Column defining chronological order.
    n_splits : int
        Number of expanding-window folds.

    Returns
    -------
    pd.DataFrame
        One row per fold with columns ``fold``, ``train_rows``, ``test_rows``,
        and the metrics ``mae``, ``rmse``, ``rmsle``, ``r2``.

    """
    ordered = df.sort_values(time_column).reset_index(drop=True)
    X = ordered[feature_columns]
    y = ordered[target_column]

    splitter = TimeSeriesSplit(n_splits=n_splits)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[test_idx])

        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                **regression_metrics(y.iloc[test_idx], predictions),
            }
        )

    return pd.DataFrame(rows)
