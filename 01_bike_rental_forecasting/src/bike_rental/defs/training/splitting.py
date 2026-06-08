"""Chronological train/test splitting for the forecasting workflow."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TrainTestSplit:
    """A chronological train/test partition of a modeling feature set.

    Attributes
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices for the training and test periods.
    y_train, y_test : pd.Series
        Target vectors for the training and test periods.

    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def _validate_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    time_column: str,
) -> None:
    """Raise if any configured feature, target, or time column is absent."""
    required = [*feature_columns, target_column, time_column]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"modeling_feature_set is missing required columns: {missing}. ")


def chronological_split(
    modeling_feature_set: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    time_column: str,
    train_ratio: float,
) -> TrainTestSplit:
    """Split a modeling feature set chronologically into train and test sets.

    The split is fully deterministic: the input is sorted by ``time_column``
    and partitioned at ``train_ratio``. The same input and configuration always
    yield identical partitions, which allows independent assets to derive the
    same split without persisting it as a data artifact.

    Parameters
    ----------
    modeling_feature_set : pd.DataFrame
        Modeling-ready dataset containing the feature, target, and time columns.
    feature_columns : list[str]
        Columns used as model input features.
    target_column : str
        Name of the forecasting target column.
    time_column : str
        Name of the column defining chronological order.
    train_ratio : float
        Fraction of the (time-ordered) rows assigned to the training set.

    Returns
    -------
    TrainTestSplit
        The chronological train/test partition.

    Raises
    ------
    ValueError
        If any configured feature, target, or time column is missing.

    """
    _validate_columns(modeling_feature_set, feature_columns, target_column, time_column)

    df = modeling_feature_set.sort_values(time_column).reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)

    X = df[feature_columns]
    y = df[target_column]

    return TrainTestSplit(
        X_train=X.iloc[:split_idx],
        X_test=X.iloc[split_idx:],
        y_train=y.iloc[:split_idx],
        y_test=y.iloc[split_idx:],
    )
