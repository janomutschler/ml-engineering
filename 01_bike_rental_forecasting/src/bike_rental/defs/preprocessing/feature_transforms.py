"""Reusable feature transformation utilities for forecasting feature engineering workflows."""

import numpy as np
import pandas as pd


def one_hot_encode_column(
    df: pd.DataFrame,
    column: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """One-hot encode a categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Name of the categorical column to encode.
    prefix : str | None, default=None
        Prefix for generated columns. If None, the column name is used.

    Returns
    -------
    pd.DataFrame
        Dataset with one-hot encoded features.

    """
    df = df.copy()

    encoded = pd.get_dummies(
        df[column],
        prefix=prefix or column,
        dtype=int,
    )

    return pd.concat(
        [df, encoded],
        axis=1,
    )


def add_cyclical_features(
    df: pd.DataFrame,
    column: str,
    period: int,
) -> pd.DataFrame:
    """Add sine and cosine encodings for a cyclical feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    column : str
        Name of the cyclical feature column.
    period : int
        Number of unique steps in one full cycle.

    Returns
    -------
    pd.DataFrame
        Dataset with additional sine and cosine encoded features.

    """
    df = df.copy()

    df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / period)
    df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / period)

    return df


def add_historical_demand_features(
    df: pd.DataFrame,
    target_column: str,
    hour_column: str,
    weekday_column: str,
) -> pd.DataFrame:
    """Add lag and context-aware historical demand features.

    Parameters
    ----------
    df : pd.DataFrame
        Chronologically sorted input dataset.
    target_column : str
        Name of the demand target column.
    hour_column : str
        Name of the hour feature column.
    weekday_column : str
        Name of the weekday feature column.

    Returns
    -------
    pd.DataFrame
        Dataset enriched with historical demand features.

    Notes
    -----
        The lag and historical aggregation features are currently generated using
        row-based shifts and rolling windows. This approach assumes a continuous
        hourly sequence and does not explicitly account for missing timestamps.

        For the current dataset the impact is minimal, but a production-grade
        implementation should use timestamp-aware lag generation to ensure strict
        temporal alignment when gaps are present.

    """
    df = df.copy()

    df["lag_24h"] = df[target_column].shift(24)
    df["lag_168h"] = df[target_column].shift(168)

    df["same_hour_mean_7d"] = df.groupby(hour_column)[target_column].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
    )

    df["same_weekday_hour_mean_4w"] = df.groupby([weekday_column, hour_column])[
        target_column
    ].transform(lambda s: s.shift(1).rolling(window=4, min_periods=1).mean())

    return df
