"""Calendar feature transformations for bike rental preprocessing."""

import pandas as pd


def create_calendar_features(
    hourly_rental_activity: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Create calendar-based time features.

    Parameters
    ----------
    hourly_rental_activity : pd.DataFrame
        Hourly rental activity containing a ``datetime_hour`` column.
    holidays : pd.DataFrame
        Holiday calendar containing a ``date`` column.

    Returns
    -------
    pd.DataFrame
        Time feature dataset containing calendar-derived features for each
        hour in the rental activity timeline.

    """
    time_features = hourly_rental_activity[["datetime_hour"]].copy()

    time_features["date"] = time_features["datetime_hour"].dt.normalize()

    time_features["hour"] = time_features["datetime_hour"].dt.hour

    time_features["weekday"] = time_features["datetime_hour"].dt.weekday

    time_features["month"] = time_features["datetime_hour"].dt.month

    time_features["is_weekend"] = time_features["weekday"] >= 5

    time_features["is_holiday"] = time_features["date"].isin(holidays["date"])

    return time_features.drop(columns="date")
