"""Build the model feature matrix for a day-ahead forecast request.

The lag and rolling features the model expects are backward-looking, so they
can be computed for the forecast horizon purely from published history. This
module reuses the *exact* pipeline transform functions, which guarantees the
features served match the features trained on (no training-serving skew).
"""

import numpy as np
import pandas as pd

from bike_rental.api.schemas import HourlyWeather
from bike_rental.defs.constants import TARGET_COLUMN
from bike_rental.defs.preprocessing.calendar_features import create_calendar_features
from bike_rental.defs.preprocessing.feature_transforms import (
    add_cyclical_features,
    add_historical_demand_features,
    one_hot_encode_column,
)

_WEATHER_FIELDS = (
    "conditions",
    "temperature_c",
    "perceived_temperature_c",
    "humidity",
    "windspeed_kmh",
)


def build_forecast_features(
    weather: list[HourlyWeather],
    history: pd.DataFrame,
    holidays: pd.DataFrame,
    feature_columns: list[str],
    horizon_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Build the feature matrix for the horizon following the last history point.

    Parameters
    ----------
    weather : list[HourlyWeather]
        Ordered hourly weather for the forecast horizon (entry ``i`` is the
        ``i``-th hour after the last observed data point).
    history : pd.DataFrame
        Published ``bike_rental_features`` (pre-modeling) with observed demand.
    holidays : pd.DataFrame
        Published holiday calendar (``date`` column).
    feature_columns : list[str]
        The exact model input columns (from the training configuration).
    horizon_hours : int, default=24
        Number of hours to forecast.

    Returns
    -------
    tuple[pd.DataFrame, pd.DatetimeIndex]
        The feature matrix (one row per horizon hour, columns == feature_columns)
        and the datetime index of the forecast hours.

    """
    history = history.sort_values("datetime_hour").reset_index(drop=True)
    start = history["datetime_hour"].max() + pd.Timedelta(hours=1)
    horizon_index = pd.date_range(start=start, periods=horizon_hours, freq="h")

    horizon = pd.DataFrame({"datetime_hour": horizon_index})
    calendar = create_calendar_features(horizon[["datetime_hour"]], holidays)
    horizon = horizon.merge(calendar, on="datetime_hour")

    for field in _WEATHER_FIELDS:
        horizon[field] = [getattr(hour, field) for hour in weather]

    horizon[TARGET_COLUMN] = np.nan

    combined = pd.concat([history, horizon], ignore_index=True)

    combined = one_hot_encode_column(combined, "conditions")
    combined = add_cyclical_features(combined, "hour", 24)
    combined = add_cyclical_features(combined, "month", 12)
    combined = add_historical_demand_features(
        combined,
        target_column=TARGET_COLUMN,
        hour_column="hour",
        weekday_column="weekday",
    )

    horizon_features = combined.tail(horizon_hours).reset_index(drop=True)

    # Reindex to the exact training columns: condition columns absent from this
    # request (or from history) are filled with 0, matching how the reference
    # category and unseen conditions behave at training time.
    feature_matrix = horizon_features.reindex(columns=feature_columns, fill_value=0)

    # Belt-and-suspenders: ensure a purely numeric matrix (float) so the model
    # never sees object/bool columns regardless of upstream dtype quirks.
    feature_matrix = feature_matrix.astype("float64")

    return feature_matrix, horizon_index
